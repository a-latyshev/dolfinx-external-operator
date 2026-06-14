from functools import singledispatchmethod

import numpy as np
import numpy.typing as npt

import basix
import ufl
from basix.ufl import _ElementBase
from dolfinx import fem
from dolfinx import mesh as _mesh
from ufl.algorithms import expand_derivatives
from ufl.constantvalue import Zero, as_ufl
from ufl.core.ufl_type import ufl_type
from ufl.corealg.dag_traverser import DAGTraverser


def get_unrolled_dofmap(function_space):
    dofmap_list = function_space.dofmap.list
    if dofmap_list.shape[0] == 0:
        return np.empty((0,), dtype=dofmap_list.dtype)
    bs = function_space.dofmap.bs
    unrolled_dofmap = (
        np.repeat(dofmap_list, bs).reshape(dofmap_list.shape[0], -1) * bs + np.tile(np.arange(bs), dofmap_list.shape[1])
    ).flatten()
    return unrolled_dofmap


def new_element_from_new_shape(element: _ElementBase, diff_shape: tuple[int, ...], mesh: _mesh.Mesh) -> _ElementBase:
    new_shape = element.reference_value_shape + diff_shape

    if element.element_family is None:
        element = basix.ufl.quadrature_element(
            mesh.topology.cell_name(),
            degree=element.degree,
            value_shape=new_shape,
        )
    else:
        element = basix.ufl.element(
            element.element_family,
            mesh.basix_cell(),
            degree=element.degree,
            shape=new_shape,
            discontinuous=element.discontinuous,
        )
    return element


@ufl_type(num_ops="varying", is_differential=True, use_default_hash=False)
class FEMExternalOperator(ufl.ExternalOperator):
    """Finite element external operator.

    The `FEMExternalOperator` class extends the functionality of the original
    `ufl.ExternalOperator` class, which symbolically represents operators that
    are not straightforwardly expressible in UFL.
    """

    # Slots are disabled here because they cause trouble in PyDOLFIN
    # multiple inheritance pattern:
    _ufl_noslots_ = True

    def __init__(
        self,
        *operands,
        function_space: fem.function.FunctionSpace,
        external_function=None,
        derivatives: tuple[int, ...] | None = None,
        name: str | None = None,
        coefficient: fem.Function | None = None,
        argument_slots=(),
        dtype: npt.DTypeLike | None = None,
    ) -> None:
        """Initializes `FEMExternalOperator`.

        Args:
            operands: operands on which the external operator acts.
            function_space: the `FunctionSpace`.
            external_function: A callable Python function defining the
                behaviour of the external operator and its derivatives.
            derivatives: A tuple specifiying the derivative multiindex with
                respect to operands.
            argument_slots: tuple composed containing expressions with
                `ufl.Argument` or `ufl.Coefficient` objects.
            name: Name of the external operator and the associated
            `fem.Function` coefficient.
            dtype: Data type of the external operator.
        """
        self.ufl_operands = tuple(map(expand_derivatives, map(as_ufl, operands)))  # expend high-level operands

        for operand in self.ufl_operands:
            if isinstance(operand, ufl.Coefficient) and operand.ufl_function_space().ufl_element().is_mixed:
                raise TypeError(
                    "Mixed element coefficients are not supported as external-operator operands: "
                    f"operand {operand} is a mixed-space coefficient."
                )

        if coefficient is not None and coefficient.function_space != function_space:
            raise TypeError("The provided coefficient must be defined on the same function space as the operator.")

        super().__init__(
            *self.ufl_operands,
            function_space=function_space,
            derivatives=derivatives,
            argument_slots=argument_slots,
        )

        # Define functional space of external operator
        diff_shape = ()  # extra indexes that come after differentiation
        for i, e in enumerate(self.derivatives):
            diff_shape += self.ufl_operands[i].ufl_shape * e

        if diff_shape != ():
            # TODO: should we decrease the degree after differentiation?

            mesh = function_space.mesh
            original_element = function_space.ufl_element()
            if original_element.is_mixed:
                all_sub_els = []
                for sub_el in original_element.sub_elements:
                    new_sub_el = new_element_from_new_shape(sub_el, diff_shape, mesh)
                    all_sub_els.append(new_sub_el)
                new_element = basix.ufl.mixed_element(all_sub_els)
            else:
                new_element = new_element_from_new_shape(original_element, diff_shape, mesh)

            self.ref_function_space = fem.functionspace(mesh, new_element)
        else:
            self.ref_function_space = function_space

        # TODO: update ufl_shape; how this will change the replacement mechanism?
        # Currently: its equal to self.arguments()[0].ufl_element()
        # Source: https://github.com/FEniCS/ufl/blob/966b0c0930bcdbd27fd939bb7ebf59fa6016faed/ufl/core/external_operator.py#L68
        # self.ufl_shape =
        # self.ref_function_space.ufl_element().reference_value_shape
        self.name = name

        # Points to evaluate operands
        self._is_mixed = self.ref_function_space.ufl_element().is_mixed
        if self._is_mixed:
            points = []
            val_sizes = []
            # First pass: collect shapes, points, and value sizes of all subspaces
            for i in range(self.ref_function_space.num_sub_spaces):
                Vi = self.ref_function_space.sub(i)
                pts_arr = Vi.element.interpolation_points
                points.append(pts_arr)
                val_shape = Vi.ufl_element().reference_value_shape
                val_size = int(np.prod(val_shape)) if len(val_shape) > 0 else 1
                val_sizes.append(val_size)
            self.eval_points = np.concatenate(points)
            # _comp_size is the maximum value dimension among all subspaces in the mixed element.
            # This represents the number of components returned by the external function per interpolation point,
            # which determines whether the output array is 2D (if _comp_size == 1) or 3D (if _comp_size > 1).
            # Examples:
            # - Mixed space (scalar pressure, scalar temp) -> val_sizes = [1, 1]
            #   -> _comp_size = 1 (2D output array)
            # - Mixed space (2D velocity vector, scalar pressure) -> val_sizes = [2, 1]
            #   -> _comp_size = 2 (3D output array)
            # - Mixed space (3D stress tensor, 3D velocity vector) -> val_sizes = [9, 3]
            #   -> _comp_size = 9 (3D output array)
            self._comp_size = max(val_sizes) if val_sizes else 1

            # Second pass: precompute metadata for slicing each subspace
            self._mixed_subspace_info = []
            offset = 0
            for i in range(self.ref_function_space.num_sub_spaces):
                Vi = self.ref_function_space.sub(i)
                n_pts = Vi.element.interpolation_points.shape[0]
                dofs_per_cell = Vi.dofmap.list.shape[1]
                val_shape = Vi.ufl_element().reference_value_shape
                val_size = val_sizes[i]

                if self._comp_size < val_size:
                    raise ValueError(
                        f"Unsupported mixed element layout for subspace {i}: "
                        f"val_shape={val_shape}, val_size={val_size}, comp_size={self._comp_size}. "
                        f"The subspace value size cannot exceed the overall operator component size."
                    )

                flat_dofs = Vi.dofmap.list.flatten()

                self._mixed_subspace_info.append(
                    {
                        "n_pts": n_pts,
                        "val_size": val_size,
                        "dofs_per_cell": dofs_per_cell,
                        "flat_dofs": flat_dofs,
                        "offset": offset,
                    }
                )
                offset += n_pts
            self._n_points_total = offset

            # Bind specialized assignment functions based on the global array dimension
            if self._comp_size == 1:
                self._assign_func = self._assign_mixed_2d
            else:
                self._assign_func = self._assign_mixed_3d
        else:
            self.eval_points = self.ref_function_space.element.interpolation_points

            # Use contiguous assignment for Quadrature/DG elements per user design decision
            element = self.ref_function_space.ufl_element()
            element_family = element.element_family
            is_contiguous = (element_family is None) or (element_family in ("DG",))

            if is_contiguous:
                self.unrolled_dofmap = None
                self._assign_func = self._assign_non_mixed_contiguous
            else:
                self.unrolled_dofmap = get_unrolled_dofmap(self.ref_function_space)
                self._assign_func = self._assign_non_mixed

        # Make the global coefficient associated to the external operator
        if coefficient is not None:
            self.ref_coefficient = coefficient
        else:
            self.ref_coefficient = fem.Function(self.ref_function_space, name=name, dtype=dtype)
        self.external_function = external_function

    def _ufl_expr_reconstruct_(
        self,
        *operands,
        function_space=None,
        derivatives=None,
        argument_slots=None,
        add_kwargs={},
    ):
        if self.ufl_element().is_cellwise_constant():  # TODO: TEST THIS
            new_shape = self.ufl_shape
            for i, e in enumerate(self.derivatives):
                new_shape += self.ufl_operands[i].ufl_shape * e
            return Zero(self.ufl_shape)
        """Return a new object of the same type with new operands."""
        coefficient = None
        d = "\N{PARTIAL DIFFERENTIAL}o"
        if derivatives is None:
            coefficient = self.ref_coefficient  # prevents additional allocations
            d_ops = ""
        else:
            d_ops = "/" + "".join(d + "o" + str(i + 1) for i, di in enumerate(derivatives) for j in range(di))
        dtype = self.ref_coefficient.dtype
        ex_op_name = d + self.ref_coefficient.name + d_ops
        return type(self)(
            *operands,
            function_space=function_space or self.ref_function_space,
            external_function=self.external_function,
            derivatives=derivatives or self.derivatives,
            argument_slots=argument_slots or self.argument_slots(),
            name=ex_op_name,
            coefficient=coefficient,
            dtype=dtype,
            **add_kwargs,
        )

    def __hash__(self):
        """Hash code for UFL AD."""
        hashdata = (
            type(self),
            tuple(hash(op) for op in self.ufl_operands),
            tuple(hash(arg) for arg in self._argument_slots),
            self.derivatives,
            hash(self.ufl_function_space()),
            self.ref_coefficient,
        )
        output = hash(hashdata)
        return output

    def __str__(self):
        """Default str string for FEMExternalOperator operators."""
        d = "\N{PARTIAL DIFFERENTIAL}"
        operator_name = self.name if self.name is not None else "e"
        derivatives = self.derivatives
        d_ops = "".join(d + "o" + str(i + 1) for i, di in enumerate(derivatives) for j in range(di))
        e = operator_name + "("
        e += ", ".join(str(op) for op in self.ufl_operands)
        e += "; "
        e += ", ".join(str(arg) for arg in reversed(self.argument_slots()))
        e += ")"
        return e + "/" + d_ops if sum(derivatives) > 0 else e

    def is_cellwise_constant(self):
        """Return whether this expression is spatially constant over each cell."""
        return self.ufl_element().is_cellwise_constant()

    def _assign_non_mixed(self, values: np.ndarray) -> None:
        self.ref_coefficient.x.array[self.unrolled_dofmap] = values

    def _assign_non_mixed_contiguous(self, values: np.ndarray) -> None:
        self.ref_coefficient.x.array[:] = values

    def _assign_mixed_2d(self, values: np.ndarray) -> None:
        """Assign evaluated values into a mixed function space where all subspaces are scalar.
        In this case, the values array has no component axis and is 2D with shape
        (n_cells, n_points_total) (or 1D which is reshaped).
        """
        coeff = self.ref_coefficient
        if values.ndim == 1:
            n_cells = values.size // self._n_points_total
            values = values.reshape(n_cells, self._n_points_total)

        n_cells = values.shape[0]
        for info in self._mixed_subspace_info:
            offset = info["offset"]
            n_pts = info["n_pts"]
            flat_dofs = info["flat_dofs"]

            # Scalar case: Direct 2D slice copy along the point dimension.
            # dofs_per_cell matches n_pts exactly for scalar spaces.
            block = values[:, offset : offset + n_pts]
            coeff.x.array[flat_dofs] = block.reshape(-1)

    def _assign_mixed_3d(self, values: np.ndarray) -> None:
        """Assign evaluated values into a mixed function space where at least one subspace has vector/tensor values.
        In this case, the values array has a component axis and is 3D with shape
        (n_cells, n_points_total, comp_size) (or 1D which is reshaped).
        """
        coeff = self.ref_coefficient
        if values.ndim == 1:
            n_cells = values.size // (self._n_points_total * self._comp_size)
            values = values.reshape(n_cells, self._n_points_total, self._comp_size)

        n_cells = values.shape[0]
        for info in self._mixed_subspace_info:
            offset = info["offset"]
            n_pts = info["n_pts"]
            flat_dofs = info["flat_dofs"]
            dofs_per_cell = info["dofs_per_cell"]

            chunk = values[:, offset : offset + n_pts, :]
            # Slice the required components along the component axis and reshape to match local dofs per cell.
            val_size = info["val_size"]
            block = chunk[:, :, :val_size].reshape(n_cells, dofs_per_cell)

            coeff.x.array[flat_dofs] = block.reshape(-1)


def evaluate_operands(
    external_operators: list[FEMExternalOperator],
    entities: np.ndarray | None = None,
) -> dict[ufl.core.expr.Expr | int, np.ndarray]:
    """Evaluates operands of external operators.

    Args:
        external_operators: A list with external operators required to be updated.
        entities: A dictionary mapping between parent mesh and sub mesh
        entities with respect to `eval` function of `fem.Expression`.
    Returns:
        A map between UFL operand and the `ndarray`, the evaluation of the
        operand.

    Note:
        User is responsible to ensure that `entities` are correctly constructed
        with respect to the codimension of the external operator.
    """
    if len(external_operators) == 0:
        return {}
    ref_function_space = external_operators[0].ref_function_space
    ufl_element = ref_function_space.ufl_element()
    mesh = ref_function_space.mesh
    # quadrature_points = basix.make_quadrature(ufl_element.cell_type, ufl_element.degree)[0]

    assert isinstance(ufl_element.pullback, ufl.pullback.IdentityPullback)
    # If no entity map is provided, assume that there is no sub-meshing
    if entities is None:
        map_c = mesh.topology.index_map(mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts
        cells = np.arange(0, num_cells, dtype=np.int32)
        entities = cells
    # Evaluate unique operands in external operators
    evaluated_operands = {}
    for external_operator in external_operators:
        # TODO: Is it possible to get the basix information out here?
        for operand in external_operator.ufl_operands:
            try:
                evaluated_operands[operand]
            except KeyError:
                # Check if we have a sub-mesh with different codim
                operand_domain = ufl.domain.extract_unique_domain(operand)
                operand_mesh = _mesh.Mesh(operand_domain.ufl_cargo(), operand_domain)
                # TODO: Stop recreating the expression every time
                if isinstance(operand, ufl.ExternalOperator):
                    evaluated_operand = evaluate_operands([operand], entities)
                else:
                    expr = fem.Expression(
                        operand,
                        external_operator.eval_points,
                        dtype=external_operator.ref_coefficient.dtype,
                    )
                    # NOTE: Using expression eval might be expensive
                    evaluated_operand = expr.eval(operand_mesh, entities)
                evaluated_operands[operand] = evaluated_operand
    return evaluated_operands


def evaluate_external_operators(
    external_operators: list[FEMExternalOperator],
    evaluated_operands: dict[ufl.core.expr.Expr | int, np.ndarray],
) -> list[list[np.ndarray]]:
    """Evaluates external operators and updates the associated coefficient.

    Args:
        external_operators: A list with external operators to evaluate.
        evaluated_operands: A dictionary mapping all operands to `ndarray`
                            containing their evaluation.

    Returns:
        A list containing the evaluation of the external operators.
    """

    evaluated_operators = []

    for external_operator in external_operators:
        ufl_operands_eval = []
        for operand in external_operator.ufl_operands:
            if isinstance(operand, ufl.ExternalOperator):
                ufl_operands_eval.extend(evaluate_external_operators([operand], evaluated_operands[operand]))
            else:
                ufl_operands_eval.append(evaluated_operands[operand])

        external_operator_eval = external_operator.external_function(external_operator.derivatives)(*ufl_operands_eval)

        # NOTE: Maybe to force the user to return always a tuple?
        if type(external_operator_eval) is tuple:
            values = external_operator_eval[0]
        else:
            values = external_operator_eval

        try:
            external_operator._assign_func(values)
        except ValueError:
            # Keep the old behaviour for diagnostics; re-raise with a clearer message.
            raise
        external_operator.ref_coefficient.x.scatter_forward()
        evaluated_operators.append(external_operator_eval)

    return evaluated_operators


def unique_external_operators(external_operators: list[FEMExternalOperator]):
    # Use a set to track unique hashes
    unique_hashes = set()
    unique_operators = []
    for ex_op in external_operators:
        h = ex_op.filtering_hash()
        if h not in unique_hashes:
            unique_hashes.add(h)
            unique_operators.append(ex_op)
    return unique_operators


def _apply_derivative_tensor(coef_expr: ufl.core.expr.Expr, arg_expr: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    """Apply a derivative tensor ``coef_expr`` to a direction ``arg_expr``.

    Interprets ``coef_expr`` as having shape ``out_shape + arg_shape`` and
    contracts over the trailing ``arg_shape`` indices.
    """
    arg_shape = arg_expr.ufl_shape
    if arg_shape == ():
        return coef_expr * arg_expr

    coef_shape = coef_expr.ufl_shape
    arg_rank = len(arg_shape)
    coef_rank = len(coef_shape)
    if coef_rank < arg_rank:
        raise ValueError(
            "Derivative coefficient has lower rank than direction argument: "
            f"coef_shape={coef_shape}, arg_shape={arg_shape}."
        )

    out_rank = coef_rank - arg_rank
    indices = ufl.indices(coef_rank)
    out_indices = indices[:out_rank]
    arg_indices = indices[out_rank:]
    return ufl.as_tensor(coef_expr[indices] * arg_expr[arg_indices], out_indices)


def _flatten_to_entries(expr: ufl.core.expr.Expr) -> list[ufl.core.expr.Expr]:
    """Flatten a tensor expression into scalar entries.

    UFL doesn't guarantee a public ``ufl.flatten`` helper across versions.
    We therefore flatten by explicit indexing in row-major order.
    """
    if expr.ufl_shape == ():
        return [expr]
    entries: list[ufl.core.expr.Expr] = []
    for multi_index in np.ndindex(expr.ufl_shape):
        if len(multi_index) == 1:
            entries.append(expr[multi_index[0]])
        else:
            entries.append(expr[multi_index])
    return entries


def _replace_action(action: ufl.Action):
    """Rewrite an Action involving a differentiated FEMExternalOperator.

    UFL differentiation of an ExternalOperator introduces an ``ufl.Action``.
    The left form contains an Argument (typically the test function in the
    operator's range space), and the right side provides:

    - the differentiated operator coefficient (``ref_coefficient``)
    - the direction/slot Argument (from ``argument_slots``)

    For mixed spaces, UFL flattens component shapes into a single vector shape.
    We therefore split the coefficient into components and apply the derivative
    tensor to the direction component-wise, then re-flatten to match the mixed
    Argument's shape.
    """

    # Extract the Argument associated with the (differentiated) ExternalOperator
    N_tilde = action.left().arguments()[-1]
    external_operator_argument = action.right().argument_slots()[-1]
    coefficient = action.right().ref_coefficient

    # Build a replacement expression with the same ufl_shape as N_tilde
    if coefficient.ufl_element().is_mixed:
        coef_components = ufl.split(coefficient)
        applied_components = [_apply_derivative_tensor(c, external_operator_argument) for c in coef_components]
        entries: list[ufl.core.expr.Expr] = []
        for comp in applied_components:
            entries.extend(_flatten_to_entries(comp))
        replacement = ufl.as_vector(entries)
    else:
        replacement = _apply_derivative_tensor(coefficient, external_operator_argument)

    form_replaced = ufl.algorithms.replace(action.left(), {N_tilde: replacement})
    return form_replaced, action.right()


def map_integrands(function, form, only_integral_type=None):
    """Map integrands.

    This function is a slightly modified version of `ufl/ufl/algorithms/map_integrands.py`,
    which is part of the FEniCS Project (https://www.fenicsproject.org) and
    licensed under the GNU Lesser General Public License v3.0

    Original author: Martin Sandve Alnæs

    Apply transform(expression) to each integrand expression in form, or
    to form if it is an Expr.
    """
    if isinstance(form, ufl.Form):
        mapped_integrals = [map_integrands(function, itg, only_integral_type) for itg in form.integrals()]
        nonzero_integrals = [itg for itg in mapped_integrals if not isinstance(itg.integrand(), ufl.constantvalue.Zero)]
        return ufl.Form(nonzero_integrals)
    elif isinstance(form, ufl.Integral):
        itg = form
        if (only_integral_type is None) or (itg.integral_type() in only_integral_type):
            new_itg = function(itg.integrand())
            return itg.reconstruct(integrand=new_itg)
        else:
            return itg
    elif isinstance(form, ufl.FormSum):
        mapped_components = [map_integrands(function, component, only_integral_type) for component in form.components()]
        nonzero_components = [
            (component, w)
            for component, w in zip(mapped_components, form.weights())
            # Catch ufl.Zero and ZeroBaseForm
            if component != 0
        ]

        # Simplify case with one nonzero component and the corresponding weight is 1
        if len(nonzero_components) == 1 and nonzero_components[0][1] == 1:
            return nonzero_components[0][0]
        return sum(w * component for component, w in nonzero_components)

        # return sum(component for component, _ in nonzero_components)

    elif isinstance(form, ufl.Adjoint):
        # Zeros are caught inside `Adjoint.__new__`
        return ufl.adjoint(map_integrands(function, form._form, only_integral_type))

    elif isinstance(form, ufl.Action):
        left = map_integrands(function, form._left, only_integral_type)
        right = form._right
        if isinstance(right, FEMExternalOperator):
            assert right.derivatives != ()
            interim_form, ex_op = _replace_action(ufl.Action(left, right))
            processed_form = map_integrands(function, interim_form)
            function.external_operators.extend([ex_op])
            return processed_form
        # Zeros are caught inside `Action.__new__`
        right = map_integrands(function, right, only_integral_type)
        return ufl.action(left, right)

    elif isinstance(form, ufl.ZeroBaseForm):
        arguments = tuple(map_integrands(function, arg, only_integral_type) for arg in form._arguments)
        return ufl.ZeroBaseForm(arguments)
    elif isinstance(form, ufl.core.expr.Expr | ufl.BaseForm):
        integrand = form
        return function(integrand)
    else:
        raise ValueError("Expecting Form, Integral or Expr.")


class ExternalOperatorReplacer(DAGTraverser):
    """DAGTraverser to replaced external operators with a coefficient in the appropriate space."""

    def __init__(
        self,
        compress: bool | None = True,
        visited_cache: dict[tuple, ufl.core.expr.Expr] | None = None,
        result_cache: (dict[ufl.core.expr.Expr, ufl.core.expr.Expr] | None) = None,
    ) -> None:
        """Initialise.

        Args:
            compress: If True, ``result_cache`` will be used.
            visited_cache: cache of intermediate results;
                expr -> r = self.process(expr, ...).
            result_cache: cache of result objects for memory reuse, r -> r.

        """
        super().__init__(
            compress=compress,
            visited_cache=visited_cache,
            result_cache=result_cache,
        )
        self._ex_ops = []

    @singledispatchmethod
    def process(
        self,
        o: ufl.core.expr.Expr,
    ) -> ufl.core.expr.Expr:
        """Replace external operators in UFL expressions.

        Args:
            o: `ufl.core.expr.Expr` to be processed.
        """
        self._ex_ops = []
        return super().process(o)

    @property
    def external_operators(self) -> list[FEMExternalOperator]:
        """Return the list of external operators found during processing."""
        return self._ex_ops

    @process.register(ufl.ExternalOperator)
    def _(self, o: ufl.ExternalOperator, *args) -> ufl.core.expr.Expr:
        for operand in o.ufl_operands:
            if isinstance(operand, ufl.ExternalOperator):
                if operand not in self._ex_ops:
                    self._ex_ops.append(operand)
        if o not in self._ex_ops:
            self._ex_ops.append(o)
        return o.ref_coefficient

    @process.register(ufl.core.expr.Expr)
    def _(
        self,
        o: ufl.Argument,
    ) -> ufl.core.expr.Expr:
        """Handle anything else in UFL."""
        return self.reuse_if_untouched(o)


def replace_external_operators(
    form: ufl.Form,
) -> tuple[ufl.Form, list[FEMExternalOperator]]:
    """Replace external operators with its reference coefficient.

    Args:
        expr: UFL expression.

    Returns:
        A tuple with the replaced form and the list of external operators found.
    """
    rule = ExternalOperatorReplacer()
    form = ufl.algorithms.apply_derivatives.apply_derivatives(ufl.algorithms.expand_derivatives(form))
    new_form = map_integrands(rule, form)
    return new_form, rule.external_operators
