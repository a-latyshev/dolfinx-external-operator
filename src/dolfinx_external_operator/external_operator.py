from functools import singledispatchmethod

import numpy as np
import numpy.typing as npt

import basix
import ufl
from dolfinx import fem
from dolfinx import mesh as _mesh
from ufl.constantvalue import as_ufl
from ufl.core.ufl_type import ufl_type
from ufl.corealg.dag_traverser import DAGTraverser


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
        ufl_element = function_space.ufl_element()
        if ufl_element.family_name != "quadrature":
            raise TypeError("FEMExternalOperator currently only supports Quadrature elements.")

        self.ufl_operands = tuple(map(as_ufl, operands))
        for operand in self.ufl_operands:
            if isinstance(operand, FEMExternalOperator):
                raise TypeError("Use of FEMExternalOperators as operands is not implemented.")

        if coefficient is not None and coefficient.function_space != function_space:
            raise TypeError("The provided coefficient must be defined on the same function space as the operator.")
        super().__init__(
            *operands,
            function_space=function_space,
            derivatives=derivatives,
            argument_slots=argument_slots,
        )

        new_shape = self.ufl_shape
        for i, e in enumerate(self.derivatives):
            new_shape += self.ufl_operands[i].ufl_shape * e

        if new_shape != self.ufl_shape:
            mesh = function_space.mesh
            quadrature_element = basix.ufl.quadrature_element(
                mesh.topology.cell_name(),
                degree=ufl_element.degree,
                value_shape=new_shape,
                dtype=mesh.geometry.x.dtype,
            )
            self.ref_function_space = fem.functionspace(mesh, quadrature_element)
        else:
            self.ref_function_space = function_space

        self.name = name
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
    ref_function_space = external_operators[0].ref_function_space
    ufl_element = ref_function_space.ufl_element()
    mesh = ref_function_space.mesh
    quadrature_points = basix.make_quadrature(ufl_element.cell_type, ufl_element.degree)[0]

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
                expr = fem.Expression(operand, quadrature_points, dtype=external_operator.ref_coefficient.dtype)
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
        ufl_operands_eval = [evaluated_operands[operand] for operand in external_operator.ufl_operands]
        external_operator_eval = external_operator.external_function(external_operator.derivatives)(*ufl_operands_eval)
        # NOTE: Maybe to force the user to return always a tuple?
        if type(external_operator_eval) is tuple:
            np.copyto(external_operator.ref_coefficient.x.array, external_operator_eval[0])
        else:
            np.copyto(external_operator.ref_coefficient.x.array, external_operator_eval)

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


def _replace_action(action: ufl.Action):
    # Extract the trial function associated with ExternalOperator
    N_tilde = action.left().arguments()[-1]
    external_operator_argument = action.right().argument_slots()[-1]
    coefficient = action.right().ref_coefficient
    # NOTE: Is this replace always appropriate?
    arg_dim = len(external_operator_argument.ufl_shape)
    coeff_dim = len(coefficient.ufl_shape)
    indexes = ufl.indices(coeff_dim)
    indexes_contracted = indexes[coeff_dim - arg_dim :]
    replacement = ufl.as_tensor(
        coefficient[indexes] * external_operator_argument[indexes_contracted],
        indexes[: coeff_dim - arg_dim],
    )
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
        result_cache: dict[ufl.core.expr.Expr, ufl.core.expr.Expr] | None = None,
    ) -> None:
        """Initialise.

        Args:
            compress: If True, ``result_cache`` will be used.
            visited_cache: cache of intermediate results;
                expr -> r = self.process(expr, ...).
            result_cache: cache of result objects for memory reuse, r -> r.

        """
        super().__init__(compress=compress, visited_cache=visited_cache, result_cache=result_cache)
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
        self._ex_ops.append(o)
        return o.ref_coefficient

    @process.register(ufl.core.expr.Expr)
    def _(
        self,
        o: ufl.Argument,
    ) -> ufl.core.expr.Expr:
        """Handle anything else in UFL."""
        return self.reuse_if_untouched(o)


def replace_external_operators(form: ufl.Form) -> tuple[ufl.Form, list[FEMExternalOperator]]:
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
