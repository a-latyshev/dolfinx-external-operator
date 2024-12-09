import numpy as np

import basix
import ufl
from dolfinx import fem
from dolfinx import mesh as _mesh
from ufl.constantvalue import as_ufl
from ufl.core.ufl_type import ufl_type


@ufl_type(num_ops="varying", is_differential=True)
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
        argument_slots=(),
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
        """
        ufl_element = function_space.ufl_element()
        if ufl_element.family_name != "quadrature":
            raise TypeError("FEMExternalOperator currently only supports Quadrature elements.")

        self.ufl_operands = tuple(map(as_ufl, operands))
        for operand in self.ufl_operands:
            if isinstance(operand, FEMExternalOperator):
                raise TypeError("Use of FEMExternalOperators as operands is not implemented.")

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
            )
            self.ref_function_space = fem.functionspace(mesh, quadrature_element)
        else:
            self.ref_function_space = function_space
        # Make the global coefficient associated to the external operator
        self.ref_coefficient = fem.Function(self.ref_function_space)

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
        return type(self)(
            *operands,
            function_space=function_space or self.ref_function_space,
            external_function=self.external_function,
            derivatives=derivatives or self.derivatives,
            argument_slots=argument_slots or self.argument_slots(),
            **add_kwargs,
        )


def evaluate_operands(
    external_operators: list[FEMExternalOperator], entity_maps: dict[_mesh.Mesh, np.ndarray] | None = None
) -> dict[ufl.core.expr.Expr | int, np.ndarray]:
    """Evaluates operands of external operators.

    Args:
        external_operators: A list with external operators required to be updated.
        entity_maps: A dictionary mapping between parent mesh and sub mesh entities.
    Returns:
        A map between UFL operand and the `ndarray`, the evaluation of the operand.
    """
    ref_function_space = external_operators[0].ref_function_space
    ufl_element = ref_function_space.ufl_element()
    mesh = ref_function_space.mesh
    quadrature_points = basix.make_quadrature(ufl_element.cell_type, ufl_element.degree)[0]
    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

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
                codim = operand_mesh.topology.dim - mesh.topology.dim
                expr = fem.Expression(operand, quadrature_points)
                # NOTE: Using expression eval might be expensive
                if codim == 0:
                    if operand_mesh._cpp_object != mesh._cpp_object:
                        inverted_map = np.empty(len(cells), dtype=np.int32)
                        indices = np.flatnonzero(entity_maps[mesh] >= 0)
                        inverted_map[entity_maps[mesh][indices]] = indices
                        evaluated_operand = expr.eval(operand_mesh, inverted_map)
                    else:
                        evaluated_operand = expr.eval(mesh, cells)
                elif codim == 1:
                    assert entity_maps is not None
                    # Invert entity map as it goes from parent to sub not sub to parent
                    inverted_map = np.empty(len(cells), dtype=np.int32)
                    indices = np.flatnonzero(entity_maps[mesh] >= 0)
                    inverted_map[entity_maps[mesh][indices]] = indices
                    integration_entities = fem.compute_integration_domains(
                        fem.IntegralType.exterior_facet,
                        operand.function_space.mesh.topology,
                        inverted_map,
                    )
                    evaluated_operand = expr.eval(operand_mesh, integration_entities)
                else:
                    raise NotImplementedError("Only codim 0 and 1 are supported.")
            evaluated_operands[operand] = evaluated_operand

    return evaluated_operands


def evaluate_external_operators(
    external_operators: list[FEMExternalOperator], evaluated_operands: dict[ufl.core.expr.Expr | int, np.ndarray]
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
        coefficient[indexes] * external_operator_argument[indexes_contracted], indexes[: coeff_dim - arg_dim]
    )

    form_replaced = ufl.algorithms.replace(action.left(), {N_tilde: replacement})
    return form_replaced, action.right()


def _replace_form(form: ufl.Form):
    external_operators = form.base_form_operators()
    ex_ops_map = {ex_op: ex_op.ref_coefficient for ex_op in external_operators}
    replaced_form = ufl.algorithms.replace(form, ex_ops_map)
    return replaced_form, external_operators


def replace_external_operators(form):
    replaced_form = None
    external_operators = []
    if isinstance(form, ufl.Action):
        if isinstance(form.right(), ufl.Action):
            replaced_right_part, ex_ops = replace_external_operators(form.right())
            external_operators += ex_ops
            interim_form = ufl.Action(form.left(), replaced_right_part)
            replaced_form, ex_ops = replace_external_operators(interim_form)
            external_operators += ex_ops
        elif isinstance(form.right(), FEMExternalOperator):
            replaced_form, ex_op = _replace_action(form)
            external_operators += [ex_op]
        else:
            raise RuntimeError("Expected an ExternalOperator in the right part of the Action.")
    elif isinstance(form, ufl.FormSum):
        components = form.components()
        # TODO: Modify this loop so it runs from range(0, len(components))
        replaced_form, ex_ops = replace_external_operators(components[0])
        external_operators += ex_ops
        for i in range(1, len(components)):
            replaced_form_term, ex_ops = replace_external_operators(components[i])
            replaced_form += replaced_form_term
            external_operators += ex_ops
    elif isinstance(form, ufl.Form):
        replaced_form, ex_ops = _replace_form(form)
        external_operators += ex_ops

    return replaced_form, external_operators
