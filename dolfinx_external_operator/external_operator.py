from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import basix
import ufl
from dolfinx import fem
from dolfinx.mesh import Mesh
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
        derivatives: Optional[Tuple[int, ...]] = None,
        argument_slots=(),
        hidden_operands: Optional[List[Union[fem.function.Function, np.ndarray]]] = [
        ],
    ) -> None:
        """Initializes `FEMExternalOperator`.

        Args:
            operands: operands on which the external operator acts.
            function_space: the `FunctionSpace`, or `MixedFunctionSpace`(?) on
                which to build this Function.
            external_function: A callable Python function defining the
                behaviour of the external operator and its derivatives.
            derivatives: A tuple specifiying the derivative multiindex with
                respect to operands.
            argument_slots: tuple composed containing expressions with
                `ufl.Argument` or `ufl.Coefficient` objects.
            hidden_operands: operands on which the external operator acts, but
                the differentiation with respect to which is not required.
        """
        ufl_element = function_space.ufl_element()
        if ufl_element.family_name != "quadrature":
            raise TypeError(
                "FEMExternalOperator currently only supports Quadrature elements.")

        self.ufl_operands = tuple(map(as_ufl, operands))
        for operand in self.ufl_operands:
            if isinstance(operand, FEMExternalOperator):
                raise TypeError(
                    "Use of FEMExternalOperators as operands is not implemented.")

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
            self.ref_function_space = fem.functionspace(
                mesh, quadrature_element)
        else:
            self.ref_function_space = function_space
        # Make the global coefficient associated to the external operator
        self.ref_coefficient = fem.Function(self.ref_function_space)

        self.external_function = external_function
        self.hidden_operands = hidden_operands

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
            hidden_operands=self.hidden_operands,
            **add_kwargs,
        )


def evaluate_operands(external_operators: List[FEMExternalOperator]) -> Dict[ufl.core.expr.Expr, np.ndarray]:
    """Evaluates operands of external operators.

    Args:
        external_operators: A list with external operators required to be updated.

    Returns:
        A map between UFL operand and the `ndarray`, the evaluation of the operand.
    """
    # TODO: Generalise to evaluate operands on subset of cells.
    ref_function_space = external_operators[0].ref_function_space
    ufl_element = ref_function_space.ufl_element()
    mesh = ref_function_space.mesh
    quadrature_points = basix.make_quadrature(ufl_element.cell_type, ufl_element.degree, basix.QuadratureType.Default)[
        0
    ]
    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    # Evaluate unique operands in external operators
    # Global map of unique operands presenting in provided external operators
    evaluated_operands = {}
    for external_operator in external_operators:
        # TODO: Is it possible to get the basix information out here?
        for operand in external_operator.ufl_operands:
            try:
                evaluated_operands[operand]
            except KeyError:
                # TODO: Next call is potentially expensive in parallel.
                # TODO: We do not need to project all operands, some of them are updated (the hidden ones).
                expr = fem.Expression(operand, quadrature_points)
                evaluated_operand = expr.eval(mesh, cells)
                # TODO: to optimize!
                # It's better to allocate memory in advance and just to copy it every time
                evaluated_operands[operand] = evaluated_operand

        for operand in external_operator.hidden_operands:
            try:
                evaluated_operands[operand]
            except KeyError:
                evaluated_operands[operand] = operand.x.array

    return evaluated_operands


def find_operands_and_allocate_memory(external_operators: List[FEMExternalOperator]
                                      ) -> Tuple[Dict[ufl.core.expr.Expr, Tuple[np.ndarray, fem.function.Expression]],
                                                 Dict[Union[ufl.core.expr.Expr, int], np.ndarray]]:
    """Finds operands of external operators and allocate memory for their evaulation.

    The function seeks unique operands among provided external operators and
    allocates Numpy-array of appropriate sizes for the operands future evaluation.

    Args:
        external_operators: A list with external operators.

    Returns:
        A tuple of two dictionaries. The first one maps operands that will
        be evaluated into a tuple of allocated `ndarray`-s and
        their`fem.Expression`-representation. The second one maps all operands (the
        objects or its id-s) into `ndarray` of their values.
    """
    # TODO: Generalise to evaluate operands on subset of cells.
    ref_function_space = external_operators[0].ref_function_space
    ufl_element = ref_function_space.ufl_element()
    mesh = ref_function_space.mesh
    quadrature_points = basix.make_quadrature(ufl_element.cell_type, ufl_element.degree, basix.QuadratureType.Default)[
        0
    ]
    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    # Global map of unique operands presenting in provided external operators
    evaluated_operands = {}
    # Global map of unique operands to be evaluated
    operands_to_project = {}
    for external_operator in external_operators:
        for operand in external_operator.ufl_operands:
            try:
                evaluated_operands[operand]
            except KeyError:
                # TODO: Next call is potentially expensive in parallel.
                expr = fem.Expression(operand, quadrature_points)
                evaluated_operand = expr.eval(mesh, cells)
                operands_to_project[operand] = (evaluated_operand, expr)
                evaluated_operands[operand] = evaluated_operand

        for operand in external_operator.hidden_operands:
            try:
                evaluated_operands[id(operand)]
            except KeyError:
                if isinstance(operand, fem.function.Function):
                    evaluated_operands[id(operand)] = operand.x.array
                elif isinstance(operand, np.ndarray):
                    evaluated_operands[id(operand)] = operand
                else:
                    raise TypeError(
                        "Hidden operands are either fem.Function-s or Numpy array-s.")

    return operands_to_project, evaluated_operands


def evaluate_operands_v2(
        operands_to_project: Dict[ufl.core.expr.Expr, Tuple[np.ndarray, fem.function.Expression]],
        mesh: Mesh
) -> None:
    """Evaluates operands.

    Evaluates only provided operands

    Args:
        external_operators: A dictionary for operands and a tuple of `ndarray`
        to store operands values and their expressions.

    Returns:
        None.
    """
    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    # Evaluate unique operands in external operators

    for operand in operands_to_project:
        operand_values, operand_expression = operands_to_project[operand]
        np.copyto(operand_values, operand_expression.eval(mesh, cells))


def evaluate_external_operators(
    external_operators: List[FEMExternalOperator], evaluated_operands: Dict[Union[ufl.core.expr.Expr, int], np.ndarray]
) -> None:
    """Evaluates external operators and updates their reference coefficients.

    Args:
        external_operators: A list with external operators to evaluate.
        evaluated_operands: A dictionary mapping all operands (the
        objects or its id-s) into `ndarray` of their values.

    Bug:
        evaluated_operands is Dict[ufl.core.expr.Expr, np.ndarray] but actually
        it may have int values, like id(op). The function allows it!
    Returns:
        None
    """
    for external_operator in external_operators:
        # Is it costly?
        ufl_operands_eval = [evaluated_operands[operand]
                             for operand in external_operator.ufl_operands]
        hidden_operands_eval = [evaluated_operands[id(operand)]
                                for operand in external_operator.hidden_operands]
        external_operator_eval = external_operator.external_function(
            external_operator.derivatives)(*ufl_operands_eval, *hidden_operands_eval)

        np.copyto(external_operator.ref_coefficient.x.array,
                  external_operator_eval)


def _replace_action(action: ufl.Action):
    # Extract the trial function associated with ExternalOperator
    N_tilde = action.left().arguments()[-1]
    external_operator_argument = action.right().argument_slots()[-1]
    coefficient = action.right().ref_coefficient
    # NOTE: Is this replace always appropriate?
    arg_dim = len(external_operator_argument.ufl_shape)
    coeff_dim = len(coefficient.ufl_shape)
    indexes = ufl.indices(coeff_dim)
    indexes_contracted = indexes[coeff_dim - arg_dim:]
    replacement = ufl.as_tensor(
        coefficient[indexes] *
        external_operator_argument[indexes_contracted], indexes[: coeff_dim - arg_dim]
    )

    form_replaced = ufl.algorithms.replace(
        action.left(), {N_tilde: replacement})
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
            replaced_right_part, ex_ops = replace_external_operators(
                form.right())
            external_operators += ex_ops
            interim_form = ufl.Action(form.left(), replaced_right_part)
            replaced_form, ex_ops = replace_external_operators(interim_form)
            external_operators += ex_ops
        elif isinstance(form.right(), FEMExternalOperator):
            replaced_form, ex_op = _replace_action(form)
            external_operators += [ex_op]
        else:
            raise RuntimeError(
                "Expected an ExternalOperator in the right part of the Action.")
    elif isinstance(form, ufl.FormSum):
        components = form.components()
        # TODO: Modify this loop so it runs from range(0, len(components))
        replaced_form, ex_ops = replace_external_operators(components[0])
        external_operators += ex_ops
        for i in range(1, len(components)):
            replaced_form_term, ex_ops = replace_external_operators(
                components[i])
            replaced_form += replaced_form_term
            external_operators += ex_ops
    elif isinstance(form, ufl.Form):
        replaced_form, ex_ops = _replace_form(form)
        external_operators += ex_ops

    return replaced_form, external_operators
