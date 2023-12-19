from typing import List, Optional, Tuple

import numpy as np

import basix
import basix.ufl
import ufl
import ufl.algorithms
import ufl.form
from dolfinx import fem
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
        hidden_operands: Optional[List[fem.function.Function]] = None,
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
            raise TypeError("FEMExternalOperator currently only supports Quadrature elements.")

        super().__init__(
            *operands,
            function_space=function_space,
            derivatives=derivatives,
            argument_slots=argument_slots,
        )

        self.ufl_operands = tuple(map(as_ufl, operands))
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

    def update(self, operands_eval: List[np.ndarray]) -> None:
        """Updates the global values of external operator.

        Evaluates the external operator according to its definition in
        `external_function` and updates values in the reference coefficient, a
        globally allocated coefficient associated with the external operator.

        Args:
            operands_eval: A list with values of operands, on which the derivation is performed.
        Returns:
            None
        """
        hidden_operands_eval = []
        if self.hidden_operands is not None:
            for operand in self.hidden_operands:
                # TODO: more elegant solution is required
                hidden_operands_eval.append(operand.x.array)
        all_operands_eval = operands_eval + hidden_operands_eval
        external_operator_eval = self.external_function(self.derivatives)(*all_operands_eval)
        np.copyto(self.ref_coefficient.x.array, external_operator_eval)


def evaluate_operands(external_operators: List[FEMExternalOperator]):
    """Evaluates operands of external operators.

    Args:
        external_operators: A list with external operators required to be updated.

    Returns:
        A map between quadrature type and UFL operand and the `ndarray`, the
        evaluation of the operand.
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
    evaluated_operands = {}
    for external_operator in external_operators:
        # TODO: Is it possible to get the basix information out here?
        ref_coefficient = external_operator.ref_coefficient
        ufl_element = ref_coefficient.ufl_function_space().ufl_element()
        quadrature_triple = (
            int(basix.QuadratureType.Default),
            int(ufl_element.cell_type),
            ufl_element.degree,
        )
        quadrature_points = basix.make_quadrature(
            ufl_element.cell_type, ufl_element.degree, basix.QuadratureType.Default
        )[0]

        for operand in external_operator.ufl_operands:
            try:
                evaluated_operands[(quadrature_triple, operand)]
            except KeyError:
                # TODO: Next call is potentially expensive in parallel.
                expr = fem.Expression(operand, quadrature_points)
                evaluated_operand = expr.eval(mesh, cells)
                evaluated_operand = evaluated_operand.reshape(num_cells, -1) 
                evaluated_operands[(quadrature_triple, operand)] = evaluated_operand  # TODO: to optimize!
    return evaluated_operands


def evaluate_external_operators(external_operators: List[FEMExternalOperator], evaluated_operands) -> None:
    """Evaluates external operators and updates their reference coefficients.

    Args:
        external_operators: A list with external operators to evaluate.
        evaluated_operands: A list containing operands values in `ndarray`-format.

    Returns:
        None
    """
    for external_operator in external_operators:
        # TODO: Is it possible to get the basix information out here?
        ref_coefficient = external_operator.ref_coefficient
        ufl_element = ref_coefficient.ufl_function_space().ufl_element()
        quadrature_triple = (
            int(basix.QuadratureType.Default),
            int(ufl_element.cell_type),
            ufl_element.degree,
        )
        basix.make_quadrature(ufl_element.cell_type, ufl_element.degree, basix.QuadratureType.Default)[0]

        operands_eval = []
        for operand in external_operator.ufl_operands:
            operands_eval.append(evaluated_operands[quadrature_triple, operand])

        external_operator.update(operands_eval)


def _replace_action(action: ufl.Action):
    # Extract the trial function associated with ExternalOperator
    N_tilde = action.left().arguments()[-1]
    external_operator_argument = action.right().argument_slots()[-1]
    # NOTE: Is this replace always appropriate?
    form_replaced = ufl.algorithms.replace(
        action.left(), {N_tilde: action.right().ref_coefficient * external_operator_argument}
    )
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
