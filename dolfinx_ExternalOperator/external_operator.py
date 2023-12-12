# import ufl
from dolfinx import fem
import basix

from ufl import Action, ExternalOperator
from ufl.form import Form, FormSum
from ufl.algorithms import replace
from ufl.core.ufl_type import ufl_type
from ufl.constantvalue import as_ufl

from typing import List, Tuple, Optional
import numpy as np

"""
TODO: 
Several improvements can be made:
    1. How to update operands only once?
    2. Create an ancestor for all derivatives?
    3. Allocate memory once for all operands?
    4. Inherit from both ufl.ExternalOperator and fem.Function?
    5. IDEA: Let the femExOp derivatives inherit evaluated operands and then provide a map between external operators and their evaluated operands. Write a more general algorithm that will distinguish "families" of external operators in order to evaluate operands of a "family" only once.
    6. Tests:
        1. Evaluation of an external operator that depends on another external operator.
        2. Several external operators in one form.
        3. Combination of the previous tests.
    7. Provide operands with an appropriate functional space, where they are evaluated.
"""


@ufl_type(num_ops="varying", is_differential=True)
class femExternalOperator(ExternalOperator):
    """Finite external operator.

    The `femExternalOperator` class extends the functionality of the original `ufl.ExternalOperator` class, which symbolically represents operators that are not straightforwardly expressible in UFL. The `femExternalOperator` aims to represent an external operator globally in a certain functional space in a form of a form coefficient.

    TODO: add example

    Attributes:
        ufl_operands:
        ref_function_space: A `fem.FunctionSpace` on which the global representation of the operator exists.
        ref_coefficient: A `fem.Function` coefficient representing the operator globally.
        external_function: A callable Python function defining the behaviour of the external operator and its derivatives.
        derivatives: A tuple specifiying the derivative multiindex with respect to operands.
        hidden_operands: operands on which the external operator acts, but the differentiation with respect to which is not required.
    """

    # Slots are disabled here because they cause trouble in PyDOLFIN
    # multiple inheritance pattern:
    _ufl_noslots_ = True

    def __init__(self,
                 *operands,
                 function_space: fem.function.FunctionSpace,
                 external_function=None,
                 derivatives=None,
                 argument_slots=(),
                 hidden_operands: Optional[List[fem.function.Function]] = None) -> None:
        """Initializes `femExternalOperator`.

        Args:
            operands: operands on which the external operator acts.
            function_space: the `FunctionSpace`, or `MixedFunctionSpace`(?) on which to build this Function.
            external_function: A callable Python function defining the behaviour of the external operator and its derivatives.
            derivatives: A tuple specifiying the derivative multiindex with respect to operands.
            argument_slots: tuple composed containing expressions with `ufl.Argument` or `ufl.Coefficient` objects.
            hidden_operands: operands on which the external operator acts, but the differentiation with respect to which is not required.
        """
        ufl_element = function_space.ufl_element()
        if ufl_element.family_name != 'quadrature':
            raise TypeError(
                "This implementation of ExternalOperator supports quadrature elements only.")

        super().__init__(*operands,
                         function_space=function_space,
                         derivatives=derivatives,
                         argument_slots=argument_slots)

        self.ufl_operands = tuple(map(as_ufl, operands))
        new_shape = self.ufl_shape
        for i, e in enumerate(self.derivatives):
            new_shape += self.ufl_operands[i].ufl_shape * e
        if new_shape != self.ufl_shape:
            mesh = function_space.mesh
            quadrature_element = basix.ufl.quadrature_element(
                mesh.topology.cell_name(), degree=ufl_element.degree, value_shape=new_shape)
            self.ref_function_space = fem.functionspace(
                mesh, quadrature_element)
        else:
            self.ref_function_space = function_space
        # Make the global coefficient associated to the external operator
        self.ref_coefficient = fem.Function(self.ref_function_space)

        self.external_function = external_function
        self.hidden_operands = hidden_operands

    def _ufl_expr_reconstruct_(self, *operands, function_space=None, derivatives=None,
                               argument_slots=None, add_kwargs={}):
        """Return a new object of the same type with new operands."""
        return type(self)(*operands,
                          function_space=function_space or self.ref_function_space,
                          external_function=self.external_function,
                          derivatives=derivatives or self.derivatives,
                          argument_slots=argument_slots or self.argument_slots(),
                          hidden_operands=self.hidden_operands,
                          **add_kwargs)

    def update(self, operands_eval: List[np.ndarray]) -> None:
        """Updates the global values of external operator.

        Evaluates the external operator according to its definition in `external_function` and updates values in the reference coefficient, a globally allocated coefficient associated with the external operator. 

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
        external_operator_eval = self.external_function(
            self.derivatives)(*all_operands_eval)
        np.copyto(self.ref_coefficient.x.array, external_operator_eval)

# def copy_femExternalOperator(ex_op: femExternalOperator, function_space: fem.function.FunctionSpace):
#     operands = ex_op.ufl_operands
#     derivatives = ex_op.derivatives
#     argument_slots = ex_op.argument_slots()
#     return femExternalOperator(*operands,
#                                 function_space=function_space,
#                                 derivatives=derivatives,
#                                 argument_slots=argument_slots)


def evaluate_operands(external_operators: List[femExternalOperator]):
    """Evaluates operands of external operators.

    Args:
        external_operators: A list with external operators required to be updated.

    Returns:
        A map between quadrature type and UFL operand and the `ndarray`, the evaluation of the
        operand.
    """
    # TODO: Generalise to evaluate operands on subset of cells.
    ref_function_space = external_operators[0].ref_function_space
    ufl_element = ref_function_space.ufl_element()
    mesh = ref_function_space.mesh
    quadrature_points = basix.make_quadrature(
        ufl_element.cell_type, ufl_element.degree, basix.QuadratureType.Default)[0]
    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    # Evaluate unique operands in external operators
    evaluated_operands = {}
    for external_operator in external_operators:
        # TODO: Is it possible to get the basix information out here?
        ref_coefficient = external_operator.ref_coefficient
        ufl_element = ref_coefficient.ufl_function_space().ufl_element()
        quadrature_triple = (int(basix.QuadratureType.Default), int(
            ufl_element.cell_type), ufl_element.degree)
        quadrature_points = basix.make_quadrature(
            ufl_element.cell_type, ufl_element.degree, basix.QuadratureType.Default)[0]

        for operand in external_operator.ufl_operands:
            try:
                evaluated_operands[(quadrature_triple, operand)]
            except KeyError:
                expr = fem.Expression(operand, quadrature_points)
                evaluated_operand = expr.eval(mesh, cells)
                evaluated_operands[(quadrature_triple, operand)
                                   ] = evaluated_operand  # TODO: to optimize!
    return evaluated_operands


def evaluate_external_operators(
    external_operators: List[femExternalOperator],
    evaluated_operands
) -> None:
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
        quadrature_triple = (int(basix.QuadratureType.Default), int(
            ufl_element.cell_type), ufl_element.degree)
        quadrature_points = basix.make_quadrature(
            ufl_element.cell_type, ufl_element.degree, basix.QuadratureType.Default)[0]

        operands_eval = []
        for operand in external_operator.ufl_operands:
            operands_eval.append(
                evaluated_operands[quadrature_triple, operand])

        external_operator.update(operands_eval)


def replace_Action(form: Action):
    # trial function associated with ExternalOperator
    N_tilde = form.left().arguments()[-1]
    ex_op_argument = form.right().argument_slots()[-1]  # e.g. grad u_tilde
    form_replaced = replace(form.left(), {N_tilde: form.right(
    ).ref_coefficient * ex_op_argument})  # TODO: Is it always like this ?
    return form_replaced, form.right()


def replace_Form(form: Form):
    external_operators = form.base_form_operators()
    ex_ops_map = {ex_op: ex_op.ref_coefficient for ex_op in external_operators}
    replaced_form = replace(form, ex_ops_map)
    return replaced_form, external_operators


def replace_external_operators(form):
    replaced_form = None
    external_operators = []
    if isinstance(form, Action):
        if isinstance(form.right(), Action):
            replaced_right_part, ex_ops = replace_external_operators(
                form.right())
            external_operators += ex_ops
            interim_form = Action(form.left(), replaced_right_part)
            replaced_form, ex_ops = replace_external_operators(interim_form)
            external_operators += ex_ops
        elif isinstance(form.right(), femExternalOperator):
            replaced_form, ex_op = replace_Action(form)
            external_operators += [ex_op]
        else:
            raise TypeError(
                "A femExternalOperator is expected in the right part of the Action.")
    elif isinstance(form, FormSum):
        components = form.components()
        replaced_form, ex_ops = replace_external_operators(components[0])
        external_operators += ex_ops
        for i in range(1, len(components)):
            replaced_form_term, ex_ops = replace_external_operators(
                components[i])
            replaced_form += replaced_form_term
            external_operators += ex_ops
    elif isinstance(form, Form):
        replaced_form, ex_ops = replace_Form(form)
        external_operators += ex_ops

    return replaced_form, external_operators
