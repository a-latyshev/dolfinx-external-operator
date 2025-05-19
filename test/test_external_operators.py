# Author: Andrey Latyshev
# Test dimensions of external operator

from mpi4py import MPI

import basix
import ufl
from dolfinx import fem, mesh
from dolfinx_external_operator import (
    FEMExternalOperator,
    replace_external_operators,
)


def Q_gen(domain, tensor_shape):
    Qe = basix.ufl.quadrature_element(domain.topology.cell_name(), degree=1, value_shape=tensor_shape)
    return fem.functionspace(domain, Qe)


def compute_dimensions(operand, u, test_f):
    domain = u.function_space.mesh
    V = u.function_space
    u_hat = ufl.TrialFunction(V)
    Q = Q_gen(domain, test_f.ufl_shape)
    N = FEMExternalOperator(operand, function_space=Q, external_function=None)

    dx = ufl.Measure(
        "dx", domain=N.ref_function_space.mesh, metadata={"quadrature_degree": 1, "quadrature_scheme": "default"}
    )
    F = ufl.inner(N, test_f) * dx
    J = ufl.derivative(F, u, u_hat)
    J_expanded = ufl.algorithms.expand_derivatives(J)
    J_replaced, J_ex_ops_list = replace_external_operators(J_expanded)
    dNdu = J_ex_ops_list[0]

    shape_dNdu = dNdu.ref_coefficient.ufl_shape
    shape_N = N.ref_coefficient.ufl_shape
    shape_operand = operand.ufl_shape
    return shape_dNdu == shape_N + shape_operand


def test_dimensions_after_differentiation():
    # F = \int N(operand) : test_f dx
    # dim(tensor) := len(shape(tensor))
    # dim(test_f) == dim(N)
    # J = \int (dN(operand) : doperand) : test_f dx
    # dim(dN) - dim(doperand) == dim(N) == dim(test_f)
    # shape(dN) == shape(N) + shape(operand)

    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    V = fem.functionspace(domain, ("P", 1))
    u = fem.Function(V)
    v = ufl.TestFunction(V)

    operand = u
    test_f = v
    assert compute_dimensions(operand, u, test_f)

    operand = u
    test_f = ufl.grad(v)
    assert compute_dimensions(operand, u, test_f)

    operand = ufl.grad(u)
    test_f = ufl.grad(v)
    assert compute_dimensions(operand, u, test_f)

    operand = ufl.grad(ufl.grad(u))
    test_f = ufl.grad(v)
    assert compute_dimensions(operand, u, test_f)

    operand = ufl.grad(ufl.grad(ufl.grad(u)))
    test_f = ufl.grad(v)
    assert compute_dimensions(operand, u, test_f)

    operand = ufl.grad(u)
    test_f = ufl.grad(ufl.grad(v))
    assert compute_dimensions(operand, u, test_f)

    operand = ufl.grad(ufl.grad(ufl.grad(u)))
    test_f = ufl.grad(ufl.grad(v))
    assert compute_dimensions(operand, u, test_f)


def check_replacement(form: ufl.Form, operators: list[FEMExternalOperator], operators_count_after_AD: int):
    V = operators[0].ufl_operands[0].function_space
    u = operators[0].ufl_operands[0]
    form_replaced, form_external_operators = replace_external_operators(form)
    assert len(form_external_operators) == len(operators)
    assert len(form_replaced.base_form_operators()) == 0

    J = ufl.derivative(form, u, ufl.TrialFunction(V))
    J_expanded = ufl.algorithms.expand_derivatives(J)
    J_replaced, J_external_operators = replace_external_operators(J_expanded)
    assert len(J_external_operators) == operators_count_after_AD
    assert len(J_replaced.base_form_operators()) == 0


def test_replacement_mechanism():
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    V = fem.functionspace(domain, ("P", 1))
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    Q_vec = Q_gen(domain, (2,))
    Q_scalar = Q_gen(domain, ())
    n1 = FEMExternalOperator(u, function_space=Q_scalar, external_function=None, name="n1")
    n2 = FEMExternalOperator(u, ufl.grad(u), function_space=Q_scalar, external_function=None, name="n2")
    N1 = FEMExternalOperator(u, function_space=Q_vec, external_function=None, name="N1")
    N2 = FEMExternalOperator(u, ufl.grad(u), function_space=Q_vec, external_function=None, name="N2")

    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 1, "quadrature_scheme": "default"})
    Fn1 = n1 * v * dx
    Fn2 = n2 * v * dx
    Fn3 = (n1 + n2) * v * dx
    Fn4 = (n1 + n2) * u * v * dx
    Fn5 = ufl.inner(n2 * ufl.grad(u), ufl.grad(v)) * dx
    FN1 = ufl.inner(N1, ufl.grad(v)) * dx
    FN2 = ufl.inner(N2, ufl.grad(v)) * dx
    FN3 = ufl.inner(N1 + N2, ufl.grad(v)) * dx
    FN4 = ufl.inner(N1 + N2 * u, ufl.grad(v)) * dx

    check_replacement(Fn1, [n1], operators_count_after_AD=1)
    check_replacement(Fn2, [n2], operators_count_after_AD=2)
    check_replacement(Fn3, [n1, n2], operators_count_after_AD=3)
    check_replacement(Fn1 + Fn2, [n1, n2], operators_count_after_AD=3)
    check_replacement(Fn1 + Fn1, [n1], operators_count_after_AD=1)
    check_replacement(Fn2 + Fn2, [n2], operators_count_after_AD=2)
    check_replacement(FN1, [N1], operators_count_after_AD=1)
    check_replacement(FN2, [N2], operators_count_after_AD=2)
    check_replacement(FN3, [N1, N2], operators_count_after_AD=3)
    check_replacement(FN1 + FN2, [N1, N2], operators_count_after_AD=3)
    check_replacement(FN2 + FN2, [N2], operators_count_after_AD=2)
    check_replacement(Fn1 + FN2, [n1, N2], operators_count_after_AD=3)
    check_replacement(FN3 + Fn3, [n1, n2, N1, N2], operators_count_after_AD=6)
    check_replacement(Fn4, [n1, n2], operators_count_after_AD=5)
    check_replacement(FN4, [N1, N2], operators_count_after_AD=4)
    check_replacement(Fn5 + Fn1, [n1, n2], operators_count_after_AD=4)
