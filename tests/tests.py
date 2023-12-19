from mpi4py import MPI
from petsc4py import PETSc

import basix
import ufl
from dolfinx import fem, mesh
from dolfinx_external_operator import FEMExternalOperator, replace_external_operators, evaluate_operands, evaluate_external_operators

import pytest
import numpy as np


# @pytest.fixture
def domain_2d():
    return mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)


# @pytest.fixture
# def dx():
#     return ufl.Measure("dx", domain=domain, metadata={
#                        "quadrature_degree": 1, "quadrature_scheme": "default"})


# @pytest.fixture
def V(domain):
    return fem.functionspace(domain, ("CG", 1))


def Q_gen(domain, tensor_shape):
    Qe = basix.ufl.quadrature_element(
        domain.topology.cell_name(), degree=1, value_shape=tensor_shape)
    return fem.functionspace(domain, Qe)


def compute_dimensions(N, test_f, u):
    dx = ufl.Measure("dx", domain=N.ref_function_space.mesh, metadata={
                     "quadrature_degree": 1, "quadrature_scheme": "default"})
    V = u.function_space
    u_hat = ufl.TrialFunction(V)

    operand = N.ufl_operands[0]
    F = ufl.inner(N, test_f) * dx
    J = ufl.derivative(F, u, u_hat)
    J_expanded = ufl.algorithms.expand_derivatives(J)
    J_replaced, J_ex_ops_list = replace_external_operators(J_expanded)
    dNdu = J_ex_ops_list[0]

    dim_dNdu = len(dNdu.ref_coefficient.ufl_shape)
    dim_operand = len(operand.ufl_shape)
    dim_test_f = len(test_f.ufl_shape)
    return dim_dNdu - dim_operand == dim_test_f


# def try_this(N_shape, operand, test_f):
#     Q = Q_gen(domain, N_shape)
#     N = FEMExternalOperator(operand, function_space=Q, external_function=None)
#     return compute_dimensions(N, test_f, u)


def test_dimensions_after_differentiation(V):
    # F = \int N(operand) : test_f dx
    # dim(tensor) := len(shape(tensor))
    # dim(test_f) == dim(N)
    # J = \int (dN(operand) : doperand) : test_f dx
    # dim(dN) - dim(doperand) == dim(N) == dim(test_f)

    domain = V.mesh
    v = ufl.TestFunction(V)
    u = fem.Function(V)

    Q = Q_gen(domain, ())
    operand = u
    test_f = v
    N = FEMExternalOperator(operand, function_space=Q, external_function=None)
    assert compute_dimensions(N, test_f, u)

    Q = Q_gen(domain, (2,))
    operand = u
    test_f = ufl.grad(v)
    N = FEMExternalOperator(operand, function_space=Q, external_function=None)
    assert compute_dimensions(N, test_f, u)

    Q = Q_gen(domain, (2,))
    operand = ufl.grad(u)
    test_f = ufl.grad(v)
    N = FEMExternalOperator(operand, function_space=Q, external_function=None)
    assert compute_dimensions(N, test_f, u)

    Q = Q_gen(domain, (2,))
    operand = ufl.grad(ufl.grad(u))
    test_f = ufl.grad(v)
    N = FEMExternalOperator(operand, function_space=Q, external_function=None)
    assert compute_dimensions(N, test_f, u)

    Q = Q_gen(domain, (2,))
    operand = ufl.grad(ufl.grad(ufl.grad(u)))
    test_f = ufl.grad(v)
    N = FEMExternalOperator(operand, function_space=Q, external_function=None)
    assert compute_dimensions(N, test_f, u)

    Q = Q_gen(domain, (2, 2))
    operand = ufl.grad(u)
    test_f = ufl.grad(ufl.grad(v))
    N = FEMExternalOperator(operand, function_space=Q, external_function=None)
    assert compute_dimensions(N, test_f, u)

    Q = Q_gen(domain, (2, 2))
    operand = ufl.grad(ufl.grad(ufl.grad(u)))
    test_f = ufl.grad(ufl.grad(v))
    N = FEMExternalOperator(operand, function_space=Q, external_function=None)
    assert compute_dimensions(N, test_f, u)


# def test_external_operators_composition(V):
#     domain = V.mesh
#     v = ufl.TestFunction(V)
#     u = fem.Function(V)
#     u_hat = ufl.TrialFunction(V)

#     dx = ufl.Measure("dx", domain=domain, metadata={
#                      "quadrature_degree": 1, "quadrature_scheme": "default"})

#     Q_N = Q_gen(domain, (2,))
#     Q_M = Q_gen(domain, ())
#     N = FEMExternalOperator(
#         ufl.grad(u), function_space=Q_N, external_function=None)
#     M = FEMExternalOperator(N, function_space=Q_M, external_function=None)

#     F = ufl.inner(M, v) * dx
#     J = ufl.derivative(F, u, u_hat)
#     J_expanded = ufl.algorithms.expand_derivatives(J)
#     J_replaced, J_ex_ops_list = replace_external_operators(J_expanded)
#     dNdu = J_ex_ops_list[0]
#     print(M.ufl_operands)


# test_external_operators_composition(V(domain_2d()))
