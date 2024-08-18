# Author: Andrey Latyshev
# Test dimensions of external operator

import pytest

from mpi4py import MPI

import basix
import ufl
from dolfinx import fem, mesh
from dolfinx_external_operator import (
    FEMExternalOperator,
    replace_external_operators,
)

# # @pytest.fixture
# def domain_2d():
#     return mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)


# @pytest.fixture
# def u():
#     domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
#     V = fem.functionspace(domain, ("P", 1))
#     u = fem.Function(V)
#     return u


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
    # dim_operand = len(operand.ufl_shape)
    # dim_test_f = len(test_f.ufl_shape)
    return shape_dNdu == shape_N + shape_operand


# def try_this(N_shape, operand, test_f):
#     Q = Q_gen(domain, N_shape)
#     N = FEMExternalOperator(operand, function_space=Q, external_function=None)
#     return compute_dimensions(N, test_f, u)


# @pytest.mark.parametrize("test_f", [np.float32, np.float64])
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
