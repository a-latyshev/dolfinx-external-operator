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
