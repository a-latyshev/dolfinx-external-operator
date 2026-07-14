from mpi4py import MPI
import numpy as np
import pytest
import basix.ufl
import ufl
from dolfinx import fem
from dolfinx.mesh import create_unit_square
from dolfinx_external_operator import (
    FEMExternalOperator,
    replace_external_operators,
    evaluate_external_operators,
    evaluate_operands,
)


def check_vector_matrix(F, F_explicit, u):
    V = u.function_space
    J = ufl.derivative(F, u, ufl.TrialFunction(V))
    F_replaced, F_external_operators = replace_external_operators(F)
    J_replaced, J_external_operators = replace_external_operators(J)
    evaluated_operands = evaluate_operands(F_external_operators)
    _ = evaluate_external_operators(F_external_operators, evaluated_operands)
    _ = evaluate_external_operators(J_external_operators, evaluated_operands)

    F_compiled = fem.form(F_replaced)
    J_compiled = fem.form(J_replaced)
    b_vector = fem.assemble_vector(F_compiled)
    A_matrix = fem.assemble_matrix(J_compiled)

    F_explicit_compiled = fem.form(F_explicit)
    b_explicit_vector = fem.assemble_vector(F_explicit_compiled)
    assert np.allclose(b_explicit_vector.array, b_vector.array)

    J_explicit = ufl.derivative(F_explicit, u, ufl.TrialFunction(V))
    J_explicit_compiled = fem.form(J_explicit)
    A_explicit_matrix = fem.assemble_matrix(J_explicit_compiled)
    assert np.allclose(A_explicit_matrix.to_dense(), A_matrix.to_dense())


def test_spatial_derivative_scalar_grad():
    domain = create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = fem.functionspace(domain, ("P", 1))
    u = fem.Function(V)
    u.interpolate(lambda x: x[0] * x[0] + x[1])
    
    Q = fem.functionspace(domain, ("P", 1))
    
    # N(u) = u
    def N_impl(u_):
        return u_.reshape(-1)

    # dN/du = 1
    def dNdu_impl(u_):
        n_cells = u_.shape[0]
        n_dofs = u_.shape[1]
        return np.ones(n_cells * n_dofs)

    def N_external(derivatives):
        if derivatives == (0,):
            return N_impl
        elif derivatives == (1,):
            return dNdu_impl
        else:
            raise NotImplementedError

    N = FEMExternalOperator(u, function_space=Q, external_function=N_external, name="N")
    
    # Take ufl.grad(N)
    grad_N = ufl.grad(N)
    
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=domain)
    F = ufl.inner(grad_N, ufl.grad(v)) * dx
    
    F_explicit = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    
    # Verify the vector assembly and Jacobian matrix assembly
    check_vector_matrix(F, F_explicit, u)


def test_spatial_derivative_vector_grad():
    domain = create_unit_square(MPI.COMM_WORLD, 4, 4)
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("P", 1, (gdim,)))
    u = fem.Function(V)
    u.interpolate(lambda x: (x[0] * x[0], x[1] * x[1]))
    
    Q = fem.functionspace(domain, ("P", 1, (gdim,)))
    
    # N(u) = u
    def N_impl(u_):
        return u_.reshape(-1)

    # dN/du = Id (2x2 identity matrix at each point)
    Id = np.eye(gdim)
    def dNdu_impl(u_):
        n_cells = u_.shape[0]
        n_dofs = u_.shape[1]
        return np.repeat(Id[np.newaxis, :, :], n_cells * n_dofs, axis=0).reshape(-1)

    def N_external(derivatives):
        if derivatives == (0,):
            return N_impl
        elif derivatives == (1,):
            return dNdu_impl
        else:
            raise NotImplementedError

    N = FEMExternalOperator(u, function_space=Q, external_function=N_external, name="N")
    
    # Take ufl.grad(N)
    grad_N = ufl.grad(N)
    
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=domain)
    F = ufl.inner(grad_N, ufl.grad(v)) * dx
    
    F_explicit = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    
    # Verify the vector assembly and Jacobian matrix assembly
    check_vector_matrix(F, F_explicit, u)
