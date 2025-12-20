# Author: Andrey Latyshev
# Test external evaluation of external operators

from mpi4py import MPI

import numpy as np

import basix
import ufl
from dolfinx import fem, mesh
from dolfinx_external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    evaluate_operands,
    replace_external_operators,
)
from ufl import Measure, TestFunction, TrialFunction, derivative, div, grad, inner, split


def check_vector_matrix(F, F_explicit, u):
    """ "Check that the vector and matrix assembled from the form `F`
    match those assembled from the explicit form F_explicit. The derivative is
    taken in the direction of the function `u`.
    """
    V = u.function_space
    J = derivative(F, u, TrialFunction(V))
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

    J_explicit = derivative(F_explicit, u, TrialFunction(V))
    J_explicit_compiled = fem.form(J_explicit)
    A_explicit_matrix = fem.assemble_matrix(J_explicit_compiled)
    assert np.allclose(A_explicit_matrix.to_dense(), A_matrix.to_dense())


def test_quadrature_space():
    # Test is based on the heat equation tutorial:
    # `nonlinear_heat_equation_part2.py`.
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    V = fem.functionspace(domain, ("P", 1))

    T = fem.Function(V)
    sigma = grad(T)

    T.interpolate(lambda x: x[0] ** 2 + x[1])

    quadrature_degree = 2
    Qe = basix.ufl.quadrature_element(domain.topology.cell_name(), degree=quadrature_degree, value_shape=(2,))
    Q = fem.functionspace(domain, Qe)
    dx = Measure("dx", metadata={"quadrature_degree": quadrature_degree})

    A = 1.0
    B = 1.0
    Id = np.eye(2)
    gdim = domain.geometry.dim

    def k(T):
        return 1.0 / (A + B * T)

    def q_impl(T, sigma):
        num_cells = T.shape[0]
        sigma_ = sigma.reshape((num_cells, -1, gdim))
        output = -k(T)[:, :, np.newaxis] * sigma_
        return output.reshape(-1)

    def dqdT_impl(T, sigma):
        num_cells = T.shape[0]
        sigma_ = sigma.reshape((num_cells, -1, gdim))
        output = B * (k(T) ** 2)[:, :, np.newaxis] * sigma_
        return output.reshape(-1)

    def dqdsigma_impl(T, sigma):
        output = -k(T)[:, :, np.newaxis, np.newaxis] * Id[np.newaxis, np.newaxis, :, :]
        return output.reshape(-1)

    def q_external(derivatives):
        if derivatives == (0, 0):
            return q_impl
        elif derivatives == (1, 0):
            return dqdT_impl
        elif derivatives == (0, 1):
            return dqdsigma_impl
        else:
            raise NotImplementedError

    q_ = FEMExternalOperator(T, sigma, function_space=Q, external_function=q_external)
    T_tilde = TestFunction(V)

    F = inner(q_, grad(T_tilde)) * dx
    k_explicit = 1.0 / (A + B * T)
    q_explicit = -k_explicit * sigma
    F_explicit = inner(q_explicit, grad(T_tilde)) * dx

    check_vector_matrix(F, F_explicit, T)


def test_discontinuous_space():
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("P", 1, (gdim,)))
    u = fem.Function(V)
    u.interpolate(lambda x: (x[0], x[1]))

    def N_impl(div_u_, grad_u_):
        # each operand has shape: (n_cells, n_dofs_per_cell, *math_shape)
        # N = div(u) + grad(u):grad(u)
        # shape(grad(u):grad(u)) == (n_cells, n_dofs_per_cell)
        return div_u_.reshape(-1) + np.einsum("...ij,...ij->...", grad_u_, grad_u_).reshape(-1)

    def dNddivu_impl(div_u_, grad_u_):
        n_cells = div_u_.shape[0]
        n_dofs = div_u_.shape[1]
        return np.ones(n_cells * n_dofs).reshape(-1)

    def dNdgradu_impl(div_u_, grad_u_):
        return 2 * grad_u_.reshape(-1)

    def N_external(derivatives):
        if derivatives == (0, 0):
            return N_impl
        elif derivatives == (1, 0):
            return dNddivu_impl
        elif derivatives == (0, 1):
            return dNdgradu_impl
        else:
            raise NotImplementedError

    Q = fem.functionspace(domain, ("DG", 1, ()))
    N = FEMExternalOperator(div(u), grad(u), function_space=Q, external_function=N_external)
    v = TestFunction(V)

    dx = ufl.Measure("dx", metadata={"quadrature_degree": 2})
    F = N * inner(u, v) * dx
    N_explicit = div(u) + inner(grad(u), grad(u))
    F_explicit = N_explicit * inner(u, v) * dx
    check_vector_matrix(F, F_explicit, u)


def test_continuous_space():
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("P", 1, (gdim,)))
    u = fem.Function(V)
    u.x.array[:] = 1.0

    def N_impl(u_):
        return u_.reshape(-1)

    Id = np.eye(2)

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

    N = FEMExternalOperator(u, function_space=V, external_function=N_external)
    v = TestFunction(V)
    F = inner(N, v) * ufl.dx

    N_explicit = u
    F_explicit = inner(N_explicit, v) * ufl.dx
    check_vector_matrix(F, F_explicit, u)


def test_mixed_element_space():
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    gdim = domain.geometry.dim

    Ve1 = basix.ufl.element("P", domain.topology.cell_name(), degree=1, shape=())
    Ve2 = basix.ufl.element("P", domain.topology.cell_name(), degree=2, shape=())
    V = fem.functionspace(domain, basix.ufl.mixed_element([Ve1, Ve2]))
    u = fem.Function(V)
    u.sub(0).interpolate(lambda x: x[1] + 2.0)
    u.sub(1).interpolate(lambda x: x[1] + 1.0)
    u1, u2 = split(u)
    v = TestFunction(V)

    V1 = V.sub(0)
    V2 = V.sub(1)
    local_dofs_V1 = V1.dofmap.list.shape[1]
    local_dofs_V2 = V2.dofmap.list.shape[1]
    local_size_V = local_dofs_V1 + local_dofs_V2

    def N_impl(u_):
        out = np.zeros_like(u_)
        out[:, local_dofs_V1:local_size_V] = u_[:, local_dofs_V1:local_size_V]
        return out.reshape(-1)

    def dN_impl(u_):
        out = np.zeros_like(u_)
        out[:, local_dofs_V1:local_size_V] = 1.0
        return out.reshape(-1)

    def N_external(derivatives):
        if derivatives == (0,):
            return N_impl
        elif derivatives == (1,):
            return dN_impl
        else:
            raise NotImplementedError

    # u2 from V2 will be projected onto both V1 and V2
    N = FEMExternalOperator(u2, function_space=V, name="N", external_function=N_external)
    N1, N2 = split(N)
    v1, v2 = split(v)
    F = N1 * v1 * ufl.dx + inner(grad(N2), v) * ufl.dx
    F_explicit = inner(grad(u2), v) * ufl.dx

    check_vector_matrix(F, F_explicit, u)

    # Ve1 = basix.ufl.element("P", domain.topology.cell_name(), degree=1, shape=())
    # Ve2 = basix.ufl.element("P", domain.topology.cell_name(), degree=2, shape=(gdim,))
    # V = fem.functionspace(domain, basix.ufl.mixed_element([Ve1, Ve2]))
    # u = fem.Function(V)
    # u.sub(0).interpolate(lambda x: x[1] + 2.0)
    # u.sub(1).interpolate(lambda x: (x[0], x[1]))
    # u1, u2 = split(u)
    # v = TestFunction(V)

    # V1 = V.sub(0)
    # V2 = V.sub(1)
    # local_dofs_V1 = V1.dofmap.list.shape[1]
    # local_dofs_V2 = V2.dofmap.list.shape[1]
    # local_size_V = local_dofs_V1 + local_dofs_V2

    # # N = [N1, N2]
    # # N1 = u1 + inner(u2, u2), N2 = u2
    # def N_impl(u1_, u2_):
    #     # each operand has shape: (n_cells, n_dofs_Ve1 + n_dofs_Ve2,
    #     # *math_shape)
    #     # shape(N) == (n_cells, n_dofs_Ve1 + n_dofs_Ve2, *math_shape)
    #     out = np.zeros_like(u_)
    #     out[:, local_dofs_V1:local_size_V] = u_[:, local_dofs_V1:local_size_V]
    #     return out.reshape(-1)

    # def dNdu2_impl(u1_, u2_):
    #     out = np.zeros_like(u_)
    #     out[:, local_dofs_V1:local_size_V] = 1.0
    #     return out.reshape(-1)

    # def dNdgradu1_impl(u1_, u2_):
    #     out = np.zeros_like(u_)
    #     out[:, local_dofs_V1:local_size_V] = 1.0
    #     return out.reshape(-1)

    # def N_external(derivatives):
    #     if derivatives == (0, 0):
    #         return N_impl
    #     elif derivatives == (1, 0):
    #         return dNdu2_impl
    #     elif derivative == (0, 1):
    #         return dNdgradu1_impl
    #     else:
    #         raise NotImplementedError

    # # u2 from V2 will be projected onto both V1 and V2
    # N = FEMExternalOperator(u1, u2, function_space=V, name="N", external_function=N_external)

    # N1, N2 = split(N)
    # v1, v2 = split(v)
    # F = N1 * v1 * ufl.dx + inner(N2, v) * ufl.dx
    # N1_explicit = u1 + inner(u2, u2)
    # N2_explicit = u2
    # F_explicit = N1_explicit * v1 * ufl.dx + inner(N2_explicit, v) * ufl.dxx

    # check_vector_matrix(F, F_explicit, u)
