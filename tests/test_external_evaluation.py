# Author: Andrey Latyshev
# Test external evaluation of external operators
# Test is based on the heat equation tutorial: `nonlinear_heat_equation_part2.py`.

from mpi4py import MPI

import numpy as np

import basix
import ufl
import ufl.algorithms
from dolfinx import fem, mesh
from dolfinx_external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    evaluate_operands,
    replace_external_operators,
)
from ufl import Measure, TestFunction, TrialFunction, derivative, grad, inner


def test_heat_equation():
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    V = fem.functionspace(domain, ("P", 1))

    T = fem.Function(V)
    sigma = grad(T)

    T.interpolate(lambda x: x[0] ** 2 + x[1])

    quadrature_degree = 2
    Qe = basix.ufl.quadrature_element(domain.topology.cell_name(), degree=quadrature_degree, value_shape=(2,))
    Q = fem.functionspace(domain, Qe)
    dx = Measure("dx", metadata={"quadrature_scheme": "default", "quadrature_degree": quadrature_degree})

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
            return NotImplementedError

    q_ = FEMExternalOperator(T, sigma, function_space=Q, external_function=q_external)
    T_tilde = TestFunction(V)

    F = inner(q_, grad(T_tilde)) * dx

    T_hat = TrialFunction(V)
    J = derivative(F, T, T_hat)
    J_expanded = ufl.algorithms.expand_derivatives(J)
    F_replaced, F_external_operators = replace_external_operators(F)
    J_replaced, J_external_operators = replace_external_operators(J_expanded)
    evaluated_operands = evaluate_operands(F_external_operators)
    _ = evaluate_external_operators(F_external_operators, evaluated_operands)

    _ = evaluate_external_operators(J_external_operators, evaluated_operands)
    F_compiled = fem.form(F_replaced)
    J_compiled = fem.form(J_replaced)
    b_vector = fem.assemble_vector(F_compiled)
    A_matrix = fem.assemble_matrix(J_compiled)

    k_explicit = 1.0 / (A + B * T)
    q_explicit = -k_explicit * sigma
    F_explicit = inner(q_explicit, grad(T_tilde)) * dx
    F_explicit_compiled = fem.form(F_explicit)
    b_explicit_vector = fem.assemble_vector(F_explicit_compiled)
    assert np.allclose(b_explicit_vector.array, b_vector.array)

    J_explicit = ufl.derivative(F_explicit, T, T_hat)
    J_explicit_compiled = fem.form(J_explicit)
    A_explicit_matrix = fem.assemble_matrix(J_explicit_compiled)
    assert np.allclose(A_explicit_matrix.to_dense(), A_matrix.to_dense())

    J_manual = (
        inner(B * k_explicit**2 * sigma * T_hat, grad(T_tilde)) * dx
        + inner(-k_explicit * ufl.Identity(2) * grad(T_hat), grad(T_tilde)) * dx
    )
    J_manual_compiled = fem.form(J_manual)
    A_manual_matrix = fem.assemble_matrix(J_manual_compiled)
    assert np.allclose(A_manual_matrix.to_dense(), A_matrix.to_dense())
