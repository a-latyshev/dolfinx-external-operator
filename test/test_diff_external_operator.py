"""Test an external energy differentiated with ufl.diff, and directions in argument slots."""

from collections import Counter

from mpi4py import MPI

import numpy as np
import pytest

import basix.ufl
import ufl
from dolfinx import fem, mesh
from dolfinx_external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    evaluate_operands,
    replace_external_operators,
)


def assemble(form):
    """Assemble a linear form to its owned entries, or a bilinear form to a dense local block."""
    compiled = fem.form(form)
    if compiled.rank == 1:
        b = fem.assemble_vector(compiled)
        return b.array[: b.index_map.size_local * b.block_size]
    return fem.assemble_matrix(compiled).to_dense()


def evaluate(form):
    """Replace, evaluate and assemble a form with external operators."""
    replaced, operators = replace_external_operators(form)
    evaluate_external_operators(operators, evaluate_operands(operators))
    return assemble(replaced)


@pytest.mark.parametrize("dim", [2, 3])
def test_stress_and_tangent_of_external_energy(dim):
    if dim == 2:
        domain = mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    else:
        domain = mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    V = fem.functionspace(domain, ("Lagrange", 1, (dim,)))
    u = fem.Function(V)
    u.interpolate(lambda x: np.vstack([0.1 * x[0] * x[1], -0.2 * x[0] ** 2, 0.05 * x[2]][:dim]))
    v, du = ufl.TestFunction(V), ufl.TrialFunction(V)
    F = ufl.variable(ufl.Identity(dim) + ufl.grad(u))
    invariant = ufl.tr(F.T * F) - dim

    calls = Counter()

    def W_external(derivatives):
        """Derivatives of W(I) = I**2 / 4."""
        calls[derivatives] += 1
        if derivatives == (1,):
            return lambda s: (s / 2).reshape(-1)
        if derivatives == (2,):
            return lambda s: np.full(s.size, 0.5)
        raise NotImplementedError(f"No external function for derivatives {derivatives}.")

    Qe = basix.ufl.quadrature_element(domain.topology.cell_name(), value_shape=(), degree=2)
    W = FEMExternalOperator(invariant, function_space=fem.functionspace(domain, Qe), external_function=W_external)
    try:
        stress = ufl.algorithms.expand_derivatives(ufl.diff(W, F))
    except (NotImplementedError, ValueError):
        pytest.skip("UFL cannot differentiate external operators with ufl.diff (needs FEniCS/ufl#568).")

    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2})
    residual = ufl.inner(stress, ufl.grad(v)) * dx
    reference = ufl.inner(ufl.diff(invariant**2 / 4, F), ufl.grad(v)) * dx

    # One evaluation per derivative order, although diff makes one operator per component of F
    np.testing.assert_allclose(evaluate(residual), assemble(reference), atol=1e-12)
    assert calls == {(1,): 1}

    calls.clear()
    tangent = ufl.derivative(residual, u, du)
    np.testing.assert_allclose(evaluate(tangent), assemble(ufl.derivative(reference, u, du)), atol=1e-12)
    assert calls == {(1,): 1, (2,): 1}
