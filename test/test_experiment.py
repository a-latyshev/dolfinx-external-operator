# Author: Andrey Latyshev
# Test spatial gradient of external operators in a variational formulation

from mpi4py import MPI
import numpy as np
import basix
import ufl
from dolfinx import fem, mesh
from dolfinx_external_operator import (
    FEMExternalOperator,
    replace_external_operators,
    evaluate_operands,
    evaluate_external_operators,
)
from ufl import Measure, TestFunction, grad, inner


def test_grad_external_operator():
    """Test a variational form containing the spatial gradient of an external operator:
    F = grad(N(u)) . grad(v) * dx.
    Since N(u) = u^2 is nonlinear and interpolated onto a P1 space, there will be
    a finite element interpolation error compared to the analytical gradient of u^2,
    which is verified with an appropriate tolerance.
    """
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    V = fem.functionspace(domain, ("P", 1))
    u = fem.Function(V, name="u")
    u.interpolate(lambda x: x[0]**2 + x[1])

    # N(u) = u^2
    def N_impl(u_):
        return (u_**2).flatten()

    def dNdu_impl(u_):
        return (2 * u_).flatten()

    def N_external(derivatives):
        if derivatives == (0,):
            return N_impl
        elif derivatives == (1,):
            return dNdu_impl
        else:
            raise NotImplementedError

    N = FEMExternalOperator(u, function_space=V, name="N", external_function=N_external)
    v = TestFunction(V)

    # Variational form: F = grad(N) . grad(v) * dx
    dx = Measure("dx")
    F = inner(grad(N), grad(v)) * dx

    F_expanded = ufl.algorithms.expand_derivatives(F)
    F_replaced, F_external_operators = replace_external_operators(F_expanded)

    # Explicit formulation
    N_explicit = u**2
    F_explicit = inner(grad(N_explicit), grad(v)) * dx

    evaluated_operands = evaluate_operands(F_external_operators)
    evaluate_external_operators(F_external_operators, evaluated_operands)

    F_compiled = fem.form(F_replaced)
    F_explicit_compiled = fem.form(F_explicit)

    b_vector = fem.assemble_vector(F_compiled)
    b_explicit_vector = fem.assemble_vector(F_explicit_compiled)

    # Assert with looser tolerance due to FE interpolation error of u^2 onto P1 space
    assert np.allclose(b_explicit_vector.array, b_vector.array, atol=3e-2, rtol=3e-2)
