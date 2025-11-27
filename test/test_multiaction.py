# Test nested usage of action with external operators
# Author: Jørgen S. Dokken

from mpi4py import MPI

import numpy as np
import pytest

import basix.ufl
import dolfinx
import dolfinx_external_operator
import ufl


@pytest.mark.parametrize("q_deg", [1, 3, 5])
def test_nested_action(q_deg):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)

    T = dolfinx.fem.functionspace(mesh, ("DG", 0, (3,)))
    theta = dolfinx.fem.Function(T)
    theta.interpolate(lambda x: (0.1 * x[0], 0.2 * x[1] + 0.3 * x[1], x[0] * x[1]))

    def f_mod(mod, theta, x):
        return mod.sin(mod.pi * theta[0] * x[0]) * mod.cos(mod.pi * theta[2] * x[1]) + 3.11 * theta[1]

    def f(theta, x):
        theta = theta.reshape(-1, 3).T
        x = x.reshape(-1, 2).T
        return f_mod(np, theta, x).flatten()

    def df_dtheta(theta, x):
        theta = theta.reshape(-1, 3).T
        x = x.reshape(-1, 2).T
        df_dtheta0 = np.pi * x[0] * np.cos(np.pi * theta[0] * x[0]) * np.cos(np.pi * theta[2] * x[1])
        df_dtheta1 = np.full(theta.shape[1], 3.11)
        df_dtheta2 = -np.pi * x[1] * np.sin(np.pi * theta[0] * x[0]) * np.sin(np.pi * theta[2] * x[1])
        return np.array([df_dtheta0, df_dtheta1, df_dtheta2]).T.flatten()

    def df_dx(theta, x):
        theta = theta.reshape(-1, 3).T
        x = x.reshape(-1, 2).T
        df_dx0 = np.pi * theta[0] * np.cos(np.pi * theta[0] * x[0]) * np.cos(np.pi * theta[2] * x[1])
        df_dx1 = -np.pi * theta[2] * np.sin(np.pi * theta[0] * x[0]) * np.sin(np.pi * theta[2] * x[1])
        return np.array([df_dx0, df_dx1]).T.flatten()

    def f_ext(derivatives):
        if derivatives == (0, 0):
            return f
        elif derivatives == (1, 0):
            return df_dtheta
        elif derivatives == (0, 1):
            return df_dx
        else:
            raise NotImplementedError(f"No external function is defined for the requested derivative {derivatives}.")

    Qe = basix.ufl.quadrature_element(mesh.topology.cell_name(), degree=q_deg)
    Q = dolfinx.fem.functionspace(mesh, Qe)
    x = ufl.SpatialCoordinate(mesh)
    N = dolfinx_external_operator.FEMExternalOperator(theta, x, function_space=Q, external_function=f_ext)

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    F = ufl.inner(u, v) * ufl.dx - ufl.inner(N, v) * ufl.dx

    # Create dummy variables for forward and adjoint solution
    uh = dolfinx.fem.Function(V, name="uh")
    uh.interpolate(lambda x: np.sin(x[0]))
    lmbda = dolfinx.fem.Function(V, name="lmbda")
    lmbda.interpolate(lambda x: np.cos(x[1]))

    F = ufl.action(ufl.action(F, uh), lmbda)
    dFdtheta = ufl.derivative(F, theta)

    replaced_form, external_operators = dolfinx_external_operator.replace_external_operators(dFdtheta)

    assert len(external_operators) == 1

    operands = dolfinx_external_operator.evaluate_operands(external_operators)
    dolfinx_external_operator.evaluate_external_operators(external_operators, operands)

    vec = dolfinx.fem.assemble_vector(dolfinx.fem.form(replaced_form))

    f_ref = f_mod(ufl, theta, x)
    dfdtheta = ufl.diff(f_ref, theta)
    ref_form = (
        -ufl.inner(dfdtheta, ufl.TestFunction(theta.function_space))
        * lmbda
        * ufl.dx(metadata={"quadrature_degree": q_deg})
    )
    vec_ref = dolfinx.fem.assemble_vector(dolfinx.fem.form(ref_form))
    np.testing.assert_allclose(vec.array, vec_ref.array, rtol=1e-10, atol=1e-10)
