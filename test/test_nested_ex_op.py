from mpi4py import MPI

import numpy as np
import pytest

import basix.ufl
import dolfinx
import ufl
from dolfinx_external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    evaluate_operands,
    replace_external_operators,
)


def compile_external_operator_form(
    form: dolfinx.fem.Form,
    jit_options: dict | None = None,
    form_compiler_options: dict | None = None,
    entity_maps: list[dolfinx.mesh.EntityMap] | None = None,
) -> dolfinx.fem.Form:
    form_replaced, ex_ops = replace_external_operators(form)
    compiled_form = dolfinx.fem.form(
        form_replaced,
        jit_options=jit_options,
        form_compiler_options=form_compiler_options,
        entity_maps=entity_maps,
    )
    compiled_form._ex_ops = ex_ops  # type: ignore[attr-defined]
    return compiled_form


def pack_external_operator_data(form: dolfinx.fem.Form | list[dolfinx.fem.Form]):
    if isinstance(form, dolfinx.fem.Form):
        external_operators = form._ex_ops  # type: ignore[attr-defined]
        if len(external_operators) == 0:
            return
        operands = evaluate_operands(external_operators)
        evaluate_external_operators(external_operators, operands)
    else:
        for f in form:
            external_operators = f._ex_ops  # type: ignore[attr-defined]
            if len(external_operators) == 0:
                continue
            operands = evaluate_operands(external_operators)
            evaluate_external_operators(external_operators, operands)


def p(mod, u_NN):
    return mod.sin(u_NN)


def z(mod, u_NN, p_NN):
    return mod.cos(u_NN) + p_NN**2


def p_NN(mod, derivatives):
    if derivatives == (0,):
        return lambda u_NN: p(mod, u_NN)
    else:
        raise RuntimeError(f"Not implemented for derivative {derivatives}")


def z_NN(mod, derivatives):
    if derivatives == (0, 0):
        return lambda u_NN, p_NN: z(mod, u_NN, p_NN)
    else:
        raise RuntimeError(f"Not implemented for derivative {derivatives}")


def u_NN(gdim, mod, x, theta):
    if gdim == 1:
        return mod.sin(theta[0] * x[0])
    else:
        return mod.sin(theta[0] * x[0]) * mod.sin(theta[1] * x[1])


def u_NN_impl(gdim, x, theta):
    x_vec = x.reshape(-1, gdim)
    theta_vec = theta.reshape(-1, 4)
    out = u_NN(gdim, np, x_vec.T, theta_vec.T)
    return out.flatten().copy()


def u_NN_np(gdim, derivatives):
    if derivatives == (0, 0):
        return lambda x, theta: u_NN_impl(gdim, x, theta)
    else:
        raise RuntimeError(f"No function is defined for the derivatives {derivatives}.")


@pytest.mark.parametrize("q_deg", [1, 4, 8])
@pytest.mark.parametrize("N", [4, 8, 12])
@pytest.mark.parametrize(
    "cell_type",
    [
        dolfinx.mesh.CellType.interval,
        dolfinx.mesh.CellType.triangle,
        dolfinx.mesh.CellType.quadrilateral,
        dolfinx.mesh.CellType.tetrahedron,
        dolfinx.mesh.CellType.hexahedron,
    ],
)
def test_replacement_operator(cell_type, N, q_deg):
    tdim = dolfinx.cpp.mesh.cell_dim(cell_type)
    if tdim == 1:
        mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, N)
    if tdim == 2:
        mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=cell_type)
    elif tdim == 3:
        mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, N, N, N, cell_type=cell_type)

    R = dolfinx.fem.functionspace(mesh, ("DG", 1, (4,)))
    theta = dolfinx.fem.Function(R)
    theta.x.array[:] = 0.32
    theta.x.scatter_forward()
    x = ufl.SpatialCoordinate(mesh)

    Qe = basix.ufl.quadrature_element(mesh.basix_cell(), degree=q_deg)
    Q = dolfinx.fem.functionspace(mesh, Qe)

    # Define external operator and correct quadrature space
    N = FEMExternalOperator(
        x,
        theta,
        function_space=Q,
        external_function=lambda derivatives: u_NN_np(mesh.geometry.dim, derivatives),
        name="exop",
    )

    P = FEMExternalOperator(
        N, function_space=Q, external_function=lambda derivatives: p_NN(np, derivatives), name="second_op"
    )
    Z = FEMExternalOperator(
        N, P, function_space=Q, external_function=lambda derivatives: z_NN(np, derivatives), name="third_op"
    )
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    phi = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": q_deg})

    def F(P, N, Z, phi_h):
        a = ufl.inner(ufl.grad(phi), ufl.grad(v)) * dx
        L = (Z**2 * P * N) * v * dx
        _F = a - L
        return ufl.action(_F, phi_h)

    lmbda = dolfinx.fem.Function(V, name="lmbda")
    phih = dolfinx.fem.Function(V, name="phih")
    phih.interpolate(lambda x: np.sin(np.pi * x[0]))
    lmbda.interpolate(lambda x: np.cos(3 * np.pi * x[0]))

    F_ex = F(P, N, Z, phih)
    F_compiled = compile_external_operator_form(F_ex)

    pack_external_operator_data(F_compiled)
    vec = dolfinx.fem.assemble_vector(F_compiled)
    N_ufl = u_NN(mesh.geometry.dim, ufl, x, theta)
    P_ufl = p(ufl, N_ufl)
    Z_ufl = z(ufl, N_ufl, P_ufl)
    F_ref = F(P_ufl, N_ufl, Z_ufl, phih)
    vec_ref = dolfinx.fem.assemble_vector(dolfinx.fem.form(F_ref))
    np.testing.assert_allclose(vec.array, vec_ref.array)
