# Mixed element space examples for external operators
# Copied from test/test_external_operators_evaluation.py

from mpi4py import MPI
import numpy as np
import basix.ufl
from dolfinx import fem, mesh
import ufl
from ufl import split, inner, grad, TestFunction, TrialFunction
from dolfinx_external_operator import (
    FEMExternalOperator,
    replace_external_operators,
    evaluate_operands,
    evaluate_external_operators,
)

def test_mixed_element_space():
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    gdim = domain.geometry.dim

    # 1. Simple mixed space setup
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
    pts_V1 = V1.element.interpolation_points.shape[0]
    pts_V2 = V2.element.interpolation_points.shape[0]
    pts_total = pts_V1 + pts_V2

    def N_impl(u_):
        out = np.zeros_like(u_)
        out[:, pts_V1:pts_total] = u_[:, pts_V1:pts_total]
        return out.reshape(-1)

    def dN_impl(u_):
        out = np.zeros_like(u_)
        out[:, pts_V1:pts_total] = 1.0
        return out.reshape(-1)

    def N_external(derivatives):
        if derivatives == (0,):
            return N_impl
        elif derivatives == (1,):
            return dN_impl
        else:
            raise NotImplementedError

    N_tensor = FEMExternalOperator(u2, function_space=V, name="N", external_function=N_external)
    N1, N2 = split(N_tensor)
    v1, v2 = split(v)
    F = N1 * v1 * ufl.dx + inner(grad(N2), v) * ufl.dx

    # 2. More complex (scalar + vector) mixed case
    Ve1 = basix.ufl.element("P", domain.topology.cell_name(), degree=4, shape=())
    Ve2 = basix.ufl.element("P", domain.topology.cell_name(), degree=2, shape=(gdim,))
    V = fem.functionspace(domain, basix.ufl.mixed_element([Ve1, Ve2]))
    u = fem.Function(V)
    u.sub(0).interpolate(lambda x: x[1] + 2.0)
    u.sub(1).interpolate(lambda x: (x[0], x[1]))
    u1, u2 = split(u)
    v = TestFunction(V)

    V1 = V.sub(0)
    V2 = V.sub(1)
    pts_V1 = V1.element.interpolation_points.shape[0]
    pts_V2 = V2.element.interpolation_points.shape[0]
    pts_total = pts_V1 + pts_V2

    # N = [N1, N2]
    # N1 = u1 + inner(u2, u2), N2 = u2
    def N_tensor_impl(u1_, u2_):
        n_cells = u2_.shape[0]
        out = np.zeros((n_cells, pts_total, 2), dtype=u2_.dtype)
        u1_vals = u1_.reshape(n_cells, -1)
        u2_first = u2_[:, :pts_V1, :]
        out[:, :pts_V1, 0] = u1_vals[:, :pts_V1] + np.einsum("...i,...i->...", u2_first, u2_first)
        out[:, pts_V1:, :] = u2_[:, pts_V1:, :]
        return out.reshape(-1)

    def dNdu1_impl(u1_, u2_):
        n_cells = u2_.shape[0]
        out = np.zeros((n_cells, pts_total, 2), dtype=u2_.dtype)
        out[:, :pts_V1, 0] = 1.0
        return out.reshape(-1)

    def dNdu2_impl(u1_, u2_):
        n_cells = u2_.shape[0]
        out = np.zeros((n_cells, pts_total, 4), dtype=u2_.dtype)
        out[:, :pts_V1, 0:2] = 2.0 * u2_[:, :pts_V1, :]
        out[:, pts_V1:, 0] = 1.0
        out[:, pts_V1:, 3] = 1.0
        return out.reshape(-1)

    def N_tensor_external(derivatives):
        if derivatives == (0, 0):
            return N_tensor_impl
        elif derivatives == (1, 0):
            return dNdu1_impl
        elif derivatives == (0, 1):
            return dNdu2_impl
        else:
            raise NotImplementedError

    N = FEMExternalOperator(u1, u2, function_space=V, name="N", external_function=N_tensor_external)


def test_mixed_cg_dg_space():
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    gdim = domain.geometry.dim

    Ve_u1 = basix.ufl.element("P", domain.topology.cell_name(), degree=2, shape=())
    Ve_u2 = basix.ufl.element("DG", domain.topology.cell_name(), degree=1, shape=(gdim,))
    V_u = fem.functionspace(domain, basix.ufl.mixed_element([Ve_u1, Ve_u2]))
    u = fem.Function(V_u)
    u.sub(0).interpolate(lambda x: x[0] ** 2 + x[1])
    u.sub(1).interpolate(lambda x: (x[0] - x[1], x[0] + x[1]))
    u1, u2 = split(u)

    Ve1 = basix.ufl.element("P", domain.topology.cell_name(), degree=4, shape=())
    Ve2 = basix.ufl.element("DG", domain.topology.cell_name(), degree=3, shape=(gdim,))
    V = fem.functionspace(domain, basix.ufl.mixed_element([Ve1, Ve2]))
    v = TestFunction(V)

    V1 = V.sub(0)
    V2 = V.sub(1)
    pts_V1 = V1.element.interpolation_points.shape[0]
    pts_V2 = V2.element.interpolation_points.shape[0]
    pts_total = pts_V1 + pts_V2

    def N_tensor_impl(u1_, u2_):
        n_cells = u1_.shape[0]
        out = np.zeros((n_cells, pts_total, 2), dtype=u1_.dtype)
        out[:, :pts_V1, 0] = u1_[:, :pts_V1] ** 2
        out[:, pts_V1:, 0:2] = u1_[:, pts_V1:, np.newaxis] * u2_[:, pts_V1:, :]
        return out.reshape(-1)

    def dNdu1_impl(u1_, u2_):
        n_cells = u1_.shape[0]
        out = np.zeros((n_cells, pts_total, 2), dtype=u1_.dtype)
        out[:, :pts_V1, 0] = 2.0 * u1_[:, :pts_V1]
        out[:, pts_V1:, 0:2] = u2_[:, pts_V1:, :]
        return out.reshape(-1)

    def dNdu2_impl(u1_, u2_):
        n_cells = u1_.shape[0]
        out = np.zeros((n_cells, pts_total, 4), dtype=u1_.dtype)
        out[:, pts_V1:, 0] = u1_[:, pts_V1:]
        out[:, pts_V1:, 3] = u1_[:, pts_V1:]
        return out.reshape(-1)

    def N_external(derivatives):
        if derivatives == (0, 0):
            return N_tensor_impl
        elif derivatives == (1, 0):
            return dNdu1_impl
        elif derivatives == (0, 1):
            return dNdu2_impl
        else:
            raise NotImplementedError

    N = FEMExternalOperator(u1, u2, function_space=V, name="N", external_function=N_external)
