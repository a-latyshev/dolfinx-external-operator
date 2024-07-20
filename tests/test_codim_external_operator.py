# Author: Jørgen S. Dokken
# Test external operator for mixed assembly

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


def g_impl(uuh):
    output = np.cos(uuh)  # NUMPY
    return output.reshape(-1)  # The output must be returned flattened to one dimension


def dgdu_impl(uuh):
    aa = np.sin(uuh)  # NUMPY
    return aa.reshape(-1)  # The output must be returned flattened to one dimension


def g_external(derivatives):
    if derivatives == (0,):  # no derivation, the function itself
        return g_impl
    elif derivatives == (1,):  # the derivative with respect to the operand `uh`
        return dgdu_impl
    else:
        return NotImplementedError


def f_impl(uuh):
    output = uuh**3  # NUMPY
    return output.reshape(-1)  # The output must be returned flattened to one dimension


def dfdu_impl(uuh):
    aa = 3 * uuh**2  # NUMPY
    return aa.reshape(-1)  # The output must be returned flattened to one dimension


def f_external(derivatives):
    if derivatives == (0,):  # no derivation, the function itself
        return f_impl
    elif derivatives == (1,):  # the derivative with respect to the operand `uh`
        return dfdu_impl
    else:
        return NotImplementedError


@pytest.mark.parametrize("quadrature_degree", range(1, 5))
def test_external_operator_codim_1(quadrature_degree):
    """Test assembly of codim 1 external operator"""

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 5, 5)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    ext_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: x[0] + x[1])

    submesh, sub_to_parent, _, _ = dolfinx.mesh.create_submesh(mesh, mesh.topology.dim - 1, ext_facets)
    num_entities = mesh.topology.index_map(mesh.topology.dim - 1).size_local
    parent_to_sub = np.full(num_entities, -1, dtype=np.int32)
    parent_to_sub[sub_to_parent] = np.arange(len(sub_to_parent), dtype=np.int32)
    entity_maps = {submesh: parent_to_sub}

    Qe = basix.ufl.quadrature_element(submesh.basix_cell(), degree=quadrature_degree, value_shape=())
    Q = dolfinx.fem.functionspace(submesh, Qe)

    g = FEMExternalOperator(u, function_space=Q)  # g is now Symbolic not numpy involved
    g.external_function = g_external

    ft = dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, ext_facets, np.full_like(ext_facets, 1))
    ds = ufl.Measure(
        "ds", domain=mesh, subdomain_data=ft, subdomain_id=1, metadata={"quadrature_degree": quadrature_degree}
    )

    for derivative in [0, 1]:
        if derivative == 0:
            J = g * ds
        else:
            J = ufl.algorithms.expand_derivatives(ufl.derivative(g, u) * ds)

        J_replaced, J_external_operators = replace_external_operators(J)
        J_compiled = dolfinx.fem.form(J_replaced, entity_maps=entity_maps)
        # Pack coefficients for g
        evaluated_operands = evaluate_operands(J_external_operators, entity_maps=entity_maps)
        _ = evaluate_external_operators(J_external_operators, evaluated_operands)

        Jh = dolfinx.fem.assemble_scalar(J_compiled)
        Jh = mesh.comm.allreduce(Jh, op=MPI.SUM)

        # Exact solution
        if derivative == 0:
            J_exact = ufl.cos(u) * ds
        else:
            J_exact = ufl.sin(u) * ds

        J_exact_compiled = dolfinx.fem.form(J_exact)
        J_ex = dolfinx.fem.assemble_scalar(J_exact_compiled)
        J_ref = mesh.comm.allreduce(J_ex, op=MPI.SUM)
        np.testing.assert_allclose(Jh, J_ref)


@pytest.mark.parametrize("quadrature_degree", range(1, 5))
def test_external_operator_codim_0(quadrature_degree):
    """Test assembly of codim 1 external operator"""

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 5, 5)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: x[0] + x[1])

    def bottom(x):
        return x[0] <= 0.2 + 1e-10

    cells = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, bottom)
    ct = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, cells, np.full_like(cells, 1))
    submesh, sub_to_parent, _, _ = dolfinx.mesh.create_submesh(mesh, mesh.topology.dim, ct.find(1))

    num_entities = (
        mesh.topology.index_map(mesh.topology.dim).size_local + mesh.topology.index_map(mesh.topology.dim).num_ghosts
    )
    parent_to_sub = np.full(num_entities, -1, dtype=np.int32)
    parent_to_sub[sub_to_parent] = np.arange(len(sub_to_parent), dtype=np.int32)
    entity_maps = {submesh: parent_to_sub}
    Qe = basix.ufl.quadrature_element(submesh.basix_cell(), degree=quadrature_degree, value_shape=())
    Q = dolfinx.fem.functionspace(submesh, Qe)

    f = FEMExternalOperator(u, function_space=Q)  # g is now Symbolic not numpy involved
    f.external_function = f_external

    dx = ufl.Measure(
        "dx", domain=mesh, subdomain_data=ct, subdomain_id=1, metadata={"quadrature_degree": quadrature_degree}
    )

    for derivative in [0, 1]:
        if derivative == 0:
            J = f * dx
        else:
            J = ufl.algorithms.expand_derivatives(ufl.derivative(f, u) * dx)

        J_replaced, J_external_operators = replace_external_operators(J)
        J_compiled = dolfinx.fem.form(J_replaced, entity_maps=entity_maps)
        # Pack coefficients for g
        evaluated_operands = evaluate_operands(J_external_operators, entity_maps=entity_maps)
        _ = evaluate_external_operators(J_external_operators, evaluated_operands)

        Jh = dolfinx.fem.assemble_scalar(J_compiled)
        Jh = mesh.comm.allreduce(Jh, op=MPI.SUM)

        # Exact solution
        if derivative == 0:
            J_exact = u**3 * dx
        else:
            J_exact = 3 * u**2 * dx

        J_exact_compiled = dolfinx.fem.form(J_exact)
        J_ex = dolfinx.fem.assemble_scalar(J_exact_compiled)
        J_ref = mesh.comm.allreduce(J_ex, op=MPI.SUM)
        np.testing.assert_allclose(Jh, J_ref)
