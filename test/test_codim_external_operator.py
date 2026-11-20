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


def f_impl(uuh_square):
    output = uuh_square * np.sqrt(uuh_square)  # NUMPY
    return output.reshape(-1)  # The output must be returned flattened to one dimension


def dfdu_impl(uuh_square):
    # Derivative with respect to the operand s = u**2; the direction supplies 2 * u * v
    aa = 1.5 * np.sqrt(uuh_square)  # NUMPY
    return aa.reshape(-1)  # The output must be returned flattened to one dimension


def f_external(derivatives):
    if derivatives == (0,):  # no derivation, the function itself
        return f_impl
    elif derivatives == (1,):  # the derivative with respect to the operand `uh`
        return dfdu_impl
    else:
        return NotImplementedError


def assemble(form):
    """Assemble a functional to a scalar, or a linear form to its owned entries."""
    if form.rank == 0:
        value = dolfinx.fem.assemble_scalar(form)
        return form.mesh.comm.allreduce(value, op=MPI.SUM)
    b = dolfinx.fem.assemble_vector(form)
    b.scatter_reverse(dolfinx.la.InsertMode.add)
    return b.array[: b.index_map.size_local * b.block_size]


@pytest.mark.parametrize("quadrature_degree", range(1, 5))
def test_external_operator_codim_1(quadrature_degree):
    """Test assembly of codim 1 external operator"""

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 5, 5)
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, tdim - 1)
    c_to_f = mesh.topology.connectivity(tdim, tdim - 1)
    mesh.topology.create_connectivity(tdim - 1, tdim)
    f_to_c = mesh.topology.connectivity(tdim - 1, tdim)

    ext_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: x[0] + x[1])

    submesh, entity_map, _, _ = dolfinx.mesh.create_submesh(mesh, mesh.topology.dim - 1, ext_facets)
    parent_to_sub = np.empty((len(ext_facets), 2), dtype=np.int32)

    for i, facet in enumerate(ext_facets):
        cells = f_to_c.links(facet)
        cell = cells[0]
        local_facets = c_to_f.links(cell)
        local_pos = np.flatnonzero(local_facets == facet)
        parent_to_sub[i, 0] = cell
        parent_to_sub[i, 1] = local_pos[0]
    tdim = mesh.topology.dim

    ext_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

    Qe = basix.ufl.quadrature_element(submesh.basix_cell(), degree=quadrature_degree, value_shape=())
    Q = dolfinx.fem.functionspace(submesh, Qe)

    g = FEMExternalOperator(u, function_space=Q, external_function=g_external)  # g is now Symbolic not numpy involved

    ft = dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, ext_facets, np.full_like(ext_facets, 1))
    ds = ufl.Measure(
        "ds", domain=mesh, subdomain_data=ft, subdomain_id=1, metadata={"quadrature_degree": quadrature_degree}
    )

    for derivative in [0, 1]:
        if derivative == 0:
            J = g * ds
        else:
            J = ufl.derivative(g, u) * ds

        J_replaced, J_external_operators = replace_external_operators(J)

        J_compiled = dolfinx.fem.form(J_replaced, entity_maps=[entity_map])
        # Pack coefficients for g
        evaluated_operands = evaluate_operands(J_external_operators, entities=parent_to_sub)
        _ = evaluate_external_operators(J_external_operators, evaluated_operands)

        # Exact solution: the derivative is contracted with the direction, the test function
        if derivative == 0:
            J_exact = ufl.cos(u) * ds
        else:
            J_exact = ufl.sin(u) * ufl.TestFunction(V) * ds

        np.testing.assert_allclose(assemble(J_compiled), assemble(dolfinx.fem.form(J_exact)))


@pytest.mark.parametrize("quadrature_degree", range(1, 5))
def test_external_operator_codim_0(quadrature_degree):
    """Test assembly of codim 0 external operator"""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 5, 5)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: x[0] + x[1])

    def bottom(x):
        return x[0] <= 0.2 + 1e-10

    cells = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, bottom)
    ct = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, cells, np.full_like(cells, 1))
    submesh, entity_map, _, _ = dolfinx.mesh.create_submesh(mesh, mesh.topology.dim, ct.find(1))

    Qe = basix.ufl.quadrature_element(submesh.basix_cell(), degree=quadrature_degree, value_shape=())
    Q = dolfinx.fem.functionspace(submesh, Qe)

    f = FEMExternalOperator(
        u * u, function_space=Q, external_function=f_external
    )  # g is now Symbolic not numpy involved

    dx = ufl.Measure(
        "dx", domain=mesh, subdomain_data=ct, subdomain_id=1, metadata={"quadrature_degree": quadrature_degree}
    )

    for derivative in [0, 1]:
        if derivative == 0:
            J = f * dx
        else:
            J = ufl.derivative(f, u) * dx

        J_replaced, J_external_operators = replace_external_operators(J)
        map_c = submesh.topology.index_map(submesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts
        cells = np.arange(0, num_cells, dtype=np.int32)
        submesh_cells_to_parent = entity_map.sub_topology_to_topology(cells, inverse=False)
        # Pack coefficients for g
        evaluated_operands = evaluate_operands(J_external_operators, entities=submesh_cells_to_parent)
        _ = evaluate_external_operators(J_external_operators, evaluated_operands)

        J_compiled = dolfinx.fem.form(J_replaced, entity_maps=[entity_map])

        # Exact solution: f'(u**2) contracted with the direction 2 * u * v
        if derivative == 0:
            J_exact = u**3 * dx
        else:
            J_exact = 3 * u**2 * ufl.TestFunction(V) * dx

        np.testing.assert_allclose(assemble(J_compiled), assemble(dolfinx.fem.form(J_exact)))
