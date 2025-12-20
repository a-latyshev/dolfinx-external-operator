from mpi4py import MPI

import basix
import ufl
from ufl.algorithms import expand_derivatives
from ufl.algorithms.renumbering import renumber_indices
from dolfinx import fem, mesh
from dolfinx_external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    evaluate_operands,
    replace_external_operators,
)

import numpy as np

def test_operands_evaluation():
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = fem.functionspace(domain, ("P", 1, (domain.geometry.dim,)))

    u = fem.Function(V, name="u")
    u.interpolate(lambda x: (x[0] * 0.1, x[1] * 0.3))
    v = ufl.TestFunction(V)

    # Measure 
    quadrature_degree = 2
    dx = ufl.Measure(
        "dx",
        domain=domain,
        metadata={"quadrature_scheme": "default", "quadrature_degree": quadrature_degree},
    )

    d = domain.geometry.dim
    I = ufl.Identity(d)
    F = ufl.variable(I + ufl.grad(u))
    C = F.T * F
    J = ufl.det(F)
    I1 = ufl.tr(C)

    R = fem.functionspace(domain, ("P", 1, ()))
    slope = fem.Function(R, name="slope")
    slope.x.array[:] = 1.0

    Qe = basix.ufl.quadrature_element(
        domain.topology.cell_name(),
        degree=quadrature_degree,
        value_shape=(),
    )
    Q = fem.functionspace(domain, Qe)

    N = FEMExternalOperator(I1, slope, function_space=Q, name="N")

    map_c = domain.topology.index_map(domain.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)
    quadrature_points = basix.make_quadrature(Qe.cell_type, quadrature_degree)[0]

    I1_expr = fem.Expression(I1, quadrature_points, dtype=N.ref_coefficient.dtype)
    I1_values = I1_expr.eval(N.ref_function_space.mesh, cells)
    slope_values = np.full((quadrature_points.shape[0]*num_cells,), slope.x.array[0], dtype=N.ref_coefficient.dtype)

    P = ufl.diff(I1, F) * N
    Res = ufl.inner(P, ufl.grad(v)) * dx 
    J = ufl.derivative(Res, u, ufl.TrialFunction(V))
    _, J_external_operators = replace_external_operators(J)
    evaluated_operands = evaluate_operands(J_external_operators)
    I1_expanded = N.ufl_operands[0]
    np.testing.assert_allclose(I1_values.reshape(-1), evaluated_operands[I1_expanded].reshape(-1))
    np.testing.assert_allclose(slope_values.reshape(-1), evaluated_operands[slope].reshape(-1))