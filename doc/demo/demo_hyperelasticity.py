# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: dolfinx-env
#     language: python
#     name: python3
# ---

# %%
"""
Demo: Neo-Hookean Hyperelasticity - Tensile Test of a Rectangular Specimen (2D)
Author: Adapted from FEniCSx/dolfinx and jorgensd/dolfinx-tutorial
"""

# %%
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh, fem, log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.mesh import CellType
from dolfinx.io import XDMFFile

# %%
# Geometry and mesh (2D)
L = 1.0  # Length
W = 1.0  # Height
nx, ny = 40, 40

# %%
domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [L, W]], [nx, ny], cell_type=CellType.quadrilateral)

# %%
V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))


# %%
def bottom(x):
    return np.isclose(x[1], 0.0)
def top(x):
    return np.isclose(x[1], W)


# %%
fdim = domain.topology.dim - 1
bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom)
top_facets = mesh.locate_entities_boundary(domain, fdim, top)

# %%
marked_facets = np.hstack([bottom_facets, top_facets])
marked_values = np.hstack([np.full_like(bottom_facets, 1), np.full_like(top_facets, 2)])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

bottom_dofs = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.find(1))
top_dofs = fem.locate_dofs_topological(V.sub(1), facet_tag.dim, facet_tag.find(2))
u_D_bottom = np.array((0,) * domain.geometry.dim, dtype=default_scalar_type)
u_D_top = fem.Constant(domain, 0.0)

bcs_u = [
    fem.dirichletbc(u_D_top, top_dofs, V.sub(1)),
    fem.dirichletbc(u_D_bottom, bottom_dofs, V),
]


# %%
u = fem.Function(V)
v = ufl.TestFunction(V)
d = len(u)
I = ufl.variable(ufl.Identity(d))
F = ufl.variable(I + ufl.grad(u))
C = ufl.variable(F.T * F)
Ic = ufl.variable(ufl.tr(C))
J = ufl.variable(ufl.det(F))

# %%
# Material parameters (neo-Hookean)
E = default_scalar_type(1.0e4)
nu = default_scalar_type(0.3)
mu = fem.Constant(domain, E / (2 * (1 + nu)))
lmbda = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))
# Strain energy and first Piola-Kirchhoff stress
psi = (mu / 2) * (Ic - 2) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J)) ** 2
P = ufl.diff(psi, F)
dP = ufl.diff(P, F)

# %%
# Measures for integration
metadata = {"quadrature_degree": 2}
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag, metadata=metadata)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

# %%
# from dolfinx.fem import Expression, functionspace, Function
# import basix

# # Choose quadrature degree and get quadrature points for the cell type
# quadrature_degree = 2
# cell_name = domain.topology.cell_name()
# quadrature_points, _ = basix.make_quadrature(getattr(basix.CellType, cell_name), quadrature_degree)

# # Create a quadrature element and function space for tensor-valued P
# Qe = basix.ufl.quadrature_element(cell_name, value_shape=(d, d), degree=quadrature_degree, scheme="default")
# Q = functionspace(domain, Qe)

# # Create Expression for P at quadrature points
# P_expr = Expression(P, quadrature_points, dtype=default_scalar_type)

# # Evaluate P at all cells (including ghosts)
# map_c = domain.topology.index_map(domain.topology.dim)
# num_cells = map_c.size_local + map_c.num_ghosts
# cells = np.arange(num_cells, dtype=np.int32)
# P_eval = P_expr.eval(domain, cells)

# # Optionally, assemble into a Function for visualization/post-processing
# P_func = Function(Q)
# P_func.x.array[:] = P_eval.flatten()

# %%
# Residual form (weak form)
F = ufl.inner(ufl.grad(v), P) * dx

# %%
# Nonlinear problem and Newton solver
problem = NonlinearProblem(F, u, bcs_u)
solver = NewtonSolver(domain.comm, problem)
solver.atol = 1e-8
solver.rtol = 1e-8
solver.convergence_criterion = "incremental"

# %%
with XDMFFile(domain.comm, "tensile2d_u.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)

# %%
# Apply a tensile load by incrementally increasing traction on the right edge
n_steps = 100
max_traction = 1.0
u.name = "displacement"

for step in range(1, n_steps + 1):
    u_D_top.value = step * max_traction / n_steps
    num_its, converged = solver.solve(u)
    assert converged, f"Newton solver did not converge at step {step}"
    u.x.scatter_forward()
    if domain.comm.rank == 0:
        print(f"Step {step}: Traction {u_D_top.value:.2f}, Newton its: {num_its}")
    # Save solution for each step
    with XDMFFile(domain.comm, "tensile2d_u.xdmf", "a") as xdmf:
        xdmf.write_function(u, step)

# %%
# # Post-processing: compute von Mises stress
# s = P * F.T / J  # Cauchy stress
# s_dev = s - 1.0 / 2 * ufl.tr(s) * ufl.Identity(d)
# von_Mises = ufl.sqrt(3.0 / 2 * ufl.inner(s_dev, s_dev))
# V_von_mises = fem.functionspace(domain, ("DG", 0))
# stress_expr = fem.Expression(von_Mises, V_von_mises.element.interpolation_points)
# stresses = fem.Function(V_von_mises)
# stresses.interpolate(stress_expr)
# with XDMFFile(domain.comm, "tensile2d_vonmises.xdmf", "w") as xdmf:
#     xdmf.write_mesh(domain)
#     stresses.name = "von_mises"
#     xdmf.write_function(stresses)
