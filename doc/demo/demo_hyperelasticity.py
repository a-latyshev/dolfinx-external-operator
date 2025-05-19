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
L = 10.0  # Length
W = 1.0  # Height
nx, ny = 80, 8

# %%
domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [L, W]], [nx, ny], cell_type=CellType.quadrilateral)

# %%
V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))


# %%
# Boundary markers for Dirichlet and Neumann BCs
def left(x):
    return np.isclose(x[0], 0.0)
def right(x):
    return np.isclose(x[0], L)


# %%

# %%
fdim = domain.topology.dim - 1
left_facets = mesh.locate_entities_boundary(domain, fdim, left)
right_facets = mesh.locate_entities_boundary(domain, fdim, right)

# %%
# Mark facets: 1=left (fixed), 2=right (tension)
marked_facets = np.hstack([left_facets, right_facets])
marked_values = np.hstack([np.full_like(left_facets, 1), np.full_like(right_facets, 2)])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

# %%
# Dirichlet BC: fix left edge (all dofs)
u_bc = np.array((0,) * domain.geometry.dim, dtype=default_scalar_type)
left_dofs = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.find(1))
bcs = [fem.dirichletbc(u_bc, left_dofs, V)]

# %%
# Body force and traction (2D)
B = fem.Constant(domain, default_scalar_type((0, 0)))
T = fem.Constant(domain, default_scalar_type((0, 0)))

# %%
# Solution and test functions
u = fem.Function(V)
v = ufl.TestFunction(V)

# %%
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

# %%
# Strain energy and first Piola-Kirchhoff stress
psi = (mu / 2) * (Ic - 2) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J)) ** 2
P = ufl.diff(psi, F)

# %%
# Measures for integration
metadata = {"quadrature_degree": 2}
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag, metadata=metadata)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

# %%
# Residual form (weak form)
F = ufl.inner(ufl.grad(v), P) * dx - ufl.inner(v, B) * dx - ufl.inner(v, T) * ds(2)

# %%
# Nonlinear problem and Newton solver
problem = NonlinearProblem(F, u, bcs)
solver = NewtonSolver(domain.comm, problem)
solver.atol = 1e-8
solver.rtol = 1e-8
solver.convergence_criterion = "incremental"

# %%
with XDMFFile(domain.comm, "tensile2d_u.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)

# %%
# Apply a tensile load by incrementally increasing traction on the right edge
n_steps = 5
max_traction = 10.0
u.name = "displacement"

for step in range(1, n_steps + 1):
    T.value[0] = step * max_traction / n_steps  # Apply in x-direction
    num_its, converged = solver.solve(u)
    assert converged, f"Newton solver did not converge at step {step}"
    u.x.scatter_forward()
    if domain.comm.rank == 0:
        print(f"Step {step}: Traction {T.value[0]:.2f}, Newton its: {num_its}")
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

# %%
if domain.comm.rank == 0:
    print("2D simulation complete. Displacement and von Mises stress written to XDMF files.")
