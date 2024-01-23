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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Plasticity of von Mises
#
# ## Problem formulation
#
# ## Implementation
#
# ### Preamble

# %%
import sys

from mpi4py import MPI
from petsc4py import PETSc

import gmsh
import matplotlib.pyplot as plt
import numba
import numpy as np
import solvers

import basix
import ufl
from dolfinx import fem
from dolfinx_external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    evaluate_operands,
    replace_external_operators,
)

# %%
R_e, R_i = 1.3, 1.0  # external/internal radius

# elastic parameters
E = 70e3
nu = 0.3
lambda_ = E * nu / (1 + nu) / (1 - 2 * nu)
mu_ = E / 2.0 / (1 + nu)

sigma_0 = 250.0  # yield strength
E_tangent = E / 100.0  # tangent modulus
H = E * E_tangent / (E - E_tangent)  # hardening modulus

# NOTE: Is this really necessary for intialising the Newton solver?
TPV = np.finfo(PETSc.ScalarType).eps  # très petite value

q_lim = float(2 / np.sqrt(3) * np.log(R_e / R_i) * sig0)

SQRT2 = np.sqrt(2.0)

# %% [markdown]
# ### Building the mesh

# %%
# Source: https://newfrac.github.io/fenicsx-fracture/notebooks/plasticity/plasticity.html
# mesh parameters
gdim = 2
lc = 0.3
verbosity = 0

# TODO: Put this in another file? Can execute with cell magic?
mesh_comm = MPI.COMM_WORLD
model_rank = 0
gmsh.initialize()

facet_tags_labels = {"Lx": 1, "Ly": 2, "inner": 3, "outer": 4}

cell_tags_map = {"all": 20}
if mesh_comm.rank == model_rank:
    model = gmsh.model()
    model.add("Quart_cylinder")
    model.setCurrent("Quart_cylinder")
    # Create the points
    pix = model.occ.addPoint(R_i, 0.0, 0, lc)
    pex = model.occ.addPoint(R_e, 0, 0, lc)
    piy = model.occ.addPoint(0.0, R_i, 0, lc)
    pey = model.occ.addPoint(0.0, R_e, 0, lc)
    center = model.occ.addPoint(0.0, 0.0, 0, lc)
    # Create the lines
    lx = model.occ.addLine(pix, pex, tag=facet_tags_labels["Lx"])
    lout = model.occ.addCircleArc(pex, center, pey, tag=facet_tags_labels["outer"])
    ly = model.occ.addLine(pey, piy, tag=facet_tags_labels["Ly"])
    lin = model.occ.addCircleArc(piy, center, pix, tag=facet_tags_labels["inner"])
    # Create the surface
    cloop1 = model.occ.addCurveLoop([lx, lout, ly, lin])
    surface_1 = model.occ.addPlaneSurface([cloop1], tag=cell_tags_map["all"])
    model.occ.synchronize()
    # Assign mesh and facet tags
    surface_entities = [entity[1] for entity in model.getEntities(2)]
    model.addPhysicalGroup(2, surface_entities, tag=cell_tags_map["all"])
    model.setPhysicalName(2, 2, "Quart_cylinder surface")
    for key, value in facet_tags_labels.items():
        # 1 : it is the dimension of the object (here a curve)
        model.addPhysicalGroup(1, [value], tag=value)
        model.setPhysicalName(1, value, key)
    # Finalize mesh
    model.occ.synchronize()
    gmsh.option.setNumber("General.Verbosity", verbosity)
    model.mesh.generate(gdim)

# import the mesh in fenicsx with gmshio
# TODO: After calling this line there is [WARNING] yaksa: 2 leaked handle pool objects
mesh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, mesh_comm, 0.0, gdim=2)

mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
mesh.name = "Quart_cylinder"
cell_tags.name = f"{mesh.name}_cells"
facet_tags.name = f"{mesh.name}_facets"

# %%
deg_u = 2
deg_stress = 2

W0e = basix.ufl.quadrature_element(mesh.topology.cell_name(), degree=deg_stress, value_shape=())
We = basix.ufl.quadrature_element(mesh.topology.cell_name(), degree=deg_stress, value_shape=(4,))

W0 = fem.functionspace(mesh, W0e)
W = fem.functionspace(mesh, We)
V = fem.functionspace(mesh, ("CG", deg_u, (2,)))

ds = ufl.Measure(
    "ds",
    domain=mesh,
    subdomain_data=facet_tags,
    metadata={"quadrature_degree": deg_stress, "quadrature_scheme": "default"},
)

dx = ufl.Measure(
    "dx",
    domain=mesh,
    metadata={"quadrature_degree": deg_stress, "quadrature_scheme": "default"},
)

# %%
p = fem.Function(W0, name="cumulative_plastic_strain")
u = fem.Function(V, name="total_displacement")
du = fem.Function(V, name="newton_iteration_correction")
Du = fem.Function(V, name="current_increment")

u_hat = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# %%
bottom_facets = facet_tags.find(facet_tags_labels["Lx"])
left_facets = facet_tags.find(facet_tags_labels["Ly"])

bottom_dofs_y = fem.locate_dofs_topological(V.sub(1), mesh.topology.dim - 1, bottom_facets)
left_dofs_x = fem.locate_dofs_topological(V.sub(0), mesh.topology.dim - 1, left_facets)

sym_bottom = fem.dirichletbc(np.array(0.0, dtype=PETSc.ScalarType), bottom_dofs_y, V.sub(1))
sym_left = fem.dirichletbc(np.array(0.0, dtype=PETSc.ScalarType), left_dofs_x, V.sub(0))

bcs = [sym_bottom, sym_left]

# %%
n = ufl.FacetNormal(mesh)
loading = fem.Constant(mesh, PETSc.ScalarType(0.0))


def F_ext(v):
    return -loading * ufl.inner(v, n) * ds(facet_tags_labels["inner"])


# %%
def epsilon(v):
    grad_v = ufl.grad(v)
    return ufl.as_vector([grad_v[0, 0], grad_v[1, 1], 0, SQRT2 * 0.5 * (grad_v[0, 1] + grad_v[1, 0])])


sigma = FEMExternalOperator(epsilon(u), function_space=W)

F = ufl.inner(sigma, epsilon(v)) * dx - F_ext(v)
J = ufl.derivative(F, u, u_hat)

# %% [markdown]
# ### Implementing the external operator

# %%
l, m = lambda_, mu_  # noqa: E741
C_elas = np.array(
    [[l + 2 * m, l, l, 0], [l, l + 2 * m, l, 0], [l, l, l + 2 * m, 0], [0, 0, 0, 2 * m]], dtype=PETSc.ScalarType
)

deviatoric = np.eye(4, dtype=PETSc.ScalarType)
deviatoric[:3, :3] -= np.full((3, 3), 1.0 / 3.0, dtype=PETSc.ScalarType)


def return_mapping(deps, sigma, p, dp):
    """Performs the return-mapping procedure.
    """
    sigma_elastic = sigma + C_elas @ deps
    s = deviatoric @ sigma_elastic
    sigma_eq = np.sqrt(3.0 / 2.0 * np.dot(s, s))

    f_elastic = sigma_eq - sigma_0 - H * p
    f_elastic_plus = (f_elastic + np.sqrt(f_elastic**2)) / 2.0

    dp_new = f_elastic_plus / (3 * mu_ + H)

    n_elas = s / sigma_eq * f_elastic_plus / f_elastic
    beta = 3 * mu_ * dp / sigma_eq

    sigma_new = sigma_elastic - beta * s
    
    n_elas_matrix = np.outer(n_elas, n_elas)
    C_tang = C_elas - 3 * mu_ * (3 * mu_ / (3 * mu_ + H) - beta) * n_elas_matrix - 2 * mu_ * beta * deviatoric

    return C_tang, sigma_new, dp_new 


def C_tang_impl(deps):
    deps_ = deps.reshape((-1, 4))
    sigma_ = sigma.x.array.reshape((-1, 4))
    p_ = p.reshape.x.array((-1, 1))
    dp_ = dp.x.array.reshape((-1, 1))

    C_tang_, sigma_new, dp_new = return_mapping(
        deps_, sigma_, p_, dp_,
    )
   
    sigma_[:] = sigma_new
    dp_[:] = dp_new

    return C_tang_.reshape(-1)


def sigma_impl(deps, sigma, p, dp):
    return sigma


# %%
def sigma_external(derivatives):
    if derivatives == (0,):
        return sigma_impl
    elif derivatives == (1,):
        return C_tang_impl
    else:
        return NotImplementedError


# %%
sigma.external_function = sigma_external

# %%
Du.x.array[:] = TPV  # For initialization

# %%
J_expanded = ufl.algorithms.expand_derivatives(J)
F_replaced, F_external_operators = replace_external_operators(F)
J_replaced, J_external_operators = replace_external_operators(J_expanded)


operands_to_project, evaluated_operands = find_operands_and_allocate_memory(F_external_operators)
evaluate_operands_v2(operands_to_project, mesh)
# evaluated_operands = evaluate_operands(F_external_operators)

evaluate_external_operators(F_external_operators, evaluated_operands)
evaluate_external_operators(J_external_operators, evaluated_operands)

# %%
external_operator_problem = solvers.LinearProblem(J_replaced, -F_replaced, Du, bcs=bcs)


# %%
# Defining a cell containing (Ri, 0) point, where we calculate a value of u
def find_cell_by_point(mesh, point):
    cells = []
    points_on_proc = []
    tree = bb_tree(mesh, mesh.geometry.dim)
    cell_candidates = compute_collisions_points(tree, point)
    colliding_cells = compute_colliding_cells(mesh, cell_candidates, point)
    for i, point in enumerate(point):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    return cells, points_on_proc


# %% [markdown]
# ### Solving the problem


# %%
# Defining a cell containing (Ri, 0) point, where we calculate a value of u
# It is required to run this program via MPI in order to capture the process, to which this point is attached
x_point = np.array([[R_i, 0, 0]])
cells, points_on_proc = find_cell_by_point(mesh, x_point)

Nitermax, tol = 200, 1e-8  # parameters of the manual Newton method
Nincr = 20
load_steps = np.linspace(0, 1.1, Nincr + 1)[1:] ** 0.5
results = np.zeros((Nincr + 1, 2))

# timer3 = common.Timer("Solving the problem")
# start = MPI.Wtime()
# timer3.start()

for i, t in enumerate(load_steps):
    loading.value = t * q_lim
    external_operator_problem.assemble_vector()

    nRes0 = external_operator_problem.b.norm()
    nRes = nRes0
    Du.x.array[:] = 0

    if MPI.COMM_WORLD.rank == 0:
        print(f"\nnRes0 , {nRes0} \n Increment: {i+1!s}, load = {t * q_lim}")
    niter = 0

    while nRes / nRes0 > tol and niter < Nitermax:
        external_operator_problem.assemble_matrix()
        external_operator_problem.solve(du)

        Du.vector.axpy(1, du.vector)  # Du = Du + 1*du
        Du.x.scatter_forward()

        # Evaluation of new_eps(Du):
        # evaluated_operands = evaluate_operands(F_external_operators)
        evaluate_operands_v2(operands_to_project, mesh)
        # Return-mapping procedure and stress update:
        evaluate_external_operators(J_external_operators, evaluated_operands)
        external_operator_problem.assemble_vector()
        nRes = external_operator_problem.b.norm()

        if MPI.COMM_WORLD.rank == 0:
            print(f"    it# {niter} Residual: {nRes}")
        niter += 1
    u.vector.axpy(1, Du.vector)  # u = u + 1*Du
    u.x.scatter_forward()

    # p.vector.axpy(1, dp.vector)
    # p.x.scatter_forward()
    p.x.array[:] = p.x.array + dp
    np.copyto(sig_old, sigma.ref_coefficient.x.array)

    if len(points_on_proc) > 0:
        results[i + 1, :] = (u.eval(points_on_proc, cells)[0], t)

# end = MPI.Wtime()
# timer3.stop()

# total_time = end - start
# compilation_overhead = time1 - time2

# print(f'rank#{MPI.COMM_WORLD.rank}: Time = {total_time:.3f} (s)')
# print(f'rank#{MPI.COMM_WORLD.rank}: Compilation overhead: {compilation_overhead:.3f} s')

# %%
if len(points_on_proc) > 0:
    plt.plot(results[:, 0], results[:, 1], "-o", label="via ExternalOperator")
    plt.xlabel("Displacement of inner boundary")
    plt.ylabel(r"Applied pressure $q/q_{lim}$")
    plt.savefig(f"displacement_rank{MPI.COMM_WORLD.rank:d}.png")
    plt.legend()
    plt.show()
