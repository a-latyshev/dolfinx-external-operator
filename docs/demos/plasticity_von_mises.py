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

from mpi4py import MPI
from petsc4py import PETSc

import gmsh
import numpy as np

import basix
import ufl
from dolfinx import fem
from dolfinx.io import gmshio
from dolfinx_external_operator import (
    FEMExternalOperator,
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

# NOTE: Is this really necessary for initialising the Newton solver?
TPV = np.finfo(PETSc.ScalarType).eps  # très petite value

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

mesh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, mesh_comm, 0.0, gdim=2)

mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
mesh.name = "quarter_cylinder"
cell_tags.name = f"{mesh.name}_cells"
facet_tags.name = f"{mesh.name}_facets"

k_u = 2
V = fem.functionspace(mesh, ("CG", k_u, (2,)))

k_stress = 2 * (k_u - 1)
P_element = basix.ufl.quadrature_element(mesh.topology.cell_name(), degree=k_stress, value_shape=())
S_element = basix.ufl.quadrature_element(mesh.topology.cell_name(), degree=k_stress, value_shape=(4,))

P = fem.functionspace(mesh, P_element)
S = fem.functionspace(mesh, S_element)

Du = fem.Function(V, name="displacement_increment")


def epsilon(v):
    grad_v = ufl.grad(v)
    return ufl.as_vector([grad_v[0, 0], grad_v[1, 1], 0, np.sqrt(2.0) * 0.5 * (grad_v[0, 1] + grad_v[1, 0])])


sigma_operator = FEMExternalOperator(epsilon(Du), function_space=S)

n = ufl.FacetNormal(mesh)
loading = fem.Constant(mesh, PETSc.ScalarType(0.0))

ds = ufl.Measure(
    "ds",
    domain=mesh,
    subdomain_data=facet_tags,
    metadata={"quadrature_degree": k_stress, "quadrature_scheme": "default"},
)

dx = ufl.Measure(
    "dx",
    domain=mesh,
    metadata={"quadrature_degree": k_stress, "quadrature_scheme": "default"},
)

v = ufl.TestFunction(V)
F = ufl.inner(sigma_operator, epsilon(v)) * dx - loading * ufl.inner(v, n) * ds(facet_tags_labels["inner"])

u_hat = ufl.TrialFunction(V)
J = ufl.derivative(F, Du, u_hat)

l, m = lambda_, mu_  # noqa: E741
C_elas = np.array(
    [[l + 2 * m, l, l, 0], [l, l + 2 * m, l, 0], [l, l, l + 2 * m, 0], [0, 0, 0, 2 * m]], dtype=PETSc.ScalarType
)

deviatoric = np.eye(4, dtype=PETSc.ScalarType)
deviatoric[:3, :3] -= np.full((3, 3), 1.0 / 3.0, dtype=PETSc.ScalarType)

# TODO: Can be numba or jax jitted
def return_mapping(deps, sigma, p, dp):
    """Performs the return-mapping procedure."""
    # TODO: Add loop over points.
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

# Internal state
p = fem.Function(P, name="cumulative_plastic_strain")
dp = fem.Function(P, name="incremental_plastic_strain")
sigma = fem.Function(S, name="stress")

def C_tang_impl(deps):
    # NOTE: Why these fixed shapes? Don't we have e.g. deps_ at more than one quadrature point?
    deps_ = deps.reshape((-1, 4))
    sigma_ = sigma.x.array.reshape((-1, 4))
    p_ = p.x.array.reshape((-1, 1))
    dp_ = dp.x.array.reshape((-1, 1))

    C_tang_, sigma_new, dp_new = return_mapping(
        deps_,
        sigma_,
        p_,
        dp_,
    )

    return C_tang_.reshape(-1), sigma_new, dp_new


def sigma_impl(deps):
    return sigma


def sigma_external(derivatives):
    if derivatives == (0,):
        return sigma_impl
    elif derivatives == (1,):
        return C_tang_impl
    else:
        return NotImplementedError


sigma.external_function = sigma_external

# Boundary conditions
bottom_facets = facet_tags.find(facet_tags_labels["Lx"])
left_facets = facet_tags.find(facet_tags_labels["Ly"])

bottom_dofs_y = fem.locate_dofs_topological(V.sub(1), mesh.topology.dim - 1, bottom_facets)
left_dofs_x = fem.locate_dofs_topological(V.sub(0), mesh.topology.dim - 1, left_facets)

sym_bottom = fem.dirichletbc(np.array(0.0, dtype=PETSc.ScalarType), bottom_dofs_y, V.sub(1))
sym_left = fem.dirichletbc(np.array(0.0, dtype=PETSc.ScalarType), left_dofs_x, V.sub(0))

bcs = [sym_bottom, sym_left]

# Form manipulations
J_expanded = ufl.algorithms.expand_derivatives(J)
F_replaced, F_external_operators = replace_external_operators(F)
J_replaced, J_external_operators = replace_external_operators(J_expanded)

F_replaced = fem.form(F_replaced)
J_form = fem.form(J_replaced)

num_increments = 20
load_steps = np.linspace(0, 1.1, num_increments + 1)[1:] ** 0.5
results = np.zeros((num_increments + 1, 2))
q_lim = 2.0 / np.sqrt(3.0) * np.log(R_e / R_i) * sigma_0

u = fem.Function(V, name="displacement")

# Continuation
for i, load_step in enumerate(load_steps):
    loading.value = -load_step * q_lim


