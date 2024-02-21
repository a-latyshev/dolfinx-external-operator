# -*- coding: utf-8 -*-
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
# # Limit analysis
# Based on https://fenics-optim.readthedocs.io/en/latest/demos/limit_analysis_3D_SDP.html 

# %%
from mpi4py import MPI
from petsc4py import PETSc

import jax
jax.config.update("jax_enable_x64", True) # replace by JAX_ENABLE_X64=True
import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np
from solvers import LinearProblem
from utilities import build_cylinder_quarter, find_cell_by_point

import basix
import ufl
from dolfinx import common, fem, mesh, default_scalar_type
from dolfinx_external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    evaluate_operands,
    replace_external_operators,
)

# %%
E = 70e3
nu = 0.3
lambda_ = E*nu/(1+nu)/(1-2*nu)
mu_ = E/2./(1+nu)
sig0 = 250.  # yield strength
Et = E/100.  # tangent modulus
H = E*Et/(E-Et)  # hardening modulus
TPV = np.finfo(PETSc.ScalarType).eps # très petite value 
SQRT2 = np.sqrt(2.)

c = 3.45 #[MPa]
phi = 30 * np.pi / 180
psi = 30 * np.pi / 180
theta_T = 20 * np.pi / 180
a = 0.5 * c / np.tan(phi)

l, m = lambda_, mu_
C_elas = np.array([[l+2*m, l, l, 0],
                    [l, l+2*m, l, 0],
                    [l, l, l+2*m, 0],
                    [0, 0, 0, 2*m]], dtype=PETSc.ScalarType)

dev = np.array([[2/3., -1/3., -1/3., 0],
                 [-1/3., 2/3., -1/3., 0],
                 [-1/3., -1/3., 2/3., 0],
                 [0, 0, 0, 1.]], dtype=PETSc.ScalarType)

tr = np.array([1, 1, 1, 0])

zero_vec = jnp.array([TPV, TPV, TPV, TPV])

Nitermax, tol = 200, 1e-8

# %%
L = W = H = 1.
gamma = 1.
N = 10
domain = mesh.create_box(MPI.COMM_WORLD, [np.array([0,0,0]), np.array([L, W, H])], [N, N, N])

# %%
k_u = 2
# V = fem.functionspace(domain, ("Lagrange", k_u, (3,)))
V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
Du = fem.Function(V)

def on_right(x):
    return np.isclose(x[0], L)

def on_bottom(x):
    return np.isclose(x[2], 0.)

bottom_dofs = fem.locate_dofs_geometrical(V, on_bottom)
right_dofs = fem.locate_dofs_geometrical(V, on_right)
# bcs = [fem.dirichletbc(0.0, bottom_dofs, V), fem.dirichletbc(np.array(0.0, dtype=PETSc.ScalarType), right_dofs, V)] # bug???
bcs = [
    fem.dirichletbc(np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType), bottom_dofs, V),
    fem.dirichletbc(np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType), right_dofs, V)]


# %%
def epsilon(u):
    return ufl.sym(ufl.grad(u))

k_stress = 2 * (k_u - 1)
S_element = basix.ufl.quadrature_element(domain.topology.cell_name(), degree=k_stress, value_shape=())
S = fem.functionspace(domain, S_element)
dx = ufl.Measure(
    "dx",
    domain=domain,
    metadata={"quadrature_degree": k_stress, "quadrature_scheme": "default"},
)


pi = FEMExternalOperator(epsilon(Du), function_space=S)
f = fem.Constant(domain, default_scalar_type((0, 0, -gamma)))
F = pi * dx + ufl.dot(f, u) * ufl.dx


# %% [markdown]
# ### Constitutive model

# %%
@jax.jit
def det(Mandel_vec):
    """Returns the derminante of a tensor written through Mandel notation."""
    return Mandel_vec[2] * (Mandel_vec[0]*Mandel_vec[1] - Mandel_vec[3]*Mandel_vec[3]/2.)

@jax.jit
def eigensum(Mandel_vec):
    """Returns the sum of eigenvalues modules of a tensor written through Mandel notation."""
    lambda_I = jnp.abs(Mandel_vec[2])
    A = tr @ Mandel_vec
    B = jnp.sqrt(A**2 - 4*det(Mandel_vec))
    lambda_II = jnp.abs(0.5*(A + B))
    lambda_III = jnp.abs(0.5*(A - B))
    return lambda_I + lambda_II + lambda_III

# %%
@jax.jit
def pi(deps_local):
    condition = c/np.tan(phi) * tr @ deps_local - np.sin(phi) * eigensum(deps_local)
    # == tol and tr @ deps_local > np.sin(phi) * eigensum(deps_local)
    pi_positive = lambda eps_local: c * tr @ eps_local / np.tan(phi)
    pi_negative = lambda eps_local: jnp.inf
    return jax.lax.cond(condition >= TPV, pi_positive, pi_negative, deps_local)

dpiddeps = jax.jit(jax.jacfwd(pi, argnums=(0)))

# %%
deps_local = jnp.array([1., 0., 0., 0.])
pi(deps_local)

# %%
deps_local = jnp.array([0.001, 30., 20., 0.])
pi(deps_local)

# %%
dpiddeps(deps_local)

# %%




