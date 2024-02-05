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
# # Plasticity of Mohr-Coulomb
#
# ## Problem formulation
# https://thelfer.github.io/tfel/web/MohrCoulomb.html
#
# ## Implementation
#
# ### Preamble

# %%
from dolfinx_external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    evaluate_operands,
    replace_external_operators,
)
from dolfinx import common, fem
import ufl
import basix
from utilities import build_cylinder_quarter, find_cell_by_point
from solvers import LinearProblem
import jax.numpy as jnp
from mpi4py import MPI
from petsc4py import PETSc

import matplotlib.pyplot as plt
import numba
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # replace by JAX_ENABLE_X64=True

# %% [markdown]
# ### Model parameters

# %%
R_i = 1  # [m]
R_e = 21  # [m]

E = 6778  # [MPa] Young modulus
nu = 0.25  # [-] Poisson ratio
lambda_ = E*nu/(1+nu)/(1-2*nu)
mu_ = E/2./(1+nu)
# sigma_u = 27.6 #[MPa]
P_i_value = 3.45  # [MPa]

c = 3.45  # [MPa] cohesion
phi = 30 * np.pi / 180  # [rads] friction angle
psi = 30 * np.pi / 180  # [rads] dilatancy angle
theta_T = 20 * np.pi / 180  # [rads] angle
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

TPV = np.finfo(PETSc.ScalarType).eps  # très petite value
SQRT2 = np.sqrt(2.)

zero_vec = jnp.array([TPV, TPV, TPV, TPV])

tr = np.array([1, 1, 1, 0])


# %% [markdown]
# ### JAX implementation
#
# #### Local implementation

# %%
@jax.jit
def J3(sigma_local):
    return sigma_local[2] * (sigma_local[0]*sigma_local[1] - sigma_local[3]*sigma_local[3]/2.)


@jax.jit
def sign(x):
    return jax.lax.cond(x < -TPV, lambda x: -1, lambda x: 1, x)


def coeff1(theta, angle):
    return jnp.cos(theta_T) - 1 / (jnp.sqrt(3) * jnp.sin(angle) * sign(theta) * jnp.sin(theta_T))


def coeff2(theta, angle):
    return sign(theta) * jnp.sin(theta_T) + 1 / (jnp.sqrt(3) * jnp.sin(angle) * jnp.cos(theta_T))


coeff3 = 18 * jnp.cos(3*theta_T)*jnp.cos(3*theta_T)*jnp.cos(3*theta_T)


def C(theta, angle):
    return (- jnp.cos(3*theta_T) * coeff1(theta, angle) - 3 * sign(theta) * jnp.sin(3*theta_T) * coeff2(theta, angle)) / coeff3


def B(theta, angle):
    return (sign(theta) * jnp.sin(6*theta_T) * coeff1(theta, angle) - 6 * jnp.cos(6*theta_T) * coeff2(theta, angle)) / coeff3


def A(theta, angle):
    return - 1/jnp.sqrt(3) * jnp.sin(angle) * sign(theta) * jnp.sin(theta_T) - B(theta, angle) * sign(theta) * jnp.sin(theta_T) - C(theta, angle) * jnp.sin(3 * theta_T)*jnp.sin(3 * theta_T) + jnp.cos(theta_T)


@jax.jit
def K(theta, angle):
    def K_true(theta, angle): return jnp.cos(theta) - 1 / \
        jnp.sqrt(3) * jnp.sin(angle) * jnp.sin(theta)
    def K_false(theta, angle): return A(theta, angle) + B(theta, angle) * \
        jnp.sin(3*theta) + C(theta, angle) * jnp.sin(3*theta)*jnp.sin(3*theta)
    return K_true(theta, angle)
    return jax.lax.cond(jnp.abs(theta) < theta_T, K_true, K_false, theta, angle)


@jax.jit
def a_G(angle):
    return a * jnp.tan(phi) / jnp.tan(angle)


# %%
@jax.jit
def surface(sigma_local, angle):
    s = dev @ sigma_local
    I1 = tr @ sigma_local
    J2 = 0.5 * jnp.vdot(s, s)
    arg = -3*jnp.sqrt(3) * J3(s) / (2 * jnp.sqrt(J2*J2*J2))
    arg = jnp.clip(arg, -1, 1)
    # arcsin returns nan if its argument is equal to -1 + smth around 1e-16!!!
    theta = 1/3. * jnp.arcsin(arg)
    return I1/3 * jnp.sin(angle) + jnp.sqrt(J2 * K(theta, angle)*K(theta, angle) + a_G(angle)*a_G(angle) * jnp.sin(angle)*jnp.sin(angle)) - c * jnp.cos(angle)


# %%
f_MC = jax.jit(lambda sigma_local: surface(sigma_local, phi))
g_MC = jax.jit(lambda sigma_local: surface(sigma_local, psi))


# %%
@jax.jit
def theta(sigma_local):
    s = dev @ sigma_local
    J2 = 0.5 * jnp.vdot(s, s)
    arg = -3*jnp.sqrt(3) * J3(s) / (2 * jnp.sqrt(J2*J2*J2))
    arg = jnp.clip(arg, -1, 1)
    return 1/3. * jnp.arcsin(arg)


# %%
dthetadsigma = jax.jit(jax.jacfwd(theta, argnums=(0)))
dgdsigma = jax.jit(jax.jacfwd(g_MC, argnums=(0)))

# dthetadJ2 = jax.jit(jax.jacfwd(theta, argnums=(1)))

# %%
deps_local = jnp.array([TPV, 0.0, 0.0, 0.0])
sigma_local = jnp.array([1, 1, 1.0, 1.0])
print(f"f_MC = {f_MC(sigma_local)}, dfdsigma = {sigma_local},\ntheta = {theta(sigma_local)}, dtheta = {dthetadsigma(sigma_local)}")

# %%
deps_local = jnp.array([0.0006, 0.0003, 0.0, 0.0])
sigma_local = C_elas @ deps_local
print(f"f_MC = {f_MC(sigma_local)}, dfdsigma = {sigma_local},\ntheta = {theta(sigma_local)}, dtheta = {dthetadsigma(sigma_local)}")


# %% [markdown]
# #### Vectorization

# %%
@jax.jit
def deps_p(sigma_local, dlambda, deps_local, sigma_n_local):
    sigma_elas_local = sigma_n_local + C_elas @ deps_local
    yielding = f_MC(sigma_elas_local)
    def deps_p_elastic(sigma_local, dlambda): return jnp.zeros(4)
    def deps_p_plastic(
        sigma_local, dlambda): return dlambda * dgdsigma(sigma_local)
    return jax.lax.cond(yielding <= TPV, deps_p_elastic, deps_p_plastic, sigma_local, dlambda)


@jax.jit
def r_sigma(sigma_local, dlambda, deps_local, sigma_n_local):
    deps_p_local = deps_p(sigma_local, dlambda, deps_local, sigma_n_local)
    return sigma_local - sigma_n_local - C_elas @ (deps_local - deps_p_local)


@jax.jit
def r_f(sigma_local, dlambda, deps_local, sigma_n_local):
    sigma_elas_local = sigma_n_local + C_elas @ deps_local
    yielding = f_MC(sigma_elas_local)

    def r_f_elastic(sigma_local, dlambda): return dlambda
    def r_f_plastic(sigma_local, dlambda): return f_MC(sigma_local)
    return jax.lax.cond(yielding <= TPV, r_f_elastic, r_f_plastic, sigma_local, dlambda)


dr_sigma = jax.jit(jax.jacfwd(r_sigma, argnums=(0, 1)))
dr_f = jax.jit(jax.jacfwd(r_f, argnums=(0, 1)))


@jax.jit
def j(sigma_local, dlambda, deps_local, sigma_n_local):
    dr_sigmadsigma, dr_sigmaddlambda = dr_sigma(
        sigma_local, dlambda, deps_local, sigma_n_local)
    # normally this creates two copies(?) but jit will "eat" them
    dr_sigmaddlambda_T = jnp.atleast_2d(dr_sigmaddlambda).T
    dr_fdsigma, dr_fddlambda = dr_f(
        sigma_local, dlambda, deps_local, sigma_n_local)
    return jnp.block([[dr_sigmadsigma, dr_sigmaddlambda_T],
                     [dr_fdsigma, dr_fddlambda]])


Nitermax, tol = 200, 1e-8


@jax.jit
def sigma_return_mapping(deps_local, sigma_n_local):
    niter = 0

    dlambda = jnp.array(0.)  # init guess
    sigma_local = sigma_n_local  # init guess

    res_sigma = r_sigma(sigma_local, dlambda, deps_local, sigma_n_local)
    res_f = r_f(sigma_local, dlambda, deps_local, sigma_n_local)
    res = jnp.c_['0,1,-1', res_sigma, res_f]

    norm_res0 = jnp.linalg.norm(res)
    sigma_elas_local = C_elas @ deps_local
    yielding = f_MC(sigma_n_local + sigma_elas_local)

    # jax.debug.print("norm {} {} {}", jnp.linalg.norm(res_sigma), jnp.linalg.norm(res_f), yielding)
    def cond_fun(state):
        norm_res, niter, _ = state
        return (norm_res/norm_res0 > tol) & (niter < Nitermax)

    def body_fun(state):
        norm_res, niter, history = state
        sigma_local, dlambda, deps_local, sigma_n_local, res = history
        J = j(sigma_local, dlambda, deps_local, sigma_n_local)
        j_inv_vp = jnp.linalg.solve(J, -res)
        sigma_local = sigma_local + j_inv_vp[:4]
        dlambda = dlambda + j_inv_vp[-1]

        res_sigma = r_sigma(sigma_local, dlambda, deps_local, sigma_n_local)
        res_f = r_f(sigma_local, dlambda, deps_local, sigma_n_local)
        res = jnp.c_['0,1,-1', res_sigma, res_f]
        norm_res = jnp.linalg.norm(res)
        niter += 1
        history = sigma_local, dlambda, deps_local, sigma_n_local, res
        return (norm_res, niter, history)

    history = (sigma_local, dlambda, deps_local, sigma_n_local, res)

    output = jax.lax.while_loop(
        cond_fun, body_fun, (norm_res0, niter, history))
    niter_total = output[1]
    sigma_local = output[2][0]
    # norm_res = output[0]

    return sigma_local, (sigma_local, niter_total, yielding, res)


dsigma_ddeps = jax.jacfwd(sigma_return_mapping, argnums=(0,), has_aux=True)
dsigma_ddeps_vec = jax.jit(jax.vmap(dsigma_ddeps, in_axes=(0, 0)))

# %%
mesh, facet_tags, facet_tags_labels = build_cylinder_quarter(R_e=R_e, R_i=R_i)

# %%
k_u = 2
V = fem.functionspace(mesh, ("Lagrange", k_u, (2,)))

# %%
k_stress = 2 * (k_u - 1)
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

S_element = basix.ufl.quadrature_element(
    mesh.topology.cell_name(), degree=k_stress, value_shape=(4,))
S = fem.functionspace(mesh, S_element)

P_element = basix.ufl.quadrature_element(
    mesh.topology.cell_name(), degree=k_stress, value_shape=())
P = fem.functionspace(mesh, P_element)

def epsilon(v):
    grad_v = ufl.grad(v)
    return ufl.as_vector([grad_v[0, 0], grad_v[1, 1], 0, np.sqrt(2.0) * 0.5 * (grad_v[0, 1] + grad_v[1, 0])])


# %%
Du = fem.Function(V, name="displacement_increment")
u = fem.Function(V, name="Total_displacement")
du = fem.Function(V, name="du")
v = ufl.TrialFunction(V)
u_ = ufl.TestFunction(V)

sigma = FEMExternalOperator(epsilon(Du), function_space=S)
sig_old = fem.Function(S, name="sig_old")
sig = fem.Function(S, name="sig")
p = fem.Function(P, name="p")
dp = fem.Function(P, name="dp")

# %%
bottom_facets = facet_tags.find(facet_tags_labels["Lx"])
left_facets = facet_tags.find(facet_tags_labels["Ly"])

bottom_dofs_y = fem.locate_dofs_topological(V.sub(1), mesh.topology.dim-1, bottom_facets)
left_dofs_x = fem.locate_dofs_topological(V.sub(0), mesh.topology.dim-1, left_facets)

sym_bottom = fem.dirichletbc(np.array(0.,dtype=PETSc.ScalarType), bottom_dofs_y, V.sub(1))
sym_left = fem.dirichletbc(np.array(0.,dtype=PETSc.ScalarType), left_dofs_x, V.sub(0))

bcs = [sym_bottom, sym_left]

# %%
n = ufl.FacetNormal(mesh)
P_o = fem.Constant(mesh, PETSc.ScalarType(0.0))
P_i = fem.Constant(mesh, PETSc.ScalarType(0.0))

def F_ext(v):
    return -P_o * ufl.inner(n, v)*ds(facet_tags_labels["inner"]) + P_o * ufl.inner(n, v)*ds(facet_tags_labels["outer"])


# %%
num_quadrature_points = P_element.dim

def C_tang_impl(deps, sigma, sigma_n, dp):
    num_cells = deps.shape[0]

    deps_global = deps.reshape((num_cells*num_quadrature_points, 4))
    sigma_n_global = sigma_n.reshape((num_cells*num_quadrature_points, 4))

    output = dsigma_ddeps_vec(deps_global, sigma_n_global)

    C_tang_global = output[0][0]
    sigma_global = output[1][0]
    # dp_global = output[1][1]
    # print(np.linalg.norm(dp_global))
    # np.copyto(dp, dp_global)
    dp[:] = 0.

    np.copyto(sigma, sigma_global.reshape(-1))

    niter = output[1][1]
    yielding = output[1][2]
    # res = output[1][3].reshape((27144, -1, 5))
    # norm_res = jnp.linalg.norm(res, axis=0)
    print("\tSubNewton:")
    print(f"\t  unique counts niter-s = {jnp.unique(niter, return_counts=True)}")
    print(f"\t  sigma = {np.linalg.norm(sigma)}")
    print(f"\t  max yielding = {jnp.max(yielding)}")
    # print(f"\t  norm_res = {jnp.min(norm_res), jnp.max(norm_res), jnp.mean(norm_res)}")
    # print(f"\t  res = {jnp.min(res), jnp.max(res), jnp.mean(res)}")
    # print(f"\t  nans = {jnp.argwhere(jnp.isnan(res))}")
    # print(f"\t  deps_global = {deps_global[0]}")

    return C_tang_global.reshape(-1)


# %%
def sigma_external(derivatives):
    if derivatives == (0,):
        return NotImplementedError
    elif derivatives == (1,):
        return C_tang_impl
    else:
        return NotImplementedError

sigma.external_function = sigma_external

# %%
u_hat = ufl.TrialFunction(V)
F = ufl.inner(epsilon(u_), sigma)*dx - F_ext(u_)
J = ufl.derivative(F, Du, u_hat)
J_expanded = ufl.algorithms.expand_derivatives(J)

F_replaced, F_external_operators = replace_external_operators(F)
J_replaced, J_external_operators = replace_external_operators(J_expanded)

F_form = fem.form(F_replaced)
J_form = fem.form(J_replaced)

# %%
evaluated_operands = evaluate_operands(F_external_operators)


# %%
# ((_, sigma_new, dp_new),) = 
evaluate_external_operators(J_external_operators, evaluated_operands)
