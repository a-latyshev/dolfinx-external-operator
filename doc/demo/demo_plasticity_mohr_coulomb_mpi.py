# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
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

# %% [markdown]
# # Plasticity of Mohr-Coulomb with apex-smoothing
#
# This tutorial aims to demonstrate how modern automatic algorithmic
# differentiation (AD) techniques may be used to define a complex constitutive
# model demanding a lot of by-hand differentiation. In particular, we implement
# the non-associative plasticity model of Mohr-Coulomb with apex-smoothing applied
# to a slope stability problem for soil. We use the
# [JAX](https://jax.readthedocs.io/en/latest/) package to define constitutive
# relations including the differentiation of certain terms and
# `FEMExternalOperator` class to incorporate this model into a weak formulation
# within [UFL](https://github.com/fenics/ufl).
#
# The tutorial is based on the
# [limit analysis](https://fenics-optim.readthedocs.io/en/latest/demos/limit_analysis_3D_SDP.html)
# within semi-definite programming framework, where the plasticity model was
# replaced by the MFront/TFEL
# [implementation](https://thelfer.github.io/tfel/web/MohrCoulomb.html) of
# the Mohr-Coulomb elastoplastic model with apex smoothing.
#
#
# ## Problem formulation
#
# We solve a slope stability problem of a soil domain $\Omega$ represented by a
# rectangle $[0; L] \times [0; W]$ with homogeneous Dirichlet boundary conditions
# for the displacement field $\boldsymbol{u} = \boldsymbol{0}$ on the right side
# $x = L$ and the bottom one $z = 0$. The loading consists of a gravitational body
# force $\boldsymbol{q}=[0, -\gamma]^T$ with $\gamma$ being the soil self-weight.
# The solution of the problem is to find the collapse load $q_\text{lim}$, for
# which we know an analytical solution in the case of the standard Mohr-Coulomb
# model without smoothing under plane strain assumption for associative plastic law
# {cite}`chenLimitAnalysisSoil1990`. Here we follow the same Mandel-Voigt notation
# as in the [von Mises plasticity tutorial](demo_plasticity_von_mises.py).
#
# If $V$ is a functional space of admissible displacement fields, then we can
# write out a weak formulation of the problem:
#
# Find $\boldsymbol{u} \in V$ such that
#
# $$
#     F(\boldsymbol{u}; \boldsymbol{v}) = \int\limits_\Omega
#     \boldsymbol{\sigma}(\boldsymbol{u}) \cdot
#     \boldsymbol{\varepsilon}(\boldsymbol{v}) \, \mathrm{d}\boldsymbol{x} -
#     \int\limits_\Omega \boldsymbol{q} \cdot \boldsymbol{v} \, \mathrm{d}\boldsymbol{x} = \boldsymbol{0}, \quad
#     \forall \boldsymbol{v} \in V,
# $$
# where $\boldsymbol{\sigma}$ is an external operator representing the stress tensor.
#
# ```{note}
# Although the tutorial shows the implementation of the Mohr-Coulomb model, it
# is quite general to be adapted to a wide rage of plasticity models that may
# be defined through a yield surface and a plastic potential.
# ```
#
# ## Implementation
#
# ### Preamble

# %%
import os, sys 
sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
from mpi4py import MPI
from petsc4py import PETSc

import jax
import jax.lax
import jax.numpy as jnp
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpltools import annotation  # for slope markers
from solvers import LinearProblem, SNESProblem
from utilities import find_cell_by_point

import basix
import ufl
from dolfinx import default_scalar_type, fem, mesh, common
from dolfinx_external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    evaluate_operands,
    replace_external_operators,
)

jax.config.update("jax_enable_x64", True)

# %% [markdown]
# Here we define geometrical and material parameters of the problem as well as
# some useful constants.

# %%
E =  6778  # [MPa] Young modulus
nu = 0.25  # [-] Poisson ratio
c = 3.45 # [MPa] cohesion
phi = 30 * np.pi / 180  # [rad] friction angle
psi = 30 * np.pi / 180  # [rad] dilatancy angle
theta_T = 26 * np.pi / 180  # [rad] transition angle as defined by Abbo and Sloan
a = 0.26 * c / np.tan(phi)  # [MPa] tension cuff-off parameter
import argparse

parser = argparse.ArgumentParser(description="Demo Plasticity Mohr Coulomb MPI")
parser.add_argument("--N", type=int, default=50, help="Mesh size")
args = parser.parse_args()
N = args.N

# %%
L, H = (1.2, 1.0)
Nx, Ny = (N, N)
gamma = 1.0
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([L, H])], [Nx, Ny])

# %%
domain.topology.index_map(0).size_global

# %%
k_u = 2
gdim = domain.topology.dim
V = fem.functionspace(domain, ("Lagrange", k_u, (gdim,)))


# Boundary conditions
def on_right(x):
    return np.isclose(x[0], L)


def on_bottom(x):
    return np.isclose(x[1], 0.0)


bottom_dofs = fem.locate_dofs_geometrical(V, on_bottom)
right_dofs = fem.locate_dofs_geometrical(V, on_right)

bcs = [
    fem.dirichletbc(np.array([0.0, 0.0], dtype=PETSc.ScalarType), bottom_dofs, V),
    fem.dirichletbc(np.array([0.0, 0.0], dtype=PETSc.ScalarType), right_dofs, V),
]


def epsilon(v):
    grad_v = ufl.grad(v)
    return ufl.as_vector(
        [
            grad_v[0, 0],
            grad_v[1, 1],
            0.0,
            np.sqrt(2.0) * 0.5 * (grad_v[0, 1] + grad_v[1, 0]),
        ]
    )


k_stress = 2 * (k_u - 1)

dx = ufl.Measure(
    "dx",
    domain=domain,
    metadata={"quadrature_degree": k_stress, "quadrature_scheme": "default"},
)

stress_dim = 2 * gdim
S_element = basix.ufl.quadrature_element(domain.topology.cell_name(), degree=k_stress, value_shape=(stress_dim,))
S = fem.functionspace(domain, S_element)


Du = fem.Function(V, name="Du")
u = fem.Function(V, name="Total_displacement")
du = fem.Function(V, name="du")
v = ufl.TestFunction(V)

sigma = FEMExternalOperator(epsilon(Du), function_space=S)
sigma_n = fem.Function(S, name="sigma_n")

# %% [markdown]
# ### Defining plasticity model and external operator
#
# The constitutive model of the soil is described by a non-associative plasticity
# law without hardening that is defined by the Mohr-Coulomb yield surface $f$ and
# the plastic potential $g$. Both quantities may be expressed through the
# following function $h$
#
# \begin{align*}
#     & h(\boldsymbol{\sigma}, \alpha) =
#     \frac{I_1(\boldsymbol{\sigma})}{3}\sin\alpha +
#     \sqrt{J_2(\boldsymbol{\sigma}) K^2(\alpha) + a^2(\alpha)\sin^2\alpha} -
#     c\cos\alpha, \\
#     & f(\boldsymbol{\sigma}) = h(\boldsymbol{\sigma}, \phi), \\
#     & g(\boldsymbol{\sigma}) = h(\boldsymbol{\sigma}, \psi),
# \end{align*}
# where $\phi$ and $\psi$ are friction and dilatancy angles, $c$ is a cohesion,
# $I_1(\boldsymbol{\sigma}) = \mathrm{tr} \boldsymbol{\sigma}$ is the first
# invariant of the stress tensor and $J_2(\boldsymbol{\sigma}) =
# \frac{1}{2}\boldsymbol{s} \cdot \boldsymbol{s}$ is the second invariant of the
# deviatoric part of the stress tensor. The expression of the coefficient
# $K(\alpha)$ may be found in the MFront/TFEL
# [implementation](https://thelfer.github.io/tfel/web/MohrCoulomb.html) of this plastic model.
#
# During the plastic loading the stress-strain state of the solid must satisfy
# the following system of nonlinear equations
#
# $$
#
#     \begin{cases}
#         \boldsymbol{r}_{g}(\boldsymbol{\sigma}_{n+1}, \Delta\lambda) =
#         \boldsymbol{\sigma}_{n+1} - \boldsymbol{\sigma}_n -
#         \boldsymbol{C} \cdot (\Delta\boldsymbol{\varepsilon} - \Delta\lambda
#         \frac{\mathrm{d} g}{\mathrm{d}\boldsymbol{\sigma}}(\boldsymbol{\sigma_{n+1}})) =
#         \boldsymbol{0}, \\
#          r_f(\boldsymbol{\sigma}_{n+1}) = f(\boldsymbol{\sigma}_{n+1}) = 0,
#     \end{cases}
#
# $$ (eq_MC_1)
# where $\Delta$ is associated with increments of a quantity between the next
# loading step $n + 1$ and the current loading step $n$.
#
# By introducing the residual vector $\boldsymbol{r} = [\boldsymbol{r}_{g}^T,
# r_f]^T$ and its argument vector $\boldsymbol{y}_{n+1} =
# [\boldsymbol{\sigma}_{n+1}^T, \Delta\lambda]^T$, we obtain the following
# nonlinear constitutive equation:
#
# $$
#     \boldsymbol{r}(\boldsymbol{y}_{n+1}) = \boldsymbol{0}.
# $$
#
# To solve this equation we apply the Newton method and introduce the local
# Jacobian of the residual vector $\boldsymbol{j} := \frac{\mathrm{d}
# \boldsymbol{r}}{\mathrm{d} \boldsymbol{y}}$. Thus we solve the following linear
# system at each quadrature point for the plastic phase
#
# $$
#     \begin{cases}
#         \boldsymbol{j}(\boldsymbol{y}_{n})\boldsymbol{t} = -
#         \boldsymbol{r}(\boldsymbol{y}_{n}), \\
#         \boldsymbol{x}_{n+1} = \boldsymbol{x}_n + \boldsymbol{t}.
#     \end{cases}
# $$
#
# During the elastic loading, we consider a trivial system of equations
#
# $$
#     \begin{cases}
#         \boldsymbol{\sigma}_{n+1} = \boldsymbol{\sigma}_n +
#         \boldsymbol{C} \cdot \Delta\boldsymbol{\varepsilon}, \\ \Delta\lambda = 0.
#     \end{cases}
# $$ (eq_MC_2)
#
# The algorithm solving the systems {eq}`eq_MC_1`--{eq}`eq_MC_2` is called the
# *return-mapping procedure* and the solution defines the return-mapping
# correction of the stress tensor. By implementation of the external operator
# $\boldsymbol{\sigma}$ we mean the implementation of this *algorithmic*
# procedure.
#
# The automatic differentiation tools of the JAX library are applied to calculate
# the three distinct derivatives:
# 1. $\frac{\mathrm{d} g}{\mathrm{d}\boldsymbol{\sigma}}$ - derivative
#    of the plastic potential $g$,
# 2. $j = \frac{\mathrm{d} \boldsymbol{r}}{\mathrm{d} \boldsymbol{y}}$ -
#    derivative of the local residual $\boldsymbol{r}$,
# 3. $\boldsymbol{C}_\text{tang} =
#    \frac{\mathrm{d}\boldsymbol{\sigma}}{\mathrm{d}\boldsymbol{\varepsilon}}$ -
#    stress tensor
# derivative or consistent tangent moduli.
#
# #### Defining yield surface and plastic potential
#
# First of all, we define supplementary functions that help us to express the
# yield surface $f$ and the plastic potential $g$. In the following definitions,
# we use built-in functions of the JAX package, in particular, the conditional
# primitive `jax.lax.cond`. It is necessary for the correct work of the AD tool
# and just-in-time compilation. For more details, please, visit the JAX
# [documentation](https://jax.readthedocs.io/en/latest/).


# %%
def J3(s):
    return s[2] * (s[0] * s[1] - s[3] * s[3] / 2.0)


def J2(s):
    return 0.5 * jnp.vdot(s, s)


def theta(s):
    J2_ = J2(s)
    arg = -(3.0 * np.sqrt(3.0) * J3(s)) / (2.0 * jnp.sqrt(J2_ * J2_ * J2_))
    arg = jnp.clip(arg, -1.0, 1.0)
    theta = 1.0 / 3.0 * jnp.arcsin(arg)
    return theta


def sign(x):
    return jax.lax.cond(x < 0.0, lambda x: -1, lambda x: 1, x)


def coeff1(theta, angle):
    return np.cos(theta_T) - (1.0 / np.sqrt(3.0)) * np.sin(angle) * np.sin(theta_T)


def coeff2(theta, angle):
    return sign(theta) * np.sin(theta_T) + (1.0 / np.sqrt(3.0)) * np.sin(angle) * np.cos(theta_T)


coeff3 = 18.0 * np.cos(3.0 * theta_T) * np.cos(3.0 * theta_T) * np.cos(3.0 * theta_T)


def C(theta, angle):
    return (
        -np.cos(3.0 * theta_T) * coeff1(theta, angle) - 3.0 * sign(theta) * np.sin(3.0 * theta_T) * coeff2(theta, angle)
    ) / coeff3


def B(theta, angle):
    return (
        sign(theta) * np.sin(6.0 * theta_T) * coeff1(theta, angle) - 6.0 * np.cos(6.0 * theta_T) * coeff2(theta, angle)
    ) / coeff3


def A(theta, angle):
    return (
        -(1.0 / np.sqrt(3.0)) * np.sin(angle) * sign(theta) * np.sin(theta_T)
        - B(theta, angle) * sign(theta) * np.sin(3 * theta_T)
        - C(theta, angle) * np.sin(3.0 * theta_T) * np.sin(3.0 * theta_T)
        + np.cos(theta_T)
    )


def K(theta, angle):
    def K_false(theta):
        return jnp.cos(theta) - (1.0 / np.sqrt(3.0)) * np.sin(angle) * jnp.sin(theta)

    def K_true(theta):
        return (
            A(theta, angle)
            + B(theta, angle) * jnp.sin(3.0 * theta)
            + C(theta, angle) * jnp.sin(3.0 * theta) * jnp.sin(3.0 * theta)
        )

    return jax.lax.cond(jnp.abs(theta) > theta_T, K_true, K_false, theta)


def a_g(angle):
    return a * np.tan(phi) / np.tan(angle)


dev = np.array(
    [
        [2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 0.0],
        [-1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0, 0.0],
        [-1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=PETSc.ScalarType,
)
tr = np.array([1.0, 1.0, 1.0, 0.0], dtype=PETSc.ScalarType)


def surface(sigma_local, angle):
    s = dev @ sigma_local
    I1 = tr @ sigma_local
    theta_ = theta(s)
    return (
        (I1 / 3.0 * np.sin(angle))
        + jnp.sqrt(
            J2(s) * K(theta_, angle) * K(theta_, angle) + a_g(angle) * a_g(angle) * np.sin(angle) * np.sin(angle)
        )
        - c * np.cos(angle)
    )


# %% [markdown]
# By picking up an appropriate angle we define the yield surface $f$ and the
# plastic potential $g$.


# %%
def f(sigma_local):
    return surface(sigma_local, phi)


def g(sigma_local):
    return surface(sigma_local, psi)


dgdsigma = jax.jacfwd(g)

# %% [markdown]
# #### Solving constitutive equations
#
# In this section, we define the constitutive model by solving the systems
# {eq}`eq_MC_1`--{eq}`eq_MC_2`. They must be solved at each Gauss point, so we
# apply the Newton method, implement the whole algorithm locally and then
# vectorize the final result using `jax.vmap`.
#
# In the following cell, we define locally the residual $\boldsymbol{r}$ and
# its Jacobian `drdy`.

# %%
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))
C_elas = np.array(
    [
        [lmbda + 2 * mu, lmbda, lmbda, 0],
        [lmbda, lmbda + 2 * mu, lmbda, 0],
        [lmbda, lmbda, lmbda + 2 * mu, 0],
        [0, 0, 0, 2 * mu],
    ],
    dtype=PETSc.ScalarType,
)
S_elas = np.linalg.inv(C_elas)
ZERO_VECTOR = np.zeros(stress_dim, dtype=PETSc.ScalarType)


def deps_p(sigma_local, dlambda, deps_local, sigma_n_local):
    sigma_elas_local = sigma_n_local + C_elas @ deps_local
    yielding = f(sigma_elas_local)

    def deps_p_elastic(sigma_local, dlambda):
        return ZERO_VECTOR

    def deps_p_plastic(sigma_local, dlambda):
        return dlambda * dgdsigma(sigma_local)

    return jax.lax.cond(yielding <= 0.0, deps_p_elastic, deps_p_plastic, sigma_local, dlambda)


def r_g(sigma_local, dlambda, deps_local, sigma_n_local):
    deps_p_local = deps_p(sigma_local, dlambda, deps_local, sigma_n_local)
    return sigma_local - sigma_n_local - C_elas @ (deps_local - deps_p_local)


def r_f(sigma_local, dlambda, deps_local, sigma_n_local):
    sigma_elas_local = sigma_n_local + C_elas @ deps_local
    yielding = f(sigma_elas_local)

    def r_f_elastic(sigma_local, dlambda):
        return dlambda

    def r_f_plastic(sigma_local, dlambda):
        return f(sigma_local)

    return jax.lax.cond(yielding <= 0.0, r_f_elastic, r_f_plastic, sigma_local, dlambda)


def r(y_local, deps_local, sigma_n_local):
    sigma_local = y_local[:stress_dim]
    dlambda_local = y_local[-1]

    res_g = r_g(sigma_local, dlambda_local, deps_local, sigma_n_local)
    res_f = r_f(sigma_local, dlambda_local, deps_local, sigma_n_local)

    res = jnp.c_["0,1,-1", res_g, res_f]  # concatenates an array and a scalar
    return res


drdy = jax.jacfwd(r)

# %% [markdown]
# Then we define the function `return_mapping` that implements the
# return-mapping algorithm numerically via the Newton method.

# %%
Nitermax, tol = 200, 1e-7

ZERO_SCALAR = np.array([0.0])


def return_mapping(deps_local, sigma_n_local):
    """Performs the return-mapping procedure.

    It solves elastoplastic constitutive equations numerically by applying the
    Newton method in a single Gauss point. The Newton loop is implement via
    `jax.lax.while_loop`.

    The function returns `sigma_local` two times to reuse its values after
    differentiation, i.e. as once we apply
    `jax.jacfwd(return_mapping, has_aux=True)` the ouput function will
    have an output of
    `(C_tang_local, (sigma_local, niter_total, yielding, norm_res, dlambda))`.

    Returns:
        sigma_local: The stress at the current Gauss point.
        niter_total: The total number of iterations.
        yielding: The value of the yield function.
        norm_res: The norm of the residuals.
        dlambda: The value of the plastic multiplier.
    """
    niter = 0

    dlambda = ZERO_SCALAR
    sigma_local = sigma_n_local
    y_local = jnp.concatenate([sigma_local, dlambda])

    res = r(y_local, deps_local, sigma_n_local)
    norm_res0 = jnp.linalg.norm(res)

    def cond_fun(state):
        norm_res, niter, _ = state
        return jnp.logical_and(norm_res / norm_res0 > tol, niter < Nitermax)

    def body_fun(state):
        norm_res, niter, history = state

        y_local, deps_local, sigma_n_local, res = history

        j = drdy(y_local, deps_local, sigma_n_local)
        j_inv_vp = jnp.linalg.solve(j, -res)
        y_local = y_local + j_inv_vp

        res = r(y_local, deps_local, sigma_n_local)
        norm_res = jnp.linalg.norm(res)
        history = y_local, deps_local, sigma_n_local, res

        niter += 1

        return (norm_res, niter, history)

    history = (y_local, deps_local, sigma_n_local, res)

    norm_res, niter_total, y_local = jax.lax.while_loop(cond_fun, body_fun, (norm_res0, niter, history))

    sigma_local = y_local[0][:stress_dim]
    dlambda = y_local[0][-1]
    sigma_elas_local = C_elas @ deps_local
    yielding = f(sigma_n_local + sigma_elas_local)

    return sigma_local, (sigma_local, niter_total, yielding, norm_res, dlambda)


# %% [markdown]
# #### Consistent tangent stiffness matrix
#
# Not only is the automatic differentiation able to compute the derivative of a
# mathematical expression but also a numerical algorithm. For instance, AD can
# calculate the derivative of the function performing return-mapping with respect
# to its output, the stress tensor $\boldsymbol{\sigma}$. In the context of the
# consistent tangent moduli $\boldsymbol{C}_\text{tang}$, this feature becomes
# very useful, as there is no need to write an additional program computing the
# stress derivative.
#
# JAX's AD tool permits taking the derivative of the function `return_mapping`,
# which is factually the while loop. The derivative is taken with respect to the
# first output and the remaining outputs are used as auxiliary data. Thus, the
# derivative `dsigma_ddeps` returns both values of the consistent tangent moduli
# and the stress tensor, so there is no need in a supplementary computation of the
# stress tensor.

# %%
def C_tang_local(sigma_local, dlambda_local, deps_local, sigma_n_local):
    y_local = jnp.c_["0,1,-1", sigma_local, dlambda_local]
    j = drdy(y_local, deps_local, sigma_n_local)
    return jnp.linalg.inv(j)[:4,:4] @ C_elas

C_tang_vec = jax.jit(jax.vmap(C_tang_local, in_axes=(0, 0, 0, 0)))

dsigma_ddeps = jax.jacfwd(return_mapping, has_aux=True)

# %% [markdown]
# #### Defining external operator
#
# Once we define the function `dsigma_ddeps`, which evaluates both the
# external operator and its derivative locally, we can simply vectorize it and
# define the final implementation of the external operator derivative.
#
# ```{note}
# The function `dsigma_ddeps` containing a `while_loop` is designed to be called
# at a single Gauss point that's why we need to vectorize it for the all points
# of our functional space `S`. For this purpose we use the `vmap` function of JAX.
# It creates another `while_loop`, which terminates only when all mapped loops
# terminate. Find further details in this
# [discussion](https://github.com/google/jax/discussions/15954).
# ```

# %%
dsigma_ddeps_vec = jax.jit(jax.vmap(dsigma_ddeps, in_axes=(0, 0)))
return_mapping_vec = jax.jit(jax.vmap(return_mapping, in_axes=(0, 0)))


def C_tang_impl(deps):
    deps_ = deps.reshape((-1, stress_dim))
    sigma_n_ = sigma_n.x.array.reshape((-1, stress_dim))

    # (C_tang_global, state) = dsigma_ddeps_vec(deps_, sigma_n_)
    (sigam_global, state) = return_mapping_vec(deps_, sigma_n_)
    
    sigma_global, niter, yielding, norm_res, dlambda = state

    C_tang_global = C_tang_vec(sigma_global, dlambda, deps_, sigma_n_)

    unique_iters, counts = jnp.unique(niter, return_counts=True)

    max_yielding = jnp.max(yielding)
    max_yielding_rank = MPI.COMM_WORLD.allreduce((max_yielding, MPI.COMM_WORLD.rank), op=MPI.MAXLOC)[1]

    if MPI.COMM_WORLD.rank == max_yielding_rank:
        print("\tInner Newton summary:", flush=True)
        print(f"\t\tUnique number of iterations: {unique_iters}", flush=True)
        print(f"\t\tCounts of unique number of iterations: {counts}", flush=True)
        print(f"\t\tMaximum f: {max_yielding}", flush=True)
        print(f"\t\tMaximum residual: {jnp.max(norm_res)}", flush=True)

    return C_tang_global.reshape(-1), sigma_global.reshape(-1)


# %% [markdown]
# Similarly to the von Mises example, we do not implement explicitly the
# evaluation of the external operator. Instead, we obtain its values during the
# evaluation of its derivative and then update the values of the operator in the
# main Newton loop.


# %%
def sigma_external(derivatives):
    if derivatives == (1,):
        return C_tang_impl
    else:
        return NotImplementedError


sigma.external_function = sigma_external

# %% [markdown]
# ### Defining the forms

# %%
q = fem.Constant(domain, default_scalar_type((0, -gamma)))


def F_ext(v):
    return ufl.dot(q, v) * dx


u_hat = ufl.TrialFunction(V)
F = ufl.inner(epsilon(v), sigma) * dx - F_ext(v)
J = ufl.derivative(F, Du, u_hat)
J_expanded = ufl.algorithms.expand_derivatives(J)

F_replaced, F_external_operators = replace_external_operators(F)
J_replaced, J_external_operators = replace_external_operators(J_expanded)

F_form = fem.form(F_replaced)
J_form = fem.form(J_replaced)

# %% [markdown]
# ### Variables initialization and compilation
#
# Before solving the problem we have to initialize values of the stiffness matrix,
# as it requires for the system assembling. During the first loading step, we
# expect an elastic response only, so it's enough to solve the constitutive
# equations for a relatively small displacement field at each Gauss point. This
# results in initializing the consistent tangent moduli with elastic ones.

# %%
Du.x.array[:] = 1.0
sigma_n.x.array[:] = 0.0

evaluated_operands = evaluate_operands(F_external_operators)
_ = evaluate_external_operators(J_external_operators, evaluated_operands)

# %% [markdown]
# ### Solving the problem
#
# Summing up, we apply the Newton method to solve the main weak problem. On each
# iteration of the main Newton loop, we solve elastoplastic constitutive equations
# by using the second (inner) Newton method at each Gauss point. Thanks to the
# framework and the JAX library, the final interface is general enough to be
# applied to other plasticity models.

# %%
x_point = np.array([[0, H, 0]])
cells, points_on_process = find_cell_by_point(domain, x_point)

# %%
# parameters of the manual Newton method
max_iterations, relative_tolerance = 200, 1e-7

load_steps = np.concatenate([np.linspace(1.5, 22.3, 100)[:-1]])[:10]
# load_steps = np.linspace(2, 21, 40)
num_increments = len(load_steps)
results = np.zeros((num_increments + 1, 2))

external_operator_problem = LinearProblem(J_replaced, -F_replaced, Du, bcs=bcs)

# %%
timer = common.Timer("DOLFINx_timer")
timer_total = common.Timer("Total_timer")
local_monitor = {}
performance_monitor = pd.DataFrame({
    "loading_step": np.array([], dtype=np.int64),
    "Newton_iteration": np.array([], dtype=np.int64),
    "matrix_assembling": np.array([], dtype=np.float64),
    "vector_assembling": np.array([], dtype=np.float64),
    "linear_solver": np.array([], dtype=np.float64),
    "constitutive_model_update": np.array([], dtype=np.float64),
})

if MPI.COMM_WORLD.rank == 0:
    print(f"N = {N}, n = {MPI.COMM_WORLD.Get_size()}", flush=True)

timer_total.start()
comm = MPI.COMM_WORLD
for i, load in enumerate(load_steps):
    local_monitor["loading_step"] = i
    q.value = load * np.array([0, -gamma])
    external_operator_problem.assemble_vector()

    residual_0 = external_operator_problem.b.norm()
    residual = residual_0
    Du.x.array[:] = 0

    if MPI.COMM_WORLD.rank == 0:
        print(f"Load increment #{i}, load: {load}, initial residual: {residual_0}", flush=True)

    for iteration in range(0, max_iterations):
        local_monitor["Newton_iteration"] = iteration
        if residual / residual_0 < relative_tolerance:
            break

        if MPI.COMM_WORLD.rank == 0:
            print(f"\tOuter Newton iteration #{iteration}", flush=True)

        timer.start()
        external_operator_problem.assemble_matrix()
        timer.stop()
        local_monitor["matrix_assembling"] = comm.allreduce(timer.elapsed()[0], op=MPI.MAX)

        timer.start()
        external_operator_problem.solve(du)
        timer.stop()
        local_monitor["linear_solver"] = comm.allreduce(timer.elapsed()[0], op=MPI.MAX)

        Du.x.petsc_vec.axpy(1.0, du.x.petsc_vec)
        Du.x.scatter_forward()

        timer.start()
        evaluated_operands = evaluate_operands(F_external_operators)
        ((_, sigma_new),) = evaluate_external_operators(J_external_operators, evaluated_operands)
        # Direct access to the external operator values
        sigma.ref_coefficient.x.array[:] = sigma_new
        timer.stop()
        local_monitor["constitutive_model_update"] = comm.allreduce(timer.elapsed()[0], op=MPI.MAX)

        timer.start()
        external_operator_problem.assemble_vector()
        timer.stop()
        local_monitor["vector_assembling"] = comm.allreduce(timer.elapsed()[0], op=MPI.MAX)
        performance_monitor.loc[len(performance_monitor.index)] = local_monitor

        residual = external_operator_problem.b.norm()

        if MPI.COMM_WORLD.rank == 0:
            print(f"\tResidual: {residual} Relative: {residual / residual_0}\n", flush=True)

    if MPI.COMM_WORLD.rank == 0:
        print("____________________\n", flush=True)
    u.x.petsc_vec.axpy(1.0, Du.x.petsc_vec)
    u.x.scatter_forward()

    sigma_n.x.array[:] = sigma.ref_coefficient.x.array

    if len(points_on_process) > 0:
        results[i + 1, :] = (-u.eval(points_on_process, cells)[0], load)
timer_total.stop()
total_time = timer_total.elapsed()[0]

if MPI.COMM_WORLD.rank == 0:
    print(f"Slope stability factor: {-q.value[-1]*H/c}", flush=True)
    print(f"Total time: {total_time}", flush=True)
# Load increment #97, load: 22.32070707070707 - OK
# load: 22.535353535353536 - not OK


# %% [markdown]
# ## Verification

# %% [markdown]
# ### Critical load
#
# According to {cite:t}`chenLimitAnalysisSoil1990`, we can derive analytically the
# slope stability factor $l_\text{lim}$ for the standard Mohr-Coulomb plasticity
# model (*without* apex smoothing) under plane strain assumption for associative plastic flow
#
# $$
#     l_\text{lim} = \gamma_\text{lim} H/c,
# $$
#
# where $\gamma_\text{lim}$ is an associated value of the soil self-weight. In
# particular, for the rectangular slope with the friction angle $\phi$ equal to
# $30^\circ$, $l_\text{lim} = 6.69$ {cite}`chenLimitAnalysisSoil1990`. Thus, by
# computing $\gamma_\text{lim}$ from the formula above, we can progressively
# increase the second component of the gravitational body force
# $\boldsymbol{q}=[0, -\gamma]^T$, up to the critical value
# $\gamma_\text{lim}^\text{num}$, when the perfect plasticity plateau is reached
# on the loading-displacement curve at the $(0, H)$ point and then compare
# $\gamma_\text{lim}^\text{num}$ against analytical $\gamma_\text{lim}$.
#
# By demonstrating the loading-displacement curve on the figure below we approve
# that the yield strength limit reached for $\gamma_\text{lim}^\text{num}$ is close to $\gamma_\text{lim}$.

n = MPI.COMM_WORLD.Get_size()
# %%
if len(points_on_process) > 0:
    l_lim = 6.69
    gamma_lim = l_lim / H * c
    plt.plot(results[:-3, 0], results[:-3, 1], "o-", label=r"$\gamma$")
    plt.axhline(y=gamma_lim, color="r", linestyle="--", label=r"$\gamma_\text{lim}$")
    plt.xlabel(r"Displacement of the slope $u_x$ at $(0, H)$ [mm]")
    plt.ylabel(r"Soil self-weight $\gamma$ [MPa/mm$^3$]")
    plt.grid()
    plt.legend()
    plt.savefig(f"output_data/mc_mpi_{N}x{N}_n_{n}.png")

# %%
import pickle
performance_data = {"total_time": total_time, "performance_monitor": performance_monitor}
with open(f"output_data/performance_data_{N}x{N}_n_{n}.pkl", "wb") as f:
        pickle.dump(performance_data, f)


