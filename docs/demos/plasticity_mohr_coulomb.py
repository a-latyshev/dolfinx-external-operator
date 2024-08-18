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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Plasticity of Mohr-Coulomb with apex-smoothing
#
# This tutorial aims to demonstrate how modern automatic or algorithmic differentiation (AD)
# techniques may be used to define a complex constitutive model demanding a lot of
# by-hand differentiation. In particular, we implement the non-associative
# plasticity model of Mohr-Coulomb with apex-smoothing applied to a slope
# stability problem for soil. We use the JAX package to define constitutive
# relations including the differentiation of certain terms and
# `FEMExternalOperator` framework to incorporate this model into a weak
# formulation within UFL.
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
# parallelepiped $[0; L] \times [0; W] \times [0; H]$ with homogeneous Dirichlet
# boundary conditions for the displacement field $\boldsymbol{u} = \boldsymbol{0}$
# on the right side $x = L$ and the bottom one $z = 0$. The loading consists of a
# gravitational body force $\boldsymbol{q}=[0, 0, -\gamma]^T$ with $\gamma$ being
# the soil self-weight. The solution of the problem is to find the collapse load
# $q_\text{lim}$, for which we know an analytical solution in the plane-strain
# case for the standard Mohr-Coulomb criterion. We
# follow the same Mandel-Voigt notation as in the von Mises plasticity tutorial
# but in 3D.
#
# If $V$ is a functional space of admissible displacement fields, then we can
# write out a weak formulation of the problem:
#
# Find $\boldsymbol{u} \in V$ such that
#
# $$
#     F(\boldsymbol{u}; \boldsymbol{v}) = \int\limits_\Omega
#     \boldsymbol{\sigma}(\boldsymbol{u}) \cdot
#     \boldsymbol{\varepsilon}(\boldsymbol{v}) \mathrm{d}\boldsymbol{x} +
#     \int\limits_\Omega \boldsymbol{q} \cdot \boldsymbol{v} = \boldsymbol{0}, \quad
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
from mpi4py import MPI
from petsc4py import PETSc

import jax
import jax.lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pyvista
from mpltools import annotation  # for slope markers
from solvers import LinearProblem
from utilities import find_cell_by_point, Mohr_Coulomb_yield_criterion

import basix
import dolfinx.plot as plot
import ufl
from dolfinx import common, default_scalar_type, fem, mesh
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
E = 6778  # [MPa] Young modulus
nu = 0.25  # [-] Poisson ratio
c = 3.45  # [MPa] cohesion
phi = 30 * np.pi / 180  # [rad] friction angle
psi = 30 * np.pi / 180  # [rad] dilatancy angle
theta_T = 26 * np.pi / 180  # [rad] transition angle as defined by Abbo and Sloan
a = 0.26 * c / np.tan(phi)  # [MPa] tension cuff-off parameter

# %%
L, H = (1.2, 1.0)
Nx, Ny = (50, 50)
gamma = 1.0
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([L, H])], [Nx, Ny])

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
# v = ufl.TrialFunction(V)
u_ = ufl.TestFunction(V)

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
#
# By introducing the residual vector $\boldsymbol{r} = [\boldsymbol{r}_{g}^T,
# r_f]^T$ and its argument vector $\boldsymbol{x} =
# [\boldsymbol{\sigma}_{n+1}^T, \Delta\lambda]^T$ we solve the following nonlinear
# equation:
#
# $$
#     \boldsymbol{r}(\boldsymbol{x}_{n+1}) = \boldsymbol{0}
# $$
#
# To solve this equation we apply the Newton method and introduce the Jacobian of
# the residual vector $\boldsymbol{j} = \frac{\mathrm{d} \boldsymbol{r}}{\mathrm{d}
# \boldsymbol{x}}$. Thus we solve the following linear system at each quadrature
# point for the plastic phase
#
# $$
#     \begin{cases}
#         \boldsymbol{j}(\boldsymbol{x}_{n})\boldsymbol{y} = -
#         \boldsymbol{r}(\boldsymbol{x}_{n}), \\
#         \boldsymbol{x}_{n+1} = \boldsymbol{x}_n + \boldsymbol{y}.
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
# return-mapping procedure and the solution defines the return-mapping
# correction of the stress tensor. By implementation of the external operator
# $\boldsymbol{\sigma}$ we mean the implementation of this *algorithmic* procedure.
#
# The automatic differentiation tools of the JAX library are applied to calculate
# the derivatives $\frac{\mathrm{d} g}{\mathrm{d}\boldsymbol{\sigma}}, \frac{\mathrm{d}
# \boldsymbol{r}}{\mathrm{d} \boldsymbol{x}}$ as well as the stress tensor
# derivative or the consistent tangent stiffness matrix $\boldsymbol{C}_\text{tang} =
# \frac{\mathrm{d}\boldsymbol{\sigma}}{\mathrm{d}\boldsymbol{\varepsilon}}$.
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
# its jacobian `drdx`.

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


def r(x_local, deps_local, sigma_n_local):
    sigma_local = x_local[:stress_dim]
    dlambda_local = x_local[-1]

    res_g = r_g(sigma_local, dlambda_local, deps_local, sigma_n_local)
    res_f = r_f(sigma_local, dlambda_local, deps_local, sigma_n_local)

    res = jnp.c_["0,1,-1", res_g, res_f]  # concatenates an array and a scalar
    return res


drdx = jax.jacfwd(r)

# %% [markdown]
# Then we define the function `return_mapping` that implements the
# return-mapping algorithm numerically via the Newton method.

# %%
Nitermax, tol = 200, 1e-10

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
    x_local = jnp.concatenate([sigma_local, dlambda])

    res = r(x_local, deps_local, sigma_n_local)
    norm_res0 = jnp.linalg.norm(res)

    def cond_fun(state):
        norm_res, niter, _ = state
        return jnp.logical_and(norm_res / norm_res0 > tol, niter < Nitermax)

    def body_fun(state):
        norm_res, niter, history = state

        x_local, deps_local, sigma_n_local, res = history

        j = drdx(x_local, deps_local, sigma_n_local)
        j_inv_vp = jnp.linalg.solve(j, -res)
        x_local = x_local + j_inv_vp

        res = r(x_local, deps_local, sigma_n_local)
        norm_res = jnp.linalg.norm(res)
        history = x_local, deps_local, sigma_n_local, res

        niter += 1

        return (norm_res, niter, history)

    history = (x_local, deps_local, sigma_n_local, res)

    norm_res, niter_total, x_local = jax.lax.while_loop(cond_fun, body_fun, (norm_res0, niter, history))

    sigma_local = x_local[0][:stress_dim]
    dlambda = x_local[0][-1]
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
# consistent tangent matrix $\boldsymbol{C}_\text{tang}$, this feature becomes
# very useful, as there is no need to write an additional program computing the
# stress derivative.
#
# JAX's AD tool permits taking the derivative of the function `return_mapping`,
# which is factually the while loop. The derivative is taken with respect to the
# first output and the remaining outputs are used as auxiliary data. Thus, the
# derivative `dsigma_ddeps` returns both values of the consistent tangent matrix
# and the stress tensor, so there is no need in a supplementary computation of the
# stress tensor.


# %%
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


def C_tang_impl(deps):
    deps_ = deps.reshape((-1, stress_dim))
    sigma_n_ = sigma_n.x.array.reshape((-1, stress_dim))

    (C_tang_global, state) = dsigma_ddeps_vec(deps_, sigma_n_)
    sigma_global, niter, yielding, norm_res, dlambda = state

    unique_iters, counts = jnp.unique(niter, return_counts=True)

    print("\tInner Newton summary:")
    print(f"\t\tUnique number of iterations: {unique_iters}")
    print(f"\t\tCounts of unique number of iterations: {counts}")
    print(f"\t\tMaximum f: {jnp.max(yielding)}")
    print(f"\t\tMaximum residual: {jnp.max(norm_res)}")

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
F = ufl.inner(epsilon(u_), sigma) * dx - F_ext(u_)
J = ufl.derivative(F, Du, u_hat)
J_expanded = ufl.algorithms.expand_derivatives(J)

F_replaced, F_external_operators = replace_external_operators(F)
J_replaced, J_external_operators = replace_external_operators(J_expanded)

F_form = fem.form(F_replaced)
J_form = fem.form(J_replaced)

# %% [markdown]
# ### Variables initialization and compilation
#
# Before solving the problem we have to initialize values of the consistent
# tangent matrix, as it requires for the system assembling. During the first load
# step, we expect an elastic response only, so it's enough two to solve the
# constitutive equations for any small displacements at each Gauss point. This
# results in initializing the consistent tangent matrix with elastic moduli.
#
# At the same time, we can measure the compilation overhead caused by the first
# call of JIT-ed JAX functions.

# %%
Du.x.array[:] = 1.0
sigma_n.x.array[:] = 0.0

timer = common.Timer("DOLFINx_timer")
timer.start()
evaluated_operands = evaluate_operands(F_external_operators)
_ = evaluate_external_operators(J_external_operators, evaluated_operands)
timer.stop()
pass_1 = timer.elapsed()[0]

timer.start()
evaluated_operands = evaluate_operands(F_external_operators)
_ = evaluate_external_operators(J_external_operators, evaluated_operands)
timer.stop()
pass_2 = timer.elapsed()[0]

print(f"\nJAX's JIT compilation overhead: {pass_1 - pass_2}")

# %% [markdown]
# ### Solving the problem
#
# Summing up, we apply the Newton method to solve the main weak problem. On each
# iteration of the main Newton loop, we solve elastoplastic constitutive equations
# by using the second, inner, Newton method at each Gauss point. Thanks to the
# framework and the JAX library, the final interface is general enough to be
# applied to other plasticity models.

# %%
external_operator_problem = LinearProblem(J_replaced, -F_replaced, Du, bcs=bcs)

# %%
x_point = np.array([[0, H, 0]])
cells, points_on_process = find_cell_by_point(domain, x_point)

# %%
# parameters of the manual Newton method
max_iterations, relative_tolerance = 200, 1e-8

load_steps_1 = np.linspace(2, 21, 40)
load_steps_2 = np.linspace(21, 22.75, 20)[1:]
load_steps = np.concatenate([load_steps_1, load_steps_2])
num_increments = len(load_steps)
results = np.zeros((num_increments + 1, 2))

# %% tags=["scroll-output"]

for i, load in enumerate(load_steps):
    q.value = load * np.array([0, -gamma])
    external_operator_problem.assemble_vector()

    residual_0 = external_operator_problem.b.norm()
    residual = residual_0
    Du.x.array[:] = 0

    if MPI.COMM_WORLD.rank == 0:
        print(f"Load increment #{i}, load: {load}, initial residual: {residual_0}")

    for iteration in range(0, max_iterations):
        if residual / residual_0 < relative_tolerance:
            break

        if MPI.COMM_WORLD.rank == 0:
            print(f"\tOuter Newton iteration #{iteration}")
        external_operator_problem.assemble_matrix()
        external_operator_problem.solve(du)

        Du.vector.axpy(1.0, du.vector)
        Du.x.scatter_forward()

        evaluated_operands = evaluate_operands(F_external_operators)
        ((_, sigma_new),) = evaluate_external_operators(J_external_operators, evaluated_operands)
        # Direct access to the external operator values
        sigma.ref_coefficient.x.array[:] = sigma_new

        external_operator_problem.assemble_vector()
        residual = external_operator_problem.b.norm()

        if MPI.COMM_WORLD.rank == 0:
            print(f"\tResidual: {residual}\n")

    u.vector.axpy(1.0, Du.vector)
    u.x.scatter_forward()

    sigma_n.x.array[:] = sigma.ref_coefficient.x.array

    if len(points_on_process) > 0:
        results[i + 1, :] = (-u.eval(points_on_process, cells)[0], load)

print(f"Slope stability factor: {-q.value[-1]*H/c}")

# %%
# 20 - critical load # -5.884057971014492
# Slope stability factor: -6.521739130434782


# %% [markdown]
# ## Verification

# %% [markdown]
# ### Critical load

# %%
if len(points_on_process) > 0:
    plt.plot(results[:, 0], results[:, 1], "o-")
    plt.xlabel("Displacement of the slope at (0, H)")
    plt.ylabel(r"Soil self-weight $\gamma$")
    plt.savefig(f"displacement_rank{MPI.COMM_WORLD.rank:d}.png")
    plt.show()

# %%
print(f"Slope stability factor for 2D plane strain factor [Chen]: {6.69}")
print(f"Computed slope stability factor: {22.75*H/c}")

# %%
W = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
u_tmp = fem.Function(W, name="Displacement")
u_tmp.interpolate(u)

pyvista.start_xvfb()
plotter = pyvista.Plotter(window_size=[600, 400])
topology, cell_types, x = plot.vtk_mesh(domain)
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
vals = np.zeros((x.shape[0], 3))
vals[:, : len(u_tmp)] = u_tmp.x.array.reshape((x.shape[0], len(u_tmp)))
grid["u"] = vals
warped = grid.warp_by_vector("u", factor=20)
plotter.add_text("Displacement field", font_size=11)
plotter.add_mesh(warped, show_edges=False, show_scalar_bar=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()


# %% [markdown]
# ### Yield surface
#
# We verify that the constitutive model is correctly implemented by tracing the
# yield surface. We generate several stress paths and check whether they remain
# within the yield surface. The stress tracing is performed in the
# [Haigh-Westergaard coordinates](https://en.wikipedia.org/wiki/Lode_coordinates)
# $(\xi, \rho, \theta)$ which are defined as follows
#
# $$
#     \xi = \frac{1}{\sqrt{3}}I_1, \quad \rho =
#     \sqrt{2J_2}, \quad \cos(3\theta) = \frac{3\sqrt{3}}{2}
#     \frac{J_3}{J_2^{3/2}},
# $$
# where $J_3(\boldsymbol{\sigma}) = \det(\boldsymbol{s})$ is the third invariant
# of the deviatoric part of the stress tensor, $\xi$ is the deviatoric coordinate,
# $\rho$ is the radial coordinate and the angle $\theta \in
# [-\frac{\pi}{6}, \frac{\pi}{6}]$ is called Lode or stress angle.
#
# By introducing the hydrostatic variable $p = \xi/\sqrt{3}$ The principal
# stresses can be written in Haigh-Westergaard coordinates
#
# $$
#     \begin{pmatrix}
#         \sigma_{I} \\
#         \sigma_{II} \\
#         \sigma_{III} \\
#     \end{pmatrix}
#     = p
#     \begin{pmatrix}
#         1 \\
#         1 \\
#         1 \\
#     \end{pmatrix}
#     + \sqrt{\frac{2}{3}}\rho
#     \begin{pmatrix}
#         \cos{\theta} \\
#         -\sin{\frac{\pi}{6} - \theta} \\
#         -\sin{\frac{\pi}{6} + \theta}
#     \end{pmatrix}.
# $$
#
# Firstly, we define and vectorize functions `rho`, `angle` and `sigma_tracing`
# evaluating respectively the coordinates $\rho$ and $\theta$ and the corrected
# stress tensor for a certain stress state.


# %%
def rho(sigma_local):
    s = dev @ sigma_local
    return jnp.sqrt(2.0 * J2(s))


def angle(sigma_local):
    s = dev @ sigma_local
    arg = -(3.0 * jnp.sqrt(3.0) * J3(s)) / (2.0 * jnp.sqrt(J2(s) * J2(s) * J2(s)))
    arg = jnp.clip(arg, -1.0, 1.0)
    angle = 1.0 / 3.0 * jnp.arcsin(arg)
    return angle

MC_return_mapping = Mohr_Coulomb_yield_criterion(phi, c, E, nu)

def sigma_tracing(sigma_local, sigma_n_local):
    deps_elas = S_elas @ sigma_local
    sigma_corrected, state = return_mapping(deps_elas, sigma_n_local)
    yielding = state[2]
    return sigma_corrected, yielding


angle_v = jax.jit(jax.vmap(angle, in_axes=(0)))
rho_v = jax.jit(jax.vmap(rho, in_axes=(0)))
sigma_tracing_vec = jax.jit(jax.vmap(sigma_tracing, in_axes=(0, 0)))

# %% [markdown]
# Secondly, we generate a loading path by evaluating principal stresses through
# Haigh-Westergaard coordinates, where $\rho$ and $\xi$ are fixed ones.

# %%
N_angles = 200
N_loads = 10
eps = 0.5
R = 0.7
p = 0.0

angle_values = np.linspace(0, np.pi/3, N_angles)
dsigma_path = np.zeros((N_angles, stress_dim))
dsigma_path[:, 0] = np.sqrt(2.0 / 3.0) * R * np.cos(angle_values)
dsigma_path[:, 1] = np.sqrt(2.0 / 3.0) * R * np.sin(angle_values - np.pi / 6.0)
dsigma_path[:, 2] = np.sqrt(2.0 / 3.0) * R * np.sin(-angle_values - np.pi / 6.0)

angle_results = np.empty((N_loads, N_angles))
rho_results = np.empty((N_loads, N_angles))
sigma_results = np.empty((N_loads, N_angles, stress_dim))
sigma_n_local = np.zeros_like(dsigma_path)
sigma_n_local[:, 0] = p
sigma_n_local[:, 1] = p
sigma_n_local[:, 2] = p
derviatoric_axis = tr

# %% tags=["scroll-output"]
print(f"rho = {R}, p = {p} - projection onto the octahedral plane\n")
for i in range(N_loads):
    print(f"Loading#{i}")
    dsigma, yielding = sigma_tracing_vec(dsigma_path, sigma_n_local)
    dp = dsigma @ tr / 3.0 - p
    dsigma -= np.outer(dp, derviatoric_axis)  # projection on the same octahedral plane

    sigma_results[i, :] = dsigma
    angle_results[i, :] = angle_v(dsigma)
    rho_results[i, :] = rho_v(dsigma)
    print(f"max f: {jnp.max(yielding)}\n")
    sigma_n_local[:] = dsigma

# %% [markdown]
# Finally, the stress paths are represented by a series of circles lying in each other in
# the same octahedral plane. By applying the return-mapping algorithm defined in
# the function `return_mapping`, we perform the correction of the stress
# paths. Once they get close to the elastic limit the traced curves look similar
# to the Mohr-Coulomb yield surface with apex smoothing which indicates the
# correct implementation of the constitutive model.

# %%
fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))
for j in range(12):
    for i in range(N_loads):
        ax.plot(j * np.pi / 3 - j % 2 * angle_results[i] + (1 - j % 2) * angle_results[i], rho_results[i], ".")

ax.set_yticklabels([])
fig.tight_layout()

# %% [markdown]
# ### Taylor test
#
# The derivatives, on which the form $F$ and its jacobian $J$ are based, are
# automatically derived using the JAX AD tools. In this regards, we perform the
# Taylor test to ensure that these derivatives are computed correctly.
#
# Indeed, by following the Taylor's theorem and perturbating the functional $F: V
# \to \mathbb{R}$ in the direction $h \, \boldsymbol{δu} \in V$ for $h > 0$, the
# first and second order Taylor reminders $R_0$ and $R_1$ have the following
# convergence rates
#
# $$
#     R_0 = | F(\boldsymbol{u} + h \, \boldsymbol{δu}; \boldsymbol{v}) -
# F(\boldsymbol{u}; \boldsymbol{v}) | \longrightarrow 0 \text{ at } O(h),
# $$
#
# $$
#     R_1 = | F(\boldsymbol{u} + h \, \boldsymbol{δu}; \boldsymbol{v}) -
# F(\boldsymbol{u}; \boldsymbol{v}) - \, J(\boldsymbol{u}; h\boldsymbol{δu},
# \boldsymbol{v}) | \longrightarrow 0 \text{ at } O(h^2),
# $$
#
# In the following code-blocks you may find the implementation of the Taylor test
# justifying the first and second convergence rates.

# %%
# Reset main variables to zero including the external operators values
sigma_n.x.array[:] = 0.0
sigma.ref_coefficient.x.array[:] = 0.0
J_external_operators[0].ref_coefficient.x.array[:] = 0.0
# Reset the values of the consistent tangent matrix to elastic moduli
Du.x.array[:] = 1.0
evaluated_operands = evaluate_operands(F_external_operators)
_ = evaluate_external_operators(J_external_operators, evaluated_operands)

# %% [markdown]
# As the derivatives of the constitutive model are different for elastic and
# plastic phases, we must consider two initial states for the Taylor test. For
# this reason, we solve the problem once for a certain loading value to get the
# initial state close to the one with plastic deformations.

# %%
i = 0
load = 2.0
q.value = load * np.array([0, -gamma])
external_operator_problem.assemble_vector()

residual_0 = external_operator_problem.b.norm()
residual = residual_0
Du.x.array[:] = 0

if MPI.COMM_WORLD.rank == 0:
    print(f"Load increment #{i}, load: {load}, initial residual: {residual_0}")

for iteration in range(0, max_iterations):
    if residual / residual_0 < relative_tolerance:
        break

    if MPI.COMM_WORLD.rank == 0:
        print(f"\tOuter Newton iteration #{iteration}")
    external_operator_problem.assemble_matrix()
    external_operator_problem.solve(du)

    Du.vector.axpy(1.0, du.vector)
    Du.x.scatter_forward()

    evaluated_operands = evaluate_operands(F_external_operators)
    ((_, sigma_new),) = evaluate_external_operators(J_external_operators, evaluated_operands)

    sigma.ref_coefficient.x.array[:] = sigma_new

    external_operator_problem.assemble_vector()
    residual = external_operator_problem.b.norm()

    if MPI.COMM_WORLD.rank == 0:
        print(f"\tResidual: {residual}\n")

sigma_n.x.array[:] = sigma.ref_coefficient.x.array

# Initial values of the displacement field and the stress state for the Taylor
# test
Du0 = np.copy(Du.x.array)
sigma_n0 = np.copy(sigma_n.x.array)

# %% [markdown]
# If we take into account the initial stress state `sigma_n0` computed in the cell
# above, we perform the Taylor test for the plastic phase, otherwise we stay in
# the elastic one.

# %%
R_form = fem.form(ufl.inner(ufl.grad(u_hat), ufl.grad(u_)) * ufl.dx)
R = fem.petsc.assemble_matrix(R_form, bcs=bcs)
R.assemble()
Riesz_solver = PETSc.KSP().create(domain.comm)
Riesz_solver.setType("preonly")
Riesz_solver.getPC().setType("lu")
Riesz_solver.setOperators(R)

RT = fem.Function(V, name="Riesz_representer_of_T") # T - a Taylor remainder

# %%
h_list = np.logspace(-2.0, -6.0, 5)[::-1]

def perform_Taylor_test(Du0, sigma_n0):
    # F(Du0 + h*δu) - F(Du0) - h*J(Du0)*δu
    Du.x.array[:] = Du0
    sigma_n.x.array[:] = sigma_n0
    evaluated_operands = evaluate_operands(F_external_operators)
    ((_, sigma_new),) = evaluate_external_operators(J_external_operators, evaluated_operands)
    sigma.ref_coefficient.x.array[:] = sigma_new

    F0 = fem.petsc.assemble_vector(F_form)  # F(Du0)
    F0.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    # fem.set_bc(F0, bcs)

    J0 = fem.petsc.assemble_matrix(J_form)
    J0.assemble()  # J(Du0)
    y = J0.createVecLeft()  # y = J0 @ x

    δu = fem.Function(V)
    δu.x.array[:] = Du0  # δu == Du0

    zero_order_remainder = np.zeros_like(h_list)
    first_order_remainder = np.zeros_like(h_list)

    for i, h in enumerate(h_list):
        Du.x.array[:] = Du0 + h * δu.x.array
        evaluated_operands = evaluate_operands(F_external_operators)
        ((_, sigma_new),) = evaluate_external_operators(J_external_operators, evaluated_operands)
        sigma.ref_coefficient.x.array[:] = sigma_new

        F_delta = fem.petsc.assemble_vector(F_form)  # F(Du0 + h*δu)
        F_delta.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        # fem.set_bc(F0, bcs)

        J0.mult(δu.vector, y)  # y = J(Du0)*δu
        y.scale(h)  # y = h*y

        T0 = F_delta - F0
        T1 = F_delta - F0 - y

        Riesz_solver.solve(T0, RT.vector) # RT = R^{-1} T0
        RT.x.scatter_forward()
        zero_order_remainder[i] = np.sqrt(T0.dot(RT.vector)) # sqrt{T0^T R^{-1} T0}

        Riesz_solver.solve(T1, RT.vector) # RT = R^{-1} T1
        RT.x.scatter_forward()
        first_order_remainder[i] = np.sqrt(T1.dot(RT.vector)) # sqrt{T1^T R^{-1} T1}

    return zero_order_remainder, first_order_remainder


print("Elastic phase")
zero_order_remainder_elastic, first_order_remainder_elastic = perform_Taylor_test(Du0, 0.0)
print("Plastic phase")
zero_order_remainder_plastic, first_order_remainder_plastic = perform_Taylor_test(Du0, sigma_n0)

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].loglog(h_list, zero_order_remainder_elastic, "o-", label=r"$R_0$")
axs[0].loglog(h_list, first_order_remainder_elastic, "o-", label=r"$R_1$")
annotation.slope_marker((5e-5, 5e-6), 1, ax=axs[0], poly_kwargs={"facecolor": "tab:blue"})

axs[1].loglog(h_list, zero_order_remainder_plastic, "o-", label=r"$R_0$")
annotation.slope_marker((5e-5, 5e-6), 1, ax=axs[1], poly_kwargs={"facecolor": "tab:blue"})
axs[1].loglog(h_list, first_order_remainder_plastic, "o-", label=r"$R_1$")
annotation.slope_marker((1e-4, 5e-13), 2, ax=axs[1], poly_kwargs={"facecolor": "tab:orange"})

for i in range(2):
    axs[i].set_xlabel("h")
    axs[i].set_ylabel("Taylor remainder")
    axs[i].legend()
    axs[i].grid()

plt.tight_layout()

first_order_rate = np.polyfit(np.log(h_list), np.log(zero_order_remainder_elastic), 1)[0]
second_order_rate = np.polyfit(np.log(h_list), np.log(first_order_remainder_elastic), 1)[0]
print(f"Elastic phase:\n\tthe 1st order rate = {first_order_rate:.2f}\n\tthe 2nd order rate = {second_order_rate:.2f}")
first_order_rate = np.polyfit(np.log(h_list), np.log(zero_order_remainder_plastic), 1)[0]
second_order_rate = np.polyfit(np.log(h_list[1:]), np.log(first_order_remainder_plastic[1:]), 1)[0]
print(f"Plastic phase:\n\tthe 1st order rate = {first_order_rate:.2f}\n\tthe 2nd order rate = {second_order_rate:.2f}")

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].loglog(h_list, zero_order_remainder_elastic, "o-", label=r"$R_0$")
axs[0].loglog(h_list, first_order_remainder_elastic, "o-", label=r"$R_1$")
annotation.slope_marker((5e-5, 5e-6), 1, ax=axs[0], poly_kwargs={"facecolor": "tab:blue"})

axs[1].loglog(h_list, zero_order_remainder_plastic, "o-", label=r"$R_0$")
annotation.slope_marker((5e-5, 5e-6), 1, ax=axs[1], poly_kwargs={"facecolor": "tab:blue"})
axs[1].loglog(h_list, first_order_remainder_plastic, "o-", label=r"$R_1$")
annotation.slope_marker((1e-4, 5e-13), 2, ax=axs[1], poly_kwargs={"facecolor": "tab:orange"})

for i in range(2):
    axs[i].set_xlabel("h")
    axs[i].set_ylabel("Taylor remainder")
    axs[i].legend()
    axs[i].grid()

plt.tight_layout()

first_order_rate = np.polyfit(np.log(h_list), np.log(zero_order_remainder_elastic), 1)[0]
second_order_rate = np.polyfit(np.log(h_list), np.log(first_order_remainder_elastic), 1)[0]
print(f"Elastic phase:\n\tthe 1st order rate = {first_order_rate:.2f}\n\tthe 2nd order rate = {second_order_rate:.2f}")
first_order_rate = np.polyfit(np.log(h_list), np.log(zero_order_remainder_plastic), 1)[0]
second_order_rate = np.polyfit(np.log(h_list[1:]), np.log(first_order_remainder_plastic[1:]), 1)[0]
print(f"Plastic phase:\n\tthe 1st order rate = {first_order_rate:.2f}\n\tthe 2nd order rate = {second_order_rate:.2f}")

# %% [markdown]
# For the elastic phase (on the left) the zeroth-order Taylor remainder $R_0$
# achieves the first-order convergence rate the same as for the plastic phase (on
# the right). The first-order remainder $R_1$ is constant during the elastic
# response, as the jacobian is constant in this case contrarily to the plastic
# phase, where $R_1$ has the second-order convergence.
