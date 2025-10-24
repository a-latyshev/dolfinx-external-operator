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
#     display_name: dolfinx-env (3.12.3)
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
from mpi4py import MPI
from petsc4py import PETSc

import jax
import jax.lax
import jax.numpy as jnp
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from mpltools import annotation  # for slope markers
from utilities import find_cell_by_point

import basix
import ufl
from dolfinx import default_scalar_type, fem, mesh
from dolfinx.fem.petsc import NonlinearProblem
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
Nx, Ny = (25, 25)
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
Nitermax, tol = 200, 1e-8

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
    sigma_global, niter, yielding, norm_res, _dlambda = state

    unique_iters, counts = jnp.unique(niter, return_counts=True)

    if MPI.COMM_WORLD.rank == 0:
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
        raise NotImplementedError(f"No external function is defined for the requested derivative {derivatives}.")


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
# Similarly to the von Mises tutorial, we use a Newton solver, but this time we
# rely on `SNES`, the implementation from the `PETSc` library. We implemented the
# class `PETScNonlinearProblem` that allows to call an additional routine
# `external_callback` at each iteration of SNES before the vector and matrix
# assembly.


# %%
def constitutive_update():
    evaluated_operands = evaluate_operands(F_external_operators)
    ((_, sigma_new),) = evaluate_external_operators(J_external_operators, evaluated_operands)
    # Direct access to the external operator values
    sigma.ref_coefficient.x.array[:] = sigma_new


petsc_options = {
    "snes_type": "vinewtonrsls",
    "snes_linesearch_type": "basic",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_atol": 1.0e-8,
    "snes_rtol": 1.0e-8,
    "snes_max_it": 100,
    "snes_monitor": "",
}

problem = NonlinearProblem(
    F_replaced, Du, J=J_replaced, bcs=bcs, petsc_options_prefix="demo_mohr-coulomb_", petsc_options=petsc_options
)


def assemble_residual_with_callback(snes: PETSc.SNES, x: PETSc.Vec, b: PETSc.Vec) -> None:
    """Assemble the residual F into the vector b with a callback to external functions.

    snes: the snes object
    x: Vector containing the latest solution.
    b: Vector to assemble the residual into.
    """
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    x.copy(Du.x.petsc_vec)
    Du.x.scatter_forward()

    # Call external functions, e.g. evaluation of external operators
    constitutive_update()

    with b.localForm() as b_local:
        b_local.set(0.0)
    fem.petsc.assemble_vector(b, problem._F)

    fem.petsc.apply_lifting(b, [problem._J], [bcs], [x], -1.0)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, bcs, x, -1.0)


problem.solver.setFunction(assemble_residual_with_callback, problem.b)


# %% [markdown]
# After definition of the nonlinear problem and the Newton solver, we are ready to
# get the final result.

# %% tags=["scroll-output"]
load_steps_1 = np.linspace(2, 22.9, 50)
load_steps_2 = np.array([22.96, 22.99])
load_steps = np.concatenate([load_steps_1, load_steps_2])
num_increments = len(load_steps)
results = np.zeros((num_increments + 1, 2))

x_point = np.array([[0, H, 0]])
cells, points_on_process = find_cell_by_point(domain, x_point)

for i, load in enumerate(load_steps):
    q.value = load * np.array([0, -gamma])

    if MPI.COMM_WORLD.rank == 0:
        print(f"Load increment #{i}, load: {load}")

    problem.solve()

    u.x.petsc_vec.axpy(1.0, Du.x.petsc_vec)
    u.x.scatter_forward()

    sigma_n.x.array[:] = sigma.ref_coefficient.x.array

    if len(points_on_process) > 0:
        results[i + 1, :] = (-u.eval(points_on_process, cells)[0], load)

print(f"Slope stability factor: {-q.value[-1] * H / c}")

# %% [markdown]
# ```{note}
# We demonstrated here the use of `PETSc.SNES` together with external operators
# through the `PETScNonlinearProblem` and `PETScNonlinearSolver` classes. If the
# user is familiar with original DOLFINx `NonlinearProblem`, feel free to
# use `NonlinearProblemWithCallback` covered in the [von Mises tutorial](demo_plasticity_von_mises.py).
# ```

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

# %%
if len(points_on_process) > 0:
    l_lim = 6.69
    gamma_lim = l_lim / H * c
    plt.plot(results[:, 0], results[:, 1], "o-", label=r"$\gamma$")
    plt.axhline(y=gamma_lim, color="r", linestyle="--", label=r"$\gamma_\text{lim}$")
    plt.xlabel(r"Displacement of the slope $u_x$ at $(0, H)$ [mm]")
    plt.ylabel(r"Soil self-weight $\gamma$ [MPa/mm$^3$]")
    plt.grid()
    plt.legend()

# %% [markdown]
# The slope profile reaching its stability limit:

# %%
try:
    import pyvista

    print(pyvista.global_theme.jupyter_backend)
    import dolfinx.plot

    W = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
    u_tmp = fem.Function(W, name="Displacement")
    u_tmp.interpolate(u)

    plotter = pyvista.Plotter(window_size=[600, 400], off_screen=True)
    topology, cell_types, x = dolfinx.plot.vtk_mesh(domain)
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)
    vals = np.zeros((x.shape[0], 3))
    vals[:, : len(u_tmp)] = u_tmp.x.array.reshape((x.shape[0], len(u_tmp)))
    grid["u"] = vals
    warped = grid.warp_by_vector("u", factor=20)
    plotter.add_mesh(warped, show_edges=False, show_scalar_bar=False)
    plotter.view_xy()
    plotter.camera.tight()
    image = plotter.screenshot(None, transparent_background=True, return_img=True)
    plt.imshow(image)
    plt.axis("off")

except ImportError:
    print("pyvista required for this plot")

# %% [markdown]
# ### Yield surface
#
# We verify that the constitutive model is correctly implemented by tracing the
# yield surface. We generate several stress paths and check whether they remain
# within the Mohr-Coulomb yield surface. The stress tracing is performed in the
# [Haigh-Westergaard coordinates](https://en.wikipedia.org/wiki/Lode_coordinates)
# $(\xi, \rho, \theta)$ which are defined as follows
#
# $$
#     \xi = \frac{1}{\sqrt{3}}I_1, \quad \rho =
#     \sqrt{2J_2}, \quad \sin(3\theta) = -\frac{3\sqrt{3}}{2}
#     \frac{J_3}{J_2^{3/2}},
# $$
# where $J_3(\boldsymbol{\sigma}) = \det(\boldsymbol{s})$ is the third invariant
# of the deviatoric part of the stress tensor, $\xi$ is the deviatoric coordinate,
# $\rho$ is the radial coordinate and the angle $\theta \in
# [-\frac{\pi}{6}, \frac{\pi}{6}]$ is called Lode or stress angle.
#
# To generate the stress paths we use the principal
# stresses formula written in Haigh-Westergaard coordinates as follows
#
# $$
#     \begin{pmatrix}
#         \sigma_{I} \\
#         \sigma_{II} \\
#         \sigma_{III}
#     \end{pmatrix}
#     = p
#     \begin{pmatrix}
#         1 \\
#         1 \\
#         1
#     \end{pmatrix}
#     + \frac{\rho}{\sqrt{2}}
#     \begin{pmatrix}
#         \cos\theta + \frac{\sin\theta}{\sqrt{3}} \\
#         -\frac{2\sin\theta}{\sqrt{3}} \\
#         \frac{\sin\theta}{\sqrt{3}} - \cos\theta
#     \end{pmatrix},
# $$
#
# where $p = \xi/\sqrt{3}$ is a hydrostatic variable and $\sigma_{I} \geq
# \sigma_{II} \geq \sigma_{III}$.
#
# Now we generate the loading path by evaluating principal stresses in
# Haigh-Westergaard coordinates for the Lode angle $\theta$ being varied from
# $-\frac{\pi}{6}$ to $\frac{\pi}{6}$ with fixed $\rho$ and $p$.

# %%
N_angles = 50
N_loads = 9  # number of loadings or paths
eps = 0.00001
R = 0.7  # fix the values of rho
p = 0.1  # fix the deviatoric coordinate
theta_1 = -np.pi / 6
theta_2 = np.pi / 6

theta_values = np.linspace(theta_1 + eps, theta_2 - eps, N_angles)
theta_returned = np.empty((N_loads, N_angles))
rho_returned = np.empty((N_loads, N_angles))
sigma_returned = np.empty((N_loads, N_angles, stress_dim))

# fix an increment of the stress path
dsigma_path = np.zeros((N_angles, stress_dim))
dsigma_path[:, 0] = (R / np.sqrt(2)) * (np.cos(theta_values) + np.sin(theta_values) / np.sqrt(3))
dsigma_path[:, 1] = (R / np.sqrt(2)) * (-2 * np.sin(theta_values) / np.sqrt(3))
dsigma_path[:, 2] = (R / np.sqrt(2)) * (np.sin(theta_values) / np.sqrt(3) - np.cos(theta_values))

sigma_n_local = np.zeros_like(dsigma_path)
sigma_n_local[:, 0] = p
sigma_n_local[:, 1] = p
sigma_n_local[:, 2] = p
derviatoric_axis = tr


# %% [markdown]
# Then, we define and vectorize functions `rho`, `Lode_angle` and `sigma_tracing`
# evaluating respectively the coordinates $\rho$, $\theta$ and the corrected (or
# "returned") stress tensor for a certain stress state. `sigma_tracing` calls the
# function `return_mapping`, where the constitutive model was defined via JAX
# previously.


# %%
def rho(sigma_local):
    s = dev @ sigma_local
    return jnp.sqrt(2.0 * J2(s))


def Lode_angle(sigma_local):
    s = dev @ sigma_local
    arg = -(3.0 * jnp.sqrt(3.0) * J3(s)) / (2.0 * jnp.sqrt(J2(s) * J2(s) * J2(s)))
    arg = jnp.clip(arg, -1.0, 1.0)
    angle = 1.0 / 3.0 * jnp.arcsin(arg)
    return angle


def sigma_tracing(sigma_local, sigma_n_local):
    deps_elas = S_elas @ sigma_local
    sigma_corrected, state = return_mapping(deps_elas, sigma_n_local)
    yielding = state[2]
    return sigma_corrected, yielding


Lode_angle_v = jax.jit(jax.vmap(Lode_angle, in_axes=(0)))
rho_v = jax.jit(jax.vmap(rho, in_axes=(0)))
sigma_tracing_v = jax.jit(jax.vmap(sigma_tracing, in_axes=(0, 0)))

# %% [markdown]
# For each stress path, we call the function `sigma_tracing_v` to get the
# corrected stress state and then we project it onto the deviatoric plane $(\rho,
# \theta)$ with a fixed value of $p$.

# %% tags=["scroll-output"]
for i in range(N_loads):
    print(f"Loading path#{i}")
    dsigma, yielding = sigma_tracing_v(dsigma_path, sigma_n_local)
    dp = dsigma @ tr / 3.0 - p
    dsigma -= np.outer(dp, derviatoric_axis)  # projection on the same deviatoric plane

    sigma_returned[i, :] = dsigma
    theta_returned[i, :] = Lode_angle_v(dsigma)
    rho_returned[i, :] = rho_v(dsigma)
    print(f"max f: {jnp.max(yielding)}\n")
    sigma_n_local[:] = dsigma


# %% [markdown]
# Then, by knowing the expression of the [standrad
# Mohr-Coulomb](https://en.wikipedia.org/wiki/Mohr%E2%80%93Coulomb_theory) yield
# surface in principle stresses, we can obtain an analogue expression in
# Haigh-Westergaard coordinates, which leads us to the following equation:
#
#
# $$
#     \frac{\rho}{\sqrt{6}}(\sqrt{3}\cos\theta + \sin\phi
#     \sin\theta) - p\sin\phi - c\cos\phi= 0.
# $$ (eq:standard_MC)
#
# Thus, we restore the standard Mohr-Coulomb yield surface:


# %%
def MC_yield_surface(theta_, p):
    """Restores the coordinate `rho` satisfying the standard Mohr-Coulomb yield
    criterion."""
    rho = (np.sqrt(2) * (c * np.cos(phi) + p * np.sin(phi))) / (
        np.cos(theta_) - np.sin(phi) * np.sin(theta_) / np.sqrt(3)
    )
    return rho


rho_standard_MC = MC_yield_surface(theta_values, p)

# %% [markdown]
# Finally, we plot the yield surface:

# %%
colormap = cm.plasma
colors = colormap(np.linspace(0.0, 1.0, N_loads))

fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))
# Mohr-Coulomb yield surface with apex smoothing
for i, color in enumerate(colors):
    rho_total = np.array([])
    theta_total = np.array([])
    for j in range(12):
        angles = j * np.pi / 3 - j % 2 * theta_returned[i] + (1 - j % 2) * theta_returned[i]
        theta_total = np.concatenate([theta_total, angles])
        rho_total = np.concatenate([rho_total, rho_returned[i]])

    ax.plot(theta_total, rho_total, ".", color=color)

# standard Mohr-Coulomb yield surface
theta_standard_MC_total = np.array([])
rho_standard_MC_total = np.array([])
for j in range(12):
    angles = j * np.pi / 3 - j % 2 * theta_values + (1 - j % 2) * theta_values
    theta_standard_MC_total = np.concatenate([theta_standard_MC_total, angles])
    rho_standard_MC_total = np.concatenate([rho_standard_MC_total, rho_standard_MC])
ax.plot(theta_standard_MC_total, rho_standard_MC_total, "-", color="black")
ax.set_yticklabels([])

norm = mcolors.Normalize(vmin=0.1, vmax=0.7 * 9)
sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="vertical")
cbar.set_label(r"Magnitude of the stress path deviator, $\rho$ [MPa]")

plt.show()

# %% [markdown]
# Each colour represents one loading path. The circles are associated with the
# loading during the elastic phase. Once the loading reaches the elastic limit,
# the circles start outlining the yield surface, which in the limit lay along the
# standard Mohr-Coulomb one without smoothing (black contour).

# %% [markdown]
# ### Taylor test
#
# Here, we perform a Taylor test to check that the form $F$ and its Jacobian $J$
# are consistent zeroth- and first-order approximations of the residual $F$. In
# particular, the test verifies that the program `dsigma_ddeps_vec`
# obtained by the JAX's AD returns correct values of the external operator
# $\boldsymbol{\sigma}$ and its derivative $\boldsymbol{C}_\text{tang}$, which
# define $F$ and $J$ respectively.
#
# To perform the test, we introduce the
# operators $\mathcal{F}: V \rightarrow V^\prime$ and $\mathcal{J}: V \rightarrow \mathcal{L}(V,
# V^\prime)$ defined as follows:
#
# $$
#     \langle \mathcal{F}(\boldsymbol{u}), \boldsymbol{v} \rangle :=
#     F(\boldsymbol{u}; \boldsymbol{v}), \quad \forall \boldsymbol{v} \in V,
# $$
# $$
#     \langle (\mathcal{J}(\boldsymbol{u}))(k\boldsymbol{\delta u}),
#     \boldsymbol{v} \rangle := J(\boldsymbol{u}; k\boldsymbol{\delta u},
#     \boldsymbol{v}), \quad \forall \boldsymbol{v} \in V,
# $$
#
# where $V^\prime$ is a dual space of $V$, $\langle \cdot, \cdot \rangle$ is the
# $V^\prime \times V$ duality pairing and $\mathcal{L}(V, V^\prime)$ is a space of
# bounded linear operators from $V$ to its dual.
#
# Then, by following the Taylor's theorem on Banach spaces and perturbating the
# functional $\mathcal{F}$ in the direction $k \, \boldsymbol{δu} \in V$ for $k >
# 0$, the zeroth and first order Taylor reminders $r_k^0$ and $r_k^1$ have the
# following *mesh-independent* convergence rates in the dual space $V^\prime$:
#
# $$
#     \| r_k^0 \|_{V^\prime} := \| \mathcal{F}(\boldsymbol{u} + k \,
#     \boldsymbol{\delta u}) - \mathcal{F}(\boldsymbol{u}) \|_{V^\prime}
#     \longrightarrow 0 \text{ at } O(k),
# $$ (eq:r0)
# $$
#     \| r_k^1 \|_{V^\prime} := \| \mathcal{F}(\boldsymbol{u} + k \,
#     \boldsymbol{\delta u}) - \mathcal{F}(\boldsymbol{u}) - \,
#     (\mathcal{J}(\boldsymbol{u}))(k\boldsymbol{\delta u}) \|_{V^\prime}
#     \longrightarrow 0 \text{ at } O(k^2).
# $$ (eq:r1)
#
# In order to compute the norm of an element $f \in V^\prime$ from the dual space
# $V^\prime$, we apply the Riesz representation theorem, which states that there
# is a linear isometric isomorphism $\mathcal{R} : V^\prime \to V$, which
# associates a linear functional $f$ with a unique element $\mathcal{R} f =
# \boldsymbol{u} \in V$. In practice, within a finite subspace $V_h \subset V$, the
# Riesz map $\mathcal{R}$ is represented by the matrix $\mathsf{L}^{-1}$, the
# inverse of the Laplacian operator {cite}`kirbyFunctional2010`
#
# $$
#     \mathsf{L}_{ij} = \int\limits_\Omega \nabla\varphi_i \cdot \nabla\varphi_j \mathrm{d} x , \quad i,j = 1, \dots, n,
# $$
#
# where $\{\varphi_i\}_{i=1}^{\dim V_h}$ is a set of basis function of the space
# $V_h$.
#
# If the Euclidean vectors $\mathsf{r}_k^i \in \mathbb{R}^{\dim V_h}, \, i \in
# \{0,1\}$ represent the Taylor remainders from {eq}`eq:r0`--{eq}`eq:r1` in the
# finite space, then the dual norms are computed through the following formula
# {cite}`kirbyFunctional2010`
#
# $$
#     \| r_k^i \|^2_{V^\prime_h} = (\mathsf{r}_k^i)^T \mathsf{L}^{-1} \mathsf{r}_k^i, \quad i \in \{0,1\}.
# $$ (eq:r_norms)
#
# In practice, the vectors $\mathsf{r}_k^i$ are defined through the residual
# vector $\mathsf{F} \in \mathbb{R}^{\dim V_h}$ and the Jacobian matrix
# $\mathsf{J} \in \mathbb{R}^{\dim V_h\times\dim V_h}$
#
# $$
#     \mathsf{r}_k^0 = \mathsf{F}(\mathsf{u} + k \, \mathsf{\delta u}) - \mathsf{F}(\mathsf{u}) \in \mathbb{R}^n,
# $$ (eq:vec_r0)
# $$
#     \mathsf{r}_k^1 = \mathsf{F}(\mathsf{u} + k \, \mathsf{\delta u}) -
#     \mathsf{F}(\mathsf{u}) - \, \mathsf{J}(\mathsf{u}) \cdot k\mathsf{\delta
#     u} \in \mathbb{R}^n,
# $$ (eq:vec_r1)
#
# where $\mathsf{u} \in \mathbb{R}^{\dim V_h}$ and $\mathsf{\delta u} \in
# \mathbb{R}^{\dim V_h}$ represent dispacement fields $\boldsymbol{u} \in V_h$ and
# $\boldsymbol{\delta u} \in V_h$.
#
# Now we can proceed with the Taylor test implementation. Let us first start with
# defining the Laplace operator.

# %%
L_form = fem.form(ufl.inner(ufl.grad(u_hat), ufl.grad(v)) * ufl.dx)
L = fem.petsc.assemble_matrix(L_form, bcs=bcs)
L.assemble()
Riesz_solver = PETSc.KSP().create(domain.comm)
Riesz_solver.setType("preonly")
Riesz_solver.getPC().setType("lu")
Riesz_solver.setOperators(L)
y = fem.Function(V, name="Riesz_representer_of_r")  # r - a Taylor remainder

# %% [markdown]
# Now we initialize main variables of the plasticity problem.

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
# initial state close to the one with plastic deformations but still remain in the
# elastic phase.

# %%
i = 0
load = 2.0
q.value = load * np.array([0, -gamma])
Du.x.array[:] = 1e-8

if MPI.COMM_WORLD.rank == 0:
    print(f"Load increment #{i}, load: {load}")

problem.solve()

u.x.petsc_vec.axpy(1.0, Du.x.petsc_vec)
u.x.scatter_forward()

sigma_n.x.array[:] = sigma.ref_coefficient.x.array

Du0 = np.copy(Du.x.array)
sigma_n0 = np.copy(sigma_n.x.array)

# %% [markdown]
# If we take into account the initial stress state `sigma_n0` computed in the cell
# above, we perform the Taylor test for the plastic phase, otherwise we stay in
# the elastic one.
#
# Finally, we define the function `perform_Taylor_test`, which returns the norms
# of the Taylor reminders in dual space {eq}`eq:r_norms`--{eq}`eq:vec_r1`.

# %% tags=["scroll-output"]
k_list = np.logspace(-2.0, -6.0, 5)[::-1]


def perform_Taylor_test(Du0, sigma_n0):
    # r0 = F(Du0 + k*δu) - F(Du0)
    # r1 = F(Du0 + k*δu) - F(Du0) - k*J(Du0)*δu
    Du.x.array[:] = Du0
    sigma_n.x.array[:] = sigma_n0
    evaluated_operands = evaluate_operands(F_external_operators)
    ((_, sigma_new),) = evaluate_external_operators(J_external_operators, evaluated_operands)
    sigma.ref_coefficient.x.array[:] = sigma_new

    F0 = fem.petsc.assemble_vector(F_form)  # F(Du0)
    F0.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(F0, bcs)

    J0 = fem.petsc.assemble_matrix(J_form, bcs=bcs)
    J0.assemble()  # J(Du0)
    Ju = J0.createVecLeft()  # Ju = J0 @ u

    δu = fem.Function(V)
    δu.x.array[:] = Du0  # δu == Du0

    zero_order_remainder = np.zeros_like(k_list)
    first_order_remainder = np.zeros_like(k_list)

    for i, k in enumerate(k_list):
        Du.x.array[:] = Du0 + k * δu.x.array
        evaluated_operands = evaluate_operands(F_external_operators)
        ((_, sigma_new),) = evaluate_external_operators(J_external_operators, evaluated_operands)
        sigma.ref_coefficient.x.array[:] = sigma_new

        F_delta = fem.petsc.assemble_vector(F_form)  # F(Du0 + h*δu)
        F_delta.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(F_delta, bcs)

        J0.mult(δu.x.petsc_vec, Ju)  # Ju = J(Du0)*δu
        Ju.scale(k)  # Ju = k*Ju

        r0 = F_delta - F0
        r1 = F_delta - F0 - Ju

        Riesz_solver.solve(r0, y.x.petsc_vec)  # y = L^{-1} r0
        y.x.scatter_forward()
        zero_order_remainder[i] = np.sqrt(r0.dot(y.x.petsc_vec))  # sqrt{r0^T L^{-1} r0}

        Riesz_solver.solve(r1, y.x.petsc_vec)  # y = L^{-1} r1
        y.x.scatter_forward()
        first_order_remainder[i] = np.sqrt(r1.dot(y.x.petsc_vec))  # sqrt{r1^T L^{-1} r1}

    return zero_order_remainder, first_order_remainder


print("Elastic phase")
zero_order_remainder_elastic, first_order_remainder_elastic = perform_Taylor_test(Du0, 0.0)
print("Plastic phase")
zero_order_remainder_plastic, first_order_remainder_plastic = perform_Taylor_test(Du0, sigma_n0)

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].loglog(k_list, zero_order_remainder_elastic, "o-", label=r"$\|r_k^0\|_{V^\prime}$")
axs[0].loglog(k_list, first_order_remainder_elastic, "o-", label=r"$\|r_k^1\|_{V^\prime}$")
annotation.slope_marker((2e-4, 5e-5), 1, ax=axs[0], poly_kwargs={"facecolor": "tab:blue"})
axs[0].text(0.5, -0.2, "(a) Elastic phase", transform=axs[0].transAxes, ha="center", va="top")

axs[1].loglog(k_list, zero_order_remainder_plastic, "o-", label=r"$\|r_k^0\|_{V^\prime}$")
annotation.slope_marker((2e-4, 5e-5), 1, ax=axs[1], poly_kwargs={"facecolor": "tab:blue"})
axs[1].loglog(k_list, first_order_remainder_plastic, "o-", label=r"$\|r_k^1\|_{V^\prime}$")
annotation.slope_marker((2e-4, 5e-13), 2, ax=axs[1], poly_kwargs={"facecolor": "tab:orange"})
axs[1].text(0.5, -0.2, "(b) Plastic phase", transform=axs[1].transAxes, ha="center", va="top")

for i in range(2):
    axs[i].set_xlabel("k")
    axs[i].set_ylabel("Taylor remainder norm")
    axs[i].legend()
    axs[i].grid()

plt.tight_layout()
plt.show()

first_order_rate = np.polyfit(np.log(k_list), np.log(zero_order_remainder_elastic), 1)[0]
second_order_rate = np.polyfit(np.log(k_list), np.log(first_order_remainder_elastic), 1)[0]
print(f"Elastic phase:\n\tthe 1st order rate = {first_order_rate:.2f}\n\tthe 2nd order rate = {second_order_rate:.2f}")
first_order_rate = np.polyfit(np.log(k_list), np.log(zero_order_remainder_plastic), 1)[0]
second_order_rate = np.polyfit(np.log(k_list[1:]), np.log(first_order_remainder_plastic[1:]), 1)[0]
print(f"Plastic phase:\n\tthe 1st order rate = {first_order_rate:.2f}\n\tthe 2nd order rate = {second_order_rate:.2f}")

# %% [markdown]
# For the elastic phase (a) the zeroth-order Taylor remainder $r_k^0$ achieves the
# first-order convergence rate, whereas the first-order remainder $r_k^1$ is
# computed at the level of machine precision due to the constant Jacobian.
# Similarly to the elastic flow, the zeroth-order Taylor remainder $r_k^0$ of the
# plastic phase (b) reaches the first-order convergence, whereas the first-order
# remainder $r_k^1$ achieves the second-order convergence rate, as expected.

# %% [markdown]
# ## References
# ```{bibliography}
# :filter: docname in docnames
# ```
