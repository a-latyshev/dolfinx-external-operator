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
# The current tutorial implements the non-associative plasticity model of
# Mohr-Coulomb with apex-smoothing, where the constitutive relations are defined
# using the external package JAX. Here we consider the same cylinder expansion
# problem in the two-dimensional case in a symmetric formulation, which was
# considered in the previous tutorial on von Mises plasticity.
#
# The tutorial is based on the MFront/TFEL
# [implementation](https://thelfer.github.io/tfel/web/MohrCoulomb.html) of the
# Mohr-Coulomb elastoplastic model with apex smoothing.
#
# ## Problem formulation
#
# We solve the same cylinder expansion problem from the previous tutorial of von
# Mises plasticity and follow the same Mandel-Voigt notation. Thus, we focus here on the constitutive model definition and its implementation.
#
# We consider a non-associative plasticity law without hardening that is defined by the Mohr-Coulomb yield surface $F$ and the plastic potential $G$. Both quantities may be expressed through the following function $H$
#
# \begin{align*}
#     & H(\boldsymbol{\sigma}, \alpha) = \frac{I_1(\boldsymbol{\sigma})}{3}\sin\alpha + \sqrt{J_2(\boldsymbol{\sigma}) K^2(\alpha) + a^2(\alpha)\sin^2\alpha} - c\cos\alpha, \\
#     & F(\boldsymbol{\sigma}) = H(\boldsymbol{\sigma}, \phi), \\
#     & G(\boldsymbol{\sigma}) = H(\boldsymbol{\sigma}, \psi),
# \end{align*}
# where $\phi$ and $\psi$ are friction and dilatancy angles, $c$ is a cohesion, $I_1(\boldsymbol{\sigma}) = \mathrm{tr} \boldsymbol{\sigma}$ is the first invariant of the stress tensor and $J_2(\boldsymbol{\sigma}) = \frac{1}{2}\boldsymbol{s}:\boldsymbol{s}$ is the second invariant of the deviatoric part of the stress tensor. The expression of the coefficient $K(\alpha)$ may be found in the MFront/TFEL
# [implementation](https://thelfer.github.io/tfel/web/MohrCoulomb.html).
#
# During the plastic loading the stress-strain state of the solid must satisfy the following system of nonlinear equations
# $$
#     \begin{cases}
#         \boldsymbol{r}_{G}(\boldsymbol{\sigma}_{n+1}, \Delta\lambda) = \boldsymbol{\sigma}_{n+1} - \boldsymbol{\sigma}_n - \boldsymbol{C}.(\Delta\boldsymbol{\varepsilon} - \Delta\lambda \frac{d G}{d\boldsymbol{\sigma}}(\boldsymbol{\sigma_{n+1}})) = \boldsymbol{0}, \\
#         r_F(\boldsymbol{\sigma}_{n+1}) = F(\boldsymbol{\sigma}_{n+1}) = 0,
#     \end{cases}
# $$ (eq_MC_1)
# where the index $n$ is associated with values from previous loading step.
#
# By introducing the residual vector $\boldsymbol{r} = [\boldsymbol{r}_{G}^T, r_F]^T$ and its argument vector $\boldsymbol{x} = [\sigma_{xx}, \sigma_{yy}, \sigma_{zz}, \sqrt{2}\sigma_{xy}, \Delta\lambda]^T$ we solve the following equation:
# $$
#     \boldsymbol{r}(\boldsymbol{x}_{n+1}) = \boldsymbol{0}
# $$
#
# To solve this system we apply the Newton method and then introduce the Jacobian of the residual vector $\boldsymbol{j} = \frac{\partial \boldsymbol{r}}{\partial \boldsymbol{x}}$
#
# $$
#     \boldsymbol{r}(\boldsymbol{x}_{n+1}) = \boldsymbol{r}(\boldsymbol{x}_{n}) + \boldsymbol{j}(\boldsymbol{x}_{n})(\boldsymbol{x}_{n+1} - \boldsymbol{x}_{n})
# $$
#
# $$
#     \boldsymbol{j}(\boldsymbol{x}_{n})\boldsymbol{y} = - \boldsymbol{r}(\boldsymbol{x}_{n})
# $$
#
# $$
#     \boldsymbol{x}_{n+1} = \boldsymbol{x}_n + \boldsymbol{y}
# $$
#
# During the elastic loading, we consider a trivial system of equations
# $$
#     \begin{cases}
#         \boldsymbol{\sigma}_{n+1} = \boldsymbol{\sigma}_n + \boldsymbol{C}.\Delta\boldsymbol{\varepsilon}, \\
#         \Delta\lambda = 0.
#     \end{cases}
# $$ (eq_MC_2)
#
# The algorithm solving the systems `eq`{eq_MC_1}--`eq`{eq_MC_2} is called the return-mapping procedure and the solution defines the return-mapping correction of the stress tensor. By implementation of the external operator $\boldsymbol{\sigma}$ we mean the implementation of the return-mapping procedure. By applying the automatic differentiation (AD) technique to this algorithm we may restore the stress derivative $\frac{\mathrm{d}\boldsymbol{\sigma}}{\mathrm{d}\boldsymbol{\varepsilon}}$.
#
# The JAX library was used to implement the external operator and its derivative.
#
# ```{note}
# Although the tutorial shows the implementation of the Mohr-Coulomb model, it is quite general to be adapted to a wide rage of plasticity models that may be defined through a yield surface and a plastic potential.
# ```
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
#
# Here we define geometrical and material parameters of the problem as well as some useful constants.

# %%
R_i = 1  # [m] Inner radius
R_e = 21  # [m] Outer radius

E = 6778  # [MPa] Young modulus
nu = 0.25  # [-] Poisson ratio
# sigma_u = 27.6 #[MPa]
P_i_value = 3.45  # [MPa]

c = 3.45  # [MPa] cohesion
phi = 30 * np.pi / 180  # [rad] friction angle
psi = 30 * np.pi / 180  # [rad] dilatancy angle
# [rad] transition angle as defined by Abbo and Sloan
theta_T = 20 * np.pi / 180
a = 0.5 * c / np.tan(phi)  # [MPa] tension cuff-off parameter

lmbda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu)
mu = E / 2.0 / (1.0 + nu)
C_elas = np.array(
    [
        [lmbda + 2.0 * mu, lmbda, lmbda, 0.0],
        [lmbda, lmbda + 2.0 * mu, lmbda, 0.0],
        [lmbda, lmbda, lmbda + 2.0 * mu, 0.0],
        [0.0, 0.0, 0.0, 2.0 * mu],
    ],
    dtype=PETSc.ScalarType,
)

dev = np.array([[2/3., -1/3., -1/3., 0],
                [-1/3., 2/3., -1/3., 0],
                [-1/3., -1/3., 2/3., 0],
                [0, 0, 0, 1.]], dtype=PETSc.ScalarType)

TPV = np.finfo(PETSc.ScalarType).eps  # très petite value
SQRT2 = np.sqrt(2.)
tr = np.array([1, 1, 1, 0])

# %%
mesh, facet_tags, facet_tags_labels = build_cylinder_quarter(R_e=R_e, R_i=R_i)

# %%
k_u = 2
V = fem.functionspace(mesh, ("Lagrange", k_u, (2,)))
# Boundary conditions
bottom_facets = facet_tags.find(facet_tags_labels["Lx"])
left_facets = facet_tags.find(facet_tags_labels["Ly"])

bottom_dofs_y = fem.locate_dofs_topological(
    V.sub(1), mesh.topology.dim-1, bottom_facets)
left_dofs_x = fem.locate_dofs_topological(
    V.sub(0), mesh.topology.dim-1, left_facets)

sym_bottom = fem.dirichletbc(
    np.array(0., dtype=PETSc.ScalarType), bottom_dofs_y, V.sub(1))
sym_left = fem.dirichletbc(
    np.array(0., dtype=PETSc.ScalarType), left_dofs_x, V.sub(0))

bcs = [sym_bottom, sym_left]


def epsilon(v):
    grad_v = ufl.grad(v)
    return ufl.as_vector([grad_v[0, 0], grad_v[1, 1], 0, np.sqrt(2.0) * 0.5 * (grad_v[0, 1] + grad_v[1, 0])])


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


Du = fem.Function(V, name="displacement_increment")
u = fem.Function(V, name="Total_displacement")
du = fem.Function(V, name="du")
v = ufl.TrialFunction(V)
u_ = ufl.TestFunction(V)

sigma = FEMExternalOperator(epsilon(Du), function_space=S)
sigma_n = fem.Function(S, name="sigma_n")


# %% [markdown]
# ### Defining the external operator
#
# In order to define the behaviour of the external operator and its derivatives,
# we need to implement the return-mapping procedure solving the constitutive
# equations `eq`{eq_MC_1}--`eq`{eq_MC_2} and apply the automatic differentiation
# tool to this algorithm.
#
# #### Defining yield surface and plastic potential
#
# First of all, we define supplementary functions that help us to express the
# yield surface $F$ and the plastic potential $G$. In the following definitions,
# we use built-in functions of the JAX package, in particular, the conditional
# primitive `jax.lax.cond`. It is necessary for the correct work of the AD tool
# and just-in-time compilation. For more details, please, visit the JAX
# [documentation](https://jax.readthedocs.io/en/latest/).

# %%
def J3(sigma_local):
    return sigma_local[2] * (sigma_local[0]*sigma_local[1] - sigma_local[3]*sigma_local[3]/2.)


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


def K(theta, angle):
    def K_true(theta, angle): return jnp.cos(theta) - 1 / \
        jnp.sqrt(3) * jnp.sin(angle) * jnp.sin(theta)
    def K_false(theta, angle): return A(theta, angle) + B(theta, angle) * \
        jnp.sin(3*theta) + C(theta, angle) * jnp.sin(3*theta)*jnp.sin(3*theta)
    return K_true(theta, angle)
    return jax.lax.cond(jnp.abs(theta) < theta_T, K_true, K_false, theta, angle)


def a_G(angle):
    return a * jnp.tan(phi) / jnp.tan(angle)


def surface(sigma_local, angle):
    s = dev @ sigma_local
    I1 = tr @ sigma_local
    J2 = 0.5 * jnp.vdot(s, s)
    arg = -3*jnp.sqrt(3) * J3(s) / (2 * jnp.sqrt(J2*J2*J2))
    arg = jnp.clip(arg, -1, 1)
    # arcsin returns nan if its argument is equal to -1 + smth around 1e-16!!!
    theta = 1/3. * jnp.arcsin(arg)
    return I1/3 * jnp.sin(angle) + jnp.sqrt(J2 * K(theta, angle)*K(theta, angle) + a_G(angle)*a_G(angle) * jnp.sin(angle)*jnp.sin(angle)) - c * jnp.cos(angle)


# %% [markdown]
# By picking up an appropriate angle we define the yield surface $F$ and the
# plastic potential $G$.

# %%
def f_MC(sigma_local): return surface(sigma_local, phi)
def g_MC(sigma_local): return surface(sigma_local, psi)


# %%
# NOTE: For testing/remove
@jax.jit
def theta(sigma_local):
    s = dev @ sigma_local
    J2 = 0.5 * jnp.vdot(s, s)
    arg = -3*jnp.sqrt(3) * J3(s) / (2 * jnp.sqrt(J2*J2*J2))
    arg = jnp.clip(arg, -1, 1)
    return 1/3. * jnp.arcsin(arg)


dthetadsigma = jax.jit(jax.jacfwd(theta, argnums=(0)))
dgdsigma = jax.jit(jax.jacfwd(g_MC, argnums=(0)))

deps_local = jnp.array([TPV, 0.0, 0.0, 0.0])
sigma_local = jnp.array([1, 1, 1.0, 1.0])
print(f"f_MC = {f_MC(sigma_local)}, dfdsigma = {sigma_local},\ntheta = {theta(sigma_local)}, dtheta = {dthetadsigma(sigma_local)}")

deps_local = jnp.array([0.0006, 0.0003, 0.0, 0.0])
sigma_local = C_elas @ deps_local
print(f"f_MC = {f_MC(sigma_local)}, dfdsigma = {sigma_local},\ntheta = {theta(sigma_local)}, dtheta = {dthetadsigma(sigma_local)}")


# %% [markdown]
# #### Solving constitutive equations
#
# In this section, we define the constitutive model by solving the following systems
#
# \begin{align*}
#     & \text{Plastic flow:} \\
#     & \begin{cases}
#         \boldsymbol{r}_{G}(\boldsymbol{\sigma}_{n+1}, \Delta\lambda) = \boldsymbol{\sigma}_{n+1} - \boldsymbol{\sigma}_n - \boldsymbol{C}.(\Delta\boldsymbol{\varepsilon} - \Delta\lambda \frac{d G}{d\boldsymbol{\sigma}}(\boldsymbol{\sigma_{n+1}})) = \boldsymbol{0}, \\
#         r_F(\boldsymbol{\sigma}_{n+1}) = F(\boldsymbol{\sigma}_{n+1}) = 0,
#      \end{cases} \\
#     & \text{Elastic flow:} \\
#     &\begin{cases}
#         \boldsymbol{\sigma}_{n+1} = \boldsymbol{\sigma}_n + \boldsymbol{C}.\Delta\boldsymbol{\varepsilon}, \\
#         \Delta\lambda = 0.
#     \end{cases}
# \end{align*}
#
# As the second one is trivial we focus on the first system only and rewrite it in the following form.
# $$
#     \boldsymbol{r}(\boldsymbol{x}_{n+1}) = \boldsymbol{0},
# $$
# where $\boldsymbol{x} = [\sigma_{xx}, \sigma_{yy}, \sigma_{zz}, \sqrt{2}\sigma_{xy}, \Delta\lambda]^T$.
#
# This nonlinear equation must be solved at each Gauss point, so we apply the Newton method, implement the whole algorithm locally and then vectorize the final result using `jax.vmap`.
#
# In the following cell, we define locally the residual $\boldsymbol{r}$ and its jacobian $\boldsymbol{j}$.

# %%
# NOTE: Actually, I put conditionals inside local functions, but we may
# implement two "branches" of the algo separetly and check the yielding condition
# in the main Newton loop. It may be more efficient, but idk. Anyway, as it is,
# it looks fancier.

def deps_p(sigma_local, dlambda, deps_local, sigma_n_local):
    sigma_elas_local = sigma_n_local + C_elas @ deps_local
    yielding = f_MC(sigma_elas_local)
    def deps_p_elastic(sigma_local, dlambda): return jnp.zeros(4)
    def deps_p_plastic(
        sigma_local, dlambda): return dlambda * dgdsigma(sigma_local)
    return jax.lax.cond(yielding <= TPV, deps_p_elastic, deps_p_plastic, sigma_local, dlambda)

def r_sigma(sigma_local, dlambda, deps_local, sigma_n_local):
    deps_p_local = deps_p(sigma_local, dlambda, deps_local, sigma_n_local)
    return sigma_local - sigma_n_local - C_elas @ (deps_local - deps_p_local)

def r_f(sigma_local, dlambda, deps_local, sigma_n_local):
    sigma_elas_local = sigma_n_local + C_elas @ deps_local
    yielding = f_MC(sigma_elas_local)

    def r_f_elastic(sigma_local, dlambda): return dlambda
    def r_f_plastic(sigma_local, dlambda): return f_MC(sigma_local)
    return jax.lax.cond(yielding <= TPV, r_f_elastic, r_f_plastic, sigma_local, dlambda)

dr_sigma = jax.jit(jax.jacfwd(r_sigma, argnums=(0, 1)))
dr_f = jax.jit(jax.jacfwd(r_f, argnums=(0, 1)))

def j(sigma_local, dlambda, deps_local, sigma_n_local):
    dr_sigmadsigma, dr_sigmaddlambda = dr_sigma(
        sigma_local, dlambda, deps_local, sigma_n_local)
    # normally this creates two copies(?) but jit will "eat" them
    dr_sigmaddlambda_T = jnp.atleast_2d(dr_sigmaddlambda).T
    dr_fdsigma, dr_fddlambda = dr_f(
        sigma_local, dlambda, deps_local, sigma_n_local)
    return jnp.block([[dr_sigmadsigma, dr_sigmaddlambda_T],
                     [dr_fdsigma, dr_fddlambda]])

def r(x_local, deps_local, sigma_n_local):
    # Normally, the following lines allocate new memory
    sigma_local = x_local[:4]
    dlambda_local = x_local[-1]
    res_sigma = r_sigma(sigma_local, dlambda_local, deps_local, sigma_n_local)
    res_f = r_f(sigma_local, dlambda_local, deps_local, sigma_n_local)
    # As well as this one
    res = jnp.c_['0,1,-1', res_sigma, res_f]
    return res

# NOTE: Instead of the definition of j above we may use this one
# TODO: Less efficient?
drdx = jax.jacfwd(r, argnums=(0))

# %% [markdown]
# Then we define the function `return_mapping` that implements the return-mapping
# algorithm numerically via the Newton method.

# %%
Nitermax, tol = 200, 1e-8

def sigma_return_mapping(deps_local, sigma_n_local):
    """Performs the return-mapping procedure.

    It solves elastoplastic constitutive equations numerically by applying the
    Newton method in a single Gauss point. The Newton loop is implement via
    `jax.lax.while_loop`.
    """
    niter = 0

    dlambda = jnp.array(0.)  # init guess
    sigma_local = sigma_n_local  # init guess
    x_local = jnp.c_['0,1,-1', sigma_local, dlambda]  # init guess

    # res_sigma = r_sigma(sigma_local, dlambda, deps_local, sigma_n_local)
    # res_f = r_f(sigma_local, dlambda, deps_local, sigma_n_local)
    # res = jnp.c_['0,1,-1', res_sigma, res_f]
    res = r(x_local, deps_local, sigma_n_local)

    norm_res0 = jnp.linalg.norm(res)
    sigma_elas_local = C_elas @ deps_local
    yielding = f_MC(sigma_n_local + sigma_elas_local)

    def cond_fun(state):
        norm_res, niter, _ = state
        return (norm_res/norm_res0 > tol) & (niter < Nitermax)

    def body_fun(state):
        norm_res, niter, history = state
        x_local, deps_local, sigma_n_local, res = history
        # sigma_local, dlambda, deps_local, sigma_n_local, res = history
        # x_local = jnp.c_['0,1,-1', sigma_local, dlambda]
        # J = j(sigma_local, dlambda, deps_local, sigma_n_local)
        J = drdx(x_local, deps_local, sigma_n_local)
        j_inv_vp = jnp.linalg.solve(J, -res)
        # sigma_local = sigma_local + j_inv_vp[:4]
        # dlambda = dlambda + j_inv_vp[-1]
        x_local = x_local + j_inv_vp

        # res_sigma = r_sigma(sigma_local, dlambda, deps_local, sigma_n_local)
        # res_f = r_f(sigma_local, dlambda, deps_local, sigma_n_local)
        # res = jnp.c_['0,1,-1', res_sigma, res_f]
        # x_local = jnp.c_['0,1,-1', sigma_local, dlambda]
        res = r(x_local, deps_local, sigma_n_local)
        norm_res = jnp.linalg.norm(res)
        niter += 1
        history = x_local, deps_local, sigma_n_local, res
        # history = sigma_local, dlambda, deps_local, sigma_n_local, res
        return (norm_res, niter, history)

    history = (x_local, deps_local, sigma_n_local, res)
    # history = (sigma_local, dlambda, deps_local, sigma_n_local, res)

    output = jax.lax.while_loop(
        cond_fun, body_fun, (norm_res0, niter, history))
    niter_total = output[1]
    sigma_local = output[2][0][:4] # or `at` is better?
    norm_res = output[0]

    return sigma_local, (sigma_local, niter_total, yielding, norm_res)


# %% [markdown]
# The `return_mapping` function returns a tuple with two elements. The first element is an array containing values of the external operator $\boldsymbol{\sigma}$ and the second one is another tuple containing additional data such as e.g. information on a convergence of the Newton method. Once we apply the JAX AD tool, the latter "converts" the first element of the `return_mapping` output into an array with values of the derivative $\frac{\mathrm{d}\boldsymbol{\sigma}}{\mathrm{d}\boldsymbol{\varepsilon}}$ and leaves untouched the second one. That is why we return `sigma_local` twice in the `return_mapping`: ....
#
# COMMENT: Well, looks too wordy...

# %%
dsigma_ddeps = jax.jacfwd(sigma_return_mapping, argnums=(0,), has_aux=True)

# NOTE: If we implemented the function `dsigma_ddeps` manually, it would return
# `C_tang_local, (sigma_local, niter_total, yielding, res)`

# %% [markdown]
# Once we defined the function `dsigma_ddeps`, which evaluates both the external operator and its derivative locally, we can just vectorize it and define the final implementation of the external operator derivative.

# %%
dsigma_ddeps_vec = jax.jit(jax.vmap(dsigma_ddeps, in_axes=(0, 0)))


def C_tang_impl(deps):
    deps_ = deps.reshape((-1, 4))
    sigma_n_ = sigma_n.x.array.reshape((-1, 4))

    output = dsigma_ddeps_vec(deps_, sigma_n_)

    C_tang_global = output[0][0]
    sigma_global = output[1][0]
    niter = output[1][1]
    yielding = output[1][2]
    norm_res = output[1][3]

    unique_iters, counts = jnp.unique(niter, return_counts=True)

    # NOTE: The following code prints some details about the second Newton
    # solver, solving the constitutive equations. Do we need this or it's better
    # to have the code as clean as possible?
    print("\tSubNewton:")
    print(f"\t  max F = {jnp.max(yielding)}")
    print(f"\t  {unique_iters} - unique number of iterations")
    print(f"\t  {counts} - counts of unique number of iterations")
    print(f"\t  max SubResidual = {jnp.max(norm_res)}")
    # print(f"\t  nans = {jnp.argwhere(jnp.isnan(res))}")


    return C_tang_global.reshape(-1), sigma_global.reshape(-1)


# %% [markdown]
# Similarly to the von Mises example, we do not implement explicitly the
# evaluation of the external operator. Instead, we obtain its values during the
# evaluation of its derivative and then update the values of the operator in the
# main Newton loop.

# %%
def sigma_external(derivatives):
    if derivatives == (0,):
        return NotImplementedError
    elif derivatives == (1,):
        return C_tang_impl
    else:
        return NotImplementedError


sigma.external_function = sigma_external

# %% [markdown]
# ### Defining the forms

# %%
n = ufl.FacetNormal(mesh)
P_o = fem.Constant(mesh, PETSc.ScalarType(0.0))
P_i = fem.Constant(mesh, PETSc.ScalarType(0.0))


def F_ext(v):
    return -P_i * ufl.inner(n, v)*ds(facet_tags_labels["inner"]) + P_o * ufl.inner(n, v)*ds(facet_tags_labels["outer"])


u_hat = ufl.TrialFunction(V)
F = ufl.inner(epsilon(u_), sigma)*dx - F_ext(u_)
J = ufl.derivative(F, Du, u_hat)
J_expanded = ufl.algorithms.expand_derivatives(J)

F_replaced, F_external_operators = replace_external_operators(F)
J_replaced, J_external_operators = replace_external_operators(J_expanded)

F_form = fem.form(F_replaced)
J_form = fem.form(J_replaced)

# %% [markdown]
# ### Variables initialization and compilation
# Before solving the problem it is required.
# %%
# Initialize variables to start the algorithm
# NOTE: Actually we need to evaluate operators before the Newton solver
# in order to assemble the matrix, where we expect elastic stiffness matrix
# Shell we discuss it? The same states for the von Mises.
Du.x.array[:] = 1.0  # still the elastic flow
# sigma_n.x.array[:] = TPV

timer1 = common.Timer("1st JAX pass")
timer1.start()

evaluated_operands = evaluate_operands(F_external_operators)
((_, _),) = evaluate_external_operators(
    J_external_operators, evaluated_operands)

timer1.stop()

timer2 = common.Timer("2nd JAX pass")
timer2.start()

evaluated_operands = evaluate_operands(F_external_operators)
((_, _),) = evaluate_external_operators(
    J_external_operators, evaluated_operands)

timer2.stop()

timer3 = common.Timer("3rd JAX pass")
timer3.start()

evaluated_operands = evaluate_operands(F_external_operators)
((_, _),) = evaluate_external_operators(
    J_external_operators, evaluated_operands)

timer3.stop()

# %%
# TODO: Is there a more elegant way to extract the data?
# TODO: Maybe we analyze the compilation time in-place?
common.list_timings(MPI.COMM_WORLD, [common.TimingType.wall])


# %% [markdown]
# ### Solving the problem
#
# Summing up, we apply the Newton method to solve the main weak problem. On each
# iteration of the main Newton loop, we solve elastoplastic constitutive equations
# by using the second Newton method at each Gauss point. Thanks to the framework
# and the JAX library, the final interface is general enough to be reused for
# other plasticity models.

# %%
external_operator_problem = LinearProblem(J_replaced, -F_replaced, Du, bcs=bcs)

# %%
# Defining a cell containing (Ri, 0) point, where we calculate a value of u
# It is required to run this program via MPI in order to capture the process, to which this point is attached
x_point = np.array([[R_i, 0, 0]])
cells, points_on_process = find_cell_by_point(mesh, x_point)

# parameters of the manual Newton method
max_iterations, relative_tolerance = 200, 1e-8
num_increments = 20
load_steps = np.linspace(0.9, 5, num_increments, endpoint=True)[1:]
results = np.zeros((num_increments, 2))

for (i, load) in enumerate(load_steps):
    P_i.value = load
    external_operator_problem.assemble_vector()

    residual_0 = external_operator_problem.b.norm()
    residual = residual_0
    Du.x.array[:] = 0

    if MPI.COMM_WORLD.rank == 0:
        print(
            f"Increment: {i+1!s}, load = {load}, residual0 = {residual_0} ")
    niter = 0

    while residual / residual_0 > relative_tolerance and niter < max_iterations:
        if MPI.COMM_WORLD.rank == 0:
            print(f"\tit# {niter}:")
        external_operator_problem.assemble_matrix()
        external_operator_problem.solve(du)

        Du.vector.axpy(1, du.vector)  # Du = Du + 1*du
        Du.x.scatter_forward()

        evaluated_operands = evaluate_operands(F_external_operators)
        ((_, sigma_new),) = evaluate_external_operators(
            J_external_operators, evaluated_operands)
        sigma.ref_coefficient.x.array[:] = sigma_new

        external_operator_problem.assemble_vector()
        residual = external_operator_problem.b.norm()

        if MPI.COMM_WORLD.rank == 0:
            print(f"\tresidual: {residual}\n")
        niter += 1
    u.vector.axpy(1, Du.vector)  # u = u + 1*Du
    u.x.scatter_forward()

    sigma_n.x.array[:] = sigma.ref_coefficient.x.array

    if len(points_on_process) > 0:
        results[i+1, :] = (u.eval(points_on_process, cells)[0], load)

# %%

# %% [markdown]
# ### Post-processing

# %%
if len(points_on_process) > 0:
    plt.plot(results[:, 0], results[:, 1], "-o", label="via ExternalOperator")
    plt.xlabel("Displacement of inner boundary")
    plt.ylabel(r"Applied pressure $q/q_{lim}$")
    plt.savefig(f"displacement_rank{MPI.COMM_WORLD.rank:d}.png")
    plt.legend()
    plt.show()

# %%
# TODO: Is there a more elegant way to extract the data?
# common.list_timings(MPI.COMM_WORLD, [common.TimingType.wall])

# %%
# # NOTE: There is the warning `[WARNING] yaksa: N leaked handle pool objects` for
# # the call `.assemble_vector()` and `.vector`.
# # NOTE: The following lines eleminate the leakes (except the mesh ones).
# # NOTE: To test this for the newest version of the DOLFINx.
external_operator_problem.__del__()
Du.vector.destroy()
du.vector.destroy()
u.vector.destroy()
