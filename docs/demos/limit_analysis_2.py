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
# # Limit analysis as a standard plasticity problem

# %%
from mpi4py import MPI
from petsc4py import PETSc

import jax
import jax.lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from solvers import LinearProblem
from utilities import find_cell_by_point

import basix
import ufl
from dolfinx import common, fem, mesh, default_scalar_type
from dolfinx_external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    evaluate_operands,
    replace_external_operators,
)

jax.config.update("jax_enable_x64", True)  # replace by JAX_ENABLE_X64=True

# %% [markdown]
# ### Model parameters
#
# Here we define geometrical and material parameters of the problem as well as
# some useful constants.

# %%
E = 6778  # [MPa] Young modulus
nu = 0.25  # [-] Poisson ratio
lambda_ = E*nu/(1+nu)/(1-2*nu)
mu_ = E/2./(1+nu)
# sig0 = 250.  # yield strength
Et = E/100.  # tangent modulus
H = E*Et/(E-Et)  # hardening modulus
TPV = np.finfo(PETSc.ScalarType).eps # très petite value
# SQRT2 = np.sqrt(2.)

c = 3.45 #[MPa]
phi = 30 * np.pi / 180
# psi = 30 * np.pi / 180
# theta_T = 20 * np.pi / 180
# a = 0.5 * c / np.tan(phi)

l, m = lambda_, mu_
C_elas = np.array([[l+2*m, l, l, 0, 0, 0],
                    [l, l+2*m, l, 0, 0, 0],
                    [l, l, l+2*m, 0, 0, 0],
                    [0, 0, 0, 2*m, 0, 0],
                    [0, 0, 0, 0, 2*m, 0],
                    [0, 0, 0, 0, 0, 2*m],
                    ], dtype=PETSc.ScalarType)

# dev = np.array([[2/3., -1/3., -1/3., 0],
#                  [-1/3., 2/3., -1/3., 0],
#                  [-1/3., -1/3., 2/3., 0],
#                  [0, 0, 0, 1., 0., 0.],
#                  [0, 0, 0, 0., 1., 0.],
#                  [0, 0, 0, 0., 0., 1.]
#                  ], dtype=PETSc.ScalarType)

# tr = np.array([1, 1, 1, 0])

Nitermax, tol = 200, 1e-8

# %%
L = W = H = 1.
gamma = 1.
N = 10
domain = mesh.create_box(MPI.COMM_WORLD, [np.array([0,0,0]), np.array([L, W, H])], [N, N, N])

# %%
k_u = 2
V = fem.functionspace(domain, ("Lagrange", k_u, (3,)))
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
    # return ufl.sym(ufl.grad(u))
    grad_v = ufl.grad(v)
    return ufl.as_vector([
        grad_v[0, 0], grad_v[1, 1], grad_v[2, 2],
        np.sqrt(2.0) * 0.5 * (grad_v[1, 2] + grad_v[2, 1]),
        np.sqrt(2.0) * 0.5 * (grad_v[0, 2] + grad_v[2, 0]),
        np.sqrt(2.0) * 0.5 * (grad_v[0, 1] + grad_v[1, 0]),
    ])


k_stress = 2 * (k_u - 1)
S_element = basix.ufl.quadrature_element(domain.topology.cell_name(), degree=k_stress, value_shape=(6,))
S = fem.functionspace(domain, S_element)
dx = ufl.Measure(
    "dx",
    domain=domain,
    metadata={"quadrature_degree": k_stress, "quadrature_scheme": "default"},
)

# %%
Du = fem.Function(V, name="displacement_increment")
u = fem.Function(V, name="Total_displacement")
du = fem.Function(V, name="du")
# v = ufl.TrialFunction(V)
u_ = ufl.TestFunction(V)

sigma = FEMExternalOperator(epsilon(Du), function_space=S)
sigma_n = fem.Function(S, name="sigma_n")

# %% [markdown]
# ### Defining the external operator
#
# In order to define the behaviour of the external operator and its
# derivatives, we need to implement the return-mapping procedure solving the
# constitutive equations {eq}`eq_MC_1`--{eq}`eq_MC_2` and apply the automatic
# differentiation tool to this algorithm.
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
# tr = jnp.array([1.0, 1.0, 1.0, 0.0], dtype=PETSc.ScalarType)
def principal_stresses(sigma_local):
    sym_matrix = jnp.array([
        [sigma_local[0], np.sqrt(2) * sigma_local[5], sigma_local[4] / np.sqrt(2)],
        [sigma_local[5] / np.sqrt(2), sigma_local[1], sigma_local[3] / np.sqrt(2)],
        [sigma_local[4] / np.sqrt(2), sigma_local[3] / np.sqrt(2), sigma_local[2]]
    ])
    eigvals = jnp.linalg.eigvalsh(sym_matrix)
    sigma_M = jnp.max(eigvals)
    sigma_m = jnp.min(eigvals)
    return sigma_M, sigma_m

a = (1 - np.sin(phi)) / (1 + np.sin(phi))

def G(sigma_local):
    sigma_M, sigma_m = principal_stresses(sigma_local)
    return sigma_M - a * sigma_m - 2*c * np.cos(phi) / (1 + np.sin(phi))

dGdsigma = jax.jacfwd(G)

# %% [markdown]
# By picking up an appropriate angle we define the yield surface $F$ and the
# plastic potential $G$.

# %% [markdown]
# #### Solving constitutive equations
#
# In this section, we define the constitutive model by solving the following
# systems
#
# \begin{align*}
#     & \text{Plastic flow:} \\
#     & \begin{cases}
#         \boldsymbol{r}_{G}(\boldsymbol{\sigma}_{n+1}, \Delta\lambda) =
#         \boldsymbol{\sigma}_{n+1} - \boldsymbol{\sigma}_n -
#         \boldsymbol{C}.(\Delta\boldsymbol{\varepsilon} - \Delta\lambda
#         \frac{d G}{d\boldsymbol{\sigma}}(\boldsymbol{\sigma_{n+1}})) =
#         \boldsymbol{0}, \\
#         r_F(\boldsymbol{\sigma}_{n+1}) = F(\boldsymbol{\sigma}_{n+1}) = 0,
#      \end{cases} \\
#     & \text{Elastic flow:} \\
#     &\begin{cases}
#         \boldsymbol{\sigma}_{n+1} = \boldsymbol{\sigma}_n +
#         \boldsymbol{C}.\Delta\boldsymbol{\varepsilon}, \\ \Delta\lambda = 0.
#     \end{cases}
# \end{align*}
#
# As the second one is trivial we focus on the first system only and rewrite it
# in the following form.
#
# $$
#     \boldsymbol{r}(\boldsymbol{x}_{n+1}) = \boldsymbol{0},
# $$
#
# where $\boldsymbol{x} = [\sigma_{xx}, \sigma_{yy}, \sigma_{zz},
# \sqrt{2}\sigma_{xy}, \Delta\lambda]^T$.
#
# This nonlinear equation must be solved at each Gauss point, so we apply the
# Newton method, implement the whole algorithm locally and then vectorize the
# final result using `jax.vmap`.
#
# In the following cell, we define locally the residual $\boldsymbol{r}$ and
# its jacobian $\boldsymbol{j}$.

# %%
# NOTE: Actually, I put conditionals inside local functions, but we may
# implement two "branches" of the algo separetly and check the yielding
# condition in the main Newton loop. It may be more efficient, but idk. Anyway,
# as it is, it looks fancier.

ZERO_VEC = jnp.zeros(6, dtype=PETSc.ScalarType)

def deps_p(sigma_local, dlambda, deps_local, sigma_n_local):
    sigma_elas_local = sigma_n_local + C_elas @ deps_local
    yielding = G(sigma_elas_local)

    def deps_p_elastic(sigma_local, dlambda):
        return ZERO_VEC

    def deps_p_plastic(sigma_local, dlambda):
        return dlambda * dGdsigma(sigma_local)

    return jax.lax.cond(yielding <= 0.0, deps_p_elastic, deps_p_plastic, sigma_local, dlambda)


def r_sigma(sigma_local, dlambda, deps_local, sigma_n_local):
    deps_p_local = deps_p(sigma_local, dlambda, deps_local, sigma_n_local)
    return sigma_local - sigma_n_local - C_elas @ (deps_local - deps_p_local)


def r_f(sigma_local, dlambda, deps_local, sigma_n_local):
    sigma_elas_local = sigma_n_local + C_elas @ deps_local
    yielding = G(sigma_elas_local)

    def r_f_elastic(sigma_local, dlambda):
        return dlambda

    def r_f_plastic(sigma_local, dlambda):
        return G(sigma_local)
    return jax.lax.cond(yielding <= 0.0, r_f_elastic, r_f_plastic, sigma_local, dlambda)


def r(x_local, deps_local, sigma_n_local):
    sigma_local = x_local[:4]
    dlambda_local = x_local[-1]

    res_sigma = r_sigma(sigma_local, dlambda_local, deps_local, sigma_n_local)
    res_f = r_f(sigma_local, dlambda_local, deps_local, sigma_n_local)

    res = jnp.c_["0,1,-1", res_sigma, res_f]
    return res


drdx = jax.jacfwd(r)

# %% [markdown]
# Then we define the function `return_mapping` that implements the
# return-mapping algorithm numerically via the Newton method.

# %%
Nitermax, tol = 200, 1e-8


# JSH: You need to explain somewhere here how the while_loop interacts with
# vmap.
def sigma_return_mapping(deps_local, sigma_n_local):
    """Performs the return-mapping procedure.

    It solves elastoplastic constitutive equations numerically by applying the
    Newton method in a single Gauss point. The Newton loop is implement via
    `jax.lax.while_loop`.
    """
    niter = 0

    dlambda = jnp.array([0.0])
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

        J = drdx(x_local, deps_local, sigma_n_local)
        j_inv_vp = jnp.linalg.solve(J, -res)
        x_local = x_local + j_inv_vp

        res = r(x_local, deps_local, sigma_n_local)
        norm_res = jnp.linalg.norm(res)
        history = x_local, deps_local, sigma_n_local, res

        niter += 1

        return (norm_res, niter, history)

    history = (x_local, deps_local, sigma_n_local, res)

    norm_res, niter_total, x_local = jax.lax.while_loop(cond_fun, body_fun, (norm_res0, niter, history))

    sigma_local = x_local[0][:4]
    sigma_elas_local = C_elas @ deps_local
    yielding = G(sigma_n_local + sigma_elas_local)

    return sigma_local, (sigma_local, niter_total, yielding, norm_res)
    # return sigma_local, (sigma_local,)

# %% [markdown]
# The `return_mapping` function returns a tuple with two elements. The first
# element is an array containing values of the external operator
# $\boldsymbol{\sigma}$ and the second one is another tuple containing
# additional data such as e.g. information on a convergence of the Newton
# method. Once we apply the JAX AD tool, the latter "converts" the first
# element of the `return_mapping` output into an array with values of the
# derivative
# $\frac{\mathrm{d}\boldsymbol{\sigma}}{\mathrm{d}\boldsymbol{\varepsilon}}$
# and leaves untouched the second one. That is why we return `sigma_local`
# twice in the `return_mapping`: ....
#
# COMMENT: Well, looks too wordy...
# JSH eg.
# `jax.jacfwd` returns a callable that returns the Jacobian as its first return
# argument. As we also need sigma_local, we also return sigma_local as
# auxilliary data.
#
#
# NOTE: If we implemented the function `dsigma_ddeps` manually, it would return
# `C_tang_local, (sigma_local, niter_total, yielding, norm_res)`

# %% [markdown]
# Once we defined the function `dsigma_ddeps`, which evaluates both the
# external operator and its derivative locally, we can just vectorize it and
# define the final implementation of the external operator derivative.

# %%
dsigma_ddeps = jax.jacfwd(sigma_return_mapping, has_aux=True)
dsigma_ddeps_vec = jax.jit(jax.vmap(dsigma_ddeps, in_axes=(0, 0)))


def C_tang_impl(deps):
    deps_ = deps.reshape((-1, 4))
    sigma_n_ = sigma_n.x.array.reshape((-1, 4))

    (C_tang_global, state) = dsigma_ddeps_vec(deps_, sigma_n_)
    sigma_global, niter, yielding, norm_res = state

    unique_iters, counts = jnp.unique(niter, return_counts=True)

    # NOTE: The following code prints some details about the second Newton
    # solver, solving the constitutive equations. Do we need this or it's better
    # to have the code as clean as possible?

    print("\tInner Newton iteration summary")
    print(f"\t\tUnique number of iterations: {unique_iters}")
    print(f"\t\tCounts of unique number of iterations: {counts}")
    print(f"\t\tMaximum F: {jnp.max(yielding)}")
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
f = fem.Constant(domain, default_scalar_type((0, 0, -gamma)))

def F_ext(v):
    return ufl.dot(f, v) * dx

u_hat = ufl.TrialFunction(V)
F = ufl.inner(epsilon(u_), sigma) * dx
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

timer1 = common.Timer("1st JAX pass")
timer1.start()

evaluated_operands = evaluate_operands(F_external_operators)
((_, _),) = evaluate_external_operators(J_external_operators, evaluated_operands)

timer1.stop()

timer2 = common.Timer("2nd JAX pass")
timer2.start()

evaluated_operands = evaluate_operands(F_external_operators)
((_, _),) = evaluate_external_operators(J_external_operators, evaluated_operands)

timer2.stop()

timer3 = common.Timer("3rd JAX pass")
timer3.start()

evaluated_operands = evaluate_operands(F_external_operators)
((_, _),) = evaluate_external_operators(J_external_operators, evaluated_operands)

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
# Defining a cell containing (Ri, 0) point, where we calculate a value of u It
# is required to run this program via MPI in order to capture the process, to
# which this point is attached
# x_point = np.array([[R_i, 0, 0]])
# cells, points_on_process = find_cell_by_point(mesh, x_point)

# %%
# parameters of the manual Newton method
max_iterations, relative_tolerance = 200, 1e-8
num_increments = 20
load_steps = np.linspace(0.9, 5, num_increments, endpoint=True)[1:]
results = np.zeros((num_increments, 2))

for i, load in enumerate(load_steps):
    f.value = load * np.array([0, 0, -gamma])
    external_operator_problem.assemble_vector()

    residual_0 = external_operator_problem.b.norm()
    residual = residual_0
    Du.x.array[:] = 0

    if MPI.COMM_WORLD.rank == 0:
        print(f"Load increment: {i}, load: {load}, initial residual: {residual_0}")

    for iteration in range(0, max_iterations):
        if residual / residual_0 < relative_tolerance:
            break

        if MPI.COMM_WORLD.rank == 0:
            print(f"\tOuter Newton iteration {iteration}")
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

    u.vector.axpy(1.0, Du.vector)
    u.x.scatter_forward()

    sigma_n.x.array[:] = sigma.ref_coefficient.x.array

    # if len(points_on_process) > 0:
    #     results[i + 1, :] = (u.eval(points_on_process, cells)[0], load)

# %% [markdown]
# ### Post-processing

# %%
# if len(points_on_process) > 0:
#     plt.plot(results[:, 0], results[:, 1], "-o", label="via ExternalOperator")
#     plt.xlabel("Displacement of inner boundary")
#     plt.ylabel(r"Applied pressure $q/q_{lim}$")
#     plt.savefig(f"displacement_rank{MPI.COMM_WORLD.rank:d}.png")
#     plt.legend()
#     plt.show()

# %%
# TODO: Is there a more elegant way to extract the data?
# common.list_timings(MPI.COMM_WORLD, [common.TimingType.wall])

# %%
# # NOTE: There is the warning `[WARNING] yaksa: N leaked handle pool objects`
# for # the call `.assemble_vector()` and `.vector`. # NOTE: The following
# lines eleminate the leakes (except the mesh ones). # NOTE: To test this for
# the newest version of the DOLFINx.
external_operator_problem.__del__()
Du.vector.destroy()
du.vector.destroy()
u.vector.destroy()



# %%
# #### Some tests
# def f_MC_to_plot(sigma_I, sigma_II, sigma_III):
#     sigma_local = jnp.array([sigma_I, sigma_II, sigma_III, 0])
#     return f_MC(sigma_local)

# f_MC_to_plot_vec = jax.vmap(f_MC_to_plot)
# sigma_I = sigma_II = np.array([1., 2.0])
# sigma_III = np.array([0.0, 0.0])
# f_MC_to_plot_vec(sigma_I, sigma_II, sigma_III)
# sigma_I = sigma_II = np.arange(1.0, 3.0, 0.05)
# X, Y = np.meshgrid(sigma_I, sigma_II)
# sigma_III = np.zeros_like(X)
# Z = f_MC_to_plot_vec(X.reshape(-1), Y.reshape(-1), sigma_III.reshape(-1)).reshape(X.shape)
# # %matplotlib widget
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.set_zlim(-1, 1)
# ax.plot_surface(X, Y, Z)
# sigma_local = jnp.array([0.00001, 0.00001, 0.0, 0.0])
# f_MC(sigma_local)
# def von_mises(sigma_I, sigma_II, sigma_III):
#     sigma_local = jnp.array([sigma_I, sigma_II, sigma_III, 0])
#     dev = jnp.array(
#         [
#             [2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 0.0],
#             [-1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0, 0.0],
#             [-1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0, 0.0],
#             [0.0, 0.0, 0.0, 1.0],
#         ],
#         dtype=PETSc.ScalarType,
#     )
#     s = dev @ sigma_local

#     return jnp.sqrt(3*0.5 * jnp.vdot(s, s))

# von_mises_vec = jax.vmap(von_mises)
# one = np.ones(100)
# R = 5.
# u = np.linspace(0, np.pi/2, 100)
# x = R * np.outer(np.cos(u), one)
# y = R * np.outer(np.sin(u), one)
# z = np.outer(np.linspace(0.5, 10, 100), one)
# R = 5.
# height = 10
# resolution = 100
# theta = np.linspace(0, 2*np.pi, resolution)
# z = np.linspace(0.1, height, resolution)
# theta, z = np.meshgrid(theta, z)
# x = R * np.cos(theta)
# y = R * np.sin(theta)
# # x = R * z/height * np.cos(theta)
# # y = R * z/height * np.sin(theta)
# F = f_MC_to_plot_vec(x.reshape(-1), y.reshape(-1), z.reshape(-1)).reshape(x.shape)
# F = von_mises_vec(x.reshape(-1), y.reshape(-1), z.reshape(-1)).reshape(x.shape)
# import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x, y, F)

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Generate data points for the von Mises yield surface
# theta = np.linspace(0, 2 * np.pi, 100)
# phi = np.linspace(0, np.pi, 100)
# theta, phi = np.meshgrid(theta, phi)
# sigma_1 = np.sin(theta) * np.cos(phi)
# sigma_2 = np.sin(theta) * np.sin(phi)
# sigma_3 = np.cos(theta)
# von_mises = np.sqrt(sigma_1**2 + sigma_2**2 + sigma_3**2 - sigma_1*sigma_2 - sigma_2*sigma_3 - sigma_3*sigma_1)

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the von Mises yield surface
# ax.plot_surface(sigma_1, sigma_2, sigma_3, cmap='viridis', edgecolor='none')

# # Set labels and title
# ax.set_xlabel('Sigma 1')
# ax.set_ylabel('Sigma 2')
# ax.set_zlabel('Sigma 3')
# ax.set_title('Von Mises Yield Surface')

# # Show the plot
# plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Parameters for the cone
# radius = 1
# height = 2
# resolution = 100

# # Generate data points for the surface of the cone
# theta = np.linspace(0, 2*np.pi, resolution)
# z = np.linspace(0, height, resolution)
# theta, z = np.meshgrid(theta, z)
# x = radius * (1 - z/height) * np.cos(theta)
# y = radius * (1 - z/height) * np.sin(theta)

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the surface of the cone
# ax.plot_surface(x, y, z, color='b', alpha=0.5)

# # Set labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Yield Surface: Cone')

# # Show the plot
# plt.show()
# z.shape
# u = np.linspace(0, 2 * np.pi, 100)
# v = np.linspace(0, np.pi, 100)
# v = np.full((100), np.pi/2)
# x = np.outer(np.cos(u), np.sin(v))
# x

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Generate data points for a sphere
# u = np.linspace(0, 2 * np.pi, 100)
# v = np.linspace(0, np.pi, 100)
# x = np.outer(np.cos(u), np.sin(v))
# y = np.outer(np.sin(u), np.sin(v))
# z = np.outer(np.ones(np.size(u)), np.cos(v))

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the sphere
# ax.plot_surface(x, y, z, color='b', alpha=0.5)

# # Set labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Yield Surface')

# # Show the plot
# plt.show()
