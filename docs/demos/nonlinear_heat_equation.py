# %% [markdown]
# # Nonlinear heat equation
#
# Authors: Andrey Latyshev (University of Luxembourg, Sorbonne Université, andrey.latyshev@uni.lu)
#
# In this notebook, we implement a numerical solution of a nonlinear steady-state heat equation using an external operator. Here we focus on the application of our framework to the problem, where the external operator has two operands. In addition, we leverage the flexibility of the framework to define the behaviour of the external operator using different 3rd-party libraries (here we use Numba and JAX). We strongly recommend taking a look at the simple example first in order to become familiar with the basic workflow of the application of external operators in FEniCSx.
#
# ## Problem formulation
#
# Denoting the temperature field through $T$ we consider the following system on the square domain $\Omega$:
#
# \begin{align*}
#     \Omega : \quad & \nabla \cdot (K(T) \nabla T) = 0 \\
#     \partial\Omega : \quad & T = 0
# \end{align*}
#
# where $K(T) = \frac{1}{A + BT}$ is a nonlinear thermal conductivity, $A$ and $B$ are some constants.
#
# Let $V = H^1_0(\Omega)$ be the functional space of admissible temperature fields then in a variational setting the problem can be written as follows.
#
# Find $T \in V$ such that
#
# $$
#     F(\boldsymbol{j}; \tilde{T}) = -\int\frac{1}{A + BT}\nabla T . \nabla\tilde{T} dx = \int\boldsymbol{j}(T,\boldsymbol{\sigma}(T)) . \nabla\tilde{T} dx = 0, \quad \forall T \in V,
# $$ (eqn:1)
#
# where $\boldsymbol{j} = -\frac{1}{A + BT}\nabla T = - K(T) \boldsymbol{\sigma}(T)$ is a nonlinear heat flux and through $\boldsymbol{\sigma}$ we denoted the gradient of the temperature field $\nabla T$.
#
# In order to solve the nonlinear equation {eq}`eqn:1` we apply the Newton method and calculate the Gateau derivative of the functional $F$ with respect to operand $T$ in the direction $\hat{T} \in V$ as follows:
#
# $$
#     J(\boldsymbol{j};\hat{T},\tilde{T}) = \frac{d F}{d T}(\boldsymbol{j}(T,\boldsymbol{\sigma}(T));\hat{T}, \tilde{T}) = \int\frac{d\boldsymbol{j}}{dT}(T,\boldsymbol{\sigma}(T);\hat{T}) \nabla\tilde{T} dx,
# $$
#
# where through $d \cdot / dT$ we denote the Gateau derivative.
#
# In this example, we treat the heat flux $\boldsymbol{j}$ as an external operator with two operands $T$ and $\boldsymbol{\sigma}(T) = \nabla T$. In this regard, by applying the chain rule, let us write out the explicit expression of the Gateau derivative of $\boldsymbol{j}$ here below
#
# $$
#     \frac{d\boldsymbol{j}}{dT}(T,\boldsymbol{\sigma}(T);\hat{T}) = \frac{\partial\boldsymbol{j}}{\partial T} + \frac{\partial\boldsymbol{j}}{\partial\boldsymbol{\sigma}}\frac{\partial\boldsymbol{\sigma}}{\partial T} = BK^2(T)\boldsymbol{\sigma}(T)\hat{T} - K(T)\mathbb{I}:\nabla\hat{T},
# $$
# where $\mathbb{I}$ is a second-order identity tensor.
#
# According to the current version of the framework operands of an external operator may be any UFL expression. It is worth noting that derivatives of these expressions appear as terms of the full Gateaux derivative (as per the chain rule) and are computed by UFL. Consequently, the user must define evaluation only "partial derivatives" of the external operator and leave the operand differentiation to UFL. Thus, in our example by the evaluation of the external operator $\frac{\partial\boldsymbol{j}}{\partial\boldsymbol{\sigma}}$ we mean the computation of the expression $-K(T)\mathbb{I}$. The term $\nabla\hat{T}$ is derived automatically by the AD tool of UFL and will be natively incorporated into the bilinear form $J$ after application of the `replace_external_operators` function. The same rule applies to the "first" partial derivative $\frac{\partial\boldsymbol{j}}{\partial T}$. We evaluate it as following the expression $BK^2(T)\boldsymbol{\sigma}(T)$ without the term $\hat{T}$.
#
# TODO: Rewrite? Discuss this part!
#
# ```{note}
# In general, the same function can be presented in numerous variations by selecting different operands as sub-expressions of this function. In our case, for example, we could have presented the heat flux $\boldsymbol{j}$ as a function of $K(T)$ and $\sigma(T)$ operands, but this decision would have led to more midterms due to the chain rule and therefore to more computation costs. Thus, it is important to choose wisely the operands of the external operators, which you want to use.
# ```
#
# Despite the function can be explicitly expressed via UFL, we are going to define the heat flux through `femExternalOperator` object and calls of an external function.
#
# In order to start the numerical algorithm we initialize variable `T` with the following initial guess:
#
# $$
#     T(\boldsymbol{x}) = x + 2y,
# $$
# where  $\bm{x} = (x, y)^T$ is the space variable.
#
# ## Defining the external operator
#
# FORTHEARTICLE: the framework takes care of the operands differentiation. (completely forgot to cover this!!!)

# %% [markdown]
# ## Preamble
#
# Importing required packages.
#

# %%
import solvers
import external_operator as ex_op_env
import sys
from mpi4py import MPI
from petsc4py import PETSc

import basix
import ufl
from dolfinx import fem, mesh, common
import dolfinx.fem.petsc  # there is an error without it, why?

import numpy as np
import numba
import jax
# import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

sys.path.append("../../src/dolfinx_ExternalOperator")

# %% [markdown]
# Here we build the mesh, construct the finite functional space and define main variables and zero boundary conditions.
#

# %%
nx = 5
domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, nx)
gdim = domain.geometry.dim
V = fem.functionspace(domain, ("CG", 1, ()))
T_tilde = ufl.TestFunction(V)
T_hat = ufl.TrialFunction(V)
T = fem.Function(V, name="T")
sigma = ufl.grad(T)
dT = dolfinx.fem.Function(V, name="dT")


def non_zero_guess(x):
    return x[0, :] + 2.0*x[1, :]


A = 1.0
B = 1.0


def on_boundary(x):
    return np.isclose(x[0], 0)


boundary_dofs = fem.locate_dofs_geometrical(V, on_boundary)
bc = fem.dirichletbc(PETSc.ScalarType(0), boundary_dofs, V)

# %% [markdown]
# ## Defining the external operator
#

# %% [markdown]
# The external operator must be defined in quadrature finite element space.

# %%
quadrature_degree = 2
dx = ufl.Measure("dx", metadata={
                 "quadrature_scheme": "default", "quadrature_degree": quadrature_degree})
Qe = basix.ufl.quadrature_element(
    domain.topology.cell_name(), degree=quadrature_degree, value_shape=(2,))
Q = dolfinx.fem.functionspace(domain, Qe)
num_cells = domain.topology.index_map(domain.topology.dim).size_local
num_gauss_points = Qe.custom_quadrature()[0].shape[0]

# %% [markdown]
# Now we need to define functions that will compute the exact values of the external operator and its derivatives. The framework gives the complete freedom of how these functions are implemented. The only constraints are:
# 1. They recieve `ndarray` (Numpy-like) arrays on their input.
# 2. They return a `ndarray` array, a vector holding degrees-of-freedom of the coefficient representing an external operator. This coefficient is accessible through `ref_coefficient` attribute of `femExternalOperator`.
#
# Thanks to the popularity of the Numpy package, there is plenty of other Python libraries that support the integration of `ndarray` data. Thus, there are numerous ways to define required functions. In this notebook, we focus on leverage of two powerfull packages: Numba and JAX.

# %% [markdown]
# ### Numba
#
# The package Numba allows its users to write just-in-time (JIT) compilable Python functions. Numba typically produces highly optimised machine code with runtime performance on the level of traditional compiled languages. It is strongly integrated with Numpy and supports its numerous features, including `ndarray` data. Thus, NUmba package perfectly fits as tool to define the external operators behaviour.
#
# Let us demonstrate here below, how by using simple Python loops and JIT-ed by Numba functions we define the evaluation of the heat flux $\boldsymbol{j}$ and its derivatives $\frac{d\boldsymbol{j}}{d T}$ and $\frac{d\boldsymbol{j}}{d\boldsymbol{\sigma}}$ at machine-code performance level.

# %%
I = np.eye(2)


@numba.njit
def K(T):
    return 1.0/(A + B*T)


@numba.njit
def func_j_numba(T, sigma):
    # j : scalar x vector -> vector
    T_ = T.reshape((num_cells, num_gauss_points))
    sigma_ = sigma.reshape((num_cells, num_gauss_points, 2))
    j_ = np.empty_like(sigma_)
    for i in range(0, num_cells):
        for j in range(0, num_gauss_points):
            j_[i, j] = -K(T_[i, j]) * sigma_[i, j]
    return j_.reshape(-1)


@numba.njit
def func_djdT_numba(T, sigma):
    # djdT : scalar x vector -> vector
    T_ = T.reshape((num_cells, num_gauss_points))
    sigma_ = sigma.reshape((num_cells, num_gauss_points, 2))
    djdT = np.empty_like(sigma_)

    for i in range(0, num_cells):
        for j in range(0, num_gauss_points):
            djdT[i, j] = B * K(T_[i, j])**2 * sigma_[i, j]
    return djdT.reshape(-1)


@numba.njit
def func_djdsigma_numba(T, sigma):
    # djdsigma : scalar x vector -> tensor
    T_ = T.reshape((num_cells, num_gauss_points))
    djdsigma_ = np.empty((num_cells, num_gauss_points, 2, 2),
                         dtype=PETSc.ScalarType)

    for i in range(0, num_cells):
        for j in range(0, num_gauss_points):
            djdsigma_[i, j] = -K(T_[i, j])*I
    return djdsigma_.reshape(-1)

# %%


def j_external_numba(derivatives):
    """Concrete numba implementation of external operator and its derivatives."""
    if derivatives == (0, 0):
        return func_j_numba
    elif derivatives == (1, 0):
        return func_djdT_numba
    elif derivatives == (0, 1):
        return func_djdsigma_numba
    else:
        return NotImplementedError

# %% [markdown]
# ### JAX
#
# In some applications, explicit expression of derivatives of quantity of interest either is difficult to derive or is not possible due to different causes. Automatic differentiation may help to solve this issue. In the context of Python, the JAX package provides the ne
#
# Moreover, JAX supports the just-in-time compilation and vectorization feature
#
# Note: Numba supports the vectorisation feature as well (through the `@guvectorize` decorator), but does not have the AD tool.

# %%


@jax.jit
def K(T):
    return 1.0/(A + B*T)


@jax.jit
def j_jax(T, sigma):
    return -K(T) * sigma


djdT = jax.jit(jax.jacfwd(j_jax, argnums=(0)))
djdsigma = jax.jit(jax.jacfwd(j_jax, argnums=(1)))

# vectorization: j_vec(T=(batch_size, 1), sigma=(batch_size, 2))
j_vec = jax.jit(jax.vmap(j_jax, in_axes=(0, 0)))
djdT_vec = jax.jit(jax.vmap(djdT, in_axes=(0, 0)))
djdsigma_vec = jax.jit(jax.vmap(djdsigma, in_axes=(0, 0)))


@jax.jit
def func_j_jax(T, sigma):
    T_ = T.reshape((num_cells*num_gauss_points))
    sigma_ = sigma.reshape((num_cells*num_gauss_points, 2))
    j_ = j_vec(T_, sigma_)
    return j_.reshape(-1)


@jax.jit
def func_djdT_jax(T, sigma):
    T_ = T.reshape((num_cells*num_gauss_points))
    sigma_ = sigma.reshape((num_cells*num_gauss_points, 2))
    j_ = djdT_vec(T_, sigma_)
    return j_.reshape(-1)


@jax.jit
def func_djdsigma_jax(T, sigma):
    T_ = T.reshape((num_cells*num_gauss_points))
    sigma_ = sigma.reshape((num_cells*num_gauss_points, 2))
    j_ = djdsigma_vec(T_, sigma_)
    return j_.reshape(-1)

# sigma_zero_ = jnp.array([0.0, 0.0])
# @jax.jit
# def func_j_jax(T, sigma):
#     # j : scalar x vector -> vector
#     T_vectorized = T.reshape((num_cells*num_gauss_points, 1))
#     sigma_vectorized = sigma.reshape((num_cells*num_gauss_points, 2))
#     out = vj(T_vectorized, sigma_vectorized)
#     return out.reshape(-1)

# @jax.jit
# def func_djdT_jax(T, sigma):
#     # djdT : scalar x vector -> vector
#     T_vectorized = T.reshape((num_cells*num_gauss_points, 1))
#     sigma_vectorized = sigma.reshape((num_cells*num_gauss_points, 2))
#     out = vdjdT(T_vectorized, sigma_vectorized)
#     return out.reshape(-1)

# @jax.jit
# def func_djdsigma_jax(T, sigma):
#     # djdsigma : scalar x vector -> tensor
#     T_vectorized = T.reshape((num_cells*num_gauss_points, 1))
#     sigma_vectorized = sigma.reshape((num_cells*num_gauss_points, 2))
#     out = vdjdsigma(T_vectorized, sigma_vectorized)

#     return out.reshape(-1)

# %%


def j_external_jax(derivatives):
    """Concrete JAX implementation of external operator and its derivatives."""
    if derivatives == (0, 0):
        return func_j_jax
    elif derivatives == (1, 0):
        return func_djdT_jax
    elif derivatives == (0, 1):
        return func_djdsigma_jax
    else:
        return NotImplementedError

# %% [markdown]
# ## Solving the problem using external operators


# %%
j = ex_op_env.femExternalOperator(
    T, sigma, function_space=Q, external_function=j_external_jax)

F_ext = ufl.inner(j, ufl.grad(T_tilde))*dx
J_ext = ufl.derivative(F_ext, T, T_hat)

# %%
F_replaced, F_ex_ops_list = ex_op_env.replace_external_operators(F_ext)
F_dolfinx = fem.form(F_replaced)

# %%
J_expanded = ufl.algorithms.expand_derivatives(J_ext)
J_replaced, J_ex_ops_list = ex_op_env.replace_external_operators(J_expanded)
J_dolfinx = fem.form(J_replaced)

# %%
timer1 = common.Timer("1st numba pass")
start = MPI.Wtime()
timer1.start()

evaluated_operands = ex_op_env.evaluate_operands(F_ex_ops_list)
ex_op_env.evaluate_external_operators(F_ex_ops_list, evaluated_operands)
# NOTE: Operands are re-used from previous step.
ex_op_env.evaluate_external_operators(J_ex_ops_list, evaluated_operands)

end = MPI.Wtime()
timer1.stop()
time1 = end - start

# %%
timer2 = common.Timer("1st numba pass")
start = MPI.Wtime()
timer2.start()

evaluated_operands = ex_op_env.evaluate_operands(F_ex_ops_list)
ex_op_env.evaluate_external_operators(F_ex_ops_list, evaluated_operands)
# NOTE: Operands are re-used from previous step.
ex_op_env.evaluate_external_operators(J_ex_ops_list, evaluated_operands)

end = MPI.Wtime()
timer2.stop()
time2 = end - start

# %%
T.interpolate(non_zero_guess)
evaluated_operands = ex_op_env.evaluate_operands(F_ex_ops_list)
ex_op_env.evaluate_external_operators(F_ex_ops_list, evaluated_operands)
ex_op_env.evaluate_external_operators(J_ex_ops_list, evaluated_operands)

linear_problem = solvers.LinearProblem(J_replaced, -F_replaced, T, bcs=[bc])

linear_problem.assemble_vector()
norm_residue_0 = linear_problem.b.norm()
norm_residue = norm_residue_0
tol, n_iter_max = 1e-3, 500
n_iter = 0

# %%
dT_values_ex_op = []

# %%
timer3 = common.Timer("Solving the problem")
start = MPI.Wtime()
timer3.start()

print(f"Residue0: {norm_residue_0}")
while norm_residue/norm_residue_0 > tol and n_iter < n_iter_max:
    linear_problem.assemble_matrix()
    linear_problem.solve(dT)
    T.vector.axpy(1, dT.vector)
    T.x.scatter_forward()
    dT_values_ex_op.append(dT.x.array)

    evaluated_operands = ex_op_env.evaluate_operands(F_ex_ops_list)
    ex_op_env.evaluate_external_operators(F_ex_ops_list, evaluated_operands)
    # NOTE: Operands are re-used from previous step.
    ex_op_env.evaluate_external_operators(J_ex_ops_list, evaluated_operands)

    linear_problem.assemble_vector()
    norm_residue = linear_problem.b.norm()
    print(f"Iteration# {n_iter} Residue: {norm_residue}")
    n_iter += 1

end = MPI.Wtime()
timer3.stop()

total_time_ex_op = end - start

# print(f'rank#{MPI.COMM_WORLD.rank}: Total time = {total_time_ex_op:.3f} (s)')
# print(f'rank#{MPI.COMM_WORLD.rank}: Compilation overhead: {time1 - time2:.3f} s')
# print(f'rank#{MPI.COMM_WORLD.rank}: Total time pure UFL: {total_time_pure_ufl:.3f} s')

# %% [markdown]
# ## Solving the problem using only UFL

# %%
K = 1.0/(A + B*T)
j = -K*sigma
F = ufl.inner(j, ufl.grad(T_tilde))*dx
J = ufl.derivative(F, T, T_hat)
T.interpolate(non_zero_guess)

# %%
linear_problem = solvers.LinearProblem(J, -F, T, bcs=[bc])
linear_problem.assemble_vector()

norm_residue_0 = linear_problem.b.norm()
norm_residue = norm_residue_0
norm_residue_0

n_iter = 0

# %%
dT_values_pure_ufl = []

# %%
start = MPI.Wtime()

while norm_residue/norm_residue_0 > tol and n_iter < n_iter_max:
    linear_problem.assemble_matrix()
    linear_problem.solve(dT)
    T.vector.axpy(1, dT.vector)
    T.x.scatter_forward()
    dT_values_pure_ufl.append(dT.x.array)

    linear_problem.assemble_vector()
    norm_residue = linear_problem.b.norm()

    print(f"Iteration# {n_iter} Residue: {norm_residue}")
    n_iter += 1

end = MPI.Wtime()

total_time_pure_ufl = end - start
print(f'rank#{MPI.COMM_WORLD.rank}: Total time: {total_time_pure_ufl} s')

# %%
for i in range(len(dT_values_pure_ufl)):
    print(np.max(np.abs(dT_values_pure_ufl[i] - dT_values_ex_op[i])))


# %%
dT_values_pure_ufl[3]

# %%
dT_values_ex_op[3]

# %%
