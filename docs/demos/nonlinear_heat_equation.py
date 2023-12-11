# %% [markdown]
# # Nonlinear heat equation
#
# In this notebook, we implement a numerical solution of a nonlinear heat equation using an external operator.
#
# ## Problem formulation
#
# Denoting the temperature field through $T$ and the space variable $\bm{x} = (x, y)^T$ we consider the following system on the square domain $\Omega$:
#
# \begin{align*}
# \Omega : \quad & \nabla \cdot (K(T) \nabla T) = 0 \\
# \partial\Omega : \quad & T(\pbm{x}) = 0
# \end{align*}
#
# where $K(T) = \frac{1}{a + b T}$ is a nonlinear thermal conductivity, $a$ and $b$ are some constants.
#
# Let $V = H^1_0(\Omega)$ be the functional space of admissible temperature fields then in a variational setting the problem can be written as follows.
#
# Find $T \in V$ such that
#
# $$
#     F(j; \tilde{T}) = -\int\frac{1}{a + b T}\nabla T . \nabla\tilde{T} dx = \int\mathbf{j}(T,\boldsym{\sigma}(T)) . \nabla\tilde{T} dx = 0, \quad \forall T \in V,
# $$ (eqn:1)
#
# where $\bm{j} = - K(T) \bm{\sigma}(T) = \frac{1}{a + bT}\bm{\sigma}(T)$ is a nonlinear heat flux and $\bm{\sigma}(T)$ is equal to the gradient $\nabla T$ and introduced for simplicity.
#
# In order to solve the nonlinear equation {eq}`eqn:1` we apply the Newton method and calculate the Gateau derivative of $F$ with respect to operand $T$ in the direction $\hat{T} \in V$ as follows:
#
#
# $$
#
#     J(j;\hat{T},\tilde{T}) = \frac{d F}{d T}(\bm{j}(T,\bm{\sigma}(T));\hat{T}, \tilde{T}) = \int\frac{d\bm{j}}{dT}(T,\bm{\sigma}(T);\hat{T}) \nabla\tilde{T} dx,
#
# $$
#
# where through $d \cdot / dT$ we denote the Gateau derivative with respect to operand $T$ in the direction $\hat{T} \in V$.
#
# In this example, we treat the heat flux $\bm{j}$ as an external operator with two operands $T$ and $\bm{\sigma}(T) = \nabla T$. In this regard, by applying the chain rule, let us write out the explicit expression of the Gateau derivative of $\bm{j}$ here below
#
#
# $$
#
#     \frac{d\bm{j}}{dT}(T,\bm{\sigma}(T);\hat{T}) = \frac{\partial\bm{j}}{\partial T} + \frac{\partial\bm{j}}{\partial\bm{\sigma}}\frac{\partial\bm{\sigma}}{\partial T} = -\bm{\sigma}(T)(-bK^2(T))\hat{T} - K(T)\mathbb{I}:\nabla\hat{T},
#
# $$
# where $\mathbb{I}$ is a second-order identity tensor.
#
# Despite the function can be explicitly expressed via UFL, we are going to define the heat flux through `femExternalOperator` object and calls of an external function.
#
# ```{note}
# In general, the same function can be presented in numerous variations by selecting different operands as sub-expressions of this function. In our case, for example, we could have presented the heat flux $\bm{j}$ as a function of $K(T)$ and $\sigma(T)$ operands, but this decision would have led to more midterms due to the chain rule and therefore to more computation costs. Thus, it is important to choose wisely the operands of the external operators, which you want to use.
# ```
#
# In order to start the numerical algorithm we initialize variable `T` with the following initial guess:
#
#
# $$
#
#     T(\bm{x}) = x + 2y
#
# $$
#
# ## Defining the external operator
#
# TOADD: the framework takes care of trial functions and expressions of it.
# $$
#

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
import jax.numpy as jnp
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


T.interpolate(non_zero_guess)

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
# The external operators must be defined in quadrature finite element space.
#

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
# ### Numba
#

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
    """Concrete numba implementation of external operator and its derivative."""
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
    """Concrete JAX implementation of external operator and its derivative."""
    if derivatives == (0, 0):
        return func_j_jax
    elif derivatives == (1, 0):
        return func_djdT_jax
    elif derivatives == (0, 1):
        return func_djdsigma_jax
    else:
        return NotImplementedError

# %% [markdown]
# ### Solving the problem using external operators


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
# ## Pure UFL implementation
#

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
