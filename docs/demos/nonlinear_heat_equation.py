# %% [markdown]
# $\newcommand\ul[1]{\underline{#1}}$
#
# # Nonlinear heat equation
#
# In this notebook, we implement a numerical solution of a simple example of a nonlinear heat equation using an external operator. The problem can be solved using both pure UFL formulation and the application of an external operator. This gives an advantage///..
#
# Denoting the temperature field through $T$ we consider the following system
# $$
#     \begin{align*}
#         \Omega : \quad & \nabla \cdot (K \nabla T) = 0 \\
#         \partial\Omega_D : \quad & T(\ul{x}) = 0 \\
#         \Omega : \quad & T(\ul{x},0) = x + 2y
#     \end{align*}
# $$
# where $K(T) = \frac{1}{a + b T}$ is a nonlinear thermal conductivity, $a$ and $b$ are some constants.
#
# Let $V = H^1_0(\Omega)$ be the functional space of admissible temperature fields then in a variational setting the problem can be written as follows.
#
# Find $T \in V$ such that
# $$
#     \begin{equation}
#         F(T; \tilde{T}) = -\int\frac{1}{a + b T}\underline{\nabla T} . \underline{\nabla \tilde{T}} dx = \int\underline{j}(T,\underline{\sigma}(T)) . \underline{\nabla \tilde{T}} dx = 0, \quad \forall T \in V,
#     \end{equation}
# $$
# where $\ul{j} = - K(T) \underline{\sigma}(T) = \frac{1}{a + bT}\underline{\sigma}(T)$ is a nonlinear heat flux and $\underline{\sigma}(T)$ is equal to the gradient $\underline{\nabla T}$ and introduced for simplicity.
#
# In order to solve the nonlinear equation $(1)$ we apply the Newton method and calculate the Gateau derivative of $F$ with respect to operand $T$ in the direction $\bar{T} \in V$ as following:
# $$
#     \begin{align*}
#         & J(T;\bar{T},\tilde{T}) = \frac{d F}{d T}(\ul{j}(T,\ul{\sigma}(T));\bar{T}, \tilde{T}) = \int\frac{d\ul{j}}{dT}(T,\ul{\sigma}(T);\bar{T}) \ul{\nabla \tilde{T}} dx,
#     \end{align*}
# $$
# where through $d \cdot / dT$ we denote the Gateau derivative with respect to operand $T$ in the direction $\bar{T} \in V$.
#
# In this example, we treat the heat flux $\ul{j}$ as a function of two operands $T$ and $\ul{\sigma}(T) = \ul{\nabla T}$. In this regard, by applying the chain rule, let us write out the explicit expression of the Gateau derivative of $\ul{j}$ here below
# $$
#     \begin{align*}
#         & \frac{d\ul{j}}{dT}(T,\ul{\sigma}(T);\bar{T}) = \frac{\partial\ul{j}}{\partial T} + \frac{\partial\ul{j}}{\partial\ul{\sigma}}\frac{\partial\ul{\sigma}}{\partial T} = -\ul{\sigma}(T)(-bK^2(T))\bar{T} - K(T)\ul{\ul{I}}:\nabla\bar{T}.
#     \end{align*}
# $$
#
# In our implementation, the heat flux $\ul{j}$ acts like an external operator and its behaviour as well as the behaviour of its derivatives, will be defined in an external function.
#
# In general, the same function can be presented in numerous variations by selecting different operands as sub-expressions of this function. In our case, for example, we could have presented the heat flux $\ul{j}$ as a function of $K(T)$ and $\sigma(T)$ operands, but this decision would have led to more midterms due to the chain rule and therefore to more computation costs. Thus, it is important to choose wisely the operands of the external operators, which you want to use.
#
#
# In this regard, the linear form $F$ has one operand, the heat flux $\ul{j}(T)$, and one argument $\tilde{T}$:
# $$
#     F(T; \tilde{T}) = F(\ul{j}(T,\ul{\sigma}(T)); \tilde{T}).
# $$
#
# where $\partial \cdot / \partial T$ is a gateau derivative.
#
# Can we say that a Gateau derivative can be full or partial?

# %%


# %%
import solvers
import external_operator as ex_op_env
import sys
from petsc4py import PETSc
from mpi4py import MPI
import dolfinx.fem.petsc  # there is an error without it, why?
from dolfinx import fem, mesh, common
import ufl
import basix
import jax.numpy as jnp
from jax import jit, jacrev, vmap
import numpy as np

import numba

from jax import config
config.update("jax_enable_x64", True)  # replace by JAX_ENABLE_X64=True


sys.path.append("..")


# %% [markdown]
# <div hidden>
# $\newcommand\ul[1]{\underline{#1}}$
# \vskip-\parskip
# \vskip-\baselineskip
# </div>

# %%
nx = 5

domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, nx)
quadrature_degree = 2
dx = ufl.Measure("dx", metadata={
                 "quadrature_scheme": "default", "quadrature_degree": quadrature_degree})

# %%
gdim = domain.geometry.dim
V = fem.functionspace(domain, ("CG", 1, ()))
T_bar = ufl.TrialFunction(V)
T_tilde = ufl.TestFunction(V)


def non_zero_guess(x):
    return x[0, :] + 2.0*x[1, :]


T = fem.Function(V, name="T")
T_n = fem.Function(V, name="T_n")
dT = dolfinx.fem.Function(V, name="dT")

T.interpolate(non_zero_guess)
T_n.interpolate(non_zero_guess)

Qe = basix.ufl.quadrature_element(
    domain.topology.cell_name(), degree=quadrature_degree, value_shape=(2,))
Q = dolfinx.fem.functionspace(domain, Qe)

# %%


def on_boundary(x):
    return np.isclose(x[0], 0)


boundary_dofs = fem.locate_dofs_geometrical(V, on_boundary)
bc = fem.dirichletbc(PETSc.ScalarType(0), boundary_dofs, V)

# %% [markdown]
# ## Pure UFL implementation

# %%
sigma = ufl.grad(T)
A = fem.Constant(domain, 1.0)
B = fem.Constant(domain, 1.0)
K = 1.0/(A + B*T)
j = -K*sigma
# dt = 0.001
# F = T * T_tilde * dx - dt * ufl.inner(j, ufl.grad(T_tilde))*dx - T_n * T_tilde * dx
F = ufl.inner(j, ufl.grad(T_tilde))*dx
J = ufl.derivative(F, T, T_bar)

# %%
tol, n_iter_max = 1e-3, 500
n_iter = 0

pure_ufl_problem = solvers.LinearProblem(J, -F, T, bcs=[bc])
pure_ufl_problem.assemble_vector()

norm_residue_0 = pure_ufl_problem.b.norm()
norm_residue = norm_residue_0
norm_residue_0

# %%
dT_values_pure_ufl = np.zeros((10, T.x.array.shape[0]))

# %%
start = MPI.Wtime()

while norm_residue/norm_residue_0 > tol and n_iter < n_iter_max:

    pure_ufl_problem.assemble_matrix()

    pure_ufl_problem.solve(dT)

    T.vector.axpy(1, dT.vector)
    T.x.scatter_forward()
    dT_values_pure_ufl[n_iter][:] = dT.x.array[:]

    pure_ufl_problem.assemble_vector()
    norm_residue = pure_ufl_problem.b.norm()

    n_iter += 1
    print(f"\tResidue: {norm_residue}, Iteration #{n_iter}\n")

end = MPI.Wtime()

total_time_pure_ufl = end - start
print(f'rank#{MPI.COMM_WORLD.rank}: Total time: {total_time_pure_ufl} s')

# %% [markdown]
# ## ExternalOperator approach

# %%
A = 1.0
B = 1.0

quadrature_points = basix.make_quadrature(
    basix.CellType.triangle, quadrature_degree, basix.QuadratureType.Default)
num_cells = domain.topology.index_map(domain.topology.dim).size_local
num_gauss_points = quadrature_points[0].shape[0]
I_ = np.eye(2).flatten()

# %%
num_cells

# %%
num_gauss_points

# %%


@numba.njit(fastmath=True)
def K(T):
    return 1.0/(A + B*T)


@numba.njit(fastmath=True)
def func_j(T, sigma):
    # j : scalar x vector -> vector
    out = np.zeros_like(sigma)
    for i in range(0, num_cells):
        for j in range(0, num_gauss_points):
            T_ = T[i, 1*j:j + 1]
            sigma_ = sigma[i, 2*j:2*j + 2]
            out[i, 2*j:2*j + 2] = -K(T_)*sigma_
    return out.reshape(-1)


@numba.njit(fastmath=True)
def func_djdT(T, sigma):
    # djdT : scalar x vector -> vector
    # out = np.empty_like(sigma)
    out = np.empty((num_cells, num_gauss_points * 2))

    for i in range(0, num_cells):
        for j in range(0, num_gauss_points):
            T_ = T[i, 1*j:j + 1]
            sigma_ = sigma[i, 2*j:2*j + 2]
            out[i, 2*j:2*j + 2] = B*K(T_)**2*sigma_

    return out.reshape(-1)


@numba.njit(fastmath=True)
def func_djdsigma(T, sigma):
    # djdsigma : scalar x vector -> tensor
    out = np.empty((num_cells, num_gauss_points * 4))
    for i in range(0, num_cells):
        for j in range(0, num_gauss_points):
            T_ = T[i, 1*j: j + 1]
            out[i, 4*j:4*j + 4] = -K(T_)*I_
    return out.reshape(-1)

# %%
# My Notes
# 1. Use jax object inside jax transformations (like jit)
# 2. How many jit-s to use?


@jit
def j_jax(T, sigma):
    return -1./(A + B*T) * sigma


dj = jit(jacrev(j_jax, argnums=(0, 1)))
djdT = jit(jacrev(j_jax, argnums=(0)))
djdsigma = jit(jacrev(j_jax, argnums=(1)))

# vectorization in the way: vj(T=(batch_size, 1), sigma=(batch_size, 2))
vj = jit(vmap(j_jax, in_axes=(0, 0)))
vdjdT = jit(vmap(djdT, in_axes=(0, 0)))
vdjdsigma = jit(vmap(djdsigma, in_axes=(0, 0)))

sigma_zero_ = jnp.array([0.0, 0.0])


@jit
def func_j_jax(T, sigma):
    # j : scalar x vector -> vector
    T_vectorized = T.reshape((num_cells*num_gauss_points, 1))
    sigma_vectorized = sigma.reshape((num_cells*num_gauss_points, 2))
    out = vj(T_vectorized, sigma_vectorized)
    return out.reshape(-1)


@jit
def func_djdT_jax(T, sigma):
    # djdT : scalar x vector -> vector
    T_vectorized = T.reshape((num_cells*num_gauss_points, 1))
    sigma_vectorized = sigma.reshape((num_cells*num_gauss_points, 2))
    out = vdjdT(T_vectorized, sigma_vectorized)
    return out.reshape(-1)


@jit
def func_djdsigma_jax(T, sigma):
    # djdsigma : scalar x vector -> tensor
    T_vectorized = T.reshape((num_cells*num_gauss_points, 1))
    sigma_vectorized = sigma.reshape((num_cells*num_gauss_points, 2))
    out = vdjdsigma(T_vectorized, sigma_vectorized)

    return out.reshape(-1)

# %%


def j_ext(derivatives):
    """Concrete JAX implementation of external operator and its derivative."""
    if derivatives == (0, 0):
        return func_j_jax
    elif derivatives == (1, 0):
        return func_djdT_jax
    elif derivatives == (0, 1):
        return func_djdsigma_jax
    else:
        return NotImplementedError


def j_ext(derivatives):
    """Concrete numba implementation of external operator and its derivative."""
    if derivatives == (0, 0):
        return func_j
    elif derivatives == (1, 0):
        return func_djdT
    elif derivatives == (0, 1):
        return func_djdsigma
    else:
        return NotImplementedError


# %%
j = ex_op_env.femExternalOperator(
    T, sigma, function_space=Q, external_function=j_ext)

F_ext = ufl.inner(j, ufl.grad(T_tilde))*dx
J_ext = ufl.derivative(F_ext, T, T_bar)

# %%
F_replaced, F_ex_ops_list = ex_op_env.replace_external_operators(F_ext)
F_dolfinx = fem.form(F_replaced)

# %%
J_expanded = ufl.algorithms.expand_derivatives(J_ext)
J_replaced, J_ex_ops_list = ex_op_env.replace_external_operators(J_expanded)
J_dolfinx = fem.form(J_replaced)

# %%
T.interpolate(non_zero_guess)
T_n.interpolate(non_zero_guess)

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
external_operator_problem = solvers.LinearProblem(
    J_replaced, -F_replaced, T, bcs=[bc])

n_iter = 0

external_operator_problem.assemble_vector()

norm_residue_0 = external_operator_problem.b.norm()
norm_residue = norm_residue_0

# %%
dT_values_ex_op = np.zeros((10, T.x.array.shape[0]))

# %%
timer3 = common.Timer("Solving the problem")
start = MPI.Wtime()
timer3.start()

while norm_residue/norm_residue_0 > tol and n_iter < n_iter_max:

    external_operator_problem.assemble_matrix()

    external_operator_problem.solve(dT)

    T.vector.axpy(1, dT.vector)
    T.x.scatter_forward()
    dT_values_ex_op[n_iter][:] = dT.x.array[:]

    evaluated_operands = ex_op_env.evaluate_operands(F_ex_ops_list)
    ex_op_env.evaluate_external_operators(F_ex_ops_list, evaluated_operands)
    # NOTE: Operands are re-used from previous step.
    ex_op_env.evaluate_external_operators(J_ex_ops_list, evaluated_operands)

    external_operator_problem.assemble_vector()
    norm_residue = external_operator_problem.b.norm()

    n_iter += 1
    print(f"\tResidue: {norm_residue}, Iteration #{n_iter}\n")

end = MPI.Wtime()
timer3.stop()

total_time_ex_op = end - start

print(f'rank#{MPI.COMM_WORLD.rank}: Total time = {total_time_ex_op:.3f} (s)')
print(f'rank#{MPI.COMM_WORLD.rank}: Compilation overhead: {time1 - time2:.3f} s')
print(
    f'rank#{MPI.COMM_WORLD.rank}: Total time pure UFL: {total_time_pure_ufl:.3f} s')

# %%
for i in range(4):
    print(np.max(np.abs(dT_values_pure_ufl - dT_values_ex_op)))
