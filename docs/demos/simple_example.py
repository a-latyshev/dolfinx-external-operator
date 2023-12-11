# %% [markdown]
# # Simple example
#
# Authors: Andrey Latyshev (University of Luxembourg, Sorbonne Univeristé, andrey.latyshev@uni.lu)
#
# In order to show how an external operator can be used in a variational setting in FEniCSx/DOLFINx, we want to start with a simple example.
#
# Let us denote an external operator that is not expressible through UFL by $N = N(u)$, where $u$ is its single operand from functional space $V$. In these terms, we consider the following linear form $F$.
#
# $$
#   F(N(u);v) = \int N(u)v \, dx
# $$
#
# In a variational setting, we quite often need to compute the Jacobian of the Form $F$. In other words, we need to take the Gateau derivative of the functional $F$ in the direction of $\hat{u}$. Denoting the full and partial Gateau derivatives of a functional through $\frac{d }{d u}(\cdot)$ and $\frac{\partial}{\partial u}(\cdot)$ respectively, applying the chain rule and omitting the operand of $N$, we can express the Jacobian of $F$ as following:
#
# $$
#   J(N;\hat{u}, v) = \frac{dF}{du}(N;\hat{u}, v) = \frac{\partial F}{\partial N}(N; \hat{N}, v) = \int \hat{N}(u;\hat{u})v \, dx,
# $$
#
# where $\hat{N}(u;\hat{u})$ is a new trial function, which behaviour is defined by the Gateau derivative $\frac{\partial N}{\partial u}(u;\hat{u})$.
# ___
# or
#
# $$
#   J(N;\hat{u}, v) = \frac{dF}{du}(N;\hat{u}, v) = \frac{\partial F}{\partial N}(N; \hat{N}, v) \circ \left( \hat{N} = \frac{\partial N}{\partial u}(u;\hat{u}) \right) = \int \hat{N}(u;\hat{u})v \, dx,
# $$
#
# where $\hat{N}(u;\hat{u}) = \frac{\partial N}{\partial u}(u;\hat{u})$ is a Gateau derivative of the external operator $\hat{N}$.
# ___
# or
#
# $$
#   J(N;\hat{u}, v) = \frac{dF}{du}(N;\hat{u}, v) = \frac{\partial F}{\partial N}(N; \frac{\partial N}{\partial u}(u;\hat{u}), v) = \int \hat{N}(u;\hat{u})v \, dx,
# $$
#
# where $\hat{N}(u;\hat{u}) = \frac{\partial N}{\partial u}(u;\hat{u})$ is a Gateau derivative of the external operator $\hat{N}$.
# ___
# or
#
# $$
#   J(N;\hat{u},v) = F^\prime(N; \hat{N}(u;\hat{u}),v) = (F^\prime \circ \hat{N})(u;\hat{u},v)
# $$
#
# ___
#
# Chain rule (according to [wiki](https://en.wikipedia.org/wiki/Gateaux_derivative)):
#
# \begin{align*}
#   & H(u) = (G \circ F)(u) = G(F(u)) \\
#   & H^\prime(u; \hat{u}) = (G \circ F)^\prime(u; \hat{u}) = G^\prime(F(u); \hat{F}(u; \hat{u})),
# \end{align*}
# where $\hat{G}(u; \hat{u}) = G^\prime(u; \hat{u})$.
#
# May we write ?
#
# $$
# H^\prime(u; \hat{u}) = (G \circ F)^\prime(u; \hat{u}) = G^\prime(F(u); \hat{F}(u; \hat{u})) = (G^\prime \circ \hat{F})(u; \hat{u}),
# $$
# ___
#
# Thus, the Jacobian $J$ is presented as an action of the functional $\frac{\partial F}{\partial N}$ on the trial function $\hat{N}$....
#
# The behaviour of both external operators $N$ and $\frac{d N}{d u}$ must be defined by a user via any callable Python function.
#
# Here below we demonstrate how the described above linear and bilinear forms can be easily defined in UFL using its automatic differentiation tool and the new object `femExternalOperator`.

# %% [markdown]
# For the sake of simplicity, we chose the following definition of the external operator $N$
#
# $$
#     N(u) = u^2,
# $$
#
# which is easily expressible via UFL, in order to show that our framework works correctly on this simple example.

# %% [markdown]
# ## Preamble
#
# Here we import the required Python packages, build a simple square mesh and define the finite element functional space $V$, where the main variable $u$, test and trial functions exist.

# %%
import external_operator as ex_op_env
from mpi4py import MPI
from petsc4py import PETSc

import basix
import ufl
from dolfinx import fem, mesh, common
import dolfinx.fem.petsc  # there is an error without it, why?
from ufl.algorithms import expand_derivatives

import numpy as np

import sys
sys.path.append("../../src/dolfinx_ExternalOperator")

nx = 2
domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, nx)

gdim = domain.geometry.dim
V = fem.functionspace(domain, ("CG", 1, ()))
v = ufl.TestFunction(V)
u_hat = ufl.TrialFunction(V)
u = fem.Function(V)
u.x.array[:] = 1.  # in order to get non-zero forms after assembling

# %% [markdown]
# ## Defining the external operator

# %% [markdown]
# According to the current implementation of the framework, the finite element representation of the external operator must be defined in a quadrature finite element space.

# %%
Qe = basix.ufl.quadrature_element(domain.topology.cell_name(), degree=1)
Q = dolfinx.fem.functionspace(domain, Qe)
dx = ufl.Measure("dx", domain=domain, metadata={
                 "quadrature_degree": 1, "quadrature_scheme": "default"})

# %% [markdown]
# Now we define the behaviour of the external operator $N$ and its Gateau derivative $\frac{dN}{du}(\hat{u})$.

# %%


def func_N(u):
    """Defines the output of the external operator `N`.

    Args:
        u: A numpy-array containing global values of the variable `u`.

    Returns:
        A numpy-array containing global values of the external operator coefficient.
    """
    return np.reshape(u**2, -1)


def func_dNdu(u):
    """Defines the output of the derivative of the external operator `N`.

    Computes the Gateau derivative of the external operator `N` with respect to the operand `u`. TODO: it os not correctly writen

    Args:
        u: A numpy-array containing global values of the variable `u`.

    Returns:
        A numpy-array containing global values of the derivative of the external operator coefficient.
    """
    return np.reshape(2*u, -1)

# %% [markdown]
# Then we combine both functions `func_N` and `func_dNdu` in the function `f_external` that distinguish the external operator and its derivatives using the multi-index `derivatives`, the attribute of any external operator.

# %%


def f_external(derivatives):
    """Defines the behaviour of the external operator and its derivative.

    Args:
        derivatives: A tuple of integer positive values, the multi-index of external operator.

    Returns:
        A callable function evaluating eighter values of external operator or its derivatives.
    """
    if derivatives == (0,):
        return func_N
    elif derivatives == (1,):
        return func_dNdu
    else:
        return NotImplementedError

# %% [markdown]
# Now we have all the ingredients to define the external operator $N$.


# %%
N = ex_op_env.femExternalOperator(
    u, function_space=Q, external_function=f_external)

# %% [markdown]
# ## Defining the linear and bilinear forms

# %% [markdown]
# Thanks to the automatic differentiation tool of UFL we do not need to compute manually the derivative of the form $F$ and the operator $N$. We just have to define the linear form and the direction with respect to which the differentiation will be performed.

# %%
F = N*v*dx
J = ufl.derivative(F, u, u_hat)

# %% [markdown]
# Our framework recursively looks for `femExternalOperator` objects in the form and replaces them with their representatives, the globally allocated coefficients.

# %%
F_replaced, F_ex_ops_list = ex_op_env.replace_external_operators(F)
F_dolfinx = fem.form(F_replaced)

# %%
J_expanded = ufl.algorithms.expand_derivatives(J)
J_replaced, J_ex_ops_list = ex_op_env.replace_external_operators(J_expanded)
J_dolfinx = fem.form(J_replaced)

# %% [markdown]
# ```{note}
# As `femExternalOperator` combines properties of both UFL-like symbolic objects and finite objects `fem.Function`, the user must be aware that once the expansion of the derivatives is performed, the framework creates new functional spaces for shapes of derivatives of external operators and allocates memory for the appropriate coefficients. Thus, `expand_derivatives` may lead to "hidden" memory allocations, which may not be exptected by common UFL users.
# ```

# %% [markdown]
# Now the forms `F_replaced` and `J_replaced` do not contain any UFL-like `femExternalOperator` objects. Instead, they consist of `fem.Function` objects defined on the quadrature space `Q`.

# %% [markdown]
# Then we need to update the operand values of the external operator $N$, i.e. the field $u$. In this case, we just project the values of variable $u$ from the functional space $V$ onto the quadrature space $Q$.

# %%
evaluated_operands = ex_op_env.evaluate_operands(F_ex_ops_list)

# %% [markdown]
# As the operands of an external operand and its derivatives are the same, we evaluate them only once and send their values to the function `evaluate_external_operators`. The latter updates external operators contained by providing a list of `femExternalOperator` of a form and values of evaluated operands stored in numpy-like arrays.

# %%
ex_op_env.evaluate_external_operators(F_ex_ops_list, evaluated_operands)
ex_op_env.evaluate_external_operators(J_ex_ops_list, evaluated_operands)

# %% [markdown]
# The algorithm exploits the `f_external` to update coefficients representing external operators.

# %% [markdown]
# Concrete values of external operators can be easily accessible through the `ref_coefficient` attribute.

# %%
N.ref_coefficient.x.array

# %%
dNdu = J_ex_ops_list[0]
dNdu.ref_coefficient.x.array
