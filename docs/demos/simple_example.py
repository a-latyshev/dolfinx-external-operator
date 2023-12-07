# %% [markdown]
# # A simple example of an external operator
#
# Authors: Andrey Latyshev (University of Luxembourg, Sorbonne Univeristé, andrey.latyshev@uni.lu)
#
# In order to show how an external operator can be used in a variational setting in FEniCSx/DOLFINx, we want to start with a simple example.
#
# Let us denote an external operator that is not expressible through UFL by $N = N(u)$, where $u$ is its single operand existing in a functional space $V$. In these terms, we consider the following linear form $F$
# $$
#   F(u;v) = \int N(u)v dx
# $$
#
# \begin{align*}
#   &  \\
#   & J(u;\tilde{u},v) = F^\prime(u;\tilde{u},v) = \int \hat{N}(u;\tilde{u})v dx = 2\int u \tilde{u} v dx,
# \end{align*}
#
# where $\hat{N}(u;\tilde{u}) = N^\prime(u;\tilde{u}) = 2u\tilde{u}$.
#
# In other words:
# $$
#   J(u;\tilde{u},v) = F^\prime(u; \hat{N}(u;\tilde{u}),v) = (F^\prime \circ \hat{N})(u;\tilde{u},v)
# $$
#
# Chain rule:
#
# \begin{align*}
#   & H(u) = (F \circ G)(u) = F(G(u)) \\
#   & H^\prime(u;v) = (F \circ G)^\prime(u;v) = F^\prime(G(u); \hat{G}(u;v)),
# \end{align*}
# where $\hat{G}(u;v) = G^\prime(u;v)$

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
sys.path.append("../..")

nx = 2
domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, nx)

gdim = domain.geometry.dim
V = fem.functionspace(domain, ("CG", 1, ()))
v = ufl.TestFunction(V)
u_hat = ufl.TrialFunction(V)
u = fem.Function(V)

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
    return u**2


def func_dNdu(u):
    """Defines the output of the derivative of the external operator `N`.

    Computes the Gateau derivative of the external operator `N` with respect to the operand `u`. TODO: it os not correctly writen

    Args:
        u: A numpy-array containing global values of the variable `u`.

    Returns:
        A numpy-array containing global values of the derivative of the external operator coefficient.
    """
    return 2*u

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

# %%
F = N*v*dx
J = ufl.derivative(F, u, u_hat)

# %%
F_replaced, F_ex_ops_list = ex_op_env.replace_external_operators(F)
F_dolfinx = fem.form(F_replaced)

# %%
J_expanded = ufl.algorithms.expand_derivatives(J)
J_replaced, J_ex_ops_list = ex_op_env.replace_external_operators(J_expanded)
J_dolfinx = fem.form(J_replaced)
