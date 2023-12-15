# %% [markdown]
# # Nonlinear heat equation (numpy)
#
# In this notebook we implement a numerical solution of steady-state heat
# equation with an external operator to used to define a non-linear thermal
# conductivity law relating the temperature gradient with the flux.
# 
# In this tutorial you will learn how to:
#
# - define a UFL form including an `ExternalOperator` which allows the
#   symbolically representation of an external operator,
# - define a concrete implementation of the `ExternalOperator` and its
#   derivatives using `numpy`,
# - how to assemble the Jacobian and residual operators for use inside
#   e.g. Newton's method.
#
# We assume some familiarity with finite element methods, non-linear mechanics
# and FEniCS.
#
# ## Problem formulation
#
# Denoting the temperature field $T$ and its gradient $\boldsymbol{\sigma} :=
# \nabla T$ we consider the following system on the square domain $\Omega :=
# [0, 1]^2$ with boundary $\partial \Omega$
#
# \begin{align*}
#      \nabla \cdot \boldsymbol{q} &= f \quad \mathrm{on} \; \Omega, \\
#      \boldsymbol{q}(T, \boldsymbol{\sigma}(T)) &= -k(T) \boldsymbol{\sigma}, \\
#      T &= 0 \quad \mathrm{on} \; \partial \Omega, \\
# \end{align*}
# where $f$ is a given function. With the material conductivity $k =
# \mathrm{const}$ we recover the standard Fourier heat problem. However, here
# we will assume that the thermal conductivity $k$ is some general function of
# $T$ and that we would like specify $k$ using some external (non-UFL) piece of
# code. For notational convenience in most of what follows we have omitted the
# explicit dependence of $\boldsymbol{q}$ on $T$ and $\boldsymbol{\sigma}(T) =
# \nabla T$.

# Let $V = H^1_0(\Omega)$ be the usual Sobolev space of square-integrable
# functions with square-integrable derivatives and vanishing value on the
# boundary $\partial \Omega$. Then in a variational setting the problem can be
# written in residual form as find $T \in V$ such that
# \begin{equation*}
# F(\boldsymbol{q}; \tilde{T}) = \int \boldsymbol{q} \cdot \nabla \tilde{T} - f \cdot
# \tilde{T} \; \mathrm{d}x = 0 \quad \forall \tilde{T} \in V,
# \end{equation*}
# where the semi-colon denotes the split between arguments in which the form is
# non-linear ($T$) and linear ($\tilde{T}$). In order to solve the nonlinear
# system of equation we apply Newton's method which requires the computation of
# Jacobian, or the Gateaux derivative of $F$. 
#
# \begin{equation*}
# D_{T} [ F(\boldsymbol{q}; \tilde{T}) ] \lbrace \hat{T} \rbrace := \int D_{T}[
# \boldsymbol{q} ]\lbrace \hat{T} \rbrace \cdot
# \nabla \tilde{T} \; \mathrm{d}x.
# \end{equation*}
#
# Emphasing the explicit dependence of $\boldsymbol{q}$ on $T$, i.e.
# $\boldsymbol{q} = \boldsymbol{q}(T, \sigma(T))$, we can use the chain rule to
# write
# \begin{align*}
# D_{T}[\boldsymbol{q}]\lbrace \hat{T} \rbrace &= D_T [\boldsymbol{q}]\lbrace
# D_T[T]\lbrace \hat{T} \rbrace \rbrace +
# D_{\boldsymbol{\sigma}}[\boldsymbol{q}] \lbrace
# D_T[\boldsymbol{\sigma}]\lbrace \hat{T} \rbrace \rbrace \\
# &= D_T [\boldsymbol{q}]\lbrace \hat{T} \rbrace +
# D_{\boldsymbol{\sigma}}[\boldsymbol{q}] \lbrace \nabla \hat{T} \rbrace \\
# \end{align*}
# 
# To fix ideas, we now assume the following explicit form for the material
# conductivity
# \begin{equation*}
# k(T) = \frac{1}{A + BT}
# \end{equation*}
# where $A$ and $B$ are material constants. After some algebra we can derive
# \begin{align*}
# D_T [\boldsymbol{q}]\lbrace \hat{T} \rbrace &=
# [Bk(T)k(T)\boldsymbol{\sigma}(T)] \hat{T} \\
# D_{\boldsymbol{\sigma}}[\boldsymbol{q}] \lbrace \nabla \hat{T} \rbrace &=
# [-k(T) \boldsymbol{I}] : \nabla \hat{T}  
# \end{align*}
# We now proceed to the definition of residual and Jacobian of this problem
# using the `ExternalOperator` approach.
# ```{note}
# This simple example can also be implemented in pure UFL and the Jacobian
# derived symbolically using UFL's usual `derivative` function.
# ```
#
# ## Implementation
# ### Preamble
#
# We import from the required packages, create a mesh, and a scalar valued
# function space which we will use to discretise the temperature field $T$.
# %%

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import basix
from ufl import TrialFunction, TestFunction, grad
from dolfinx import fem, mesh

from dolfinx_external_operator import FEMExternalOperator

domain = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
V = fem.functionspace(domain, ("CG", 1))

# %% [markdown]
# ### Defining the external operator
# We will begin by defining the input *operands* we need to specify the
# external operator $\boldsymbol{q}(T, \boldsymbol{\sigma})$.
# 
# Our external operator is a function of $T$ and $\boldsymbol{\sigma} := \nabla
# T$:
# %%
T = fem.Function(V)
sigma = grad(T)

# %% [markdown]
# We also need to define a `FunctionSpace` on which the output of the external
# operator $q$ should live. For optimal convergence it is important that $q$ is
# evaluated directly at the Gauss points used in the integration of the weak
# form.
# %%
quadrature_degree = 2
Qe = basix.ufl.quadrature_element(domain.topology.cell_name(),
                                  degree=quadrature_degree, value_shape=(2,))
Q = fem.functionspace(domain, Qe)

# %% [markdown]
# We now have the ingredients to define the external operator.
# %%
q = FEMExternalOperator(T, sigma, function_space=Q) 
# %% [markdown]
# Note that at this stage 
# %%

# %% Residual equation
# We now have the algebraic
# %%
dx = ufl.Measure("dx", metadata={"quadrature_scheme": "default",
                                 "quadrature_degree": quadrature_degree})


# %% [markdown]
# Now we need to define functions that will compute the exact values of the
# external operator and its derivatives. The framework gives the complete
# freedom of how these functions are implemented. The only constraints are: 1.
# They recieve `ndarray` (Numpy-like) arrays on their input. 2. They return a
# `ndarray` array, a vector holding degrees-of-freedom of the coefficient
# representing an external operator. This coefficient is accessible through
# `ref_coefficient` attribute of `femExternalOperator`.
#
# Thanks to the popularity of the Numpy package, there is plenty of other
# Python libraries that support the integration of `ndarray` data. Thus, there
# are numerous ways to define required functions. In this notebook, we focus on
# leverage of two powerfull packages: Numba and JAX.
# %% [markdown]
# ### Numba
#
# The package Numba allows its users to write just-in-time (JIT) compilable
# Python functions. Numba typically produces highly optimised machine code with
# runtime performance on the level of traditional compiled languages. It is
# strongly integrated with Numpy and supports its numerous features, including
# `ndarray` data. Thus, NUmba package perfectly fits as tool to define the
# external operators behaviour.
#
# Let us demonstrate here below, how by using simple Python loops and JIT-ed by
# Numba functions we define the evaluation of the heat flux $\boldsymbol{j}$
# and its derivatives $\frac{d\boldsymbol{j}}{d T}$ and
# $\frac{d\boldsymbol{j}}{d\boldsymbol{\sigma}}$ at machine-code performance
# level.

# %%
Id = np.eye(2)


@numba.njit
def K(T):
    return 1.0 / (A + B * T)


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
            djdT[i, j] = B * K(T_[i, j]) ** 2 * sigma_[i, j]
    return djdT.reshape(-1)


@numba.njit
def func_djdsigma_numba(T, sigma):
    # djdsigma : scalar x vector -> tensor
    T_ = T.reshape((num_cells, num_gauss_points))
    djdsigma_ = np.empty((num_cells, num_gauss_points, 2, 2), dtype=PETSc.ScalarType)

    for i in range(0, num_cells):
        for j in range(0, num_gauss_points):
            djdsigma_[i, j] = -K(T_[i, j]) * Id
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
