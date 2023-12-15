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
# - define an external definition of the external operator using `numpy`,
# - and assemble the Jacobian and residual operators for use inside
#   e.g. Newton's method.
# 
# To keep the concepts simple we do not solve the non-linear problem, leaving
# that to a subsequent demo.
#
# We assume some basic familiarity with finite element methods, non-linear
# mechanics and FEniCS.
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
# non-linear and linear. To solve the nonlinear system of equation we apply
# Newton's method which requires the computation of Jacobian, or the Gateaux
# derivative of $F$. 
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
# [-k(T) \boldsymbol{I}] \cdot \nabla \hat{T}  
# \end{align*}
# We now proceed to the definition of residual and Jacobian of this problem
# using the `ExternalOperator` approach.
# ```{note}
# This simple model can also be implemented in pure UFL and the Jacobian
# derived symbolically using UFL's `derivative` function.
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
from dolfinx_external_operator.external_operator import replace_external_operators
import ufl
import ufl.algorithms
from ufl import TrialFunction, TestFunction, grad, inner, Measure, derivative
from dolfinx import fem, mesh

from dolfinx_external_operator import FEMExternalOperator, replace_external_operators

domain = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
V = fem.functionspace(domain, ("CG", 1))

# %% [markdown]
# ### Defining the external operator
# We will begin by defining the input *operands* we need to specify the
# external operator $\boldsymbol{q}(T, \boldsymbol{\sigma}(T))$.
# %%
T = fem.Function(V)
sigma = grad(T)

# %% [markdown]
# We also need to define a `FunctionSpace` on which the output of the external
# operator $\boldsymbol{q}$ should live. For optimal convergence it is
# necessary that $\boldsymbol{q}$ is evaluated directly at the Gauss points
# used in the integration of the weak form. This can be enforced by
# constructing a quadrature function space.
# %%
quadrature_degree = 2
Qe = basix.ufl.quadrature_element(domain.topology.cell_name(),
                                  degree=quadrature_degree, value_shape=(2,))
Q = fem.functionspace(domain, Qe)

# %% [markdown]
# We now have all of the ingredients to define the external operator.
# %%
q_ = FEMExternalOperator(T, sigma, function_space=Q) 

# %% [markdown]
# Note that at this stage `q_` is symbolic and we have not
# yet defined the `numpy` code to compute it.

# ```{note}
# FEMExternalOperator holds memory to store its values, like `fem.Function`. 
# ```
# %%

# %% [markdown]
# ### Residual
# The external operator can be used in the definition of the residual $F$.
# %%
dx = Measure("dx", metadata={"quadrature_scheme": "default",
                             "quadrature_degree": quadrature_degree})
T_tilde = TestFunction(V)
F = inner(q_, grad(T_tilde))*dx

# %% [markdown]
# ### Implementing the external operator
# The external operator is implemented using functional programming techniques
# that closely maps the way that `interpolate` is implemented in DOLFINx.
#
# In the first step, the user must define function(s) which accept `np.ndarray`
# containing the operands (here, $T$ and $\boldsymbol{\sigma}$) evaluated at
# all of the global interpolation points associated with the output function
# space. These function(s) must return `np.ndarray` objects containing the
# evaluation of the external operator at all of the global interpolation points.
# 
# We begin by defining the Python functions for the material conductivity $k$
# and the flux $q$
# \begin{align*}
# k(T) &= \frac{1}{A + BT} \\
# \boldsymbol{q} &= -k(T) \boldsymbol{\sigma}
# \end{align*}
# %%

def k(T):
    return 1.0 / (A + B * T)


def q(T, sigma):
    T_ = T.reshape((num_cells, num_gauss_points))
    sigma_ = sigma.reshape((num_cells, num_gauss_points, 2))
    j_ = np.empty_like(sigma_)
    for i in range(0, num_cells):
        for j in range(0, num_gauss_points):
            j_[i, j] = -K(T_[i, j]) * sigma_[i, j]
    return j_.reshape(-1)

# %% [markdown]
# Because we also wish to assemble the Jacobian we will also require
# implementations of the derivative
# \begin{equation*}
# D_T [\boldsymbol{q}]\lbrace \hat{T} \rbrace =
# [Bk(T)k(T)\boldsymbol{\sigma}(T)] \hat{T}
# \end{equation*}
# %%

def dqdT(T, sigma):
    T_ = T.reshape((num_cells, num_gauss_points))
    sigma_ = sigma.reshape((num_cells, num_gauss_points, 2))
    djdT = np.empty_like(sigma_)

    for i in range(0, num_cells):
        for j in range(0, num_gauss_points):
            djdT[i, j] = B * K(T_[i, j]) ** 2 * sigma_[i, j]
    return djdT.reshape(-1)

# %% [markdown]
# and the derivative
# \begin{equation*}
# D_{\boldsymbol{\sigma}}[\boldsymbol{q}] \lbrace \nabla \hat{T} \rbrace =
# [-k(T) \boldsymbol{I}] \cdot \nabla \hat{T}  
# \end{equation*}
# %%

def dqdsigma(T, sigma):
    T_ = T.reshape((num_cells, num_gauss_points))
    djdsigma_ = np.empty((num_cells, num_gauss_points, 2, 2), dtype=PETSc.ScalarType)
    Id = np.eye(2)

    for i in range(0, num_cells):
        for j in range(0, num_gauss_points):
            djdsigma_[i, j] = -K(T_[i, j]) * Id
    return djdsigma_.reshape(-1)

# %% [markdown]
# Note that we do not need to explicitly incorporate the action of the finite
# element trial function $\hat{T}$; it will be handled by DOLFINx during
# assembly.
#
# The final function that the user must define is a higher-order function (a
# function that returns other functions) that takes in a derivative multi-index
# as its only argument and returns the appropriate function from the three
# previous definitions.
# %%

def q(derivatives):
    if derivatives == (0, 0):
        return q_
    elif derivatives == (1, 0):
        return dqdT
    elif derivatives == (0, 1):
        return dqdsigma
    else:
        return NotImplementedError

# %% [markdown]
# We can now attach the implementation of the external function `q` to our
# `FEMExternalOperator` symbolic object `q_`.
# %%
q_.external_function = q

# %% [markdown]
# ### Jacobian
# We can now use UFL's built in `derivative` method to derive the Jacobian
# automatically.
# %%
T_hat = TrialFunction(V)
J = derivative(F, T, T_hat)

# %% [markdown]
# To apply the chain rule and obtain something symbolically similar to
# \begin{equation*}
# J(T; \hat{T}, \tilde{T}) = D_T [\boldsymbol{q}]\lbrace \hat{T} \rbrace +
# D_{\boldsymbol{\sigma}}[\boldsymbol{q}] \lbrace \nabla \hat{T} \rbrace \\
# \end{equation*}
# we apply UFL's derivative expansion algorithm.
# %%
J_expanded = ufl.algorithms.expand_derivatives(J)

# %% [markdown]
# In order to assemble `F` and `J` we must apply a further transformation which
# replaces the UFL external operators in the forms with their associated
# `fem.Function`.
# %%
F_replaced, F_external_operators = replace_external_operators(F)
J_replaced, J_external_operators = replace_external_operators(J_expanded)

print(J_replaced)
