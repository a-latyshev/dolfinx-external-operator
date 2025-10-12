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
#     display_name: dolfinx-env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # More complex case
#
# In this notebook, we alter the previous implementation by wrapping a more complex
# expression through an external operator. The objective remains the same: to
# assemble the Jacobian and residual of a steady-state heat equation with an
# external operator to be used to define a non-linear flux law.
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
# \end{align*}
# where $f$ is a given function. With flux $\boldsymbol{q} = -k
# \boldsymbol{\sigma}$ for $k = \mathrm{const}$ we recover the standard Fourier
# heat problem. However, here we will assume that the $\boldsymbol{q}$ is some
# general function of $T$ and $\boldsymbol{\sigma}(T)$ that we would like to
# specify $\boldsymbol{q}$ using some external (non-UFL) piece of code.
#
# Let $V = H^1_0(\Omega)$ be the usual Sobolev space of square-integrable
# functions with square-integrable derivatives and vanishing value on the
# boundary $\partial \Omega$. Then in a variational setting, the problem can be
# written in residual form as find $T \in V$ such that
#
# $$
#   F(T; \tilde{T}) = \int \boldsymbol{q}(T, \nabla T) \cdot \nabla \tilde{T} - f \cdot
#   \tilde{T} \; \mathrm{d}x = 0 \quad \forall \tilde{T} \in V,
# $$ (eq_2)
#
# where the semi-colon denotes the split between arguments in which the form is
# non-linear (on the left) and linear (on the right).
#
# To solve the nonlinear equation {eq}`eq_2` we apply Newton's method which
# requires the computation of Jacobian, or the Gateaux derivative of $F$.
#
# \begin{equation*}
#   J(T; \hat{T}, \tilde{T})
#   := D_{T} [ F(T; \tilde{T}) ] \lbrace \hat{T} \rbrace
#   := \int D_T[\boldsymbol{q}(T, \nabla T)] \lbrace \hat{T} \rbrace \cdot \nabla \tilde{T} \; \mathrm{d}x
# \end{equation*}
#
# ```{note}
# The above result uses the product rule $D_{x}(fg)\lbrace \hat{x} \rbrace =
# (D_x(f)\lbrace \hat{x} \rbrace) g + f(D_x(g)\lbrace \hat{x} \rbrace)$ and
# that the Gateaux derivative and integral can be exchanged safely.
# ```
#
# Dropping the explicit dependence of $\boldsymbol{q}$ on $T$ and $\nabla T$
# for notational convenience we can use the chain rule to write
# \begin{align*}
#   D_{T}[\boldsymbol{q}]\lbrace \hat{T} \rbrace &= D_T [\boldsymbol{q}]\lbrace
#   D_T[T]\lbrace \hat{T} \rbrace \rbrace +
#   D_{\boldsymbol{\sigma}}[\boldsymbol{q}] \lbrace
#   D_T[\boldsymbol{\sigma}]\lbrace \hat{T} \rbrace \rbrace \\
#   &= D_T [\boldsymbol{q}]\lbrace \hat{T} \rbrace +
#   D_{\boldsymbol{\sigma}}[\boldsymbol{q}] \lbrace \nabla \hat{T} \rbrace \\
# \end{align*}
#
# To fix ideas, we now assume the following explicit form for the material
# conductivity
# \begin{equation*}
#   k(T) = \frac{1}{A + BT}
# \end{equation*}
# where $A$ and $B$ are material constants. After some algebra we can derive
# \begin{align*}
#   D_T [\boldsymbol{q}]\lbrace \hat{T} \rbrace &=
#   [Bk^2(T)\boldsymbol{\sigma}(T)] \hat{T} \\
#   D_{\boldsymbol{\sigma}}[\boldsymbol{q}] \lbrace \nabla \hat{T} \rbrace &=
#   [-k(T) \boldsymbol{I}] \cdot \nabla \hat{T}
# \end{align*}
# We now proceed to the definition of residual and Jacobian of this problem
# where $\boldsymbol{q}$ will be defined using the `FEMExternalOperator` approach
# and an external implementation using `NumPy`.
# ```{note}
# This simple model can also be implemented in pure UFL and the Jacobian
# derived symbolically using UFL's `derivative` function.
# ```
#
# ## Implementation
#
# ### Preamble
#
# We import from the required packages, create a mesh, and a scalar-valued
# function space which we will use to discretise the temperature field $T$.

# %%

from mpi4py import MPI

import numpy as np

import basix
import ufl
import ufl.algorithms
from dolfinx import fem, mesh
from dolfinx_external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    evaluate_operands,
    replace_external_operators,
)
from ufl import Measure, TestFunction, TrialFunction, derivative, grad, inner

domain = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
V = fem.functionspace(domain, ("CG", 1))

# %% [markdown]
# ### Defining the external operator
# We will begin by defining the input *operands* $T$ and $\boldsymbol{\sigma}$
# needed to specify the external operator $\boldsymbol{q}(T,
# \boldsymbol{\sigma}(T))$. The operands of an `FEMExternalOperator` can be any
# `ufl.Expr` object.

# %%
T = fem.Function(V)
sigma = grad(T)

# %% [markdown]
# To start the Newton method we require non-zero assembled residual and Jacobian,
# thus we initialize the variable `T` with the following non-zero function
#
# $$
#     T = x^2 + y
# $$

# %%
T.interpolate(lambda x: x[0] ** 2 + x[1])

# %% [markdown]
# We also need to define a `fem.FunctionSpace` in which the output of the external
# operator $\boldsymbol{q}$ will live. For optimal convergence, $\boldsymbol{q}$
# must be evaluated directly at the Gauss points used in the integration of the
# weak form. This can be enforced by constructing a quadrature function space
# and an integration measure `dx` using the same rule.

# %%
quadrature_degree = 2
Qe = basix.ufl.quadrature_element(domain.topology.cell_name(), degree=quadrature_degree, value_shape=(2,))
Q = fem.functionspace(domain, Qe)
dx = Measure("dx", metadata={"quadrature_scheme": "default", "quadrature_degree": quadrature_degree})

# %% [markdown]
# We now have all of the ingredients to define the external operator.

# %%
q_ = FEMExternalOperator(T, sigma, function_space=Q)

# %% [markdown]
# Note that at this stage the `q_` is symbolic and we have not defined the
# `numpy` code to compute it. This will be done later in the example.
# ```{note}
# `FEMExternalOperator` holds a `fem.Function` to store its evaluation.
# ```

# %% [markdown]
# ### Residual
# The external operator can be used in the definition of the residual $F$.

# %%
T_tilde = TestFunction(V)
F = inner(q_, grad(T_tilde)) * dx

# %% [markdown]
# ### Implementing the external operator
# The symbolic `FEMExternalOperator` is linked to its implementation using
# functional programming techniques. This approach is similar to how the
# `interpolate` function works in DOLFINx.
#
# In the first step, the user must define Python functions(s) that accept
# `np.ndarray` containing the evaluated operands (here, $T$ and
# $\boldsymbol{\sigma}$) at the global interpolation points associated with the
# output function space. These Function(s) must return an `np.ndarray` object
# containing the evaluation of the external operator at all of the global
# interpolation points. We discuss the sizing of these arrays directly in the
# code below.
#
# We begin by defining the Python functions for the left part of
# \begin{equation*}
#     [\boldsymbol{q}(T, \nabla T)] \cdot \nabla \tilde{T}
# \end{equation*}
# here we recall
# \begin{align*}
#     \boldsymbol{q} &= -k(T) \boldsymbol{\sigma}, \\
#     k(T) &= \frac{1}{A + BT}
# \end{align*}

# %%
A = 1.0
B = 1.0
Id = np.eye(2)
gdim = domain.geometry.dim


def k(T):
    return 1.0 / (A + B * T)


def q_impl(T, sigma):
    # T has shape `(num_cells, num_interpolation_points_per_cell)`
    num_cells = T.shape[0]
    # sigma has shape `(num_cells, num_interpolation_points_per_cell*value_shape)`
    # We reshape `sigma` to have shape `(num_cells,
    # num_interpolation_points_per_cell, np.prod(value_shape))`
    sigma_ = sigma.reshape((num_cells, -1, gdim))
    # Array for output with shape `(num_cells,
    # num_interpolation_points_per_cell, np.prod(value_shape))`
    output = -k(T)[:, :, np.newaxis] * sigma_
    # The output must be returned flattened to one dimension
    return output.reshape(-1)


# %% [markdown]
# Because we also wish to assemble the Jacobian we will also require
# implementations of the left part of the derivative
# \begin{equation*}
#     D_T [\boldsymbol{q}]\lbrace \hat{T} \rbrace =
#     [Bk^2(T)\boldsymbol{\sigma}(T)] \hat{T}
# \end{equation*}


# %%
def dqdT_impl(T, sigma):
    num_cells = T.shape[0]
    sigma_ = sigma.reshape((num_cells, -1, gdim))
    output = B * (k(T) ** 2)[:, :, np.newaxis] * sigma_
    return output.reshape(-1)


# %% [markdown]
# and the left part of the derivative
# \begin{equation*}
# D_{\boldsymbol{\sigma}}[\boldsymbol{q}] \lbrace \nabla \hat{T} \rbrace =
# [-k(T) \boldsymbol{I}] \cdot \nabla \hat{T}
# \end{equation*}


# %%
def dqdsigma_impl(T, sigma):
    output = -k(T)[:, :, np.newaxis, np.newaxis] * Id[np.newaxis, np.newaxis, :, :]
    return output.reshape(-1)


# %% [markdown]
# Note that we do not need to explicitly incorporate the action of the finite
# element trial $\tilde{T}$ or test functions $\hat{T}$; it will be handled by
# DOLFINx during assembly.
#
# The final function that the user must define is a higher-order function (a
# function that returns other functions) that takes in a derivative multi-index
# as its only argument and returns the appropriate function from the three
# previous definitions.


# %%
def q_external(derivatives):
    if derivatives == (0, 0):
        return q_impl
    elif derivatives == (1, 0):
        return dqdT_impl
    elif derivatives == (0, 1):
        return dqdsigma_impl
    else:
        raise NotImplementedError(f"No external function is defined for the requested derivative {derivatives}.")


# %% [markdown]
# We can now attach the implementation of the external function `q` to our
# `FEMExternalOperator` symbolic object `q_`.


# %%
q_.external_function = q_external

# %% [markdown]
# ### System assembling
#
# The remaining part of the modeling remains unchanged.

# %%
T_hat = TrialFunction(V)
J = derivative(F, T, T_hat)
J_expanded = ufl.algorithms.expand_derivatives(J)
F_replaced, F_external_operators = replace_external_operators(F)
J_replaced, J_external_operators = replace_external_operators(J_expanded)
evaluated_operands = evaluate_operands(F_external_operators)
_ = evaluate_external_operators(F_external_operators, evaluated_operands)

_ = evaluate_external_operators(J_external_operators, evaluated_operands)
F_compiled = fem.form(F_replaced)
J_compiled = fem.form(J_replaced)
b_vector = fem.assemble_vector(F_compiled)
A_matrix = fem.assemble_matrix(J_compiled)

k_explicit = 1.0 / (A + B * T)
q_explicit = -k_explicit * sigma
F_explicit = inner(q_explicit, grad(T_tilde)) * dx
F_explicit_compiled = fem.form(F_explicit)
b_explicit_vector = fem.assemble_vector(F_explicit_compiled)
assert np.allclose(b_explicit_vector.array, b_vector.array)

J_explicit = ufl.derivative(F_explicit, T, T_hat)
J_explicit_compiled = fem.form(J_explicit)
A_explicit_matrix = fem.assemble_matrix(J_explicit_compiled)
assert np.allclose(A_explicit_matrix.to_dense(), A_matrix.to_dense())

J_manual = (
    inner(B * k_explicit**2 * sigma * T_hat, grad(T_tilde)) * dx
    + inner(-k_explicit * ufl.Identity(2) * grad(T_hat), grad(T_tilde)) * dx
)
J_manual_compiled = fem.form(J_manual)
A_manual_matrix = fem.assemble_matrix(J_manual_compiled)
assert np.allclose(A_manual_matrix.to_dense(), A_matrix.to_dense())
