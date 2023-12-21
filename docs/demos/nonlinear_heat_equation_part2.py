# %% [markdown]
# # More complex example
#
# In this notebook, we assemble the Jacobian and residual of a steady-state heat
# equation with an external operator to be used to define a non-linear flux law.
#
# To keep the concepts simple we do not solve the non-linear problem, which we
# leave to a subsequent demo.
#
# In this tutorial you will learn how to:
#
# - define a UFL form including an `FEMExternalOperator` which symbolically
#   represents an external operator,
# - define the concrete external definition operator using `numpy` and
#   functional programming techniques, and then attach it to the symbolic
#   `FEMExternalOperator`,
# - and assemble the Jacobian and residual operators that can be used inside a
#   linear or non-linear solver.
#
# We assume some basic familiarity with finite element methods, non-linear
# variational problems and FEniCS.
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
# $$ (eq_1)
#
# where the semi-colon denotes the split between arguments in which the form is
# non-linear (on the left) and linear (on the right).
#
# To solve the nonlinear equation {eq}`eq_1` we apply Newton's method which
# requires the computation of Jacobian, or the Gateaux derivative of $F$.
#
# \begin{equation*}
#   J(T; \hat{T}, \tilde{T}) := D_{T} [ F(T; \tilde{T}) ] \lbrace \hat{T} \rbrace := \int D_T[\boldsymbol{q}(T, \nabla T)] \lbrace \hat{T} \rbrace \cdot \nabla \tilde{T} \; \mathrm{d}x
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
# and an external implementation using `numpy`.
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
from petsc4py import PETSc

import numpy as np

import basix
import ufl
import ufl.algorithms
from dolfinx import fem, mesh
from dolfinx_external_operator import FEMExternalOperator, replace_external_operators
from dolfinx_external_operator.external_operator import evaluate_external_operators, evaluate_operands
from ufl import Measure, TestFunction, TrialFunction, derivative, grad, inner

domain = mesh.create_unit_square(MPI.COMM_WORLD, 1, 1)
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
Qe = basix.ufl.quadrature_element(
    domain.topology.cell_name(), degree=quadrature_degree, value_shape=(2,))
Q = fem.functionspace(domain, Qe)
dx = Measure("dx", metadata={
             "quadrature_scheme": "default", "quadrature_degree": quadrature_degree})

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
    q = np.empty_like(sigma_)

    # TODO: Rewrite using numpy vectorised operation?
    # Loop over cells
    for i in range(0, num_cells):
        # Loop over interpolation points in cell
        for j in range(0, sigma_.shape[1]):
            q[i, j] = -k(T[i, j]) * sigma_[i, j]
    # The output must be returned flattened to one dimension
    return q.reshape(-1)

# %% [markdown]
# Because we also wish to assemble the Jacobian we will also require
# implementations of the left part of the derivative
# \begin{equation*}
# D_T [\boldsymbol{q}]\lbrace \hat{T} \rbrace =
# [Bk(T)k(T)\boldsymbol{\sigma}(T)] \hat{T}
# \end{equation*}

# %%


def dqdT_impl(T, sigma):
    num_cells = T.shape[0]
    sigma_ = sigma.reshape((num_cells, -1, gdim))
    dqdT = np.empty_like(sigma_)

    for i in range(0, num_cells):
        for j in range(0, T.shape[1]):
            dqdT[i, j] = B * k(T[i, j]) ** 2 * sigma_[i, j]
    return dqdT.reshape(-1)

# %% [markdown]
# and the left part of the derivative
# \begin{equation*}
# D_{\boldsymbol{\sigma}}[\boldsymbol{q}] \lbrace \nabla \hat{T} \rbrace =
# [-k(T) \boldsymbol{I}] \cdot \nabla \hat{T}
# \end{equation*}

# %%


def dqdsigma_impl(T, sigma):
    num_cells = T.shape[0]
    dqdsigma_ = np.empty(
        (num_cells, T.shape[1], gdim, gdim), dtype=PETSc.ScalarType)

    Id = np.eye(2)
    for i in range(0, num_cells):
        for j in range(0, T.shape[1]):
            dqdsigma_[i, j] = -k(T[i, j]) * Id
    return dqdsigma_.reshape(-1)

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
        return NotImplementedError

# %% [markdown]
# We can now attach the implementation of the external function `q` to our
# `FEMExternalOperator` symbolic object `q_`.


# %%
q_.external_function = q_external

# %% [markdown]
# ### Jacobian
# We can now use UFL's built-in `derivative` method to derive the Jacobian
# automatically.

# %%
T_hat = TrialFunction(V)
J = derivative(F, T, T_hat)

# %% [markdown]
# ### Transformations
# TODO: Explain the motivation (?) why we need to replace and not just assemble the form.
# To apply the chain rule and obtain a new form symbolically equivalent to
#
# \begin{equation*}
#     J(T; \hat{T}, \tilde{T}) = \int (D_T [\boldsymbol{q}]\lbrace \hat{T} \rbrace +
# D_{\boldsymbol{\sigma}}[\boldsymbol{q}] \lbrace \nabla \hat{T} \rbrace) \cdot \nabla \tilde{T} \; \mathrm{d}x \\
# \end{equation*}
#
# and which can be assembled via DOLFINx, we apply UFL's derivative expansion
# algorithm. This algorithm is aware of the `FEMExternalOperator` semantics and
# the chain rule, and creates a new form containing new `FEMExternalOperator`
# objects associated with the terms $D_T [\boldsymbol{q}]\lbrace \hat{T} \rbrace$
# and $D_{\boldsymbol{\sigma}}[\boldsymbol{q}] \lbrace \nabla \hat{T} \rbrace$.

# %%
J_expanded = ufl.algorithms.expand_derivatives(J)

# %% [markdown]
# In order to assemble `F` and `J` we must apply a further transformation that
# replaces the `FEMExternalOperator` in the forms with their owned `fem.Function`,
# which are accessible through `ref_coefficient` attribute of the
# `FEMExternalOperator` object.

# %%
F_replaced, F_external_operators = replace_external_operators(F)
J_replaced, J_external_operators = replace_external_operators(J_expanded)

# %% [markdown]
# ```{note}
# `*_replaced` contain standard `ufl.Form` objects mathematically similar to
# `F` and `J_expanded` but with `FEMExternalOperators` replaced with the
# `fem.Function` associated with the `FEMExternalOperator`.
# `*_external_operators` are lists of the `FEMExternalOperator` objects found
# `F` and `J_expanded`.
# ```
#
# ### Assembly
# We can now proceed with the finite element assembly in three key steps.
# 1. Evaluate the operands (`T` and `sigma`) associated with the
# `FEMExternalOperator`(s) on the quadrature space `Q`.

# %%
evaluated_operands = evaluate_operands(F_external_operators)

# %% [markdown]
# ```{note}
# `evaluated_operands` represents a map between operands `ufl.Expr` and their
# evaluations stored in `np.ndarray`-s.
# ```

# %% [markdown]
# 2a. Using the evaluated operands, evaluate the external operators in
# `F_external_operators` and assemble the result into the `fem.Function` object
# in `F_replaced`. This calls `q_impl` defined above.

# %%
evaluate_external_operators(F_external_operators, evaluated_operands)

# %% [markdown]
# 2b. Using the evaluated operands, evaluate the external operators in
# `J_external_operators` and assemble the results into `fem.Function` objects
# in `J_replaced`. This calls `dqdT_impl` and `dqdsigma_impl` defined above.

# %%
evaluate_external_operators(J_external_operators, evaluated_operands)

# %% [markdown]
# ```{note}
# Because all external operators share the same operands we can reuse
# the map `evaluated_operands`.
# ```
# 3. The finite element forms can be assembled using the standard DOLFINx
# assembly routines.

# %%
F_compiled = fem.form(F_replaced)
J_compiled = fem.form(J_replaced)
b_vector = fem.assemble_vector(F_compiled)
A_matrix = fem.assemble_matrix(J_compiled)

# %% [markdown]
# ### Comparison with pure UFL
# This output of the external operator approach can be directly checked against
# a pure UFL implementation. Firstly the residual

# %%
k_explicit = 1.0 / (A + B * T)
q_explicit = -k_explicit * sigma
F_explicit = inner(q_explicit, grad(T_tilde)) * dx
F_explicit_compiled = fem.form(F_explicit)
b_explicit_vector = fem.assemble_vector(F_explicit_compiled)
assert np.allclose(b_explicit_vector.array, b_vector.array)

# %% [markdown]
# and then the Jacobian

# %%
J_explicit = ufl.derivative(F_explicit, T, T_hat)
J_explicit_compiled = fem.form(J_explicit)
A_explicit_matrix = fem.assemble_matrix(J_explicit_compiled)
assert np.allclose(A_explicit_matrix.to_dense(), A_matrix.to_dense())

# %% [markdown]
# and a hand-derived Jacobian

# %%
J_manual = (
    inner(B * k_explicit**2 * sigma * T_hat, grad(T_tilde)) * dx
    + inner(-k_explicit * ufl.Identity(2) * grad(T_hat), grad(T_tilde)) * dx
)
J_manual_compiled = fem.form(J_manual)
A_manual_matrix = fem.assemble_matrix(J_manual_compiled)
assert np.allclose(A_manual_matrix.to_dense(), A_matrix.to_dense())
