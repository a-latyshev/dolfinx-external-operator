# %% [markdown]
# # Non-linear heat equation
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
# - define the concrete external definition operator using `Numpy` and
#   functional programming techniques, and then attach it to the symbolic
#   `FEMExternalOperator`,
# - and assemble the Jacobian and residual operators that can be used inside a
#   linear or non-linear solver.
#
# We assume some basic familiarity with finite element methods, non-linear
# variational problems and FEniCS.
#
# ## Recall
#
# TODO: Remind the definition of external operators, operands, arguments.
#
# ## Problem formulation
#
# Denoting the temperature field $T$ and its gradient $\boldsymbol{\sigma} :=
# \nabla T$ we consider the following system on the square domain $\Omega :=
# [0, 1]^2$ with boundary $\partial \Omega$
#
# \begin{align*}
#      \nabla \cdot \boldsymbol{q} &= f \quad \mathrm{on} \; \Omega, \\
#      \boldsymbol{q}(T) &= -k(T) \nabla T, \\
# \end{align*}
#
# where $f$ is a given function. With flux $\boldsymbol{q} =
# -k\boldsymbol{\sigma}$ for the thermal conductivity $k = \mathrm{const}$ we
# recover the standard Fourier heat problem. However, here we will assume that $k$
# is some general function of $T$ that we would like to specify using some
# external (non-UFL) piece of code.
#
# Let $V = H^1_0(\Omega)$ be the usual Sobolev space of square-integrable
# functions with square-integrable derivatives and vanishing value on the
# boundary $\partial \Omega$. Then in a variational setting, the problem can be
# written in residual form as find $T \in V$ such that
#
# $$
#   F(T; \tilde{T}) = - \int k(T) \nabla T \cdot \nabla \tilde{T} - f \cdot
#   \tilde{T} \; \mathrm{d}x = 0 \quad \forall \tilde{T} \in V,
# $$ (eq_1)
#
# where the semi-colon denotes the split between operands (on the left) and
# arguments (on the right) in which the form is non-linear and linear respectively.
#
# To solve the nonlinear equation {eq}`eq_1` we apply Newton's method which
# requires the computation of Jacobian, or the Gateaux derivative of $F$.
#
# \begin{equation*}
#   J(T; \hat{T}, \tilde{T}) := D_{T} [ F(T; \tilde{T}) ] \lbrace \hat{T} \rbrace := -\int D_T[k(T) \nabla T] \lbrace \hat{T} \rbrace \cdot \nabla \tilde{T} \; \mathrm{d}x
# \end{equation*}
#
# Now we apply the chain rule to write
#
# \begin{align*}
#   D_{T}[k \nabla T]\lbrace \hat{T} \rbrace &= D_T [k]\lbrace
#   D_T[T]\lbrace \hat{T} \rbrace \rbrace\nabla T +
#   k(T)
#   D_T[\nabla T]\lbrace \hat{T} \rbrace \\
#   &= D_T [k]\lbrace \hat{T} \rbrace \nabla T +
#   [k(T) \boldsymbol{I}] \cdot \nabla \hat{T},  \\
# \end{align*}
# where $\boldsymbol{I}$ is the 2x2 identity matrix.
#
# To fix ideas, we now assume the following explicit form for the material
# conductivity
# \begin{equation*}
#   k(T) = \frac{1}{A + BT}
# \end{equation*}
# where $A$ and $B$ are material constants. After some algebra we can derive
# \begin{equation*}
#   D_T [k]\lbrace \hat{T} \rbrace =
#   [-Bk^2(T)] \hat{T}
# \end{equation*}
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

import numpy as np

import basix
import ufl
import ufl.algorithms
from dolfinx import fem, mesh
from dolfinx_external_operator import FEMExternalOperator, replace_external_operators
from dolfinx_external_operator.external_operator import evaluate_external_operators, evaluate_operands
from ufl import Identity, Measure, TestFunction, TrialFunction, derivative, grad, inner

domain = mesh.create_unit_square(MPI.COMM_WORLD, 1, 1)
V = fem.functionspace(domain, ("CG", 1))

# %% [markdown]
# ### Defining the external operator
# We will begin by defining the input *operand* $T$ needed to specify the external
# operator $k(T)$. Operands of a `FEMExternalOperator` can be any `ufl.Expr`
# object.

# %%
T = fem.Function(V)

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
# operator $k$ will live. For optimal convergence, $k$
# must be evaluated directly at the Gauss points used in the integration of the
# weak form. This can be enforced by constructing a quadrature function space
# and an integration measure `dx` using the same rule.

# %%
quadrature_degree = 2
Qe = basix.ufl.quadrature_element(
    domain.topology.cell_name(), degree=quadrature_degree, value_shape=())
Q = fem.functionspace(domain, Qe)
dx = Measure("dx", metadata={
             "quadrature_scheme": "default", "quadrature_degree": quadrature_degree})

# %% [markdown]
# We can create the external operator $k$.

# %%
k = FEMExternalOperator(T, function_space=Q)

# %% [markdown]
# Note that at this stage the object `k` is symbolic and we have not defined the
# `numpy` code to compute it. This will be done later in the example.
# ```{note}
# `FEMExternalOperator` holds the `ref_coefficient` (a `fem.Function`) attribute to store its evaluation.
# ```

# %% [markdown]
# ### Residual
# The external operator can be used in the definition of the residual $F$.

# %%
T_tilde = TestFunction(V)
F = inner(-k*grad(T), grad(T_tilde)) * dx

# %% [markdown]
# ### Implementing the external operator
# The symbolic `FEMExternalOperator` is linked to its implementation using
# functional programming techniques. This approach is similar to how the
# `interpolate` function works in DOLFINx.
#
# In the first step, the user must define Python function(s) that accept
# `np.ndarray` containing values of the evaluated operands (here, $T$) at the
# global interpolation points associated with the output function space. These
# function(s) must return an `np.ndarray` object containing the evaluation of the
# external operator at all of the global interpolation points. We discuss the
# sizing of these arrays directly in the code below.
#
# We begin by defining the Python function evaluating $k$, so here we recall
# \begin{align*}
#     k(T) &= \frac{1}{A + BT}.
# \end{align*}

# %%
A = 1.0
B = 1.0
gdim = domain.geometry.dim


def k_impl(T):
    # The input T is a `np.ndarray` and has the shape
    # `(num_cells, num_interpolation_points_per_cell)` TODO: Is it always the case??
    output = 1.0 / (A + B * T)
    # The output must be returned flattened to one dimension
    return output.reshape(-1)

# %% [markdown]
# Because we also wish to assemble the Jacobian we will also require
# implementations of the left part of the derivative
# \begin{equation*}
#     D_{T}[k(T)] \lbrace \hat{T} \rbrace =
#     [-Bk^2(T)] \cdot \hat{T}
# \end{equation*}

# %%


def dkdT_impl(T):
    return -B * k_impl(T)**2

# %% [markdown]
# Note that we do not need to explicitly incorporate the action of the finite
# element trial $\tilde{T}$ or test functions $\hat{T}$; it will be handled by
# DOLFINx during assembly.
#
# The final function that the user must define is a higher-order function (a
# function that returns other functions defining the behaviour of the external
# operator and its derivative) that takes in a derivative multi-index as its only
# argument and returns the appropriate function from the two previous definitions.

# %%


def k_external(derivatives):
    if derivatives == (0,):
        return k_impl
    elif derivatives == (1,):
        return dkdT_impl
    else:
        return NotImplementedError

# %% [markdown]
# We can now attach the implementation of the external function `k_external` to our
# `FEMExternalOperator` symbolic object `k`.


# %%
k.external_function = k_external

# %% [markdown]
# ### Jacobian
# We can now use UFL's built-in `derivative` method to derive the Jacobian
# automatically.

# %%
T_hat = TrialFunction(V)
J = derivative(F, T, T_hat)

# %% [markdown]
# ### Transformations
# TODO: Explain the motivation (?) why the user needs to replace and not just assemble the form.
#
# To apply the chain rule and obtain a new form symbolically equivalent to
#
# \begin{equation*}
#     J(T; \hat{T}, \tilde{T}) = \int (D_T [\boldsymbol{q}]\lbrace \hat{T} \rbrace +
#     D_{\boldsymbol{\sigma}}[\boldsymbol{q}] \lbrace \nabla \hat{T} \rbrace) \cdot \nabla \tilde{T} \; \mathrm{d}x \\
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
q_explicit = -k_explicit * grad(T)
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
    inner(B * k_explicit**2 * grad(T) * T_hat, grad(T_tilde)) * dx +
    inner(-k_explicit * Identity(2) * grad(T_hat) , grad(T_tilde)) * dx
)
J_manual_compiled = fem.form(J_manual)
A_manual_matrix = fem.assemble_matrix(J_manual_compiled)
assert np.allclose(A_manual_matrix.to_dense(), A_matrix.to_dense())
