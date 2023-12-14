# %% [markdown]
# # Simple example: implementation
#
# Authors: Andrey Latyshev (University of Luxembourg, Sorbonne Université, andrey.latyshev@uni.lu)
#
# In this notebook, we implement the simple example of the external operator $N
# = N(u)$ with one operand in the FEniCSx environment. In the following
# paragraphs, you may find a description of the interface of our framework.
# Then, in the next cells, you will find the implementation of the example.
#
# ## Describing the implementation
#
# The external operator $N$ is defined through the new UFL-like object
# `femExternalOperator` developed as a part of this framework. The object
# combines features of symbolic UFL objects and a globally allocated
# `fem.Function` coefficient holding a vector with degrees-of-freedom. In other
# words, we can natively use the `femExternalOperator` object within the
# symbolic representation of finite element forms, evaluate it in a certain
# finite element space and store the values. The latter is accessible through
# the `ref_coefficient` attribute of the `femExternalOperator` object.
#
# Once a form containing the external operator is defined it can be
# successfully differentiated using the UFL function `expand_derivatives`. The
# latter recursively applies the automatic differentiation (AD) feature of UFL
# to the form and calculates the Gateau derivative $\frac{d N}{d u}$ of the
# external operator according to the chain rule. During this procedure, the
# algorithm creates a new `femExternalOperator` object representing the
# derivative $\frac{d N}{d u}$. It preserves an appropriate local shape of the
# newly created symbolic object (the `ufl_shape` attribute) and changes the
# multi-index `derivatives`, one more attribute of `femExternalOperator`. This
# multi-index helps to define the operand with respect to which the derivative
# was taken.
#
# The behaviour of both external operator $N$ and its derivative $\frac{d N}{d
# u}$ is managed by a callable Python function (the `external_function`
# argument of the `femExternalOperator` constructor). It must be defined by the
# user and contain subfunctions computing values of the external operator and
# its derivative respectively. The signature is the same for all subfunctions
# and consists of `ndarray` (Numpy-like) array representing the
# degrees-of-freedom of the operand $u$. The framework uses the multi-index
# `derivatives` of the `femExternalOperator` object to pick up an appropriate
# subfunction evaluating values of either $N$ or $\frac{d N}{d u}$.
#
# Once both linear and bilinear forms are defined we can apply the framework
# function `replace_external_operators`. It replaces `femExternalOperator`
# objects in a form with their finite element representatives, the
# `ref_coefficient` variable. The new "replaced" forms can be easily assembled
# using the standard functionality of DOLFINx.
#
# Throughout a FEM simulation, we often need to re-evaluate operators presented
# in linear and bilinear forms, e.g. in the Newton method. As operands of
# external operators may be any UFL expressions, first of all, we must evaluate
# them in quadratures using the framework function `evaluate_operands`. Only
# then, using newly calculated values of the operands, we evaluate external
# operands via the framework function `evaluate_external_operators`. It updates
# the values of the `ref_coefficient` field of each `femExternalOperator`
# object.
#
# As a result, our framework provides DOLFINx users with a very simple
# interface allowing to use
#
# As a summary, in order to solve a variational problem using external operator the user needs just:
# 1. Choose appropriate operands of the external operator, e.g. the
# `fem.Function` variable $u$ in our case.
# 2. Define callable Python functions that evaluate the external operator and
# its derivatives by manipulating the degrees-of-freedom of the operands as
# `ndarray` data.
# 3. Define external operator through the class `femExternalOperator`.
# 4.
#
# For the sake of simplicity, we chose the following definition of the external operator $N$
#
# $$
#     N(u) = u^2,
# $$
#
# which is easily expressible via UFL, in order to show that our framework
# works correctly on this simple example.

# %% [markdown]
# ## Preamble
#
# Here we import the required Python packages, build a simple square mesh and
# define the finite element functional space $V$, where the main variable $u$,
# test and trial functions exist.

# %%
from mpi4py import MPI
from petsc4py import PETSc

import basix
import ufl
from dolfinx import fem, mesh
import dolfinx.fem.petsc  # there is an error without it, why?

import numpy as np

from dolfinx_external_operator import FEMExternalOperator, replace_external_operators, evaluate_external_operators, evaluate_operands

nx = 2
domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, nx)

gdim = domain.geometry.dim
V = fem.functionspace(domain, ("CG", 1))
v = ufl.TestFunction(V)

u_hat = ufl.TrialFunction(V)
u = fem.Function(V)
u.x.array[:] = 2.0  # in order to get non-zero forms after assembling

# %% [markdown]
# ## Defining the external operator

# %% [markdown]
# According to the current implementation of the framework, the finite element
# representation of the external operator must be defined in a quadrature
# finite element space.

# %%
Qe = basix.ufl.quadrature_element(domain.topology.cell_name(), degree=1)
Q = dolfinx.fem.functionspace(domain, Qe)
dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 1, "quadrature_scheme": "default"})

# %% [markdown]
# Now we define the behaviour of the external operator $N = N(u)$ and its
# Gateau derivative $\frac{dN}{du}(u; \hat{u})$ in subfunctions `N` and `dNdu`
# respectively.

# %%


def N_external(u):
    return np.reshape(u**2, -1)


def dNdu_external(u):
    return np.reshape(2 * u, -1)


# %% [markdown]
# Then we combine both subfunctions `func_N` and `func_dNdu` in the function
# `f_external` that distinguishes the external operator and its derivatives
# using the multi-index `derivatives`, the attribute of every external
# operator.

# %%


def Ns_external(derivatives):
    if derivatives == (0,):
        return N_external
    elif derivatives == (1,):
        return dNdu_external
    else:
        return NotImplementedError


# %% [markdown]
# Now we have all the ingredients to define the external operator $N$.


# %%
N = FEMExternalOperator(u, function_space=Q, external_function=Ns_external)

# %% [markdown]
# ## Defining the linear and bilinear forms

# %% [markdown]
# Thanks to the automatic differentiation tool of UFL we do not need to compute
# manually the derivative of the form $F$ and the operator $N$. We just have to
# define the linear form and the direction with respect to which the
# differentiation will be performed.

# %%
F = N * v * dx
J = ufl.derivative(F, u, u_hat)

# %% [markdown]
# Our framework recursively looks for `femExternalOperator` objects in the form
# and replaces them with their representatives, the globally allocated
# coefficients.

# %%
F_replaced, F_ex_ops_list = replace_external_operators(F)
F_dolfinx = fem.form(F_replaced)

# %%
J_expanded = ufl.algorithms.expand_derivatives(J)
J_replaced, J_ex_ops_list = replace_external_operators(J_expanded)
J_dolfinx = fem.form(J_replaced)

# %% [markdown]
# ```{note}
# As `femExternalOperator` combines properties of both UFL-like symbolic
# objects and finite objects `fem.Function`, the user must be aware that once
# the expansion of the derivatives is performed, the framework creates new
# functional spaces for shapes of derivatives of external operators and
# allocates memory for the appropriate coefficients. Thus, `expand_derivatives`
# may lead to "hidden" memory allocations, which may not be exptected by UFL
# users.
# ```
#
# Now the forms `F_replaced` and `J_replaced` do not contain any UFL-like
# `femExternalOperator` objects. Instead, they consist of `fem.Function`
# objects defined on the quadrature space `Q`, the `ref_coefficient` attribute
# of `femExternalOperator`.

# %% [markdown]
# Then we need to update the operand values of the external operator $N$, i.e.
# the field $u$. In this case, we just project the values of variable $u$ from
# the functional space $V$ onto the quadrature space $Q$.

# %%
evaluated_operands = evaluate_operands(F_ex_ops_list)

# %% [markdown]
# As the operands of an external operand and its derivatives are the same, we
# evaluate them only once and send their values to the function
# `evaluate_external_operators`. The latter evaluates external operators and
# updates values of the `ref_cofficient` contained in provided lists of
# `femExternalOperator` of a form and values of evaluated operands stored in
# numpy-like arrays.

# %%
evaluate_external_operators(F_ex_ops_list, evaluated_operands)
evaluate_external_operators(J_ex_ops_list, evaluated_operands)

# %% [markdown]
# The algorithm exploits the `f_external` to update coefficients representing
# external operators.

# %% [markdown]
# Concrete values of external operators can be easily accessible through the
# `ref_coefficient` attribute.
