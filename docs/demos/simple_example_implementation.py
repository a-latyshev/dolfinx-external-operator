# %% [markdown]
# # Simple example: implementation
#
# Authors: Andrey Latyshev (University of Luxembourg, Sorbonne Université, andrey.latyshev@uni.lu)
#
# In this notebook, we implement the simple example of the external operator $N = N(u)$ with one operand in the FEniCSx environment.
#
# The external operator is defined through the new UFL-like object `femExternalOperator` developed as a part of this framework. The object combines features of symbolic UFL objects, i.e. we can use it natively within the symbolic representation of finite element forms, and a globally allocated `fem.Function` coefficient holding a vector with degrees-of-freedom. The vector data is accessible through the `ref_coefficient` attribute of the `femExternalOperator` object.
#
# Once the form containing an external operator is defined it can be successfully derivated using the UFL function `expand_derivatives`. It recursively applies automatic differentiation of UFL and calculates Gateau derivatives of an external operator according to the chain rule. During this procedure, the algorithm creates new `femExternalOperator` objects representing derivatives of external operators of the original form. It preserves appropriate local shapes of new symbolic objects (the `ufl_shape` attribute) and changes the multi-index `derivatives`, an attribute of `femExternalOperator`. The latter helps to define the direction to which operand the derivative was taken, with respect to
#
# The behaviour of both external operator $N$ and its derivative $\frac{d N}{d u}$ is managed by a callable Python function (the `external_function` argument of the `femExternalOperator` constructor). It must be defined by the user and contain subfunctions computing values of the external operator and its derivative respectively. The signature is the same for all subfunctions and consists of `ndarray` (Numpy-like) array representing the degrees-of-freedom of the operand $u$ of the external operator. The framework uses the multi-index `derivatives` of the `femExternalOperator` object to pick up an appropriate subfunction evaluating, in our case, values of either $N$ or $\frac{d N}{d u}$.
#
# Once both linear and bilinear forms are defined we can apply `replace_external_operators` to them, the function of this framework. It replaces `femExternalOperator` objects with their finite element representatives, the `ref_coefficient`. The new "replaced" forms can be easily assembled and
#
#
# Here below we demonstrate how the described above linear and bilinear forms can be easily defined in UFL using its automatic differentiation tool and the new object `femExternalOperator`.
#
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
from mpi4py import MPI
from petsc4py import PETSc

import basix
import ufl
from dolfinx import fem, mesh
import dolfinx.fem.petsc  # there is an error without it, why?

import numpy as np

import dolfinx_ExternalOperator.external_operator as ex_op_env

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

# %%
