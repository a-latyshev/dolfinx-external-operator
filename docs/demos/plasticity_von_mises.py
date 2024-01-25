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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# %% [markdown]
# # Plasticity of von Mises
#
# This tutorial aims to demonstrate an efficient implementation of the plasticity
# model of von Mises using an external operator defining the elastoplastic
# constitutive relations written with the help of the 3rd party package `Numba`.
# Here we consider a cylinder expansion problem in the two-dimensional case in a
# symmetric formulation.
#
# This tutorial is based on the
# [original implementation](https://comet-fenics.readthedocs.io/en/latest/demo/2D_plasticity/vonMises_plasticity.py.html)
# of the problem via legacy FEniCS 2019 and
# [its extension](https://github.com/a-latyshev/convex-plasticity/tree/main) for
# the modern FEniCSx in the setting of convex optimization. A detailed conclusion
# of the von Mises plastic model in the case of the cylinder expansion problem can
# be found in {cite}`bonnet2014`. Do not hesitate to visit the mentioned sources
# for more information.
#
# We assume the knowledge of the return-mapping procedure, commonly used in the
# solid mechanics community to solve elastoplasticity problems.
#
# ## Notation
#
# Denoting the displacement vector $\boldsymbol{u}$ we define the strain tensor
# $\boldsymbol{\varepsilon}$ as follows
#
# $$
#     \boldsymbol{\varepsilon} = \frac{1}{2}\left( \nabla\boldsymbol{u} + \nabla\boldsymbol{u}^T \right).
# $$
#
# Throughout the tutorial, we stick to the Mandel-Voigt notation, according to
# which the stress tensor $\boldsymbol{\sigma}$ and the strain tensor
# $\boldsymbol{\varepsilon}$ are written as 4-size vectors with the following
# components
#
# \begin{align*}
#     & \boldsymbol{\sigma} = [\sigma_{xx}, \sigma_{yy}, \sigma_{zz}, \sqrt{2}\sigma_{xy}]^T, \\
#     & \boldsymbol{\varepsilon} = [\varepsilon_{xx}, \varepsilon_{yy}, \varepsilon_{zz}, \sqrt{2}\varepsilon_{xy}]^T.
# \end{align*}
#
# Denoting the deviatoric operator $\mathrm{dev}$, we introduce two additional
# quantities of interest: the cumulative plastic strain $p$ and the equivalent
# stress $\sigma_\text{eq}$ defined by the following formulas:
#
# \begin{align*}
#     & p = \sqrt{\frac{2}{3} \boldsymbol{e} . \boldsymbol{e}}, \\
#     & \sigma_\text{eq} = \sqrt{\frac{3}{2}\boldsymbol{s}.\boldsymbol{s}},
# \end{align*}
#
# where $\boldsymbol{e} = \mathrm{dev}\boldsymbol{\varepsilon}$ and
# $\boldsymbol{s} = \mathrm{dev}\boldsymbol{\sigma}$ are deviatoric parts of the
# stain and stress tensors respectively.
#
#
# ## Problem formulation
#
# The domain of the problem $\Omega$ represents the first quarter of the hollow
# cylinder with inner $R_i$ and outer $R_o$ radii, where symmetry conditions are
# set on the left and bottom sides and pressure is set on the inner wall $\partial\Omega_\text{inner}$. The behaviour of cylinder material is defined by the von Mises yield criterion $f$
# with the linear isotropic hardening law {eq}`eq_von_Mises`
#
# $$
#     f(\boldsymbol{\sigma}) = \sigma_\text{eq}(\boldsymbol{\sigma}) - \sigma_0 - Hp \leq 0,
# $$ (eq_von_Mises)
#
# where $\sigma_0$ is a uniaxial strength and $H$ is an isotropic hardening
# modulus, which is defined through the Young modulus $E$ and the tangent elastic
# modulus $E_t = \frac{EH}{E+H}$.
#
# Let V be the functional space of admissible displacement fields. Then, the weak
# formulation of this problem can be written as follows:
#
# Find $\boldsymbol{u} \in V$ such that
# $$
#     F(\boldsymbol{u}; \boldsymbol{v}) = \int\limits_\Omega \boldsymbol{\sigma}(\boldsymbol{u}) . \boldsymbol{\varepsilon(v)} d\boldsymbol{x} - F_\text{ext}(\boldsymbol{v}) = 0, \quad \forall \boldsymbol{v} \in V.
# $$ (eq_main)
#
# The external force $F_{\text{ext}}(\boldsymbol{v})$ represents the pressure inside the cylinder and is written as the following Neumann condition
#
# $$
#     F_\text{ext}(\boldsymbol{v}) = q \int\limits_{\partial\Omega_\text{inner}} \boldsymbol{n} .\boldsymbol{v} d\boldsymbol{x},
# $$
# where the vector $\boldsymbol{n}$ is a normal to the cylinder surface and the loading parameter $q$ is progressively increased from 0 to
# $q_\text{lim} = \frac{2}{\sqrt{3}}\sigma_0\log\left(\frac{R_o}{R_i}\right)$, the
# analytical collapse load for the perfect plasticity model without hardening.
#
# The modelling is performed under assumptions of the plane strain and
# an associative plasticity law.
#
# In the above nonlinear problem {eq}`eq_main` the elastoplastic constitutive
# relation $\boldsymbol{\sigma}(\boldsymbol{u})$ is restored by applying the
# return-mapping procedure. The main bottleneck of this procedure is a computation
# of derivatives of quantities of interest including one of the stress tensor,
# so-called the tangent stiffness matrix $\boldsymbol{C}_\text{tang}$ required for
# the Newton method to solve the nonlinear equation {eq}`eq_main`. The advantage
# of the von Mises model is that the return-mapping procedure may be performed
# analytically, so the derivatives may be expressed explicitly.
#
# In this tutorial, we treat the stress tensor $\boldsymbol{\sigma}$ as an
# external operator acting on the displacement field $\boldsymbol{u}$ and
# represent it through a `FEMExternalOperator` object. The analytical
# return-mapping procedure of von Mises plasticity is used to define the behviour
# of the stress operator and its derivative. The procedure is implemented using
# some external (non-UFL) piece of code.
#
# ## Implementation
#
# ### Preamble

# %%
import matplotlib.pyplot as plt
import sys
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
from solvers import LinearProblem
from utilities import build_cylinder_quarter, find_cell_by_point

import basix
import ufl
import numba
from dolfinx import fem, common
from dolfinx_external_operator import (
    FEMExternalOperator,
    replace_external_operators,
    evaluate_external_operators,
    evaluate_operands
)

# %%
R_e, R_i = 1.3, 1.0  # external/internal radius

# elastic parameters
E = 70e3
nu = 0.3

sigma_0 = 250.0  # yield strength


# %% [markdown]
# ### Building the mesh

# %%
mesh, facet_tags, facet_tags_labels = build_cylinder_quarter()

# %%
k_u = 2
V = fem.functionspace(mesh, ("Lagrange", k_u, (2,)))


def epsilon(v):
    grad_v = ufl.grad(v)
    return ufl.as_vector([grad_v[0, 0], grad_v[1, 1], 0, np.sqrt(2.0) * 0.5 * (grad_v[0, 1] + grad_v[1, 0])])


k_stress = 2 * (k_u - 1)
ds = ufl.Measure(
    "ds",
    domain=mesh,
    subdomain_data=facet_tags,
    metadata={"quadrature_degree": k_stress, "quadrature_scheme": "default"},
)

dx = ufl.Measure(
    "dx",
    domain=mesh,
    metadata={"quadrature_degree": k_stress, "quadrature_scheme": "default"},
)

Du = fem.Function(V, name="displacement_increment")
S_element = basix.ufl.quadrature_element(
    mesh.topology.cell_name(), degree=k_stress, value_shape=(4,))
S = fem.functionspace(mesh, S_element)
sigma = FEMExternalOperator(epsilon(Du), function_space=S)

n = ufl.FacetNormal(mesh)
loading = fem.Constant(mesh, PETSc.ScalarType(0.0))

v = ufl.TestFunction(V)
# TODO: think about the sign later
F = ufl.inner(sigma, epsilon(v)) * dx + loading * \
    ufl.inner(v, n) * ds(facet_tags_labels["inner"])


lmbda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu)
mu = E / 2.0 / (1.0 + nu)
C_elas = np.array(
    [
        [lmbda + 2.0 * mu, lmbda, lmbda, 0.0],
        [lmbda, lmbda + 2.0 * mu, lmbda, 0.0],
        [lmbda, lmbda, lmbda + 2.0 * mu, 0.0],
        [0.0, 0.0, 0.0, 2.0 * mu],
    ],
    dtype=PETSc.ScalarType,
)

deviatoric = np.eye(4, dtype=PETSc.ScalarType)
deviatoric[:3, :3] -= np.full((3, 3), 1.0 / 3.0, dtype=PETSc.ScalarType)

E_tangent = E / 100.0  # tangent modulus
H = E * E_tangent / (E - E_tangent)  # hardening modulus

# Internal state
P_element = basix.ufl.quadrature_element(
    mesh.topology.cell_name(), degree=k_stress, value_shape=())
P = fem.functionspace(mesh, P_element)

p = fem.Function(P, name="cumulative_plastic_strain")
dp = fem.Function(P, name="incremental_plastic_strain")
sigma_n = fem.Function(S, name="stress_n")

# Boundary conditions
bottom_facets = facet_tags.find(facet_tags_labels["Lx"])
left_facets = facet_tags.find(facet_tags_labels["Ly"])

bottom_dofs_y = fem.locate_dofs_topological(
    V.sub(1), mesh.topology.dim - 1, bottom_facets)
left_dofs_x = fem.locate_dofs_topological(
    V.sub(0), mesh.topology.dim - 1, left_facets)

sym_bottom = fem.dirichletbc(
    np.array(0.0, dtype=PETSc.ScalarType), bottom_dofs_y, V.sub(1))
sym_left = fem.dirichletbc(
    np.array(0.0, dtype=PETSc.ScalarType), left_dofs_x, V.sub(0))

bcs = [sym_bottom, sym_left]


# %%
# num_quadrature_points = P_element.dim
# NOTE: if we have the above why not to have this one
# num_cells = int(p.x.array.shape[0]/num_quadrature_points)

# The underscore means "_" with "appropriate" shape
# "_local" means local
# without underscore means raw global data or tmp outputs
@numba.njit
def return_mapping(deps_, sigma_n_, p_):
    """Performs the return-mapping procedure."""
    num_cells = deps_.shape[0]
    num_quadrature_points = deps_.shape[1]

    C_tang_ = np.empty((num_cells, num_quadrature_points,
                       4, 4), dtype=PETSc.ScalarType)
    sigma_ = np.empty_like(sigma_n_)
    dp_ = np.empty_like(p_)

    # NOTE: LLVM will inline this function call.
    def _kernel(deps_local, sigma_n_local, p_local):
        sigma_elastic = sigma_n_local + C_elas @ deps_local
        s = deviatoric @ sigma_elastic
        sigma_eq = np.sqrt(3.0 / 2.0 * np.dot(s, s))

        f_elastic = sigma_eq - sigma_0 - H * p_local
        f_elastic_plus = (f_elastic + np.sqrt(f_elastic**2)) / 2.0

        dp = f_elastic_plus / (3 * mu + H)

        n_elas = s / sigma_eq * f_elastic_plus / f_elastic
        beta = 3 * mu * dp / sigma_eq

        sigma = sigma_elastic - beta * s

        n_elas_matrix = np.outer(n_elas, n_elas)
        C_tang = C_elas - 3 * mu * \
            (3 * mu / (3 * mu + H) - beta) * \
            n_elas_matrix - 2 * mu * beta * deviatoric

        return C_tang, sigma, dp

    for i in range(0, num_cells):
        for j in range(0, num_quadrature_points):
            C_tang_[i, j], sigma_[i, j], dp_[i, j] = _kernel(
                deps_[i, j], sigma_n_[i, j], p_[i, j])

    return C_tang_, sigma_, dp_


def C_tang_impl(deps):
    num_cells = deps.shape[0]
    num_quadrature_points = int(deps.shape[1]/4)

    deps_ = deps.reshape((num_cells, num_quadrature_points, 4))
    # Current state
    sigma_n_ = sigma_n.x.array.reshape((num_cells, num_quadrature_points, 4))
    p_ = p.x.array.reshape((num_cells, num_quadrature_points))

    C_tang_, sigma_, dp_ = return_mapping(deps_, sigma_n_, p_)

    return C_tang_.reshape(-1), sigma_.reshape(-1), dp_.reshape(-1)


def sigma_impl(deps):
    return NotImplementedError


def sigma_external(derivatives):
    if derivatives == (0,):
        return sigma_impl
    elif derivatives == (1,):
        return C_tang_impl
    else:
        return NotImplementedError


sigma.external_function = sigma_external


# Form manipulations
u_hat = ufl.TrialFunction(V)
J = ufl.derivative(F, Du, u_hat)
J_expanded = ufl.algorithms.expand_derivatives(J)

F_replaced, F_external_operators = replace_external_operators(F)
J_replaced, J_external_operators = replace_external_operators(J_expanded)

F_form = fem.form(F_replaced)
J_form = fem.form(J_replaced)

# %%
# NOTE: Small test, to remove/move.
# Avoid divide by zero
eps = np.finfo(PETSc.ScalarType).eps
Du.x.array[:] = eps

# %%
timer1 = common.Timer("1st numba pass")
timer1.start()
evaluated_operands = evaluate_operands(F_external_operators)
((_, sigma_new, dp_new),) = evaluate_external_operators(
    J_external_operators, evaluated_operands)
timer1.stop()

timer2 = common.Timer("2nd numba pass")
timer2.start()
evaluated_operands = evaluate_operands(F_external_operators)
((_, sigma_new, dp_new),) = evaluate_external_operators(
    J_external_operators, evaluated_operands)
timer2.stop()

timer3 = common.Timer("3nd numba pass")
timer3.start()
evaluated_operands = evaluate_operands(F_external_operators)
((_, sigma_new, dp_new),) = evaluate_external_operators(
    J_external_operators, evaluated_operands)
timer3.stop()

# %%
common.list_timings(MPI.COMM_WORLD, [common.TimingType.wall])

# %%
u = fem.Function(V, name="displacement")
du = fem.Function(V, name="Newton_correction")
external_operator_problem = LinearProblem(J_replaced, -F_replaced, Du, bcs=bcs)

# %%
# Defining a cell containing (Ri, 0) point, where we calculate a value of u
# It is required to run this program via MPI in order to capture the process, to which this point is attached
x_point = np.array([[R_i, 0, 0]])
cells, points_on_proc = find_cell_by_point(mesh, x_point)

# %%
q_lim = 2.0 / np.sqrt(3.0) * np.log(R_e / R_i) * sigma_0
Nitermax, tol = 200, 1e-8  # parameters of the manual Newton method
num_increments = 20
load_steps = np.linspace(0, 1.1, num_increments + 1)[1:] ** 0.5
results = np.zeros((num_increments + 1, 2))

# timer3 = common.Timer("Solving the problem")
# start = MPI.Wtime()
# timer3.start()

# TODO: Why can't we use NewtonSolver?
# ANSWER: May be not for this example? it's heavy enough.
# Mohr-Coulomb where the algo is more general suits better to introduce the nonlinear solver + external operators, IMHO.
for i, t in enumerate(load_steps):
    loading.value = t * q_lim
    external_operator_problem.assemble_vector()

    nRes0 = external_operator_problem.b.norm()
    nRes = nRes0
    Du.x.array[:] = 0

    if MPI.COMM_WORLD.rank == 0:
        print(f"\nnRes0 , {nRes0} \n Increment: {i+1!s}, load = {t * q_lim}")
    niter = 0

    while nRes / nRes0 > tol and niter < Nitermax:
        external_operator_problem.assemble_matrix()
        external_operator_problem.solve(du)

        Du.vector.axpy(1, du.vector)  # Du = Du + 1*du
        Du.x.scatter_forward()

        evaluated_operands = evaluate_operands(F_external_operators)
        ((_, sigma_new, dp_new),) = evaluate_external_operators(
            J_external_operators, evaluated_operands)
        sigma.ref_coefficient.x.array[:] = sigma_new
        dp.x.array[:] = dp_new

        external_operator_problem.assemble_vector()
        nRes = external_operator_problem.b.norm()

        if MPI.COMM_WORLD.rank == 0:
            print(f"    it# {niter} Residual: {nRes}")
        niter += 1
    u.vector.axpy(1, Du.vector)  # u = u + 1*Du
    u.x.scatter_forward()

    p.vector.axpy(1, dp.vector)
    p.x.scatter_forward()
    np.copyto(sigma_n.x.array, sigma.ref_coefficient.x.array)

    if len(points_on_proc) > 0:
        results[i + 1, :] = (u.eval(points_on_proc, cells)[0], t)

# end = MPI.Wtime()
# timer3.stop()

# total_time = end - start
# compilation_overhead = time1 - time2

# print(f'rank#{MPI.COMM_WORLD.rank}: Time = {total_time:.3f} (s)')
# print(f'rank#{MPI.COMM_WORLD.rank}: Compilation overhead: {compilation_overhead:.3f} s')

# %%
if len(points_on_proc) > 0:
    plt.plot(results[:, 0], results[:, 1], "-o", label="via ExternalOperator")
    plt.xlabel("Displacement of inner boundary")
    plt.ylabel(r"Applied pressure $q/q_{lim}$")
    plt.savefig(f"displacement_rank{MPI.COMM_WORLD.rank:d}.png")
    plt.legend()
    plt.show()
