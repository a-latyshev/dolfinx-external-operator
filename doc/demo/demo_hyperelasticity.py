# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: fenicsx-0.10.0
#     language: python
#     name: python3
# ---

# %% [markdown]
# # [Preview] Hyperelasticity via Input-Convex Neural Networks (ICNN) (PyTorch)
#
# This tutorial demonstrates how to define a complex hyperelastic constitutive
# model using PyTorch automatic differentiation (AD) and integrate it with
# FEniCSx. In solid mechanics, standard models (like Neo-Hookean) have simple
# analytical derivatives. However, data-driven constitutive equations (like
# neural network surrogates) require automatic differentiation to evaluate
# stresses and tangent operators (Jacobians) at the quadrature points of the
# finite element mesh.
#
# Here, we stick to the Arruda-Boyce hyperelasticity model. The tutorial is
# based on the work of {cite:t}`thakolkaranNNEUCLID2022` on unsupervised
# deep-learning hyperelasticity without stress data (NN-EUCLID):
# https://github.com/EUCLID-code/EUCLID-hyperelasticity-NN.
#
# To get familiarized with hyperelasticity problems, we advise to take a look
# first at the following basic tutorials:
# * [Numerical Tours of Computational Mechanics with
#   FEniCSx](https://bleyerj.github.io/comet-fenicsx/intro/hyperelasticity/hyperelasticity.html)
#   by Jérémy Bleyer.
# * [The FEniCSx
#   tutorial](https://jsdokken.com/dolfinx-tutorial/chapter2/hyperelasticity.html)
#   by Jørgen S. Dokken and Garth N. Wells.
#
# ## Problem Formulation
#
# We solve a hyperelasticity boundary value problem on a 2D square domain
# $\Omega$ containing elliptic holes, subjected to displacement-controlled
# tensile loading.
#
# Let $V$ be the functional space of admissible displacement fields. Under the
# assumption of quasi-static displacement-controlled loading and in the absence
# of body forces, the weak formulation of the equilibrium equation is expressed
# as:
#
# Find $\mathbf{u} \in V$ satisfying the Dirichlet boundary conditions such
# that:
#
# $$ F(\mathbf{u}; \mathbf{v}) = \int\limits_\Omega \mathbf{P}(\mathbf{F}) :
#     \nabla\mathbf{v} \, \mathrm{d}\mathbf{x} = 0, \quad \forall \mathbf{v} \in
# V $$
#
# where:
# - $\mathbf{F} = \mathbf{I} + \nabla\mathbf{u}$ is the deformation gradient.
# - $\mathbf{P}(\mathbf{F}) = \frac{\partial W}{\partial \mathbf{F}}$ is the
#   first Piola-Kirchhoff stress tensor derived from the strain energy density
#   $W(\mathbf{F})$.
# - $\mathbf{v}$ is a test function vanishing on the Dirichlet boundaries.
#
# ## Implementation
#
# ### Preamble

# %%
from utilities import build_square_with_elliptic_holes

import basix
import dolfinx.plot
import ufl
from dolfinx import default_scalar_type, fem, mesh
from dolfinx.fem.petsc import NonlinearProblem

# Define external function for Piola-Kirchhoff stress and its derivative
from dolfinx_external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    evaluate_operands,
    replace_external_operators,
)
from dolfinx_external_operator.petsc import assemble_residual_with_callback

# isort: split

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pyvista
import torch

# %% [markdown]
# ### Geometry and boundary conditions
#
# We solve a 2D tension problem of a square specimen with elliptic holes. The boundary
# conditions consist of zero displacement at the bottom, and a prescribed vertical
# displacement at the top. We use quadratic Lagrange elements for the displacement field $\mathbf{u}$.

# %%
L = 1.0  # Length
W = 1.0  # Height

domain, facet_tags, facet_tags_labels = build_square_with_elliptic_holes(L=L, lc=0.02)
V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))

def bottom(x):
    return np.isclose(x[1], 0.0)

def top(x):
    return np.isclose(x[1], W)


fdim = domain.topology.dim - 1
bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom)
top_facets = mesh.locate_entities_boundary(domain, fdim, top)

marked_facets = np.hstack([bottom_facets, top_facets])
marked_values = np.hstack([np.full_like(bottom_facets, 1), np.full_like(top_facets, 2)])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

bottom_dofs = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.find(1))
top_dofs_uy = fem.locate_dofs_topological(V.sub(1), facet_tag.dim, facet_tag.find(2))
top_dofs_ux = fem.locate_dofs_topological(V.sub(0), facet_tag.dim, facet_tag.find(2))
u_D_bottom = np.array((0,) * domain.geometry.dim, dtype=default_scalar_type)
u_D_top = fem.Constant(domain, 0.0)

bcs_u = [
    fem.dirichletbc(u_D_top, top_dofs_uy, V.sub(1)),
    fem.dirichletbc(fem.Constant(domain, 0.0), top_dofs_ux, V.sub(0)),
    fem.dirichletbc(u_D_bottom, bottom_dofs, V),
]


# %% [markdown]
# ### Constitutive Formulation - hyperelasticity via Input-Convex Neural Network (ICNN)
#
# We model the hyperelastic response using an Input-Convex Neural Network (ICNN)
# designed to output a strain energy density $W(\mathbf{F})$ that is convex with
# respect to the right Cauchy-Green strain invariants. Objectivity is satisfied
# by expressing $W$ as a function of the right Cauchy-Green deformation tensor
# $\mathbf{C} = \mathbf{F}^T\mathbf{F}$.
#
# Following {cite}`thakolkaranNNEUCLID2022`, the strain energy density is formulated
# as:
#
# $$ W(\mathbf{F}) = W_{\mathbf{Q},
#     \mathbf{\mathcal{A}}}^{\text{NN}}(\mathbf{E}(\mathbf{F})) + W^0 +
# \mathbf{H}:\mathbf{E} $$
# where:
# - $W_{\mathbf{Q}, \mathbf{\mathcal{A}}}^{\text{NN}}$ is the neural network
#   mapping the strain invariants $\mathbf{E}(\mathbf{F})$ to a scalar energy.
# - $W^0$ is a scalar correction offsetting the energy density to zero in the
#   reference configuration:
#
# $$ W^0 = -\left. W_{\mathbf{Q},
#       \mathbf{\mathcal{A}}}^{\text{NN}}(\mathbf{E}(\mathbf{F}))
#   \right|_{\mathbf{F} = \mathbf{I}} $$
#
# - $\mathbf{H}$ is a stress correction tensor ensuring that the stress vanishes
#   in the undeformed state ($\mathbf{F}=\mathbf{I}$):
#
# $$ \mathbf{H} = -\left.
#       \frac{\partial W_{\mathbf{Q},
#   \mathbf{\mathcal{A}}}^{\text{NN}}(\mathbf{E})}{\partial \mathbf{F}}
#   \right|_{\mathbf{F}=\mathbf{I}} $$
#
# - $\mathbf{E}$ is the Green-Lagrange strain tensor $\mathbf{E} =
#   \frac{1}{2}(\mathbf{C} - \mathbf{I})$.
#
# The first Piola-Kirchhoff stress $\mathbf{P}$ is computed as:
#
# $$ \mathbf{P}(\mathbf{F}) = \frac{\partial W(\mathbf{F})}{\partial \mathbf{F}}
#     = \frac{\partial W^{\text{NN}}}{\partial \mathbf{F}} +
# \mathbf{F}\mathbf{H} $$
#
# and the tangent modulus is:
#
# $$ \mathbb{C}_{ijkl} = \frac{\partial P_{ij}(\mathbf{F})}{\partial F_{kl}} =
#     \frac{\partial^2 W^{\text{NN}}}{\partial F_{ij} \partial F_{kl}} +
# \delta_{ik} H_{lj} $$
#
# We will load a pre-trained network modeling the Arruda-Boyce material
# behaviour to evaluate the first Piola-Kirchhoff stress $\mathbf{P}$ values.
# Then, using PyTorch automatic differentiation (AD), we compute the energy
# derivative $\frac{\partial^2 W^{\text{NN}}}{\partial F_{ij} \partial F_{kl}}$
# to evaluate values of the tangent $\mathbb{C}_{ijkl}$. Both
# $\mathbf{P}(\mathbf{F})$ and $\mathbb{C}_{ijkl}$ can be then integrated into
# FEniCSx as a `FEMExternalOperator`.

# %% [markdown]
# ### Input-Convex Neural Network (ICNN) in PyTorch
#
# We construct the ICNN model in PyTorch using custom layers that enforce positive
# weights in the hidden layers to maintain input-convexity.

# %%
class convexLinear(torch.nn.Module):
    """Custom linear layer with positive weights and no bias"""

    def __init__(
        self,
        size_in,
        size_out,
    ):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = torch.nn.Parameter(weights)

        # initialize weights
        torch.nn.init.kaiming_uniform_(self.weights, a=np.sqrt(5))

    def forward(self, x):
        w_times_x = torch.mm(x, torch.nn.functional.softplus(self.weights.t()))
        return w_times_x


class ICNN(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output, dropout):
        super().__init__()
        # Create Module dicts for the hidden and skip-connection layers
        self.layers = torch.nn.ModuleDict()
        self.skip_layers = torch.nn.ModuleDict()
        self.depth = len(n_hidden)
        self.dropout = dropout[0]
        self.p_dropout = dropout[1]

        self.layers[str(0)] = torch.nn.Linear(n_input, n_hidden[0]).float()
        # Create create NN with number of elements in n_hidden as depth
        for i in range(1, self.depth):
            self.layers[str(i)] = convexLinear(n_hidden[i - 1], n_hidden[i]).float()
            self.skip_layers[str(i)] = torch.nn.Linear(n_input, n_hidden[i]).float()

        self.layers[str(self.depth)] = convexLinear(n_hidden[self.depth - 1], n_output).float()
        self.skip_layers[str(self.depth)] = convexLinear(n_input, n_output).float()

    def forward(self, x):
        # Get F components
        F11 = x[:, 0:1]
        F12 = x[:, 1:2]
        F21 = x[:, 2:3]
        F22 = x[:, 3:4]

        # Compute right Cauchy green strain Tensor
        C11 = F11**2 + F21**2
        C12 = F11 * F12 + F21 * F22
        C21 = F11 * F12 + F21 * F22
        C22 = F12**2 + F22**2

        # Compute computeStrainInvariants
        I1 = C11 + C22 + 1.0
        C11 + C22 - C12 * C21 + C11 * C22
        I3 = C11 * C22 - C12 * C21

        # Apply transformation to invariants
        K1 = I1 * torch.pow(I3, -1 / 3) - 3.0
        K2 = (I1 + I3 - 1) * torch.pow(I3, -2 / 3) - 3.0
        J = torch.sqrt(I3)
        K3 = (J - 1) ** 2

        # Concatenate feature
        x_input = torch.cat((K1, K2, K3), 1).float()

        z = x_input.clone()
        z = self.layers[str(0)](z)
        for layer in range(1, self.depth):
            skip = self.skip_layers[str(layer)](x_input)
            z = self.layers[str(layer)](z)
            z += skip
            z = torch.nn.functional.softplus(z)
            z = 1 / 12.0 * torch.square(z)
            if self.training:
                if self.dropout:
                    z = torch.nn.functional.dropout(z, p=self.p_dropout)
        y = self.layers[str(self.depth)](z) + self.skip_layers[str(self.depth)](x_input)
        return y


n_input = 3
n_output = 1
n_hidden = [64, 64, 64]
dropout = [True, 0.2]
model = ICNN(n_input=n_input, n_hidden=n_hidden, n_output=n_output, dropout=dropout)

model.load_state_dict(torch.load("ArrudaBoyce_noise=high.pth"))
model.eval()

# %% [markdown]
# ### PyTorch-based Tangent and Stress Evaluation
#
# We define the FEniCSx function variables and set up the PyTorch-based consistent
# tangent stiffness evaluation. In contrast to JAX, where vectorization over
# batch dimensions is natively supported using `jax.vmap` and `jax.jacfwd`, PyTorch
# requires iterating row-by-row over components to compute the full consistent tangent
# Jacobian tensor $\mathbb{C}$.

# %%
u = fem.Function(V)
v = ufl.TestFunction(V)
d = len(u)
gradU = ufl.variable(ufl.Identity(d) + ufl.grad(u))

# Zero input deformation gradient + track gradients
F_0 = torch.zeros((1, 4))
F_0[:, 0] = 1
F_0[:, 3] = 1

F_0.requires_grad = True
W_NN_0 = model(F_0)
P_NN_0 = torch.autograd.grad(W_NN_0, F_0, torch.ones_like(W_NN_0), create_graph=True)[0].squeeze()
# P_cor = torch.zeros_like(P_NN)


def dP_dF_impl(Fvals):
    F = torch.from_numpy(np.ascontiguousarray(Fvals)).reshape(-1, 4)  # modify a -> t_shared changes
    F.requires_grad = True

    # Predict strain energy (uncorrected)
    W_NN = model(F)
    # Compute the gradient of W_NN with respect to the entire F tensor
    P_NN = torch.autograd.grad(W_NN, F, torch.ones_like(W_NN), create_graph=True)[0]

    P_cor = torch.zeros_like(P_NN)

    P_cor[:, 0] = F[:, 0] * -P_NN_0[0] + F[:, 1] * -P_NN_0[2]
    P_cor[:, 1] = F[:, 0] * -P_NN_0[1] + F[:, 1] * -P_NN_0[3]
    P_cor[:, 2] = F[:, 2] * -P_NN_0[0] + F[:, 3] * -P_NN_0[2]
    P_cor[:, 3] = F[:, 2] * -P_NN_0[1] + F[:, 3] * -P_NN_0[3]

    P = P_NN + P_cor

    # Initialize a tensor to store the full Jacobian (second derivative)
    _batch_size, num_components = F.shape
    jacobian_rows = []

    # Compute the Jacobian row by row
    for i in range(num_components):
        grad_output = torch.zeros_like(P)
        grad_output[:, i] = 1.0  # Select the i-th component
        jacobian_rows.append(torch.autograd.grad(P, F, grad_output, create_graph=True)[0])

    dP = torch.stack(jacobian_rows, dim=1)  # Shape: (batch_size, num_components, num_components)

    return dP.reshape(-1).detach(), P.reshape(-1).detach()


def P_external(derivatives):
    if derivatives == (1,):
        return dP_dF_impl
    else:
        raise NotImplementedError(f"No external function is defined for the requested derivative {derivatives}.")


# %% [markdown]
# ### FEniCSx Integration and External Operator
#
# We wrap the PyTorch execution code into a python callable, `dP_dF_impl`, and
# define a `FEMExternalOperator` in UFL.

# %%
# Create a quadrature element and function space for tensor-valued P
quadrature_degree = 2
cell_name = domain.topology.cell_name()
Qe = basix.ufl.quadrature_element(cell_name, value_shape=(d, d), degree=quadrature_degree, scheme="default")
Q = fem.functionspace(domain, Qe)

# Replace P with FEMExternalOperator
P = FEMExternalOperator(gradU, function_space=Q, external_function=P_external)

# %% [markdown]
# ## JAX AD vs. PyTorch AD
#
# While both JAX and PyTorch are powerful libraries for deep learning and
# automatic differentiation, they have distinct paradigms:
#
# 1. **Execution Model**:
#    - **JAX** is strictly functional and trace-based. Functions are traced to
#      generate an intermediate representation (jaxpr) that is then JIT-compiled
#      to XLA.
#    - **PyTorch** builds dynamic computation graphs on-the-fly (tape-based). It
#      executes imperatively, making it highly interactive but requiring
#      tracking of tensor histories.
# 2. **Batch Jacobians / Tangents**:
#    - **JAX** has first-class support for vectorization via `jax.vmap` and
#      forward-mode/reverse-mode Jacobian calculations via `jax.jacfwd` and
#      `jax.jacrev`. This allows native batch-wise consistent tangent stiffness
#      evaluation.
#    - **PyTorch** traditionally evaluates gradients for scalar values or needs
#      manual looping for Jacobian rows (`torch.autograd.grad` with one-hot
#      output vectors) in batch mode. Modern PyTorch supports `torch.func`
#      (inspired by JAX), but in finite element environments, tape-based
#      automatic differentiation using custom batch wrappers is still common.
#
# In this tutorial, we will use PyTorch's dynamic graph features and manual
# batch looping to compute the first Piola-Kirchhoff stress tensor and its
# consistent tangent stiffness matrix for each quadrature point.

# %%
# Measures for integration
metadata = {"quadrature_degree": 2}
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag, metadata=metadata)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

# %% [markdown]
# ### Variational Forms and Residual Replacing
#
# We define the weak form using the UFL representation of the external operator,
# compute its directional derivative, and replace it with the `replace_external_operators` function.

# %%
metadata = {"quadrature_degree": 2}
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

u_hat = ufl.TrialFunction(V)
F = ufl.inner(ufl.grad(v), P) * dx
J = ufl.derivative(F, u, u_hat)
J_expanded = ufl.algorithms.expand_derivatives(J)

F_replaced, F_external_operators = replace_external_operators(F)
J_replaced, J_external_operators = replace_external_operators(J_expanded)

F_form = fem.form(F_replaced)
J_form = fem.form(J_replaced)


# %% [markdown]
# ### Newton Solver and Constitutive Update Callback
#
# We set up the PETSc SNES solver using a Newton solver with line search, wrapping
# the constitutive update in a callback function.


# %%
def constitutive_update(
    F_external_operators: list[FEMExternalOperator],
    J_external_operators: list[FEMExternalOperator],
):
    """Update the constitutive model by evaluating the external operators."""
    evaluated_operands = evaluate_operands(F_external_operators)
    ((_, P_new),) = evaluate_external_operators(J_external_operators, evaluated_operands)
    P.ref_coefficient.x.array[:] = P_new


petsc_options = {
    "snes_type": "vinewtonrsls",
    "snes_linesearch_type": "basic",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_atol": 1.0e-7,
    "snes_rtol": 1.0e-7,
    "snes_max_it": 50,
    "snes_monitor": "",
}

problem = NonlinearProblem(
    F_replaced, u, J=J_replaced, bcs=bcs_u, petsc_options_prefix="demo_hyperelasticity_", petsc_options=petsc_options
)

assemble_residual_with_callback_ = partial(
    assemble_residual_with_callback,
    problem.u,
    problem.F,
    problem.J,
    bcs_u,
    constitutive_update,  # external callback with respect to SNES
    [F_external_operators, J_external_operators],  # input arguments of the callback
)
problem.solver.setFunction(assemble_residual_with_callback_, problem.b)

# %% [markdown]
# ### Solving the Problem
#
# We incrementally apply a displacement-controlled tensile load by moving the top boundary.

# %% tags=["scroll-output"]
# Apply a tensile load by incrementally increasing traction on the right edge
n_steps = 100
max_traction = 0.5
u.name = "displacement"
u.x.array[:] = 0
n_steps_total_tmp = 10
for step in range(1, n_steps_total_tmp + 1):
    u_D_top.value = step * max_traction / n_steps
    num_its, converged = problem.solve()
    assert converged, f"Newton solver did not converge at step {step}"
    u.x.scatter_forward()
    if domain.comm.rank == 0:
        print(f"Step {step}: Traction {u_D_top.value:.2f}, Newton its: {num_its}")

# %% [markdown]
# ### Visualizing the Deformation
#
# We use PyVista to plot the deformed configuration of the specimen.

# %%
try:
    import pyvista

    print(pyvista.global_theme.jupyter_backend)
    import dolfinx.plot

    plotter = pyvista.Plotter(window_size=[600, 400], off_screen=True)
    topology, cell_types, x = dolfinx.plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)
    vals = np.zeros((x.shape[0], 3))
    vals[:, : len(u)] = u.x.array.reshape((x.shape[0], len(u)))
    grid["u"] = vals
    warped = grid.warp_by_vector("u", factor=1)
    plotter.add_mesh(warped, show_edges=False, show_scalar_bar=False)
    plotter.view_xy()
    plotter.camera.tight()
    image = plotter.screenshot(None, transparent_background=True, return_img=True)
    plt.imshow(image)
    plt.axis("off")

except ImportError:
    print("pyvista required for this plot")

# %% [markdown]
# ## Verification against Analytical UFL Baseline
#
# To verify the accuracy of the neural network external operator solver, we compare
# its results against the analytical UFL implementation of the Arruda-Boyce model
# (relying on a 5-term Taylor series expansion). We solve the same problem and calculate
# the maximum relative $L^2$ error between the displacement fields.

# %%
u_UFL = fem.Function(V)
v = ufl.TestFunction(V)
d = len(u_UFL)
# Deformation gradient
F_ = ufl.variable(ufl.Identity(d) + ufl.grad(u_UFL))

C = F_.T * F_
J_ = ufl.det(F_)
I1 = ufl.tr(C) + 1.0
I2 = I1 + J_**2 - 1.0

# Arruda-Boyce model parameters
mu = fem.Constant(domain, 2.5)
N_c = fem.Constant(domain, 28.0)
K = fem.Constant(domain, 3.0)

C1 = 1.0 / 2.0
C2 = 1.0 / 20.0
C3 = 11.0 / 1050.0
C4 = 19.0 / 7000.0
C5 = 519.0 / 673750.0

I1_bar = (J_ ** (-2.0 / 3.0)) * I1

W_AB = (
    mu
    * (
        C1 * (I1_bar - 3.0)
        + (C2 / N_c) * (I1_bar**2 - 3.0**2)
        + (C3 / N_c**2) * (I1_bar**3 - 3.0**3)
        + (C4 / N_c**3) * (I1_bar**4 - 3.0**4)
        + (C5 / N_c**4) * (I1_bar**5 - 3.0**5)
    )
    + (K / 2.0) * (J_ - 1.0) ** 2
)

P = ufl.diff(W_AB, F_)

# %%
metadata = {"quadrature_degree": 2}
dx = ufl.Measure("dx", domain=domain, metadata=metadata)
F = ufl.inner(ufl.grad(v), P) * dx

problem_UFL = NonlinearProblem(
    F, u_UFL, bcs=bcs_u, petsc_options=petsc_options, petsc_options_prefix="UFL_hyperelasticity_"
)

# %% tags=["scroll-output"]
# Apply a tensile load by incrementally increasing traction on the right edge
n_steps = 100
max_traction = 0.5
u_UFL.name = "displacement"
u_UFL.x.array[:] = 0
for step in range(1, n_steps_total_tmp + 1):
    u_D_top.value = step * max_traction / n_steps
    num_its, converged = problem_UFL.solve()
    assert converged, f"Newton solver did not converge at step {step}"
    u_UFL.x.scatter_forward()
    if domain.comm.rank == 0:
        print(f"Step {step}: Traction {u_D_top.value:.3f}, Newton its: {num_its}")

# %%
try:
    import pyvista

    print(pyvista.global_theme.jupyter_backend)
    import dolfinx.plot

    plotter = pyvista.Plotter(window_size=[600, 400], off_screen=True)
    topology, cell_types, x = dolfinx.plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)
    vals = np.zeros((x.shape[0], 3))
    vals[:, : len(u_UFL)] = u.x.array.reshape((x.shape[0], len(u_UFL)))
    grid["u"] = vals
    warped = grid.warp_by_vector("u", factor=1)
    plotter.add_mesh(warped, show_edges=False, show_scalar_bar=False)
    plotter.view_xy()
    plotter.camera.tight()
    image = plotter.screenshot(None, transparent_background=True, return_img=True)
    plt.imshow(image)
    plt.axis("off")
    
except ImportError:
    print("pyvista required for this plot")

# %%
np.abs(u.x.array[:] - u_UFL.x.array[:]).max() / np.abs(u_UFL.x.array[:]).max()


# %%
# MPI.COMM_WORLD.allreduce(fem.assemble_scalar(fem.form((u - u_UFL)**2), op=MPI.SUM))

# %%
# # Post-processing: compute von Mises stress
# s = P * F.T / J  # Cauchy stress
# s_dev = s - 1.0 / 2 * ufl.tr(s) * ufl.Identity(d)
# von_Mises = ufl.sqrt(3.0 / 2 * ufl.inner(s_dev, s_dev))
# V_von_mises = fem.functionspace(domain, ("DG", 0))
# stress_expr = fem.Expression(von_Mises, V_von_mises.element.interpolation_points)
# stresses = fem.Function(V_von_mises)
# stresses.interpolate(stress_expr)
# with XDMFFile(domain.comm, "tensile2d_vonmises.xdmf", "w") as xdmf:
#     xdmf.write_mesh(domain)
#     stresses.name = "von_mises"
#     xdmf.write_function(stresses)

# %% [markdown]
# ## References
# ```{bibliography}
# :filter: docname in docnames
# ```
