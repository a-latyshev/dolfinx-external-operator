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
#     display_name: dolfinx-env (3.12.3)
#     language: python
#     name: python3
# ---

# %%
"""
Demo: Neo-Hookean Hyperelasticity - Tensile Test of a Rectangular Specimen (2D)
Author: Adapted from FEniCSx/dolfinx and jorgensd/dolfinx-tutorial
"""

# %%
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh, fem, log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.mesh import CellType
from dolfinx.io import XDMFFile
from dolfinx.fem import Expression, Function
import basix

# %%
# Geometry and mesh (2D)
L = 1.0  # Length
W = 1.0  # Height
nx, ny = 40, 40
domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [L, W]], [nx, ny], cell_type=CellType.quadrilateral)
V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))


# %%
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
top_dofs = fem.locate_dofs_topological(V.sub(1), facet_tag.dim, facet_tag.find(2))
u_D_bottom = np.array((0,) * domain.geometry.dim, dtype=default_scalar_type)
u_D_top = fem.Constant(domain, 0.0)

bcs_u = [
    fem.dirichletbc(u_D_top, top_dofs, V.sub(1)),
    fem.dirichletbc(u_D_bottom, bottom_dofs, V),
]


# %%
u = fem.Function(V)
v = ufl.TestFunction(V)
d = len(u)
I = ufl.variable(ufl.Identity(d))
F = ufl.variable(I + ufl.grad(u))
C = ufl.variable(F.T * F)
Ic = ufl.variable(ufl.tr(C))
detF = ufl.variable(ufl.det(F))

# %%
import torch
import numpy as np

class convexLinear(torch.nn.Module):
    """ Custom linear layer with positive weights and no bias """
    def __init__(self, size_in, size_out, ):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = torch.nn.Parameter(weights)

        # initialize weights
        torch.nn.init.kaiming_uniform_(self.weights, a=np.sqrt(5))

    def forward(self, x):
        w_times_x= torch.mm(x, torch.nn.functional.softplus(self.weights.t()))
        return w_times_x

class ICNN(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output, dropout):
        super(ICNN, self).__init__()
        # Create Module dicts for the hidden and skip-connection layers
        self.layers = torch.nn.ModuleDict()
        self.skip_layers = torch.nn.ModuleDict()
        self.depth = len(n_hidden)
        self.dropout = dropout[0]
        self.p_dropout = dropout[1]

        self.layers[str(0)] = torch.nn.Linear(n_input, n_hidden[0]).float()
        # Create create NN with number of elements in n_hidden as depth
        for i in range(1, self.depth):
            self.layers[str(i)] = convexLinear(n_hidden[i-1], n_hidden[i]).float()
            self.skip_layers[str(i)] = torch.nn.Linear(n_input, n_hidden[i]).float()

        self.layers[str(self.depth)] = convexLinear(n_hidden[self.depth-1], n_output).float()
        self.skip_layers[str(self.depth)] = convexLinear(n_input, n_output).float()

    def forward(self, x):

        # Get F components
        F11 = x[:,0:1]
        F12 = x[:,1:2]
        F21 = x[:,2:3]
        F22 = x[:,3:4]

        # Compute right Cauchy green strain Tensor
        C11 = F11**2 + F21**2
        C12 = F11*F12 + F21*F22
        C21 = F11*F12 + F21*F22
        C22 = F12**2 + F22**2

        # Compute computeStrainInvariants
        I1 = C11 + C22 + 1.0
        I2 = C11 + C22 - C12*C21 + C11*C22
        I3 = C11*C22 - C12*C21

        # Apply transformation to invariants
        K1 = I1 * torch.pow(I3,-1/3) - 3.0
        K2 = (I1 + I3 - 1) * torch.pow(I3,-2/3) - 3.0
        J = torch.sqrt(I3)
        K3 = (J-1)**2

        # Concatenate feature
        x_input = torch.cat((K1,K2,K3),1).float()

        z = x_input.clone()
        z = self.layers[str(0)](z)
        for layer in range(1,self.depth):
            skip = self.skip_layers[str(layer)](x_input)
            z = self.layers[str(layer)](z)
            z += skip
            z = torch.nn.functional.softplus(z)     
            z = 1/12.*torch.square(z)
            if self.training:
                if self.dropout:
                    z = torch.nn.functional.dropout(z,p=self.p_dropout)
        y = self.layers[str(self.depth)](z) + self.skip_layers[str(self.depth)](x_input)
        return y

n_input = 3
n_output = 1
n_hidden = [64,64,64]
dropout = [True,0.2]
model = ICNN(n_input=n_input,
                n_hidden=n_hidden,
                n_output=n_output,
                dropout=dropout)

model.load_state_dict(torch.load('ArrudaBoyce_noise=high.pth'))
model.eval()

# Create dummy deformation gradients based on uniaxial tension
# F = torch.zeros(50, 4)
# gamma = torch.linspace(0,0.5,50)
# for a in range(50):
#     F[a,0] = 1 + gamma[a]
#     F[a,1] = 0
#     F[a,2] = 0
#     F[a,3] = 1

# F.requires_grad = True

# Zero input deformation gradient + track gradients
F_0 = torch.zeros((1, 4))
F_0[:, 0] = 1
F_0[:, 3] = 1

F_0.requires_grad = True

# %%
# # Predict strain energy (uncorrected)
# W_NN = model(F)
# # Compute the gradient of W_NN with respect to the entire F tensor
# P_NN = torch.autograd.grad(W_NN, F, torch.ones_like(W_NN), create_graph=True)[0]

# W_NN_0 = model(F_0)
# P_NN_0 = torch.autograd.grad(W_NN_0, F_0, torch.ones_like(W_NN_0), create_graph=True)[0].squeeze()

# P_cor = torch.zeros_like(P_NN)

# P_cor[:,0] = F[:,0]*-P_NN_0[0] + F[:,1]*-P_NN_0[2]
# P_cor[:,1] = F[:,0]*-P_NN_0[1] + F[:,1]*-P_NN_0[3]
# P_cor[:,2] = F[:,2]*-P_NN_0[0] + F[:,3]*-P_NN_0[2]
# P_cor[:,3] = F[:,2]*-P_NN_0[1] + F[:,3]*-P_NN_0[3]

# P = P_NN + P_cor

# # Initialize a tensor to store the full Jacobian (second derivative)
# batch_size, num_components = F.shape
# jacobian_rows = []

# # Compute the Jacobian row by row
# for i in range(num_components):
#     grad_output = torch.zeros_like(P)
#     grad_output[:, i] = 1.0  # Select the i-th component
#     jacobian_rows.append(torch.autograd.grad(P, F, grad_output, create_graph=True)[0])

# dP = torch.stack(jacobian_rows, dim=1)  # Shape: (batch_size, num_components, num_components)

# print("Jacobian shape:", dP.shape)

# %%
# Material parameters (neo-Hookean)
E = default_scalar_type(1.0e4)
nu = default_scalar_type(0.3)
mu = fem.Constant(domain, E / (2 * (1 + nu)))
lmbda = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))
# Strain energy and first Piola-Kirchhoff stress
psi = (mu / 2) * (Ic - 2) - mu * ufl.ln(detF) + (lmbda / 2) * (ufl.ln(detF)) ** 2
P = ufl.diff(psi, F)
dP = ufl.diff(P, F)

# Define external function for Piola-Kirchhoff stress and its derivative
import numpy as np
from dolfinx_external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    evaluate_operands,
    replace_external_operators,
)
from solvers import NonlinearProblemWithCallback

# # Fval: (num_cells, 2, 2)
# mu_val = float(mu.value)
# lmbda_val = float(lmbda.value)
# I = np.eye(2)
# C = np.matmul(Fval.transpose(0, 2, 1), Fval)
# Ic = np.trace(C, axis1=1, axis2=2)
# J = np.linalg.det(Fval)
# psi = (mu_val / 2) * (Ic - 2) - mu_val * np.log(J) + (lmbda_val / 2) * (np.log(J)) ** 2
# # dpsi/dF (analytical, as in original code)
# FinvT = np.linalg.inv(Fval).transpose(0, 2, 1)
# P = mu_val * (Fval - FinvT) + lmbda_val * np.log(J)[:, None, None] * FinvT

quadrature_degree = 2
cell_name = domain.topology.cell_name()
quadrature_points, _ = basix.make_quadrature(getattr(basix.CellType, cell_name), quadrature_degree)
map_c = domain.topology.index_map(domain.topology.dim)
num_cells = map_c.size_local + map_c.num_ghosts
cells = np.arange(num_cells, dtype=np.int32)
P_expr = Expression(P, quadrature_points, dtype=default_scalar_type)
dP_expr = Expression(dP, quadrature_points, dtype=default_scalar_type)



def P_impl(Fval):
    P_eval = P_expr.eval(domain, cells)
    return P_eval.reshape(-1)

def dP_dF_impl(Fval):
    dP_eval = dP_expr.eval(domain, cells)
    return dP_eval.reshape(-1)

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

    P_cor[:,0] = F[:,0]*-P_NN_0[0] + F[:,1]*-P_NN_0[2]
    P_cor[:,1] = F[:,0]*-P_NN_0[1] + F[:,1]*-P_NN_0[3]
    P_cor[:,2] = F[:,2]*-P_NN_0[0] + F[:,3]*-P_NN_0[2]
    P_cor[:,3] = F[:,2]*-P_NN_0[1] + F[:,3]*-P_NN_0[3]

    P = P_NN + P_cor

    # Initialize a tensor to store the full Jacobian (second derivative)
    batch_size, num_components = F.shape
    jacobian_rows = []

    # Compute the Jacobian row by row
    for i in range(num_components):
        grad_output = torch.zeros_like(P)
        grad_output[:, i] = 1.0  # Select the i-th component
        jacobian_rows.append(torch.autograd.grad(P, F, grad_output, create_graph=True)[0])

    dP = torch.stack(jacobian_rows, dim=1)  # Shape: (batch_size, num_components, num_components)

    return dP.reshape(-1).detach(), P.reshape(-1).detach()

def P_external(derivatives):
    # if derivatives == (0,):
    #     return P_impl
    if derivatives == (1,):
        return dP_dF_impl
    else:
        raise NotImplementedError(f"No external function is defined for the requested derivative {derivatives}.")

# Create a quadrature element and function space for tensor-valued P
quadrature_degree = 2
cell_name = domain.topology.cell_name()
Qe = basix.ufl.quadrature_element(cell_name, value_shape=(d, d), degree=quadrature_degree, scheme="default")
Q = fem.functionspace(domain, Qe)

# Replace P with FEMExternalOperator
P = FEMExternalOperator(F, function_space=Q, external_function=P_external)

# %%
P_expr.eval(domain, cells).shape

# %%
# Measures for integration
metadata = {"quadrature_degree": 2}
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag, metadata=metadata)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

# %%
# from dolfinx.fem import Expression, Function
# import basix

# # Choose quadrature degree and get quadrature points for the cell type
# quadrature_degree = 2
# cell_name = domain.topology.cell_name()
# quadrature_points, _ = basix.make_quadrature(getattr(basix.CellType, cell_name), quadrature_degree)

# # Create Expression for P at quadrature points
# P_expr = Expression(P, quadrature_points, dtype=default_scalar_type)

# # Evaluate P at all cells (including ghosts)
# map_c = domain.topology.index_map(domain.topology.dim)
# num_cells = map_c.size_local + map_c.num_ghosts
# cells = np.arange(num_cells, dtype=np.int32)
# P_eval = P_expr.eval(domain, cells)

# # Optionally, assemble into a Function for visualization/post-processing
# P_func = Function(Q)
# P_func.x.array[:] = P_eval.flatten()

# # Create Expression for F at quadrature points
# F_expr = Expression(F, quadrature_points, dtype=default_scalar_type)

# # Evaluate F at all cells (including ghosts)
# F_eval = F_expr.eval(domain, cells)

# # Optionally, assemble into a Function for visualization/post-processing
# F_func = Function(Q)
# F_func.x.array[:] = F_eval.flatten()

# %%
u_hat = ufl.TrialFunction(V)
F = ufl.inner(ufl.grad(v), P) * dx
J = ufl.derivative(F, u, u_hat)
J_expanded = ufl.algorithms.expand_derivatives(J)

F_replaced, F_external_operators = replace_external_operators(F)
J_replaced, J_external_operators = replace_external_operators(J_expanded)

F_form = fem.form(F_replaced)
J_form = fem.form(J_replaced)

# %%
from petsc4py import PETSc

def constitutive_update():
    evaluated_operands = evaluate_operands(F_external_operators)
    # _ = evaluate_external_operators(F_external_operators, evaluated_operands)
    # _ = evaluate_external_operators(J_external_operators, evaluated_operands)
    ((_, P_new),) = evaluate_external_operators(J_external_operators, evaluated_operands)
    P.ref_coefficient.x.array[:] = P_new

problem = NonlinearProblemWithCallback(F_replaced, u, bcs=bcs_u, J=J_replaced, external_callback=constitutive_update)
solver = NewtonSolver(domain.comm, problem)
solver.atol = 1e-8
solver.rtol = 1e-8
solver.convergence_criterion = "incremental"
ksp = solver.krylov_solver
opts = PETSc.Options()  # type: ignore
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

# %%
# # Nonlinear problem and Newton solver
# problem = NonlinearProblem(F, u, bcs_u)
# solver = NewtonSolver(domain.comm, problem)
# solver.atol = 1e-8
# solver.rtol = 1e-8
# solver.convergence_criterion = "incremental"

# %%
# with XDMFFile(domain.comm, "tensile2d_u.xdmf", "w") as xdmf:
#     xdmf.write_mesh(domain)

# %%
# Apply a tensile load by incrementally increasing traction on the right edge
n_steps = 100
max_traction = 0.5 
u.name = "displacement"
u.x.array[:] = 0
for step in range(1, n_steps + 1):
    u_D_top.value = step * max_traction / n_steps
    num_its, converged = solver.solve(u)
    assert converged, f"Newton solver did not converge at step {step}"
    u.x.scatter_forward()
    if domain.comm.rank == 0:
        print(f"Step {step}: Traction {u_D_top.value:.2f}, Newton its: {num_its}")
    # Save solution for each step
    # with XDMFFile(domain.comm, "tensile2d_u.xdmf", "a") as xdmf:
    #     xdmf.write_function(u, step)

# %%
import pyvista
import dolfinx.plot
import matplotlib.pyplot as plt

# %%
# pyvista.start_xvfb()
plotter = pyvista.Plotter(window_size=[600, 400], off_screen=True)
topology, cell_types, x = dolfinx.plot.vtk_mesh(domain)
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

# plotter.add_axes()
# plotter.show()

# %%
vals

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
