# %% [markdown]
# # Plasticity of von Mises
# 
# This tutorial aims to demonstrate an efficient implementation of the plasticity
# model of von Mises using an external operator defining the elastoplastic
# constitutive relations written with the help of the 3rd-party package `Numba`.
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
#     \boldsymbol{\varepsilon} = \frac{1}{2}\left( \nabla\boldsymbol{u} +
#     \nabla\boldsymbol{u}^T \right).
# $$
# 
# Throughout the tutorial, we stick to the Mandel-Voigt notation, according to
# which the stress tensor $\boldsymbol{\sigma}$ and the strain tensor
# $\boldsymbol{\varepsilon}$ are written as 4-size vectors with the following
# components
# 
# \begin{align*}
#     & \boldsymbol{\sigma} = [\sigma_{xx}, \sigma_{yy}, \sigma_{zz},
#     \sqrt{2}\sigma_{xy}]^T, \\
#     & \boldsymbol{\varepsilon} = [\varepsilon_{xx}, \varepsilon_{yy},
#     \varepsilon_{zz}, \sqrt{2}\varepsilon_{xy}]^T.
# \end{align*}
# 
# Denoting the deviatoric operator $\mathrm{dev}$, we introduce two additional
# quantities of interest: the cumulative plastic strain $p$ and the equivalent
# stress $\sigma_\text{eq}$ defined by the following formulas:
# 
# \begin{align*}
#     & p = \sqrt{\frac{2}{3} \boldsymbol{e} \cdot \boldsymbol{e}}, \\
#     & \sigma_\text{eq} = \sqrt{\frac{3}{2}\boldsymbol{s} \cdot \boldsymbol{s}},
# \end{align*}
# 
# where $\boldsymbol{e} = \mathrm{dev}\boldsymbol{\varepsilon}$ and
# $\boldsymbol{s} = \mathrm{dev}\boldsymbol{\sigma}$ are deviatoric parts of the
# stain and stress tensors respectively.
# 
# ## Problem formulation
# 
# The domain of the problem $\Omega$ represents the first quarter of the hollow
# cylinder with inner $R_i$ and outer $R_o$ radii, where symmetry conditions
# are set on the left and bottom sides and pressure is set on the inner wall
# $\partial\Omega_\text{inner}$. The behaviour of cylinder material is defined
# by the von Mises yield criterion $f$ with the linear isotropic hardening law
# {eq}`eq_von_Mises`
# 
# $$
#     f(\boldsymbol{\sigma}) = \sigma_\text{eq}(\boldsymbol{\sigma}) - \sigma_0
#     - Hp \leq 0,
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
# 
# $$
#     F(\boldsymbol{u}; \boldsymbol{v}) = \int\limits_\Omega
#     \boldsymbol{\sigma}(\boldsymbol{\varepsilon}(\boldsymbol{u})) \cdot \boldsymbol{\varepsilon(v)}
#     \,\mathrm{d}\boldsymbol{x} - F_\text{ext}(\boldsymbol{v}) = 0, \quad \forall
#     \boldsymbol{v} \in V.
# $$ (eq_von_Mises_main)
# 
# The external force $F_{\text{ext}}(\boldsymbol{v})$ represents the pressure
# inside the cylinder and is written as the following Neumann condition
# 
# $$
#     F_\text{ext}(\boldsymbol{v}) =
#     \int\limits_{\partial\Omega_\text{inner}} (-q \boldsymbol{n}) \cdot \boldsymbol{v}
#     \,\mathrm{d}\boldsymbol{x},
# $$
# where the vector $\boldsymbol{n}$ is the outward normal to the cylinder
# surface and the loading parameter $q$ is progressively adjusted from 0 to
# $q_\text{lim} = \frac{2}{\sqrt{3}}\sigma_0\log\left(\frac{R_o}{R_i}\right)$,
# the analytical collapse load for the perfect plasticity model without
# hardening.
# 
# The modelling is performed under assumptions of the plane strain and
# an associative plasticity law.
# 
# In this tutorial, we treat the stress tensor $\boldsymbol{\sigma}$ as an
# external operator acting on the strain tensor
# $\boldsymbol{\varepsilon}(\boldsymbol{u})$ and represent it through a
# `FEMExternalOperator` object. By the implementation of this external operator,
# we mean an implementation of the return-mapping procedure, the most common
# approach to solve plasticity problems. With the help of this procedure, we
# compute both values of the stress tensor $\boldsymbol{\sigma}$ and its
# derivative, so-called the tangent moduli $\boldsymbol{C}_\text{tang}$.
# 
# As before, in order to solve the nonlinear equation {eq}`eq_von_Mises_main`
# we need to compute the Gateaux derivative of $F$ in the direction
# $\boldsymbol{\hat{u}} \in V$:
# 
# $$
#     J(\boldsymbol{u}; \boldsymbol{\hat{u}},\boldsymbol{v}) :=
#     D_{\boldsymbol{u}} [F(\boldsymbol{u};
#     \boldsymbol{v})]\{\boldsymbol{\hat{u}}\} := \int\limits_\Omega
# \left( \boldsymbol{C}_\text{tang}(\boldsymbol{\varepsilon}(\boldsymbol{u}))
# \cdot \boldsymbol{\varepsilon}(\boldsymbol{\hat{u}}) \right) \cdot
#     \boldsymbol{\varepsilon(v)} \,\mathrm{d}\boldsymbol{x}, \quad \forall \boldsymbol{v}
#     \in V.
# $$
# 
# The advantage of the von Mises model is that the return-mapping procedure may be
# performed analytically, so the stress tensor and the tangent moduli may be
# expressed explicitly using any package. In our case, we the Numba library to
# define the behaviour of the external operator and its derivative.
# 
# ## Implementation
# 
# ### Preamble

# %%
from mpi4py import MPI
from petsc4py import PETSc

import matplotlib.pyplot as plt
import numba
import numpy as np
from demo_plasticity_von_mises_pure_ufl import plasticity_von_mises_pure_ufl
from solvers import PETScNonlinearProblem, PETScNonlinearSolver
from utilities import build_cylinder_quarter, find_cell_by_point

import basix
import ufl
from dolfinx import fem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx_external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    evaluate_operands,
    replace_external_operators,
)

# %% [markdown]
# Here we define geometrical and material parameters of the problem as well as some useful constants.

# %%
R_e, R_i = 1.3, 1.0  # external/internal radius

E, nu = 70e3, 0.3  # elastic parameters
E_tangent = E / 100.0  # tangent modulus
H = E * E_tangent / (E - E_tangent)  # hardening modulus
sigma_0 = 250.0  # yield strength
sigt = 250.0 # tensile strength
sigc = 250.0 # compression strength

lmbda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu)
mu = E / 2.0 / (1.0 + nu)
# stiffness matrix
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

# %%
mesh, facet_tags, facet_tags_labels = build_cylinder_quarter()

# %%
k_u = 2
V = fem.functionspace(mesh, ("Lagrange", k_u, (mesh.geometry.dim,)))
# Boundary conditions
bottom_facets = facet_tags.find(facet_tags_labels["Lx"])
left_facets = facet_tags.find(facet_tags_labels["Ly"])

bottom_dofs_y = fem.locate_dofs_topological(V.sub(1), mesh.topology.dim - 1, bottom_facets)
left_dofs_x = fem.locate_dofs_topological(V.sub(0), mesh.topology.dim - 1, left_facets)

sym_bottom = fem.dirichletbc(np.array(0.0, dtype=PETSc.ScalarType), bottom_dofs_y, V.sub(1))
sym_left = fem.dirichletbc(np.array(0.0, dtype=PETSc.ScalarType), left_dofs_x, V.sub(0))

bcs = [sym_bottom, sym_left]


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
S_element = basix.ufl.quadrature_element(mesh.topology.cell_name(), degree=k_stress, value_shape=(4,))
S = fem.functionspace(mesh, S_element)
sigma = FEMExternalOperator(epsilon(Du), function_space=S)

n = ufl.FacetNormal(mesh)
loading = fem.Constant(mesh, PETSc.ScalarType(0.0))

v = ufl.TestFunction(V)
F = ufl.inner(sigma, epsilon(v)) * dx - ufl.inner(loading * -n, v) * ds(facet_tags_labels["inner"])

# Internal state
P_element = basix.ufl.quadrature_element(mesh.topology.cell_name(), degree=k_stress)
P = fem.functionspace(mesh, P_element)

p = fem.Function(P, name="cumulative_plastic_strain")
dp = fem.Function(P, name="incremental_plastic_strain")
sigma_n = fem.Function(S, name="stress_n")

# %% [markdown]
# ### Defining the external operator
# 
# During the automatic differentiation of the form $F$, the following terms will
# appear in the Jacobian
# 
# $$
#     \frac{\mathrm{d} \boldsymbol{\sigma}}{\mathrm{d}
#     \boldsymbol{\varepsilon}}(\boldsymbol{\varepsilon}(\boldsymbol{u})) \cdot
#     \boldsymbol{\varepsilon}(\boldsymbol{\hat{u}}) =
#     \boldsymbol{C}_\text{tang}(\boldsymbol{\varepsilon}(\boldsymbol{u})) \cdot
#     \boldsymbol{\varepsilon}(\boldsymbol{\hat{u}}),
# $$
# 
# where the "trial" part $\boldsymbol{\varepsilon}(\boldsymbol{\hat{u}})$ will be
# handled by the framework and the derivative of the operator
# $\frac{\mathrm{d} \boldsymbol{\sigma}}{\mathrm{d} \boldsymbol{\varepsilon}}$
# must be implemented by the user. In this tutorial, we implement the derivative using the Numba package.
# 
# First of all, we implement the return-mapping procedure locally in the
# function `_kernel`. It computes the values of the stress tensor, the tangent
# moduli and the increment of cumulative plastic strain at a single Gausse
# node. For more details, visit the [original
# implementation](https://comet-fenics.readthedocs.io/en/latest/demo/2D_plasticity/vonMises_plasticity.py.html)
# of this problem for the legacy FEniCS 2019.
# 
# Then we iterate over each Gauss node and compute the quantities of interest
# globally in the `return_mapping` function with the `@numba.njit` decorator.
# This guarantees that the function will be compiled during its first call and
# ordinary `for`-loops will be efficiently processed.

# %%
num_quadrature_points = P_element.dim


@numba.njit
def return_mapping(deps_, sigma_n_, p_):
    """Performs the return-mapping procedure."""
    num_cells = deps_.shape[0]

    C_tang_ = np.empty((num_cells, num_quadrature_points, 4, 4), dtype=PETSc.ScalarType)
    sigma_ = np.empty_like(sigma_n_)
    dp_ = np.empty_like(p_)

    def _kernel(deps_local, sigma_n_local, p_local):
        """Performs the return-mapping procedure locally."""
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
        C_tang = C_elas - 3 * mu * (3 * mu / (3 * mu + H) - beta) * n_elas_matrix - 2 * mu * beta * deviatoric

        return C_tang, sigma, dp

    for i in range(0, num_cells):
        for j in range(0, num_quadrature_points):
            C_tang_[i, j], sigma_[i, j], dp_[i, j] = _kernel(deps_[i, j], sigma_n_[i, j], p_[i, j])

    return C_tang_, sigma_, dp_

# %%
import cvxpy as cp
from scipy.sparse import block_diag

class IsotropicElasticity:
    """A constitutive law of isotropic elasticity.
    
    Attributes: 
        E: Young's modulus [Pa].
        nu: Poisson coefficient [-].   
        lambda: Lame's first parameter [Pa].
        mu: shear modulus [Pa] .
    """
    def __init__(self, E, nu):
        """Inits an  IsotropicElasticity class."""
        self.E = E
        self.nu = nu
        self.lambda_ = E*nu/(1+nu)/(1-2*nu)
        self.mu_ = E/2/(1+nu)        

    def C(self):
        """Returns a 4x4 Voigt elastic tensor."""
        l, m = self.lambda_, self.mu_
        return np.array([[l+2*m, l, l, 0],
                         [l, l+2*m, l, 0],
                         [l, l, l+2*m, 0],
                         [0, 0, 0, 2*m]])

class Material:
    """An abstract 2D material class.
    
    Attributes:
        C: A 4x4 Voigt elastic tensor.
        yield_criterion: A yield criterion.
        plane_stress: A boolean flag showing whether we consider a plane stress problem or not.    
    """
    def __init__(self, constitutive_law, yield_criterion, plane_stress: bool = False):
        """Inits Material class."""
        self.C = constitutive_law.C()
        self.constitutive_law = constitutive_law
        self.yield_criterion = yield_criterion
        self.plane_stress = plane_stress
        
class Rankine():
    def __init__(self, ft: np.float64, fc: np.float64, hardening: np.float64):
        self.fc = ft
        self.ft = fc
        self.H = hardening

    def criterion(self, sig: cp.expressions.variable.Variable, p: cp.expressions.variable.Variable):
        N = p.size

        ft = np.repeat(self.ft, N)
        fc = np.repeat(self.fc, N)

        sigma_max = []
        sigma_min = []
        for i in range(N):
            SIG = cp.bmat([[sig[0,i], sig[3,i]/np.sqrt(2), 0],
                           [sig[3,i]/np.sqrt(2), sig[1,i], 0],
                           [0, 0, sig[2,i]]])
            sigma_max.append(cp.lambda_max(SIG))
            sigma_min.append(cp.lambda_min(SIG))

        return [cp.hstack(sigma_max) <= ft + p * self.H, cp.hstack(sigma_min) >= -fc - p * self.H]

class ReturnMapping:
    """An implementation of return-mapping procedure via convex problems solving.

    Attributes:
        deps:
        sig_old:
        sig:
        p_old:
        p:
        C_tang:
        e:
        opt_problem:
        solver:
    """
    def __init__(self, material:Material, N:int, solver=cp.SCS):
        """Inits ReturnMapping class.
        
        Args:
            material: An appropriate material.
            solver: A convex optimization solver
            
        Note:
            We use here `cp.SCS` as it allows to calculate the derivatives of target variables.
        """
        self.N = N
        self.deps = cp.Parameter((4, N), name='deps')
        self.sig_old = cp.Parameter((4, N), name='sig_old')
        sig_elas = self.sig_old + material.C @ self.deps
        self.sig = cp.Variable((4, N), name='sig')
        
        self.p_old = cp.Parameter((N,), nonneg=True, name='p_old')
        self.p = cp.Variable((N,),nonneg=True, name='p')

        self.sig_old.value = np.zeros((4, N))
        self.deps.value = np.zeros((4, N))
        self.p_old.value = np.zeros((N,))
        self.C_tang = np.zeros((N, 4, 4))

        S = np.linalg.inv(material.C)
        delta_sig = self.sig - sig_elas
        # energy = []
        # for i in range(N):
        #     energy.append(cp.quad_form(delta_sig[:, i], S))
        # target_expression = cp.sum(cp.hstack(energy)) + material.yield_criterion.H * cp.sum_squares(self.p - self.p_old)
        
        # energy = cp.sum(cp.diag(delta_sig.T @ S_sparsed @ delta_sig))
        
        S_sparsed = block_diag([S for _ in range(N)])
        delta_sig_vector = cp.reshape(delta_sig, (N*4))

        elastic_energy = cp.quad_form(delta_sig_vector, S_sparsed, assume_PSD=True)
        # target_expression = 0.5*elastic_energy + 0.5*material.yield_criterion.H * cp.sum_squares(self.p - self.p_old)
        D = material.yield_criterion.H * np.eye(N)
        target_expression = 0.5*elastic_energy + 0.5*cp.quad_form(self.p - self.p_old, D)

        constrains = material.yield_criterion.criterion(self.sig, self.p) 

        if material.plane_stress:
            constrains.append(self.sig[2] == 0) #TO MODIFY!

        self.opt_problem = cp.Problem(cp.Minimize(target_expression), constrains)
        self.solver = solver
    
    def solve(self, **kwargs):
        """Solves a minimization problem and calculates the derivative of `sig` variable.
        
        Args:
            **kwargs: additional solver attributes, such as tolerance, etc.
        """
        self.opt_problem.solve(solver=self.solver, requires_grad=False, ignore_dpp=False, **kwargs)
        
    def solve_and_derivate(self, **kwargs):
        """Solves a minimization problem and calculates the derivative of `sig` variable.
        
        Args:
            **kwargs: additional solver attributes, such as tolerance, etc.
        """

        with common.Timer() as t: 
            self.opt_problem.solve(solver=self.solver, requires_grad=True, **kwargs)
            self.convex_solving_time = t.elapsed()[0] 
        
        with common.Timer() as t: 
            for i in range(4):
                for j in range(self.N):
                    e = np.zeros((4, self.N))
                    e[i, j] = 1
                    self.deps.delta = e
                    self.opt_problem.derivative()
                    self.C_tang[j, :, i] = self.sig.delta[:, j] 
            
            self.differentiation_time = t.elapsed()[0] # time.time() - start
    

# %%
rankine = Rankine(sigt, sigc, H)
material = Material(IsotropicElasticity(E, nu), rankine)

patch_size = 3
return_mapping_cvxpy = ReturnMapping(material, patch_size, 'SCS')
tol = 1.0e-13
scs_params = {'eps': tol, 'eps_abs': tol, 'eps_rel': tol}
conic_solver_params = scs_params

# %% [markdown]
# Now nothing stops us from defining the implementation of the external operator
# derivative (the tangent tensor $\boldsymbol{C}_\text{tang}$) in the
# function `C_tang_impl`. It returns global values of the derivative, stress
# tensor and the cumulative plastic increment.

# %%
def sigma_impl(deps):
    num_cells, num_quadrature_points, _ = deps.shape

    # deps_ = deps.reshape((num_cells, num_quadrature_points, 4))
    # sigma_n_ = sigma_n.x.array.reshape((num_cells, num_quadrature_points, 4))
    # p_ = p.x.array.reshape((num_cells, num_quadrature_points))

    # _, sigma_, dp_ = return_mapping(deps_, sigma_n_, p_)
    N_patches = int(num_quadrature_points / patch_size)
    residue_size = num_quadrature_points % patch_size
    p_values = p.x.array[:num_quadrature_points - residue_size].reshape((-1, patch_size))
    p_old_values = p_old.x.array[:num_quadrature_points - residue_size].reshape((-1, patch_size))
    deps_values = deps.x.array[:4*(num_quadrature_points - residue_size)].reshape((-1, patch_size, 4))
    sig_values = sig.x.array[:4*(num_quadrature_points - residue_size)].reshape((-1, patch_size, 4))
    sig_old_values = sig_old.x.array[:4*(num_quadrature_points - residue_size)].reshape((-1, patch_size, 4))

    if residue_size != 0:
        return_mapping_residue = return_mapping_cvxpy(material, residue_size, 'SCS')
        p_values_residue = p.x.array[num_quadrature_points - residue_size:].reshape((1, residue_size))
        p_old_values_residue = p_old.x.array[num_quadrature_points - residue_size:].reshape((1, residue_size))
        deps_values_residue = deps.x.array[4*(num_quadrature_points - residue_size):].reshape((1, residue_size, 4))
        sig_values_residue = sig.x.array[4*(num_quadrature_points - residue_size):].reshape((1, residue_size, 4))
        sig_old_values_residue = sig_old.x.array[4*(num_quadrature_points - residue_size):].reshape((1, residue_size, 4))
    
    for q in range(N_patches):
        return_mapping.deps.value[:] = deps_values[q,:].T
        return_mapping.sig_old.value[:] = sig_old_values[q,:].T
        return_mapping.p_old.value = p_old_values[q,:]
        
        return_mapping.solve(**conic_solver_params)

        sig_values[q,:] = return_mapping.sig.value[:].T
        p_values[q,:] = return_mapping.p.value

    if residue_size != 0: #how to improve ?
        return_mapping_residue.deps.value[:] = deps_values_residue[0,:].T
        return_mapping_residue.sig_old.value[:] = sig_old_values_residue[0,:].T
        return_mapping_residue.p_old.value = p_old_values_residue[0,:]
        
        return_mapping_residue.solve(**conic_solver_params)

        sig_values_residue[0,:] = return_mapping_residue.sig.value[:].T
        p_values_residue[0,:] = return_mapping_residue.p.value
    return sigma_.reshape(-1), dp_.reshape(-1)

global_size = int(sigma.ref_coefficient.x.array.size / 4.0)
C_elas_ = np.empty((global_size, 4, 4), dtype=PETSc.ScalarType)
for i in range(global_size):
    C_elas_[i] = C_elas

def C_tang_impl(deps):
    # num_cells, num_quadrature_points, _ = deps.shape
   
    # deps_ = deps.reshape((num_cells, num_quadrature_points, 4))
    # sigma_n_ = sigma_n.x.array.reshape((num_cells, num_quadrature_points, 4))
    # p_ = p.x.array.reshape((num_cells, num_quadrature_points))

    # C_tang_, sigma_, dp_ = return_mapping(deps_, sigma_n_, p_)

    return C_elas_.reshape(-1)

# %% [markdown]
# It is worth noting that at the time of the derivative evaluation, we compute the
# values of the external operator as well. Thus, there is no need for a separate
# implementation of the operator $\boldsymbol{\sigma}$. We will reuse the output
# of the `C_tang_impl` to update values of the external operator further in the
# Newton loop.

# %%
def sigma_external(derivatives):
    if derivatives == (0,):
        return sigma_impl
    if derivatives == (1,):
        return C_tang_impl
    else:
        return NotImplementedError


sigma.external_function = sigma_external

# %% [markdown]
# ```{note}
# The framework allows implementations of external operators and its derivatives
# to return additional outputs. In our example, alongside with the values of the
# derivative, the function `C_tang_impl` returns, the values of the stress tensor
# and the cumulative plastic increment. Both additional outputs may be reused by
# the user afterwards in the Newton loop.
# ```

# %% [markdown]
# ### Form manipulations
# 
# As in the previous tutorials before solving the problem we need to perform
# some transformation of both linear and bilinear forms.

# %%
u_hat = ufl.TrialFunction(V)
J = ufl.derivative(F, Du, u_hat)
J_expanded = ufl.algorithms.expand_derivatives(J)

F_replaced, F_external_operators = replace_external_operators(F)
J_replaced, J_external_operators = replace_external_operators(J_expanded)

F_form = fem.form(F_replaced)
J_form = fem.form(J_replaced)

# %% [markdown]
# ```{note}
#  We remind that in the code above we replace `FEMExternalOperator` objects by
#  their `fem.Function` representatives, the coefficients which are allocated
#  during the call of the `FEMExternalOperator` constructor. The access to these
#  coefficients may be carried out through the field `ref_coefficient` of an
#  `FEMExternalOperator` object. For example, the following code returns the
#  finite coefficient associated with the tangent matrix
#  `C_tang = J_external_operators[0].ref_coefficient`.
# ```

# %% [markdown]
# ### Solving the problem
# 
# Once we prepared the forms containing external operators, we can defind the
# nonlinear problem and its solver. Here we modified the original DOLFINx
# `NonlinearProblem` and called it `NonlinearProblemWithCallback` to let the
# solver evaluate external operators at each iteration. For this matter we define
# the function `constitutive_update` with external operators evaluations and
# update of the internal variable `dp`.

# %%
def constitutive_update():
    evaluated_operands = evaluate_operands(F_external_operators)
    ((_, dp_new),) = evaluate_external_operators(F_external_operators, evaluated_operands)
    _ = evaluate_external_operators(J_external_operators, evaluated_operands)
    # This avoids having to evaluate the external operators of F.
    # sigma.ref_coefficient.x.array[:] = sigma_new
    dp.x.array[:] = dp_new


problem = PETScNonlinearProblem(Du, F_replaced, J_replaced, bcs=bcs, external_callback=constitutive_update)

petsc_options = {
    "snes_type": "qn",
    "snes_qn_type": "lbfgs", #lbfgs broyden, badbroyden
    "snes_qn_m": 100,
    "snes_qn_scale_type": "jacobian", #<diagonal,none,scalar,jacobian> 	
    "snes_qn_restart_type": "none", #<powell,periodic,none> 
    "pc_type": "cholesky", # cholesky >> hypre > gamg,sor 
    "snes_linesearch_type": "basic",
    "ksp_type": "preonly",
    "snes_atol": 1.0e-8,
    "snes_rtol": 1.0e-8,
    "snes_max_it": 100,
    "snes_monitor": "",
}


solver = PETScNonlinearSolver(mesh.comm, problem, petsc_options=petsc_options)  # PETSc.SNES wrapper

# %% [markdown]
# Now we are ready to solve the problem.

# %%
u = fem.Function(V, name="displacement")

x_point = np.array([[R_i, 0, 0]])
cells, points_on_process = find_cell_by_point(mesh, x_point)

q_lim = 2.0 / np.sqrt(3.0) * np.log(R_e / R_i) * sigma_0
num_increments = 20
load_steps = np.linspace(0, 1.1, num_increments, endpoint=True) ** 0.5
loadings = q_lim * load_steps
results = np.zeros((num_increments, 2))

eps = np.finfo(PETSc.ScalarType).eps

for i, loading_v in enumerate(loadings):
    if MPI.COMM_WORLD.rank == 0:
        print(f"Load increment #{i}, load: {loading_v:.3f}")

    loading.value = loading_v
    Du.x.array[:] = eps

    iters, _ = solver.solve(Du)
    print(f"\tInner Newton iterations: {iters}")

    u.x.petsc_vec.axpy(1.0, Du.x.petsc_vec)
    u.x.scatter_forward()

    p.x.petsc_vec.axpy(1.0, dp.x.petsc_vec)
    sigma_n.x.array[:] = sigma.ref_coefficient.x.array

    if len(points_on_process) > 0:
        results[i, :] = (u.eval(points_on_process, cells)[0], loading.value / q_lim)

# %% [markdown]
# ### Post-processing
# 
# In order to verify the correctness of obtained results, we perform their
# comparison against a "pure UFl" implementation. Thanks to simplicity of the von
# Mises model we can express stress tensor and tangent moduli analytically within
# the variational setting and so in UFL. Such a performant implementation is
# presented by the function `plasticity_von_mises_pure_ufl`.

# %%
results_pure_ufl = plasticity_von_mises_pure_ufl(verbose=True)

# %% [markdown]
# Here below we plot the displacement of the inner boundary of the cylinder
# $u_x(R_i, 0)$ with respect to the applied pressure in the von Mises model with
# isotropic hardening. The plastic deformations are reached at the pressure
# $q_{\lim}$ equal to the analytical collapse load for perfect plasticity.

# %%
if len(points_on_process) > 0:
    plt.plot(results_pure_ufl[:, 0], results_pure_ufl[:, 1], "o-", label="pure UFL")
    plt.plot(results[:, 0], results[:, 1], "*-", label="dolfinx-external-operator (Numba)")
    plt.xlabel(r"Displacement of inner boundary $u_x$ at $(R_i, 0)$ [mm]")
    plt.ylabel(r"Applied pressure $q/q_{\text{lim}}$ [-]")
    plt.legend()
    plt.grid()
    plt.savefig("output.png")
    plt.show()

# %% [markdown]
# ## References
# 
# ```{bibliography}
# :filter: docname in docnames
# ```


