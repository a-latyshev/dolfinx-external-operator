# Author: Andrey Latyshev 
# 
# This script is a modified version of the tutorial on the Mohr-Coulomb model
# https://a-latyshev.github.io/dolfinx-external-operator/ and is a part of the
# suplementary material to the paper "Expressing general constitutive models in
# FEniCSx using external operators and algorithmic automatic differentiation"
# (preprint: hal.science/hal-04735022v1). The difference is routed in
# implementation of `C_tang`, where the implicit function theorem is applied
# instead of passing AD through unrolled Newton loop. The script is compatible
# with the JAX version 0.5.0, the image ghcr.io/fenics/dolfinx/dolfinx:nightly
# ID 28eb839aff48 and git hash ecfa039132622b8d6539ef21288cdba7f41dcf76 of the
# repository https://github.com/a-latyshev/dolfinx-external-operator. 
# 
# Typical run: `mpirun -n 2 python demo_plasticity_mohr_coulomb_mpi.py --N 200`

import os, sys 
sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
from mpi4py import MPI
from petsc4py import PETSc

import jax
jax.config.update("jax_enable_x64", True)
import jax.lax
import jax.numpy as jnp
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpltools import annotation  # for slope markers
from solvers import LinearProblem, SNESProblem
from utilities import find_cell_by_point

import basix
import ufl
from dolfinx import default_scalar_type, fem, mesh, common
from dolfinx_external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    evaluate_operands,
    replace_external_operators,
)
E =  6778  # [MPa] Young modulus
nu = 0.25  # [-] Poisson ratio
c = 3.45 # [MPa] cohesion
phi = 30 * np.pi / 180  # [rad] friction angle
psi = 30 * np.pi / 180  # [rad] dilatancy angle
theta_T = 26 * np.pi / 180  # [rad] transition angle as defined by Abbo and Sloan
a = 0.26 * c / np.tan(phi)  # [MPa] tension cuff-off parameter
import argparse

parser = argparse.ArgumentParser(description="Demo Plasticity Mohr Coulomb MPI")
parser.add_argument("--N", type=int, default=200, help="Mesh size")
args = parser.parse_args()
N = args.N
L, H = (1.2, 1.0)
Nx, Ny = (N, N)
gamma = 1.0
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([L, H])], [Nx, Ny])
domain.topology.index_map(0).size_global
k_u = 2
gdim = domain.topology.dim
V = fem.functionspace(domain, ("Lagrange", k_u, (gdim,)))


# Boundary conditions
def on_right(x):
    return np.isclose(x[0], L)


def on_bottom(x):
    return np.isclose(x[1], 0.0)


bottom_dofs = fem.locate_dofs_geometrical(V, on_bottom)
right_dofs = fem.locate_dofs_geometrical(V, on_right)

bcs = [
    fem.dirichletbc(np.array([0.0, 0.0], dtype=PETSc.ScalarType), bottom_dofs, V),
    fem.dirichletbc(np.array([0.0, 0.0], dtype=PETSc.ScalarType), right_dofs, V),
]


def epsilon(v):
    grad_v = ufl.grad(v)
    return ufl.as_vector(
        [
            grad_v[0, 0],
            grad_v[1, 1],
            0.0,
            np.sqrt(2.0) * 0.5 * (grad_v[0, 1] + grad_v[1, 0]),
        ]
    )


k_stress = 2 * (k_u - 1)

dx = ufl.Measure(
    "dx",
    domain=domain,
    metadata={"quadrature_degree": k_stress, "quadrature_scheme": "default"},
)

stress_dim = 2 * gdim
S_element = basix.ufl.quadrature_element(domain.topology.cell_name(), degree=k_stress, value_shape=(stress_dim,))
S = fem.functionspace(domain, S_element)


Du = fem.Function(V, name="Du")
u = fem.Function(V, name="Total_displacement")
du = fem.Function(V, name="du")
v = ufl.TestFunction(V)

sigma = FEMExternalOperator(epsilon(Du), function_space=S)
sigma_n = fem.Function(S, name="sigma_n")

def J3(s):
    return s[2] * (s[0] * s[1] - s[3] * s[3] / 2.0)


def J2(s):
    return 0.5 * jnp.vdot(s, s)


def theta(s):
    J2_ = J2(s)
    arg = -(3.0 * np.sqrt(3.0) * J3(s)) / (2.0 * jnp.sqrt(J2_ * J2_ * J2_))
    arg = jnp.clip(arg, -1.0, 1.0)
    theta = 1.0 / 3.0 * jnp.arcsin(arg)
    return theta


def sign(x):
    return jax.lax.cond(x < 0.0, lambda x: -1, lambda x: 1, x)


def coeff1(theta, angle):
    return np.cos(theta_T) - (1.0 / np.sqrt(3.0)) * np.sin(angle) * np.sin(theta_T)


def coeff2(theta, angle):
    return sign(theta) * np.sin(theta_T) + (1.0 / np.sqrt(3.0)) * np.sin(angle) * np.cos(theta_T)


coeff3 = 18.0 * np.cos(3.0 * theta_T) * np.cos(3.0 * theta_T) * np.cos(3.0 * theta_T)


def C(theta, angle):
    return (
        -np.cos(3.0 * theta_T) * coeff1(theta, angle) - 3.0 * sign(theta) * np.sin(3.0 * theta_T) * coeff2(theta, angle)
    ) / coeff3


def B(theta, angle):
    return (
        sign(theta) * np.sin(6.0 * theta_T) * coeff1(theta, angle) - 6.0 * np.cos(6.0 * theta_T) * coeff2(theta, angle)
    ) / coeff3


def A(theta, angle):
    return (
        -(1.0 / np.sqrt(3.0)) * np.sin(angle) * sign(theta) * np.sin(theta_T)
        - B(theta, angle) * sign(theta) * np.sin(3 * theta_T)
        - C(theta, angle) * np.sin(3.0 * theta_T) * np.sin(3.0 * theta_T)
        + np.cos(theta_T)
    )


def K(theta, angle):
    def K_false(theta):
        return jnp.cos(theta) - (1.0 / np.sqrt(3.0)) * np.sin(angle) * jnp.sin(theta)

    def K_true(theta):
        return (
            A(theta, angle)
            + B(theta, angle) * jnp.sin(3.0 * theta)
            + C(theta, angle) * jnp.sin(3.0 * theta) * jnp.sin(3.0 * theta)
        )

    return jax.lax.cond(jnp.abs(theta) > theta_T, K_true, K_false, theta)


def a_g(angle):
    return a * np.tan(phi) / np.tan(angle)


dev = np.array(
    [
        [2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 0.0],
        [-1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0, 0.0],
        [-1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=PETSc.ScalarType,
)
tr = np.array([1.0, 1.0, 1.0, 0.0], dtype=PETSc.ScalarType)


def surface(sigma_local, angle):
    s = dev @ sigma_local
    I1 = tr @ sigma_local
    theta_ = theta(s)
    return (
        (I1 / 3.0 * np.sin(angle))
        + jnp.sqrt(
            J2(s) * K(theta_, angle) * K(theta_, angle) + a_g(angle) * a_g(angle) * np.sin(angle) * np.sin(angle)
        )
        - c * np.cos(angle)
    )
def f(sigma_local):
    return surface(sigma_local, phi)


def g(sigma_local):
    return surface(sigma_local, psi)


dgdsigma = jax.jacfwd(g)
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))
C_elas = np.array(
    [
        [lmbda + 2 * mu, lmbda, lmbda, 0],
        [lmbda, lmbda + 2 * mu, lmbda, 0],
        [lmbda, lmbda, lmbda + 2 * mu, 0],
        [0, 0, 0, 2 * mu],
    ],
    dtype=PETSc.ScalarType,
)
S_elas = np.linalg.inv(C_elas)
ZERO_VECTOR = np.zeros(stress_dim, dtype=PETSc.ScalarType)


def deps_p(sigma_local, dlambda, deps_local, sigma_n_local):
    sigma_elas_local = sigma_n_local + C_elas @ deps_local
    yielding = f(sigma_elas_local)

    def deps_p_elastic(sigma_local, dlambda):
        return ZERO_VECTOR

    def deps_p_plastic(sigma_local, dlambda):
        return dlambda * dgdsigma(sigma_local)

    return jax.lax.cond(yielding <= 0.0, deps_p_elastic, deps_p_plastic, sigma_local, dlambda)


def r_g(sigma_local, dlambda, deps_local, sigma_n_local):
    deps_p_local = deps_p(sigma_local, dlambda, deps_local, sigma_n_local)
    return sigma_local - sigma_n_local - C_elas @ (deps_local - deps_p_local)


def r_f(sigma_local, dlambda, deps_local, sigma_n_local):
    sigma_elas_local = sigma_n_local + C_elas @ deps_local
    yielding = f(sigma_elas_local)

    def r_f_elastic(sigma_local, dlambda):
        return dlambda

    def r_f_plastic(sigma_local, dlambda):
        return f(sigma_local)

    return jax.lax.cond(yielding <= 0.0, r_f_elastic, r_f_plastic, sigma_local, dlambda)


def r(y_local, deps_local, sigma_n_local):
    sigma_local = y_local[:stress_dim]
    dlambda_local = y_local[-1]

    res_g = r_g(sigma_local, dlambda_local, deps_local, sigma_n_local)
    res_f = r_f(sigma_local, dlambda_local, deps_local, sigma_n_local)

    res = jnp.c_["0,1,-1", res_g, res_f]  # concatenates an array and a scalar
    return res


drdy = jax.jacfwd(r)
Nitermax, tol = 200, 1e-7

ZERO_SCALAR = np.array([0.0])


def return_mapping(deps_local, sigma_n_local):
    """Performs the return-mapping procedure.

    It solves elastoplastic constitutive equations numerically by applying the
    Newton method in a single Gauss point. The Newton loop is implement via
    `jax.lax.while_loop`.

    The function returns `sigma_local` two times to reuse its values after
    differentiation, i.e. as once we apply
    `jax.jacfwd(return_mapping, has_aux=True)` the ouput function will
    have an output of
    `(C_tang_local, (sigma_local, niter_total, yielding, norm_res, dlambda))`.

    Returns:
        sigma_local: The stress at the current Gauss point.
        niter_total: The total number of iterations.
        yielding: The value of the yield function.
        norm_res: The norm of the residuals.
        dlambda: The value of the plastic multiplier.
    """
    niter = 0

    dlambda = ZERO_SCALAR
    sigma_local = sigma_n_local
    y_local = jnp.concatenate([sigma_local, dlambda])

    res = r(y_local, deps_local, sigma_n_local)
    norm_res0 = jnp.linalg.norm(res)

    def cond_fun(state):
        norm_res, niter, _ = state
        return jnp.logical_and(norm_res / norm_res0 > tol, niter < Nitermax)

    def body_fun(state):
        norm_res, niter, history = state

        y_local, deps_local, sigma_n_local, res = history

        j = drdy(y_local, deps_local, sigma_n_local)
        j_inv_vp = jnp.linalg.solve(j, -res)
        y_local = y_local + j_inv_vp

        res = r(y_local, deps_local, sigma_n_local)
        norm_res = jnp.linalg.norm(res)
        history = y_local, deps_local, sigma_n_local, res

        niter += 1

        return (norm_res, niter, history)

    history = (y_local, deps_local, sigma_n_local, res)

    norm_res, niter_total, y_local = jax.lax.while_loop(cond_fun, body_fun, (norm_res0, niter, history))

    sigma_local = y_local[0][:stress_dim]
    dlambda = y_local[0][-1]
    sigma_elas_local = C_elas @ deps_local
    yielding = f(sigma_n_local + sigma_elas_local)

    return sigma_local, (sigma_local, niter_total, yielding, norm_res, dlambda)
def C_tang_local(sigma_local, dlambda_local, deps_local, sigma_n_local):
    # impilicit function theorem and automatic differentiation:
    # http://implicit-layers-tutorial.org/implicit_functions/ 
    y_local = jnp.c_["0,1,-1", sigma_local, dlambda_local]
    j = drdy(y_local, deps_local, sigma_n_local)
    return jnp.linalg.inv(j)[:4,:4] @ C_elas

return_mapping_vec = jax.jit(jax.vmap(return_mapping, in_axes=(0, 0)))
C_tang_vec = jax.jit(jax.vmap(C_tang_local, in_axes=(0, 0, 0, 0)))

def C_tang_impl(deps):
    deps_ = deps.reshape((-1, stress_dim))
    sigma_n_ = sigma_n.x.array.reshape((-1, stress_dim))

    (sigam_global, state) = return_mapping_vec(deps_, sigma_n_)
    sigma_global, niter, yielding, norm_res, dlambda = state

    C_tang_global = C_tang_vec(sigma_global, dlambda, deps_, sigma_n_)

    unique_iters, counts = jnp.unique(niter, return_counts=True)

    max_yielding = jnp.max(yielding)
    max_yielding_rank = MPI.COMM_WORLD.allreduce((max_yielding, MPI.COMM_WORLD.rank), op=MPI.MAXLOC)[1]

    if MPI.COMM_WORLD.rank == max_yielding_rank:
        print("\tInner Newton summary:", flush=True)
        print(f"\t\tUnique number of iterations: {unique_iters}", flush=True)
        print(f"\t\tCounts of unique number of iterations: {counts}", flush=True)
        print(f"\t\tMaximum f: {max_yielding}", flush=True)
        print(f"\t\tMaximum residual: {jnp.max(norm_res)}", flush=True)

    return C_tang_global.reshape(-1), sigma_global.reshape(-1)
def sigma_external(derivatives):
    if derivatives == (1,):
        return C_tang_impl
    else:
        return NotImplementedError


sigma.external_function = sigma_external
q = fem.Constant(domain, default_scalar_type((0, -gamma)))


def F_ext(v):
    return ufl.dot(q, v) * dx


u_hat = ufl.TrialFunction(V)
F = ufl.inner(epsilon(v), sigma) * dx - F_ext(v)
J = ufl.derivative(F, Du, u_hat)
J_expanded = ufl.algorithms.expand_derivatives(J)

F_replaced, F_external_operators = replace_external_operators(F)
J_replaced, J_external_operators = replace_external_operators(J_expanded)

F_form = fem.form(F_replaced)
J_form = fem.form(J_replaced)
Du.x.array[:] = 1.0
sigma_n.x.array[:] = 0.0

evaluated_operands = evaluate_operands(F_external_operators)
_ = evaluate_external_operators(J_external_operators, evaluated_operands)
x_point = np.array([[0, H, 0]])
cells, points_on_process = find_cell_by_point(domain, x_point)

# parameters of the manual Newton method
max_iterations, relative_tolerance = 200, 1e-7

load_steps = np.concatenate([np.linspace(1.5, 22.3, 100)[:-1]])[:10]
num_increments = len(load_steps)
results = np.zeros((num_increments + 1, 2))

external_operator_problem = LinearProblem(J_replaced, -F_replaced, Du, bcs=bcs)

timer = common.Timer("DOLFINx_timer")
timer_total = common.Timer("Total_timer")
local_monitor = {}
performance_monitor = pd.DataFrame({
    "loading_step": np.array([], dtype=np.int64),
    "Newton_iteration": np.array([], dtype=np.int64),
    "matrix_assembling": np.array([], dtype=np.float64),
    "vector_assembling": np.array([], dtype=np.float64),
    "linear_solver": np.array([], dtype=np.float64),
    "constitutive_model_update": np.array([], dtype=np.float64),
})

if MPI.COMM_WORLD.rank == 0:
    print(f"N = {N}, n = {MPI.COMM_WORLD.Get_size()}", flush=True)

timer_total.start()
comm = MPI.COMM_WORLD
for i, load in enumerate(load_steps):
    local_monitor["loading_step"] = i
    q.value = load * np.array([0, -gamma])
    external_operator_problem.assemble_vector()

    residual_0 = external_operator_problem.b.norm()
    residual = residual_0
    Du.x.array[:] = 0

    if MPI.COMM_WORLD.rank == 0:
        print(f"Load increment #{i}, load: {load}, initial residual: {residual_0}", flush=True)

    for iteration in range(0, max_iterations):
        local_monitor["Newton_iteration"] = iteration
        if residual / residual_0 < relative_tolerance:
            break

        if MPI.COMM_WORLD.rank == 0:
            print(f"\tOuter Newton iteration #{iteration}", flush=True)

        timer.start()
        external_operator_problem.assemble_matrix()
        timer.stop()
        local_monitor["matrix_assembling"] = comm.allreduce(timer.elapsed().total_seconds(), op=MPI.MAX)

        timer.start()
        external_operator_problem.solve(du)
        timer.stop()
        local_monitor["linear_solver"] = comm.allreduce(timer.elapsed().total_seconds(), op=MPI.MAX)

        Du.x.petsc_vec.axpy(1.0, du.x.petsc_vec)
        Du.x.scatter_forward()

        timer.start()
        evaluated_operands = evaluate_operands(F_external_operators)
        ((_, sigma_new),) = evaluate_external_operators(J_external_operators, evaluated_operands)
        # Direct access to the external operator values
        sigma.ref_coefficient.x.array[:] = sigma_new
        timer.stop()
        local_monitor["constitutive_model_update"] = comm.allreduce(timer.elapsed().total_seconds(), op=MPI.MAX)

        timer.start()
        external_operator_problem.assemble_vector()
        timer.stop()
        local_monitor["vector_assembling"] = comm.allreduce(timer.elapsed().total_seconds(), op=MPI.MAX)
        performance_monitor.loc[len(performance_monitor.index)] = local_monitor

        residual = external_operator_problem.b.norm()

        if MPI.COMM_WORLD.rank == 0:
            print(f"\tResidual: {residual} Relative: {residual / residual_0}\n", flush=True)

    if MPI.COMM_WORLD.rank == 0:
        print("____________________\n", flush=True)
    u.x.petsc_vec.axpy(1.0, Du.x.petsc_vec)
    u.x.scatter_forward()

    sigma_n.x.array[:] = sigma.ref_coefficient.x.array

    if len(points_on_process) > 0:
        results[i + 1, :] = (-u.eval(points_on_process, cells)[0], load)
timer_total.stop()
total_time = timer_total.elapsed().total_seconds()

if MPI.COMM_WORLD.rank == 0:
    print(f"Total time: {total_time}", flush=True)

n = MPI.COMM_WORLD.Get_size()

if len(points_on_process) > 0:
    l_lim = 6.69
    gamma_lim = l_lim / H * c
    plt.plot(results[:-3, 0], results[:-3, 1], "o-", label=r"$\gamma$")
    plt.axhline(y=gamma_lim, color="r", linestyle="--", label=r"$\gamma_\text{lim}$")
    plt.xlabel(r"Displacement of the slope $u_x$ at $(0, H)$ [mm]")
    plt.ylabel(r"Soil self-weight $\gamma$ [MPa/mm$^3$]")
    plt.grid()
    plt.legend()
    plt.savefig(f"mc_mpi_{N}x{N}_n_{n}.png")

import pickle
performance_data = {"total_time": total_time, "performance_monitor": performance_monitor}
with open(f"performance_data_{N}x{N}_n_{n}.pkl", "wb") as f:
        pickle.dump(performance_data, f)
