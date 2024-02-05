from dolfinx_external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    evaluate_operands,
    replace_external_operators,
)
from dolfinx import common, fem
import ufl
import basix
from utilities import build_cylinder_quarter, find_cell_by_point
from solvers import LinearProblem
import jax.numpy as jnp
from mpi4py import MPI
from petsc4py import PETSc

import matplotlib.pyplot as plt
import numba
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # replace by JAX_ENABLE_X64=True


E = 6778  # [MPa]
nu = 0.25  # [-]
lambda_ = E*nu/(1+nu)/(1-2*nu)
mu_ = E/2./(1+nu)  # 6.9#[GPa]
# sigma_u = 27.6 #[MPa]
P_i_value = 3.45  # [MPa]

c = 3.45  # [MPa]
phi = 30 * np.pi / 180
psi = 30 * np.pi / 180
theta_T = 20 * np.pi / 180
a = 0.5 * c / np.tan(phi)

l, m = lambda_, mu_
C_elas = np.array([[l+2*m, l, l, 0],
                   [l, l+2*m, l, 0],
                   [l, l, l+2*m, 0],
                   [0, 0, 0, 2*m]], dtype=PETSc.ScalarType)

dev = np.array([[2/3., -1/3., -1/3., 0],
                [-1/3., 2/3., -1/3., 0],
                [-1/3., -1/3., 2/3., 0],
                [0, 0, 0, 1.]], dtype=PETSc.ScalarType)

TPV = np.finfo(PETSc.ScalarType).eps  # très petite value
SQRT2 = np.sqrt(2.)

zero_vec = jnp.array([TPV, TPV, TPV, TPV])


tr = np.array([1, 1, 1, 0])


@jax.jit
def J3(sigma_local):
    return sigma_local[2] * (sigma_local[0]*sigma_local[1] - sigma_local[3]*sigma_local[3]/2.)


@jax.jit
def sign(x):
    return jax.lax.cond(x < -TPV, lambda x: -1, lambda x: 1, x)


def coeff1(theta, angle):
    return jnp.cos(theta_T) - 1 / (jnp.sqrt(3) * jnp.sin(angle) * sign(theta) * jnp.sin(theta_T))


def coeff2(theta, angle):
    return sign(theta) * jnp.sin(theta_T) + 1 / (jnp.sqrt(3) * jnp.sin(angle) * jnp.cos(theta_T))


coeff3 = 18 * jnp.cos(3*theta_T)*jnp.cos(3*theta_T)*jnp.cos(3*theta_T)


def C(theta, angle):
    return (- jnp.cos(3*theta_T) * coeff1(theta, angle) - 3 * sign(theta) * jnp.sin(3*theta_T) * coeff2(theta, angle)) / coeff3


def B(theta, angle):
    return (sign(theta) * jnp.sin(6*theta_T) * coeff1(theta, angle) - 6 * jnp.cos(6*theta_T) * coeff2(theta, angle)) / coeff3


def A(theta, angle):
    return - 1/jnp.sqrt(3) * jnp.sin(angle) * sign(theta) * jnp.sin(theta_T) - B(theta, angle) * sign(theta) * jnp.sin(theta_T) - C(theta, angle) * jnp.sin(3 * theta_T)*jnp.sin(3 * theta_T) + jnp.cos(theta_T)


@jax.jit
def K(theta, angle):
    def K_true(theta, angle): return jnp.cos(theta) - 1 / \
        jnp.sqrt(3) * jnp.sin(angle) * jnp.sin(theta)
    def K_false(theta, angle): return A(theta, angle) + B(theta, angle) * \
        jnp.sin(3*theta) + C(theta, angle) * jnp.sin(3*theta)*jnp.sin(3*theta)
    return K_true(theta, angle)
    return jax.lax.cond(jnp.abs(theta) < theta_T, K_true, K_false, theta, angle)


@jax.jit
def a_G(angle):
    return a * jnp.tan(phi) / jnp.tan(angle)


@jax.jit
def surface(sigma_local, angle):
    s = dev @ sigma_local
    I1 = tr @ sigma_local
    J2 = 0.5 * jnp.vdot(s, s)
    arg = -3*jnp.sqrt(3) * J3(s) / (2 * jnp.sqrt(J2*J2*J2))
    arg = jnp.clip(arg, -1, 1)
    # arcsin returns nan if its argument is equal to -1 + smth around 1e-16!!!
    theta = 1/3. * jnp.arcsin(arg)
    return I1/3 * jnp.sin(angle) + jnp.sqrt(J2 * K(theta, angle)*K(theta, angle) + a_G(angle)*a_G(angle) * jnp.sin(angle)*jnp.sin(angle)) - c * jnp.cos(angle)


f_MC = jax.jit(lambda sigma_local: surface(sigma_local, phi))
g_MC = jax.jit(lambda sigma_local: surface(sigma_local, psi))


@jax.jit
def theta(sigma_local):
    s = dev @ sigma_local
    J2 = 0.5 * jnp.vdot(s, s)
    arg = -3*jnp.sqrt(3) * J3(s) / (2 * jnp.sqrt(J2*J2*J2))
    arg = jnp.clip(arg, -1, 1)
    return 1/3. * jnp.arcsin(arg)


dthetadsigma = jax.jit(jax.jacfwd(theta, argnums=(0)))
# dthetadJ2 = jax.jit(jax.jacfwd(theta, argnums=(1)))

deps_local = jnp.array([TPV, 0.0, 0.0, 0.0])
sigma_local = jnp.array([1, 1, 1.0, 1.0])
print(f"f_MC = {f_MC(sigma_local)}, dfdsigma = {sigma_local},\ntheta = {theta(sigma_local)}, dtheta = {dthetadsigma(sigma_local)}")

deps_local = jnp.array([0.0006, 0.0003, 0.0, 0.0])
sigma_local = C_elas @ deps_local
print(f"f_MC = {f_MC(sigma_local)}, dfdsigma = {sigma_local},\ntheta = {theta(sigma_local)}, dtheta = {dthetadsigma(sigma_local)}")
