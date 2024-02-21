# %% [markdown]
# Based on https://fenics-optim.readthedocs.io/en/latest/demos/limit_analysis_3D_SDP.html 

# %%
from petsc4py import PETSc

import jax
jax.config.update("jax_enable_x64", True) # replace by JAX_ENABLE_X64=True
import jax.numpy as jnp

import numpy as np
import matplotlib.pyplot as plt

# %%
E = 70e3
nu = 0.3
lambda_ = E*nu/(1+nu)/(1-2*nu)
mu_ = E/2./(1+nu)
sig0 = 250.  # yield strength
Et = E/100.  # tangent modulus
H = E*Et/(E-Et)  # hardening modulus
TPV = np.finfo(PETSc.ScalarType).eps # très petite value 
SQRT2 = np.sqrt(2.)

c = 3.45 #[MPa]
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

tr = np.array([1, 1, 1, 0])

zero_vec = jnp.array([TPV, TPV, TPV, TPV])

Nitermax, tol = 200, 1e-8

# %%
@jax.jit
def det(Mandel_vec):
    """Returns the derminante of a tensor written through Mandel notation."""
    return Mandel_vec[2] * (Mandel_vec[0]*Mandel_vec[1] - Mandel_vec[3]*Mandel_vec[3]/2.)

@jax.jit
def eigensum(Mandel_vec):
    """Returns the sum of eigenvalues modules of a tensor written through Mandel notation."""
    lambda_I = jnp.abs(Mandel_vec[2])
    A = tr @ Mandel_vec
    B = jnp.sqrt(A**2 - 4*det(Mandel_vec))
    lambda_II = jnp.abs(0.5*(A + B))
    lambda_III = jnp.abs(0.5*(A - B))
    return lambda_I + lambda_II + lambda_III

# %%
@jax.jit
def pi(deps_local):
    condition = tr @ deps_local - np.sin(phi) * eigensum(deps_local)
    # == tol and tr @ deps_local > np.sin(phi) * eigensum(deps_local)
    pi_positive = lambda eps_local: c * tr @ eps_local / np.tan(phi)
    pi_negative = lambda eps_local: jnp.inf
    return jax.lax.cond(condition >= TPV, pi_positive, pi_negative, deps_local)

dpiddeps = jax.jit(jax.jacfwd(pi, argnums=(0)))

# %%
deps_local = jnp.array([1., 0., 0., 0.])
pi(deps_local)

# %%
deps_local = jnp.array([0.001, 30., 20., 0.])
pi(deps_local)

# %%
tr @ deps_local 

# %%
tr @ deps_local - np.sin(phi) * eigensum(deps_local)


# %%
dpiddeps(deps_local)

# %%



