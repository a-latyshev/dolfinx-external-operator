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
# # Application of external package
#
# Here we demonstrate the capabilities of our framework to define the behaviour of
# the external operator via a 3-rd party Python package. In this example, we use
# the JAX library to define the external operator $\boldsymbol{q}$ and its
# derivatives. JAX provides powerful automatic differentiation (AD) and function
# vectorization features as well as it is able to compile functions
# just-in-time. For more details, visit the
# [JAX's documentation](https://jax.readthedocs.io/en/latest/index.html).
#
# The code solving the heat equation problem from the previous notebook remains
# almost unchanged. The only functions that must be redefined are `q_impl`,
# `dqdT_impl` and `dqdsigma_impl`.

# %%
import jax

# enables the work with double precision under float64
jax.config.update("jax_enable_x64", True)

# %% [markdown]
# We use JAX's auto-vectorisation decorator `jax.vmap` to allow us to write the
# function definition for a single interpolation point, and automatically extend
# that definition to a batch of interpolation points. Then we need to reshape the global
# representation of variables $T$ and $\boldsymbol{\sigma}$ and send them to the
# vectorized version of the operator $\boldsymbol{q}$ (the function `q_global`).
# automatically extend that definition to a batch of interpolation points.

# %%
A = 1.0
B = 1.0


def k(T):
    # In contrast to the previous implementations, the input `T` here is a scalar.
    return 1.0 / (A + B * T)


def q(T, sigma):
    # The input `T is a scalar and sigma is an array with the shape (2,).
    return -k(T) * sigma


# vectorization in the following way: q_global(T=(batch_size, 1),
# sigma=(batch_size, 2))
q_global = jax.vmap(q, in_axes=(0, 0))


@jax.jit
def q_impl(T, sigma):
    # Here we evaluate q globally, so inputs `T` and `sigma` are `np.ndarray`
    # with sizes # cells number * number of interpolation points per cell * local
    # size (which is equal to 1 for T and 2 for sigma). By applying
    # `reshape((-1, 1))` function we prepare the data for batching. For example,
    # `sigma_vectorized` has the shape of (cells number * number of interpolation
    # points per cell, 2)
    T_vectorized = T.reshape((-1, 1))
    sigma_vectorized = sigma.reshape((-1, 2))
    out = q_global(T_vectorized, sigma_vectorized)
    return out.reshape(-1)


# %% [markdown]
# Thus we do not need to think about how to manipulate data globally. Instead, we
# just define the local behaviour of the operator.
#
# Once the operator $\boldsymbol{q}$ is defined through a callable Python function
# we may take the derivative of it using the AD tool via the JAX's function
# `jax.jacfwd`.


# %%
dqdT = jax.jacfwd(q, argnums=(0))
dqdsigma = jax.jacfwd(q, argnums=(1))

# %% [markdown]
# As the function `q` acts locally so do the functions `dqdT` and `dqdsigma`
# because they have the same signature. We can vectorize them and define the
# global behaviour of the external operator.

# %%
dqdT_global = jax.vmap(dqdT, in_axes=(0, 0))
dqdsigma_global = jax.vmap(dqdsigma, in_axes=(0, 0))


@jax.jit
def dqdT_impl(T, sigma):
    T_vectorized = T.reshape((-1, 1))
    sigma_vectorized = sigma.reshape((-1, 2))
    out = dqdT_global(T_vectorized, sigma_vectorized)
    return out.reshape(-1)


@jax.jit
def dqdsigma_impl(T, sigma):
    T_vectorized = T.reshape((-1, 1))
    sigma_vectorized = sigma.reshape((-1, 2))
    out = dqdsigma_global(T_vectorized, sigma_vectorized)
    return out.reshape(-1)


# %% [markdown]
# The decorator `@jax.jit` guarantees that the first function call will take place at compile time.
