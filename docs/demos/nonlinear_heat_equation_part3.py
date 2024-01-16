# %% [markdown]
# # Application of external package
#
# Here we demonstrate the capabilities of our framework to define the behaviour of the external operator via a 3-rd party Python package. In this example, we use the JAX library to define the external operator $\boldsymbol{q}$ and its derivatives. JAX provides powerful automatic differentiation (AD) and function vectorization features as well as it is able to compile functions just-in-time.
#
# The code solving the heat equation problem from the previous notebook remains almost unchanged. The only functions that must be redefined are `q_impl`, `dqdT_impl` and `dqdsigma_impl`.

# %%
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

# %% [markdown]
# By using the vectorization technique we need just to define the behaviour of the function locally at one node of an element. Then we need to reshape the global representation of variables $T$ and $\boldsymbol{\sigma}$ and send them to the vectorized version of the operator $\boldsymbol{q}$ (the function `q_global`). We use the JAX's function `vmap` for the vectorization.

# %%


def k(T):
    return 1.0 / (A + B * T)


@jax.jit
def q(T, sigma):
    return -k(T) * sigma


# vectorization in the following way: q_global(T=(batch_size, 1), sigma=(batch_size, 2))
q_global = jax.jit(jax.vmap(q, in_axes=(0, 0)))


@jax.jit
def q_impl(T, sigma):
    T_vectorized = T.reshape((-1, 1))
    sigma_vectorized = sigma.reshape((-1, 2))
    out = q_global(T_vectorized, sigma_vectorized)
    return out.reshape(-1)

# %% [markdown]
# Thus we do not need to think about how to manipulate data globally. Instead, we just define the local behaviour of the operator.
#
# Once the operator $\boldsymbol{q}$ is defined through a callable Python function we may take the derivative of it using the AD tool via the JAX's function `jacrev`.


# %%
dqdT = jax.jit(jax.jacrev(q, argnums=(0)))
dqdsigma = jax.jit(jax.jacrev(q, argnums=(1)))

# %% [markdown]
# As the function `q` acts locally so do the functions `dqdT` and `dqdsigma` because they have the same signature. We can vectorize them and define the global behaviour of the external operator.

# %%
dqdT_global = jax.jit(jax.vmap(dqdT, in_axes=(0, 0)))
dqdsigma_global = jax.jit(jax.vmap(dqdsigma, in_axes=(0, 0)))


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
