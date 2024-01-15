# %% [markdown]
# # Application of external package

# %%
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

# %%


def k(T):
    return 1.0 / (A + B * T)


@jax.jit
def q(T, sigma):
    return -k(T) * sigma


dqdT = jax.jit(jax.jacrev(q, argnums=(0)))
dqdsigma = jax.jit(jax.jacrev(q, argnums=(1)))

# vectorization in the way: vj(T=(batch_size, 1), sigma=(batch_size, 2))
q_global = jax.jit(jax.vmap(q, in_axes=(0, 0)))
dqdT_global = jax.jit(jax.vmap(qjdT, in_axes=(0, 0)))
dqdsigma_global = jax.jit(jax.vmap(dqdsigma, in_axes=(0, 0)))


@jax.jit
def q_impl(T, sigma):
    T_vectorized = T.reshape((num_cells*num_gauss_points, 1))
    sigma_vectorized = sigma.reshape((num_cells*num_gauss_points, 2))
    out = q_global(T_vectorized, sigma_vectorized)
    return out.reshape(-1)


@jax.jit
def dqdT_impl(T, sigma):
    T_vectorized = T.reshape((num_cells*num_gauss_points, 1))
    sigma_vectorized = sigma.reshape((num_cells*num_gauss_points, 2))
    out = dqdT_global(T_vectorized, sigma_vectorized)
    return out.reshape(-1)


@jax.jit
def dqdsigma_impl(T, sigma):
    T_vectorized = T.reshape((num_cells*num_gauss_points, 1))
    sigma_vectorized = sigma.reshape((num_cells*num_gauss_points, 2))
    out = dqdsigma_global(T_vectorized, sigma_vectorized)
    return out.reshape(-1)
