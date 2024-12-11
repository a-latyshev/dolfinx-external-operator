import os, sys 
sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)

from mpi4py import MPI

import jax
import jax.lax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)
import time
from timeit import timeit
import matplotlib.pyplot as plt
from jax.sharding import PartitionSpec as P
from jax._src import distributed

cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
local_device_ids = [int(i) for i in cuda_visible_devices.split(",")]
print(f"CUDA_VISIBLE_DEVICES = {local_device_ids} ")
jax.distributed.initialize(local_device_ids=local_device_ids)
print(distributed.global_state.client)
print(distributed.global_state.service)
print(distributed.global_state.process_id)
# import os
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
# jax.distributed.initialize() 
print(f"Backend: {jax.default_backend()}")
print(f"Global devices: {jax.devices()}\n")
print(f"Local devices: {jax.local_devices()}\n")

# A = jax.random.uniform(jax.random.key(0), (3,3,3), dtype=jnp.float64)
# A_sym = 0.5 * (A + A.T)
# result = jax.lax.linalg.eigh(A, lower=True, symmetrize_input=True,
# sort_eigenvalues=True, subset_by_index=None)

# LU_vec = jax.jit(jax.vmap(jax.lax.linalg.lu, in_axes=(0)))
# result = LU_vec(A_sym)
# print(result)


E = 6778  # [MPa] Young modulus
nu = 0.25  # [-] Poisson ratio
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))
C_elas = np.array(
    [
        [lmbda + 2 * mu, lmbda, lmbda, 0],
        [lmbda, lmbda + 2 * mu, lmbda, 0],
        [lmbda, lmbda, lmbda + 2 * mu, 0],
        [0, 0, 0, 2 * mu],
    ],
    dtype=np.float64,
)
def f(eps):
    return C_elas @ eps
f_vec = jax.vmap(f, in_axes=(0))

# def f(x):  # function we're benchmarking (works in both NumPy & JAX)
#   return x.T @ (x - x.mean(axis=0))

f_vec_jit = jax.jit(f_vec)

N_list = 12*np.array([10, 100, 1000, 10000, 100000, 1000000])

def benchmarking(sharding=None):
    def measurement(N, sharding=sharding):
        eps_np = np.ones((N, 4), dtype=np.float64)  # same as JAX default dtype
        # x_np = np.ones((N, N), dtype=np.float64)  # same as JAX default dtype

        time_numpy = timeit(lambda: f_vec(eps_np), number=1)

        start = time.time()
        eps_jax = jax.device_put(eps_np, sharding)  # measure JAX device transfer time
        print(f"Devices: eps_jax = {eps_jax.devices()}")
        time_data_transfer = time.time() - start

        time_jit_overhead = timeit(lambda: f_vec_jit(eps_jax).block_until_ready(), number=1)  # measure JAX compilation time
        time_jax = timeit(lambda: f_vec_jit(eps_jax).block_until_ready(), number=1)  # measure JAX runtime

        return time_numpy, time_data_transfer, time_jit_overhead, time_jax

    all_time = np.empty((len(N_list), 4))
    for i, N in enumerate(N_list):
        all_time[i,:] = measurement(N)
    
    return all_time

devices_list = [1, 2, 3, 4]

if MPI.COMM_WORLD.Get_rank() == 0:

    GPU_scaling_data = np.empty((len(devices_list), 4))
    for n, devices_num in enumerate(devices_list):
        device_mesh = jax.make_mesh((devices_num,), ('x',))
        sharding = jax.sharding.NamedSharding(device_mesh, P('x'))
        time_data_sharding = benchmarking(sharding=sharding)
        GPU_scaling_data[n,:] = time_data_sharding[-1,:]

        fig, ax = plt.subplots()
        ax.loglog(N_list, time_data_sharding[:,0], 'o-', label="NoJIT")
        ax.loglog(N_list, time_data_sharding[:,1], 'o-', label="NumPy->JAX transfer")
        ax.loglog(N_list, time_data_sharding[:,2], 'o-', label="JIT overhead")
        ax.loglog(N_list, time_data_sharding[:,3], 'o-', label="JAX")
        ax.loglog(N_list, time_data_sharding[:,1] + time_data_sharding[:,3], 'o-', label="JAX+transfer")
        ax.set_xlabel("N")
        ax.set_ylabel("Time (s)")
        ax.set_title(rf"$C \cdot \varepsilon$, N times on {devices_num} GPUs")
        ax.legend()
        ax.grid()
        fig.savefig(f"sharding_benchmark_devices_{devices_num}.png")

    fig, ax = plt.subplots()
    ax.loglog(devices_list, GPU_scaling_data[:,0], 'o-', label="NoJIT")
    ax.loglog(devices_list, GPU_scaling_data[:,1], 'o-', label="NumPy->JAX transfer")
    ax.loglog(devices_list, GPU_scaling_data[:,2], 'o-', label="JIT overhead")
    ax.loglog(devices_list, GPU_scaling_data[:,3], 'o-', label="JAX")
    ax.loglog(devices_list, GPU_scaling_data[:,1] + GPU_scaling_data[:,3], 'o-', label="JAX+transfer")
    ax.set_xlabel("Num. of GPUs")
    ax.set_ylabel("Time (s)")
    ax.set_title(r"$C \cdot \varepsilon$, N times")
    ax.legend()
    ax.grid()
    fig.savefig("GPU_scaling.png")

# x_jax_sharded = jax.device_put(x_jax, sharding)
# print(f"After sharding = {x_jax_sharded.devices()}")

print("====Done====")