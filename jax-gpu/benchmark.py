import os
from mpi4py import MPI
from dolfinx import mesh, fem, common
import basix
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from constitutive_model import constitutive_response
import numpy as np

def run_benchmark(N):
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, mesh.CellType.triangle)
    stress_dim = 4
    Q_element = basix.ufl.quadrature_element(domain.topology.cell_name(), degree=1, value_shape=(stress_dim,))
    Q = fem.functionspace(domain, Q_element)
    sigma_n = fem.Function(Q)
    sigma = fem.Function(Q)
    sigma_n_np = sigma_n.x.array.reshape((-1, stress_dim))

    local_size = int(sigma_n.x.array.shape[0]/stress_dim)
    dsigma_path_np = np.zeros((local_size, stress_dim))
    R = 0.1
    angle = 0
    # formulas for angle \in [-pi/6, pi/6]
    for i in range(local_size):
        angle = np.random.uniform(-np.pi/6, np.pi/6)
        dsigma_path_np[i,0] = (R / np.sqrt(2)) * (np.cos(angle) + np.sin(angle) / np.sqrt(3))
        dsigma_path_np[i,1] = (R / np.sqrt(2)) * (-2 * np.sin(angle) / np.sqrt(3))
        dsigma_path_np[i,2] = (R / np.sqrt(2)) * (np.sin(angle) / np.sqrt(3) - np.cos(angle))
    # input data
    if MPI.COMM_WORLD.rank == 0:
        print(f"Backend: {jax.default_backend()}")
        print(f"Global devices: {jax.devices()}")
        print(f"Globally: #DoFs(Q): {Q.dofmap.index_map.size_global:6d}\n", flush=True)
    print(f"rank = {MPI.COMM_WORLD.rank} Locally: #DoFs(Q): {Q.dofmap.index_map.size_local:6d} shape(sigma_n_np): {sigma_n_np.shape}", flush=True)
    print(f"rank = {MPI.COMM_WORLD.rank} Local devices: {jax.local_devices()}", flush=True)

    timer = common.Timer("Total_timer")
    dconstitutive_response = jax.jacfwd(constitutive_response, has_aux=True)
    dconstitutive_response_v = jax.jit(jax.vmap(dconstitutive_response, in_axes=(0, 0)))
    timer.start()
    N_loads = 100  # number of loadings or paths
    for i in range(N_loads):
        _, (sigma_corrected, yielding) = dconstitutive_response_v(dsigma_path_np, sigma_n_np)
        sigma_n_np[:] = sigma_corrected
        if MPI.COMM_WORLD.rank == 0:
            print(f"yielding max: {jnp.max(yielding)}")
    timer.stop()
    total_time = MPI.COMM_WORLD.allreduce(timer.elapsed()[0], op=MPI.MAX)
    if MPI.COMM_WORLD.rank == 0:
        print(f"Total time: {total_time} \n", flush=True)
    print(f"rank = {MPI.COMM_WORLD.rank} sigma_corrected is on: {sigma_corrected.devices()}", flush=True)

    return total_time