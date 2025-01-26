import os, sys 
sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)

from mpi4py import MPI
from dolfinx import mesh, fem, common
import basix
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax._src import distributed
import numpy as np
import pickle
from constitutive_model import constitutive_response

def run_benchmark(params):
    N = params["N"]
    benchmark_type = params["type"]
    
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, mesh.CellType.triangle)
    stress_dim = 4
    Q_element = basix.ufl.quadrature_element(domain.topology.cell_name(), degree=1, value_shape=(stress_dim,))
    Q = fem.functionspace(domain, Q_element)
    sigma_n = fem.Function(Q)
    sigma = fem.Function(Q)
    sigma_n_np = sigma_n.x.array.reshape((-1, stress_dim))
    # input data
    if MPI.COMM_WORLD.rank == 0:
        print(f"type: {benchmark_type} N: {N}", flush=True)
        print(f"JAX backend: {jax.default_backend()}", flush=True)
        print(f"JAX global devices: {jax.devices()}", flush=True)
        # print(f"JAX client: {distributed.global_state.client}", flush=True)
        # print(f"JAX service: {distributed.global_state.service}", flush=True)
        print(f"Globally: #DoFs(Q): {Q.dofmap.index_map.size_global:6d}\n", flush=True)
    print(f"rank = {MPI.COMM_WORLD.rank} Locally: #DoFs(Q): {Q.dofmap.index_map.size_local:6d} shape(sigma_n_np): {sigma_n_np.shape}", flush=True)
    
    print(f"rank = {MPI.COMM_WORLD.rank} JAX local devices: {jax.local_devices()}", flush=True)
    print(f"rank = {MPI.COMM_WORLD.rank} JAX process_id: {distributed.global_state.process_id}", flush=True)

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

    timer = common.Timer("Total_timer")
    timer_in = common.Timer("Inner_timer")
    time_data_transfer = 0
    time_data_copy = 0
    dconstitutive_response = jax.jacfwd(constitutive_response, has_aux=True)
    dconstitutive_response_v = jax.jit(jax.vmap(dconstitutive_response, in_axes=(0, 0)))
    dconstitutive_response_p = jax.pmap(dconstitutive_response, in_axes=(0, 0))
    timer.start()
    N_loads = 100  # number of loadings or paths

    device_mesh = jax.make_mesh((len(jax.local_devices()),), ('x',))
    sharding = jax.sharding.NamedSharding(device_mesh, P('x'))
    for i in range(N_loads):
        # timer_in.start()
        # dsigma_path = jax.device_put(dsigma_path_np, sharding)
        # timer_in.stop()
        dsigma_path = dsigma_path_np
        # time_data_transfer += MPI.COMM_WORLD.allreduce(timer_in.elapsed()[0], op=MPI.MAX)

        _, (sigma_corrected, yielding) = dconstitutive_response_p(dsigma_path, sigma_n_np)
        timer_in.start()
        sigma_n_np[:] = sigma_corrected
        timer_in.stop()
        time_data_copy += MPI.COMM_WORLD.allreduce(timer_in.elapsed()[0], op=MPI.MAX)
        if MPI.COMM_WORLD.rank == 0:
            print(f"sigma shape {sigma_corrected.shape}")
            print(f"yielding max: {jnp.max(yielding)}")
    timer.stop()
    total_time = MPI.COMM_WORLD.allreduce(timer.elapsed()[0], op=MPI.MAX)
    if MPI.COMM_WORLD.rank == 0:
        print(f"Total time: {total_time} \n", flush=True)
    print(f"rank = {MPI.COMM_WORLD.rank} sigma_corrected is on: {sigma_corrected.devices()}", flush=True)

    n = MPI.COMM_WORLD.Get_size()
    pkl_file = benchmark_type + f"_N_{N}_n_{n}.pkl"
    output = {**params, "total_time": total_time, "time_data_copy": time_data_copy, "n": n, "pkl_file": pkl_file}
    with open(f"output/"+pkl_file, "wb") as f:
        pickle.dump(output, f)
    return output