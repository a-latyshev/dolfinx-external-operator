import os, sys 
sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)

from mpi4py import MPI
from dolfinx import mesh, fem, common
import basix
import jax
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_transfer_guard", "disallow")
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax._src import distributed
import numpy as np
import pickle
from constitutive_model import constitutive_response

def data_splitter(data, n_devices: int):
    """Reshape data to be distributed among devices.
    
    If the data size is not divisible by the number of devices, the data is
    padded with zeros.
    """
    local_size = data.shape[0] # local mpi size
    if local_size % n_devices != 0:
        splitted_data = np.full((local_size + (n_devices - local_size % n_devices), int(data.size/local_size)), 0.00000001)        
        splitted_data[:local_size, :] = data.reshape((local_size, -1))
    else:
        splitted_data = data
    return splitted_data.reshape(n_devices, -1, data.shape[1])

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
    local_jax_devices = len(jax.local_devices())

    # input data
    if MPI.COMM_WORLD.rank == 0:
        print(f"type: {benchmark_type} N: {N}", flush=True)
        print(f"JAX backend: {jax.default_backend()}", flush=True)
        print(f"JAX global devices: {jax.devices()}", flush=True)
        # print(f"JAX client: {distributed.global_state.client}", flush=True)
        # print(f"JAX service: {distributed.global_state.service}", flush=True)
        print(f"Globally: #DoFs(Q): {Q.dofmap.index_map.size_global:6d}\n", flush=True)
    print(f"rank = {MPI.COMM_WORLD.rank} Locally: #DoFs(Q): {Q.dofmap.index_map.size_local:6d} shape(sigma_n_np): {sigma_n_np.shape} JAX process_id: {distributed.global_state.process_id} JAX #(local devices): {local_jax_devices} JAX local devices: {jax.local_devices()}", flush=True)

    local_mpi_size = int(sigma_n.x.array.shape[0]/stress_dim)
    dsigma_path_np = np.zeros((local_mpi_size, stress_dim))
    R = 0.1
    angle = 0
    # formulas for angle \in [-pi/6, pi/6]
    for i in range(local_mpi_size):
        angle = np.random.uniform(-np.pi/6, np.pi/6)
        dsigma_path_np[i,0] = (R / np.sqrt(2)) * (np.cos(angle) + np.sin(angle) / np.sqrt(3))
        dsigma_path_np[i,1] = (R / np.sqrt(2)) * (-2 * np.sin(angle) / np.sqrt(3))
        dsigma_path_np[i,2] = (R / np.sqrt(2)) * (np.sin(angle) / np.sqrt(3) - np.cos(angle))

    timer = common.Timer("Total_timer")
    timer_in = common.Timer("Inner_timer")
    time_data_transfer = 0
    time_data_copy = 0
    time_computation = 0
    dconstitutive_response = jax.jacfwd(constitutive_response, has_aux=True)
    dconstitutive_response_v = jax.jit(jax.vmap(dconstitutive_response, in_axes=(0, 0)))
    dconstitutive_response_p = jax.pmap(dconstitutive_response, in_axes=(0, 0))
    dconstitutive_response_pv = jax.pmap(jax.vmap(dconstitutive_response, in_axes=(0, 0)), in_axes=(0, 0))
    sigma_n_np_split = data_splitter(sigma_n_np, local_jax_devices)
    dsigma_path_np_split = data_splitter(dsigma_path_np, local_jax_devices)
    
    device_mesh = jax.make_mesh((local_jax_devices,), ('x',))
    sharding = jax.sharding.NamedSharding(device_mesh, P('x'))
    sigma_n_jax = jax.device_put(sigma_n_np_split, sharding)

    timer.start()
    N_loads = 100  # number of loadings or paths

    for i in range(N_loads):
        timer_in.start()
        dsigma_path_np_split.reshape((-1, stress_dim))[:local_mpi_size, :] = dsigma_path_np
        # dsigma_path = jax.device_put(dsigma_path_np, sharding)
        timer_in.stop()
        time_data_transfer += MPI.COMM_WORLD.allreduce(timer_in.elapsed()[0], op=MPI.MAX)

        timer_in.start()
        _, (sigma_corrected_split, yielding) = dconstitutive_response_pv(dsigma_path_np_split, sigma_n_jax)
        timer_in.stop()
        time_computation += MPI.COMM_WORLD.allreduce(timer_in.elapsed()[0], op=MPI.MAX)
        
        timer_in.start()
        sigma_n_jax = sigma_corrected_split
        tmp = jax.device_get(sigma_corrected_split)
        # tmp = sigma_corrected_split.copy_to_host_async()
        sigma_n.x.array[:] = tmp.reshape(-1)[:sigma_n.x.array.size]
        timer_in.stop()
        time_data_copy += MPI.COMM_WORLD.allreduce(timer_in.elapsed()[0], op=MPI.MAX)
        if MPI.COMM_WORLD.rank == 0:
            # print(f"it#{i} sigma shape {sigma_corrected_split.shape}")
            print(f"it#{i} yielding max: {jnp.max(yielding)}")
    timer.stop()
    total_time = MPI.COMM_WORLD.allreduce(timer.elapsed()[0], op=MPI.MAX)
    n = MPI.COMM_WORLD.Get_size()
    pkl_file = benchmark_type + f"_N_{N}_n_{n}_d_{len(jax.devices())}.pkl"
    if MPI.COMM_WORLD.rank == 0:
        print(f"Total time: {total_time} \n", flush=True)
        print(f"Loaded pkl_file: {pkl_file}", flush=True)
    print(f"rank = {MPI.COMM_WORLD.rank} sigma_corrected is on: {sigma_corrected_split.devices()}", flush=True)
    output = {**params, "total_time": total_time, "time_data_copy": time_data_copy, "time_data_transfer": time_data_transfer, "time_computation": time_computation,"n": n, "pkl_file": pkl_file}
    with open(f"output/"+pkl_file, "wb") as f:
        pickle.dump(output, f)
    return output

def run_benchmark_no_devices(params):
    N = params["N"]
    benchmark_type = params["type"]
    
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, mesh.CellType.triangle)
    stress_dim = 4
    Q_element = basix.ufl.quadrature_element(domain.topology.cell_name(), degree=1, value_shape=(stress_dim,))
    Q = fem.functionspace(domain, Q_element)
    sigma_n = fem.Function(Q)
    sigma = fem.Function(Q)
    sigma_n_np = sigma_n.x.array.reshape((-1, stress_dim))
    # sigma_n_jax = jnp.array(sigma_n_np)
    # input data
    local_jax_devices = len(jax.local_devices())
    if MPI.COMM_WORLD.rank == 0:
        print(f"type: {benchmark_type} N: {N}", flush=True)
        print(f"JAX backend: {jax.default_backend()}", flush=True)
        print(f"JAX global devices: {jax.devices()}", flush=True)
        print(f"Globally: #DoFs(Q): {Q.dofmap.index_map.size_global:6d}\n", flush=True)
    print(f"rank = {MPI.COMM_WORLD.rank} Locally: #DoFs(Q): {Q.dofmap.index_map.size_local:6d} shape(sigma_n_np): {sigma_n_np.shape} JAX process_id: {distributed.global_state.process_id} JAX #(local devices): {local_jax_devices} JAX local devices: {jax.local_devices()}", flush=True)
    # print(f"rank = {MPI.COMM_WORLD.rank} JAX local devices: {local_jax_devices}", flush=True)
    # print(f"rank = {MPI.COMM_WORLD.rank} JAX process_id: {distributed.global_state.process_id}", flush=True)

    local_mpi_size = int(sigma_n.x.array.shape[0]/stress_dim)
    dsigma_path_np = np.zeros((local_mpi_size, stress_dim))
    R = 0.1
    angle = 0
    # formulas for angle \in [-pi/6, pi/6]
    for i in range(local_mpi_size):
        angle = np.random.uniform(-np.pi/6, np.pi/6)
        dsigma_path_np[i,0] = (R / np.sqrt(2)) * (np.cos(angle) + np.sin(angle) / np.sqrt(3))
        dsigma_path_np[i,1] = (R / np.sqrt(2)) * (-2 * np.sin(angle) / np.sqrt(3))
        dsigma_path_np[i,2] = (R / np.sqrt(2)) * (np.sin(angle) / np.sqrt(3) - np.cos(angle))

    timer = common.Timer("Total_timer")
    timer_in = common.Timer("Inner_timer")
    time_data_transfer = 0
    time_data_copy = 0
    time_computation = 0
    dconstitutive_response = jax.jacfwd(constitutive_response, has_aux=True)
    dconstitutive_response_v = jax.jit(jax.vmap(dconstitutive_response, in_axes=(0, 0)))

    timer.start()
    N_loads = 100  # number of loadings or paths

    for i in range(N_loads):
        timer_in.start()
        _, (sigma_corrected, yielding) = dconstitutive_response_v(dsigma_path_np, sigma_n_np)
        timer_in.stop()
        time_computation += MPI.COMM_WORLD.allreduce(timer_in.elapsed()[0], op=MPI.MAX)
        
        timer_in.start()
        sigma_n_np = np.asarray(sigma_corrected)
        sigma_n.x.array[:] = sigma_corrected.reshape(-1)
        timer_in.stop()
        time_data_copy += MPI.COMM_WORLD.allreduce(timer_in.elapsed()[0], op=MPI.MAX)
        if MPI.COMM_WORLD.rank == 0:
            print(f"it#{i} yielding max: {jnp.max(yielding)}")
    timer.stop()
    total_time = MPI.COMM_WORLD.allreduce(timer.elapsed()[0], op=MPI.MAX)
    n = MPI.COMM_WORLD.Get_size()
    pkl_file = benchmark_type + f"_N_{N}_n_{n}.pkl"
    if MPI.COMM_WORLD.rank == 0:
        print(f"Total time: {total_time} \n", flush=True)
        print(f"Loaded pkl_file: {pkl_file}", flush=True)
    print(f"rank = {MPI.COMM_WORLD.rank} sigma_corrected is on: {sigma_corrected.devices()}", flush=True)

    output = {**params, "total_time": total_time, "time_data_copy": time_data_copy, "time_data_transfer": time_data_transfer, "time_computation": time_computation,"n": n, "pkl_file": pkl_file}
    with open(f"output/"+pkl_file, "wb") as f:
        pickle.dump(output, f)
    return output