import argparse
parser = argparse.ArgumentParser(description="JAX-CPU-GPU")
parser.add_argument("--N", type=int, default=50, help="Mesh size")
parser.add_argument("--d", type=int, default=2, help="JAX devices")
args = parser.parse_args()
d = args.d

import os
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={d}'
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8 --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4 inter_op_parallelism_threads=4'
import jax
jax.config.update("jax_enable_x64", True)
# jax.distributed.initialize() # https://github.com/jax-ml/jax/issues/5022

import os, sys
path_to_src = os.path.join(os.path.dirname(__file__), "../src")
sys.path.append(path_to_src)
from benchmark import run_benchmark


N = args.N
params = {"N": N, "d":d, "type": "jax_multithreading"}
run_benchmark(params)