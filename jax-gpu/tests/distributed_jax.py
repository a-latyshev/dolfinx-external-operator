import jax
jax.config.update("jax_enable_x64", True)
jax.distributed.initialize()

import os, sys
path_to_src = os.path.join(os.path.dirname(__file__), "../src")
sys.path.append(path_to_src)
from benchmark import run_benchmark_no_devices

import argparse
parser = argparse.ArgumentParser(description="JAX-CPU-GPU")
parser.add_argument("--N", type=int, default=50, help="Mesh size")
args = parser.parse_args()
N = args.N

params = {"N": N, "type": "distributed_jax"}
run_benchmark_no_devices(params)