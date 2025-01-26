import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=2'
import jax
jax.config.update("jax_enable_x64", True)
jax.distributed.initialize() # https://github.com/jax-ml/jax/issues/5022

import os, sys
path_to_src = os.path.join(os.path.dirname(__file__), "../src")
sys.path.append(path_to_src)
from benchmark import run_benchmark

import argparse
parser = argparse.ArgumentParser(description="JAX-CPU-GPU")
parser.add_argument("--N", type=int, default=50, help="Mesh size")
args = parser.parse_args()
N = args.N

params = {"N": N, "type": "distributed_jax_several_hosts"}
run_benchmark(params)