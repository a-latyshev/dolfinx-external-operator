import argparse
parser = argparse.ArgumentParser(description="JAX-CPU-GPU")
parser.add_argument("--N", type=int, default=50, help="Mesh size")
parser.add_argument("--d", type=int, default=2, help="JAX devices")
args = parser.parse_args()
d = args.d

import os, sys
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
import jax
jax.config.update("jax_enable_x64", True)
# jax.distributed.initialize()

path_to_src = os.path.join(os.path.dirname(__file__), "../src")
sys.path.append(path_to_src)
from benchmark import run_benchmark

import argparse
parser = argparse.ArgumentParser(description="JAX-CPU-GPU")
parser.add_argument("--N", type=int, default=50, help="Mesh size")
args = parser.parse_args()
N = args.N

params = {"N": N, "d":d, "type": "jax_gpu"}
run_benchmark(params)