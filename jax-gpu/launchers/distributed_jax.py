import jax
jax.config.update("jax_enable_x64", True)
jax.distributed.initialize()
from benchmark import run_benchmark

