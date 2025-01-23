import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'
import jax
jax.config.update("jax_enable_x64", True)
jax.distributed.initialize()
from benchmark import run_benchmark

