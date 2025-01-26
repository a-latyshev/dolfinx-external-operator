# MESH_SIZE="200"
# sbatch -n 4 launchers/launcher_distributed_jax.sh "$MESH_SIZE"
# sbatch -n 4 launchers/launcher_standard.sh "$MESH_SIZE"

MESH_SIZE="50"
# sbatch -n 4 launchers/launcher_distributed_jax.sh "$MESH_SIZE"
# sbatch -n 4 launchers/launcher_standard.sh "$MESH_SIZE"
# sbatch -n 2 launchers/launcher_distributed_jax_several_hosts.sh "$MESH_SIZE"

# sbatch -n 4 launchers/launcher_distributed_jax_several_hosts.sh "$MESH_SIZE"

# MESH_SIZE="200"
# sbatch -n 2 launchers/launcher_distributed_jax_several_cores.sh "$MESH_SIZE"


# MESH_SIZE="200"
# JAX_DEVICES="1"
# sbatch -n 8 launchers/launcher_jax_multithreading.sh "$MESH_SIZE" "$JAX_DEVICES"
# JAX_DEVICES="2"
# sbatch -n 8 launchers/launcher_jax_multithreading.sh "$MESH_SIZE" "$JAX_DEVICES"
# JAX_DEVICES="4"
# sbatch -n 8 launchers/launcher_jax_multithreading.sh "$MESH_SIZE" "$JAX_DEVICES"
# JAX_DEVICES="8"
# sbatch -n 8 launchers/launcher_jax_multithreading.sh "$MESH_SIZE" "$JAX_DEVICES"
# JAX_DEVICES="16"
# sbatch -n 8 launchers/launcher_jax_multithreading.sh "$MESH_SIZE" "$JAX_DEVICES"
# JAX_DEVICES="32"
# sbatch -n 8 launchers/launcher_jax_multithreading.sh "$MESH_SIZE" "$JAX_DEVICES"
# JAX_DEVICES="64"
# sbatch -n 8 launchers/launcher_jax_multithreading.sh "$MESH_SIZE" "$JAX_DEVICES"
# JAX_DEVICES="132"
# sbatch -n 8 launchers/launcher_jax_multithreading.sh "$MESH_SIZE" "$JAX_DEVICES"
# JAX_DEVICES="256"
# sbatch -n 8 launchers/launcher_jax_multithreading.sh "$MESH_SIZE" "$JAX_DEVICES"


# MESH_SIZE="200"
# sbatch -n 8 launchers/launcher_standard.sh "$MESH_SIZE"

MESH_SIZE="300"
# sbatch -n 2 launchers/launcher_jax_gpu.sh "$MESH_SIZE"
# sbatch -n 4 launchers/launcher_jax_gpu.sh "$MESH_SIZE"
# sbatch -n 8 launchers/launcher_jax_gpu.sh "$MESH_SIZE"
# sbatch -n 16 launchers/launcher_jax_gpu.sh "$MESH_SIZE"

sbatch -n 2 launchers/launcher_standard.sh "$MESH_SIZE"
sbatch -n 4 launchers/launcher_standard.sh "$MESH_SIZE"
sbatch -n 8 launchers/launcher_standard.sh "$MESH_SIZE"
sbatch -n 16 launchers/launcher_standard.sh "$MESH_SIZE"