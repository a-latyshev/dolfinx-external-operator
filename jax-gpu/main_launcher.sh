# MESH_SIZE="200"
# sbatch -n 4 launchers/launcher_distributed_jax.sh "$MESH_SIZE"
# sbatch -n 4 launchers/launcher_standard.sh "$MESH_SIZE"

MESH_SIZE="50"
# sbatch -n 4 launchers/launcher_distributed_jax.sh "$MESH_SIZE"
# sbatch -n 4 launchers/launcher_standard.sh "$MESH_SIZE"
sbatch -n 2 launchers/launcher_distributed_jax_several_hosts.sh "$MESH_SIZE"

# sbatch -n 4 launchers/launcher_distributed_jax_several_hosts.sh "$MESH_SIZE"

# MESH_SIZE="200"
# sbatch -n 2 launchers/launcher_distributed_jax_several_cores.sh "$MESH_SIZE"
