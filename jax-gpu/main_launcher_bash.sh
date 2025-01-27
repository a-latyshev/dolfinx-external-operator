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


MESH_SIZE=200
JAX_DEVICES=1
mpirun -n 8 python tests/jax_multithreading.py --N $MESH_SIZE --d $JAX_DEVICES
JAX_DEVICES=2
mpirun -n 8 python tests/jax_multithreading.py --N $MESH_SIZE --d $JAX_DEVICES
JAX_DEVICES=4
mpirun -n 8 python tests/jax_multithreading.py --N $MESH_SIZE --d $JAX_DEVICES
JAX_DEVICES=8
mpirun -n 8 python tests/jax_multithreading.py --N $MESH_SIZE --d $JAX_DEVICES
JAX_DEVICES=16
mpirun -n 8 python tests/jax_multithreading.py --N $MESH_SIZE --d $JAX_DEVICES
JAX_DEVICES=32
mpirun -n 8 python tests/jax_multithreading.py --N $MESH_SIZE --d $JAX_DEVICES
JAX_DEVICES=64
mpirun -n 8 python tests/jax_multithreading.py --N $MESH_SIZE --d $JAX_DEVICES
JAX_DEVICES=128
mpirun -n 8 python tests/jax_multithreading.py --N $MESH_SIZE --d $JAX_DEVICES
# JAX_DEVICES=256
# mpirun -n 8 python tests/jax_multithreading.py --N $MESH_SIZE --d $JAX_DEVICES
# MESH_SIZE="200"
# sbatch -n 8 launchers/launcher_standard.sh "$MESH_SIZE"

MESH_SIZE="300"
# sbatch -n 2 launchers/launcher_jax_gpu.sh "$MESH_SIZE"
# sbatch -n 4 launchers/launcher_jax_gpu.sh "$MESH_SIZE"
# sbatch -n 8 launchers/launcher_jax_gpu.sh "$MESH_SIZE"
# sbatch -n 16 launchers/launcher_jax_gpu.sh "$MESH_SIZE"

# sbatch -n 2 launchers/launcher_standard.sh "$MESH_SIZE"
# sbatch -n 4 launchers/launcher_standard.sh "$MESH_SIZE"
# sbatch -n 8 launchers/launcher_standard.sh "$MESH_SIZE"
# sbatch -n 16 launchers/launcher_standard.sh "$MESH_SIZE"