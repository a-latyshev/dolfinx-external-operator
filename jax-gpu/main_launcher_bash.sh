MESH_SIZE=200
# JAX_DEVICES=1
# mpirun -n 8 python tests/jax_multithreading.py --N $MESH_SIZE --d $JAX_DEVICES
# JAX_DEVICES=2
# mpirun -n 8 python tests/jax_multithreading.py --N $MESH_SIZE --d $JAX_DEVICES
# JAX_DEVICES=4
# mpirun -n 8 python tests/jax_multithreading.py --N $MESH_SIZE --d $JAX_DEVICES
# JAX_DEVICES=8
# mpirun -n 8 python tests/jax_multithreading.py --N $MESH_SIZE --d $JAX_DEVICES
# JAX_DEVICES=16
# mpirun -n 8 python tests/jax_multithreading.py --N $MESH_SIZE --d $JAX_DEVICES
# JAX_DEVICES=32
# mpirun -n 8 python tests/jax_multithreading.py --N $MESH_SIZE --d $JAX_DEVICES
# JAX_DEVICES=64
# mpirun -n 8 python tests/jax_multithreading.py --N $MESH_SIZE --d $JAX_DEVICES
# JAX_DEVICES=128
# mpirun -n 8 python tests/jax_multithreading.py --N $MESH_SIZE --d $JAX_DEVICES
# JAX_DEVICES=256
# mpirun -n 8 python tests/jax_multithreading.py --N $MESH_SIZE --d $JAX_DEVICES
# MESH_SIZE="200"
# sbatch -n 8 launchers/launcher_standard.sh "$MESH_SIZE"

MESH_SIZE=50
JAX_DEVICES=1
mpirun -n 1 python tests/jax_gpu.py --N $MESH_SIZE --d $JAX_DEVICES 
JAX_DEVICES=4
mpirun -n 4 python tests/jax_gpu.py --N $MESH_SIZE --d $JAX_DEVICES