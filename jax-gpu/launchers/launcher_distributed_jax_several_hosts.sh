#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -p batch

#SBATCH --time=0-01:00:00
#SBATCH --job-name=jax-distributed-several-hosts
#SBATCH --chdir=slurm_output/

echo "== Starting run at $(date)"
echo "== Job name: ${SLURM_JOB_NAME}"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir: ${SLURM_SUBMIT_DIR}"
echo "== Number of tasks: ${SLURM_NTASKS}"

spack env activate fenicsx-v09
cd ~/dolfinx-external-operator/jax-gpu

srun -c 1 --exclusive -n $SLURM_NTASKS python tests/distributed_jax_several_hosts.py --N $1

echo ""
echo "== Finished at $(date)"