#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH -G 4
##SBATCH --mem-per-gpu=16G
#SBATCH --cpus-per-task=1
##SBATCH -c 1
#SBATCH --partition=gpu

#SBATCH --time=0-01:00:00
#SBATCH --job-name=jax-gpu
#SBATCH --chdir=slurm_output/

echo "== Starting run at $(date)"
echo "== Job name: ${SLURM_JOB_NAME}"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir: ${SLURM_SUBMIT_DIR}"
echo "== Number of tasks: ${SLURM_NTASKS}"

spack env activate fenicsx-v09
cd ~/dolfinx-external-operator/jax-gpu

srun -c 1 -n $SLURM_NTASKS python tests/jax_gpu.py --N $1 --d $2

echo ""
echo "== Finished at $(date)"