#!/bin/bash -l
#SBATCH --nodes=2
#SBATCH -G 8
#SBATCH -c 28
#SBATCH --partition=gpu

##SBATCH --ntasks-per-node=28
##SBATCH --cpus-per-task=1
##SBATCH -p batch
##SBATCH --qos=normal

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

# Workaround for broken Python module find for adios2
export PYTHONPATH=$(find $SPACK_ENV/.spack-env -type d -name 'site-packages' | grep venv):$PYTHONPATH

python jax-gpu.py

echo ""
echo "== Finished at $(date)"