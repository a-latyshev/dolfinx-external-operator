#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=0-02:00:00
#SBATCH -p batch

#SBATCH --job-name=mohr_coulomb
#SBATCH --chdir=slurm_output/

echo "== Starting run at $(date)"
echo "== Job name: ${SLURM_JOB_NAME}"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir: ${SLURM_SUBMIT_DIR}"
echo "== Number of tasks: ${SLURM_NTASKS}"

spack env activate fenicsx-v09
cd ~/dolfinx-external-operator/doc/demo

srun -c 1 --exclusive -n $SLURM_NTASKS python demo_plasticity_mohr_coulomb_mpi.py --N 50

echo ""
echo "== Finished at $(date)"