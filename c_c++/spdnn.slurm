#!/bin/bash
#SBATCH --job-name=spdnn
#SBATCH --output=spdnn.out
#SBATCH --error=spdnn.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --cluster=smp
#SBATCH --partition=high-mem


echo "SLURM_JOB_ID="$SLURM_JOB_ID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_CORES_NODES"=$SLURM_CPUS_PER_TASK
echo "SLURM_TASKS"=$SLURM_NTASKS
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
echo "************************************************"

module purge
module load gcc/5.4.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

DATA_PERFIX="/zfs1/cs3580_2017F/moh18/sdnn/data/"
NEURONS=("1024")
#NEURONS=("1024" "4096" "16384" "65536")
LAYERS=("120" "480" "1920")
#THREADS=("1" "8" "14" "28")

for N in "${NEURONS[@]}"
do
    for L in "${LAYERS[@]}"
    do
 #       for T in "${THREADS[@]}"
   #         do
#	    export OMP_NUM_THREADS=${T}
	    echo ${T}
            CMD="./baseline -n ${N} -l ${L} ${DATA_PERFIX}/MNIST/ ${DATA_PERFIX}/DNN/"
            echo "Command=${CMD}"
            ${CMD}
  #      done
    done
done
exit;


