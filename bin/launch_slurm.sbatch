#!/bin/bash
#SBATCH -J snap
#SBATCH -N 1                  # 1 node
#SBATCH -n 4                  # 4 tasks/ranks total
#SBATCH -t 00:10:00
#SBATCH -p debug

# Pick master addr/port
MASTER_ADDR=$(hostname)
MASTER_PORT=29500

export BACKEND=gloo

srun --mpi=none bash -c '
  export RANK=$SLURM_PROCID
  export WORLD_SIZE=$SLURM_NTASKS
  export LOCAL_RANK=$SLURM_LOCALID
  export NODE_RANK=$SLURM_NODEID
  export MASTER_ADDR='"$MASTER_ADDR"'
  export MASTER_PORT='"$MASTER_PORT"'

  echo "Launching rank $RANK/$WORLD_SIZE on node $(hostname), local_rank=$LOCAL_RANK"

  ./run.release
'
