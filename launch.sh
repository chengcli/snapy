#!/usr/bin/env bash
# launch_local.sh
set -euo pipefail

# How many processes (ranks)?
WORLD_SIZE=${1:-2}

# Backend: gloo (CPU) or nccl (GPU)
BACKEND=${BACKEND:-gloo}

# Rendezvous address/port
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

echo "Launching ${WORLD_SIZE} processes with BACKEND=${BACKEND}"
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

for (( RANK=0; RANK<${WORLD_SIZE}; RANK++ )); do
  echo "Starting rank ${RANK}"
  RANK=${RANK} \
  WORLD_SIZE=${WORLD_SIZE} \
  BACKEND=${BACKEND} \
  MASTER_ADDR=${MASTER_ADDR} \
  MASTER_PORT=${MASTER_PORT} \
  ./test_torchrun.release &
done

wait
echo "All ranks finished."
