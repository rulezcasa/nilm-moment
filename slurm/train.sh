#!/bin/bash
# MOMENT final training: finetune, frozen and random for one appliance and seed, packed on one GPU.
# Submit from ~/nilm-moment on coe-hpc3:
#   mkdir -p slurm/logs && sbatch --array=0-8%2 slurm/train.sh   # 3 appliances x 3 seeds, at most 2 GPUs
#   sbatch --array=4 slurm/train.sh                              # rerun one task (restarts from scratch)
# Task i: appliance = i / 3 (0 Kettle, 1 Washing Machine, 2 Fridge-Freezer), seed = 42 + i % 3.
#SBATCH --job-name=moment-train
#SBATCH --partition=nsfq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out

set -o pipefail
TARGETS=("Kettle" "Washing Machine" "Fridge-Freezer")
TARGET="${TARGETS[$((SLURM_ARRAY_TASK_ID / 3))]}"
SEED=$((42 + SLURM_ARRAY_TASK_ID % 3))
SLUG=$(echo "$TARGET" | tr 'A-Z -' 'a-z__')
LOGS="slurm/logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p "$LOGS"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nilm
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "target=$TARGET seed=$SEED node=$(hostname) start=$(date)"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 60 > "$LOGS/gpu.csv" &
SMI=$!

pids=()
for MODEL in moment_finetune moment_frozen moment_random; do
    python -u -m scripts.train --model "$MODEL" --target "$TARGET" --seed "$SEED" > "$LOGS/$MODEL-$SLUG-$SEED.log" 2>&1 &
    pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
kill "$SMI"
echo "end=$(date) status=$status"
exit $status
