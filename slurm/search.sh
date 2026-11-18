#!/bin/bash
# Grid search: MODELS for one appliance, packed on one GPU; grids come from configs/models/<model>.yaml.
# Submit from ~/nilm-moment on coe-hpc3:
#   mkdir -p slurm/logs
#   sbatch --array=0-2%2 --export=ALL,MODELS="moment_finetune moment_frozen moment_random" slurm/search.sh
#   resubmit one appliance (resumes, finished trials are skipped): same command with --array=<i>
# Task i: 0 Kettle, 1 Washing Machine, 2 Fridge-Freezer.
#SBATCH --job-name=moment-search
#SBATCH --partition=nsfq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out

set -o pipefail
: "${MODELS:?set MODELS, e.g. --export=ALL,MODELS=\"cnn lstm\"}"
MODELS=($MODELS)
TARGETS=("Kettle" "Washing Machine" "Fridge-Freezer")
TARGET="${TARGETS[$SLURM_ARRAY_TASK_ID]}"
SLUG=$(echo "$TARGET" | tr 'A-Z -' 'a-z__')
LOGS="slurm/logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p "$LOGS"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nilm
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "target=$TARGET models=${MODELS[*]} node=$(hostname) start=$(date)"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 60 > "$LOGS/gpu.csv" &
SMI=$!

pids=()
for MODEL in "${MODELS[@]}"; do
    python -u -m scripts.search --model "$MODEL" --target "$TARGET" > "$LOGS/$MODEL-$SLUG.log" 2>&1 &
    pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
kill "$SMI"
echo "end=$(date) status=$status"
exit $status
