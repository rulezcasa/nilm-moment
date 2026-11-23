#!/bin/bash
# Prediction-vs-actual spans: every model's best seed on one test span per appliance, into spans/.
# Submit from ~/nilm-moment on coe-hpc3:
#   mkdir -p slurm/logs
#   sbatch slurm/predict_span.sh
#   different span length: sbatch --export=ALL,HOURS=12 slurm/predict_span.sh
#SBATCH --job-name=moment-span
#SBATCH --partition=nsfq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -o pipefail
HOURS="${HOURS:-24}"
TARGETS=("Kettle" "Washing Machine" "Fridge-Freezer")

cd "$SLURM_SUBMIT_DIR"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nilm
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "hours=$HOURS node=$(hostname) start=$(date)"
status=0
for TARGET in "${TARGETS[@]}"; do
    python -u -m scripts.predict_span --target "$TARGET" --hours "$HOURS" || status=1
done
echo "end=$(date) status=$status"
exit $status
