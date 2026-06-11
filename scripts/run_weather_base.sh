#!/bin/bash
#SBATCH --job-name=weather_base
#SBATCH -p segal.q
#SBATCH --gres=gpu:L40S:1
#SBATCH -c 3
#SBATCH --mem=40G
#SBATCH --time=6:00:00
#SBATCH --output=/home/evandro/checkpoints_segmoe/logs/%x-%j.out
#SBATCH --error=/home/evandro/checkpoints_segmoe/logs/%x-%j.err

set -euo pipefail
# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
JAFAR_ROOT=/home/evandro
CHECKPOINT_ROOT="$JAFAR_ROOT/checkpoints_segmoe"
# Important: the SBATCH log directory must exist before sbatch is called.
mkdir -p "$CHECKPOINT_ROOT/logs"
# ---------------------------------------------------------------------
# Activate GPU environment
# ---------------------------------------------------------------------
source /home/evandro/anaconda3/etc/profile.d/conda.sh
conda activate torch_stable

cd "$JAFAR_ROOT/src/segmoe_forecast"
echo "===== JOB INFO ====="
echo "date:   $(date)"
echo "host:   $(hostname)"
echo "pwd:    $(pwd)"
echo "python: $(which python)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-N/A}"
#nvidia-smi

python - <<'PY'
import sys
import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("====================\n")
PY

python -u -m segmoe_forecast.utils.run_benchmarks --model-size base \
  --block-size 512 --patch-width 8 --width-factor 4 --channels 21 \
  --set exp_route_dropout=0. --set exp_route_temperature=1.0 \
  --exp-segment-size "[5,5,4,4,3,3]" \
  --epochs 25 --max-lr 3.2e-4 --min-lr 1.2e-5 \
  --weight-decay 1e-4 --warmup-portion 0.1 --setup-opt \
  --bf16 --moe-metrics --clip-grad None \
  --no-show-tqdm --save-plots --no-plot-cut-first \
  --dataset-name "Weather" --no-from-csv --batch-size 192 \
  --verbose --checkpoint-dir "$CHECKPOINT_ROOT" --seed 44