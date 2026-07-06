#!/bin/bash
#SBATCH --job-name=ecl_base
#SBATCH -p segal.q
#SBATCH --gres=gpu:L40S:1
#SBATCH -c 3
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/home/evandro/checkpoints_segmoe/logs/%x-%j.out
#SBATCH --error=/home/evandro/checkpoints_segmoe/logs/%x-%j.err

set -euo pipefail
ulimit -v unlimited 2>/dev/null || true
echo "vmem: soft=$(ulimit -Sv) hard=$(ulimit -Hv)"
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
  --block-size 512 --patch-width 8 --width-factor 4 --channels 321 \
  --set exp_route_dropout=0.1 --set exp_route_temperature=1.0 \
  --exp-segment-size "[5,5,4,4,3,3]" \
  --epochs 15 --max-lr 2.6e-5 --min-lr 3.2e-6 \
  --weight-decay 1e-4 --warmup-portion 0.1 --setup-opt \
  --bf16 --moe-metrics --clip-grad None \
  --no-show-tqdm --save-plots --no-plot-cut-first \
  --dataset-name "ECL" --no-from-csv --batch-size 14 \
  --verbose --checkpoint-dir "$CHECKPOINT_ROOT" --seed 62