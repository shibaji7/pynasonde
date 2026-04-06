#!/bin/bash
#SBATCH --job-name=nn_polan_s1
#SBATCH --partition=gpu              # VEGA GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/stage1_%j.out
#SBATCH --error=logs/stage1_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chakras4@erau.edu

# ---------------------------------------------------------------------------
# Configuration — edit before submitting
# ---------------------------------------------------------------------------
PYNASONDE_ROOT="${HOME}/Research/CodeBase/pynasonde"
DATA_DIR="/scratch/${USER}/nn_polan/synthetic"
OUT_DIR="/scratch/${USER}/nn_polan/checkpoints/stage1"

EPOCHS=50
BATCH=256
LR=3e-4
WORKERS=6
LATENT=256
FEAT=128
LAMBDA_PHY=1.0
LAMBDA_MONO=0.01
LAMBDA_BG=0.1

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
module load python/3.11
module load cuda/12.1
module load gcc/12

source "${HOME}/.venvs/pynasonde_train/bin/activate"
export PYTHONPATH="${PYNASONDE_ROOT}:${PYTHONPATH}"

mkdir -p "${OUT_DIR}" logs

echo "Stage 1 training on $(hostname) at $(date)"
echo "  DATA_DIR: ${DATA_DIR}"
echo "  OUT_DIR:  ${OUT_DIR}"
echo "  GPUs:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
python "${PYNASONDE_ROOT}/pynasonde/vipir/analysis/nn_inversion/training/trainer_stage1.py" \
    --data_dir   "${DATA_DIR}" \
    --out_dir    "${OUT_DIR}" \
    --epochs     "${EPOCHS}" \
    --batch      "${BATCH}" \
    --lr         "${LR}" \
    --workers    "${WORKERS}" \
    --latent_dim "${LATENT}" \
    --feat_dim   "${FEAT}" \
    --lambda_phy "${LAMBDA_PHY}" \
    --lambda_mono "${LAMBDA_MONO}" \
    --lambda_bg  "${LAMBDA_BG}" \
    --device     cuda \
    --verbose

echo "Done at $(date)"
