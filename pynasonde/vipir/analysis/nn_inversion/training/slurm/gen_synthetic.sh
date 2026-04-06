#PBS -S /bin/bash

# ---------------------------------------------------------------------------
# Configuration — edit these before submitting
# ---------------------------------------------------------------------------
PYNASONDE_ROOT="${HOME}/Research/CodeBase/pynasonde"
OUT_DIR="/scratch/${USER}/nn_polan/"
LOG_DIR="${OUT_DIR}/logs"
DATA_DIR="${OUT_DIR}/synthetic"

mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${DATA_DIR}"

#PBS -N nn_polan_gen
#PBS -q shortq                        # VEGA queue — adjust as needed
#PBS -l nodes=1:ppn=4
#PBS -l mem=64gb
#PBS -l walltime=02:00:00
#PBS -t 0-269                        # 270 shards = one per (year, doy) pair
#PBS -o ${OUT_DIR}/gen_${PBS_JOBID}_${PBS_ARRAYID}.out
#PBS -e ${OUT_DIR}/gen_${PBS_JOBID}_${PBS_ARRAYID}.err
#PBS -m a
#PBS -M chakras4@erau.edu


N_SHARDS=270       # must match -t upper bound + 1


# ---------------------------------------------------------------------------
# Run shard  (PBS_ARRAY_INDEX replaces SLURM_ARRAY_TASK_ID)
# ---------------------------------------------------------------------------
echo "Starting shard ${PBS_ARRAYID}/${N_SHARDS} on $(hostname) at $(date)"
echo "  Output: ${OUT_DIR}/shard_$(printf '%05d' ${PBS_ARRAYID}).nc"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
module purge
module use /apps/spack/share/spack/modules/linux-rocky8-zen4
module load anaconda3 cmake

source activate pynasonde

export PYTHONPATH="${PYNASONDE_ROOT}:${PYTHONPATH}"
export OMNIDATA_PATH="${HOME}/OMNI"   # matches synthetic_data.py OMNI_DATA_PATH


python "${PYNASONDE_ROOT}/pynasonde/vipir/analysis/nn_inversion/training/synthetic_data.py" \
    --shard     "${PBS_ARRAYID}" \
    --n_shards  "${N_SHARDS}" \
    --out_dir   "${OUT_DIR}" \
    --verbose

echo "Finished shard ${PBS_ARRAYID} at $(date)"
