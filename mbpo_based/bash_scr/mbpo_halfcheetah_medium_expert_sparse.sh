#!/bin/bash
cd "$(dirname "$0")/.."  # run from mbpo_based/
# Multi-seed MBPO training for HalfCheetah-Medium-Expert-v2 (Sparse 72.5%)
# Usage: bash mbpo_based/bash_scr/mbpo_halfcheetah_medium_expert_sparse.sh

echo "============================================"
echo "Multi-Seed MBPO Training: HalfCheetah-Medium-Expert-Sparse3"
echo "============================================"
echo ""

# Array of random seeds to test
seeds=(42 123 456)

# Shared results file for all seeds
RESULTS_FILE="results/halfcheetah-medium-expert-v2_sparse_72.5/kde/mbpo_multiseed_results.csv"

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo ">>> Training MBPO with seed = $seed"
    echo "=========================================="

    python mopo.py \
        --config configs/kde/gormpo_halfcheetah_medium_expert_sparse.yaml \
        --algo-name mbpo \
        --reward-penalty-coef 0.0 \
        --seed $seed \
        --epoch 3000 \
        --devid 0 \
        --results_output $RESULTS_FILE \
         --dynamics-model-dir auto 
    echo "✓ MBPO training complete for seed $seed"
    echo ""
done

echo "============================================"
echo "All MBPO multi-seed experiments completed!"
echo "============================================"
echo ""

# Compute normalized scores across all seeds
echo "Computing normalized scores..."
python ../helpers/normalizer.py $RESULTS_FILE halfcheetah-medium-expert-v2
