#!/bin/bash
cd "$(dirname "$0")/.."  # run from mbpo_based/
# Multi-seed GORMPO-RealNVP training for Walker2d-Medium-Expert-v2 (Sparse 72.5%)
# Usage: bash mbpo_based/bash_scr/gormpo_realnvp_walker2d_medium_expert_sparse.sh

echo "============================================"
echo "Multi-Seed GORMPO-RealNVP Training: Walker2d-Medium-Expert-Sparse3"
echo "============================================"
echo ""

# Array of random seeds to test
seeds=(42 123 456)

# Shared results file for all seeds
RESULTS_FILE="results/walker2d-medium-expert-v2_sparse_78/realnvp/gormpo_realnvp_multiseed_results.csv"

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo ">>> Training with seed = $seed"
    echo "=========================================="

    # Step 1: Train RealNVP density estimator for this seed
    # echo "Step 1/2: Training RealNVP density estimator (seed $seed)..."
    # python ../density_models/realnvp_module/realnvp.py \
    #     --config ../density_models/configs/realnvp/walker2d_medium_expert_sparse.yaml \
    #     --seed $seed \
    #     --model_save_path checkpoints/walker2d_medium_expert_sparse/realnvp_$seed/realnvp_$seed \
    #     --device cuda:5
    # echo "✓ RealNVP training complete for seed $seed"
    # echo ""

    # Step 2: Train GORMPO policy using the trained RealNVP model
    echo "Step 2/2: Training GORMPO-RealNVP policy (seed $seed)..."
    python mopo.py \
        --config configs/realnvp/gormpo_walker2d_medium_expert_sparse.yaml \
        --seed $seed \
        --rollout-length 5 \
        --classifier_model_name checkpoints/walker2d_medium_expert_sparse/realnvp_$seed/realnvp_$seed \
        --epoch 3000 \
        --devid 1\
        --results_output $RESULTS_FILE \
        --dynamics-model-dir auto \

    echo "✓ GORMPO-RealNVP training complete for seed $seed"
    echo ""
done

echo "============================================"
echo "All GORMPO-RealNVP multi-seed experiments completed!"
echo "============================================"
echo ""

# Compute normalized scores across all seeds
echo "Computing normalized scores..."
python ../helpers/normalizer.py $RESULTS_FILE walker2d-medium-expert-v2
