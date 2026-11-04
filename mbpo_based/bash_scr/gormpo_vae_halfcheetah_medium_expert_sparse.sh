#!/bin/bash
cd "$(dirname "$0")/.."  # run from mbpo_based/
# Multi-seed GORMPO-VAE training for HalfCheetah-Medium-Expert-v2 (Sparse 72.5%)
# Usage: bash mbpo_based/bash_scr/gormpo_vae_halfcheetah_medium_expert_sparse.sh

echo "============================================"
echo "Multi-Seed GORMPO-VAE Training: HalfCheetah-Medium-Expert-Sparse3"
echo "============================================"
echo ""

# Array of random seeds to test
seeds=(42 123 456)

# Shared results file for all seeds
RESULTS_FILE="results/halfcheetah-medium-expert-v2_sparse_72.5/vae/gormpo_vae_multiseed_results.csv"

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo ">>> Training with seed = $seed"
    echo "=========================================="

    # Step 1: Train VAE density estimator for this seed
    # echo "Step 1/2: Training VAE density estimator (seed $seed)..."
    # python ../density_models/vae_module/vae.py \
    #     --config ../density_models/configs/vae/halfcheetah_medium_expert_sparse.yaml \
    #     --seed $seed \
    #     --model_save_path checkpoints/halfcheetah_medium_expert_sparse/vae_$seed/vae_$seed \
    #     --device cuda:2 
    # echo "✓ VAE training complete for seed $seed"
    # echo ""

    # Step 2: Train GORMPO policy using the trained VAE model
    echo "Step 2/2: Training GORMPO-VAE policy (seed $seed)..."
    python mopo.py \
        --config configs/vae/gormpo_halfcheetah_medium_expert_sparse.yaml \
        --seed $seed \
        --classifier_model_name checkpoints/halfcheetah_medium_expert_sparse/vae_$seed/vae_$seed \
        --epoch 3000 \
        --devid 7\
        --results_output $RESULTS_FILE \
        --dynamics-model-dir auto 
    echo "✓ GORMPO-VAE training complete for seed $seed"
    echo ""
done

echo "============================================"
echo "All GORMPO-VAE multi-seed experiments completed!"
echo "============================================"
echo ""

# Compute normalized scores across all seeds
echo "Computing normalized scores..."
python ../helpers/normalizer.py $RESULTS_FILE halfcheetah-medium-expert-v2
