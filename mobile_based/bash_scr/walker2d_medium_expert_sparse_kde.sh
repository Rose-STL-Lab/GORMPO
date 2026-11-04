#!/bin/bash
cd "$(dirname "$0")/.."  # run from mobile_based/

TASK="walker2d-medium-expert-v2"
DATASET_PATH="data/sparse_datasets/walker2d_medium_expert_sparse_73.pkl"
declare -A DEVICES
DEVICES[42]="cuda:4"
DEVICES[123]="cuda:5"
DEVICES[456]="cuda:6"
CLASSIFIER='kde'
DATASET_NAME=$(basename "$DATASET_PATH" .pkl)
mkdir -p results
RESULTS_FILE="results/results_${DATASET_NAME}_${CLASSIFIER}.csv"

# Create results file with header if it doesn't exist
if [ ! -f "$RESULTS_FILE" ]; then
    echo "task,dataset,seed,final_normalized_reward,final_reward_std" > "$RESULTS_FILE"
fi

find_log_dir () {
    local seed=$1
    find "log/$TASK/" -mindepth 2 -maxdepth 2 -type d -name "seed_${seed}&timestamp_*" -path "*classifier_type=${CLASSIFIER}*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-
}

# Train dynamics model only for seed 42 (no --dynamics_path, so train.py trains
# a fresh one) and run its policy training to completion first.
echo "Running MOBILE with seed 42 (training dynamics model from scratch)"
python train.py \
--dataset-path "$DATASET_PATH" \
--device ${DEVICES[42]} \
--config config_gormpo/walker2d_medium_expert_sparse.yaml \
--seed 42 \
--epoch 3000 \
--reward_penalty_coef 0.3 \
--task $TASK \
--classifier_model_name checkpoints/walker2d_medium_expert_sparse/kde_42/kde_42

LOG_DIR_42=$(find_log_dir 42)
if [ -z "$LOG_DIR_42" ]; then
    echo "Error: could not find log directory for seed 42 after training. Aborting."
    exit 1
fi
DYNAMICS_PATH="$LOG_DIR_42/model"
echo "Trained dynamics model for seed 42 at: $DYNAMICS_PATH"

# Launch seeds 123 and 456 in parallel, reusing the seed-42 dynamics model
declare -A PIDS
for seed in 123 456
do
    echo "Running MOBILE with seed $seed"
    echo "Reusing dynamics path from seed 42: $DYNAMICS_PATH"
    python train.py \
    --dataset-path "$DATASET_PATH" \
    --device ${DEVICES[$seed]} \
    --config config_gormpo/walker2d_medium_expert_sparse.yaml \
    --seed $seed \
    --epoch 3000 \
    --reward_penalty_coef 0.3 \
    --task $TASK \
    --classifier_model_name checkpoints/walker2d_medium_expert_sparse/kde_${seed}/kde_${seed} \
    --dynamics_path "$DYNAMICS_PATH" &
    PIDS[$seed]=$!
done

# Wait for seeds 123 and 456 to complete
for seed in 123 456
do
    wait ${PIDS[$seed]}
    echo "Seed $seed finished"
done

# Collect results for all seeds
for seed in 42 123 456
do
    LOG_DIR=$(find_log_dir $seed)

    if [ -n "$LOG_DIR" ]; then
        CSV_FILE="$LOG_DIR/record/policy_training_progress.csv"
        if [ -f "$CSV_FILE" ]; then
            FINAL_REWARD=$( (head -n 1 "$CSV_FILE"; tail -n 1 "$CSV_FILE") | python3 -c "
import sys
import csv
reader = csv.DictReader(sys.stdin)
for row in reader:
    reward = row.get('eval/normalized_episode_reward', 'N/A')
    reward_std = row.get('eval/normalized_episode_reward_std', 'N/A')
    print(f'{reward},{reward_std}')
" 2>/dev/null)

            if [ -n "$FINAL_REWARD" ]; then
                echo "$TASK,$DATASET_NAME,$seed,$FINAL_REWARD" >> "$RESULTS_FILE"
                echo "Results appended: $TASK, seed $seed -> $FINAL_REWARD"
            else
                echo "Warning: Could not extract results for seed $seed"
            fi
        else
            echo "Warning: CSV file not found at $CSV_FILE"
        fi
    else
        echo "Warning: Log directory not found for seed $seed"
    fi
done

echo ""
echo "All runs completed! Results saved to $RESULTS_FILE"
