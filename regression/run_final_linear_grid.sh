#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

DATA_SEEDS=(1 2 3)
RANKS=(16 32)
LEARNING_RATES=("1e-4" "5e-5")
MODEL_SEED=74
TARGET_SCALE="linear"
EXPERIMENT_DATE="${EXPERIMENT_DATE:-$(date +"%y%m%d")}"

total=$(( ${#DATA_SEEDS[@]} * ${#RANKS[@]} * ${#LEARNING_RATES[@]} ))
count=0

mkdir -p logs

for data_seed in "${DATA_SEEDS[@]}"; do
  for rank in "${RANKS[@]}"; do
    for learning_rate in "${LEARNING_RATES[@]}"; do
      count=$((count + 1))
      alpha=$((rank * 2))

      if [[ "$rank" -eq 32 ]]; then
        dropout="0.1"
      else
        dropout="0.05"
      fi

      log_file="logs/${EXPERIMENT_DATE}_linear_s${data_seed}_r${rank}_lr${learning_rate}.out"

      echo "[$count/$total] data_seed=$data_seed rank=$rank alpha=$alpha lr=$learning_rate"

      python main_logit_cp_share.py \
        --model_name "meta-llama/Llama-3.1-8B" \
        --data_seed "$data_seed" \
        --seed "$MODEL_SEED" \
        --load_in_4bit 1 \
        --bnb_4bit_use_double_quant 1 \
        --bnb_4bit_compute_dtype "float16" \
        --bnb_4bit_quant_type "nf4" \
        --r "$rank" \
        --lora_alpha "$alpha" \
        --lora_dropout "$dropout" \
        --epochs 15 \
        --batch_size 4 \
        --lr "$learning_rate" \
        --weight_decay 0.01 \
        --warmup_steps 20 \
        --saving_checkpoint 1 \
        --interp_method "linear" \
        --min_g 1e-10 \
        --target_scale "$TARGET_SCALE" \
        --experiment_date "$EXPERIMENT_DATE" \
        2>&1 | tee "$log_file"
    done
  done
done

python summarize_grid_results.py \
  --collect \
  --results-root results \
  --experiment "$EXPERIMENT_DATE"

python summarize_grid_results.py \
  --csv "results/${EXPERIMENT_DATE}/final_summary_results.csv"
