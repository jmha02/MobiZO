#!/bin/sh

for SEED in 42 1337 2025
do
  echo "Running WNLI FO Config 1 (fp32, no PEFT, lr=1e-4, seed=$SEED)"
  python main.py \
    --compute_dtype "fp32" \
    --eps 0.01 \
    --learning_rate 0.001 \
    --logging_steps 100 \
    --lora_alpha 32 \
    --lora_rank 16 \
    --max_iterations 1000 \
    --max_length 512 \
    --mixed_precision "true" \
    --model_name_or_path "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --n 1 \
    --num_train_epochs 1000 \
    --optimizer "fo" \
    --peft "no" \
    --per_device_eval_batch_size 16 \
    --per_device_train_batch_size 16 \
    --quantization "no" \
    --seed "$SEED" \
    --split false \
    --task_name "wnli" \
    --torch_optimizer "sgd" \
    --zo_mode "single"

  echo "Running WNLI FO Config 2 (fp32, lora-fa, lr=1e-3, seed=$SEED)"
  python main.py \
    --compute_dtype "fp32" \
    --eps 0.01 \
    --learning_rate 0.005 \
    --logging_steps 100 \
    --lora_alpha 32 \
    --lora_rank 16 \
    --max_iterations 1000 \
    --max_length 512 \
    --mixed_precision "true" \
    --model_name_or_path "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --n 1 \
    --num_train_epochs 1000 \
    --optimizer "fo" \
    --peft "lora-fa" \
    --per_device_eval_batch_size 16 \
    --per_device_train_batch_size 16 \
    --quantization "no" \
    --seed "$SEED" \
    --split false \
    --task_name "wnli" \
    --torch_optimizer "sgd" \
    --zo_mode "single"
done
