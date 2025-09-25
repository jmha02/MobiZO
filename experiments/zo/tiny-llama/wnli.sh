#!/bin/sh

for SEED in 42 1337 2025
do
  python main.py \
    --compute_dtype "fp16" \
    --eps 0.001 \
    --learning_rate 0.000001 \
    --logging_steps 500 \
    --lora_alpha 32 \
    --lora_rank 16 \
    --max_iterations 20000 \
    --max_length 512 \
    --mixed_precision false \
    --model_name_or_path "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --n 1 \
    --num_train_epochs 1000 \
    --optimizer "zo" \
    --peft "no" \
    --per_device_eval_batch_size 16 \
    --per_device_train_batch_size 16 \
    --quantization "no" \
    --seed "$SEED" \
    --split false \
    --task_name "wnli" \
    --zo_mode "single"

  python main.py \
    --compute_dtype "fp16" \
    --eps 0.01 \
    --learning_rate 0.0001 \
    --logging_steps 500 \
    --lora_alpha 32 \
    --lora_rank 16 \
    --max_iterations 20000 \
    --max_length 512 \
    --mixed_precision false \
    --model_name_or_path "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --n 1 \
    --num_train_epochs 1000 \
    --optimizer "zo" \
    --peft "lora-fa" \
    --per_device_eval_batch_size 16 \
    --per_device_train_batch_size 16 \
    --quantization "no" \
    --seed "$SEED" \
    --split false \
    --task_name "wnli" \
    --torch_optimizer "adam" \
    --zo_mode "single"

  python main.py \
    --compute_dtype "fp16" \
    --eps 0.01 \
    --learning_rate 0.0001 \
    --logging_steps 500 \
    --lora_alpha 32 \
    --lora_rank 16 \
    --max_iterations 20000 \
    --max_length 512 \
    --mixed_precision false \
    --model_name_or_path "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --n 16 \
    --num_train_epochs 1000 \
    --optimizer "zo" \
    --peft "lora-fa" \
    --per_device_eval_batch_size 16 \
    --per_device_train_batch_size 1 \
    --quantization "no" \
    --seed "$SEED" \
    --split false \
    --task_name "wnli" \
    --torch_optimizer "adam" \
    --zo_mode "single"

  python main.py \
    --compute_dtype "fp16" \
    --eps 0.01 \
    --learning_rate 0.0001 \
    --logging_steps 500 \
    --lora_alpha 32 \
    --lora_rank 16 \
    --max_iterations 20000 \
    --max_length 512 \
    --mixed_precision false \
    --model_name_or_path "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --n 4 \
    --num_train_epochs 1000 \
    --optimizer "zo" \
    --peft "lora-fa" \
    --per_device_eval_batch_size 16 \
    --per_device_train_batch_size 4 \
    --quantization "no" \
    --seed "$SEED" \
    --split false \
    --task_name "wnli" \
    --torch_optimizer "adam" \
    --zo_mode "single"
done