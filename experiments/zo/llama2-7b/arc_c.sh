#!/bin/sh

for SEED in 42 1337 2025
do
  echo "Running arc_c Config 1 (fp16, no PEFT, n=1, seed=$SEED)"
  python main.py \
    --compute_dtype "fp16" \
    --eps 0.001 \
    --learning_rate 0.0000005 \
    --logging_steps 500 \
    --lora_alpha 32 \
    --lora_rank 16 \
    --max_iterations 20000 \
    --max_length 512 \
    --mixed_precision false \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --n 1 \
    --num_train_epochs 1000 \
    --optimizer "zo" \
    --peft "no" \
    --per_device_eval_batch_size 16 \
    --per_device_train_batch_size 16 \
    --quantization "no" \
    --seed "$SEED" \
    --split false \
    --task_name "arc_c" \
    --zo_mode "single"

  echo "Running arc_c Config 2 (fp16, lora-fa, n=1, seed=$SEED)"
  python main.py \
    --compute_dtype "fp16" \
    --eps 0.01 \
    --learning_rate 0.00005 \
    --logging_steps 500 \
    --lora_alpha 32 \
    --lora_rank 16 \
    --max_iterations 20000 \
    --max_length 512 \
    --mixed_precision false \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --n 1 \
    --num_train_epochs 1000 \
    --optimizer "zo" \
    --peft "lora-fa" \
    --per_device_eval_batch_size 16 \
    --per_device_train_batch_size 16 \
    --quantization "no" \
    --seed "$SEED" \
    --split false \
    --task_name "arc_c" \
    --zo_mode "single"

  echo "Running arc_c Config 3 (fp16, lora-fa, n=4, seed=$SEED)"
  python main.py \
    --compute_dtype "fp16" \
    --eps 0.01 \
    --learning_rate 0.00006 \
    --logging_steps 500 \
    --lora_alpha 32 \
    --lora_rank 16 \
    --max_iterations 20000 \
    --max_length 512 \
    --mixed_precision false \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --n 4 \
    --num_train_epochs 1000 \
    --optimizer "zo" \
    --peft "lora-fa" \
    --per_device_eval_batch_size 16 \
    --per_device_train_batch_size 4 \
    --quantization "no" \
    --seed "$SEED" \
    --split false \
    --task_name "arc_c" \
    --zo_mode "single"

  echo "Running arc_c Config 4 (fp16, lora-fa, n=16, seed=$SEED)"
  python main.py \
    --compute_dtype "fp16" \
    --eps 0.01 \
    --learning_rate 0.00006 \
    --logging_steps 500 \
    --lora_alpha 32 \
    --lora_rank 16 \
    --max_iterations 20000 \
    --max_length 512 \
    --mixed_precision false \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --n 16 \
    --num_train_epochs 1000 \
    --optimizer "zo" \
    --peft "lora-fa" \
    --per_device_eval_batch_size 16 \
    --per_device_train_batch_size 1 \
    --quantization "no" \
    --seed "$SEED" \
    --split false \
    --task_name "arc_c" \
    --zo_mode "single"
done
