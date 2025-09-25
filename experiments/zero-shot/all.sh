#!/bin/sh

for TASK in sst2 rte mrpc qqp qnli wnli
do
  for SEED in 42 1337 2025
  do
    echo "Running ZO Config on task=$TASK with seed=$SEED"
    python main.py \
      --compute_dtype "fp16" \
      --max_length 512 \
      --mixed_precision false \
      --model_name_or_path "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
      --peft "no" \
      --per_device_eval_batch_size 16 \
      --quantization "no" \
      --seed "$SEED" \
      --split false \
      --task_name "$TASK" \
      --zero_shot true
  done
done

for TASK in sst2 rte boolq wsc wic multirc copa winogrande arc_e arc_c
do
  for SEED in 42 1337 2025
  do
    echo "Running ZO Config on task=$TASK with seed=$SEED"
    python main.py \
      --compute_dtype "fp16" \
      --max_length 512 \
      --mixed_precision false \
      --model_name_or_path "meta-llama/Llama-2-7b-hf" \
      --peft "no" \
      --per_device_eval_batch_size 2 \
      --quantization "no" \
      --seed "$SEED" \
      --split false \
      --task_name "$TASK" \
      --zero_shot true
  done
done
