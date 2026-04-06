# ExecuTorch Training Baseline

This directory contains the ExecuTorch CPU training baseline used in this
project.

- `et_export.py`: host-side training export and optional host-only validation
- `et_run.py`: Android build, deploy, and run wrapper
- `training_runner/`: native Android runner sources

The export path follows the official ExecuTorch joint forward/backward pattern
used by `examples/llm_pte_finetuning`, but it targets the local GPT-2 model in
`android/models/gpt2.py`.

## What This Baseline Does

1. build a GPT-2 training module that returns loss first
2. export a joint forward/backward program with `_export_forward_backward`
3. save the result as a `.pte`, optionally with external constants in a `.ptd`
4. run the training program on Android with the native `gpt2_training_runner`

This is a CPU baseline. It does not use QNN or the NPU.

## Prerequisites

- ExecuTorch source tree available through `$EXECUTORCH_ROOT`
- Android ExecuTorch build available, for example
  `$EXECUTORCH_ROOT/build-android-qnn237`
- Android NDK configured in `$ANDROID_NDK_ROOT`
- host ExecuTorch pybindings available for export
- raw WikiText files available under
  `/home/jmha/MobiZO/data/wikitext2/wikitext-2-raw`
- optional local GPT-2 pretrained checkpoint directory for `gpt2_small`

## Build the Host Pybindings

The export script uses ExecuTorch host pybindings such as `_portable_lib` and
`_training_lib`.

```bash
cd /home/jmha/hetllm/third_party/executorch
source /home/jmha/hetllm/scripts/env.sh

cmake -S . -B cmake-out-training-host-lite \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_PYBIND=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TRAINING=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_BUILD_XNNPACK=OFF \
  -DEXECUTORCH_ENABLE_LOGGING=ON \
  -DPYTHON_EXECUTABLE=$(which python)

cmake --build cmake-out-training-host-lite -j8 --target portable_lib _training_lib
cmake --install cmake-out-training-host-lite --prefix cmake-out-training-host-lite/install
```

## Environment

```bash
cd /home/jmha/MobiZO
source /home/jmha/hetllm/scripts/env.sh
export PYTHONPATH=/home/jmha/hetllm/third_party/executorch:$PYTHONPATH
```

## Host Validation

This runs the tiny model on the host and checks that the training loss moves in
the expected direction.

```bash
python android/et/et_export.py --no_xnnpack
```

## Export Only

This generates the training `.pte` without running the host-side loop.

```bash
python android/et/et_export.py \
  --no_xnnpack \
  --export_only \
  --artifact /home/jmha/MobiZO/et_training_artifacts/gpt2_tiny_training_android.pte
```

## Run on Android

This command builds the native runner from `android/et/training_runner/`,
pushes the model and optional token files, and executes the CPU training
baseline on the phone.

```bash
python android/et/et_run.py \
  -s <adb-serial> \
  -b $EXECUTORCH_ROOT/build-android-qnn237 \
  --artifact_path /home/jmha/MobiZO/et_training_artifacts/gpt2_tiny_training_android.pte \
  --artifact_dir ./gpt2_training_phone_baseline \
  --steps 50 \
  --eval_steps 8
```

## Realistic GPT-2 Small Baseline

To use a more realistic CPU baseline with pretrained GPT-2 small weights and
WikiText tokens:

```bash
python android/et/et_export.py \
  --model_preset gpt2_small \
  --pretrained_dir /home/jmha/MobileFineTuner/gpt2_lora_finetune/pretrained/gpt2 \
  --prepare_raw_dir /home/jmha/MobiZO/data/wikitext2/wikitext-2-raw \
  --prepare_data_dir /home/jmha/MobiZO/et_training_artifacts/gpt2_small_tokens \
  --artifact /home/jmha/MobiZO/et_training_artifacts/gpt2_small_bs32_cpu_training.pte \
  --external_constants \
  --export_only \
  --no_xnnpack \
  --block_size 32 \
  --batch_size 1 \
  --steps 1 \
  --eval_steps 1 \
  --num_batches 1
```

```bash
python android/et/et_run.py \
  -s <adb-serial> \
  -b $EXECUTORCH_ROOT/build-android-qnn237 \
  --model_preset gpt2_small \
  --artifact_dir ./gpt2_small_phone_bs32_cpu \
  --artifact_path /home/jmha/MobiZO/et_training_artifacts/gpt2_small_bs32_cpu_training.pte \
  --ptd_path /home/jmha/MobiZO/et_training_artifacts/_default_external_constant.ptd \
  --train_tokens_path /home/jmha/MobiZO/et_training_artifacts/gpt2_small_tokens/wiki.train.gpt2.int32.bin \
  --eval_tokens_path /home/jmha/MobiZO/et_training_artifacts/gpt2_small_tokens/wiki.valid.gpt2.int32.bin \
  --skip_export \
  --steps 1 \
  --eval_steps 1 \
  --batch_size 1 \
  --block_size 32 \
  --dataset_stride 32
```

## Outputs

`et_run.py` saves device logs under the chosen artifact directory. The runner
prints `RESULT ...` lines for:

- pre-training eval loss
- post-training eval loss
- last training loss
- average training step time
- wall clock time
- RSS and HWM memory

If `--save_ptd_path` is provided, the updated parameters are also pulled back to
the host.
