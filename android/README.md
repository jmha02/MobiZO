# Android Environment Setup with ExecuTorch

### Step 1: Create Conda Environment

```bash
conda create -n android python=3.11
conda activate android
```

### Step 2: Download ExecuTorch (v0.6)
```bash
git clone -b release/0.6 https://github.com/pytorch/executorch.git
```

### Step 2: Install Qualcomm AI Engine Direct SDK and Android NDK
Download Qualcomm AI Engine Direct SDK [QNN 2.28.0.241029](https://softwarecenter.qualcomm.com/api/download/software/qualcomm_neural_processing_sdk/v2.28.0.241029.zip) and Android NDK [r26d](https://github.com/android/ndk/releases).

### Step 3: Set Environment Variables
Change the `QNN_SDK_ROOT` and `ANDROID_NDK_ROOT` path accordingly. 
```bash
export QNN_SDK_ROOT=/absolute_path_to/qairt/2.28.0.241029
export ANDROID_NDK_ROOT=/absolute_path_to/android-ndk-r26d

export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang/:$LD_LIBRARY_PATH
export EXECUTORCH_ROOT=/absolute_path_to/executorch
export PYTHONPATH=$EXECUTORCH_ROOT/..
```

### Step 4: Build ExecuTorch

```bash
sh build.sh
```

### Step 5: Connect Android Device

```bash
adb get-serialno
```

### Step 6: Run Compile & Offload Script
```bash
python qnn.py -s 7500709c -m SM8650 -b $EXECUTORCH_ROOT/build-android
```

* -s: Device serial number
* -m: SoC identifier
* -b: Path to the Android build directory for ExecuTorch

### Check detailed tutorial from [ExecuTorch](https://pytorch.org/executorch/stable/build-run-qualcomm-ai-engine-direct-backend.html)

## CPU Training Baseline

For the official ExecuTorch-style CPU training baseline, use the export/run
helpers under `android/et/` together with the native Android runner sources in
`android/et/training_runner/`.

Export the training artifact:

```bash
cd /home/jmha/MobiZO
source /home/jmha/hetllm/scripts/env.sh
export PYTHONPATH=$EXECUTORCH_ROOT:$PYTHONPATH

python android/et/et_export.py \
  --no_xnnpack \
  --export_only \
  --artifact /home/jmha/MobiZO/et_training_artifacts/gpt2_tiny_training_android.pte
```

Run it on device:

```bash
python android/et/et_run.py \
  -s <adb-serial> \
  -b $EXECUTORCH_ROOT/build-android-qnn237 \
  --artifact_path /home/jmha/MobiZO/et_training_artifacts/gpt2_tiny_training_android.pte \
  --artifact_dir ./gpt2_training_phone_baseline \
  --steps 50 \
  --eval_steps 8
```

The runner prints `RESULT ...` lines for:
- pre/post eval loss
- average step time
- RSS/HWM memory
- optional trained weight save via `--save_ptd_path`

On the current setup (`R3CR80BDZAY`, tiny GPT2, 50 steps), the baseline ran on
device with about `9.72 ms/step`, `~6.4 MB` RSS, and post-eval loss dropping
from `3.489` to `1.876`.
