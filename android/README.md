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