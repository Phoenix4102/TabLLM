### Set up Python on your PC:

```powershell
python -m venv .venv
```
In Windows Powershell:

```powershell
.venv\Scripts\Activate.ps1
```

In Linux type OS:
```bash
source .venv/bin/activate
```

### Install dependencies:
```terminal
pip install --upgrade pip
pip install torch transformers huggingface_hub safetensors numpy
```
Optional:
```terminal
pip install accelerate
```

**Inside Powershell:**

```powershell
# CMake (if not already installed)
winget install Kitware.CMake

# Ninja
winget install Ninja-build.Ninja
```

### For exporting the model to .tllm
```terminal
python export_hf_to_tllm.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --out smollm2-135m.tllm \
  --ctx 1024
```

This will download the real model and create:

`smollm2-135m.tllm`

`smollm2-135m.tllm.tokenizer/`

The `.tllm` file is the file the C++ engine loads.

### Encoding the prompt:

*1. For an instruction model, use --chat*

```terminal
python encode_prompt.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --prompt "What is the capital of France?" \
  --chat \
  --out prompt_tokens.bin \
  --eos-file eos.txt
```

This creates:

- prompt_tokens.bin
- eos.txt

*2. If using a base model instead of an instruct/chat model, use:*

```terminal
python encode_prompt.py \
  --model HuggingFaceTB/SmolLM2-135M \
  --prompt "The capital of France is" \
  --bos \
  --out prompt_tokens.bin \
  --eos-file eos.txt
```

### Copy files to the Tab S10+ or any attached mobile device:

Attach your device to the PC, and then verify if the device is attached by the following command:

```terminal
adb devices
```
If the device is attached, it will show up in terminal; then

From PC:

```terminal
adb shell "mkdir -p /data/local/tmp/tabllm"
```

Pushing files from PC onto device:

```terminal
adb push smollm2-135m.tllm /data/local/tmp/tabllm
adb push prompt_tokens.bin /data/local/tmp/tabllm
adb push eos.txt /data/local/tmp/tabllm
```

**Setting up and Building the C++ binary**

> In Powershell:

```powershell
$ndkRoot = Get-ChildItem "$env:LOCALAPPDATA\Android\Sdk\ndk" -Directory |
    Sort-Object Name -Descending |
    Select-Object -First 1

if (-not $ndkRoot) {
    throw "Android NDK not found"
}

Write-Host "NDK: $($ndkRoot.FullName)" -ForegroundColor Green

# Verify toolchain file exists
$toolchain = Join-Path $ndkRoot.FullName "build\cmake\android.toolchain.cmake"
if (-not (Test-Path $toolchain)) {
    throw "Toolchain file not found: $toolchain"
}

Write-Host "Toolchain: $toolchain" -ForegroundColor Green

$toolchain = ($ndkRoot.FullName + "\build\cmake\android.toolchain.cmake") -replace '\\', '/'

Remove-Item -Recurse -Force build-android -ErrorAction SilentlyContinue

cmake -B build-android -S . -G Ninja `
  -DCMAKE_TOOLCHAIN_FILE="$toolchain" `
  -DANDROID_ABI=arm64-v8a `
  -DANDROID_PLATFORM=android-26 `
  -DANDROID_STL=c++_static `
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-android -j
```

You should now have:

- build-android/tabllm

**Push libomp.so library manually to the mobile device (Tab S10+):**

In the Powershell terminal:

```powershell
$libomp = Get-ChildItem "$env:ANDROID_HOME\ndk" -Recurse -Filter "libomp.so" | Where-Object { $_.FullName -like "*aarch64*" } | Select-Object -First 1

adb push $libomp.FullName /data/local/tmp/tabllm/
```

**Pushing the C++ binary:**

```terminal
adb push build-android/tabllm /data/local/tmp/tabllm
adb push run.sh /data/local/tmp/tabllm
adb shell "chmod +x /data/local/tmp/tabllm/tabllm"
```
In terminal, enter:

```terminal
adb shell
```

After the above command, you will drop into adb shell running inside device on your PC.

Enter the following commands inside adb shell:

```terminal
chmod +x /data/local/tmp/tabllm/run.sh
/data/local/tmp/tabllm/run.sh 4 32 0.7 40
```

Where `4` is the number of `OMP_NUM_THREADS`, `32` is the generated token length, `0.7` is the temperature, and `40` is the top-K value.

These are default values for now. Can pass values of your choosing.

> The engine will print generated token IDs like:
> 
> 1234 5678 90 ...

### Decoding the generated tokens:

Copy the generated tokens back to shared storage:

On the PC:

```terminal
adb pull /data/local/tmp/tabllm/out_tokens.bin
```

Decode tokens with the following command:

```terminal
python decode_tokens.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --input out_tokens.bin
```

You should see text on the terminal.