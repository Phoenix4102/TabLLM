## Milestone 3: Optimization 3 4 Bit Block Quantization

### Export FP16 model
Use the updated `export_hf_to_tllm_q4_quantization.py` file to export model to fp16 weights

```terminal
python export_hf_to_tllm_q4_quantization.py `
  --model HuggingFaceTB/SmolLM2-135M-Instruct `
  --out smollm2-135m-q4.tllm `
  --ctx 1024 `
  --dtype q4
```

### Copy files to the Tab S10+ or any attached mobile device:

Attach your device to the PC, and then verify if the device is attached by the following command:

```terminal
adb devices
```
If the device is attached, it will show up in terminal; then


Pushing the following files from PC onto device:

```terminal
adb push smollm2-135m-q4.tllm /data/local/tmp/tabllm
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

# Verify toolchain file exists
$toolchain = Join-Path $ndkRoot.FullName "build\cmake\android.toolchain.cmake"
if (-not (Test-Path $toolchain)) {
    throw "Toolchain file not found: $toolchain"
}

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


**Pushing the new C++ binary and tokenizer:**

```terminal
adb push build-android/tabllm /data/local/tmp/tabllm
adb push run_text_q4.sh /data/local/tmp/tabllm
adb shell "chmod +x /data/local/tmp/tabllm/tabllm"
```
Push `tokenizer.json` if not already pushed:

```terrminal
adb push smollm2-135m-q4.tllm.tokenizer/tokenizer.json /data/local/tmp/tabllm/tokenizer.json
```
In the terminal, enter:

```terminal
adb shell
```

After the above command, you will drop into adb shell running inside device on your PC.

Enter the following commands inside adb shell:

```terminal
chmod +x /data/local/tmp/tabllm/run_text_q4.sh
/data/local/tmp/tabllm/run_text_q4.sh "What is the capital of France?" 64 0.7 40
```
Where `64` is the generated token length, `0.7` is the temperature, and `40` is the top-K value. Can pass values of your choosing.