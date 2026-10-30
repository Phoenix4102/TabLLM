## Milestone 3: Baseline Performance
Same as Milestone 2b.

### Copy files to the Tab S10+ or any attached mobile device:

Attach your device to the PC, and then verify if the device is attached by the following command:

```terminal
adb devices
```
If the device is attached, it will show up in terminal; then


Pushing the following files from PC onto device, if not already pushed:

```terminal
adb push smollm2-135m.tllm /data/local/tmp/tabllm
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

**Push libomp.so library manually to the mobile device (Tab S10+) if haven't already:**

In the Powershell terminal:

```powershell
$libomp = Get-ChildItem "$env:ANDROID_HOME\ndk" -Recurse -Filter "libomp.so" | Where-Object { $_.FullName -like "*aarch64*" } | Select-Object -First 1

adb push $libomp.FullName /data/local/tmp/tabllm/
```

**Pushing the new C++ binary and tokenizer:**

```terminal
adb push build-android/tabllm /data/local/tmp/tabllm
adb push run_3.sh /data/local/tmp/tabllm
adb shell "chmod +x /data/local/tmp/tabllm/tabllm"
```
Push `tokenizer.json` if not already pushed:

```terrminal
adb push "../Milestone 1/smollm2-135m.tllm.tokenizer/tokenizer.json" /data/local/tmp/tabllm/tokenizer.json
```
In the terminal, enter:

```terminal
adb shell
```

After the above command, you will drop into adb shell running inside device on your PC.

Enter the following commands inside adb shell:

```terminal
chmod +x /data/local/tmp/tabllm/run_3.sh
/data/local/tmp/tabllm/run_3.sh "What is the capital of France?" 32 0.7 40
```
Where `32` is the generated token length, `0.7` is the temperature, and `40` is the top-K value.
These are default values for now. Can pass values of your choosing.


If you are using a base model instead of a chat/instruct model:

```terminal
adb push run_base.sh /data/local/tmp/tabllm/
adb shell "chmod +x /data/local/tmp/tabllm/run_base.sh"
adb shell
```

Run inside adb shell:

```terminal
/data/local/tmp/tabllm/run_base.sh "The capital of France is"
```

*Should now get text output directly like:*

> The capital of France is Paris.