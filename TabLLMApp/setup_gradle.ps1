$gradleUrl = "https://services.gradle.org/distributions/gradle-8.5-bin.zip"
$zipFile = "$env:TEMP\gradle-8.5-bin.zip"
$extractDir = "C:\Gradle"

Write-Host "Downloading Gradle 8.5..." -ForegroundColor Cyan
Invoke-WebRequest -Uri $gradleUrl -OutFile $zipFile

Write-Host "Extracting to C:\Gradle..." -ForegroundColor Cyan
if (-not (Test-Path $extractDir)) { New-Item -ItemType Directory -Path $extractDir -Force | Out-Null }
Expand-Archive -Path $zipFile -DestinationPath $extractDir -Force
Remove-Item $zipFile

$gradleBin = "$extractDir\gradle-8.5\bin"
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")

if ($userPath -notlike "*$gradleBin*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$gradleBin", "User")
    Write-Host "Added Gradle to User PATH." -ForegroundColor Green
}

# Update current session PATH
$env:Path = "$env:Path;$gradleBin"

Write-Host "Verifying installation..." -ForegroundColor Yellow
gradle --version