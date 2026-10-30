# 1. Stop any lingering background Gradle processes holding file locks
# .\gradlew --stop

# 2. Clean the project build folders
Remove-Item -Recurse -Force .gradle, app\build, build -ErrorAction SilentlyContinue

# 3. Delete the failed NDK download from the Android SDK temp folder
Remove-Item -Recurse -Force "C:\Android\Sdk\temp\*" -ErrorAction SilentlyContinue

# 4. (Optional) Nuke Gradle's global dependency cache if you are still low on space
# This forces it to re-download dependencies next time, but reclaims a lot of GBs.
# Remove-Item -Recurse -Force "$env:USERPROFILE\.gradle\caches" -ErrorAction SilentlyContinue