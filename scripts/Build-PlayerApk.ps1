<#
.SYNOPSIS
    One command from the packaged Player to an installable Android APK.

.DESCRIPTION
    Stages scenes + shaders into android/app/src/main/assets, runs the Gradle Release (default)
    or Debug build for
    the chosen ABI(s), copies each APK to an output folder under a dated name and optionally
    installs it with adb. Every step stops with a clear message naming the step that failed.

    The scene set comes from dist/RayTrophiPlayer (scripts/package_player.ps1); shader contents
    come from x64/Release/shaders (see stage_player_android_assets.py).

.EXAMPLE
    .\scripts\Build-PlayerApk.ps1                      # arm64 phone/tablet APK, all scenes
.EXAMPLE
    .\scripts\Build-PlayerApk.ps1 -Abi x86_64 -Install # emulator round, installs on the emulator
.EXAMPLE
    .\scripts\Build-PlayerApk.ps1 -Abi both -Install -Device HVA751MH
#>
[CmdletBinding()]
param(
    # arm64 = real phones and tablets, x86_64 = emulator, both = two separate APKs.
    [ValidateSet('arm64', 'x86_64', 'both')]
    [string]$Abi = 'arm64',

    # Comma-separated scene folder names; empty = every packaged scene (library shows a card each).
    [string]$Scenes = '',

    # Maximum embedded texture edge in pixels (Android default 512).
    [int]$MaxTexture = 512,

    # Install with `adb install -r` after the build.
    [switch]$Install,

    # adb serial when more than one device is attached (see `adb devices`).
    [string]$Device = '',

    # Skip asset staging (reuse what is already in android/app/src/main/assets).
    [switch]$SkipStage,

    # Release is the default: on ARM devices the optimized native build is markedly faster
    # (Lenovo first frame: Debug 163.6 s -> Release 63.8 s). The release APK is signed with
    # the local debug keystore (~/.android/debug.keystore), i.e. installable for testing,
    # not a store upload. Debug keeps jniDebuggable for native debugging.
    [ValidateSet('Release', 'Debug')]
    [string]$Configuration = 'Release',

    [string]$OutputDir = 'E:/rt_android_build',
    [string]$JavaHome = 'C:\Program Files\Java\jdk-17',
    [string]$AndroidSdk = (Join-Path $env:LOCALAPPDATA 'Android\Sdk'),
    [string]$Sdl2Dir = 'E:/SDL2-2.30.4/SDL2-2.30.11',
    [string]$Sdl2ImageDir = 'E:/SDL2-2.30.4/SDL2_image-2.8.2_src'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$started = Get-Date

function Write-Step([string]$text) {
    Write-Host ''
    Write-Host "==> $text" -ForegroundColor Cyan
}

function Stop-Build([string]$step, [string]$why) {
    Write-Host ''
    Write-Host "FAILED at step: $step" -ForegroundColor Red
    Write-Host "  $why" -ForegroundColor Red
    exit 1
}

# --- 1. Tools --------------------------------------------------------------------------------
Write-Step '1/5 Checking tools'
if (-not (Test-Path (Join-Path $JavaHome 'bin\java.exe'))) {
    Stop-Build 'tools' "JDK 17 not found at '$JavaHome'. Pass -JavaHome <path>."
}
if (-not (Test-Path $AndroidSdk)) {
    Stop-Build 'tools' "Android SDK not found at '$AndroidSdk'. Pass -AndroidSdk <path>."
}
foreach ($dir in @($Sdl2Dir, $Sdl2ImageDir)) {
    if (-not (Test-Path $dir)) { Stop-Build 'tools' "SDL source tree not found: '$dir'." }
}
$adb = Join-Path $AndroidSdk 'platform-tools\adb.exe'
if ($Install -and -not (Test-Path $adb)) {
    Stop-Build 'tools' "adb not found at '$adb' (install SDK platform-tools)."
}
$python = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $SkipStage -and -not $python) { Stop-Build 'tools' 'python is not on PATH.' }

$env:JAVA_HOME = $JavaHome
$env:ANDROID_HOME = $AndroidSdk
$env:ANDROID_SDK_ROOT = $AndroidSdk
Write-Host "  JDK      $JavaHome"
Write-Host "  SDK      $AndroidSdk"

# --- 2. Assets -------------------------------------------------------------------------------
if ($SkipStage) {
    Write-Step '2/5 Asset staging skipped (-SkipStage)'
} else {
    Write-Step "2/5 Staging scenes and shaders (max texture $MaxTexture px)"
    # Android ships the Vulkan 1.1 (SPIR-V 1.3) build of the shader set on every device.
    & python (Join-Path $root 'scripts\compile_player_android_vk11.py')
    if ($LASTEXITCODE) {
        Stop-Build 'staging' ('compile_player_android_vk11.py failed: a raster/compute shader ' +
            'does not compile for Vulkan 1.1 (see ERROR lines above).')
    }
    $stageArgs = @((Join-Path $root 'scripts\stage_player_android_assets.py'),
                   '--max-texture', "$MaxTexture")
    if ($Scenes) { $stageArgs += @('--scenes', $Scenes) } else { $stageArgs += '--all' }
    & python @stageArgs
    if ($LASTEXITCODE) {
        Stop-Build 'staging' ('stage_player_android_assets.py failed. If it says dist/RayTrophiPlayer ' +
            'is missing, run scripts/package_player.ps1 first.')
    }
}

# --- 3. Build + 4. Copy ----------------------------------------------------------------------
$abis = switch ($Abi) {
    'arm64'  { @('arm64-v8a') }
    'x86_64' { @('x86_64') }
    'both'   { @('arm64-v8a', 'x86_64') }
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$gradlew = Join-Path $root 'android\gradlew.bat'
$gradleTask = if ($Configuration -eq 'Release') { 'assembleRelease' } else { 'assembleDebug' }
$apkSource = if ($Configuration -eq 'Release') {
    Join-Path $root 'android\app\build\outputs\apk\release\app-release-unsigned.apk'
} else {
    Join-Path $root 'android\app\build\outputs\apk\debug\app-debug.apk'
}
$apksigner = $null
$keystore = Join-Path $env:USERPROFILE '.android\debug.keystore'
if ($Configuration -eq 'Release') {
    $apksigner = Get-ChildItem (Join-Path $AndroidSdk 'build-tools') -Directory -ErrorAction SilentlyContinue |
        Sort-Object { [version]($_.Name -replace '[^0-9.].*$', '') } -Descending |
        ForEach-Object { Join-Path $_.FullName 'apksigner.bat' } |
        Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $apksigner) { Stop-Build 'tools' 'apksigner.bat not found under Android SDK build-tools.' }
    if (-not (Test-Path $keystore)) {
        Stop-Build 'tools' ('Debug keystore missing: ' + $keystore + '. Build one Debug APK first ' +
            '(Gradle creates it) or pass -Configuration Debug.')
    }
}
$stamp = Get-Date -Format 'yyyy-MM-dd_HHmm'
$built = @()

foreach ($abiName in $abis) {
    Write-Step "3/5 Gradle build for $abiName (first build compiles all C++ with the NDK: slow)"
    if (Test-Path $apkSource) { Remove-Item $apkSource -Force }
    & $gradlew -p (Join-Path $root 'android') $gradleTask --no-daemon `
        "-PrtSdl2Dir=$Sdl2Dir" "-PrtSdl2ImageDir=$Sdl2ImageDir" "-PrtAbis=$abiName"
    if ($LASTEXITCODE -or -not (Test-Path $apkSource)) {
        Stop-Build "gradle ($abiName)" 'Build failed; the first compiler/linker error above is the cause.'
    }

    Write-Step "4/5 Copying APK ($abiName)"
    $short = if ($abiName -eq 'arm64-v8a') { 'arm64' } else { $abiName }
    $suffix = if ($Configuration -eq 'Release') { 'release' } else { 'debug' }
    $target = Join-Path $OutputDir "RayTrophiPlayer-$short-$suffix-$stamp.apk"
    if ($Configuration -eq 'Release') {
        & $apksigner sign --ks $keystore --ks-pass pass:android --key-pass pass:android `
            --ks-key-alias androiddebugkey --out $target $apkSource
        if ($LASTEXITCODE -or -not (Test-Path $target)) {
            Stop-Build "sign ($abiName)" 'apksigner failed; see the message above.'
        }
    } else {
        Copy-Item $apkSource $target -Force
    }
    $size = [math]::Round((Get-Item $target).Length / 1MB, 1)
    Write-Host "  $target ($size MB)"
    $built += [pscustomobject]@{ Abi = $abiName; Path = $target }
}

# --- 5. Install ------------------------------------------------------------------------------
if (-not $Install) {
    Write-Step '5/5 Install skipped (add -Install to install with adb)'
} else {
    Write-Step '5/5 Installing'
    $devices = & $adb devices | Select-Object -Skip 1 | Where-Object { $_ -match '\tdevice$' } |
        ForEach-Object { ($_ -split "`t")[0] }
    if (-not $devices) {
        Stop-Build 'install' ('No device in `adb devices`. Phone: enable USB debugging and accept ' +
            'the prompt. Emulator: start it first.')
    }
    $serial = $Device
    if (-not $serial) {
        if (@($devices).Count -gt 1) {
            Stop-Build 'install' ("More than one device attached ($($devices -join ', ')). " +
                'Pass -Device <serial>.')
        }
        $serial = @($devices)[0]
    }
    # An emulator reports x86_64, a phone arm64-v8a: install the matching APK.
    $deviceAbi = (& $adb -s $serial shell getprop ro.product.cpu.abi).Trim()
    $apk = $built | Where-Object { $_.Abi -eq $deviceAbi } | Select-Object -First 1
    if (-not $apk) {
        Stop-Build 'install' ("Device $serial is $deviceAbi but this run built " +
            "$($built.Abi -join ', '). Rerun with the matching -Abi.")
    }
    & $adb -s $serial install -r $apk.Path
    if ($LASTEXITCODE) { Stop-Build 'install' "adb install failed on $serial." }
    Write-Host "  Installed on $serial ($deviceAbi)."
}

$minutes = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)
Write-Host ''
Write-Host "DONE in $minutes min." -ForegroundColor Green
$built | ForEach-Object { Write-Host "  $($_.Path)" }
