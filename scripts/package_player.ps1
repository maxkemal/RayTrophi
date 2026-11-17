<#
.SYNOPSIS
  Builds a self-contained RayTrophi Player folder (and optional zip) from the
  current Release output and a list of .rtp scenes.

.DESCRIPTION
  Layout of the package:
    RayTrophiPlayer.exe + the DLLs it actually loads (resolved with dumpbin)
    VC++ runtime DLLs (app-local, from the Visual Studio redist folder)
    shaders\                 every .spv the Player can load
    scenes\<folder>\         .rtp + .rtp.bin + .rtp.shared + sidecars + <stem>_audio\
    player_scenes.json       scene list; the FIRST entry opens at launch, F6 switches
    licenses\                third-party notices
    redist\vc_redist.x64.exe fallback installer if app-local runtime is refused

  Not packaged on purpose:
    vulkan-1.dll             comes with the GPU driver
    Vulkan pipeline cache    keyed to GPU + driver version (VkPipelineCache
                             header); the target machine builds its own on the
                             first run. The Player's raster path does not use
                             the RT pipeline cache at all.
    CUDA/OptiX/OIDN/Embree/Python   Studio only; the Player does not link them.

.EXAMPLE
  .\scripts\package_player.ps1
  .\scripts\package_player.ps1 -Zip
  .\scripts\package_player.ps1 -Scenes 'RayTrophiStudio\assets\scenes\a\a.rtp' -OutDir dist\Test
#>
[CmdletBinding()]
param(
    # Scene order matters: the first one opens at launch.
    [string[]]$Scenes = @(
        'RayTrophiStudio\assets\scenes\pastel-paint-workshop\workshop.rtp',
        'RayTrophiStudio\assets\scenes\pastel-play-room\pastel-play-room-simple.rtp'
    ),
    # Display names for the F6 list, same order as -Scenes. Empty = file name.
    [string[]]$SceneNames = @('Boyama Atolyesi', 'Oyun Odasi'),
    [string]$OutDir = 'dist\RayTrophiPlayer',
    [switch]$Zip,
    # Leave the scenes' embedded audio out (third-party clips without a
    # confirmed licence live there; see commit d1c1709a).
    [switch]$NoSceneAudio
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$release = Join-Path $repo 'x64\Release'
$exe = Join-Path $release 'RayTrophiPlayer.exe'
$out = if ([System.IO.Path]::IsPathRooted($OutDir)) { $OutDir } else { Join-Path $repo $OutDir }
$marker = '.raytrophi_player_package'
$warnings = New-Object System.Collections.Generic.List[string]
function Warn([string]$m) { $warnings.Add($m); Write-Warning $m }

# --- 0. Sanity ---------------------------------------------------------------
if (-not (Test-Path $exe)) { throw "Player exe not found: $exe (build RayTrophiPlayer first)" }
$exeTime = (Get-Item $exe).LastWriteTime
$playerMain = Join-Path $repo 'RayTrophiStudio\source\src\Core\PlayerMain.cpp'
if ((Test-Path $playerMain) -and (Get-Item $playerMain).LastWriteTime -gt $exeTime) {
    Warn "PlayerMain.cpp is newer than RayTrophiPlayer.exe ($exeTime): the exe may be stale (no scene list / F6)."
}
if ($Scenes.Count -lt 1) { throw 'At least one scene is required.' }

# Fresh output folder. Only a folder this script made earlier is removed.
if (Test-Path $out) {
    if (-not (Test-Path (Join-Path $out $marker))) {
        throw "Output folder exists and is not a Player package: $out"
    }
    Remove-Item -Recurse -Force $out
}
New-Item -ItemType Directory -Force $out | Out-Null
Set-Content -Path (Join-Path $out $marker) -Value "made by scripts/package_player.ps1 $(Get-Date -Format s)" -Encoding ascii

# --- 1. Exe + DLLs it loads (transitive, from x64\Release) ---------------------
$dumpbin = Get-ChildItem 'C:\Program Files\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe' -ErrorAction SilentlyContinue |
    Sort-Object FullName -Descending | Select-Object -First 1
$fallbackDlls = @('SDL2.dll', 'SDL2_image.dll', 'openvdb.dll', 'Imath-3_1.dll', 'tbb12.dll', 'blosc.dll',
                  'zlib1.dll', 'lz4.dll', 'snappy.dll', 'zstd.dll', 'libssl-3-x64.dll', 'libcrypto-3-x64.dll')
$localDlls = @{}
if ($dumpbin) {
    $queue = New-Object System.Collections.Generic.Queue[string]
    $queue.Enqueue($exe)
    $seen = @{}
    while ($queue.Count -gt 0) {
        $bin = $queue.Dequeue()
        $deps = & $dumpbin.FullName /nologo /dependents $bin | ForEach-Object { $_.Trim() } | Where-Object { $_ -match '\.dll$' }
        foreach ($d in $deps) {
            $key = $d.ToLowerInvariant()
            if ($seen.ContainsKey($key)) { continue }
            $seen[$key] = $true
            $candidate = Join-Path $release $d
            if (Test-Path $candidate) {
                $localDlls[$key] = $candidate
                $queue.Enqueue($candidate)
            }
        }
    }
} else {
    Warn 'dumpbin not found; using the fixed DLL list (may be out of date).'
    foreach ($d in $fallbackDlls) {
        $p = Join-Path $release $d
        if (Test-Path $p) { $localDlls[$d.ToLowerInvariant()] = $p } else { Warn "missing DLL: $d" }
    }
}
Copy-Item $exe $out
foreach ($p in $localDlls.Values) { Copy-Item $p $out }
foreach ($bad in @('cudart64_12.dll', 'embree4.dll', 'OpenImageDenoise.dll', 'python311.dll')) {
    if ($localDlls.ContainsKey($bad.ToLowerInvariant())) { Warn "Player now depends on Studio-only $bad" }
}

# --- 2. VC++ runtime, app-local ---------------------------------------------------
$redistRoot = Get-ChildItem 'C:\Program Files\Microsoft Visual Studio\*\*\VC\Redist\MSVC\*' -Directory -ErrorAction SilentlyContinue |
    Where-Object { Test-Path (Join-Path $_.FullName 'x64\Microsoft.VC143.CRT') } |
    Sort-Object Name -Descending | Select-Object -First 1
if ($redistRoot) {
    Copy-Item (Join-Path $redistRoot.FullName 'x64\Microsoft.VC143.CRT\*.dll') $out
    $omp = Join-Path $redistRoot.FullName 'x64\Microsoft.VC143.OpenMP\vcomp140.dll'
    if (Test-Path $omp) { Copy-Item $omp $out } else { Warn 'vcomp140.dll (OpenMP) not found in redist' }
    $installer = Join-Path $redistRoot.FullName 'vc_redist.x64.exe'
    if (Test-Path $installer) {
        New-Item -ItemType Directory -Force (Join-Path $out 'redist') | Out-Null
        Copy-Item $installer (Join-Path $out 'redist')
    }
} else {
    Warn 'VC++ redist folder not found; target needs vc_redist.x64 installed.'
}

# --- 3. Shaders ------------------------------------------------------------------
Copy-Item (Join-Path $release 'shaders') (Join-Path $out 'shaders') -Recurse

# --- 4. Scenes -------------------------------------------------------------------
$sceneDir = Join-Path $out 'scenes'
New-Item -ItemType Directory -Force $sceneDir | Out-Null
$listEntries = @()
for ($i = 0; $i -lt $Scenes.Count; $i++) {
    $rtp = if ([System.IO.Path]::IsPathRooted($Scenes[$i])) { $Scenes[$i] } else { Join-Path $repo $Scenes[$i] }
    if (-not (Test-Path $rtp)) { throw "Scene not found: $rtp" }
    $src = Split-Path -Parent $rtp
    $file = Split-Path -Leaf $rtp
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($file)
    # A save in progress (or a failed one) leaves .tmp siblings: the copy could be half old.
    $tmp = Get-ChildItem $src -Filter "$file*.tmp" -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -ge (Get-Item $rtp).LastWriteTime }
    if ($tmp) {
        Warn "$file has newer .tmp files ($($tmp.Name -join ', ')): a save may be in progress or have failed. Re-save in Studio and repackage."
    }
    $dst = Join-Path $sceneDir (Split-Path -Leaf $src)
    New-Item -ItemType Directory -Force $dst | Out-Null
    foreach ($suffix in @('', '.bin', '.shared', '.aux.json', '.balloons.json')) {
        $p = "$rtp$suffix"
        if (Test-Path $p) { Copy-Item $p $dst }
        elseif ($suffix -in @('', '.bin')) { throw "Scene part missing: $p" }
    }
    $audio = Join-Path $src "${stem}_audio"
    if ((Test-Path $audio) -and -not $NoSceneAudio) {
        Copy-Item $audio (Join-Path $dst "${stem}_audio") -Recurse
        Warn "$file embeds audio from ${stem}_audio: check the clips' licence before giving this package to a customer."
    }
    $name = if ($i -lt $SceneNames.Count -and $SceneNames[$i]) { $SceneNames[$i] } else { $stem }
    $listEntries += [ordered]@{ name = $name; path = "scenes/$(Split-Path -Leaf $src)/$file" }
}
$listJson = [ordered]@{ scenes = $listEntries } | ConvertTo-Json -Depth 4
[System.IO.File]::WriteAllText((Join-Path $out 'player_scenes.json'), $listJson, (New-Object System.Text.UTF8Encoding($false)))

# --- 5. Licences + readme ---------------------------------------------------------
$assets = Join-Path $repo 'RayTrophiStudio\assets'
if (Test-Path (Join-Path $assets 'licenses')) { Copy-Item (Join-Path $assets 'licenses') (Join-Path $out 'licenses') -Recurse }
if (Test-Path (Join-Path $assets 'THIRD_PARTY_ASSETS.md')) { Copy-Item (Join-Path $assets 'THIRD_PARTY_ASSETS.md') $out }
$readme = @"
RayTrophi Player
================
Run RayTrophiPlayer.exe. The first scene in player_scenes.json opens at launch.

Keys
  F6        scene list (1-9 or click to open)
  F11       fullscreen        F1  HUD
  F2        render scale      F3  60 Hz / uncapped
  F9        clean the room (paint + balloons)
  F5        debug camera

Command line
  RayTrophiPlayer.exe [scene.rtp] --fullscreen --no-hud --render-scale 0.75

Requirements
  Windows 10/11 x64, a GPU with a current Vulkan 1.2 driver.
  If the program does not start, run redist\vc_redist.x64.exe once.
  Install into a user-writable folder (not Program Files): logs are written next to the exe.
  The first launch on a new GPU/driver compiles shaders in the driver and is slower.
"@
Set-Content -Path (Join-Path $out 'README.txt') -Value $readme -Encoding utf8

# --- 6. Zip + report -----------------------------------------------------------
$size = (Get-ChildItem $out -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("Package: {0}  ({1:N1} MB, {2} DLLs, {3} scenes)" -f $out, $size, $localDlls.Count, $Scenes.Count)
if ($Zip) {
    $zipPath = "$out.zip"
    if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
    Compress-Archive -Path (Join-Path $out '*') -DestinationPath $zipPath -CompressionLevel Optimal
    Write-Host ("Zip: {0}  ({1:N1} MB)" -f $zipPath, ((Get-Item $zipPath).Length / 1MB))
}
if ($warnings.Count -gt 0) {
    Write-Host ''
    Write-Host "$($warnings.Count) warning(s):"
    $warnings | ForEach-Object { Write-Host "  - $_" }
}
