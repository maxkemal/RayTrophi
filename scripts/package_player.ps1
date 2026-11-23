<#
.SYNOPSIS
  Builds a self-contained RayTrophi Player folder (and optional zip) from the
  current Release output and a list of .rtp scenes.

.DESCRIPTION
  Layout of the package:
    RayTrophiPlayer.exe + the DLLs it actually loads (resolved with dumpbin)
    VC++ runtime DLLs (app-local, from the Visual Studio redist folder)
    shaders\                 Release shaders selected by player_shader_manifest.json
    scenes\<folder>\         .rtp + .rtp.bin + .rtp.shared + sidecars + <stem>_audio\
    player_scenes.json       scene list; F6 opens the game library
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
    # Scene order determines the order of cards in the library.
    [string[]]$Scenes = @(
        'RayTrophiStudio\assets\scenes\pastel-paint-workshop\workshop.rtp',
        'RayTrophiStudio\assets\scenes\kids-play-room\kids-play-room.rtp'
    ),
    # Display names for the F6 list, same order as -Scenes. Empty = file name.
    [string[]]$SceneNames = @('Gizli Oyuncaklar', 'Pastel Oyun Odası'),
    # Canonical shared output. Use other destinations only for explicit diagnostics.
    [string]$OutDir = 'dist\RayTrophiPlayer',
    [switch]$Zip,
    # Leave the scenes' embedded audio out (third-party clips without a
    # confirmed licence live there; see commit d1c1709a).
    [switch]$NoSceneAudio,
    [switch]$NoPerception,
    [switch]$ShaderInventoryOnly,
    [string]$BridgePython = 'tools\perception_bridge\.venv\Scripts\python.exe',
    [string]$PoseModel = 'tools\perception_bridge\models\pose_landmarker.task'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$release = Join-Path $repo 'x64\Release'
$exe = Join-Path $release 'RayTrophiPlayer.exe'
$out = if ([System.IO.Path]::IsPathRooted($OutDir)) { $OutDir } else { Join-Path $repo $OutDir }
$packageTarget = [IO.Path]::GetFullPath($out)
$marker = '.raytrophi_player_package'
$warnings = New-Object System.Collections.Generic.List[string]
function Warn([string]$m) { $warnings.Add($m); Write-Warning $m }

$shaderDirectory = Join-Path $release 'shaders'
$shaderManifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'player_shader_manifest.json') -Raw |
    ConvertFrom-Json
$shaderGroups = @{}
foreach ($classification in @('required', 'conditional')) {
    foreach ($group in $shaderManifest.$classification.PSObject.Properties) {
        foreach ($name in $group.Value) {
            if ($shaderGroups.ContainsKey($name) -or $name -in $shaderManifest.studioOnly) {
                throw "Conflicting Player shader classification: $name"
            }
            $shaderGroups[$name] = "$classification/$($group.Name)"
            if ($classification -eq 'required' -and
                -not (Test-Path -LiteralPath (Join-Path $shaderDirectory $name) -PathType Leaf)) {
                throw "Required Player shader missing from Release: $name"
            }
        }
    }
}
$shaderFiles = @(Get-ChildItem -LiteralPath $shaderDirectory -File | Sort-Object Name)
$playerShaderFiles = @($shaderFiles | Where-Object { $_.Name -notin $shaderManifest.studioOnly })

if ($ShaderInventoryOnly) {
    $inventory = foreach ($file in $shaderFiles | Where-Object Extension -eq '.spv') {
        $bytes = [IO.File]::ReadAllBytes($file.FullName)
        if ($bytes.Length -lt 20 -or ($bytes.Length % 4) -ne 0 -or
            [BitConverter]::ToUInt32($bytes, 0) -ne 0x07230203) {
            throw "Invalid SPIR-V header: $($file.Name)"
        }
        $version = [BitConverter]::ToUInt32($bytes, 4)
        $group = if ($file.Name -in $shaderManifest.studioOnly) { 'studioOnly' }
                 elseif ($shaderGroups.ContainsKey($file.Name)) { $shaderGroups[$file.Name] }
                 else { 'retained/unclassified' }
        [PSCustomObject]@{
            name = $file.Name
            group = $group
            spirv = '{0}.{1}' -f (($version -shr 16) -band 255), (($version -shr 8) -band 255)
        }
    }
    $inventory | ConvertTo-Json -Depth 3
    return
}

# --- 0. Sanity ---------------------------------------------------------------
if (-not (Test-Path $exe)) { throw "Player exe not found: $exe (build RayTrophiPlayer first)" }
$exeTime = (Get-Item $exe).LastWriteTime
$playerMain = Join-Path $repo 'RayTrophiStudio\source\src\Core\PlayerMain.cpp'
if ((Test-Path $playerMain) -and (Get-Item $playerMain).LastWriteTime -gt $exeTime) {
    Warn "PlayerMain.cpp is newer than RayTrophiPlayer.exe ($exeTime): the exe may be stale (no scene list / F6)."
}
if ($Scenes.Count -lt 1) { throw 'At least one scene is required.' }

# Assemble away from the active package. Preserve its runtime settings/tokens/logs.
if (Test-Path -LiteralPath $packageTarget) {
    if (-not (Test-Path -LiteralPath (Join-Path $packageTarget $marker))) {
        throw "Output folder exists and is not a Player package: $packageTarget"
    }
}
$out = $packageTarget + '.staging-' + [Guid]::NewGuid().ToString('N')
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
$packageShaderDirectory = Join-Path $out 'shaders'
New-Item -ItemType Directory -Force $packageShaderDirectory | Out-Null
foreach ($shaderFile in $playerShaderFiles) {
    Copy-Item -LiteralPath $shaderFile.FullName -Destination $packageShaderDirectory
}
# Shared locale catalogs are product resources; local language preferences are preserved.
$localeSource = Join-Path $repo 'RayTrophiStudio/assets/i18n'
if (-not (Test-Path -LiteralPath (Join-Path $localeSource 'locales.json'))) {
    throw 'Missing interface language catalogs.'
}
New-Item -ItemType Directory -Force (Join-Path $out 'assets') | Out-Null
Copy-Item -LiteralPath $localeSource -Destination (Join-Path $out 'assets/i18n') -Recurse -Force


# --- 4. Scenes -------------------------------------------------------------------
$sceneDir = Join-Path $out 'scenes'
New-Item -ItemType Directory -Force $sceneDir | Out-Null
$listEntries = @()
$audioEntries = @()
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
    # Intentional game cover + field-test instructions (not verification captures).
    foreach ($assetName in @("$stem.preview.png", "$stem.TEST.md")) {
        $assetPath = Join-Path $src $assetName
        if (Test-Path -LiteralPath $assetPath) { Copy-Item -LiteralPath $assetPath -Destination $dst }
    }
    if ($i -eq 0 -and (Test-Path -LiteralPath (Join-Path $src "$stem.TEST.md"))) {
        Copy-Item -LiteralPath (Join-Path $src "$stem.TEST.md") -Destination (Join-Path $out 'KAMERA_TESTI.md')
    }
    # Distribute a reviewed live design snapshot without changing the authored .rtp.
    $distributionPath = Join-Path $src "$stem.distribution.json"
    if (Test-Path -LiteralPath $distributionPath) {
        $distribution = Get-Content -LiteralPath $distributionPath -Raw | ConvertFrom-Json
        if ($distribution.schema -ne 1 -or -not $distribution.game_design.balloon_demo) {
            throw "Invalid scene distribution design: $distributionPath"
        }
        $projectCopy = Join-Path $dst $file
        $document = Get-Content -LiteralPath $projectCopy -Raw | ConvertFrom-Json
        $document | Add-Member -NotePropertyName game_design -NotePropertyValue $distribution.game_design -Force
        $document | Add-Member -NotePropertyName project_name -NotePropertyValue $distribution.name -Force
        $document | Add-Member -NotePropertyName description -NotePropertyValue $distribution.description -Force
        [IO.File]::WriteAllText($projectCopy, ($document | ConvertTo-Json -Depth 100),
                              (New-Object Text.UTF8Encoding($false)))
        # Keep the legacy fallback consistent too: no obsolete fixture node names.
        [IO.File]::WriteAllText("$projectCopy.balloons.json",
            ($distribution.game_design.balloon_demo | ConvertTo-Json -Depth 100),
            (New-Object Text.UTF8Encoding($false)))
        Copy-Item -LiteralPath $distributionPath -Destination $dst
    }
    # World HDRI / sky overlay images saved next to the project (ProjectManager).
    $envDir = Join-Path $src "${stem}_environment"
    if (Test-Path $envDir) { Copy-Item $envDir (Join-Path $dst "${stem}_environment") -Recurse }
    $audio = Join-Path $src "${stem}_audio"
    if ((Test-Path $audio) -and -not $NoSceneAudio) {
        Copy-Item $audio (Join-Path $dst "${stem}_audio") -Recurse
        Warn "$file embeds audio from ${stem}_audio: check the clips' licence before giving this package to a customer."
    }
    if (-not $NoSceneAudio) {
        $scene = Get-Content -LiteralPath $rtp -Raw | ConvertFrom-Json
        foreach ($asset in $scene.audio.assets) {
            if (-not $asset.path) { continue }
            if ([IO.Path]::IsPathRooted($asset.path)) {
                throw "Audio asset must be scene-relative before packaging: $file / $($asset.id)"
            }
            $audioFile = [IO.Path]::GetFullPath((Join-Path $dst $asset.path))
            $sceneRoot = [IO.Path]::GetFullPath($dst).TrimEnd('\') + '\'
            if (-not $audioFile.StartsWith($sceneRoot, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Audio asset escapes its packaged scene: $file / $($asset.id)"
            }
            if (-not (Test-Path -LiteralPath $audioFile -PathType Leaf)) {
                throw "Packaged audio asset missing: $file / $($asset.id) / $($asset.path)"
            }
            $sourceAudio = Join-Path $src $asset.path
            $hash = (Get-FileHash -LiteralPath $audioFile -Algorithm SHA256).Hash
            if ($hash -ne (Get-FileHash -LiteralPath $sourceAudio -Algorithm SHA256).Hash) {
                throw "Packaged audio differs from source: $file / $($asset.id)"
            }
            $audioEntries += [ordered]@{
                scene = $file; id = $asset.id; path = $asset.path; sha256 = $hash
            }
        }
    }
    $name = if ($i -lt $SceneNames.Count -and $SceneNames[$i]) { $SceneNames[$i] } else { $stem }
    $listEntries += [ordered]@{ name = $name; path = "scenes/$(Split-Path -Leaf $src)/$file" }
}
# The scene folders carry runtime assets only, including in recursive sidecars.
$sourceAssets = @(Get-ChildItem -LiteralPath $sceneDir -Recurse -File |
    Where-Object { $_.Name -match '\.(blend\d*|glb|gltf)$' })
if ($sourceAssets.Count -ne 0) {
    throw "Authoring files in staged scenes: $($sourceAssets.FullName -join ', ')"
}
# A clean recipient gets portable library roots, no camera IDs/calibration/tokens.
$defaults = [ordered]@{ version = 1; roots = @('scenes'); locale = 'tr'; appearance = 'camera' }
[IO.File]::WriteAllText((Join-Path $out 'player_operator.json'),
    ($defaults | ConvertTo-Json -Depth 4), (New-Object Text.UTF8Encoding($false)))
$listJson = [ordered]@{ scenes = $listEntries } | ConvertTo-Json -Depth 4
[System.IO.File]::WriteAllText((Join-Path $out 'player_scenes.json'), $listJson, (New-Object System.Text.UTF8Encoding($false)))

# --- 5. Licences + readme ---------------------------------------------------------
$assets = Join-Path $repo 'RayTrophiStudio\assets'
if (Test-Path (Join-Path $assets 'licenses')) { Copy-Item (Join-Path $assets 'licenses') (Join-Path $out 'licenses') -Recurse }
if (Test-Path (Join-Path $assets 'THIRD_PARTY_ASSETS.md')) { Copy-Item (Join-Path $assets 'THIRD_PARTY_ASSETS.md') $out }
$playerExeHash = (Get-FileHash -LiteralPath (Join-Path $out 'RayTrophiPlayer.exe') -Algorithm SHA256).Hash
$readme = @"
RayTrophi Player
================
Run RayTrophiPlayer.exe. With multiple bundled games, select a card in the library.
A single bundled game opens directly.
For this demo, read KAMERA_TESTI.md for mouse/camera selection and camera setup.

Keys
  F6        game library
  F11       fullscreen        F1  HUD
  F2        operator console  F3  60 Hz / uncapped
  F8        clear paint and in-flight paint drops
  F9        clean the room (paint + balloons)
  F5        debug camera

Command line
  RayTrophiPlayer.exe [scene.rtp] --fullscreen --no-hud --render-scale 0.75

Requirements
  Windows 10/11 x64, a GPU and driver supporting Vulkan 1.3 and the
  RayFusion and scene compute features used by the selected project.
  If the program does not start, run redist\vc_redist.x64.exe once.
  Install into a user-writable folder (not Program Files): logs are written next to the exe.
  The first launch on a new GPU/driver compiles shaders in the driver and is slower.

Vulkan 1.2 candidate test
  Extract the complete package to a writable folder. Launch RayTrophiPlayer.exe.
  Check that the workshop appears, pop a balloon, and check that paint reaches
  the room surfaces. If the camera is configured, also check the depth card.
  Report which step passed or failed. Include SceneLog.txt from the package
  folder before launching again. If generated during the test, also include
  StartupCrashStack.txt. Do not send the entire folder.
  Packaged Player EXE SHA-256: $playerExeHash
"@
Set-Content -Path (Join-Path $out 'README.txt') -Value $readme -Encoding utf8

# --- 6. External perception runtime (no Python link in Player) ------------------
if (-not $NoPerception) {
    $bridgeExe = if ([IO.Path]::IsPathRooted($BridgePython)) { $BridgePython } else { Join-Path $repo $BridgePython }
    $modelFile = if ([IO.Path]::IsPathRooted($PoseModel)) { $PoseModel } else { Join-Path $repo $PoseModel }
    if (-not (Test-Path -LiteralPath $bridgeExe)) { throw "Bridge Python missing: $bridgeExe" }
    if (-not (Test-Path -LiteralPath $modelFile)) { throw "Pose model missing: $modelFile" }
    & $bridgeExe (Join-Path $repo 'tools\perception_bridge\package_runtime.py') `
        --output (Join-Path $out 'perception') --model $modelFile
    if ($LASTEXITCODE -ne 0) { throw 'Perception runtime packaging/portable import validation failed.' }
    Add-Content -LiteralPath (Join-Path $out 'README.txt') -Encoding utf8 -Value @"

Perception runtime
  perception\python\python.exe is a separate portable Python runtime.
  Game scripts use perception\bridge\game_bridge.py (BridgeProcess context manager).
  The worker requires --model and --calibration; --video replays an existing MP4.
  F2 > Camera selects a device, previews, calibrates and starts tracking.
  F2 > Game settings > Camera / skeleton controls toggles body input; mouse remains usable.
  Frames stay in memory. No camera recordings or machine-specific calibration are included.
  See perception\manifest.json for dependency versions and the model SHA-256.
"@
}

# --- 7. Publish the complete package without deleting runtime-owned files ------
$manifest = [ordered]@{
    schema = 1
    generated_at = (Get-Date -Format o)
    canonical_output = 'dist/RayTrophiPlayer'
    scene_audio = (-not $NoSceneAudio)
    perception = (-not $NoPerception)
    audio_assets = @($audioEntries)
    scenes = @($listEntries)
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $out 'package_manifest.json') -Encoding utf8
if ($Zip) {
    # Archive generated payload only, never the active package's local IPC tokens/logs.
    $zipPath = "$packageTarget.zip"
    $zipTemporary = "$out.zip"
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    try {
        [IO.Compression.ZipFile]::CreateFromDirectory(
            $out, $zipTemporary, [IO.Compression.CompressionLevel]::Fastest, $false)
        if ([IO.File]::Exists($zipPath)) {
            [IO.File]::Replace($zipTemporary, $zipPath, [NullString]::Value)
        } else {
            [IO.File]::Move($zipTemporary, $zipPath)
        }
    } finally {
        if ([IO.File]::Exists($zipTemporary)) {
            Remove-Item -LiteralPath $zipTemporary
        }
    }
}
New-Item -ItemType Directory -Force $packageTarget | Out-Null
# Explicit lean-package switches must also work when refreshing an existing package.
$excludedTargets = @()
if ($NoPerception) { $excludedTargets += (Join-Path $packageTarget 'perception') }
if ($NoSceneAudio) {
    foreach ($entry in $listEntries) {
        $scenePath = Join-Path $packageTarget $entry.path
        $audioName = [IO.Path]::GetFileNameWithoutExtension($scenePath) + '_audio'
        $excludedTargets += (Join-Path (Split-Path -Parent $scenePath) $audioName)
    }
}
foreach ($excludedTarget in $excludedTargets) {
    $resolved = [IO.Path]::GetFullPath($excludedTarget)
    if (-not $resolved.StartsWith($packageTarget.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Excluded generated content is outside the package.'
    }
    if (Test-Path -LiteralPath $resolved) { Remove-Item -LiteralPath $resolved -Recurse -Force }
}
# The generated scene tree is replaced, so removing a scene from -Scenes really
# removes its .rtp and assets from distribution and recursive library discovery.
$publishedScenes = [IO.Path]::GetFullPath((Join-Path $packageTarget 'scenes'))
$packagePrefix = $packageTarget.TrimEnd('\') + '\'
if (-not $publishedScenes.StartsWith($packagePrefix, [StringComparison]::OrdinalIgnoreCase) -or
    [IO.Path]::GetFileName($publishedScenes) -ne 'scenes' -or
    -not (Test-Path -LiteralPath (Join-Path $out $marker))) {
    throw 'Refusing to reconcile an unverified generated scene directory.'
}
if (Test-Path -LiteralPath $publishedScenes) {
    Remove-Item -LiteralPath $publishedScenes -Recurse -Force
}
foreach ($item in Get-ChildItem -LiteralPath $out -Force) {
    # Keep this local installation's settings; the zip carries only clean defaults.
    if ($item.Name -eq 'player_operator.json' -and
        (Test-Path -LiteralPath (Join-Path $packageTarget $item.Name))) { continue }

    Copy-Item -LiteralPath $item.FullName -Destination $packageTarget -Recurse -Force
}
# Reconcile the known Studio-only files left by an older generated package.
$publishedShaders = Join-Path $packageTarget 'shaders'
foreach ($name in $shaderManifest.studioOnly) {
    if ($name -ne [IO.Path]::GetFileName($name) -or $name -notlike '*.spv') {
        throw "Invalid Studio-only shader filename: $name"
    }
    $oldShader = Join-Path $publishedShaders $name
    if (Test-Path -LiteralPath $oldShader -PathType Leaf) {
        Remove-Item -LiteralPath $oldShader -Force
    }
}
# Remove this invocation's verified, generated staging tree after publication.
$stagePath = [IO.Path]::GetFullPath($out)
if (-not $stagePath.StartsWith($packageTarget + '.staging-', [StringComparison]::OrdinalIgnoreCase) -or
    -not (Test-Path -LiteralPath (Join-Path $stagePath $marker))) {
    throw 'Refusing to remove an unverified package staging directory.'
}
Remove-Item -LiteralPath $stagePath -Recurse -Force
$out = $packageTarget

# --- 8. Zip + report -----------------------------------------------------------
$size = (Get-ChildItem $out -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("Package: {0}  ({1:N1} MB, {2} DLLs, {3} scenes)" -f $out, $size, $localDlls.Count, $Scenes.Count)
if ($Zip) {
    Write-Host ("Zip: {0}  ({1:N1} MB)" -f $zipPath, ((Get-Item $zipPath).Length / 1MB))
}
if ($warnings.Count -gt 0) {
    Write-Host ''
    Write-Host "$($warnings.Count) warning(s):"
    $warnings | ForEach-Object { Write-Host "  - $_" }
}
