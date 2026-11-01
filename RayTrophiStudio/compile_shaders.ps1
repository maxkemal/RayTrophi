param(
    [switch]$ValidateOnly,
    [switch]$VolumeOnly,
    [switch]$SplitVolumeOnly,
    [ValidateSet('Auto', 'Off', 'Full', 'Modular')]
    [string]$VolumeOptimization = 'Off'
)

$ErrorActionPreference = 'Stop'
if ($VolumeOnly -and $SplitVolumeOnly) {
    throw 'Choose VolumeOnly or SplitVolumeOnly, not both.'
}
if ($VolumeOptimization -eq 'Modular') {
    throw ('Modular deployment is disabled: its validated SPIR-V crashed during ' +
        'RT driver compilation. Use -VolumeOptimization Off. ' +
        'Function-boundary experiments remain available only in scripts/test/diagnose_volume_compiler.ps1.')
}
$shaderRoot = Join-Path $PSScriptRoot 'source\shaders'
$compilerPath = $null
$validatorPath = $null
. (Join-Path $PSScriptRoot 'compile_volume_modular.ps1')
. (Join-Path $PSScriptRoot 'volume_split_manifest.ps1')

function Publish-Spirv {
    param([string]$TemporaryPath, [string]$TargetPath)
    for ($attempt = 0; $attempt -lt 8; $attempt++) {
        try {
            $bytes = [IO.File]::ReadAllBytes($TemporaryPath)
            if ($bytes.Length -lt 20 -or $bytes.Length % 4 -ne 0 -or
                [BitConverter]::ToUInt32($bytes, 0) -ne 0x07230203) {
                throw [IO.InvalidDataException]::new('Compiler output is not valid SPIR-V')
            }
            if ([IO.File]::Exists($TargetPath)) {
                # PowerShell 5 can bind $null to an empty backup path. Use a
                # real unique path, then remove the backup after publication.
                $backupPath = Join-Path ([IO.Path]::GetDirectoryName($TargetPath)) (
                    '.shader-backup-' + [Guid]::NewGuid().ToString('N') + '.tmp')
                [IO.File]::Replace($TemporaryPath, $TargetPath, $backupPath)
                try { [IO.File]::Delete($backupPath) }
                catch { Write-Host "Previous-SPV backup retained: $backupPath" }
            } else {
                [IO.File]::Move($TemporaryPath, $TargetPath)
            }
            return
        } catch [IO.IOException] {
            if ($_.Exception -is [IO.InvalidDataException] -or $attempt -eq 7) {
                throw
            }
            Start-Sleep -Milliseconds 200
        } catch [UnauthorizedAccessException] {
            if ($attempt -eq 7) { throw }
            Start-Sleep -Milliseconds 200
        }
    }
}

function Compile-Shader {
    param(
        [string]$SourcePath,
        [string]$TargetPath,
        [string[]]$CompilerFlags,
        [bool]$OptimizationFallback = $false,
        [bool]$ValidateOutput = $false
    )
    Write-Host ('Compiling: ' + [IO.Path]::GetFileName($SourcePath))
    $fallbackUsed = $false
    for ($attempt = 0; $attempt -lt 4; $attempt++) {
        # Compile beside the target under a unique name. Never truncate the old
        # working module; publish only after compiler success and header validation.
        $temporaryPath = Join-Path ([IO.Path]::GetDirectoryName($TargetPath)) (
            '.shader-' + [Guid]::NewGuid().ToString('N') + '.tmp')
        try {
            Write-Host ('  Flags: ' + ($CompilerFlags -join ' '))
            $savedPreference = $ErrorActionPreference
            try {
                $ErrorActionPreference = 'Continue'
                $messages = @(& $compilerPath $SourcePath '-o' $temporaryPath @CompilerFlags 2>&1)
                $compilerExit = $LASTEXITCODE
            } finally {
                $ErrorActionPreference = $savedPreference
            }
            foreach ($message in $messages) { Write-Host $message.ToString() }
            $failure = $null
            if ($compilerExit -ne 0) {
                $outputBusy = ($messages -join "`n") -match 'cannot open output file'
                if ($outputBusy -and $attempt -lt 3) {
                    Write-Host 'Temporary output unavailable; retrying with a new name...'
                    Start-Sleep -Milliseconds 200
                    continue
                }
                $failure = "glslc exit $compilerExit"
            } elseif ($ValidateOutput) {
                # Validate the staged module before it can replace the working
                # shader. A successful compiler exit alone is insufficient.
                $savedPreference = $ErrorActionPreference
                try {
                    $ErrorActionPreference = 'Continue'
                    # Match GL_EXT_scalar_block_layout and the scalarBlockLayout
                    # device feature requested by VulkanBackend. Vulkan target
                    # selection alone does not enable this optional layout rule.
                    $validation = @(& $validatorPath '--target-env' 'vulkan1.3' `
                        '--scalar-block-layout' $temporaryPath 2>&1)
                    $validationExit = $LASTEXITCODE
                } finally {
                    $ErrorActionPreference = $savedPreference
                }
                foreach ($message in $validation) { Write-Host $message.ToString() }
                if ($validationExit -ne 0) {
                    $failure = "spirv-val exit $validationExit"
                }
            }
            if ($null -ne $failure) {
                if ($OptimizationFallback -and !$fallbackUsed -and
                    ($CompilerFlags -contains '-O') -and $attempt -lt 3) {
                    Write-Warning "Volume optimization failed ($failure); retrying with -O0."
                    $CompilerFlags = @($CompilerFlags | ForEach-Object {
                        if ($_ -eq '-O') { '-O0' } else { $_ }
                    })
                    $fallbackUsed = $true
                    continue
                }
                throw "FAILED: $([IO.Path]::GetFileName($SourcePath)) ($failure); previous SPV preserved"
            }
            try {
                Publish-Spirv -TemporaryPath $temporaryPath -TargetPath $TargetPath
            } catch {
                throw "Cannot publish '$TargetPath'; previous SPV preserved. $($_.Exception.Message)"
            }
            Write-Host ('  OK: ' + [IO.Path]::GetFileName($TargetPath) +
                ' [' + ($CompilerFlags -join ' ') + ']')
            return
        } finally {
            if ([IO.File]::Exists($temporaryPath)) {
                Remove-Item -LiteralPath $temporaryPath -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

$plan = [Collections.Generic.List[object]]::new()
function Add-Shader {
    param([string]$SourcePath, [string]$OutputName, [string[]]$CompilerFlags)
    $plan.Add([pscustomobject]@{
        Source = $SourcePath
        Target = Join-Path $shaderRoot ($OutputName + '.spv')
        Flags = $CompilerFlags
    })
}

foreach ($extension in @('comp', 'vert', 'frag', 'geom', 'rgen', 'rmiss', 'rchit', 'rahit', 'rint', 'rcall')) {
    foreach ($shader in Get-ChildItem -LiteralPath $shaderRoot -File | Where-Object {
        $_.Extension -eq ('.' + $extension)
    } | Sort-Object Name) {
        if ($shader.Name -eq 'shadow_anyhit.rchit') { continue }
        $flags = @('--target-env=vulkan1.3', '-O')
        if ($shader.Name -eq 'volume_closesthit.rchit' -and
            $VolumeOptimization -in @('Off', 'Modular')) {
            # Explicit reference mode for performance/image comparisons.
            # Auto attempts -O and falls back on compile/validation failure;
            # Full requires optimization to succeed, preserving the old SPV otherwise.
            # Modular optimizes separately after retaining SPIR-V function boundaries.
            $flags = @('--target-env=vulkan1.3', '-O0')
        }
        if ($extension -in @('rgen', 'rmiss', 'rchit', 'rahit', 'rint', 'rcall')) {
            $flags += '--target-spv=spv1.4'
        }
        Add-Shader $shader.FullName $shader.BaseName $flags
    }
}
$splitFlags = @('--target-env=vulkan1.3', '--target-spv=spv1.4')
$splitFlags += if ($VolumeOptimization -eq 'Off') { '-O0' } else { '-O' }
Add-Shader (Join-Path $shaderRoot 'volume_closesthit.rchit') 'volume_fog_closesthit' `
    ($splitFlags + '-DRT_VOLUME_FOG_STAGE=1')
Add-Shader (Join-Path $shaderRoot 'volume_closesthit.rchit') 'volume_surface_closesthit' `
    ($splitFlags + '-DRT_VOLUME_SURFACE_STAGE=1')
Add-Shader (Join-Path $shaderRoot 'realtime_svgf.comp') 'realtime_svgf32' @(
    '-DRAW32=1', '--target-env=vulkan1.3', '-O')
Add-Shader (Join-Path $shaderRoot 'upscale_inputs.comp') 'upscale_inputs32' @(
    '-DRAW32=1', '--target-env=vulkan1.3', '-O')
$windowShaders = @(
    'sim_fluid_divergence', 'sim_fluid_divergence_var',
    'sim_fluid_cg_build_diag', 'sim_fluid_cg_build_diag_var',
    'sim_fluid_cg_spmv', 'sim_fluid_cg_spmv_var',
    'sim_fluid_cg_jacobi', 'sim_fluid_cg_copy', 'sim_fluid_cg_axpy', 'sim_fluid_cg_zpby',
    'sim_fluid_cg_dot', 'sim_fluid_cg_axpy_dev', 'sim_fluid_cg_zpby_dev',
    'sim_fluid_cg_jacobi_dot', 'sim_fluid_cg_spmv_dot', 'sim_fluid_cg_spmv_dot_var',
    'sim_fluid_cg_axpy2_dev'
)
foreach ($name in $windowShaders) {
    Add-Shader (Join-Path $shaderRoot ($name + '.comp')) ($name + '_window') @(
        '-DRT_FLUID_WINDOW=1', '--target-env=vulkan1.2', '-O')
}
Add-Shader (Join-Path $shaderRoot 'material_preview_frag.frag') 'material_preview_covered' @(
    '-DPREVIEW_COVERED_SHADING=1', '--target-env=vulkan1.3', '-O')
$sculptPath = Join-Path $PSScriptRoot 'shaders\sculpt.comp'
if (Test-Path -LiteralPath $sculptPath) {
    Add-Shader $sculptPath 'sculpt' @('--target-env=vulkan1.3', '-O')
}
if ($VolumeOnly) {
    $plan = @($plan | Where-Object {
        [IO.Path]::GetFileName($_.Target) -eq 'volume_closesthit.spv'
    })
    if ($plan.Count -ne 1) { throw 'Expected exactly one volume shader in the plan.' }
}
if ($SplitVolumeOnly) {
    $plan = @($plan | Where-Object {
        [IO.Path]::GetFileName($_.Target) -in @(
            'raygen.spv', 'volume_fog_closesthit.spv', 'volume_surface_closesthit.spv')
    })
    if ($plan.Count -ne 3) { throw 'Expected raygen and both split volume stages.' }
}
if ($ValidateOnly) {
    foreach ($item in $plan) {
        if (!(Test-Path -LiteralPath $item.Source -PathType Leaf)) {
            throw "Missing shader source: $($item.Source)"
        }
        if ([IO.Path]::GetFileName($item.Source) -eq 'volume_closesthit.rchit') {
            Write-Host ('Volume optimization: ' + $VolumeOptimization +
                '; flags: ' + ($item.Flags -join ' '))
            if ($VolumeOptimization -eq 'Modular') {
                Write-Host 'Postprocess: preserve function boundaries -> spirv-opt -O -> validation'
            }
        }
    }
    Write-Host "PASS shader plan: $($plan.Count) entries; compiler was not invoked"
    exit 0
}

$mutex = $null
$ownsMutex = $false
try {
    if ([string]::IsNullOrWhiteSpace($env:VULKAN_SDK)) {
        throw 'VULKAN_SDK is not set. Check the Vulkan SDK installation.'
    }
    $compilerPath = Join-Path $env:VULKAN_SDK 'Bin\glslc.exe'
    if (!(Test-Path -LiteralPath $compilerPath -PathType Leaf)) {
        throw "glslc not found: $compilerPath. Check VULKAN_SDK."
    }
    $validatorPath = Join-Path $env:VULKAN_SDK 'Bin\spirv-val.exe'
    if (!(Test-Path -LiteralPath $validatorPath -PathType Leaf)) {
        throw "spirv-val not found: $validatorPath. Volume shaders require validation."
    }
    if ($VolumeOptimization -eq 'Modular') {
        $modularOptimizer = Join-Path $env:VULKAN_SDK 'Bin\spirv-opt.exe'
        $modularBoundaryTool = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot `
            '..\scripts\shaders\preserve_spirv_functions.py'))
        foreach ($dependency in @($modularOptimizer, $modularBoundaryTool)) {
            if (!(Test-Path -LiteralPath $dependency -PathType Leaf)) {
                throw "Missing Modular build dependency: $dependency"
            }
        }
        $modularPython = (Get-Command python.exe -ErrorAction Stop).Source
    }
    $hasher = [Security.Cryptography.SHA256]::Create()
    try {
        $digest = $hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($shaderRoot.ToLowerInvariant()))
        $lockName = 'Local\RayTrophiShaderBuild-' + [BitConverter]::ToString($digest).Replace('-', '')
    } finally { $hasher.Dispose() }
    $mutex = [Threading.Mutex]::new($false, $lockName)
    try { $ownsMutex = $mutex.WaitOne(0) }
    catch [Threading.AbandonedMutexException] { $ownsMutex = $true }
    if (!$ownsMutex) { throw 'Another compile_shaders build is running for this folder.' }

    if ($SplitVolumeOnly) {
        $splitBackup = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot (
            '..\docs\dev\.captures\volume-split-rollback-' + [guid]::NewGuid().ToString('N'))))
        $backupDirectories = @{
            source = $shaderRoot
            release = Join-Path $PSScriptRoot '..\x64\Release\shaders'
            debug = Join-Path $PSScriptRoot '..\x64\Debug\shaders'
            build_release = Join-Path $PSScriptRoot '..\build\Release\shaders'
        }
        foreach ($label in $backupDirectories.Keys) {
            if (!(Test-Path -LiteralPath $backupDirectories[$label] -PathType Container)) { continue }
            $destination = Join-Path $splitBackup $label
            [IO.Directory]::CreateDirectory($destination) | Out-Null
            foreach ($name in @('raygen.spv', 'volume_fog_closesthit.spv',
                    'volume_surface_closesthit.spv', 'volume_split_stage_set.txt')) {
                $previous = Join-Path $backupDirectories[$label] $name
                if (Test-Path -LiteralPath $previous -PathType Leaf) {
                    Copy-Item -LiteralPath $previous -Destination (Join-Path $destination $name)
                }
            }
        }
        Write-Host "Pre-split shader backup: $splitBackup"
    }

    foreach ($item in $plan) {
        $isVolume = [IO.Path]::GetFileName($item.Source) -eq 'volume_closesthit.rchit'
        $validateStage = $isVolume -or [IO.Path]::GetFileName($item.Target) -eq 'raygen.spv'
        if ($isVolume -and $VolumeOptimization -eq 'Modular') {
            Compile-ModularVolumeShader $item.Source $item.Target $item.Flags `
                -PythonPath $modularPython -OptimizerPath $modularOptimizer `
                -BoundaryToolPath $modularBoundaryTool
        } else {
            Compile-Shader $item.Source $item.Target $item.Flags `
                -OptimizationFallback ($isVolume -and $VolumeOptimization -eq 'Auto') `
                -ValidateOutput $validateStage
        }
    }
    if (!$VolumeOnly) { Write-VolumeSplitManifest $shaderRoot }
    # sphere_closesthit.spv: the foam-sphere hit group reuses closesthit.spv (2026-10-13).
    $staleNames = @('gradient_test.spv', 'miss.rmiss.spv', 'raygen.rgen.spv', 'rgen.spv', 'sphere_closesthit.spv', 'volume_closesthit_rayquery.spv')
    if ($VolumeOnly -or $SplitVolumeOnly) { $staleNames = @() }
    foreach ($name in $staleNames) {
        $path = Join-Path $shaderRoot $name
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Force }
    }
    foreach ($relative in @('..\x64\Release', '..\x64\Debug', '..\build\Release')) {
        $runtimeRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot $relative))
        if (!(Test-Path -LiteralPath $runtimeRoot -PathType Container)) { continue }
        $runtimeShaders = Join-Path $runtimeRoot 'shaders'
        [IO.Directory]::CreateDirectory($runtimeShaders) | Out-Null
        $artifacts = if ($VolumeOnly -or $SplitVolumeOnly) {
            $plan | ForEach-Object { Get-Item -LiteralPath $_.Target }
        } else {
            Get-ChildItem -LiteralPath $shaderRoot -Filter '*.spv' -File
        }
        foreach ($artifact in $artifacts) {
            $destination = Join-Path $runtimeShaders $artifact.Name
            $temporaryPath = Join-Path $runtimeShaders ('.shader-' + [Guid]::NewGuid().ToString('N') + '.tmp')
            try {
                [IO.File]::Copy($artifact.FullName, $temporaryPath)
                Publish-Spirv $temporaryPath $destination
            } finally {
                if (Test-Path -LiteralPath $temporaryPath) {
                    Remove-Item -LiteralPath $temporaryPath -Force -ErrorAction SilentlyContinue
                }
            }
        }
        if (!$VolumeOnly) { Write-VolumeSplitManifest $runtimeShaders }
        foreach ($name in $staleNames) {
            $path = Join-Path $runtimeShaders $name
            if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Force }
        }
    }
    Write-Host '===== All shaders compiled and deployed successfully ====='
    exit 0
} catch {
    Write-Host $_.Exception.Message
    exit 1
} finally {
    if ($ownsMutex) { $mutex.ReleaseMutex() }
    if ($null -ne $mutex) { $mutex.Dispose() }
}
