param([switch]$ValidateOnly)

$ErrorActionPreference = 'Stop'
$shaderRoot = Join-Path $PSScriptRoot 'source\shaders'
$compilerPath = $null

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
    param([string]$SourcePath, [string]$TargetPath, [string[]]$CompilerFlags)
    Write-Host ('Compiling: ' + [IO.Path]::GetFileName($SourcePath))
    for ($attempt = 0; $attempt -lt 3; $attempt++) {
        # Compile beside the target under a unique name. Never truncate the old
        # working module; publish only after compiler success and header validation.
        $temporaryPath = Join-Path ([IO.Path]::GetDirectoryName($TargetPath)) (
            '.shader-' + [Guid]::NewGuid().ToString('N') + '.tmp')
        try {
            $savedPreference = $ErrorActionPreference
            try {
                $ErrorActionPreference = 'Continue'
                $messages = @(& $compilerPath $SourcePath '-o' $temporaryPath @CompilerFlags 2>&1)
                $compilerExit = $LASTEXITCODE
            } finally {
                $ErrorActionPreference = $savedPreference
            }
            foreach ($message in $messages) { Write-Host $message.ToString() }
            if ($compilerExit -ne 0) {
                $outputBusy = ($messages -join "`n") -match 'cannot open output file'
                if ($outputBusy -and $attempt -lt 2) {
                    Write-Host 'Temporary output unavailable; retrying with a new name...'
                    Start-Sleep -Milliseconds 200
                    continue
                }
                throw "FAILED: $([IO.Path]::GetFileName($SourcePath)) (glslc exit $compilerExit)"
            }
            try {
                Publish-Spirv -TemporaryPath $temporaryPath -TargetPath $TargetPath
            } catch {
                throw "Cannot publish '$TargetPath'; previous SPV preserved. $($_.Exception.Message)"
            }
            Write-Host ('  OK: ' + [IO.Path]::GetFileName($TargetPath))
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

foreach ($extension in @('comp', 'vert', 'frag', 'geom', 'rgen', 'rmiss', 'rchit', 'rahit', 'rint')) {
    foreach ($shader in Get-ChildItem -LiteralPath $shaderRoot -File | Where-Object {
        $_.Extension -eq ('.' + $extension)
    } | Sort-Object Name) {
        if ($shader.Name -eq 'shadow_anyhit.rchit') { continue }
        $flags = @('--target-env=vulkan1.3', '-O')
        if ($extension -in @('rgen', 'rmiss', 'rchit', 'rahit', 'rint')) {
            $flags += '--target-spv=spv1.4'
        }
        Add-Shader $shader.FullName $shader.BaseName $flags
    }
}
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
Add-Shader (Join-Path $shaderRoot 'closesthit.rchit') 'sphere_closesthit' @(
    '-DSPHERE_HIT=1', '--target-env=vulkan1.3', '--target-spv=spv1.4', '-O')
$sculptPath = Join-Path $PSScriptRoot 'shaders\sculpt.comp'
if (Test-Path -LiteralPath $sculptPath) {
    Add-Shader $sculptPath 'sculpt' @('--target-env=vulkan1.3', '-O')
}
if ($ValidateOnly) {
    foreach ($item in $plan) {
        if (!(Test-Path -LiteralPath $item.Source -PathType Leaf)) {
            throw "Missing shader source: $($item.Source)"
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
    $hasher = [Security.Cryptography.SHA256]::Create()
    try {
        $digest = $hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($shaderRoot.ToLowerInvariant()))
        $lockName = 'Local\RayTrophiShaderBuild-' + [BitConverter]::ToString($digest).Replace('-', '')
    } finally { $hasher.Dispose() }
    $mutex = [Threading.Mutex]::new($false, $lockName)
    try { $ownsMutex = $mutex.WaitOne(0) }
    catch [Threading.AbandonedMutexException] { $ownsMutex = $true }
    if (!$ownsMutex) { throw 'Another compile_shaders build is running for this folder.' }

    foreach ($item in $plan) {
        Compile-Shader $item.Source $item.Target $item.Flags
    }
    $staleNames = @('gradient_test.spv', 'miss.rmiss.spv', 'raygen.rgen.spv', 'rgen.spv')
    foreach ($name in $staleNames) {
        $path = Join-Path $shaderRoot $name
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Force }
    }
    foreach ($relative in @('..\x64\Release', '..\x64\Debug', '..\build\Release')) {
        $runtimeRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot $relative))
        if (!(Test-Path -LiteralPath $runtimeRoot -PathType Container)) { continue }
        $runtimeShaders = Join-Path $runtimeRoot 'shaders'
        [IO.Directory]::CreateDirectory($runtimeShaders) | Out-Null
        foreach ($artifact in Get-ChildItem -LiteralPath $shaderRoot -Filter '*.spv' -File) {
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
