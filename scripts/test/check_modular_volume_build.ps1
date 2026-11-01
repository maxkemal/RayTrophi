# Build/publication contract tests with mock tools only. No real shader tools run.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\..\RayTrophiStudio\compile_volume_modular.ps1')
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$parent = Join-Path $repoRoot 'docs\dev\.captures'
$scratch = Join-Path $parent ('modular-volume-test-' + [guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($scratch) | Out-Null
$target = Join-Path $scratch 'volume_closesthit.spv'
$validatorPath = 'Mock-Validator'

function Write-MockModule([string]$Path, [byte]$Marker) {
    $bytes = New-Object byte[] 20
    [BitConverter]::GetBytes([uint32]0x07230203).CopyTo($bytes, 0)
    $bytes[16] = $Marker
    [IO.File]::WriteAllBytes($Path, $bytes)
}
function Compile-Shader {
    param($SourcePath, $TargetPath, $CompilerFlags, $ValidateOutput)
    $script:events.Add('compile')
    if (!$ValidateOutput -or !($CompilerFlags -contains '-O0') -or
        $TargetPath -eq $target) { throw 'Reference compilation contract violated.' }
    if ($script:case -eq 'compile_failure') { throw 'Mock compile failure' }
    Write-MockModule $TargetPath 11
}
function Mock-Boundaries {
    $script:events.Add('boundaries')
    if ($script:case -eq 'boundary_failure') {
        $global:LASTEXITCODE = 1
        return
    }
    $outputIndex = [Array]::IndexOf(@($args), '--output') + 1
    Write-MockModule $args[$outputIndex] 22
    $global:LASTEXITCODE = 0
}
function Mock-Optimizer {
    $script:events.Add('optimize')
    if (!(@($args) -ccontains '-O')) { throw 'Missing optimizer -O' }
    if ($script:case -eq 'optimizer_failure') {
        $global:LASTEXITCODE = -1073741571
        return
    }
    $outputIndex = [Array]::IndexOf(@($args), '-o') + 1
    Write-MockModule $args[$outputIndex] 33
    $global:LASTEXITCODE = 0
}
function Mock-Validator {
    if (!(@($args) -ccontains '--scalar-block-layout')) { throw 'Missing scalar layout' }
    $bytes = [IO.File]::ReadAllBytes($args[-1])
    $stage = if ($bytes[16] -eq 22) { 'boundaries' } else { 'optimized' }
    $script:events.Add('validate-' + $stage)
    $global:LASTEXITCODE = if ($script:case -eq ($stage + '_validation_failure')) { 1 } else { 0 }
}
function Publish-Spirv {
    param($TemporaryPath, $TargetPath)
    $script:events.Add('publish')
    if ($script:case -eq 'publish_failure') { throw 'Mock publication failure' }
    [IO.File]::WriteAllBytes($TargetPath, [IO.File]::ReadAllBytes($TemporaryPath))
}

try {
    foreach ($script:case in @('success', 'compile_failure', 'boundary_failure',
            'boundaries_validation_failure', 'optimizer_failure',
            'optimized_validation_failure', 'publish_failure')) {
        $script:events = [Collections.Generic.List[string]]::new()
        [IO.File]::WriteAllBytes($target, [byte[]](1, 2, 3, 4))
        $failed = $false
        try {
            Compile-ModularVolumeShader 'mock.rchit' $target @(
                '--target-env=vulkan1.3', '--target-spv=spv1.4', '-O0') `
                -PythonPath 'Mock-Boundaries' -OptimizerPath 'Mock-Optimizer' `
                -BoundaryToolPath 'mock-boundaries.py'
        } catch {
            $failed = $true
        }
        if ($failed -ne ($script:case -ne 'success')) { throw "Wrong outcome: $script:case" }
        $actual = [IO.File]::ReadAllBytes($target)
        if ($script:case -eq 'success') {
            if ($actual[16] -ne 33 -or ($events -join ',') -ne
                'compile,boundaries,validate-boundaries,optimize,validate-optimized,publish') {
                throw 'Published the wrong module or wrong validation order.'
            }
        } elseif (($actual -join ',') -ne '1,2,3,4') {
            throw "Old working module changed after failure: $script:case"
        }
        if ($script:case -notin @('success', 'publish_failure') -and $events.Contains('publish')) {
            throw "Publication attempted before successful validation: $script:case"
        }
        if (@(Get-ChildItem -LiteralPath $scratch -Filter '.shader-modular-*.tmp').Count) {
            throw "Temporary module leaked: $script:case"
        }
        Write-Output "PASS $script:case"
    }
    Write-Output 'PASS: 7 Modular publication cases; only mock tools invoked.'
    # The production entrypoint must now refuse this driver-rejected mode before
    # dependencies, compilation, publication or deployment, even for plan checks.
    $savedPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $driver = Join-Path $repoRoot 'RayTrophiStudio\compile_shaders.ps1'
        $blocked = @(& (Join-Path $PSHOME 'powershell.exe') -NoProfile `
            -ExecutionPolicy Bypass -File $driver -VolumeOnly `
            -VolumeOptimization Modular -ValidateOnly 2>&1)
        $blockedExit = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $savedPreference
    }
    if ($blockedExit -eq 0 -or ($blocked -join "`n") -notmatch 'Modular deployment is disabled') {
        throw 'Driver-rejected mode was not blocked at the production entrypoint.'
    }
    Write-Output 'PASS: production Modular mode refused before shader tools run.'
} finally {
    $resolved = [IO.Path]::GetFullPath($scratch)
    if (!$resolved.StartsWith($parent.TrimEnd('\') + '\',
            [StringComparison]::OrdinalIgnoreCase) -or
        ![IO.Path]::GetFileName($resolved).StartsWith('modular-volume-test-')) {
        throw "Unsafe cleanup path: $resolved"
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
