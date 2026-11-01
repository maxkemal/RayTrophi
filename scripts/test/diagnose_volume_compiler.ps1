# User-run shader diagnosis. Outputs stay in ignored scratch; nothing is deployed.
param(
    [switch]$ValidateOnly,
    [string]$ReferenceModule,
    [switch]$PreserveFunctions,
    [ValidateRange(10, 7200)]
    [int]$TimeoutSeconds = 600
)

$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$shaderRoot = Join-Path $repoRoot 'RayTrophiStudio\source\shaders'
$source = Join-Path $shaderRoot 'volume_closesthit.rchit'
if ([string]::IsNullOrWhiteSpace($env:VULKAN_SDK)) {
    throw 'VULKAN_SDK is not set.'
}
$sdkBin = Join-Path $env:VULKAN_SDK 'Bin'
$compiler = Join-Path $sdkBin 'glslc.exe'
$optimizer = Join-Path $sdkBin 'spirv-opt.exe'
$validator = Join-Path $sdkBin 'spirv-val.exe'
foreach ($required in @($source, $compiler, $optimizer, $validator)) {
    if (!(Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing input/tool: $required"
    }
}
if (![string]::IsNullOrWhiteSpace($ReferenceModule)) {
    $ReferenceModule = (Resolve-Path -LiteralPath $ReferenceModule).Path
}
$boundaryTool = Join-Path $PSScriptRoot '..\shaders\preserve_spirv_functions.py'
$pythonPath = $null
if ($PreserveFunctions) {
    if (!(Test-Path -LiteralPath $boundaryTool -PathType Leaf)) {
        throw "Missing boundary tool: $boundaryTool"
    }
    $pythonPath = (Get-Command python.exe -ErrorAction Stop).Source
}
if ($ValidateOnly) {
    Write-Output 'PASS diagnosis plan: reference -> validation -> inline -> validation -> full -> validation'
    Write-Output "Timeout per process: $TimeoutSeconds seconds"
    Write-Output "Existing reference: $ReferenceModule; preserve functions: $PreserveFunctions"
    Write-Output 'No process invoked; no output directory created.'
    exit 0
}

$scratch = Join-Path $repoRoot ('docs\dev\.captures\volume-compiler-' +
    [guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($scratch) | Out-Null
$results = [Collections.Generic.List[object]]::new()
$summaryPath = Join-Path $scratch 'outcome.json'

function Save-Outcome {
    [pscustomobject]@{
        source = $source
        outputs = $scratch
        timeout_seconds = $TimeoutSeconds
        phases = @($results.ToArray())
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
}

function Invoke-DiagnosticPhase {
    param([string]$Name, [string]$Tool, [string[]]$ToolArguments)
    # All paths are locally resolved Windows paths (no embedded quote characters).
    # Quote every argument for Windows command-line serialization.
    $argumentLine = ($ToolArguments | ForEach-Object {
        if ($_.Contains('"')) { throw 'Unexpected quote in diagnostic argument.' }
        '"' + $_ + '"'
    }) -join ' '
    $stdout = Join-Path $scratch ($Name + '.stdout.log')
    $stderr = Join-Path $scratch ($Name + '.stderr.log')
    Write-Host "Starting $Name (limit $TimeoutSeconds seconds)"
    $timer = [Diagnostics.Stopwatch]::StartNew()
    $process = $null
    $stdoutTask = $null
    $stderrTask = $null
    $started = $false
    $completed = $false
    try {
        # Own the Process handle from Start through ExitCode retrieval. Windows
        # PowerShell Start-Process can return a detached Process with null ExitCode
        # when a short-lived redirected child finishes before the handle is retained.
        $process = [Diagnostics.Process]::new()
        $process.StartInfo.FileName = $Tool
        $process.StartInfo.Arguments = $argumentLine
        $process.StartInfo.WorkingDirectory = $shaderRoot
        $process.StartInfo.UseShellExecute = $false
        $process.StartInfo.CreateNoWindow = $true
        $process.StartInfo.RedirectStandardOutput = $true
        $process.StartInfo.RedirectStandardError = $true
        if (!$process.Start()) { throw "Cannot start diagnostic phase: $Name" }
        $started = $true
        # Drain both streams concurrently so verbose tools cannot block on a full pipe.
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        while (!$process.WaitForExit(500)) {
            if ($timer.Elapsed.TotalSeconds -ge $TimeoutSeconds) { break }
        }
        $completed = $process.HasExited
        if ($completed) { $process.WaitForExit() }
        $phaseExit = if ($completed) { $process.ExitCode } else { $null }
        $status = if (!$completed) { 'timeout' } elseif ($null -eq $phaseExit) {
            'runner-error'
        } elseif ($phaseExit -eq 0) {
            'passed'
        } else { 'failed' }
        $results.Add([pscustomobject]@{
            phase = $Name
            status = $status
            exit_code = $phaseExit
            elapsed_seconds = [math]::Round($timer.Elapsed.TotalSeconds, 3)
            tool = $Tool
            arguments = $ToolArguments
        })
        Save-Outcome
        Write-Host "$Name : $status (exit=$phaseExit)"
        return ($status -eq 'passed')
    } finally {
        # Only terminate this script's own still-running child, including on Ctrl+C.
        if ($null -ne $process) {
            if ($started -and !$process.HasExited) {
                $process.Kill()
                $process.WaitForExit()
            }
            if ($null -ne $stdoutTask) {
                [IO.File]::WriteAllText($stdout, $stdoutTask.GetAwaiter().GetResult())
            }
            if ($null -ne $stderrTask) {
                [IO.File]::WriteAllText($stderr, $stderrTask.GetAwaiter().GetResult())
            }
            $process.Dispose()
        }
    }
}

Write-Output "Diagnosis outputs: $scratch"
try {
    # Include hashes provide provenance even when the entry source is unchanged.
    $inputs = @(Get-ChildItem -LiteralPath $shaderRoot -Recurse -File | Where-Object {
        $_.Extension -in @('.glsl', '.h', '.rchit')
    } | ForEach-Object { $_.FullName }) + @($compiler, $optimizer, $validator)
    if ($ReferenceModule) { $inputs += $ReferenceModule }
    if ($PreserveFunctions) { $inputs += [IO.Path]::GetFullPath($boundaryTool) }
    $inputs | ForEach-Object { Get-FileHash -LiteralPath $_ -Algorithm SHA256 } |
        Select-Object Path, Hash | ConvertTo-Json |
        Set-Content -LiteralPath (Join-Path $scratch 'input-hashes.json') -Encoding UTF8
    Save-Outcome
    [pscustomobject]@{
        reference_module = $ReferenceModule
        preserve_functions = [bool]$PreserveFunctions
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $scratch 'experiment.json')
    $null = Invoke-DiagnosticPhase 'versions' $compiler @('--version')
    $reference = Join-Path $scratch 'reference.spv'
    if ($ReferenceModule) {
        # A fresh copy prevents tools from writing into the original capture.
        Copy-Item -LiteralPath $ReferenceModule -Destination $reference
    } else {
        $referenceResult = Invoke-DiagnosticPhase 'reference' $compiler @(
            $source, '--target-env=vulkan1.3', '--target-spv=spv1.4', '-O0', '-o', $reference)
        if (!$referenceResult) { throw 'Reference compile did not pass.' }
    }
    $validation = Invoke-DiagnosticPhase 'reference-validation' $validator @(
        '--target-env', 'vulkan1.3', '--scalar-block-layout', $reference)
    if (!$validation) { throw 'Reference validation did not pass.' }
    if ($PreserveFunctions) {
        $protectedReference = Join-Path $scratch 'function-boundaries.spv'
        if (!(Invoke-DiagnosticPhase 'function-boundaries' $pythonPath @(
                    [IO.Path]::GetFullPath($boundaryTool), $reference,
                    '--output', $protectedReference))) {
            throw 'Function boundary transformation did not pass.'
        }
        if (!(Invoke-DiagnosticPhase 'function-boundaries-validation' $validator @(
                    '--target-env', 'vulkan1.3', '--scalar-block-layout', $protectedReference))) {
            throw 'Function boundary validation did not pass.'
        }
        $reference = $protectedReference
    }
    foreach ($mode in @('inline', 'full')) {
        $flag = if ($mode -eq 'inline') { '--inline-entry-points-exhaustive' } else { '-O' }
        $target = Join-Path $scratch ($mode + '.spv')
        $phase = Invoke-DiagnosticPhase $mode $optimizer @(
            '--target-env=vulkan1.3', '--scalar-block-layout', $flag,
            $reference, '-o', $target)
        if ($phase) {
            $valid = Invoke-DiagnosticPhase ($mode + '-validation') $validator @(
                '--target-env', 'vulkan1.3', '--scalar-block-layout', $target)
            if ($valid -and $PreserveFunctions) {
                $null = Invoke-DiagnosticPhase ($mode + '-inspection') $pythonPath @(
                    (Join-Path $PSScriptRoot 'inspect_volume_spirv.py'), $target)
            }
        }
    }
} finally {
    Save-Outcome
    Write-Output "Results: $summaryPath"
}
if (@($results | Where-Object { $_.status -ne 'passed' }).Count -gt 0) { exit 1 }
