# User-run checkpoint diagnosis; never deploys modules or compiles source.
param(
    [Parameter(Mandatory = $true)]
    [string]$ReferenceModule,
    [switch]$ValidateOnly,
    [ValidateRange(1, 44)]
    [int]$StartPass = 1,
    [switch]$CompactBeforeSsa,
    [ValidateRange(10, 7200)]
    [int]$TimeoutSeconds = 600
)
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$source = (Resolve-Path -LiteralPath $ReferenceModule).Path
$shaderRoot = Join-Path $repoRoot 'RayTrophiStudio\source\shaders'
$sdkBin = Join-Path $env:VULKAN_SDK 'Bin'
$optimizer = Join-Path $sdkBin 'spirv-opt.exe'
$validator = Join-Path $sdkBin 'spirv-val.exe'
foreach ($tool in @($optimizer, $validator)) {
    if (!(Test-Path -LiteralPath $tool -PathType Leaf)) { throw "Missing tool: $tool" }
}
# Exact -O order advertised by the installed SDK inspected on 2026-10-30.
# Checkpoint serialization can change ID allocation/pass interaction; a failure
# localizes this experiment, and success does not clear the single-process -O.
$passes = @'
wrap-opkill
eliminate-dead-branches
merge-return
inline-entry-points-exhaustive
eliminate-dead-functions
eliminate-dead-code-aggressive
private-to-local
eliminate-local-single-block
eliminate-local-single-store
eliminate-dead-code-aggressive
scalar-replacement=100
convert-local-access-chains
eliminate-local-single-block
eliminate-local-single-store
eliminate-dead-code-aggressive
ssa-rewrite
eliminate-dead-code-aggressive
ccp
eliminate-dead-code-aggressive
loop-unroll
eliminate-dead-branches
redundancy-elimination
combine-access-chains
simplify-instructions
scalar-replacement=100
convert-local-access-chains
eliminate-local-single-block
eliminate-local-single-store
eliminate-dead-code-aggressive
ssa-rewrite
eliminate-dead-code-aggressive
vector-dce
eliminate-dead-inserts
eliminate-dead-branches
simplify-instructions
if-conversion
copy-propagate-arrays
reduce-load-size
eliminate-dead-code-aggressive
merge-blocks
redundancy-elimination
eliminate-dead-branches
merge-blocks
simplify-instructions
'@ -split '\r?\n'

$tokens = $null
$parseErrors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile(
    (Join-Path $PSScriptRoot 'diagnose_volume_compiler.ps1'),
    [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -gt 0) { throw ($parseErrors | Out-String) }
foreach ($name in @('Save-Outcome', 'Invoke-DiagnosticPhase')) {
    $definitions = @($ast.FindAll({
        param($node)
        $node -is [Management.Automation.Language.FunctionDefinitionAst]
    }, $true) | Where-Object { $_.Name -eq $name })
    if ($definitions.Count -ne 1) { throw "Missing runner function: $name" }
    Invoke-Expression $definitions[0].Extent.Text
}
if ($ValidateOnly) {
    Write-Output "PASS: $($passes.Count) ordered passes; start=$StartPass; compact-before-ssa=$CompactBeforeSsa; no shader tools invoked."
    exit 0
}
$scratch = Join-Path $repoRoot ('docs\dev\.captures\volume-passes-' +
    [guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($scratch) | Out-Null
$summaryPath = Join-Path $scratch 'outcome.json'
$results = [Collections.Generic.List[object]]::new()
Write-Output "Pass diagnosis outputs: $scratch"
try {
    Get-FileHash -LiteralPath $source, $optimizer, $validator -Algorithm SHA256 |
        Select-Object Path, Hash | ConvertTo-Json |
        Set-Content -LiteralPath (Join-Path $scratch 'input-hashes.json') -Encoding UTF8
    $passes | Set-Content -LiteralPath (Join-Path $scratch 'pass-order.txt')
    [pscustomobject]@{
        start_pass = $StartPass
        compact_before_ssa = [bool]$CompactBeforeSsa
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $scratch 'experiment.json')
    if (!(Invoke-DiagnosticPhase 'input-validation' $validator @(
                '--target-env', 'vulkan1.3', '--scalar-block-layout', $source))) {
        throw 'Input validation failed.'
    }
    $previous = $source
    for ($index = $StartPass - 1; $index -lt $passes.Count; $index++) {
        $pass = $passes[$index]
        $name = '{0:D2}-{1}' -f ($index + 1), $pass.Replace('=', '-')
        if ($CompactBeforeSsa -and $pass -eq 'ssa-rewrite') {
            $compact = Join-Path $scratch ($name + '-compact-input.spv')
            if (!(Invoke-DiagnosticPhase ($name + '-compact') $optimizer @(
                        '--target-env=vulkan1.3', '--scalar-block-layout',
                        '--compact-ids', $previous, '-o', $compact))) {
                throw "ID compaction failed before $name"
            }
            if (!(Invoke-DiagnosticPhase ($name + '-compact-validation') $validator @(
                        '--target-env', 'vulkan1.3', '--scalar-block-layout', $compact))) {
                throw "Compacted input invalid before $name"
            }
            $compactHeader = [IO.File]::ReadAllBytes($compact)
            [pscustomobject]@{
                pass = $name + '-compact-input'
                bytes = $compactHeader.Length
                id_bound = [BitConverter]::ToUInt32($compactHeader, 12)
            } | ConvertTo-Json -Compress |
                Add-Content -LiteralPath (Join-Path $scratch 'module-sizes.jsonl')
            $previous = $compact
        }
        $target = Join-Path $scratch ($name + '.spv')
        if (!(Invoke-DiagnosticPhase $name $optimizer @(
                    '--target-env=vulkan1.3', '--scalar-block-layout',
                    ('--' + $pass), $previous, '-o', $target))) {
            throw "Stopped at pass $name; previous valid checkpoint: $previous"
        }
        if (!(Invoke-DiagnosticPhase ($name + '-validation') $validator @(
                    '--target-env', 'vulkan1.3', '--scalar-block-layout', $target))) {
            throw "Invalid output at pass $name; previous valid checkpoint: $previous"
        }
        $header = [IO.File]::ReadAllBytes($target)
        [pscustomobject]@{
            pass = $name
            bytes = $header.Length
            id_bound = [BitConverter]::ToUInt32($header, 12)
        } | ConvertTo-Json -Compress |
            Add-Content -LiteralPath (Join-Path $scratch 'module-sizes.jsonl')
        $previous = $target
    }
} finally {
    Save-Outcome
    Write-Output "Results: $summaryPath"
}
