# Exercise process handling with PowerShell children only; no shader tools/builds.
$ErrorActionPreference = 'Stop'
$runner = Join-Path $PSScriptRoot 'diagnose_volume_compiler.ps1'
$tokens = $null
$parseErrors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile(
    $runner, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -ne 0) { throw ($parseErrors | Out-String) }
$definition = @($ast.FindAll({
    param($node)
    $node -is [Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq 'Invoke-DiagnosticPhase'
}, $true))
if ($definition.Count -ne 1) { throw 'Expected one diagnostic phase function.' }
Invoke-Expression $definition[0].Extent.Text
function Save-Outcome { }
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$parent = Join-Path $repoRoot 'docs\dev\.captures'
$scratch = Join-Path $parent ('volume-runner-test-' + [guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($scratch) | Out-Null
$shaderRoot = $repoRoot
$results = [Collections.Generic.List[object]]::new()
$TimeoutSeconds = 10
$helper = Join-Path $PSHOME 'powershell.exe'
try {
    foreach ($case in @(
        @{ Name = 'success'; Code = 0; Status = 'passed' },
        @{ Name = 'failure'; Code = 7; Status = 'failed' }
    )) {
        $command = "[Console]::Out.WriteLine('stdout-marker'); " +
            "[Console]::Error.WriteLine('stderr-marker'); exit $($case.Code)"
        $passed = Invoke-DiagnosticPhase $case.Name $helper @('-NoProfile', '-Command', $command)
        $last = $results[$results.Count - 1]
        if ($last.exit_code -ne $case.Code -or $last.status -ne $case.Status -or
            $passed -ne ($case.Code -eq 0)) { throw "Wrong result: $($case.Name)" }
        foreach ($stream in @('stdout', 'stderr')) {
            $log = Join-Path $scratch ($case.Name + '.' + $stream + '.log')
            if (!(Get-Content -LiteralPath $log -Raw).Contains($stream + '-marker')) {
                throw "Missing $stream output"
            }
        }
    }
    $TimeoutSeconds = 1
    $passed = Invoke-DiagnosticPhase 'timeout' $helper @(
        '-NoProfile', '-Command', 'Start-Sleep -Seconds 30')
    $last = $results[$results.Count - 1]
    if ($passed -or $last.status -ne 'timeout' -or $null -ne $last.exit_code) {
        throw 'Timeout was confused with process failure/success.'
    }
    Write-Output 'PASS: success (0), failure (7), stream capture and timeout; no shader tools invoked.'
} finally {
    $resolved = [IO.Path]::GetFullPath($scratch)
    if (!$resolved.StartsWith($parent.TrimEnd('\') + '\',
            [StringComparison]::OrdinalIgnoreCase) -or
        ![IO.Path]::GetFileName($resolved).StartsWith('volume-runner-test-')) {
        throw "Unsafe test cleanup path: $resolved"
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
