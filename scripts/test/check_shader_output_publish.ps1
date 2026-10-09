# File-I/O test only. Load one function from the AST, never the compiler driver.
$ErrorActionPreference = 'Stop'
$walk = [IO.DirectoryInfo]$PSScriptRoot
while ($null -ne $walk -and !(Test-Path -LiteralPath (
    Join-Path $walk.FullName 'RayTrophiStudio\compile_shaders.ps1'))) {
    $walk = $walk.Parent
}
if ($null -eq $walk) { throw 'Repository shader driver not found' }
$driverPath = Join-Path $walk.FullName 'RayTrophiStudio\compile_shaders.ps1'
$parseTokens = $null
$parseErrors = $null
$driverAst = [Management.Automation.Language.Parser]::ParseFile(
    $driverPath, [ref]$parseTokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw ($parseErrors | Out-String) }
$definition = $driverAst.FindAll({ param($node)
    $node -is [Management.Automation.Language.FunctionDefinitionAst] -and
    $node.Name -eq 'Publish-Spirv'
}, $true)
if ($definition.Count -ne 1) { throw 'Publication function missing or ambiguous' }
Invoke-Expression $definition[0].Extent.Text

$probeDirectory = Join-Path $PSScriptRoot ('.shader-publish-test-' + [Guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($probeDirectory) | Out-Null
$target = Join-Path $probeDirectory 'target.spv'
$temporary = Join-Path $probeDirectory 'output.tmp'
$lockHandle = $null
try {
    [byte[]]$expected = New-Object byte[] 20
    [BitConverter]::GetBytes([uint32]0x07230203).CopyTo($expected, 0)
    [IO.File]::WriteAllBytes($target, [byte[]]@(1, 2, 3, 4))
    [IO.File]::WriteAllBytes($temporary, $expected)
    Publish-Spirv $temporary $target
    if ([Convert]::ToBase64String([IO.File]::ReadAllBytes($target)) -ne
        [Convert]::ToBase64String($expected)) { throw 'Valid replacement failed' }

    [IO.File]::WriteAllBytes($temporary, [byte[]]@(1, 2))
    $rejected = $false
    try { Publish-Spirv $temporary $target } catch { $rejected = $true }
    if (!$rejected -or [Convert]::ToBase64String([IO.File]::ReadAllBytes($target)) -ne
        [Convert]::ToBase64String($expected)) { throw 'Invalid output damaged target' }

    [byte[]]$replacement = $expected.Clone()
    $replacement[19] = 42
    [IO.File]::WriteAllBytes($temporary, $replacement)
    $lockHandle = [IO.File]::Open($target, [IO.FileMode]::Open,
        [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    $rejected = $false
    try { Publish-Spirv $temporary $target } catch { $rejected = $true }
    $lockHandle.Dispose()
    $lockHandle = $null
    if (!$rejected -or [Convert]::ToBase64String([IO.File]::ReadAllBytes($target)) -ne
        [Convert]::ToBase64String($expected)) { throw 'Locked target was not preserved' }
    Publish-Spirv $temporary $target
    if ([Convert]::ToBase64String([IO.File]::ReadAllBytes($target)) -ne
        [Convert]::ToBase64String($replacement)) { throw 'Unlocked retry failed' }
    Write-Output 'PASS publish, invalid-output preservation, locked-target preservation and unlocked retry; no compiler invoked'
} finally {
    if ($null -ne $lockHandle) { $lockHandle.Dispose() }
    foreach ($probeFile in Get-ChildItem -LiteralPath $probeDirectory -File) {
        Remove-Item -LiteralPath $probeFile.FullName -Force
    }
    Remove-Item -LiteralPath $probeDirectory
}
