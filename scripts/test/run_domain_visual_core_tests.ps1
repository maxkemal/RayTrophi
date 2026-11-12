# User-run standalone mathematical tests. Does not build or launch Studio.
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$taskRepo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$taskCompiler = Get-Command cl.exe -ErrorAction Stop
$taskOutput = Join-Path ([System.IO.Path]::GetTempPath()) (
    'raytrophi_domain_visual_' + [System.Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $taskOutput | Out-Null

$taskTests = @('domain_field_sampling_test', 'domain_visual_runtime_test',
    'domain_visual_settings_test')
foreach ($taskTest in $taskTests) {
    $taskSource = Join-Path $taskRepo "tests/domain_fields/$taskTest.cpp"
    $taskExecutable = Join-Path $taskOutput "$taskTest.exe"
    $taskObject = Join-Path $taskOutput "$taskTest.obj"
    & $taskCompiler.Source /nologo /std:c++17 /EHsc /W4 /fp:precise `
        "/I$(Join-Path $taskRepo 'RayTrophiStudio/source/include')" `
        $taskSource "/Fe:$taskExecutable" "/Fo:$taskObject"
    if ($LASTEXITCODE -ne 0) {
        throw "Compilation failed: $taskTest. Outputs: $taskOutput"
    }
    & $taskExecutable
    if ($LASTEXITCODE -ne 0) {
        throw "Test failed: $taskTest. Outputs: $taskOutput"
    }
}
Write-Output "Domain visual core tests PASS. Outputs: $taskOutput"
