$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$vcpkgExe = Join-Path $repoRoot 'vcpkg\vcpkg.exe'

if (-not (Test-Path -LiteralPath $vcpkgExe)) {
    throw "Repository-local vcpkg executable was not found: $vcpkgExe"
}

& $vcpkgExe install pybind11:x64-windows
if ($LASTEXITCODE -ne 0) {
    throw "vcpkg failed with exit code $LASTEXITCODE"
}

# numpy backs rt.mesh.positions/normals/uvs (Faz 3a) — installed into vcpkg's
# embeddable python3 tool folder so the post-build "python/" copy step
# (RayTrophiStudio.vcxproj) picks it up alongside the stdlib automatically.
$pythonExe = Join-Path $repoRoot 'vcpkg\installed\x64-windows\tools\python3\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "vcpkg python3 tool was not found: $pythonExe"
}

& $pythonExe -m ensurepip --upgrade
if ($LASTEXITCODE -ne 0) {
    throw "ensurepip failed with exit code $LASTEXITCODE"
}

& $pythonExe -m pip install --upgrade numpy
if ($LASTEXITCODE -ne 0) {
    throw "pip install numpy failed with exit code $LASTEXITCODE"
}

Write-Host 'RayTrophi Python scripting dependencies are ready.' -ForegroundColor Green
