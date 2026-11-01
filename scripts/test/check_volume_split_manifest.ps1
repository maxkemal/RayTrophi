# Consistency-stamp tests using mock SPIR-V headers; never invokes shader tools.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\..\RayTrophiStudio\volume_split_manifest.ps1')
$parent = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\docs\dev\.captures'))
$scratch = Join-Path $parent ('volume-stamp-test-' + [guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($scratch) | Out-Null
try {
    foreach ($name in @('raygen.spv', 'volume_fog_closesthit.spv', 'volume_surface_closesthit.spv')) {
        $bytes = New-Object byte[] 20
        [BitConverter]::GetBytes([uint32]0x07230203).CopyTo($bytes, 0)
        [IO.File]::WriteAllBytes((Join-Path $scratch $name), $bytes)
    }
    Write-VolumeSplitManifest $scratch
    $stamp = Join-Path $scratch 'volume_split_stage_set.txt'
    $initial = [IO.File]::ReadAllText($stamp)
    Write-VolumeSplitManifest $scratch
    if ([IO.File]::ReadAllText($stamp) -ne $initial) { throw 'Non-deterministic stamp' }
    $rows = [IO.File]::ReadAllLines($stamp)
    if ($rows.Count -ne 4 -or $rows[0] -ne 'volume-split-v1') { throw 'Wrong stamp format' }
    # Known independently computed FNV-1a word-vector value is checked below.
    foreach ($row in $rows[1..3]) {
        if ($row -notmatch '^\S+ 20 45F09DD2$') { throw 'Wrong module row/checksum' }
    }
    [IO.File]::WriteAllBytes((Join-Path $scratch 'volume_surface_closesthit.spv'), [byte[]](1, 2, 3))
    $rejected = $false
    try { Write-VolumeSplitManifest $scratch } catch { $rejected = $true }
    if (!$rejected -or [IO.File]::ReadAllText($stamp) -ne $initial) {
        throw 'Bad input replaced a valid stamp'
    }
    if (@(Get-ChildItem -LiteralPath $scratch -Filter '.volume-split-*').Count) {
        throw 'Manifest temporary leaked'
    }
    Write-Output 'PASS: stamp creation, replacement, determinism and malformed-input preservation; no compiler invoked.'
} finally {
    $resolved = [IO.Path]::GetFullPath($scratch)
    if (!$resolved.StartsWith($parent.TrimEnd('\') + '\',
            [StringComparison]::OrdinalIgnoreCase) -or
        ![IO.Path]::GetFileName($resolved).StartsWith('volume-stamp-test-')) {
        throw "Unsafe cleanup path: $resolved"
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
