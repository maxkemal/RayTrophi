# A consistency stamp for the three mutually dependent split-stage modules.
# FNV-1a over uint32 SPIR-V words is an accidental-mix check, not a security hash.
function Write-VolumeSplitManifest {
    param([string]$Directory)
    $rows = @('volume-split-v1')
    foreach ($name in @('raygen.spv', 'volume_fog_closesthit.spv', 'volume_surface_closesthit.spv')) {
        $bytes = [IO.File]::ReadAllBytes((Join-Path $Directory $name))
        if ($bytes.Length -lt 20 -or $bytes.Length % 4 -ne 0 -or
            [BitConverter]::ToUInt32($bytes, 0) -ne 0x07230203) {
            throw "Invalid split-stage input: $name"
        }
        [uint32]$hash = 2166136261
        for ($offset = 0; $offset -lt $bytes.Length; $offset += 4) {
            $hash = $hash -bxor [BitConverter]::ToUInt32($bytes, $offset)
            $hash = [uint32](([uint64]$hash * 16777619L) -band 0xffffffffL)
        }
        $rows += ('{0} {1} {2:X8}' -f $name, $bytes.Length, $hash)
    }
    $target = Join-Path $Directory 'volume_split_stage_set.txt'
    $temporary = Join-Path $Directory ('.volume-split-' + [guid]::NewGuid().ToString('N') + '.tmp')
    $backup = $temporary + '.backup'
    try {
        [IO.File]::WriteAllLines($temporary, $rows, [Text.Encoding]::ASCII)
        if ([IO.File]::Exists($target)) {
            [IO.File]::Replace($temporary, $target, $backup)
        } else {
            [IO.File]::Move($temporary, $target)
        }
    } finally {
        if ([IO.File]::Exists($temporary)) { [IO.File]::Delete($temporary) }
        if ([IO.File]::Exists($backup)) { [IO.File]::Delete($backup) }
    }
}
