$ErrorActionPreference = 'Stop'
Import-Module "$PSScriptRoot/RtIpc.psm1" -Force
try {
    $items = @(foreach ($m in (Invoke-RtIpc material.list)) {
        if ($m.type -ne 'principled') { throw "Unsupported material: $($m.name)" }
        [pscustomobject]@{
            name = $m.name
            transmission = Invoke-RtIpc material.get_param @{material_name=$m.name;param='transmission'}
            opacity = Invoke-RtIpc material.get_param @{material_name=$m.name;param='opacity'}
            textures = @(Invoke-RtIpc material.textures @{material_name=$m.name})
        }
    })
    $items | ConvertTo-Json -Depth 12 | Set-Content "$PSScriptRoot/../../tmp/raster-material-snapshot.json" -Encoding UTF8
    $items | Where-Object { $_.transmission -gt 0 -or $_.opacity -lt 1 -or ($_.textures.slot -contains 'opacity') } | ConvertTo-Json -Depth 8
} finally { Disconnect-RtIpc }
