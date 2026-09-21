$ErrorActionPreference = 'Stop'
Import-Module "$PSScriptRoot/RtIpc.psm1" -Force
try {
    $saved = Get-Content "$PSScriptRoot/../../tmp/raster-material-snapshot.json" -Raw | ConvertFrom-Json
    $errors = @()
    $checked = 0
    foreach ($m in $saved) {
        foreach ($param in @('transmission','opacity')) {
            if (($param -eq 'transmission' -and $m.transmission -gt 0) -or
                ($param -eq 'opacity' -and $m.textures.slot -contains 'opacity')) {
                $value = Invoke-RtIpc material.get_param @{material_name=$m.name;param=$param}
                $checked++
                if ([math]::Abs([double]$value - [double]$m.$param) -gt 0.000001) {
                    $errors += "$($m.name): $param expected $($m.$param), got $value"
                }
            }
        }
    }
    $state = [pscustomobject]@{
        checked=$checked; mismatches=$errors
        world=Invoke-RtIpc world.get
        shading=Invoke-RtIpc viewport.shading
        lighting=Invoke-RtIpc viewport.preview_lighting
    }
    $state | ConvertTo-Json -Depth 15 | Set-Content "$PSScriptRoot/../../tmp/raster-probe-restored.json" -Encoding UTF8
    $state | ConvertTo-Json -Depth 15
    if ($errors.Count) { throw ($errors -join "`n") }
} finally { Disconnect-RtIpc }
