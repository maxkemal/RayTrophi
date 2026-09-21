$ErrorActionPreference='Stop'
Import-Module "$PSScriptRoot/RtIpc.psm1" -Force
try {
    $materials=@(foreach($m in (Invoke-RtIpc material.list)) {
        if($m.type -ne 'principled' -or $m.name -notlike 'foliage_*') { continue }
        $textures=@(Invoke-RtIpc material.textures @{material_name=$m.name})
        if($textures.slot -notcontains 'opacity') { continue }
        [pscustomobject]@{name=$m.name;cutout=(Invoke-RtIpc material.get_param @{material_name=$m.name;param='alpha_cutout'});transmission=(Invoke-RtIpc material.get_param @{material_name=$m.name;param='transmission'});textures=$textures}
    })
    $result=[pscustomobject]@{materials=$materials;quality=(Invoke-RtIpc viewport.quality);shadow=(Invoke-RtIpc viewport.rt_shadow)}
    $result | ConvertTo-Json -Depth 12 | Tee-Object -FilePath "$PSScriptRoot/../../tmp/foliage-coverage-inspection.json"
} finally { Disconnect-RtIpc }
