param([Parameter(Mandatory=$true)][string]$MaterialName)
$ErrorActionPreference = 'Stop'
Import-Module "$PSScriptRoot/../ipc/RtIpc.psm1" -Force
$old = Invoke-RtIpc material.get_param @{material_name=$MaterialName;param='alpha_cutout'}
try {
    foreach ($value in @(1,0,1)) {
        Invoke-RtIpc material.set_param @{material_name=$MaterialName;param='alpha_cutout';value=$value} | Out-Null
        if ((Invoke-RtIpc material.get_param @{material_name=$MaterialName;param='alpha_cutout'}) -ne $value) { throw 'roundtrip failed' }
    }
    foreach ($invalid in @(-1,0.5,2)) {
        $rejected = $false
        try { Invoke-RtIpc material.set_param @{material_name=$MaterialName;param='alpha_cutout';value=$invalid} | Out-Null }
        catch { $rejected = $true }
        if (!$rejected) { throw "Accepted invalid value: $invalid" }
        if ((Invoke-RtIpc material.get_param @{material_name=$MaterialName;param='alpha_cutout'}) -ne 1) { throw 'rejected write mutated state' }
    }
} finally {
    try { Invoke-RtIpc material.set_param @{material_name=$MaterialName;param='alpha_cutout';value=$old} | Out-Null }
    finally { Disconnect-RtIpc }
}
Write-Output 'PASS: cutout IPC roundtrip, validation, restoration'
