$ErrorActionPreference='Stop'
Import-Module "$PSScriptRoot/../ipc/RtIpc.psm1" -Force
$old=Invoke-RtIpc viewport.automatic_cutout
try {
    foreach($value in @($false,$true)) {
        Invoke-RtIpc viewport.set_automatic_cutout @{enabled=$value} | Out-Null
        if((Invoke-RtIpc viewport.automatic_cutout).enabled -ne $value) { throw 'Cutout readback failed' }
    }
    foreach($invalid in @(0,1,'true',$null)) {
        $rejected=$false
        try { Invoke-RtIpc viewport.set_automatic_cutout @{enabled=$invalid} | Out-Null }
        catch { $rejected=$true }
        if(!$rejected) { throw 'Non-boolean accepted' }
        if(!(Invoke-RtIpc viewport.automatic_cutout).enabled) { throw 'Rejected write changed state' }
    }
} finally {
    try { Invoke-RtIpc viewport.set_automatic_cutout @{enabled=[bool]$old.enabled} | Out-Null }
    finally { Disconnect-RtIpc }
}
'PASS: viewport cutout IPC roundtrip, rejection and restoration'
