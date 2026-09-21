param(
    [Parameter(Mandatory=$true)][string[]]$MaterialNames,
    [ValidateSet(0,1)][int]$Enabled = 1
)
$ErrorActionPreference = 'Stop'
Import-Module "$PSScriptRoot/RtIpc.psm1" -Force
$snapshot = @()
$touched = @()
try {
    # Read ALL original values first; an old build or missing name changes nothing.
    foreach ($name in $MaterialNames) {
        $snapshot += [pscustomobject]@{name=$name;value=(Invoke-RtIpc material.get_param @{material_name=$name;param='alpha_cutout'})}
    }
    foreach ($entry in $snapshot) {
        $touched += $entry
        Invoke-RtIpc material.set_param @{material_name=$entry.name;param='alpha_cutout';value=$Enabled} | Out-Null
        $actual = Invoke-RtIpc material.get_param @{material_name=$entry.name;param='alpha_cutout'}
        if ($actual -ne $Enabled) { throw "Cutout write/readback mismatch: $($entry.name)" }
        Write-Output "$($entry.name): alpha_cutout=$actual (was $($entry.value))"
    }
} catch {
    $failure = $_
    foreach ($entry in $touched) {
        try { Invoke-RtIpc material.set_param @{material_name=$entry.name;param='alpha_cutout';value=$entry.value} | Out-Null }
        catch { Write-Warning "Restore failed for $($entry.name): $_" }
    }
    throw $failure
} finally { Disconnect-RtIpc }
