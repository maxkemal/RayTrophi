$ErrorActionPreference='Stop'
Import-Module "$PSScriptRoot/RtIpc.psm1" -Force
try {
    Write-Output ('PROJECT=' + ((Invoke-RtIpc project.path) | ConvertTo-Json -Compress))
} finally { Disconnect-RtIpc }
$snapshot=Get-Content "$PSScriptRoot/../../tmp/foliage-coverage-inspection.json" -Raw | ConvertFrom-Json
$names=@($snapshot.materials | Where-Object { $_.transmission -eq 0 -and $_.textures.slot -notcontains 'transmission' } | ForEach-Object name)
if(!$names.Count) { throw 'No inspected coverage candidates' }
& "$PSScriptRoot/Set-FoliageCutout.ps1" -MaterialNames $names -Enabled 1
