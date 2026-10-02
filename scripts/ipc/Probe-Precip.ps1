<#
.SYNOPSIS
    Atmosfer Faz 4b-1 kabul testi: yagis perdeleri (docs/dev/ATMOSPHERE_WEATHER.md §3.1).
    1) nemli+kararsiz iklim -> clouds.precipitation.rate > 0 (turetilmis)
    2) zeminde bir izgarada yagis: bir kisim > 0 (cekirdek alti), bir kisim 0 (acik)
    3) bulut tabaninin ustunde yagis = 0
    Baslangic iklimi ve bulutlari geri yazilir.
#>
Import-Module "$PSScriptRoot\RtIpc.psm1" -Force
$fails = 0
function Check([bool]$ok, [string]$what) {
    if ($ok) { Write-Host "  OK   $what" -ForegroundColor Green } else { Write-Host "  FAIL $what" -ForegroundColor Red; $script:fails++ }
}
$c0 = Invoke-RtIpc world.get_climate
$g0 = Invoke-RtIpc world.get_clouds
try {
    $null = Invoke-RtIpc world.set_clouds @{ derive_from_climate = $true }
    $null = Invoke-RtIpc world.set_climate @{ surface_temperature_k = 295.15; surface_relative_humidity = 0.93; instability = 0.9 }
    $g = Invoke-RtIpc world.get_clouds
    Check ($g.precipitation.rate_mm_h -gt 1.0) ("derived rate {0:N1} mm/h, snow {1}, reach {2:N2}" -f $g.precipitation.rate_mm_h, $g.precipitation.snow, $g.precipitation.ground_fraction)

    $alt = (Invoke-RtIpc world.get_atmosphere).altitude
    $yLow = 0.3 * ($g.layers[0].base_altitude_m - $alt)   # under the base (a derived base can be < 200 m)
    $pts = @(); foreach ($i in -10..10) { foreach ($k in -10..10) { $pts += ,@(($i * 2000.0), $yLow, ($k * 2000.0)) } }
    $v = (Invoke-RtIpc world.sample_clouds @{ mode = 'precipitation'; points = $pts }).values
    $wet = @($v | Where-Object { $_ -gt 0.01 }).Count
    Check ($wet -gt 0 -and $wet -lt $v.Count) ("ground grid: {0}/{1} points wet, max {2:N1} mm/h" -f $wet, $v.Count, ($v | Measure-Object -Maximum).Maximum)

    $yAbove = $g.layers[0].base_altitude_m - $alt + 100.0
    $pa = @(); foreach ($p in $pts) { $pa += ,@($p[0], $yAbove, $p[2]) }
    $va = (Invoke-RtIpc world.sample_clouds @{ mode = 'precipitation'; points = $pa }).values
    Check (($va | Measure-Object -Maximum).Maximum -eq 0) "above the base: no precipitation"
}
finally {
    $null = Invoke-RtIpc world.set_climate @{ surface_temperature_k = $c0.surface_temperature_k; surface_relative_humidity = $c0.surface_relative_humidity; instability = $c0.instability }
    $null = Invoke-RtIpc world.set_clouds @{ derive_from_climate = $g0.derive_from_climate; layers = $g0.layers; precipitation = $g0.precipitation }
}
if ($fails -eq 0) { Write-Host "ALL PASS" -ForegroundColor Green; exit 0 }
Write-Host "$fails FAILED" -ForegroundColor Red; exit 1
