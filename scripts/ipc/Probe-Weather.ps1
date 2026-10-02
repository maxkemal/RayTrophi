<#
.SYNOPSIS
    Atmosfer Faz 4a kabul testi: iklimden turetilmis hava (docs/dev/ATMOSPHERE_WEATHER.md §2).
    Kisa ve sayisal: 1) kuru hava -> bulut yok, 2) nemli+kararsiz -> Cb + yagis,
    3) bulut tabani = LCL (125 m x ciy noktasi farki) ve katman 0 onu izler.
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
    $null = Invoke-RtIpc world.set_climate @{ surface_temperature_k = 293.15; surface_relative_humidity = 0.3; instability = 0.2 }
    $w = (Invoke-RtIpc world.get_weather).derived
    Check ($w.cloud_coverage -lt 0.01 -and $w.precipitation_mm_h -eq 0) ("dry air: coverage {0:N3}, precip {1:N2}" -f $w.cloud_coverage, $w.precipitation_mm_h)

    $null = Invoke-RtIpc world.set_climate @{ surface_relative_humidity = 0.92; instability = 0.95 }
    $w = (Invoke-RtIpc world.get_weather).derived
    Check ($w.cloud_type -gt 0.8 -and $w.precipitation_mm_h -gt 1.0) ("moist+unstable: type {0:N2}, precip {1:N1} mm/h" -f $w.cloud_type, $w.precipitation_mm_h)

    $alt = (Invoke-RtIpc world.get_atmosphere).altitude
    $lcl = 125.0 * (293.15 - $w.dew_point_k)
    $l0 = (Invoke-RtIpc world.get_clouds).layers[0]
    Check ([math]::Abs($w.cloud_base_m - ($alt + [math]::Max(150, $lcl))) -lt 1.0 -and [math]::Abs($l0.base_altitude_m - $w.cloud_base_m) -lt 1.0) `
          ("base {0:N0} m = LCL {1:N0} m + altitude; layer 0 follows ({2:N0} m)" -f $w.cloud_base_m, $lcl, $l0.base_altitude_m)
}
finally {
    $null = Invoke-RtIpc world.set_climate @{ surface_temperature_k = $c0.surface_temperature_k; surface_relative_humidity = $c0.surface_relative_humidity; instability = $c0.instability }
    $null = Invoke-RtIpc world.set_clouds @{ derive_from_climate = $g0.derive_from_climate; layers = $g0.layers }
}
if ($fails -eq 0) { Write-Host "ALL PASS" -ForegroundColor Green; exit 0 }
Write-Host "$fails FAILED" -ForegroundColor Red; exit 1
