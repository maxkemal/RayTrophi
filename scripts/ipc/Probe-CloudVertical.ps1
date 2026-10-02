<#
.SYNOPSIS
    Atmosfer Faz 3e kabul testi: bulut dikey yapisi (docs/dev/ATMOSPHERE_CLOUDS.md).
    'scattered' preset (cumulus), 12x12 sutun (2 km aralik), her sutunda dikey tarama (base_density).
    1) tepeler DAGILIR (max/medyan > 1.4) ve en yuksek katmanin %70'ine ulasir
    2) ruzgar kesmesi: ruzgar +X 15 m/s iken ust yarinin agirlik merkezi alt yaridan +X yonunde
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
    $null = Invoke-RtIpc world.apply_cloud_preset @{ preset = 'scattered' }
    $null = Invoke-RtIpc world.set_climate @{ wind_speed_mps = 15.0; wind_direction = @(1.0, 0.0, 0.0) }
    $g = Invoke-RtIpc world.get_clouds
    $alt = (Invoke-RtIpc world.get_atmosphere).altitude
    $L = $g.layers[0]; $y0 = $L.base_altitude_m - $alt; $nz = 24
    $pts = @()
    foreach ($i in 0..11) { foreach ($k in 0..11) { foreach ($z in 0..($nz - 1)) {
        $pts += ,@(($i * 2000.0 - 11000.0), ($y0 + ($z + 0.5) / $nz * $L.thickness_m), ($k * 2000.0 - 11000.0)) } } }
    $v = (Invoke-RtIpc world.sample_clouds @{ mode = 'base_density'; points = $pts }).values
    $tops = @(); $xLo = 0.0; $wLo = 0.0; $xHi = 0.0; $wHi = 0.0
    for ($c = 0; $c -lt 144; $c++) {
        $top = -1
        for ($z = 0; $z -lt $nz; $z++) { if ($v[$c * $nz + $z] -gt 0) { $top = $z } }
        # Shear per column: density-weighted x of its own upper vs lower half.
        for ($z = 0; $z -le $top; $z++) {
            $d = $v[$c * $nz + $z]; $x = $pts[$c * $nz + $z][0]
            if ($z -lt ($top + 1) / 2) { $xLo += $x * $d; $wLo += $d } else { $xHi += $x * $d; $wHi += $d }
        }
        if ($top -ge 0) { $tops += $top + 1 }
    }
    $s = $tops | Sort-Object
    $med = if ($s.Count) { $s[[int]($s.Count / 2)] } else { 0 }
    $mx = ($s | Measure-Object -Maximum).Maximum
    Check ($s.Count -gt 5 -and $mx / [math]::Max($med, 1) -gt 1.4 -and $mx -ge 0.7 * $nz) ("tops: {0} cloudy columns, median {1}/{3}, max {2}/{3} (tallest must reach 70% of the layer)" -f $s.Count, $med, $mx, $nz)
    $dx = if ($wLo -gt 0 -and $wHi -gt 0) { $xHi / $wHi - $xLo / $wLo } else { [double]::NaN }
    Check ($dx -gt 0) ("shear: upper half centroid {0:N0} m downwind of lower half" -f $dx)
}
finally {
    $null = Invoke-RtIpc world.set_climate @{ wind_speed_mps = $c0.wind_speed_mps; wind_direction = $c0.wind_direction }
    $null = Invoke-RtIpc world.set_clouds @{ derive_from_climate = $g0.derive_from_climate; layers = $g0.layers; precipitation = $g0.precipitation; cirrus = $g0.cirrus }
}
if ($fails -eq 0) { Write-Host "ALL PASS" -ForegroundColor Green; exit 0 }
Write-Host "$fails FAILED" -ForegroundColor Red; exit 1
