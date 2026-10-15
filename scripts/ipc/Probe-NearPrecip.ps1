<#
.SYNOPSIS
    Kamera onu yagis N1 kabul testi (docs/dev/ATMOSPHERE_NEAR_PRECIP.md).

.DESCRIPTION
    Tamamen SAYISAL; goruntu okumaz. world.sample_clouds mode near_precip,
    yakin damlalarin CIZILDIGI durumu shader'in kendi fonksiyonuyla olcer
    (near_precip.glsl nearPrecipForms).

      1. Nemli + kararsiz iklim -> turetilmis yagis > 0; zeminde en islak nokta.
      2. TEK KAYNAK: near_precip.rate_mm_h = precipitation modunun degeri,
         dominant_form = precipitation_form modunun degeri (ayni nokta).
      3. Kuru noktada damla yok (rate 0, number_density 0).
      4. Cizgi boyu = |dusus + yuzey ruzgari| x shutter_s; shutter 3 katina
         cikinca cizgi de 3 kat.
      5. Kar zorlaninca: dominant snow, dusus 1 m/s, yaricap 2.5 mm, ayni
         oranda yagmurdan DAHA COK tane (kar mm su basina cok daha fazla sacar).
      6. near_enabled false -> near_range_m = curtain_start_m = 0 (perde
         kameradan baslar).
      7. shutter_s 0..0.5 s disinda reddedilir, durum degismez. (Eski
         dosyalardaki lightning.shutter_s yalniz DOSYADAN okunur: IPC yamasinda
         ust duzey anahtar her zaman onu ezer.)
      8. world.get_weather_effects artik havadaki alan tasimiyor (visibility,
         visual_mode yok).

    ★ En sinsi basarisizlik: hepsi PASS ama goruntude damla yok -> kamera
      yagisin ALTINDA degil (1. adimdaki nokta kamera degil). Goz testi icin
      kamerayi o noktaya tasi; Measure at camera panelde ayni sayilari verir.

    Baslangic iklimi ve bulutlari geri yazilir.
#>
[CmdletBinding()]
param([string]$Backend = 'viewport')

Import-Module "$PSScriptRoot\RtIpc.psm1" -Force
$ErrorActionPreference = 'Stop'

$fails = 0
function Check([bool]$ok, [string]$what) {
    if ($ok) { Write-Host "  OK   $what" -ForegroundColor Green }
    else     { Write-Host "  FAIL $what" -ForegroundColor Red; $script:fails++ }
}
function Near($p) {
    (Invoke-RtIpc world.sample_clouds @{ mode = 'near_precip'; points = @(,@($p[0], $p[1], $p[2])); backend = $Backend }).points[0]
}
function Values([string]$mode, $pts) {
    (Invoke-RtIpc world.sample_clouds @{ mode = $mode; points = $pts; backend = $Backend }).values
}

$c0 = Invoke-RtIpc world.get_climate
$g0 = Invoke-RtIpc world.get_clouds
try {
    Write-Host "1. raining scene"
    $null = Invoke-RtIpc world.set_clouds @{ derive_from_climate = $true; shutter_s = (1.0 / 30.0);
        precipitation = @{ form = 'auto'; near_enabled = $true; near_range_m = 15.0 } }
    $null = Invoke-RtIpc world.set_climate @{ surface_temperature_k = 295.15; surface_relative_humidity = 0.93; instability = 0.9 }
    $g = Invoke-RtIpc world.get_clouds
    Check ($g.precipitation.rate_mm_h -gt 1.0) ("derived rate {0:N1} mm/h" -f $g.precipitation.rate_mm_h)
    $alt = (Invoke-RtIpc world.get_atmosphere).altitude
    $y = [math]::Max(2.0, 0.05 * ($g.layers[0].base_altitude_m - $alt))
    $pts = @(); foreach ($i in -10..10) { foreach ($k in -10..10) { $pts += ,@(($i * 2000.0), $y, ($k * 2000.0)) } }
    $v = Values 'precipitation' $pts
    $wi = 0; $di = -1
    for ($i = 0; $i -lt $v.Count; $i++) {
        if ($v[$i] -gt $v[$wi]) { $wi = $i }
        if ($v[$i] -le 0 -and $di -lt 0) { $di = $i }
    }
    $wet = $pts[$wi]
    Check ($v[$wi] -gt 0.01) ("wettest ground point ({0:0}, {1:0}, {2:0}): {3:N2} mm/h" -f $wet[0], $wet[1], $wet[2], $v[$wi])

    Write-Host "2. one source"
    $n = Near $wet
    $form = (Invoke-RtIpc world.sample_clouds @{ mode = 'precipitation_form'; points = @(,$wet); backend = $Backend }).forms[0]
    Check ([math]::Abs($n.rate_mm_h - $v[$wi]) -le 1e-4 * [math]::Max(1.0, $v[$wi])) ("near rate {0:N4} = curtain rate {1:N4}" -f $n.rate_mm_h, $v[$wi])
    Check ($n.dominant_form -eq $form) ("near form '{0}' = precipitation_form '{1}'" -f $n.dominant_form, $form)
    Check ($n.number_density_per_m3 -gt 0) ("{0:N0} drops/m3, radius {1:N4} m" -f $n.number_density_per_m3, $n.radius_m)
    Check ([math]::Abs($n.near_range_m - 15.0) -lt 1e-3 -and $n.curtain_start_m -eq $n.near_range_m) ("range {0} m = curtain start" -f $n.near_range_m)

    Write-Host "3. dry point"
    if ($di -ge 0) {
        $d = Near $pts[$di]
        Check ($d.rate_mm_h -eq 0 -and $d.number_density_per_m3 -eq 0 -and $d.dominant_form -eq 'none') "no rate, no drops"
    } else { Write-Host "  SKIP every grid point is wet" -ForegroundColor Yellow }

    Write-Host "4. streak = speed x shutter"
    $c = Invoke-RtIpc world.get_climate
    # Near-surface air = 0.6 x climate wind (CloudParams.h kSurfaceWindFraction,
    # the curtain's foot drag uses the same share).
    $surf = 0.6 * $c.wind_speed_mps
    $speed = [math]::Sqrt($n.fall_speed_mps * $n.fall_speed_mps + $surf * $surf)
    $want = $speed / 30.0
    Check ([math]::Abs($n.streak_length_m - $want) -le 0.01 * $want) ("streak {0:N4} m = {1:N4} m (fall {2:N2}, surface wind {3:N2} m/s)" -f $n.streak_length_m, $want, $n.fall_speed_mps, $surf)
    $null = Invoke-RtIpc world.set_clouds @{ shutter_s = 0.1 }
    $n3 = Near $wet
    Check ([math]::Abs($n3.streak_length_m - 3.0 * $n.streak_length_m) -le 0.01 * $n3.streak_length_m) ("shutter 0.1 s: streak {0:N4} m = 3 x" -f $n3.streak_length_m)

    Write-Host "5. forced snow"
    $null = Invoke-RtIpc world.set_clouds @{ shutter_s = (1.0 / 30.0); precipitation = @{ form = 'rain' } }
    $rain = Near $wet
    $null = Invoke-RtIpc world.set_clouds @{ precipitation = @{ form = 'snow' } }
    $snow = Near $wet
    Check ($snow.dominant_form -eq 'snow' -and [math]::Abs($snow.fall_speed_mps - 1.0) -lt 1e-3 -and [math]::Abs($snow.radius_m - 0.0025) -lt 1e-6) ("snow: fall {0} m/s, radius {1} m" -f $snow.fall_speed_mps, $snow.radius_m)
    Check ($rain.dominant_form -eq 'rain' -and $snow.number_density_per_m3 -gt $rain.number_density_per_m3) ("same rate: snow {0:N0} > rain {1:N0} drops/m3" -f $snow.number_density_per_m3, $rain.number_density_per_m3)
    $null = Invoke-RtIpc world.set_clouds @{ precipitation = @{ form = 'auto' } }

    Write-Host "6. near drops off"
    $null = Invoke-RtIpc world.set_clouds @{ precipitation = @{ near_enabled = $false } }
    $off = Near $wet
    Check ($off.near_range_m -eq 0 -and $off.curtain_start_m -eq 0) "range 0: the curtain starts at the camera"
    $null = Invoke-RtIpc world.set_clouds @{ precipitation = @{ near_enabled = $true } }

    Write-Host "7. shutter_s is validated"
    $rejected = $false
    try { $null = Invoke-RtIpc world.set_clouds @{ shutter_s = 0.6 } } catch { $rejected = $true }
    $sh = (Invoke-RtIpc world.get_clouds).shutter_s
    Check ($rejected -and [math]::Abs($sh - 1.0 / 30.0) -lt 1e-6) ("shutter_s 0.6 rejected, still {0:N4}" -f $sh)

    Write-Host "8. surface weather only"
    $we = Invoke-RtIpc world.get_weather_effects
    $names = @($we.PSObject.Properties.Name)
    Check (-not ($names -contains 'visibility') -and -not ($names -contains 'visual_mode')) ("weather_effects keys: {0}" -f ($names -join ','))
}
finally {
    $null = Invoke-RtIpc world.set_climate @{ surface_temperature_k = $c0.surface_temperature_k; surface_relative_humidity = $c0.surface_relative_humidity; instability = $c0.instability }
    $null = Invoke-RtIpc world.set_clouds @{ derive_from_climate = $g0.derive_from_climate; layers = $g0.layers; precipitation = $g0.precipitation; shutter_s = $g0.shutter_s }
}
if ($fails -eq 0) { Write-Host "ALL PASS" -ForegroundColor Green; exit 0 }
Write-Host "$fails FAILED" -ForegroundColor Red; exit 1
