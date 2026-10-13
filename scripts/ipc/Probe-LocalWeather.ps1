<#
.SYNOPSIS
    Yerel hava kabul testi (parti 17): uygunluk, son yagis, yerel sis.
    docs/dev/ATMOSPHERE_LOCAL_WINDS_FOG.md §2-4, NEXT_BUILD_CHECKS "parti 17".

.DESCRIPTION
    Tamamen SAYISAL; goruntu okumaz.

      1. Yuzey: local_weather blogu get_clouds'ta; iki yeni metot cevap veriyor.
      2. Gunduz + ruzgar: radyasyon sisi YOK (drivers.night ~0 veya calm ~0).
      3. Gece + sakin + nemli + acik gok: harita aktif, sis var; noktada
         zemin ustunde sonum > 0, sis tepesinin 50 m ustunde 0.
      4. Cukur > sirt: arazi varsa, yerel ortalamanin altindaki noktalarin
         ortalama sonumu ustundekilerden buyuk.
      5. GPU: viewport cihazindaki sis revizyonu haritanin revizyonuna esit
         (bulut march'i yeni haritayi okuyor).
      5b. Sekil: world.sample_clouds yogunlugu sis icinde > 0 ve zarfi asmiyor.
      6. Guc 0 ve fog_enabled=false: harita pasif.
      7. Uygunluk: arazi + 10 m/s ruzgar -> hem ruzgar ustu (orographic) hem
         ruzgar alti (lee) noktalar var; arazi yoksa her yer 1.
      8. Determinizm: ayni nokta iki kez -> ayni deger.

    ★ En sinsi basarisizlik: 5'te revizyon geride kalir ama 3 gecer. CPU
      haritasi dogru, ekrandaki sis eski: harita GPU'ya gitmemis. IPC "sis
      var" der, goruntude yoktur.

    Baslangic iklimi, gunes yuksekligi ve bulutlar sonunda geri yazilir.
#>
[CmdletBinding()]
param()

Import-Module "$PSScriptRoot\RtIpc.psm1" -Force

$fails = 0
function Check([bool]$ok, [string]$what) {
    if ($ok) { Write-Host "  OK   $what" -ForegroundColor Green }
    else     { Write-Host "  FAIL $what" -ForegroundColor Red; $script:fails++ }
}
# The map is rebuilt by the main loop: let it run a couple of frames.
function Tick { $null = Invoke-RtIpc viewport.render_frames @{ count = 3 } }

$clouds0 = Invoke-RtIpc world.get_clouds
$climate0 = Invoke-RtIpc world.get_climate
$sun0 = (Invoke-RtIpc world.get).sun_elevation

function Grid([object]$map, [int]$k) {
    $pts = @()
    for ($i = 0; $i -lt $k; $i++) {
        for ($j = 0; $j -lt $k; $j++) {
            $x = $map.origin_m[0] + ($i + 0.5) / $k * $map.size_m
            $z = $map.origin_m[1] + ($j + 0.5) / $k * $map.size_m
            $pts += ,@($x, 0.0, $z)
        }
    }
    return ,$pts
}

try {
    # ── 1. Yuzey ──────────────────────────────────────────────────────────
    Write-Host "1. surface"
    Check ($null -ne $clouds0.local_weather) "get_clouds carries local_weather"
    $st = Invoke-RtIpc world.local_weather_stats
    Check ($null -ne $st.fog_map) "local_weather_stats has fog_map"
    Check ($st.backends.Count -eq 2) "two backend rows"
    $null = Invoke-RtIpc world.set_clouds @{ local_weather = @{ fog_enabled = $true; fog_strength = 1.0; terrain_suitability = $true } }

    # ── 2. Gunduz + ruzgar ────────────────────────────────────────────────
    Write-Host "2. day, windy: no radiation fog"
    $null = Invoke-RtIpc world.set_sun_elevation @{ sun_elevation = 40.0 }
    $null = Invoke-RtIpc world.set_climate @{ wind_speed_mps = 8.0; surface_relative_humidity = 0.95 }
    Tick
    $st = Invoke-RtIpc world.local_weather_stats
    Write-Host ("  drivers night={0:N2} calm={1:N2}" -f $st.fog_map.drivers.night, $st.fog_map.drivers.calm)
    Check ($st.fog_map.drivers.night -lt 0.05) "night ~0 at 40 deg sun"
    $s = Invoke-RtIpc world.sample_weather_local @{ points = (Grid $st.fog_map 4) }
    $rad = ($s.points | ForEach-Object { $_.fog.radiation_per_m } | Measure-Object -Maximum).Maximum
    Check ($rad -lt 1e-6) ("radiation fog 0 by day (max {0:E2})" -f $rad)

    # ── 3. Gece + sakin + nemli + acik gok ────────────────────────────────
    Write-Host "3. night, calm, humid, clear"
    $null = Invoke-RtIpc world.set_sun_elevation @{ sun_elevation = -8.0 }
    $null = Invoke-RtIpc world.set_climate @{ wind_speed_mps = 0.0; surface_relative_humidity = 0.96 }
    $null = Invoke-RtIpc world.set_clouds @{ derive_from_climate = $false; layers = @(@{ coverage = 0.1 }) }
    Tick
    $st = Invoke-RtIpc world.local_weather_stats
    $m = $st.fog_map
    Write-Host ("  map {0:N1} km over_terrain={1} fogged={2:P0} peak={3:E2} rev={4}" -f ($m.size_m / 1000), $m.over_terrain, $m.fogged_fraction, $m.max_extinction_per_m, $m.revision)
    Check $m.active "map active"
    Check ($m.fogged_fraction -gt 0.05) "fog over > 5% of the map"
    $pts = Grid $m 8
    $s = Invoke-RtIpc world.sample_weather_local @{ points = $pts }
    $best = $s.points | Sort-Object { $_.fog.radiation_per_m } -Descending | Select-Object -First 1
    Check ($best.fog.radiation_per_m -gt 1e-4) ("densest point radiation {0:E2} 1/m" -f $best.fog.radiation_per_m)
    $idx = [array]::IndexOf($s.points, $best)
    $px = $pts[$idx][0]; $pz = $pts[$idx][2]
    $low  = @($px, ($best.ground_y + 2.0), $pz)
    $high = @($px, ($best.fog.radiation_top_y + 50.0), $pz)
    $v = Invoke-RtIpc world.sample_weather_local @{ points = @($low, $high) }
    Write-Host ("  at ground+2 m {0:E2}, at top+50 m {1:E2} (top y {2:N0}, ground {3:N0})" -f $v.points[0].extinction_per_m, $v.points[1].extinction_per_m, $best.fog.radiation_top_y, $best.ground_y)
    Check ($v.points[0].extinction_per_m -gt 1e-4) "fog at the ground"
    Check ($v.points[1].extinction_per_m -lt $v.points[0].extinction_per_m * 0.05 + 1e-6) "no radiation fog 50 m above its top"

    # ── 4. Cukur > sirt ───────────────────────────────────────────────────
    Write-Host "4. hollows hold more fog than ridges"
    if ($m.over_terrain) {
        $hol = @($s.points | Where-Object { $_.on_terrain -and ($_.ground_y -lt $_.local_mean_y - 30) })
        $rid = @($s.points | Where-Object { $_.on_terrain -and ($_.ground_y -gt $_.local_mean_y + 30) })
        if ($hol.Count -gt 0 -and $rid.Count -gt 0) {
            $h = ($hol | ForEach-Object { $_.fog.radiation_per_m } | Measure-Object -Average).Average
            $r = ($rid | ForEach-Object { $_.fog.radiation_per_m } | Measure-Object -Average).Average
            Write-Host ("  hollow {0:E2} ({1} pts)  ridge {2:E2} ({3} pts)" -f $h, $hol.Count, $r, $rid.Count)
            Check ($h -gt $r * 1.5) "hollow fog > 1.5x ridge fog"
        } else { Write-Host "  SKIP no hollow/ridge pair on the 8x8 grid" -ForegroundColor Yellow }
    } else { Write-Host "  SKIP no terrain" -ForegroundColor Yellow }

    # ── 5. GPU ────────────────────────────────────────────────────────────
    Write-Host "5. the cloud march reads this map"
    Tick
    $st = Invoke-RtIpc world.local_weather_stats
    $vp = $st.backends | Where-Object { $_.role -eq 'viewport' }
    Write-Host ("  viewport cloud_rendering={0} gpu rev={1} map rev={2} uploads={3}" -f $vp.cloud_rendering, $vp.fog_revision_on_gpu, $st.fog_map.revision, $vp.fog_uploads)
    Check ($vp.cloud_rendering) "viewport draws the cloud field (fog is in it)"
    Check ($vp.fog_revision_on_gpu -eq $st.fog_map.revision) "GPU texels = published map"

    # ── 5b. Sekil: cizilen yogunluk ───────────────────────────────────────
    Write-Host "5b. drawn density (cloud march) inside the fog, under the envelope"
    $col = @(); for ($k = 0; $k -lt 6; $k++) { $col += ,@($px, ($best.ground_y + 3.0 + $k * 6.0), $pz) }
    $dens = Invoke-RtIpc world.sample_clouds @{ mode = 'density'; points = $col; backend = 'viewport' }
    $env = Invoke-RtIpc world.sample_weather_local @{ points = $col }
    $maxD = ($dens.values | Measure-Object -Maximum).Maximum
    # The bound is the column's envelope peak (fog + mist), not the value at
    # each height: the drawn lid undulates above the envelope's own lid.
    $over = 0; for ($k = 0; $k -lt $col.Count; $k++) {
        $peak = $env.points[$k].fog.radiation_per_m + $env.points[$k].fog.mist_per_m
        if ($dens.values[$k] -gt $peak * 1.05 + 1e-6) { $over++ } }
    Write-Host ("  drawn max {0:E2} 1/m, envelope at ground {1:E2}" -f $maxD, $env.points[0].extinction_per_m)
    Check ($maxD -gt 1e-5) "the march sees fog in the hollow"
    Check ($over -eq 0) "drawn density never above the envelope peak"

    # ── 6. Kapatma ────────────────────────────────────────────────────────
    Write-Host "6. strength 0 / disabled"
    $null = Invoke-RtIpc world.set_clouds @{ local_weather = @{ fog_strength = 0.0 } }
    Tick
    Check (-not (Invoke-RtIpc world.local_weather_stats).fog_map.active) "strength 0 -> inactive"
    $null = Invoke-RtIpc world.set_clouds @{ local_weather = @{ fog_strength = 1.0; fog_enabled = $false } }
    Tick
    Check (-not (Invoke-RtIpc world.local_weather_stats).fog_map.active) "fog_enabled false -> inactive"
    $null = Invoke-RtIpc world.set_clouds @{ local_weather = @{ fog_enabled = $true } }

    # ── 7. Uygunluk ───────────────────────────────────────────────────────
    Write-Host "7. birth suitability"
    $null = Invoke-RtIpc world.set_climate @{ wind_speed_mps = 10.0 }
    $null = Invoke-RtIpc world.set_sun_elevation @{ sun_elevation = 40.0 }
    Tick
    $st = Invoke-RtIpc world.local_weather_stats
    $s = Invoke-RtIpc world.sample_weather_local @{ points = (Grid $st.fog_map 10) }
    if ($st.environment.terrain.valid) {
        $oro = @($s.points | Where-Object { $_.suitability.orographic -gt 0.5 }).Count
        $lee = @($s.points | Where-Object { $_.suitability.lee -gt 0.5 }).Count
        $vals = $s.points | ForEach-Object { $_.suitability.value } | Measure-Object -Minimum -Maximum
        Write-Host ("  windward {0}  lee {1}  value {2:N2}..{3:N2}" -f $oro, $lee, $vals.Minimum, $vals.Maximum)
        Check ($oro -gt 0 -and $lee -gt 0) "both windward and lee points exist"
        Check ($vals.Maximum - $vals.Minimum -gt 0.3) "suitability varies over the terrain"
    } else {
        $ones = @($s.points | Where-Object { [math]::Abs($_.suitability.value - 1.0) -lt 1e-6 }).Count
        Check ($ones -eq $s.points.Count) "no terrain: suitability 1 everywhere"
    }

    # ── 8. Determinizm ────────────────────────────────────────────────────
    Write-Host "8. determinism"
    $p = @(,@(($st.fog_map.origin_m[0] + 0.37 * $st.fog_map.size_m), 5.0, ($st.fog_map.origin_m[1] + 0.61 * $st.fog_map.size_m)))
    $a = (Invoke-RtIpc world.sample_weather_local @{ points = $p }) | ConvertTo-Json -Depth 6 -Compress
    $b = (Invoke-RtIpc world.sample_weather_local @{ points = $p }) | ConvertTo-Json -Depth 6 -Compress
    Check ($a -eq $b) "same point twice -> same answer"
}
finally {
    $null = Invoke-RtIpc world.set_climate @{
        wind_speed_mps = $climate0.wind_speed_mps
        surface_relative_humidity = $climate0.surface_relative_humidity }
    $null = Invoke-RtIpc world.set_sun_elevation @{ sun_elevation = $sun0 }
    $null = Invoke-RtIpc world.set_clouds @{
        derive_from_climate = $clouds0.derive_from_climate
        layers = @(@{ coverage = $clouds0.layers[0].coverage })
        local_weather = @{
            terrain_suitability = $clouds0.local_weather.terrain_suitability
            fog_enabled = $clouds0.local_weather.fog_enabled
            fog_strength = $clouds0.local_weather.fog_strength } }
    Write-Host "restored climate, sun and clouds" -ForegroundColor Cyan
}

if ($fails -eq 0) { Write-Host "PASS" -ForegroundColor Green } else { Write-Host "FAIL ($fails)" -ForegroundColor Red; exit 1 }
