<#
.SYNOPSIS
    Atmosfer Faz 3a kabul testi: bulut otoritesi + GPU bulut alani.
    docs/dev/ATMOSPHERE_CLOUDS.md §7, NEXT_BUILD_CHECKS "Faz 3a".

.DESCRIPTION
    Tamamen SAYISAL; goruntu okumaz (3a'da yeni resim yok, eski hacim ciziyor).

      1. Yuzey: get_clouds anahtarlari, 8 preset.
      2. Kismi yama: layers eleman bazinda birlesir (katman 1 degisir, 0 degismez).
      3. Red: gecersiz sonum ve cakisan katmanlar REDDEDILIR, durum degismez.
      4. Preset: fair_weather_cumulus beklenen katmani kurar.
      5. Alan: acik gokte yogunluk 0; cumulus kabugunda ortunun makul kesri
         dolu, en yuksek yogunluk <= katman sonumu.
      6. Opaklik: dolu bir sutunun gecirgenligi ~0, bos sutununki 1.
      7. Parite: render ve viewport cihazlari AYNI noktada AYNI deger.
      8. Determinizm: ayni sorgu iki kez -> bayt ayni.
      9. Ruzgar: iklim ruzgari + zaman, alani KAYDIRIR; weather_generations
         SABIT kalir (harita yeniden uretilmez, yalniz bakis kayar).

    ★ En sinsi basarisizlik: 7'de iki cihaz FARKLI deger verir ama ikisi de
      makul. Bu, harita/noise'in cihaz basina farkli uretildigi (tohum,
      bicim) anlamina gelir; 3c'de RT ile RayFusion bulutlari "biraz farkli"
      gorunur ve kimse hata demez.

    Baslangic bulutlari, iklimi ve kare sonunda geri yazilir.
#>
[CmdletBinding()]
param()

Import-Module "$PSScriptRoot\RtIpc.psm1" -Force

$fails = 0
function Check([bool]$ok, [string]$what) {
    if ($ok) { Write-Host "  OK   $what" -ForegroundColor Green }
    else     { Write-Host "  FAIL $what" -ForegroundColor Red; $script:fails++ }
}
function Near([double]$a, [double]$b, [double]$tol) { [math]::Abs($a - $b) -le $tol }

$g0 = Invoke-RtIpc world.get_clouds
$c0 = Invoke-RtIpc world.get_climate
$f0 = (Invoke-RtIpc timeline.get_frame)
$alt = (Invoke-RtIpc world.get_atmosphere).altitude
Write-Host ("start: any_enabled={0} revision={1} altitude={2}" -f $g0.any_enabled, $g0.revision, $alt) -ForegroundColor Cyan

function SceneY([double]$altitudeAboveSea) { return $altitudeAboveSea - $alt }

try {
    # ── 1. Yuzey ──────────────────────────────────────────────────────────
    Write-Host "1. surface"
    Check ($g0.layers.Count -eq 3) "3 layers"
    Check ($null -ne $g0.cirrus -and $null -ne $g0.weather -and $null -ne $g0.quality) "cirrus / weather / quality blocks"
    Check ($g0.presets.Count -eq 8) ("8 presets ({0})" -f ($g0.presets -join ','))

    # ── 2. Kismi yama ─────────────────────────────────────────────────────
    Write-Host "2. partial patch merges layers element-wise"
    $null = Invoke-RtIpc world.apply_cloud_preset @{ preset = 'clear' }
    $before = Invoke-RtIpc world.get_clouds
    $null = Invoke-RtIpc world.set_clouds @{ layers = @(@{}, @{ coverage = 0.42 }) }
    $after = Invoke-RtIpc world.get_clouds
    Check (Near $after.layers[1].coverage 0.42 1e-5) "layer 1 coverage -> 0.42"
    Check (Near $after.layers[0].coverage $before.layers[0].coverage 1e-6) "layer 0 untouched"
    Check ($after.revision -gt $before.revision) "revision bumped"

    # ── 3. Red ────────────────────────────────────────────────────────────
    Write-Host "3. rejection"
    $rev = (Invoke-RtIpc world.get_clouds).revision
    $rejected = $false
    try { $null = Invoke-RtIpc world.set_clouds @{ layers = @(@{ extinction_per_m = -1.0 }) } } catch { $rejected = $true }
    Check $rejected "negative extinction rejected"
    $rejected = $false
    try {
        $null = Invoke-RtIpc world.set_clouds @{ layers = @(
            @{ enabled = $true; base_altitude_m = 1000; thickness_m = 2000 },
            @{ enabled = $true; base_altitude_m = 2000; thickness_m = 1000 }) }
    } catch { $rejected = $true }
    Check $rejected "overlapping enabled layers rejected"
    Check ((Invoke-RtIpc world.get_clouds).revision -eq $rev) "state unchanged after rejections"
    $rejected = $false
    try { $null = Invoke-RtIpc world.apply_cloud_preset @{ preset = 'no_such_sky' } } catch { $rejected = $true }
    Check $rejected "unknown preset rejected"

    # ── 4. Preset ─────────────────────────────────────────────────────────
    Write-Host "4. preset"
    $null = Invoke-RtIpc world.apply_cloud_preset @{ preset = 'fair_weather_cumulus' }
    $cl = Invoke-RtIpc world.get_clouds
    $L = $cl.layers[0]
    Check ($L.enabled -and (Near $L.base_altitude_m 1200 1e-3) -and (Near $L.thickness_m 1200 1e-3)) "layer 0 enabled, 1200 m base, 1200 m thick"
    Check ($cl.any_enabled) "any_enabled"

    # ── 5. Alan ───────────────────────────────────────────────────────────
    Write-Host "5. density field"
    $mid = SceneY 1800.0
    $pts = @()
    for ($i = 0; $i -lt 32; $i++) { for ($k = 0; $k -lt 32; $k++) { $pts += ,@(($i - 16) * 350.0, $mid, ($k - 16) * 350.0) } }
    $d = (Invoke-RtIpc world.sample_clouds @{ mode = 'density'; points = $pts }).values
    $filled = @($d | Where-Object { $_ -gt 1e-5 }).Count / $d.Count
    $maxD = ($d | Measure-Object -Maximum).Maximum
    Check ($filled -gt 0.03 -and $filled -lt 0.6) ("filled fraction {0:N3} in (0.03, 0.6) for coverage {1}" -f $filled, $L.coverage)
    Check ($maxD -le $L.extinction_per_m + 1e-6) ("max density {0:E3} <= extinction {1}" -f $maxD, $L.extinction_per_m)
    $ground = (Invoke-RtIpc world.sample_clouds @{ mode = 'density'; points = @(,@(0.0, (SceneY 200.0), 0.0)) }).values[0]
    $above = (Invoke-RtIpc world.sample_clouds @{ mode = 'density'; points = @(,@(0.0, (SceneY 6000.0), 0.0)) }).values[0]
    Check ($ground -eq 0 -and $above -eq 0) "zero below and above the shell"

    # ── 6. Opaklik ────────────────────────────────────────────────────────
    Write-Host "6. column transmittance"
    $segs = @()
    for ($i = 0; $i -lt 64; $i++) {
        $x = ($i - 32) * 211.0; $z = ($i % 8) * 433.0
        $segs += ,@(@($x, (SceneY 1150.0), $z), @($x, (SceneY 2450.0), $z))
    }
    $t = (Invoke-RtIpc world.sample_clouds @{ mode = 'transmittance'; segments = $segs; steps = 256 }).values
    $minT = ($t | Measure-Object -Minimum).Minimum
    $maxT = ($t | Measure-Object -Maximum).Maximum
    Check ($minT -lt 0.05) ("some column is opaque (min T {0:E2})" -f $minT)
    Check (Near $maxT 1.0 1e-4) ("some column is clear (max T {0:N5})" -f $maxT)

    # ── 7. Parite ─────────────────────────────────────────────────────────
    Write-Host "7. render vs viewport device parity"
    $stats = Invoke-RtIpc world.cloud_stats
    $vp = $stats.backends | Where-Object { $_.role -eq 'viewport' }
    if ($vp -and $vp.is_vulkan) {
        $dv = (Invoke-RtIpc world.sample_clouds @{ mode = 'density'; points = $pts; backend = 'viewport' }).values
        $maxDiff = 0.0
        for ($i = 0; $i -lt $d.Count; $i++) { $maxDiff = [math]::Max($maxDiff, [math]::Abs($d[$i] - $dv[$i])) }
        Check ($maxDiff -lt 1e-6) ("max |render - viewport| = {0:E2}" -f $maxDiff)
    } else { Write-Host "  SKIP viewport backend is not Vulkan" -ForegroundColor Yellow }

    # ── 8. Determinizm ────────────────────────────────────────────────────
    Write-Host "8. determinism"
    $d2 = (Invoke-RtIpc world.sample_clouds @{ mode = 'density'; points = $pts }).values
    $same = $true
    for ($i = 0; $i -lt $d.Count; $i++) { if ($d[$i] -ne $d2[$i]) { $same = $false; break } }
    Check $same "same query twice -> identical values"

    # ── 9. Ruzgar kaydirmasi ──────────────────────────────────────────────
    Write-Host "9. wind drifts the lookup, not the map"
    $wg0 = (($stats.backends | Where-Object { $_.role -eq 'render' }).weather_generations)
    $null = Invoke-RtIpc world.set_climate @{ wind_speed_mps = 20.0; wind_direction = @(1.0, 0.0, 0.0) }
    $null = Invoke-RtIpc timeline.set_frame @{ frame = 240 }
    # The cloud time follows the frame in the UI frame loop (TimelineWidget),
    # which a script call does not run: wait for it instead of assuming.
    $deadline = (Get-Date).AddSeconds(5)
    do { Start-Sleep -Milliseconds 100; $cl = Invoke-RtIpc world.get_clouds } while ($cl.time_seconds -lt 1.0 -and (Get-Date) -lt $deadline)
    Check ($cl.wind_offset_m[0] -gt 1.0) ("drift x {0:N1} m > 0 at t={1:N2} s" -f $cl.wind_offset_m[0], $cl.time_seconds)
    $d3 = (Invoke-RtIpc world.sample_clouds @{ mode = 'density'; points = $pts }).values
    $changed = 0
    for ($i = 0; $i -lt $d.Count; $i++) { if ([math]::Abs($d[$i] - $d3[$i]) -gt 1e-6) { $changed++ } }
    Check ($changed -gt 0) ("field moved: {0} of {1} samples changed" -f $changed, $d.Count)
    $wg1 = (((Invoke-RtIpc world.cloud_stats).backends | Where-Object { $_.role -eq 'render' }).weather_generations)
    Check ($wg1 -eq $wg0) ("weather_generations flat ({0} -> {1})" -f $wg0, $wg1)
}
finally {
    $null = Invoke-RtIpc world.set_climate @{ wind_speed_mps = $c0.wind_speed_mps; wind_direction = $c0.wind_direction }
    try { $null = Invoke-RtIpc timeline.set_frame @{ frame = [int]$f0 } } catch { }
    $restore = @{ layers = $g0.layers; cirrus = $g0.cirrus; weather = $g0.weather; quality = $g0.quality }
    $null = Invoke-RtIpc world.set_clouds $restore
}

Write-Host ""
if ($fails -eq 0) { Write-Host "ALL PASS" -ForegroundColor Green; exit 0 }
Write-Host "$fails FAILED" -ForegroundColor Red
exit 1
