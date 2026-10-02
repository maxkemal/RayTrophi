<#
.SYNOPSIS
    Atmosfer Faz 2 kabul testi: iklim -> fizik tuketicileri (tek yonlu).
    docs/dev/ATMOSPHERE_SYSTEM.md, NEXT_BUILD_CHECKS.md "Faz 2".

.DESCRIPTION
    Tamamen SAYISAL. Kapili tuketiciyi acar, iklimi degistirir, tuketicinin
    RAPORLADIGI etkin degerin iklimi izledigini olcer, sonra her sey geri yazilir.

      1. world.get_thermal: yeni anahtarlar var mi, varsayilan davranis DEGISMEDI mi
         (inherit kapali -> effective == yerel ambient, kaynak 'local').
      2. inherit acik -> effective_ambient_kelvin iklimin T'sini izler
         (ambient_source 'atmosphere'), KAPALIYKEN izlemez. Yerel deger korunur.
      3. reference_kelvin ambient'tan AYRI: ambient degisince reference sabit.
      4. Red: negatif reference_kelvin reddedilir, durum yarim kalmaz.
      5. Partikul: rüzgar 0 iken effective_air_wind SIFIR (sakin dunya = eski
         davranis), rüzgar > 0 iken iklim rüzgarina esit; inherit kapaliyken 0.
      6. Foliage / okyanus / gaz: -ScatterGroup / -WaterSurface / -GasDomain
         verilirse ayni sinama; verilmezse ATLANIR (sahne varligi gerekir).

    ★ En sinsi basarisizlik: effective_* alani iklimi izliyor ama tuketici
      HAM alani okuyor (kod yolu kacagi). Bu betik raporu olcer; gercekten
      simulasyona ulastigi bir KARE OYNATILARAK ayrica dogrulanmali
      (NEXT_BUILD_CHECKS Faz 2, gorsel/oynatma maddeleri).

    Baslangic durumu sonunda geri yazilir.
#>
[CmdletBinding()]
param(
    [string]$ScatterGroup = '',
    [string]$WaterSurface = '',
    [string]$GasDomain = ''
)

Import-Module "$PSScriptRoot\RtIpc.psm1" -Force

$fails = 0
$skips = 0
function Check([bool]$ok, [string]$what) {
    if ($ok) { Write-Host "  OK   $what" -ForegroundColor Green }
    else     { Write-Host "  FAIL $what" -ForegroundColor Red; $script:fails++ }
}
function Skip([string]$what) { Write-Host "  SKIP $what" -ForegroundColor Yellow; $script:skips++ }
function Near([double]$a, [double]$b, [double]$tol) { [math]::Abs($a - $b) -le $tol }

$c0 = Invoke-RtIpc world.get_climate
$t0 = Invoke-RtIpc world.get_thermal
$p0 = $null
try { $p0 = Invoke-RtIpc particle.get_physics } catch { }
Write-Host ("start: climate T={0} K wind={1} m/s | thermal ambient={2} inherit={3}" -f
    $c0.surface_temperature_k, $c0.wind_speed_mps, $t0.ambient_kelvin, $t0.inherit_atmosphere) -ForegroundColor Cyan

try {
    # Bilinen bir baslangic: kapali, sakin.
    $null = Invoke-RtIpc world.set_thermal @{ inherit_atmosphere = $false; ambient_kelvin = 293.0 }
    $null = Invoke-RtIpc world.set_climate @{ surface_temperature_k = 300.0; wind_speed_mps = 0.0; wind_direction = @(1.0, 0.0, 0.0) }

    # ── 1. Varsayilan davranis ────────────────────────────────────────────
    Write-Host "1. thermal surface, inherit OFF is the old behaviour"
    $t = Invoke-RtIpc world.get_thermal
    Check ($null -ne $t.effective_ambient_kelvin -and $null -ne $t.reference_kelvin -and $null -ne $t.ambient_source) "new keys present (effective_ambient_kelvin, reference_kelvin, ambient_source)"
    Check ($t.inherit_atmosphere -eq $false) "inherit_atmosphere off"
    Check (Near $t.effective_ambient_kelvin 293.0 1e-3) ("effective ambient {0} == local 293" -f $t.effective_ambient_kelvin)
    Check ($t.ambient_source -eq 'local') "ambient_source 'local'"
    Check (Near $t.drying_scale 1.0 1e-6) "drying_scale 1.0 while not inheriting"

    # ── 2. Iklimi izleme ──────────────────────────────────────────────────
    Write-Host "2. inherit ON follows the climate; OFF does not"
    $null = Invoke-RtIpc world.set_thermal @{ inherit_atmosphere = $true }
    $t = Invoke-RtIpc world.get_thermal
    Check ($t.ambient_source -eq 'atmosphere') "ambient_source 'atmosphere'"
    Check (Near $t.effective_ambient_kelvin 300.0 0.05) ("effective ambient {0} ~= climate 300 (scene origin)" -f $t.effective_ambient_kelvin)
    Check (Near $t.ambient_kelvin 293.0 1e-3) "local ambient_kelvin KEPT (293), not overwritten"
    $null = Invoke-RtIpc world.set_climate @{ surface_temperature_k = 280.0 }
    $t = Invoke-RtIpc world.get_thermal
    Check (Near $t.effective_ambient_kelvin 280.0 0.05) ("effective follows a climate change: {0} ~= 280" -f $t.effective_ambient_kelvin)
    $null = Invoke-RtIpc world.set_climate @{ surface_relative_humidity = 0.25 }
    $t = Invoke-RtIpc world.get_thermal
    Check (Near $t.drying_scale 0.75 1e-3) ("drying_scale {0} == 1 - RH (0.75)" -f $t.drying_scale)
    $null = Invoke-RtIpc world.set_thermal @{ inherit_atmosphere = $false }
    $null = Invoke-RtIpc world.set_climate @{ surface_temperature_k = 310.0 }
    $t = Invoke-RtIpc world.get_thermal
    Check (Near $t.effective_ambient_kelvin 293.0 1e-3) ("OFF: effective stays local {0} despite climate 310" -f $t.effective_ambient_kelvin)

    # ── 3. Kalibrasyon sifiri ambient'tan ayri ────────────────────────────
    Write-Host "3. reference_kelvin is NOT the ambient"
    $null = Invoke-RtIpc world.set_thermal @{ reference_kelvin = 293.0 }
    $null = Invoke-RtIpc world.set_thermal @{ inherit_atmosphere = $true }
    $null = Invoke-RtIpc world.set_climate @{ surface_temperature_k = 270.0 }
    $t = Invoke-RtIpc world.get_thermal
    Check (Near $t.reference_kelvin 293.0 1e-3) ("reference {0} unchanged by a 270 K ambient" -f $t.reference_kelvin)
    Check (Near $t.effective_ambient_kelvin 270.0 0.05) "while the ambient moved to 270"

    # ── 4. Red ────────────────────────────────────────────────────────────
    Write-Host "4. rejection leaves state untouched"
    $rejected = $false
    try { $null = Invoke-RtIpc world.set_thermal @{ reference_kelvin = -5.0; oxygen_availability = 0.5 } } catch { $rejected = $true }
    Check $rejected "negative reference_kelvin rejected"
    $t = Invoke-RtIpc world.get_thermal
    Check (Near $t.reference_kelvin 293.0 1e-3) "reference unchanged after the rejected call"

    # ── 5. Partikul ruzgari ───────────────────────────────────────────────
    Write-Host "5. particle drag targets the climate wind"
    if ($null -eq $p0) { Skip "particle.get_physics unavailable (no particle system)" }
    else {
        $null = Invoke-RtIpc particle.set_physics @{ inherit_atmosphere = $true }
        $p = Invoke-RtIpc particle.get_physics
        $w = $p.effective_air_wind
        Check ((Near $w[0] 0 1e-6) -and (Near $w[1] 0 1e-6) -and (Near $w[2] 0 1e-6)) "calm world: effective_air_wind is exactly zero (old behaviour)"
        $null = Invoke-RtIpc world.set_climate @{ wind_speed_mps = 8.0; wind_direction = @(0.0, 0.0, 1.0) }
        $p = Invoke-RtIpc particle.get_physics
        $w = $p.effective_air_wind
        Check ((Near $w[2] 8.0 1e-3) -and (Near $w[0] 0 1e-3)) ("8 m/s along +Z: air = ({0}, {1}, {2})" -f $w[0], $w[1], $w[2])
        $null = Invoke-RtIpc particle.set_physics @{ inherit_atmosphere = $false }
        $p = Invoke-RtIpc particle.get_physics
        $w = $p.effective_air_wind
        Check ((Near $w[0] 0 1e-6) -and (Near $w[2] 0 1e-6)) "inherit OFF: air is zero despite an 8 m/s climate"
    }

    # ── 6. Ek tuketiciler ─────────────────────────────────────────────────
    Write-Host "6. foliage / ocean / gas (need scene objects)"
    $null = Invoke-RtIpc world.set_climate @{ wind_speed_mps = 10.0; wind_direction = @(0.0, 0.0, 1.0) }
    if ($ScatterGroup) {
        $orig = Invoke-RtIpc scatter.get_wind @{ group = $ScatterGroup }
        $null = Invoke-RtIpc scatter.set_wind @{ group = $ScatterGroup; enabled = $true; inherit_atmosphere = $true; speed = 1.0; strength = 0.1 }
        $g = Invoke-RtIpc scatter.get_wind @{ group = $ScatterGroup }
        Check ($g.wind_source -eq 'atmosphere') "scatter wind_source 'atmosphere'"
        Check (Near $g.effective_speed 2.0 1e-3) ("effective_speed {0} == speed x (10/5) = 2.0" -f $g.effective_speed)
        Check (Near $g.effective_strength 0.4 1e-3) ("effective_strength {0} == strength x (10/5)^2 = 0.4" -f $g.effective_strength)
        Check (Near $g.effective_direction.z 1.0 1e-3) "effective_direction follows the climate (+Z)"
        $null = Invoke-RtIpc world.set_climate @{ wind_speed_mps = 0.0 }
        $g = Invoke-RtIpc scatter.get_wind @{ group = $ScatterGroup }
        Check (Near $g.effective_strength 0.0 1e-6) "calm world: an inheriting group is still (strength 0)"
        $null = Invoke-RtIpc scatter.set_wind @{ group = $ScatterGroup; enabled = $orig.enabled; inherit_atmosphere = $orig.inherit_atmosphere; speed = $orig.speed; strength = $orig.strength }
    } else { Skip "foliage: pass -ScatterGroup <name>" }

    if ($WaterSurface) {
        $null = Invoke-RtIpc world.set_climate @{ wind_speed_mps = 12.0; wind_direction = @(0.0, 0.0, 1.0) }
        $ow = Invoke-RtIpc water.get_wind @{ surface = $WaterSurface }
        $null = Invoke-RtIpc water.set_wind @{ surface = $WaterSurface; inherit_atmosphere = $true }
        $wv = Invoke-RtIpc water.get_wind @{ surface = $WaterSurface }
        Check ($wv.wind_source -eq 'atmosphere') "water wind_source 'atmosphere'"
        Check (Near $wv.effective_speed_mps 12.0 1e-3) ("effective speed {0} == 12" -f $wv.effective_speed_mps)
        Check (Near $wv.effective_direction_degrees 90.0 0.5) ("effective direction {0} == 90 deg (+Z)" -f $wv.effective_direction_degrees)
        $null = Invoke-RtIpc water.set_wind @{ surface = $WaterSurface; inherit_atmosphere = $ow.inherit_atmosphere }
    } else { Skip "ocean: pass -WaterSurface <name>" }

    if ($GasDomain) {
        $gs0 = Invoke-RtIpc gas.get_settings @{ domain = $GasDomain }
        $null = Invoke-RtIpc world.set_climate @{ wind_speed_mps = 0.0; lapse_rate_k_per_m = 0.0065 }
        $null = Invoke-RtIpc gas.set_settings @{ domain = $GasDomain; inherit_atmosphere = $true }
        $gs = Invoke-RtIpc gas.get_settings @{ domain = $GasDomain }
        $kpu = (Invoke-RtIpc world.get_thermal).kelvin_per_unit
        $expected = (9.80665 / 1004.0 - 0.0065) / $kpu
        Check (Near $gs.effective_ambient_stratification $expected ($expected * 0.01)) ("stratification {0:E3} == (g/cp - L)/kpu = {1:E3}" -f $gs.effective_ambient_stratification, $expected)
        $null = Invoke-RtIpc world.set_climate @{ lapse_rate_k_per_m = -0.01 }   # inversion
        $gs2 = Invoke-RtIpc gas.get_settings @{ domain = $GasDomain }
        Check ($gs2.effective_ambient_stratification -gt $gs.effective_ambient_stratification) "an inversion (negative lapse) is MORE stable"
        $null = Invoke-RtIpc gas.set_settings @{ domain = $GasDomain; inherit_atmosphere = $gs0.inherit_atmosphere }
    } else { Skip "gas: pass -GasDomain <name>" }
}
finally {
    $null = Invoke-RtIpc world.set_climate @{
        surface_temperature_k = $c0.surface_temperature_k
        lapse_rate_k_per_m = $c0.lapse_rate_k_per_m
        surface_relative_humidity = $c0.surface_relative_humidity
        surface_pressure_pa = $c0.surface_pressure_pa
        wind_direction = $c0.wind_direction
        wind_speed_mps = $c0.wind_speed_mps
    }
    $null = Invoke-RtIpc world.set_thermal @{
        ambient_kelvin = $t0.ambient_kelvin
        reference_kelvin = $(if ($null -ne $t0.reference_kelvin) { $t0.reference_kelvin } else { $t0.ambient_kelvin })
        inherit_atmosphere = [bool]$t0.inherit_atmosphere
    }
    if ($null -ne $p0) { $null = Invoke-RtIpc particle.set_physics @{ inherit_atmosphere = [bool]$p0.inherit_atmosphere } }
}

Write-Host ""
if ($fails -eq 0) { Write-Host ("ALL PASS ({0} skipped)" -f $skips) -ForegroundColor Green; exit 0 }
Write-Host ("$fails FAILED ({0} skipped)" -f $skips) -ForegroundColor Red
exit 1
