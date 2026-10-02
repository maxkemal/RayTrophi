<#
.SYNOPSIS
    Atmosfer Faz 1a kabul testi: iklim otoritesi (world.*_climate).
    docs/dev/ATMOSPHERE_SYSTEM.md

.DESCRIPTION
    Tamamen SAYISAL; goruntu okumaz. Dort sey olcer:

      1. Ayna: set_climate -> get_climate'in applied_* alanlari (RENDER
         PAKETINDEN okunur) otoriteyle AYNI mi. Ayrisirsa panel yalan soyluyor.
      2. ISA profili: sample_climate 0 m ile 1000 m arasinda lapse rate kadar
         soguyor, basinc ve hava yogunlugu dusuyor mu.
      3. Emekli anahtar: world.set_atmosphere @{ humidity } HATA vermeli.
         Sessizce kabul edilirse eski scriptler "calisip" hicbir sey yapmaz.
      4. Red: gecersiz girdi (negatif nem, 0 K) kirpilmamali, REDDEDILMELI.

    Baslangictaki iklim sonunda geri yazilir.

    ★ En sinsi basarisizlik: 1. adimda applied_mie_humidity_scale sabit 1.0
      kaliyor ama surface_relative_humidity degisiyor. Bu, ayna senkronunun
      koptugu ve gokyuzunun nemi hala GORMEDIGI anlamina gelir.
#>
[CmdletBinding()]
param()

Import-Module "$PSScriptRoot\RtIpc.psm1" -Force

$fails = 0
function Check([bool]$ok, [string]$what) {
    if ($ok) { Write-Host "  OK   $what" -ForegroundColor Green }
    else     { Write-Host "  FAIL $what" -ForegroundColor Red; $script:fails++ }
}

$c0 = Invoke-RtIpc world.get_climate
Write-Host ("start: T={0} K  RH={1}  lapse={2}  p={3}  wind={4} m/s" -f $c0.surface_temperature_k,
    $c0.surface_relative_humidity, $c0.lapse_rate_k_per_m, $c0.surface_pressure_pa, $c0.wind_speed_mps) -ForegroundColor Cyan

try {
    # ── 1. Ayna ────────────────────────────────────────────────────────────
    Write-Host "1. authority -> render packet mirror"
    $null = Invoke-RtIpc world.set_climate @{ surface_temperature_k = 300.0; surface_relative_humidity = 0.75 }
    $c = Invoke-RtIpc world.get_climate
    Check ([math]::Abs($c.applied_temperature_c - (300.0 - 273.15)) -lt 1e-3) ("applied_temperature_c {0} == 26.85" -f $c.applied_temperature_c)
    $expectedScale = [math]::Pow(1.0 - 0.75, -0.5)   # 2.0
    Check ([math]::Abs($c.applied_mie_humidity_scale - $expectedScale) -lt 1e-3) ("applied_mie_humidity_scale {0} == {1}" -f $c.applied_mie_humidity_scale, $expectedScale)
    $null = Invoke-RtIpc world.set_climate @{ surface_relative_humidity = 0.0 }
    $c = Invoke-RtIpc world.get_climate
    Check ([math]::Abs($c.applied_mie_humidity_scale - 1.0) -lt 1e-4) ("dry air scale {0} == 1.0" -f $c.applied_mie_humidity_scale)

    # ── 2. ISA profili ────────────────────────────────────────────────────
    Write-Host "2. ISA profile (sample_climate)"
    $null = Invoke-RtIpc world.set_climate @{ surface_temperature_k = 288.15; lapse_rate_k_per_m = 0.0065; surface_pressure_pa = 101325.0 }
    $alt0 = (Invoke-RtIpc world.get_atmosphere).altitude
    $s0 = Invoke-RtIpc world.sample_climate @{ position = @(0.0, (0.0 - $alt0), 0.0) }
    $s1 = Invoke-RtIpc world.sample_climate @{ position = @(0.0, (1000.0 - $alt0), 0.0) }
    Check ([math]::Abs($s0.temperature_k - 288.15) -lt 1e-2) ("T(0 m) {0} == 288.15" -f $s0.temperature_k)
    Check ([math]::Abs(($s0.temperature_k - $s1.temperature_k) - 6.5) -lt 1e-2) ("T(0)-T(1000) {0} == 6.5 K" -f ($s0.temperature_k - $s1.temperature_k))
    # ISA reference: p(1000 m) = 89875 Pa, rho(0) = 1.225 kg/m3
    Check ([math]::Abs($s1.pressure_pa - 89875.0) -lt 60.0) ("p(1000 m) {0} ~= 89875 Pa" -f $s1.pressure_pa)
    Check ([math]::Abs($s0.air_density_kg_m3 - 1.225) -lt 2e-3) ("rho(0 m) {0} ~= 1.225" -f $s0.air_density_kg_m3)

    # ── 3. Emekli anahtar ─────────────────────────────────────────────────
    Write-Host "3. retired keys on world.set_atmosphere"
    $rejected = $false
    try { $null = Invoke-RtIpc world.set_atmosphere @{ humidity = 0.5 } } catch { $rejected = $true }
    Check $rejected "world.set_atmosphere humidity -> error (not silently ignored)"

    # ── 4. Red, kirpma degil ──────────────────────────────────────────────
    Write-Host "4. invalid input is rejected, not clamped"
    $before = Invoke-RtIpc world.get_climate
    $rejected = $false
    try { $null = Invoke-RtIpc world.set_climate @{ surface_relative_humidity = -0.2 } } catch { $rejected = $true }
    $after = Invoke-RtIpc world.get_climate
    Check ($rejected -and $after.surface_relative_humidity -eq $before.surface_relative_humidity) "RH -0.2 rejected, value unchanged"
    $rejected = $false
    try { $null = Invoke-RtIpc world.set_climate @{ wind_direction = @(0.0, 1.0, 0.0) } } catch { $rejected = $true }
    Check $rejected "vertical-only wind direction rejected"
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
}

if ($fails -eq 0) { Write-Host "PASS: climate authority" -ForegroundColor Green }
else { Write-Host "FAIL: $fails check(s)" -ForegroundColor Red; exit 1 }
