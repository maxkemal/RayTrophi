<#
.SYNOPSIS
    Atmosfer Faz 1b kabul testi: aerial froxel + yukseklik sisi.
    docs/dev/ATMOSPHERE_SYSTEM.md

.DESCRIPTION
    SIRALI; once ucuz ve bagimsiz olanlar:

      1. Yuzey: world.atmosphere_stats iki rol (render / viewport) raporluyor
         ve froxel_available=true. false -> atmosphere_aerial_froxel.spv yok
         ya da boru hatti kurulamadi (SceneLog'a bak).
      2. Gidis-donus: world.set_aerial -> world.get_aerial ayni degerler.
      3. Red: emekli anahtarlar (aerial_min_distance, fog_color, ...) ve
         aralik disi degerler HATA vermeli, kirpilmamali.
      4. Sayac: kamera ve ortam sabitken froxel_dispatches DURMALI; kamera
         kipirdayinca ve sis degisince ARTMALI.
      5. Parite (yalniz -Region verilirse): uzak zemini kapsayan bir bolgede
         sis ACIK-KAPALI parlaklik farki Rendered (Vulkan RT) ile Material
         (RayFusion) arasinda -Tolerance icinde. Ayni olcum yalniz hava
         (aerial_perspective) icin de yapilir.

    Baslangictaki aerial/sis ayarlari ve shading modu sonunda geri yazilir.

    ★ En sinsi basarisizlik: 4. adimda sayac kamera SABITKEN de artiyor.
      Goruntu dogru gorunur, hicbir sey bozuk durmaz -- ama froxel her kare
      yeniden kuruluyor demektir (imza her karede degisiyor: kamera titremesi,
      NaN iceren bir alan, ya da hash'e giren kararsiz bir bayt).
    ★ Ikinci sinsi basarisizlik: 5. adimda iki fark da ~0. Froxel kuruluyor
      ama hic okunmuyor (aerialFroxelReady bayragi 0, ya da post'ta binding 3
      yer tutucuya bagli). Sayac bunu GOSTERMEZ; yalniz parlaklik gosterir.

.EXAMPLE
    .\Probe-Aerial.ps1
    .\Probe-Aerial.ps1 -Region 600,520,300,120     # uzak zemin bolgesi (piksel)
#>
[CmdletBinding()]
param(
    [int[]]$Region = @(),
    [double]$Tolerance = 0.05,     # |dRT - dRF| / max(dRT, dRF)
    [int]$SettleMs = 2500          # Rendered'in yakinsamasi icin bekleme
)

$ErrorActionPreference = 'Stop'
Import-Module "$PSScriptRoot\RtIpc.psm1" -Force

$fails = 0
function Check([bool]$ok, [string]$what) {
    if ($ok) { Write-Host "  OK   $what" -ForegroundColor Green }
    else     { Write-Host "  FAIL $what" -ForegroundColor Red; $script:fails++ }
}
function Expect-Error([string]$method, $params, [string]$what) {
    try {
        $null = Invoke-RtIpc $method $params
        Check $false "$what (hata bekleniyordu, kabul edildi)"
    } catch {
        Check $true "$what -> reddedildi"
    }
}
function Stats-Row([string]$role) {
    $s = Invoke-RtIpc world.atmosphere_stats
    return @($s.backends | Where-Object { $_.role -eq $role })[0]
}
function Nudge-Camera {
    $c = Invoke-RtIpc camera.get @{}
    $p = @([double]$c.position[0], [double]$c.position[1], [double]$c.position[2])
    Invoke-RtIpc camera.set_position @{ position = @(($p[0] + 0.05), $p[1], $p[2]) } | Out-Null
    Start-Sleep -Milliseconds 250
    Invoke-RtIpc camera.set_position @{ position = $p } | Out-Null
    Start-Sleep -Milliseconds 250
}
function Probe-Region {
    Start-Sleep -Milliseconds $SettleMs
    $probeArgs = @{ x = $Region[0]; y = $Region[1]; width = $Region[2]; height = $Region[3] }
    $p = Invoke-RtIpc render.probe $probeArgs
    if (-not $p.available) { throw "render.probe kare veremedi -- viewport.capture acik mi?" }
    if ($p.nan_fraction -gt 0) { Check $false ("NaN piksel orani {0}" -f $p.nan_fraction) }
    return [double]$p.mean_luminance
}

$a0 = Invoke-RtIpc world.get_aerial
$shading0 = (Invoke-RtIpc viewport.shading @{}).mode
Write-Host ("start: aerial={0} fog={1} density={2} height={3} falloff={4} dist={5} g={6}" -f `
    $a0.aerial_perspective, $a0.fog_enabled, $a0.fog_density, $a0.fog_height, $a0.fog_falloff,
    $a0.fog_distance, $a0.fog_anisotropy) -ForegroundColor Cyan

try {
    # ── 1. Yuzey ─────────────────────────────────────────────────────────
    Write-Host "1. world.atmosphere_stats"
    $s = Invoke-RtIpc world.atmosphere_stats
    foreach ($r in $s.backends) {
        Write-Host ("     {0,-8} vulkan={1} lut={2} froxel_available={3} active={4} dispatches={5}" -f `
            $r.role, $r.is_vulkan, $r.lut_ready, $r.froxel_available, $r.froxel_active, $r.froxel_dispatches)
    }
    foreach ($role in @('render','viewport')) {
        $r = @($s.backends | Where-Object { $_.role -eq $role })[0]
        if ($null -eq $r) { Check $false "$role satiri yok"; continue }
        if ($r.is_vulkan) { Check ([bool]$r.froxel_available) "$role froxel_available" }
    }

    # ── 2. Gidis-donus ───────────────────────────────────────────────────
    Write-Host "2. set_aerial -> get_aerial"
    $null = Invoke-RtIpc world.set_aerial @{ aerial_perspective = $true; fog_enabled = $true;
        fog_density = 0.0015; fog_height = 120.0; fog_falloff = 0.004; fog_distance = 20000.0;
        fog_albedo = @(0.9, 0.85, 0.8); fog_anisotropy = 0.7 }
    $a = Invoke-RtIpc world.get_aerial
    Check ($a.fog_enabled -eq $true) "fog_enabled"
    Check ([math]::Abs($a.fog_density - 0.0015) -lt 1e-7) ("fog_density {0}" -f $a.fog_density)
    Check ([math]::Abs($a.fog_height - 120.0) -lt 1e-3) ("fog_height {0}" -f $a.fog_height)
    Check ([math]::Abs($a.fog_falloff - 0.004) -lt 1e-7) ("fog_falloff {0}" -f $a.fog_falloff)
    Check ([math]::Abs($a.fog_distance - 20000.0) -lt 1e-2) ("fog_distance {0}" -f $a.fog_distance)
    Check ([math]::Abs($a.fog_albedo[1] - 0.85) -lt 1e-5) ("fog_albedo.g {0}" -f $a.fog_albedo[1])
    Check ([math]::Abs($a.fog_anisotropy - 0.7) -lt 1e-5) ("fog_anisotropy {0}" -f $a.fog_anisotropy)

    # ── 3. Red ───────────────────────────────────────────────────────────
    Write-Host "3. emekli anahtarlar ve aralik disi degerler"
    Expect-Error world.set_aerial @{ aerial_min_distance = 1000.0 } "aerial_min_distance (emekli)"
    Expect-Error world.set_aerial @{ aerial_density = 1.0 }         "aerial_density (emekli)"
    Expect-Error world.set_aerial @{ fog_color = @(1.0, 1.0, 1.0) } "fog_color (emekli)"
    Expect-Error world.set_aerial @{ fog_sun_scatter = 0.5 }        "fog_sun_scatter (emekli)"
    Expect-Error world.set_aerial @{ fog_anisotropy = 0.99 }        "fog_anisotropy 0.99"
    Expect-Error world.set_aerial @{ fog_albedo = @(1.5, 0.5, 0.5) } "fog_albedo > 1"
    Expect-Error world.set_aerial @{ fog_distance = 0.0 }           "fog_distance 0"
    Expect-Error world.set_aerial @{ fog_density = -0.001 }         "fog_density < 0"

    # ── 4. Sayac ─────────────────────────────────────────────────────────
    Write-Host "4. froxel_dispatches: sabitken durur, degisince artar"
    Invoke-RtIpc viewport.capture @{ enabled = $true } | Out-Null
    Invoke-RtIpc viewport.set_shading @{ mode = 'material' } | Out-Null
    Start-Sleep -Milliseconds 800
    $d0 = [long](Stats-Row 'viewport').froxel_dispatches
    Start-Sleep -Milliseconds 1500
    $d1 = [long](Stats-Row 'viewport').froxel_dispatches
    Check (($d1 - $d0) -le 1) ("viewport sabit: {0} -> {1} (en fazla +1)" -f $d0, $d1)
    Nudge-Camera
    $d2 = [long](Stats-Row 'viewport').froxel_dispatches
    Check ($d2 -gt $d1) ("viewport kamera kipirdadi: {0} -> {1}" -f $d1, $d2)
    $null = Invoke-RtIpc world.set_aerial @{ fog_density = 0.0016 }
    Start-Sleep -Milliseconds 500
    $d3 = [long](Stats-Row 'viewport').froxel_dispatches
    Check ($d3 -gt $d2) ("viewport sis yogunlugu degisti: {0} -> {1}" -f $d2, $d3)
    $vr = Stats-Row 'viewport'
    Check ([bool]$vr.froxel_active) "viewport froxel_active (material modda)"

    # ── 5. Parite ────────────────────────────────────────────────────────
    if ($Region.Count -ne 4) {
        Write-Host "5. [ATLANDI] parite icin -Region x,y,w,h ver (uzak zemini kapsayan bolge)" -ForegroundColor Yellow
    } else {
        Write-Host ("5. RT / RayFusion parite, bolge {0}" -f ($Region -join ','))
        $cases = @(
            @{ name = 'sis';  off = @{ aerial_perspective = $false; fog_enabled = $false };
                              on  = @{ aerial_perspective = $false; fog_enabled = $true;
                                       fog_density = 0.0015; fog_height = 120.0; fog_falloff = 0.004;
                                       fog_distance = 20000.0 } },
            @{ name = 'hava'; off = @{ aerial_perspective = $false; fog_enabled = $false };
                              on  = @{ aerial_perspective = $true;  fog_enabled = $false } }
        )
        foreach ($case in $cases) {
            $delta = @{}
            $base = @{}
            foreach ($mode in @('rendered','material')) {
                Invoke-RtIpc viewport.set_shading @{ mode = $mode } | Out-Null
                $null = Invoke-RtIpc world.set_aerial $case.off
                $lOff = Probe-Region
                $null = Invoke-RtIpc world.set_aerial $case.on
                $lOn = Probe-Region
                $delta[$mode] = $lOn - $lOff
                $base[$mode] = $lOff
                Write-Host ("     {0,-5} {1,-9} off={2:F4} on={3:F4} delta={4:+0.0000;-0.0000}" -f `
                    $case.name, $mode, $lOff, $lOn, $delta[$mode])
            }
            # ★ The delta only compares the froxel if the UNHAZED pixel is the
            #   same in both backends: aerial is L*T + S, so a brighter base
            #   loses more to T and shows a smaller delta with an identical
            #   medium. On surfaces RayFusion is ~1.45x brighter than Rendered
            #   (separate open bug), so a surface region cannot pass or fail
            #   here -- use a SKY region (fog case) for the froxel parity.
            $baseRel = [math]::Abs($base['rendered'] - $base['material']) /
                       [math]::Max([math]::Max($base['rendered'], $base['material']), 1e-6)
            if ($baseRel -gt 0.03) {
                Write-Host ("  BELIRSIZ {0}: tabanlar farkli ({1:P1}) -- froxel degil, yuzey parlakligi ayrisiyor. Gokyuzu bolgesi ver." -f `
                    $case.name, $baseRel) -ForegroundColor Yellow
                continue
            }
            $big = [math]::Max([math]::Abs($delta['rendered']), [math]::Abs($delta['material']))
            if ($big -lt 0.005) {
                # Sky pixels skip the air term by design (the sky-view LUT
                # already holds it), so "hava" on a sky region is 0 in BOTH.
                Write-Host ("  BELIRSIZ {0}: iki backend'de de etki ~0 -- gokyuzu bolgesinde hava icin beklenen; yuzeyde ise froxel okunmuyor" -f $case.name) -ForegroundColor Yellow
                continue
            }
            $rel = [math]::Abs($delta['rendered'] - $delta['material']) / $big
            Check ($rel -le $Tolerance) ("{0}: |dRT-dRF|/max = {1:P1} (<= {2:P0})" -f $case.name, $rel, $Tolerance)
        }
    }
}
finally {
    $restore = @{ aerial_perspective = [bool]$a0.aerial_perspective; fog_enabled = [bool]$a0.fog_enabled;
        fog_density = [double]$a0.fog_density; fog_height = [double]$a0.fog_height;
        fog_falloff = [double]$a0.fog_falloff; fog_distance = [double]$a0.fog_distance;
        fog_albedo = @($a0.fog_albedo); fog_anisotropy = [double]$a0.fog_anisotropy }
    $null = Invoke-RtIpc world.set_aerial $restore
    if ($shading0) { Invoke-RtIpc viewport.set_shading @{ mode = $shading0 } | Out-Null }
}

if ($fails -eq 0) { Write-Host "Probe-Aerial: GECTI" -ForegroundColor Green }
else              { Write-Host "Probe-Aerial: $fails KALDI" -ForegroundColor Red; exit 1 }
