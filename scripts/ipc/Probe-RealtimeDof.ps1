<#
.SYNOPSIS
  Realtime (raster) alan derinliginin GERCEKTEN bulanistirdigini ve DOGRU
  yerde bulanistirdigini dogrular.

.DESCRIPTION
  ★★★★ Bu betigin varlik sebebi: "DoF acik" demek "DoF calisiyor" demek
  DEGILDIR. Dort kapi ust uste: ayar, shading modu, kamera tipi ve LENS. Hepsi
  gecilmeden goruntude hicbir sey olmaz, ve olmadigini gormek icin bakmak
  gerekir. `viewport.get_depth_of_field` bu yuzden `active` ve
  `inactive_reason` doner -- "kapali" ile "olculemedi" ayni sey degildir.

  Olcum: `render.probe` bir bolgenin min/mean/max parlakligini verir.
  Bulaniklik KONTRASTI (max-min) dusurur ama ORTALAMAYI korur. Ikisini
  birlikte olcmek sart:
    - yalnizca kontrast dususune bakmak, karartmayi da "bulaniklik" sayardi;
    - yalnizca ortalamaya bakmak hicbir sey soylemezdi.

  ★★★ Ve DoF "her seyi bulanistirmak" degildir: odaktaki nesne KESKIN kalmali.
  5. kapi tam olarak bunu olcer -- odak mesafesini ozneye verip kontrastin
  KORUNDUGUNU, sonra uzaga verip DUSTUGUNU gorur. Bu kapi olmadan, ekrani
  komple bulanistiran bozuk bir uygulama da "GECTI" derdi.

.NOTES
  ★ -Region ZORUNLU gibidir: bolge bir KENAR icermeli. Duz bir zemin parcasinda
    kontrast zaten sifirdir ve betik bunu "OLCULEMEDI" diye raporlar, "GECTI"
    demez.
  ★ Material shading modunda kosar; Rendered (path tracer) kendi lensini
    kullanir ve bu ayardan etkilenmez.
  ★★ KAMERANIN kendi DoF anahtari ayri bir kapidir; anahtarin kendisini
    Probe-CameraDofSwitch.ps1 olcer. Bu betik onu yalnizca ACAR.
#>
[CmdletBinding()]
param(
    [int[]]$Region = @(),
    [double]$SubjectDistance = 0.0,   # 0 = kameranin mevcut odak mesafesi
    [double]$Tolerance = 0.02
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'RtIpc.psm1') -Force

$fails = 0
function Check($name, $ok, $detail) {
    if ($ok) { Write-Host ("  [GECTI] {0} {1}" -f $name, $detail) }
    else     { Write-Host ("  [KALDI] {0} {1}" -f $name, $detail) -ForegroundColor Red; $script:fails++ }
}

$script:CamBase = $null
function Force-Frame {
    param([int]$TimeoutMs = 4000)
    if ($null -eq $script:CamBase) {
        $c = Invoke-RtIpc camera.get @{}
        $script:CamBase = @([double]$c.position[0], [double]$c.position[1], [double]$c.position[2])
    }
    $b = $script:CamBase
    $before = Invoke-RtIpc viewport.frame_telemetry @{}
    Invoke-RtIpc camera.set_position @{ position = @(($b[0] + 0.01), $b[1], $b[2]) } | Out-Null
    Invoke-RtIpc camera.set_position @{ position = @($b[0], $b[1], $b[2]) } | Out-Null
    if (-not $before.available) { Start-Sleep -Milliseconds 900; return $true }
    $sw = [Diagnostics.Stopwatch]::StartNew()
    while ($sw.ElapsedMilliseconds -lt $TimeoutMs) {
        Start-Sleep -Milliseconds 120
        $after = Invoke-RtIpc viewport.frame_telemetry @{}
        if ($after.frames_submitted -gt $before.frames_submitted) { Start-Sleep -Milliseconds 150; return $true }
    }
    return $false
}

function Probe-Region {
    if (-not (Force-Frame)) { throw "Viewport yeni kare cizmedi -- olcum BAYAT olurdu." }
    $a = @{}
    if ($Region.Count -eq 4) { $a = @{ x=$Region[0]; y=$Region[1]; width=$Region[2]; height=$Region[3] } }
    $p = Invoke-RtIpc render.probe $a
    if (-not $p.available) { throw "render.probe kare veremedi -- viewport.capture acik mi?" }
    return [pscustomobject]@{
        Mean     = [double]$p.mean_luminance
        Contrast = [double]$p.max_luminance - [double]$p.min_luminance
    }
}

$cam0 = Invoke-RtIpc camera.get @{}
$dof0 = Invoke-RtIpc viewport.get_depth_of_field @{}
Invoke-RtIpc viewport.capture @{ enabled = $true } | Out-Null
Invoke-RtIpc viewport.set_shading @{ mode = 'material' } | Out-Null

$subject = if ($SubjectDistance -gt 0.0) { $SubjectDistance } else { [double]$cam0.focus_distance }
if ($subject -le 0.0) { $subject = 5.0 }

try {
    # ── 1. ★★★ KAPI RAPORU: "kapali" ile "olculemedi" ayrilmali ─────────────
    Write-Host "1. lens kapaliyken active=false ve SEBEBI yazili olmali"
    Invoke-RtIpc viewport.set_depth_of_field @{ enabled = $true; max_coc_pixels = 24.0; max_taps = 32 } | Out-Null
    # ★★★ Kamera anahtari AYRI bir kapidir (2026-09-06 II). Once onu acmazsak
    #   asagidaki "aciklik 0" kapisina hic ulasilmaz ve betik yanlis kapiyi
    #   dogruladigini sanirdi -- olcu aletinin kendi korlugu.
    Invoke-RtIpc camera.set_depth_of_field @{ enabled = $true } | Out-Null
    Invoke-RtIpc camera.set_aperture @{ aperture = 0.0 } | Out-Null
    $d = Invoke-RtIpc viewport.get_depth_of_field @{}
    Check "enabled dogru raporlaniyor" ($d.enabled -eq $true) ("enabled={0}" -f $d.enabled)
    Check "active=false" ($d.active -eq $false) ("active={0}" -f $d.active)
    Check "sebep bos degil" (-not [string]::IsNullOrWhiteSpace([string]$d.inactive_reason)) `
          ("reason='{0}'" -f $d.inactive_reason)

    # ── 2. ★★ Gecersiz deger REDDEDILMELI, sessizce kirpilmamali ────────────
    Write-Host "2. gecersiz maliyet tavanlari reddedilmeli"
    $rejected = $false
    try { Invoke-RtIpc viewport.set_depth_of_field @{ enabled = $true; max_coc_pixels = 999.0; max_taps = 32 } | Out-Null }
    catch { $rejected = $true }
    Check "max_coc_pixels=999 reddedildi" $rejected "kirpma DEGIL reddetme"
    $rejected = $false
    try { Invoke-RtIpc viewport.set_depth_of_field @{ enabled = $true; max_coc_pixels = 24.0; max_taps = 4 } | Out-Null }
    catch { $rejected = $true }
    Check "max_taps=4 reddedildi" $rejected "alt sinir 8"

    # ── 3. ★★★ KORLUK KALIBRASYONU: bolgede olculecek KENAR var mi ──────────
    Write-Host "3. olcum bolgesi bir kenar iceriyor mu"
    Invoke-RtIpc viewport.set_depth_of_field @{ enabled = $false; max_coc_pixels = 24.0; max_taps = 32 } | Out-Null
    Invoke-RtIpc camera.set_aperture @{ aperture = 0.0 } | Out-Null
    $sharp = Probe-Region
    Write-Host ("     keskin: mean={0:F4} contrast={1:F4}" -f $sharp.Mean, $sharp.Contrast)
    $measurable = $sharp.Contrast -gt 0.15
    Check "kontrast olculebilir" $measurable ("contrast={0:F4}, gereken > 0.15" -f $sharp.Contrast)
    if (-not $measurable) {
        Write-Host "     ★ Bolge duz gorunuyor. -Region x,y,w,h ile bir KENARIN uzerine otur." -ForegroundColor Yellow
        Write-Host "       Bu kapi kalirken asagisinin GECMESI hicbir sey soylemez." -ForegroundColor Yellow
    }

    # ── 4. ★★★ Odakta KESKIN kalmali ────────────────────────────────────────
    Write-Host "4. odak ozneye ayarliyken bulaniklik OLMAMALI"
    Invoke-RtIpc viewport.set_depth_of_field @{ enabled = $true; max_coc_pixels = 24.0; max_taps = 32 } | Out-Null
    Invoke-RtIpc camera.set_depth_of_field @{ enabled = $true } | Out-Null
    Invoke-RtIpc camera.set_focus_distance @{ focus_distance = $subject } | Out-Null
    Invoke-RtIpc camera.set_aperture @{ aperture = 0.25 } | Out-Null
    $focused = Probe-Region
    $d = Invoke-RtIpc viewport.get_depth_of_field @{}
    Write-Host ("     odakta: mean={0:F4} contrast={1:F4} active={2}" -f $focused.Mean, $focused.Contrast, $d.active)
    Check "active=true" ($d.active -eq $true) ("reason='{0}'" -f $d.inactive_reason)
    if ($measurable) {
        Check "odaktaki kontrast korunuyor" `
              ($focused.Contrast -gt ($sharp.Contrast * 0.75)) `
              ("{0:F4} -> {1:F4}" -f $sharp.Contrast, $focused.Contrast)
    }

    # ── 5. ★★★★ Odak UZAKLASINCA bulanmali, ama KARARMAMALI ─────────────────
    # Kontrast dusup ortalama korunuyorsa bu bir BULANIKLIKTIR. Ortalama da
    # dusuyorsa yaptigimiz sey bulaniklik degil karartmadir -- ve o, "DoF
    # calisiyor" diye raporlanan en sinsi yanlis olurdu.
    Write-Host "5. odak uzaklasinca kontrast DUSMELI, ortalama KORUNMALI"
    Invoke-RtIpc camera.set_focus_distance @{ focus_distance = ($subject * 12.0 + 50.0) } | Out-Null
    $blurred = Probe-Region
    Write-Host ("     bulanik: mean={0:F4} contrast={1:F4}" -f $blurred.Mean, $blurred.Contrast)
    if ($measurable) {
        Check "kontrast dustu" ($blurred.Contrast -lt ($focused.Contrast * 0.85)) `
              ("{0:F4} -> {1:F4}" -f $focused.Contrast, $blurred.Contrast)
        Check "ortalama korundu (karartma DEGIL)" `
              ([Math]::Abs($blurred.Mean - $focused.Mean) -lt ([Math]::Max($focused.Mean, 0.05) * 0.25)) `
              ("{0:F4} -> {1:F4}" -f $focused.Mean, $blurred.Mean)
    } else {
        Write-Host "  [ATLANDI] bulaniklik kapilari -- kalibrasyon kapisi kalmisti." -ForegroundColor Yellow
    }

    # ── 6. ★★ Kapatinca geri donmeli (ayarin gercekten kapisi mi) ───────────
    Write-Host "6. ayari kapatmak bulanikligi KALDIRMALI"
    Invoke-RtIpc viewport.set_depth_of_field @{ enabled = $false; max_coc_pixels = 24.0; max_taps = 32 } | Out-Null
    $off = Probe-Region
    $d = Invoke-RtIpc viewport.get_depth_of_field @{}
    Write-Host ("     kapali: mean={0:F4} contrast={1:F4} reason='{2}'" -f $off.Mean, $off.Contrast, $d.inactive_reason)
    Check "active=false" ($d.active -eq $false) ("reason='{0}'" -f $d.inactive_reason)
    if ($measurable) {
        Check "kontrast geri geldi" ($off.Contrast -gt ($blurred.Contrast * 1.1)) `
              ("{0:F4} -> {1:F4}" -f $blurred.Contrast, $off.Contrast)
    }
}
finally {
    Invoke-RtIpc camera.set_depth_of_field @{ enabled = [bool]$cam0.depth_of_field } | Out-Null
    Invoke-RtIpc camera.set_aperture @{ aperture = [double]$cam0.aperture } | Out-Null
    Invoke-RtIpc camera.set_focus_distance @{ focus_distance = [double]$cam0.focus_distance } | Out-Null
    Invoke-RtIpc viewport.set_depth_of_field @{
        enabled = [bool]$dof0.enabled
        max_coc_pixels = [double]$dof0.max_coc_pixels
        max_taps = [int]$dof0.max_taps } | Out-Null
    if ($script:CamBase) {
        Invoke-RtIpc camera.set_position @{ position = @($script:CamBase[0], $script:CamBase[1], $script:CamBase[2]) } | Out-Null
    }
    Invoke-RtIpc viewport.capture @{ enabled = $false } | Out-Null
}

Write-Host ""
if ($fails -eq 0) { Write-Host "TUM KAPILAR GECTI" -ForegroundColor Green }
else { Write-Host ("{0} KAPI KALDI" -f $fails) -ForegroundColor Red; exit 1 }
