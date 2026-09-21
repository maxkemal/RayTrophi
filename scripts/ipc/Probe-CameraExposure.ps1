<#
.SYNOPSIS
  Fiziksel pozlama ucgeninin (ISO / enstantane / diyafram) gercekten ve DOGRU
  yonde calistigini, ve KAPISININ post zincirinde oldugunu dogrular.

.DESCRIPTION
  Model GORELIDIR: carpan referans ucgene gore hesaplanir. Mutlak fotometrik
  formul BU MOTORDA KULLANILAMAZ -- isik siddeti keyfi birimde ve mutlak formul
  her sahneyi karartir (~2.1e-4). Bu betik tam olarak bunu bekler: bir DURAK
  (stop) = 2x.

  ★★★★ KAPI 2026-09-06'da TASINDI. Pozlama modunun sahibi artik kamera degil
  POST ZINCIRIDIR (`post.get_exposure` -> mode):
    manual         : yalnizca post'un kendi EV'si; kamera terimi 1.0
    physical       : ISO / enstantane / f-stop OKUNUR
    auto_histogram : HDR karesinden olculur
  Kameranin `auto_exposure` / `use_physical_exposure` bayraklari MIRASTIR:
  `rtpost::syncDisplay` physical modda ikisini de zorlar, yani onlari
  cevirmek goruntuyu DEGISTIRMEZ. Viewport'taki pozlama ucgeni ve kamera
  paneli artik modu surer; bu betik onlarin surdugu yolu surer.

  Kapilar:
    1. ★★★ KAPININ VARLIGI: manual modda kadranlar goruntuye DOKUNMAMALI ve
       `camera.get.exposure_factor` 1.0 olmali (UYGULANAN carpandir, ayar degil);
    2. turetilmis alanlar preset indeksini cozuyor (indeks olcum degildir);
    3. ★★ duyarlilik: post EV +1 olculebiliyor mu (yoksa 4-5. kapilar
       korlukten gecer);
    4. ★★★ physical modda ISO yukseltmek parlatmali VE bildirilen carpanla
       olculen yon AYNI olmali;
    5. ★★★★ MODUN KENDISI: ayni kadranlarla manual -> physical gecisi
       carpani 1.0'dan cikarmali. Bu, HUD ucgeninin ve panelin surdugu tek
       anahtardir; kalirsa "kadran cevirdim, hicbir sey olmadi" geri gelir.

.NOTES
  ★ -Region ver: tam kare cogu sahnede ozneyi olcmez.
#>
[CmdletBinding()]
param(
    [int[]]$Region = @(),
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

function Probe-Mean {
    if (-not (Force-Frame)) { throw "Viewport yeni kare cizmedi -- olcum BAYAT olurdu." }
    $a = @{}
    if ($Region.Count -eq 4) { $a = @{ x=$Region[0]; y=$Region[1]; width=$Region[2]; height=$Region[3] } }
    $p = Invoke-RtIpc render.probe $a
    if (-not $p.available) { throw "render.probe kare veremedi -- viewport.capture acik mi?" }
    return [double]$p.mean_luminance
}

function Set-Presets($isoIdx, $shIdx, $fsIdx) {
    Invoke-RtIpc camera.set_iso_preset     @{ index = [int]$isoIdx } | Out-Null
    Invoke-RtIpc camera.set_shutter_preset @{ index = [int]$shIdx }  | Out-Null
    Invoke-RtIpc camera.set_fstop_preset   @{ index = [int]$fsIdx }  | Out-Null
}

# ★★★ Modun sahibi burasi. Panel, HUD ucgeni ve bu betik AYNI cagriyi kullanir.
function Set-ExposureMode($mode) {
    Invoke-RtIpc post.configure_exposure @{ settings = @{ mode = $mode } } | Out-Null
}
function Set-PostEv($ev) {
    Invoke-RtIpc post.configure_exposure @{ settings = @{ ev = [double]$ev } } | Out-Null
}
function Get-Cam  { Invoke-RtIpc camera.get @{} }
function Get-Post { Invoke-RtIpc post.get_exposure @{} }

$cam0  = Get-Cam
$post0 = Get-Post
Invoke-RtIpc viewport.capture @{ enabled = $true } | Out-Null
Invoke-RtIpc viewport.set_shading @{ mode = 'material' } | Out-Null

try {
    # ── 1. ★★★ KAPININ VARLIGI: manual modda kadranlar OKUNMAZ ─────────────
    # Kadranlarin "olu" gorunmesinin en sik sebebi budur ve ariza DEGILDIR.
    # Dogrulanmazsa 4. kapinin sifiri hangi sebepten geldigi bilinemez.
    Write-Host "1. manual modda preset'ler goruntuye dokunmamali"
    Set-ExposureMode 'manual'
    Set-PostEv 0.0
    Set-Presets 1 1 4
    $a1 = Probe-Mean
    Set-Presets 5 5 1
    $a2 = Probe-Mean
    $g = Get-Cam
    Check "manual'de UYGULANAN carpan 1.0" ([Math]::Abs($g.exposure_factor - 1.0) -lt 1e-4) `
          ("exposure_factor={0:F5}" -f $g.exposure_factor)
    Check "manual'de goruntu sabit" ([Math]::Abs($a2 - $a1) -lt $Tolerance) ("{0:F4} -> {1:F4}" -f $a1, $a2)

    # ── 2. Turetilmis alanlar preset indeksini COZUYOR mu ──────────────────
    # ★ Indeks bir olcum degildir; ajanin "hangi ISO" sorusuna cevabi bu.
    Write-Host "2. turetilmis alanlar indeksi cozuyor mu"
    Set-Presets 1 1 4
    $r1 = Get-Cam
    Set-Presets 5 5 1
    $r2 = Get-Cam
    Write-Host ("     idx1: ISO {0} / {1:F5}s / f{2}" -f $r1.iso_value, $r1.shutter_seconds, $r1.f_number)
    Write-Host ("     idx5: ISO {0} / {1:F5}s / f{2}" -f $r2.iso_value, $r2.shutter_seconds, $r2.f_number)
    Check "iso_value degisti" ($r1.iso_value -ne $r2.iso_value) "indeks farkli deger cozmeli"
    Check "shutter_seconds degisti" ($r1.shutter_seconds -ne $r2.shutter_seconds) "indeks farkli deger cozmeli"

    # ── 3. ★★ DUYARLILIK: bu kurulum bir duragi gorebiliyor mu ─────────────
    # ★★★ Sinyal POST'un EV'si: o HER modda uygulanir. Kameranin
    #   `ev_compensation`i artik yalnizca physical modda okunur, yani
    #   duyarlilik kalibrasyonu icin KULLANILAMAZ -- kullanilsaydi bu kapi
    #   "olcemiyorum" derdi ve sebebi kalibrasyon degil KAPI olurdu.
    Write-Host "3. duyarlilik: post EV +1 olculebiliyor mu"
    Set-PostEv 0.0
    $e0 = Probe-Mean
    Set-PostEv 1.0
    $e1 = Probe-Mean
    Set-PostEv 0.0
    $delta = $e1 - $e0
    Write-Host ("     EV 0 -> +1 : {0:F4} -> {1:F4}   delta={2:F4}" -f $e0, $e1, $delta)
    $sensitive = $delta -gt (2.0 * $Tolerance)
    Check "duyarlilik" $sensitive ("delta={0:F4}, gereken > {1:F4}" -f $delta, (2.0*$Tolerance))
    if (-not $sensitive) {
        Write-Host "     ★ Bolge ozneyi kapsamiyor olabilir. -Region x,y,w,h ver." -ForegroundColor Yellow
        Write-Host "       Bu kapi kalirken 4-5. kapilarin GECMESI hicbir sey soylemez." -ForegroundColor Yellow
    }

    # ── 4. ★★★ physical mod: ISO yukseltmek PARLATMALI ─────────────────────
    Write-Host "4. physical modda ISO preset'i parlakligi surmeli"
    Set-ExposureMode 'physical'

    Set-Presets 1 1 4 ; $lowCam = Get-Cam ; $low = Probe-Mean
    Set-Presets 5 1 4 ; $hiCam  = Get-Cam ; $hi  = Probe-Mean
    Write-Host ("     ISO {0} (factor {1:F4}) mean={2:F4}" -f $lowCam.iso_value, $lowCam.exposure_factor, $low)
    Write-Host ("     ISO {0} (factor {1:F4}) mean={2:F4}" -f $hiCam.iso_value,  $hiCam.exposure_factor,  $hi)

    # ★★★ Kapi, olculen yon ile BILDIRILEN carpanin yonunun AYNI olmasi.
    #   Yalnizca "parladi mi" demek yetmez: carpan dusup goruntu parlarsa
    #   model ile goruntu ayrisiyordur ve o sessiz bir aksakliktir.
    $factorUp   = $hiCam.exposure_factor -gt $lowCam.exposure_factor
    $measuredUp = $hi -gt ($low + $Tolerance)
    Check "carpan ve goruntu AYNI yonde" ($factorUp -eq $measuredUp) `
          ("factor {0} / olculen {1}" -f $factorUp, $measuredUp)
    if ($sensitive) {
        Check "yuksek ISO daha parlak" ($factorUp -and $measuredUp) "ikisi de artmali"
    } else {
        Write-Host "  [ATLANDI] parlaklik kapisi -- duyarlilik kapisi kalmisti." -ForegroundColor Yellow
    }

    # ── 5. ★★★★ MODUN KENDISI ANAHTAR MI ───────────────────────────────────
    # HUD ucgeni ve kamera paneli tam olarak bu anahtari cevirir. Ayni
    # kadranlarla yalnizca modu degistirmek carpani KIMILDATMALI.
    Write-Host "5. modun kendisi: manual <-> physical ayni kadranlarla farkli carpan"
    Set-Presets 5 1 4
    Set-ExposureMode 'manual'   ; $mMan  = (Get-Cam).exposure_factor ; $vMan  = Probe-Mean
    Set-ExposureMode 'physical' ; $mPhys = (Get-Cam).exposure_factor ; $vPhys = Probe-Mean
    Write-Host ("     manual factor={0:F4} mean={1:F4} | physical factor={2:F4} mean={3:F4}" -f $mMan, $vMan, $mPhys, $vPhys)
    Check "manual carpani 1.0" ([Math]::Abs($mMan - 1.0) -lt 1e-4) ("{0:F5}" -f $mMan)
    Check "physical carpani 1.0 DEGIL" ([Math]::Abs($mPhys - 1.0) -gt 1e-3) ("{0:F5}" -f $mPhys)
    if ($sensitive) {
        Check "mod degisimi goruntude gorunuyor" ([Math]::Abs($vPhys - $vMan) -gt $Tolerance) `
              ("delta={0:F4}" -f [Math]::Abs($vPhys - $vMan))
    } else {
        Write-Host "  [ATLANDI] mod-goruntu kapisi -- duyarlilik kapisi kalmisti." -ForegroundColor Yellow
    }

    # ★ post.get_exposure ile camera.get AYNI sayiyi soylemeli: biri ayarin,
    #   digeri uygulananin aynasi; ayrisirlarsa panel yalan soyluyor demektir.
    $p = Get-Post
    $c = Get-Cam
    Check "post.camera_exposure == camera.exposure_factor" `
          ([Math]::Abs([double]$p.camera_exposure - [double]$c.exposure_factor) -lt 1e-4) `
          ("{0:F5} / {1:F5}" -f $p.camera_exposure, $c.exposure_factor)
}
finally {
    Set-Presets $cam0.iso_preset_index $cam0.shutter_preset_index $cam0.fstop_preset_index
    Invoke-RtIpc camera.set_ev_compensation @{ ev = [double]$cam0.ev_compensation } | Out-Null
    Invoke-RtIpc post.configure_exposure @{ settings = @{ mode = [string]$post0.mode; ev = [double]$post0.ev } } | Out-Null
    if ($script:CamBase) {
        Invoke-RtIpc camera.set_position @{ position = @($script:CamBase[0], $script:CamBase[1], $script:CamBase[2]) } | Out-Null
    }
    Invoke-RtIpc viewport.capture @{ enabled = $false } | Out-Null
}

Write-Host ""
if ($fails -eq 0) { Write-Host "TUM KAPILAR GECTI" -ForegroundColor Green }
else { Write-Host ("{0} KAPI KALDI" -f $fails) -ForegroundColor Red; exit 1 }
