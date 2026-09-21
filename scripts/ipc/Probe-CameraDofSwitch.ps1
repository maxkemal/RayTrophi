<#
.SYNOPSIS
  Alan derinligi ANAHTARININ gercekten bir anahtar oldugunu dogrular:
  kapanabildigini, kapanirken degeri YOK ETMEDIGINI, ve f-stop kadraninin
  onu zorla acmadigini.

.DESCRIPTION
  ★★★★ Bu betigin varlik sebebi somut bir arizadir (2026-09-06): tek
  kapali-anahtar `aperture == 0` idi. O bir DEGER degil bir SENTINEL'di --
  hicbir f-sayisi sifir aciklik uretmez. F-stop kadrani aciklaga baglanir
  baglanmaz DoF bir kez acildi ve BIR DAHA KAPANMADI, cunku geri donus icin
  ulasilamaz bir sayi gerekiyordu.

  ★★★ Bu yuzden asagidaki kapilarin en onemlisi 3. ve 4.: "kapanabiliyor mu"
  ve "kapatinca deger duruyor mu". Ikincisi olmadan anahtar bir SIFIRLAYICI
  olurdu ve kullanici her actiginda kadranlari yeniden ayarlardi -- kimsenin
  bug diye raporlamadigi, yalnizca "bu program sinir bozucu" denen tur.

.NOTES
  ★ Bu kapilar GORUNTU olcmez, SOZLESME olcer: hangi degerin hangi kapidan
    gectigini. Bulanikligin gercekten ekranda oldugunu Probe-RealtimeDof.ps1
    olcer; ikisi birlikte kosmali.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'RtIpc.psm1') -Force

$fails = 0
function Check($name, $ok, $detail) {
    if ($ok) { Write-Host ("  [GECTI] {0} {1}" -f $name, $detail) }
    else     { Write-Host ("  [KALDI] {0} {1}" -f $name, $detail) -ForegroundColor Red; $script:fails++ }
}

$cam0 = Invoke-RtIpc camera.get @{}

try {
    # ── 1. Anahtar RAPORLANIYOR mu ──────────────────────────────────────────
    Write-Host "1. camera.get anahtari ve UYGULANAN yaricapi raporlamali"
    Invoke-RtIpc camera.set_depth_of_field @{ enabled = $true } | Out-Null
    Invoke-RtIpc camera.set_aperture @{ aperture = 0.24 } | Out-Null
    $c = Invoke-RtIpc camera.get @{}
    Check "depth_of_field alani var" ($null -ne $c.depth_of_field) ("depth_of_field={0}" -f $c.depth_of_field)
    Check "effective_lens_radius = aperture/2" `
          ([Math]::Abs([double]$c.effective_lens_radius - 0.12) -lt 0.002) `
          ("beklenen 0.12, olculen {0:F4}" -f $c.effective_lens_radius)

    # ── 2. ★★★ F-STOP KADRANI ANAHTARI ZORLA ACMAZ ──────────────────────────
    # Kullanicinin belirtisi buydu: f-stop'a dokunmak DoF'u kalici olarak
    # acti. Artik f-sayisi ACIKLIGI yazar, ANAHTARI degil.
    Write-Host "2. f-stop presetini degistirmek DoF'u zorla ACMAMALI"
    Invoke-RtIpc camera.set_depth_of_field @{ enabled = $false } | Out-Null
    Invoke-RtIpc camera.set_fstop_preset @{ index = 3 } | Out-Null
    $c = Invoke-RtIpc camera.get @{}
    Check "anahtar hala kapali" ($c.depth_of_field -eq $false) ("depth_of_field={0}" -f $c.depth_of_field)
    Check "kapaliyken etkin yaricap 0" ([double]$c.effective_lens_radius -lt 1e-6) `
          ("effective_lens_radius={0}" -f $c.effective_lens_radius)
    Check "ama aciklik YAZILDI (f-sayisi fiziksel)" ([double]$c.aperture -gt 1e-5) `
          ("aperture={0:F4}" -f $c.aperture)

    # ── 3. ★★★★ PANEL ILE SCRIPT AYNI f-SAYISINI GORMELI ────────────────────
    # Eskiden preset yalnizca indeksi yazardi, panel ise f-sayisini aciklidan
    # turetirdi: ucgeni cevirince panelin sayisi kimildamiyordu.
    Write-Host "3. preset f-sayisi ile turetilmis f-sayisi ayrismamali"
    Invoke-RtIpc camera.set_fstop_preset @{ index = 5 } | Out-Null
    $c5 = Invoke-RtIpc camera.get @{}
    Invoke-RtIpc camera.set_fstop_preset @{ index = 8 } | Out-Null
    $c8 = Invoke-RtIpc camera.get @{}
    Check "preset degisince f_number degisti" `
          ([Math]::Abs([double]$c8.f_number - [double]$c5.f_number) -gt 0.05) `
          ("{0:F2} -> {1:F2}" -f $c5.f_number, $c8.f_number)
    Check "preset degisince aciklik da degisti" `
          ([Math]::Abs([double]$c8.aperture - [double]$c5.aperture) -gt 1e-5) `
          ("{0:F4} -> {1:F4}" -f $c5.aperture, $c8.aperture)
    Check "dar diyafram = kucuk aciklik" ([double]$c8.aperture -lt [double]$c5.aperture) `
          ("f/{0:F1} -> f/{1:F1}" -f $c5.f_number, $c8.f_number)

    # ── 4. ★★★★★ KAPATIP ACMAK DEGERI YOK ETMEMELI ─────────────────────────
    Write-Host "4. anahtari kapatip acmak ayni bulanikligi geri getirmeli"
    Invoke-RtIpc camera.set_depth_of_field @{ enabled = $true } | Out-Null
    Invoke-RtIpc camera.set_aperture @{ aperture = 0.3 } | Out-Null
    $on1 = Invoke-RtIpc camera.get @{}
    Invoke-RtIpc camera.set_depth_of_field @{ enabled = $false } | Out-Null
    $offc = Invoke-RtIpc camera.get @{}
    Invoke-RtIpc camera.set_depth_of_field @{ enabled = $true } | Out-Null
    $on2 = Invoke-RtIpc camera.get @{}
    Check "kapaliyken etkin yaricap 0" ([double]$offc.effective_lens_radius -lt 1e-6) `
          ("effective_lens_radius={0}" -f $offc.effective_lens_radius)
    Check "kapaliyken aciklik KORUNDU" ([Math]::Abs([double]$offc.aperture - 0.3) -lt 1e-4) `
          ("aperture={0:F4}" -f $offc.aperture)
    Check "acinca ayni yaricap geri geldi" `
          ([Math]::Abs([double]$on2.effective_lens_radius - [double]$on1.effective_lens_radius) -lt 1e-6) `
          ("{0:F4} -> {1:F4}" -f $on1.effective_lens_radius, $on2.effective_lens_radius)

    # ── 5. ★★★ KAPI RAPORU DOGRU KAPIYI SOYLEMELI ───────────────────────────
    # "aperture is 0" demek burada YANLIS teshis olurdu: aciklik dolu.
    Write-Host "5. viewport raporu kapali anahtari DOGRU adlandirmali"
    Invoke-RtIpc viewport.set_shading @{ mode = 'material' } | Out-Null
    Invoke-RtIpc viewport.set_depth_of_field @{ enabled = $true; max_coc_pixels = 24.0; max_taps = 32 } | Out-Null
    Invoke-RtIpc camera.set_depth_of_field @{ enabled = $false } | Out-Null
    $d = Invoke-RtIpc viewport.get_depth_of_field @{}
    Check "active=false" ($d.active -eq $false) ("active={0}" -f $d.active)
    Check "sebep KAMERA anahtarini gosteriyor" `
          ([string]$d.inactive_reason -match 'camera depth of field') `
          ("reason='{0}'" -f $d.inactive_reason)
    Check "camera_aperture ETKIN degeri raporluyor" ([double]$d.camera_aperture -lt 1e-5) `
          ("camera_aperture={0} (fiziksel deger degil)" -f $d.camera_aperture)

    # ── 6. Anahtar acikken kapi gercekten aciliyor mu ───────────────────────
    Write-Host "6. anahtar acikken kapi gecilmeli"
    Invoke-RtIpc camera.set_depth_of_field @{ enabled = $true } | Out-Null
    Invoke-RtIpc camera.set_focus_distance @{ focus_distance = 5.0 } | Out-Null
    $d = Invoke-RtIpc viewport.get_depth_of_field @{}
    Check "active=true" ($d.active -eq $true) ("reason='{0}'" -f $d.inactive_reason)
}
finally {
    Invoke-RtIpc camera.set_depth_of_field @{ enabled = [bool]$cam0.depth_of_field } | Out-Null
    Invoke-RtIpc camera.set_aperture @{ aperture = [double]$cam0.aperture } | Out-Null
    Invoke-RtIpc camera.set_focus_distance @{ focus_distance = [double]$cam0.focus_distance } | Out-Null
    Invoke-RtIpc camera.set_fstop_preset @{ index = [int]$cam0.fstop_preset_index } | Out-Null
}

Write-Host ""
if ($fails -eq 0) { Write-Host "TUM KAPILAR GECTI" -ForegroundColor Green }
else { Write-Host ("{0} KAPI KALDI" -f $fails) -ForegroundColor Red; exit 1 }
