<#
.SYNOPSIS
  Vizor AF noktalarinin script yuzeyini dogrular: kapilar ADIYLA raporlaniyor
  mu, gecersiz indeks REDDEDILIYOR mu, ve AF-C gercekten odak mesafesini
  YAZIYOR mu.

.DESCRIPTION
  ★★★★ AF bir izleme overlay'i DEGILDIR. AF-C modunda secili nokta her karede
  sahneyi olcup `camera.focus_distance`i EZER. Yani bu, kamera durumunu
  degistiren bir aractir -- ve 2026-09-06'ya kadar yalnizca bir viewport
  popup'inda yasiyordu, script'ten hic erisilemiyordu (CLAUDE.md kural 1).

  ★★★ 4. kapi bu yuzden en degerlisi: AF-C acikken odak mesafesini yazip
  GERI ALINDIGINI gorur. Bu kapi olmadan, "focus_distance ayarladim ama
  tutmuyor" sorusunun cevabi hicbir yerde olmazdi.

.NOTES
  ★ Nokta sayisi alan modundan TURETILIR (Zone21 = 5x5 = 25, digerleri 3x3 = 9).
    Betik bunu iki yerde tutmaz, `viewport.get_af`in `point_count`ini okur --
    yoksa olcu aleti olctugu seyle ayni varsayimi tekrarlardi.
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

$af0  = Invoke-RtIpc viewport.get_af @{}
$cam0 = Invoke-RtIpc camera.get @{}

try {
    # ── 1. Alan modu nokta sayisini BELIRLER ────────────────────────────────
    Write-Host "1. point_count alan modundan turetilmeli"
    Invoke-RtIpc viewport.set_af @{ enabled = $true; area_mode = 1; focus_mode = 0; selected_point = 4 } | Out-Null
    $a9 = Invoke-RtIpc viewport.get_af @{}
    Invoke-RtIpc viewport.set_af @{ enabled = $true; area_mode = 2; focus_mode = 0; selected_point = 4 } | Out-Null
    $a25 = Invoke-RtIpc viewport.get_af @{}
    Check "Zone9 -> 9 nokta"  ([int]$a9.point_count -eq 9)   ("point_count={0}" -f $a9.point_count)
    Check "Zone21 -> 25 nokta" ([int]$a25.point_count -eq 25) ("point_count={0}" -f $a25.point_count)

    # ── 2. ★★★ GECERSIZ INDEKS REDDEDILMELI, sessizce kirpilmamali ──────────
    Write-Host "2. alan modunun disindaki nokta indeksi reddedilmeli"
    Invoke-RtIpc viewport.set_af @{ enabled = $true; area_mode = 1; focus_mode = 0; selected_point = 4 } | Out-Null
    $rejected = $false
    try { Invoke-RtIpc viewport.set_af @{ enabled = $true; area_mode = 1; focus_mode = 0; selected_point = 20 } | Out-Null }
    catch { $rejected = $true }
    Check "Zone9'da 20. nokta reddedildi" $rejected "kirpma DEGIL reddetme"
    $cur = Invoke-RtIpc viewport.get_af @{}
    Check "reddedilen cagri hicbir seyi degistirmedi" ([int]$cur.selected_point -eq 4) `
          ("selected_point={0}" -f $cur.selected_point)
    $rejected = $false
    try { Invoke-RtIpc viewport.set_af @{ enabled = $true; area_mode = 9; focus_mode = 0; selected_point = 0 } | Out-Null }
    catch { $rejected = $true }
    Check "gecersiz alan modu reddedildi" $rejected "area_mode araligi [0,4]"

    # ── 3. ★★★ KAPI RAPORU: HUD kapaliyken cizim YOK, ve bu SOYLENMELI ──────
    Write-Host "3. Camera HUD kapaliyken active=false ve sebep yazili olmali"
    Invoke-RtIpc viewport.set_af @{ enabled = $false; area_mode = 1; focus_mode = 0; selected_point = 4 } | Out-Null
    $d = Invoke-RtIpc viewport.get_af @{}
    Check "kapaliyken active=false" ($d.active -eq $false) ("active={0}" -f $d.active)
    Check "sebep bos degil" (-not [string]::IsNullOrWhiteSpace([string]$d.inactive_reason)) `
          ("reason='{0}'" -f $d.inactive_reason)

    # ── 4. ★★★★ AF-C ODAK MESAFESINI EZER -- ve bu OLCULEBILMELI ────────────
    # "focus_distance ayarladim ama tutmuyor"un cevabi burasi.
    Write-Host "4. AF-C acikken yazilan odak mesafesi GERI ALINMALI (beklenen davranis)"
    Invoke-RtIpc viewport.set_af @{ enabled = $true; area_mode = 1; focus_mode = 2; selected_point = 4 } | Out-Null
    $probe = Invoke-RtIpc viewport.get_af @{}
    if ($probe.active -ne $true) {
        Write-Host ("  [ATLANDI] AF cizilmiyor: '{0}'" -f $probe.inactive_reason) -ForegroundColor Yellow
        Write-Host "     ★ Camera HUD'i acip sahnede bir nesne olmasi gerekir; aksi halde" -ForegroundColor Yellow
        Write-Host "       bu kapinin GECMESI hicbir sey soylemez." -ForegroundColor Yellow
    } else {
        Invoke-RtIpc camera.set_focus_distance @{ focus_distance = 42.0 } | Out-Null
        Start-Sleep -Milliseconds 600
        $c = Invoke-RtIpc camera.get @{}
        Check "AF-C odagi geri aldi" ([Math]::Abs([double]$c.focus_distance - 42.0) -gt 0.05) `
              ("focus_distance={0:F3} (42 yazilmisti)" -f $c.focus_distance)

        # ── 5. MF modunda YAZDIGIN deger DURMALI ────────────────────────────
        Write-Host "5. MF modunda yazilan odak mesafesi DURMALI"
        Invoke-RtIpc viewport.set_af @{ enabled = $true; area_mode = 1; focus_mode = 0; selected_point = 4 } | Out-Null
        Invoke-RtIpc camera.set_focus_distance @{ focus_distance = 42.0 } | Out-Null
        Start-Sleep -Milliseconds 600
        $c = Invoke-RtIpc camera.get @{}
        Check "MF'te odak korundu" ([Math]::Abs([double]$c.focus_distance - 42.0) -lt 0.05) `
              ("focus_distance={0:F3}" -f $c.focus_distance)
    }
}
finally {
    Invoke-RtIpc viewport.set_af @{
        enabled = [bool]$af0.enabled
        area_mode = [int]$af0.area_mode
        focus_mode = [int]$af0.focus_mode
        selected_point = [int]$af0.selected_point } | Out-Null
    Invoke-RtIpc camera.set_focus_distance @{ focus_distance = [double]$cam0.focus_distance } | Out-Null
}

Write-Host ""
if ($fails -eq 0) { Write-Host "TUM KAPILAR GECTI" -ForegroundColor Green }
else { Write-Host ("{0} KAPI KALDI" -f $fails) -ForegroundColor Red; exit 1 }
