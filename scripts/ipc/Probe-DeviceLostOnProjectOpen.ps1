<#
.SYNOPSIS
  Proje acilisindaki VK_ERROR_DEVICE_LOST'u BILEREK tekrar uretir ve sonucu
  DEGER olarak raporlar.

.DESCRIPTION
  Arka plan: "hicbir sahne-acma yolu agir viewport modunu zorlayamaz" kurali
  (scene_ui.h) konulunca cokme gitti -- ama kok neden bulunmadan gitti. Kural,
  arizali yolu hic kullandirmiyor; dolayisiyla kok nedeni arayan tripwire
  (bayat material-preview descriptor set'i) ARTIK ASLA TETIKLENEMEZ.

  ★★★ Ve tam burada bu deponun bilinen tuzagi var:
      "Tripwire'in susmasi yoklugu kanitlamaz. Enstrumanin anahtari, olctugu
       seyle cakismamali."
  Kalkan acikken tripwire'in sessizligi bir OLCUM DEGILDIR. Bu betik kalkani
  olcum suresince kapatir, ariza penceresini acar, sonra kalkani GERI ACAR.

  Uc olasi sonuc ve anlamlari:
    1. stale_descset_rebuilds ARTTI  -> KOK NEDEN BULUNDU. Bir doku purge'u,
       descriptor set'i sokmeden altindaki VkImage'lari yok ediyor.
    2. device_lost = true, sayac ARTMADI -> o sinif ELENDI; ariza baska bir
       yerden geliyor. Siradaki adim RAYTROPHI_VK_VALIDATION=1.
    3. Ikisi de yok -> ariza bu kosulda URETILEMEDI. "Duzeldi" DEMEK DEGILDIR;
       senaryo yeterince agir olmayabilir (bkz. -ProjectA / -ProjectB secimi).

  ★ ProjectA DOLU bir sahne olmali. Bilinen A/B farki: bos uygulamaya agir
    sahne yuklemek SORUNSUZ; ariza yalnizca ORTADA SOKULECEK dolu bir sahne
    varken goruluyor. A'yi hafif secmek 3. sonucu garanti eder ve hicbir sey
    olcmez.

.EXAMPLE
  .\Probe-DeviceLostOnProjectOpen.ps1 -ProjectA 'E:\sahneler\agir1.rtp' `
                                      -ProjectB 'E:\sahneler\agir2.rtp'
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ProjectA,
    [Parameter(Mandatory = $true)][string]$ProjectB,
    # Material moduna gectikten sonra kac kare cizdirilsin. Descriptor set'in
    # GERCEKTEN yazilmis ve bir gonderimde KULLANILMIS olmasi sart: damga o
    # anda basiliyor, yani bu bekleme olcumun on kosulu.
    [int]$WarmupMs = 1500,
    [int]$SettleMs = 1200
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'RtIpc.psm1') -Force

function Get-Tel { @(Invoke-RtIpc viewport.frame_telemetry @{})[-1] }
function Get-Shading { @(Invoke-RtIpc viewport.shading @{})[-1] }

function Show-Diag([string]$label) {
    $t = Get-Tel
    # ★ Bu iki alan `available` false iken de yayinlanir -- surucu kaybi zaten
    #   ring'i olduren seydir, yani "olcum yok" dedigi an tam da okunmasi
    #   gereken andir.
    $stale = [int64]$t.stale_descset_rebuilds
    $lost  = [bool]$t.device_lost
    Write-Host ("  {0,-22} stale_descset_rebuilds={1}  device_lost={2}  available={3}" -f
                $label, $stale, $lost, $t.available)
    return [pscustomobject]@{ stale = $stale; lost = $lost }
}

Write-Host "=== Device-lost tekrar uretimi ===" -ForegroundColor Cyan

$guardRestored = $false
try {
    # --- 1. A sahnesini KALKAN ACIKKEN yukle (bu adim ariza penceresi degil) --
    Invoke-RtIpc viewport.set_scene_load_guard @{ enabled = $true } | Out-Null
    Write-Host "[1] ProjectA yukleniyor (kalkan ACIK, guvenli)..."
    Invoke-RtIpc project.open @{ path = $ProjectA } | Out-Null
    Start-Sleep -Milliseconds $SettleMs

    # --- 2. Material'a gec ve descriptor set'in KULLANILMASINI bekle ---------
    Write-Host "[2] Material moduna geciliyor ve kareler isitiliyor..."
    Invoke-RtIpc viewport.set_shading @{ mode = 'material' } | Out-Null
    Start-Sleep -Milliseconds $WarmupMs
    $mode0 = (Get-Shading).mode
    if ($mode0 -ne 'material') {
        # ★ Sessizce devam etmek olcumu sahte yapar: Material bagli DEGILSE
        #   ariza penceresi hic acilmamis olur ve "cokmedi" sonucu yalandir.
        throw "Viewport 'material' moduna gecmedi (mode=$mode0). Bu makinede raster viewport yok olabilir; olcum yapilamaz."
    }
    $before = Show-Diag 'A yuklendi'

    # --- 3. KALKANI KAPAT: ariza penceresi burada acilir --------------------
    Write-Host "[3] Kalkan KAPATILIYOR -- bundan sonraki acilis Material bagliyken kosacak." -ForegroundColor Yellow
    Invoke-RtIpc viewport.set_scene_load_guard @{ enabled = $false } | Out-Null

    Write-Host "[4] ProjectB aciliyor (ARIZA PENCERESI)..." -ForegroundColor Yellow
    try {
        Invoke-RtIpc project.open @{ path = $ProjectB } | Out-Null
    } catch {
        # Surucu kaybi cagrinin kendisini de dusurebilir; uygulama ayakta kalir
        # ve telemetri yine okunur. Hatayi yutmuyoruz, RAPORLUYORUZ.
        Write-Host "    project.open hata dondurdu: $($_.Exception.Message)" -ForegroundColor Red
    }
    Start-Sleep -Milliseconds ($SettleMs * 2)
    $after = Show-Diag 'B acildi'

    # --- 5. Karar -----------------------------------------------------------
    $delta = $after.stale - $before.stale
    Write-Host ""
    Write-Host "=== SONUC ===" -ForegroundColor Cyan
    if ($delta -gt 0) {
        Write-Host "KOK NEDEN BULUNDU: stale_descset_rebuilds +$delta." -ForegroundColor Green
        Write-Host "  Bir doku purge'u material-preview descriptor set'ini sokmeden altindaki"
        Write-Host "  VkImage'lari yok ediyor. Scene Log'daki 'STALE material-preview descriptor"
        Write-Host "  set' satirindaki kusak numaralari hangi purge oldugunu daraltir."
    } elseif ($after.lost) {
        Write-Host "SINIF ELENDI: device_lost = true ama tripwire ARTMADI." -ForegroundColor Yellow
        Write-Host "  Ariza gercek ve tekrar uretildi, ama sebebi bayat descriptor set DEGIL."
        Write-Host "  Siradaki adim: RAYTROPHI_VK_VALIDATION=1 ile ayni senaryoyu tekrarla."
    } else {
        Write-Host "URETILEMEDI: ne cokme ne tripwire." -ForegroundColor Yellow
        Write-Host "  ★ Bu 'duzeldi' DEMEK DEGILDIR -- olcum, arizayi goremedigini soyluyor."
        Write-Host "  ProjectA yeterince DOLU mu? Bilinen fark: bos uygulamaya agir sahne"
        Write-Host "  yuklemek sorunsuz; ariza ortada SOKULECEK dolu bir sahne ister."
    }
}
finally {
    # ★★★★ Kalkan HER DURUMDA geri acilir. Kapali unutulmus bir kalkan, aylar
    #   sonra gelen bir device-lost raporunun gorunmeyen sebebi olur.
    try {
        Invoke-RtIpc viewport.set_scene_load_guard @{ enabled = $true } | Out-Null
        $guardRestored = $true
    } catch {
        $guardRestored = $false
    }
    if ($guardRestored) {
        Write-Host "[son] Kalkan GERI ACILDI." -ForegroundColor Green
    } else {
        Write-Host "[son] ★★★ UYARI: kalkan geri acilamadi (IPC cevapsiz olabilir)." -ForegroundColor Red
        Write-Host "      Uygulamayi yeniden baslat; kalkan varsayilan olarak ACIK gelir."
    }
}
