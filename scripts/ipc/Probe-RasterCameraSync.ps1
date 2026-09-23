<#
.SYNOPSIS
  Raster viewport'un kamera hareketine neden tepki vermedigini UC olasilik
  arasinda ayirir: CPU kamerasi kimildamiyor mu, backend senkronu mu eksik,
  yoksa yeniden cizim kapisi mi kapali.

.DESCRIPTION
  ★★★★★ Bu betik bir TAHMINI degil, iki basarisiz tahmini takip ediyor. Bu
  arizanin kokunu once "tek bayrak / iki VkDevice", sonra "scene.initialized
  kapisi" diye okudum; IKISI DE YANLISTI. Ortak hatam ayni: belirtiyi daha
  derin okumaya calistim, oysa gereken sey ONU AYIRAN KOSULU olcmekti.

  ★★★ Betigin tasarim fikri sudur: butun hareketleri IPC'den surer, cunku IPC
  yolu (`rtapi::navChanged`) hem `g_viewport_backend`i hem `g_ctx->backend_ptr`yi
  ACIKCA senkronlar. Fare yolu ise yalnizca `g_camera_dirty` yazar. Yani ayni
  hareketin iki yolu farkli senkron sozlesmesine sahip, ve fark tam olarak
  olculecek sey.

  ★★ Kritik kapi 3'tur: AYNI senkron yolundan gecen orbit ile pan farkli
  sonuc veriyorsa, ariza senkronda DEGIL, backend'in yeniden cizim kapisinda
  ve o kapi konuma duyarsiz demektir. Ayni sonuc veriyorsa ariza fare
  yolundadir.

.NOTES
  ★ Uygulama REALTIME RASTER modunda olmali (RT modda ariza zaten yok).
    Sahneyi de kullanicinin bozuk gordugu durumda birak (default sahne).
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'RtIpc.psm1') -Force

function Frame-Hash {
    # Tek bir sayiya indirgenmis kare. Piksel piksel karsilastirmaya gerek yok;
    # sorulan soru "kare DEGISTI mi", "ne kadar degisti" degil.
    $r = Invoke-RtIpc viewport.get_screenshot @{}
    if (-not $r.image_base64) { return $null }
    $bytes = [Convert]::FromBase64String([string]$r.image_base64)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    return [BitConverter]::ToString($sha.ComputeHash($bytes))
}

function Cam-Signature {
    # ★★★ Vec3'ler IPC'den DIZI olarak doner ([x,y,z]), nesne olarak degil.
    #   `.x` ile okumak sessizce BOS string uretir ve her imza ayni cikar --
    #   yani olcu aleti "kamera hic kimildamadi" diye RAPORLAR. Bir olcumun
    #   yanlis olmasindan beter olan tek sey, yanlis oldugu belli olmayan bir
    #   olcumdur; bu betik tam da o hataya dusmustu.
    $c = Invoke-RtIpc camera.get @{}
    return ("{0:F4},{1:F4},{2:F4} -> {3:F4},{4:F4},{5:F4}" -f `
        [double]$c.position[0], [double]$c.position[1], [double]$c.position[2], `
        [double]$c.target[0],   [double]$c.target[1],   [double]$c.target[2])
}

Invoke-RtIpc viewport.capture @{ enabled = $true } | Out-Null
Start-Sleep -Milliseconds 400

$results = @()
function Gesture($label, $method, $params) {
    $cam0 = Cam-Signature
    $img0 = Frame-Hash
    Invoke-RtIpc $method $params | Out-Null
    # ★ IPC yazisi ile olcusu ARASINA kare koy: toggle uygulanir ama sayac
    #   onceki partiyi olcer (bkz. feedback_ipc_counter_needs_a_frame_between...).
    Start-Sleep -Milliseconds 500
    $cam1 = Cam-Signature
    $img1 = Frame-Hash

    $camMoved = ($cam0 -ne $cam1)
    $imgMoved = ($img0 -ne $img1) -and ($null -ne $img0) -and ($null -ne $img1)
    $script:results += [pscustomobject]@{
        Hareket = $label; CPU_kamera = $camMoved; GPU_kare = $imgMoved
    }
    Write-Host ("  {0,-22} CPU kamera: {1,-5}  GPU kare: {2,-5}" -f $label, $camMoved, $imgMoved)
    if (-not $imgMoved -and $null -eq $img0) {
        Write-Host "    (uyari: ekran goruntusu alinamadi - viewport.capture acik mi?)" -ForegroundColor Yellow
    }
}

Write-Host "IPC'den surulen hareketler (navChanged iki backend'i de senkronlar):"
Gesture "orbit (yonelim)"  camera.orbit @{ yaw = 25.0; pitch = 0.0 }
Gesture "pan (konum)"      camera.pan   @{ right = 2.0; up = 0.0 }
Gesture "dolly (konum)"    camera.dolly @{ factor = 0.6 }
Gesture "orbit (geri)"     camera.orbit @{ yaw = -25.0; pitch = 0.0 }

Write-Host ""
Write-Host "=== OKUMA ===" -ForegroundColor Cyan

$orbit = $results | Where-Object { $_.Hareket -like 'orbit*' }
$konum = $results | Where-Object { $_.Hareket -like 'pan*' -or $_.Hareket -like 'dolly*' }

$cpuAllMoved = -not ($results | Where-Object { -not $_.CPU_kamera })
if (-not $cpuAllMoved) {
    Write-Host "CPU kamerasi bazi hareketlerde KIMILDAMIYOR." -ForegroundColor Red
    Write-Host "  -> Ariza cizimde degil, hareketin kendisinde. Kapiyi backend'de arama."
    exit 1
}
Write-Host "CPU kamerasi her harekette kimildiyor (hareket matematigi saglam)."

$orbitDrew = ($orbit | Where-Object { $_.GPU_kare }).Count -gt 0
$konumDrew = ($konum | Where-Object { $_.GPU_kare }).Count -gt 0

if ($orbitDrew -and -not $konumDrew) {
    Write-Host ""
    Write-Host "★★★★★ AYRIM BULUNDU: orbit cizdiriyor, pan/dolly cizdirmiyor --" -ForegroundColor Green
    Write-Host "      ve IKISI DE ayni IPC senkron yolundan gecti." -ForegroundColor Green
    Write-Host "  -> Ariza senkronda DEGIL. Backend'in yeniden cizim kapisi KONUMA"
    Write-Host "     duyarsiz. Bakilacak yer: VulkanViewportBackend.cpp ~3630 kapisi"
    Write-Host "     ve ~3555 hashCamera -- ozellikle m_camera'nin gercekten"
    Write-Host "     GUNCELLENDIGI yer (syncCamera'yi bu sinif override ediyor mu?)."
}
elseif ($orbitDrew -and $konumDrew) {
    Write-Host ""
    Write-Host "★★★★ IPC'den UCU DE cizdiriyor." -ForegroundColor Green
    Write-Host "  -> Backend ve kapi saglam. Ariza FARE YOLUNDA: o yol yalnizca"
    Write-Host "     g_camera_dirty yaziyor, IPC yolu ise backend'i acikca"
    Write-Host "     senkronluyor. Simdi ayni hareketleri FAREYLE yap ve bu betigi"
    Write-Host "     tekrar kosmadan viewport.get_screenshot ile karsilastir."
    Write-Host "  ★ Duzeltmeyi TEK GOVDEYE ekle, dorduncu bir kopya cikarma."
}
elseif (-not $orbitDrew -and -not $konumDrew) {
    Write-Host ""
    Write-Host "★★★★ HICBIRI cizdirmiyor (CPU kamerasi kimildadigi halde)." -ForegroundColor Yellow
    Write-Host "  -> Raster present yolu bu sahne durumunda tamamen kapali."
    Write-Host "     Kamerayla ilgisi yok. Ayni testi PROJE ACIKKEN tekrarla:"
    Write-Host "     orada cizdiriyorsa fark sahne durumunda, kamerada degil."
}
else {
    Write-Host ""
    Write-Host "Beklenmedik desen: konum cizdiriyor, yonelim cizdirmiyor." -ForegroundColor Yellow
    Write-Host "  -> Tablodaki ham degerleri aynen paylas; varsayimlarimin disinda."
}

Write-Host ""
$results | Format-Table -AutoSize
