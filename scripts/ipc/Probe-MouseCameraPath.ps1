<#
.SYNOPSIS
  FARE yolunu olcer: bir gestur sirasinda CPU kamerasi kimildiyor mu, ve GPU
  karesi degisiyor mu. Geri sayim boyunca hareketi KULLANICI yapar.

.DESCRIPTION
  ★★★★★ Bu betik, uc yanlis tahminden sonra yazildi. Sirasiyla su kokler
  onerildi ve UCU DE IPC'den olculup ELENDI:
    1. "tek bayrak / iki VkDevice"  -> g_camera_dirty yutuluyor sanildi
    2. "scene.initialized kapisi"   -> File>New ile denendi, degismedi
    3. "senkron eksik"              -> camera.set_position (yalnizca
       g_camera_dirty yazar, viewport syncCamera YOK) kareyi GUNCELLEDI
  Yani hareket matematigi, backend senkronu ve yeniden cizim kapisi
  UCU DE SAGLAM. Geriye tek bir olculmemis halka kaldi: farenin kendisi.

  ★★★ Olculen sey kasten IKI AYRI SORU:
    - CPU kamerasi kimildadi mi?  -> olay kameraya ULASTI mi
    - GPU karesi degisti mi?      -> ulastiysa CIZDIRDI mi
  Belirti "pan sadece imgui objelerde etkili" ise beklenen cevap
  CPU=True / GPU=False'dur. Ama CPU=False cikarsa olay kameraya hic
  ulasmiyordur ve butun backend hipotezleri bir kerede elenir.

.PARAMETER Label
  Denenen gestur ("rotate", "pan", "zoom"). Yalnizca raporda gorunur.

.PARAMETER Seconds
  Geri sayim suresi. Bu sure boyunca gesturu YAP.

.EXAMPLE
  .\Probe-MouseCameraPath.ps1 -Label rotate   # orta tus surukle
  .\Probe-MouseCameraPath.ps1 -Label pan      # SHIFT + orta tus surukle
  .\Probe-MouseCameraPath.ps1 -Label zoom     # tekerlek

.NOTES
  ★ Uygulama REALTIME RASTER modunda ve arizanin gorundugu sahnede olmali.
  ★ Gesturu viewport'un UZERINDE yap - panel uzerinde yaparsan olay zaten
    ImGui'ye gider ve olctugun sey ariza degil, dogru davranis olur.
#>
[CmdletBinding()]
param(
    [string]$Label = "gestur",
    [int]$Seconds = 6
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'RtIpc.psm1') -Force

function FrameHash {
    $r = Invoke-RtIpc viewport.get_screenshot @{}
    if (-not $r.image_base64) { return $null }
    $b = [Convert]::FromBase64String([string]$r.image_base64)
    return [BitConverter]::ToString(([System.Security.Cryptography.SHA256]::Create()).ComputeHash($b))
}
function CamSig {
    # ★ Vec3'ler IPC'den DIZI doner ([x,y,z]). `.x` ile okumak sessizce bos
    #   string uretir ve olcu aleti "hic kimildamadi" diye RAPORLAR.
    $c = Invoke-RtIpc camera.get @{}
    return ("{0:F4},{1:F4},{2:F4} -> {3:F4},{4:F4},{5:F4}" -f `
        [double]$c.position[0], [double]$c.position[1], [double]$c.position[2], `
        [double]$c.target[0],   [double]$c.target[1],   [double]$c.target[2])
}

Invoke-RtIpc viewport.capture @{ enabled = $true } | Out-Null
Start-Sleep -Milliseconds 400

$cam0 = CamSig
$img0 = FrameHash
Write-Host ""
Write-Host ("SIMDI '{0}' GESTURUNU YAP - viewport uzerinde, {1} saniyen var:" -f $Label, $Seconds) -ForegroundColor Cyan
for ($i = $Seconds; $i -gt 0; $i--) {
    Write-Host ("  {0}..." -f $i) -NoNewline
    Start-Sleep -Seconds 1
}
Write-Host ""
Start-Sleep -Milliseconds 400
$cam1 = CamSig
$img1 = FrameHash

$camMoved = ($cam0 -ne $cam1)
$imgMoved = ($img0 -ne $img1)

Write-Host ""
Write-Host ("=== {0} ===" -f $Label) -ForegroundColor Cyan
Write-Host ("  once : {0}" -f $cam0)
Write-Host ("  sonra: {0}" -f $cam1)
Write-Host ("  CPU kamera kimildadi : {0}" -f $camMoved)
Write-Host ("  GPU kare degisti     : {0}" -f $imgMoved)
Write-Host ""

if (-not $camMoved -and -not $imgMoved) {
    Write-Host "★★★★★ OLAY KAMERAYA HIC ULASMIYOR." -ForegroundColor Green
    Write-Host "  -> Butun backend/senkron/cizim hipotezleri ELENDI. Ariza SDL olay"
    Write-Host "     kapilarinda: Main.cpp 3670 (MOUSEMOTION) / 3766 (MOUSEWHEEL)"
    Write-Host "     -- mouse_control_enabled, input_locked, WantCaptureMouse."
    Write-Host "  ★ Ayni gesturu IPC'den surmek CALISIYOR, yani kod degil KAPI."
}
elseif ($camMoved -and -not $imgMoved) {
    Write-Host "★★★★ Kamera kimildiyor ama kare cizilmiyor." -ForegroundColor Yellow
    Write-Host "  -> Belirtiyle birebir ortusuyor. Ama IPC yolu ayni durumda"
    Write-Host "     cizdirebiliyordu, yani fark bu gesturun frame icindeki ZAMANI"
    Write-Host "     olabilir: bayragi kimin ONCE tukettigi. g_camera_dirty'yi"
    Write-Host "     temizleyen alti yeri sirala (2292/4539/5208/5782/5822/6653)."
}
elseif ($camMoved -and $imgMoved) {
    Write-Host "★★★ Bu gestur TAM CALISIYOR (kamera da kare de degisti)." -ForegroundColor Green
    Write-Host "  -> Bunu bozuk gesturle karsilastir; fark ikisinin AYRILDIGI yerdedir."
}
else {
    Write-Host "Kare degisti ama kamera degismedi - baska bir sey ciziyor." -ForegroundColor Yellow
    Write-Host "  -> Gesturu viewport uzerinde yaptigindan emin ol; animasyon/sim"
    Write-Host "     kosuyorsa kare zaten her karede degisir ve bu olcum anlamsizdir."
}
