<#
.SYNOPSIS
  Realtime (raster onizleme) ile Rendered'in AYNI goruntuleme donusumunu
  kullandigini dogrular.

.DESCRIPTION
  2026-09-02'ye kadar bu iki mod UC ayri operator kullaniyordu:
    Rendered            : sabit Reinhard + sRGB      (tonemap.comp)
    Realtime nesneler   : sabit ACES + pow(1/2.2)    (material_preview_frag)
    Realtime gokyuzu    : sabit ACES + pow(1/2.2)    (material_preview_sky)
  ve dogrudan difuz teriminde 1/PI eksikti; yerine YALNIZCA dusuk kalite
  dalinda 0.35 sihirli katsayisi vardi. Sonuc: kalite preset'i degistirmek
  POZLAMAYI degistiriyordu.

  Bu betik dort testi SIRALI kosar. Sirasi onemli: once ucuz ve bagimsiz
  olanlar, sonra otekilerin sonucunu maskeleyebilecekler.

.NOTES
  ★ Kamera ve sahne sabit tutulmali. Betik hicbir sahne degisikligi yapmaz;
    yalnizca post ve kalite kadranlarini oynatip geri koyar.
#>
[CmdletBinding()]
param(
    [double]$Tolerance = 0.01,       # mean_luminance mutlak tolerans
    # ★★★ Olcum bolgesi. Varsayilan TAM KARE, ve tam kare cogu sahnede YANLIS
    #   olcektir: 2026-09-02'de bu betik "yayilim 0.0000, gecti" dedi -- kare
    #   %97 zemin izgarasiydi ve olculecek kup piksellerin %0.5'iydi. Ozneyi
    #   kapsayan bir bolge ver: -Region 1950,870,380,400
    [int[]]$Region = @()
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'RtIpc.psm1') -Force

$fails = 0
function Check($name, $ok, $detail) {
    if ($ok) { Write-Host ("  [GECTI] {0} {1}" -f $name, $detail) }
    else     { Write-Host ("  [KALDI] {0} {1}" -f $name, $detail) -ForegroundColor Red; $script:fails++ }
}

# ★★★ Olcum ALMADAN ONCE kareyi ZORLA yenile.
#
#   Ilk surumde burada yalnizca `Start-Sleep` vardi ve dort kalite preset'i
#   BIREBIR ayni sayiyi dondurdu (0.6297 x4). "Yayilim sifir" gorunuyordu ama
#   olculen sey sabit bir BAYAT KAREYDI: sahne temizken viewport yeni kare
#   cizmiyor. 0 == 0 yesil -- bu deponun bilinen sahte-gecis sinifi.
#
#   Kamera once minik oynatilip SONRA tam olarak eski yerine konuyor: ikinci
#   yazma da bir kare kirletir ve kamera olcum aninda ORIJINAL konumdadir,
#   yani karsilastirilan kareler ayni bakis acisindan.
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
    param([switch]$Rendered)
    if ($Rendered) {
        # Rendered'da kamera hareketi birikimi sifirlar; kareyi render_frames
        # topluyor, o yuzden zorlama YOK -- cagiran render_frames'i kendi yapar.
        Start-Sleep -Milliseconds 300
    } else {
        if (-not (Force-Frame)) {
            throw "Viewport yeni kare cizmedi -- olcum BAYAT olurdu, durduruldu."
        }
    }
    $args = @{}
    if ($Region.Count -eq 4) {
        $args = @{ x = $Region[0]; y = $Region[1]; width = $Region[2]; height = $Region[3] }
    }
    $p = Invoke-RtIpc render.probe $args
    if (-not $p.available) { throw "render.probe kare veremedi -- viewport.capture acik mi?" }
    return [double]$p.mean_luminance
}

Invoke-RtIpc viewport.capture @{ enabled = $true } | Out-Null
$post0 = Invoke-RtIpc post.get @{}
$shading0 = (Invoke-RtIpc viewport.shading @{}).mode
$quality0 = (Invoke-RtIpc viewport.quality @{}).preset

try {
    # ── 1. Ayna: onizlemenin GORDUGU post, ayarla ayni mi ────────────────────
    Write-Host "1. post.get ile shader'a giden degerler ayni mi"
    $pl = Invoke-RtIpc viewport.preview_lighting @{}
    if ($null -eq $pl.display_tone_mapping) {
        Write-Host "  [ATLANDI] display_* alanlari yok -- eski binary." -ForegroundColor Yellow
    } else {
        Check "tone_mapping" ($pl.display_tone_mapping -eq $post0.tone_mapping) `
              ("{0} / {1}" -f $pl.display_tone_mapping, $post0.tone_mapping)
        Check "exposure" ([Math]::Abs($pl.display_exposure - $post0.exposure) -lt 1e-4) `
              ("{0} / {1}" -f $pl.display_exposure, $post0.exposure)
        Check "gamma" ([Math]::Abs($pl.display_gamma - $post0.gamma) -lt 1e-4) `
              ("{0} / {1}" -f $pl.display_gamma, $post0.gamma)
        Check "saturation" ([Math]::Abs($pl.display_saturation - $post0.saturation) -lt 1e-4) `
              ("{0} / {1}" -f $pl.display_saturation, $post0.saturation)
    }


    # ── 1b. ★★★★ DUYARLILIK KALIBRASYONU ────────────────────────────────────
    # Bir kapinin "gecmesi", o kapinin BASARISIZLIGI GOREBILECEGINI kanitlamaz.
    # Duzeltilen hata dogrudan difuz terimde ~%21'lik bir POZLAMA kaymasiydi;
    # bu blok, bu kurulumun boyle bir kaymayi gorup goremeyecegini olcer:
    # exposure 1.0 -> 1.2 arasindaki gercek farki ariyoruz. Kimildamiyorsa
    # asagidaki preset testi bir OLCUM DEGIL, korlukten gelen bir sifirdir.
    Write-Host "1b. duyarlilik: bu kurulum %20'lik bir kaymayi gorebiliyor mu"
    Invoke-RtIpc viewport.set_shading @{ mode = 'material' } | Out-Null
    Invoke-RtIpc post.set_exposure @{ exposure = 1.0 } | Out-Null
    $cal1 = Probe-Mean
    Invoke-RtIpc post.set_exposure @{ exposure = 1.2 } | Out-Null
    $cal2 = Probe-Mean
    Invoke-RtIpc post.set_exposure @{ exposure = 1.0 } | Out-Null
    $calDelta = [Math]::Abs($cal2 - $cal1)
    Write-Host ("     exposure 1.0 -> 1.2 : {0:F4} -> {1:F4}   delta={2:F4}" -f $cal1, $cal2, $calDelta)
    $sensitive = $calDelta -gt (3.0 * $Tolerance)
    Check "duyarlilik" $sensitive ("delta={0:F4}, gereken > {1:F4}" -f $calDelta, (3.0*$Tolerance))
    if (-not $sensitive) {
        Write-Host "     ★ Bu kurulum kucuk degisimleri goremiyor: olcum bolgesi ozneyi" -ForegroundColor Yellow
        Write-Host "       kapsamiyor olabilir. -Region x,y,w,h ver. Asagidaki preset" -ForegroundColor Yellow
        Write-Host "       testi bu haliyle ANLAMSIZ." -ForegroundColor Yellow
    }

    # ── 2. ★★★ Kalite preset'i POZLAMAYI degistirmemeli ──────────────────────
    # Bu testin varlik sebebi: 0.35/1.0 dallanmasi tam olarak bunu yapiyordu
    # (olculdu: Performance 0.4055 vs Balanced 0.4893, %21).
    # ★ Kalan kucuk fark (~%0.2) NORMALDIR: qualityMode artik lobun BICIMINI
    #   seciyor (Blinn-Phong vs GGX), siddetini degil.
    Write-Host "2. kalite preset'i parlakligi degistirmemeli"
    $means = @{}
    foreach ($q in @('performance','balanced','quality','full')) {
        Invoke-RtIpc viewport.set_quality @{ preset = $q } | Out-Null
        $means[$q] = Probe-Mean
        Write-Host ("     {0,-12} mean={1:F4}" -f $q, $means[$q])
    }
    $spread = ($means.Values | Measure-Object -Maximum).Maximum - ($means.Values | Measure-Object -Minimum).Minimum
    Check "preset yayilimi" ($spread -lt $Tolerance) ("yayilim={0:F4} (tolerans {1})" -f $spread, $Tolerance)

    # ── 3. Exposure her iki modda AYNI orani uretmeli ────────────────────────
    Write-Host "3. exposure iki modda ayni orani uretmeli"
    Invoke-RtIpc viewport.set_quality @{ preset = 'balanced' } | Out-Null
    Invoke-RtIpc post.set_exposure @{ exposure = 1.0 } | Out-Null
    $prev1 = Probe-Mean
    Invoke-RtIpc post.set_exposure @{ exposure = 2.0 } | Out-Null
    $prev2 = Probe-Mean
    Invoke-RtIpc post.set_exposure @{ exposure = 1.0 } | Out-Null

    Invoke-RtIpc viewport.set_shading @{ mode = 'rendered' } | Out-Null
    Invoke-RtIpc viewport.render_frames @{ count = 16 } | Out-Null
    $rend1 = Probe-Mean -Rendered
    Invoke-RtIpc post.set_exposure @{ exposure = 2.0 } | Out-Null
    Invoke-RtIpc viewport.render_frames @{ count = 16 } | Out-Null
    $rend2 = Probe-Mean -Rendered
    Invoke-RtIpc post.set_exposure @{ exposure = 1.0 } | Out-Null

    $rPrev = if ($prev1 -gt 1e-6) { $prev2 / $prev1 } else { 0 }
    $rRend = if ($rend1 -gt 1e-6) { $rend2 / $rend1 } else { 0 }
    Write-Host ("     preview  {0:F4} -> {1:F4}  oran {2:F3}" -f $prev1, $prev2, $rPrev)
    Write-Host ("     rendered {0:F4} -> {1:F4}  oran {2:F3}" -f $rend1, $rend2, $rRend)
    Check "exposure orani" ([Math]::Abs($rPrev - $rRend) -lt 0.10) ("fark={0:F3}" -f [Math]::Abs($rPrev - $rRend))

    # ── 4. Mutlak parity (BILGI, kapi degil) ────────────────────────────────
    # ★★ Bu bir PASS/FAIL degil: onizleme hala bir yaklasiklik (tek ornekli
    #   golge, GI yok). Beklenen, farkin 1.45x'ten belirgin sekilde kucululmesi.
    Write-Host "4. mutlak parity (bilgi)"
    Write-Host ("     preview  mean={0:F4}" -f $prev1)
    Write-Host ("     rendered mean={0:F4}" -f $rend1)
    if ($rend1 -gt 1e-6) {
        Write-Host ("     oran preview/rendered = {0:F3}   (2026-09-02 duzeltmeden ONCE: 1.449)" -f ($prev1 / $rend1))
    }
}
finally {
    Invoke-RtIpc post.set_exposure @{ exposure = [double]$post0.exposure } | Out-Null
    if ($script:CamBase) {
        Invoke-RtIpc camera.set_position @{ position = @($script:CamBase[0], $script:CamBase[1], $script:CamBase[2]) } | Out-Null
    }
    if ($quality0) { Invoke-RtIpc viewport.set_quality @{ preset = $quality0 } | Out-Null }
    if ($shading0) { Invoke-RtIpc viewport.set_shading @{ mode = $shading0 } | Out-Null }
    Invoke-RtIpc viewport.capture @{ enabled = $false } | Out-Null
}

Write-Host ""
if ($fails -eq 0) { Write-Host "TUM KAPILAR GECTI" -ForegroundColor Green }
else { Write-Host ("{0} KAPI KALDI" -f $fails) -ForegroundColor Red; exit 1 }
