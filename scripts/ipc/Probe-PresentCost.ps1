<#
.SYNOPSIS
  Raster viewport sunum maliyetini olcer: karenin ne kadari pikselleri GPU'dan
  CPU'ya indirmekle ve SDL yuzeyine kopyalamakla geciyor.

.DESCRIPTION
  Sorulan soru: "pikselleri CPU'ya indirmeyi birakirsak ne kazaniriz?"
  Cevap iki alanda:
    host_read_ms : GPU readback buffer -> framebuffer (invalidate + memcpy)
    present_ms   : framebuffer -> SDL_Surface (memcpy)

  ★★★ TUZAK 1 -- viewport.render_frames BU OLCUM ICIN KULLANILAMAZ.
  O yol `render_progressive_pass(nullptr, nullptr, ...)` cagiriyor, yani SDL
  surface NULL; `presentCachedRasterFrame` icindeki memcpy hic calismiyor.
  Onunla olcmek, olctugun seyin olmadigi bir yolu olcmektir. Bu yuzden burada
  kare DOGAL goruntu dongusunden aliniyor: kamera oynatilir, beklenir, okunur.

  ★★★ TUZAK 2 -- sahne temizse telemetri SON GECERLI karede DONAR. Ayni sayiyi
  10 kez okuyup "kararli" sanmak bu deponun bilinen hatasi. Betik her turda
  frames_submitted'in ARTTIGINI dogrular; artmadiysa turu ATAR.
#>
[CmdletBinding()]
param(
    [int]$Samples = 20,
    [int]$SettleMs = 150,
    [double]$NudgeMetres = 0.01
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'RtIpc.psm1') -Force

$status = Invoke-RtIpc viewport.status @{}
if (-not $status.available) { throw "Viewport yok." }
$shading = Invoke-RtIpc viewport.shading @{}
$t0 = Invoke-RtIpc viewport.frame_telemetry @{}
if (-not $t0.available) { throw "Raster telemetrisi yok (interaktif raster viewport kurulmamis)." }

$mb = $t0.width * $t0.height * 4 / 1MB
Write-Host ("Mod: {0}   backend: {1}   {2}x{3}   kare basina {4:N2} MB   async_present={5}" -f `
    $shading.mode, $status.backend, $t0.width, $t0.height, $mb, $t0.async_present)
Write-Host ""

$cam = Invoke-RtIpc camera.get @{}
$bx = [double]$cam.position[0]; $by = [double]$cam.position[1]; $bz = [double]$cam.position[2]

$rows = @()
$skipped = 0
for ($i = 0; $i -lt ($Samples + 1); $i++) {
    # ★ SIRA ONEMLI: sayac once okunur. set_position'dan SONRA okumak, kare
    # zaten cizilmisse artisi yutar ve her tur "yeni kare yok" diye atilir.
    $before = Invoke-RtIpc viewport.frame_telemetry @{}

    $d = $NudgeMetres * (($i % 2) * 2 - 1)
    $nx = $bx + $d
    Invoke-RtIpc camera.set_position @{ position = @($nx, $by, $bz) } | Out-Null

    Start-Sleep -Milliseconds $SettleMs
    $after = Invoke-RtIpc viewport.frame_telemetry @{}

    # ★ Donmus kare reddedilir: yeni kare gonderilmediyse bu bir olcum degildir.
    if ($after.frames_submitted -le $before.frames_submitted) { $skipped++; continue }
    if ($i -eq 0) { continue }   # isinma

    $rows += [pscustomobject]@{
        N          = $rows.Count + 1
        FrameMs    = [double]$after.frame_ms
        CpuRecMs   = [double]$after.cpu_record_ms
        SubmitMs   = [double]$after.submit_ms
        HostReadMs = [double]$after.host_read_ms
        PresentMs  = [double]$after.present_ms
        PostMs     = if ($after.display_available) { [double]$after.display_post_ms } else { [double]::NaN }
        TexUpMs    = if ($after.display_available) { [double]$after.display_texture_upload_ms } else { [double]::NaN }
        LoopMs     = if ($after.display_available) { [double]$after.display_loop_period_ms } else { [double]::NaN }
        NoopCopy   = if ($after.display_available) { [bool]$after.display_post_was_noop_copy } else { $null }
        dSubmitted = [int]($after.frames_submitted - $before.frames_submitted)
        dConsumed  = [int]($after.frames_consumed  - $before.frames_consumed)
        dStale     = [int]($after.stale_presents   - $before.stale_presents)
    }
}

Invoke-RtIpc camera.set_position @{ position = @($bx, $by, $bz) } | Out-Null

$rows | Format-Table -AutoSize
if ($skipped -gt 0) { Write-Host ("[{0} tur atlandi: yeni kare gonderilmedi]" -f $skipped) }
if ($rows.Count -eq 0) { throw "Hic gecerli ornek yok -- viewport hic kare cizmedi." }

function Stat($name, $vals) {
    $sorted = @($vals | Sort-Object)
    "{0,-13} ort={1,7:F3}  medyan={2,7:F3}  min={3,7:F3}  max={4,7:F3}" -f `
        $name, ($vals | Measure-Object -Average).Average, $sorted[[int]($sorted.Count/2)], $sorted[0], $sorted[-1]
}

$hasDisplay = ($rows | Where-Object { -not [double]::IsNaN($_.PostMs) }).Count -gt 0

Write-Host ""
Write-Host "=== backend (renderProgressive icinde) ms/kare ==="
Stat 'frame_ms'      @($rows.FrameMs)
Stat 'cpu_record_ms' @($rows.CpuRecMs)
Stat 'submit_ms'     @($rows.SubmitMs)
Stat 'host_read_ms'  @($rows.HostReadMs)
Stat 'present_ms'    @($rows.PresentMs)

$fAvg = ($rows.FrameMs | Measure-Object -Average).Average
$d1   = (@($rows.HostReadMs) | Measure-Object -Average).Average
$d2   = (@($rows.PresentMs)  | Measure-Object -Average).Average
$copies = 2
$dAvg = $d1 + $d2

if ($hasDisplay) {
    Write-Host ""
    Write-Host "=== ekran yolu (ana dongu, backend'in DISINDA) ms/kare ==="
    Stat 'post_ms'          @($rows.PostMs)
    Stat 'texture_upload'   @($rows.TexUpMs)
    Stat 'loop_period_ms'   @($rows.LoopMs)
    $p1 = (@($rows.PostMs)   | Measure-Object -Average).Average
    $p2 = (@($rows.TexUpMs)  | Measure-Object -Average).Average
    $dAvg += $p1 + $p2
    $copies = 4
    $noop = @($rows | Where-Object { $_.NoopCopy -eq $true }).Count
    Write-Host ("post gecisi {0}/{1} karede NO-OP DUZ KOPYA idi (kaldirilabilir is)" -f $noop, $rows.Count)
    $loopAvg = (@($rows.LoopMs) | Measure-Object -Average).Average
} else {
    Write-Host ""
    Write-Host "! display_available false -- ekran yolu OLCULMEDI (eski binary?)."
    Write-Host "  Asagidaki toplam yalnizca backend yarisini kapsar, yani ALT SINIRDIR."
    $loopAvg = 0.0
}

$bw = if ($dAvg -gt 0) { ($copies * $mb / 1024.0) / ($dAvg / 1000.0) } else { 0 }
Write-Host ""
Write-Host ("CPU'ya inis TOPLAMI ({0} tam-kare gecis) : ort {1:F3} ms" -f $copies, $dAvg)
Write-Host ("  -> backend karesinin %{0:F1}'i" -f (100.0*$dAvg/[Math]::Max($fAvg,1e-6)))
if ($loopAvg -gt 0) { Write-Host ("  -> ana dongu periyodunun %{0:F1}'i ({1:F2} ms)" -f (100.0*$dAvg/$loopAvg), $loopAvg) }
Write-Host ("  -> kare basina {0:N2} MB x{1} = efektif {2:F1} GB/s" -f $mb, $copies, $bw)
Write-Host ""
Write-Host "* Kopya piksel sayisiyla olcekleniyor: 1080p'de bu sayinin ~1/4'unu bekle."
Write-Host "* host_read_ms 0.00 iken dConsumed>0 ise alet yalan soyluyor: 30 MB memcpy sifir ms surmez."
