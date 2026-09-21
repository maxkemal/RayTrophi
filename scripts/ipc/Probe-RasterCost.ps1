<#
.SYNOPSIS
  "Kare 142 ms -- maliyet NEREDEN geliyor?" sorusunu bilesenlere ayirir.

.DESCRIPTION
  viewport.frame_telemetry zaten karenin parcalarini yayinliyor:
      cpu_record_ms + slot_wait_ms + submit_ms + host_read_ms + present_ms
  Bu betik onlari DOGAL goruntu dongusunden toplar ve en onemli tureti
  hesaplar:

  ★★★ ARTIK (residual) = frame_ms - (yukaridakilerin toplami).
  Artik buyukse cevap "su alan buyuk" DEGILDIR; cevap **aletin kor oldugu**dur.
  Bir bilesen tablosunun toplami frame_ms'i aciklamiyorsa, o tabloya bakip
  "darbogaz present" demek olcum degil tahmindir.

  ★★★ IKI REJIM AYRI OLCULUR ve karistirilmamalidir:
    A) SUREKLI  -- yalnizca kamera oynar, geometri sabit. Etkilesimli maliyet.
    B) DEGISIM  -- sahne/kalite degisir, geometri yeniden kurulur. Tek seferlik.
  40 ms'i asan kareler cogunlukla B'dir; B'yi A sanip "raster yavas" demek
  bu deponun bilinen hata sinifi (uretici != tuketici).

  ★★ resource_drains SUREKLI rejimde ~0 OLMALI: kamera oynamak GPU'nun okudugu
  bir kaynagi mutasyona ugratmamali. Kare basina ~1 cikarsa iki slotlu halka
  hic ortusme uretmiyor demektir ve bu, cizim maliyetinden bagimsiz bir aciktir.

  ★★★ DUYARLILIK KAPISI: hicbir kol sayiyi kimildatmiyorsa betik "GECTI" demez,
  "OLCULEMEDI" der. Basarisizligi goremeyen kapi gecmesiyle hicbir sey soylemez.
#>
[CmdletBinding()]
param(
    [int]$Samples = 24,
    [int]$SettleMs = 140,
    [double]$NudgeMetres = 0.01,
    [switch]$SkipLevers,
    # B REJIMI: geometri/pipeline degistiren islemlerden SONRA zirve kareyi ara.
    # 40 ms'i asan kareler burada yasar; SUREKLI rejimde degil.
    [switch]$ChangeSweep,
    [int]$SpikePollMs = 25,
    [int]$SpikeWindowMs = 2500
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'RtIpc.psm1') -Force

function Get-Tel { Invoke-RtIpc viewport.frame_telemetry @{} }

# Bir rejimde N gecerli kare topla.
# ★ Donmus kare REDDEDILIR: frames_submitted artmadiysa bu bir olcum degildir.
function Measure-Regime {
    param([int]$Count, [double]$BaseX, [double]$BaseY, [double]$BaseZ)

    $rows = @(); $skipped = 0
    for ($i = 0; $i -lt ($Count + 1); $i++) {
        $before = Get-Tel                      # ★ SIRA: sayac ONCE okunur
        $d = $NudgeMetres * (($i % 2) * 2 - 1)
        Invoke-RtIpc camera.set_position @{ position = @(($BaseX + $d), $BaseY, $BaseZ) } | Out-Null
        Start-Sleep -Milliseconds $SettleMs
        $a = Get-Tel
        if ($a.frames_submitted -le $before.frames_submitted) { $skipped++; continue }
        if ($i -eq 0) { continue }             # isinma

        $parts = [double]$a.cpu_record_ms + [double]$a.slot_wait_ms + [double]$a.submit_ms +
                 [double]$a.image_readback_ms + [double]$a.host_read_ms + [double]$a.present_ms
        $rows += [pscustomobject]@{
            FrameMs    = [double]$a.frame_ms
            CpuRecMs   = [double]$a.cpu_record_ms
            SlotWaitMs = [double]$a.slot_wait_ms
            SubmitMs   = [double]$a.submit_ms
            ReadbackMs = [double]$a.image_readback_ms
            HostReadMs = [double]$a.host_read_ms
            PresentMs  = [double]$a.present_ms
            PartsMs    = $parts
            ResidualMs = [double]$a.frame_ms - $parts
            PostMs     = [double]$a.display_post_ms
            TexUpMs    = [double]$a.display_texture_upload_ms
            LoopMs     = [double]$a.display_loop_period_ms
            dSubmitted = [int]($a.frames_submitted - $before.frames_submitted)
            dDrains    = [int]($a.resource_drains  - $before.resource_drains)
            dStale     = [int]($a.stale_presents   - $before.stale_presents)
            VisTris    = [int64]$a.visible_triangles
            ProxyTris  = [int64]$a.proxy_triangles
            DrawCalls  = [int64]$a.draw_calls
        }
    }
    [pscustomobject]@{ Rows = $rows; Skipped = $skipped }
}

function Avg($vals) { if (@($vals).Count -eq 0) { [double]::NaN } else { (@($vals) | Measure-Object -Average).Average } }
function Med($vals) {
    $s = @($vals | Sort-Object); if ($s.Count -eq 0) { return [double]::NaN }
    $s[[int]([math]::Floor($s.Count / 2))]
}
function Line($name, $vals) {
    $s = @($vals | Sort-Object)
    "  {0,-15} medyan={1,8:F3}  ort={2,8:F3}  min={3,8:F3}  max={4,8:F3}" -f `
        $name, (Med $vals), (Avg $vals), $s[0], $s[-1]
}

function Report {
    param([string]$Title, $R)
    $rows = $R.Rows
    Write-Host ""
    Write-Host "=== $Title  (n=$($rows.Count), atlanan=$($R.Skipped)) ==="
    if ($rows.Count -eq 0) { Write-Host "  ORNEK YOK -- viewport kare cizmedi."; return }
    Write-Host "-- backend (renderProgressive) --"
    Line 'frame_ms'       @($rows.FrameMs)
    Line 'cpu_record'     @($rows.CpuRecMs)
    Line 'slot_wait'      @($rows.SlotWaitMs)
    Line 'submit'         @($rows.SubmitMs)
    Line 'image_readback' @($rows.ReadbackMs)
    Line 'host_read'      @($rows.HostReadMs)
    Line 'present'        @($rows.PresentMs)
    Write-Host "-- turetilmis --"
    Line 'PARCA TOPLAMI'  @($rows.PartsMs)
    Line 'ARTIK (*)'      @($rows.ResidualMs)
    Write-Host "-- ekran yolu (ana dongu, backend DISINDA) --"
    Line 'display_post'   @($rows.PostMs)
    Line 'tex_upload'     @($rows.TexUpMs)
    Line 'loop_period'    @($rows.LoopMs)
    Write-Host "-- halka / geometri --"
    Write-Host ("  gonderilen={0}  drain={1} ({2:F2}/kare)  stale={3}  vis_tris={4}  proxy_tris={5}  draw={6}" -f `
        (@($rows.dSubmitted) | Measure-Object -Sum).Sum,
        (@($rows.dDrains) | Measure-Object -Sum).Sum,
        (Avg @($rows.dDrains)),
        (@($rows.dStale) | Measure-Object -Sum).Sum,
        $rows[-1].VisTris, $rows[-1].ProxyTris, $rows[-1].DrawCalls)
}

# -- Baglam ------------------------------------------------------------------
$status = Invoke-RtIpc viewport.status @{}
if (-not $status.available) { throw "Viewport yok." }
$shading0 = Invoke-RtIpc viewport.shading @{}
$quality0 = Invoke-RtIpc viewport.quality @{}
$t0 = Get-Tel
if (-not $t0.available) { throw "Raster telemetrisi yok." }

$mb = $t0.width * $t0.height * 4 / 1MB
Write-Host ("backend={0}  mod={1}  kalite={2}  {3}x{4}  kare={5:N2} MB  async={6}  slotlar={7}" -f `
    $status.backend, $shading0.mode, $quality0.preset, $t0.width, $t0.height, $mb, $t0.async_present, $t0.slot_count)
Write-Host ("geometri: total_instances={0}  draw_calls={1}  full_tris={2}  proxy_tris={3}  gpu_cull={4}  global_inst_buf={5}" -f `
    $t0.total_instances, $t0.draw_calls, $t0.full_triangles, $t0.proxy_triangles, $t0.gpu_culling, $t0.global_instance_buffer)

# ★★ gpu_culling=false TEK BASINA ariza DEGIL. Ariza global_instance_buffer
#    ACIKKEN gpu_culling'in kurulamamasidir: o zaman ne CPU culling ne proxy
#    kosar. Global buffer kapaliyken eski CPU yolu ikisini de yapiyor.
if ($t0.global_instance_buffer -and -not $t0.gpu_culling) {
    Write-Host "  *** ARIZA: global instance buffer ACIK ama GPU culling kurulamadi -> sahne CULLING'SIZ ve PROXY'SIZ." -ForegroundColor Red
} elseif (-not $t0.gpu_culling) {
    Write-Host "  (gpu_cull=0 + global_inst_buf=0 => eski CPU yolu; culling ve proxy CPU'da. Ariza DEGIL.)"
}

$cam = Invoke-RtIpc camera.get @{}
$bx = [double]$cam.position[0]; $by = [double]$cam.position[1]; $bz = [double]$cam.position[2]

# -- A) SUREKLI rejim: mevcut ayarlarda --------------------------------------
$base = Measure-Regime -Count $Samples -BaseX $bx -BaseY $by -BaseZ $bz
Report "A) SUREKLI  mod=$($shading0.mode) kalite=$($quality0.preset)" $base

$verdicts = @()

if (-not $SkipLevers) {
    # Kol 1: gölgeleme modu (fragment/materyal shader maliyeti)
    $other = if ($shading0.mode -eq 'solid') { 'material' } else { 'solid' }
    Invoke-RtIpc viewport.set_shading @{ mode = $other } | Out-Null
    Start-Sleep -Milliseconds 400
    $alt = Measure-Regime -Count $Samples -BaseX $bx -BaseY $by -BaseZ $bz
    Report "B) SUREKLI  mod=$other" $alt
    Invoke-RtIpc viewport.set_shading @{ mode = $shading0.mode } | Out-Null
    Start-Sleep -Milliseconds 400

    # Kol 2: kalite preset (proxy LOD / shader kademesi)
    $qAlt = if ($quality0.preset -eq 'performance') { 'full' } else { 'performance' }
    Invoke-RtIpc viewport.set_quality @{ preset = $qAlt } | Out-Null
    Start-Sleep -Milliseconds 400
    $qm = Measure-Regime -Count $Samples -BaseX $bx -BaseY $by -BaseZ $bz
    Report "C) SUREKLI  kalite=$qAlt (mod=$($shading0.mode))" $qm
    Invoke-RtIpc viewport.set_quality @{ preset = $quality0.preset } | Out-Null
    Start-Sleep -Milliseconds 400

    # -- DUYARLILIK KAPISI ---------------------------------------------------
    $mA = Med @($base.Rows.FrameMs); $mB = Med @($alt.Rows.FrameMs); $mC = Med @($qm.Rows.FrameMs)
    $spanShading = [math]::Abs($mA - $mB)
    $spanQuality = [math]::Abs($mA - $mC)
    Write-Host ""
    Write-Host "=== KOL DUYARLILIGI ==="
    Write-Host ("  mod    {0,-12} {1,7:F3} ms  ->  {2,-12} {3,7:F3} ms   fark={4:F3}" -f $shading0.mode, $mA, $other, $mB, $spanShading)
    Write-Host ("  kalite {0,-12} {1,7:F3} ms  ->  {2,-12} {3,7:F3} ms   fark={4:F3}" -f $quality0.preset, $mA, $qAlt, $mC, $spanQuality)
    if ($spanShading -lt 0.05 -and $spanQuality -lt 0.05) {
        Write-Host "  *** OLCULEMEDI: iki kol da sayiyi kimildatmadi. Ya sahne cizim maliyetiyle sinirli DEGIL, ya da olcum kareyi yakalamiyor." -ForegroundColor Yellow
        $verdicts += 'OLCULEMEDI: kollar olu'
    } else {
        Write-Host "  alet DUYARLI (en az bir kol frame_ms'i kimildatti)."
    }
}

# -- SONUC -------------------------------------------------------------------
$rows = $base.Rows
if ($rows.Count -gt 0) {
    $f = Med @($rows.FrameMs)
    $res = Med @($rows.ResidualMs)
    $drainPerFrame = Avg @($rows.dDrains)
    $loop = Med @($rows.LoopMs)
    $display = (Med @($rows.PostMs)) + (Med @($rows.TexUpMs))

    Write-Host ""
    Write-Host "=== SONUC (SUREKLI rejim, medyan) ==="
    Write-Host ("  backend frame_ms      {0,8:F3}" -f $f)
    Write-Host ("  ekran yolu (post+tex) {0,8:F3}" -f $display)
    Write-Host ("  ana dongu periyodu    {0,8:F3}   <- kullanicinin HISSETTIGI sayi" -f $loop)
    if ($f -gt 0) {
        Write-Host ("  (*) artik / frame_ms  {0,7:F1}%   (buyukse alet KOR, bilesen tablosu cevap DEGIL)" -f (100.0 * $res / $f))
    }
    if ($drainPerFrame -ge 0.5) {
        Write-Host ("  ** resource_drains {0:F2}/kare -- kamera oynatmak GPU-okunur kaynagi mutasyona ugratiyor." -f $drainPerFrame) -ForegroundColor Yellow
        $verdicts += 'drain/kare >= 0.5'
    }
    # ★ Ana dongu periyodu backend karesinden COK buyukse darbogaz raster DEGIL.
    if ($loop -gt ($f * 3) -and $loop -gt 8.0) {
        Write-Host ("  *** Ana dongu ({0:F2} ms) backend karesinin ({1:F2} ms) {2:F1} kati -- maliyet RASTER CIZIMINDE DEGIL." -f $loop, $f, ($loop / $f)) -ForegroundColor Cyan
        $verdicts += 'dongu >> backend karesi'
    }
}

Invoke-RtIpc camera.set_position @{ position = @($bx, $by, $bz) } | Out-Null

# -- B REJIMI: DEGISIM kareleri ----------------------------------------------
# ★★★ Bu bolum SUREKLI rejimle AYNI TABLOYA yazilamaz. Oradaki sayi
#     "etkilesimli maliyet", buradaki "bir kereye mahsus kurulum". Ikisini
#     ortalamak, 40 ms'i asan kareyi 1 ms'lik kareler icinde eritir ve
#     "her sey yolunda" der.
if ($ChangeSweep) {
    # Bir islemi uygula, sonra kisa araliklarla telemetriyi tarayarak ZIRVE
    # kareyi yakala. Telemetri SON kareyi bildirir, o yuzden zirve ancak
    # yeterince sik bakilirsa gorulur.
    function Sweep-Change {
        param([string]$Name, [scriptblock]$Apply)

        $t0 = Get-Tel
        & $Apply | Out-Null
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $peak = $null; $peakMs = -1.0; $seen = 0
        while ($sw.ElapsedMilliseconds -lt $SpikeWindowMs) {
            $a = Get-Tel
            if ($a.frames_submitted -gt $t0.frames_submitted) {
                $seen++
                if ([double]$a.frame_ms -gt $peakMs) { $peakMs = [double]$a.frame_ms; $peak = $a }
            }
            Start-Sleep -Milliseconds $SpikePollMs
        }
        $t1 = Get-Tel
        if ($null -eq $peak) {
            Write-Host ("  {0,-26} KARE YOK -- islem yeni kare uretmedi." -f $Name) -ForegroundColor Yellow
            return
        }
        $parts = [double]$peak.cpu_record_ms + [double]$peak.slot_wait_ms + [double]$peak.submit_ms +
                 [double]$peak.image_readback_ms + [double]$peak.host_read_ms + [double]$peak.present_ms
        $res = $peakMs - $parts
        $pct = if ($peakMs -gt 0) { 100.0 * $res / $peakMs } else { 0.0 }
        Write-Host ("  {0,-26} zirve={1,8:F2} ms  parca={2,6:F2}  ARTIK={3,8:F2} ({4,5:F1}%)  kare={5,3}  drain={6,3}" -f `
            $Name, $peakMs, $parts, $res, $pct,
            [int]($t1.frames_submitted - $t0.frames_submitted),
            [int]($t1.resource_drains - $t0.resource_drains))
    }

    Write-Host ""
    Write-Host "=== D) DEGISIM rejimi -- zirve kare ve ARTIK ==="
    Write-Host ("  (poll={0} ms, pencere={1} ms; telemetri SON kareyi bildirir)" -f $SpikePollMs, $SpikeWindowMs)

    $objs = Invoke-RtIpc scene.list_objects @{}
    $victim = if ($objs.Count -gt 0) { [string]$objs[0] } else { $null }

    Sweep-Change 'kamera fov (kontrol)' { Invoke-RtIpc camera.set_fov @{ fov = 46.0 } }
    Sweep-Change 'kamera fov geri'      { Invoke-RtIpc camera.set_fov @{ fov = [double]$cam.fov } }
    Sweep-Change 'shading -> solid'     { Invoke-RtIpc viewport.set_shading @{ mode = 'solid' } }
    Sweep-Change 'shading -> material'  { Invoke-RtIpc viewport.set_shading @{ mode = 'material' } }
    Sweep-Change 'kalite -> performance' { Invoke-RtIpc viewport.set_quality @{ preset = 'performance' } }
    Sweep-Change 'kalite -> full'       { Invoke-RtIpc viewport.set_quality @{ preset = 'full' } }
    if ($victim) {
        $tr = Invoke-RtIpc scene.get_transform @{ name = $victim }
        $p = $tr.translation
        Sweep-Change "set_transform ($victim)" {
            Invoke-RtIpc scene.set_transform @{ name = $victim; translation = @(([double]$p[0] + 0.05), [double]$p[1], [double]$p[2]) }
        }
        Sweep-Change 'set_transform geri' {
            Invoke-RtIpc scene.set_transform @{ name = $victim; translation = @([double]$p[0], [double]$p[1], [double]$p[2]) }
        }
    }
    Invoke-RtIpc viewport.set_shading @{ mode = $shading0.mode } | Out-Null
    Invoke-RtIpc viewport.set_quality @{ preset = $quality0.preset } | Out-Null

    Write-Host ""
    Write-Host "  ★ ARTIK yuzdesi buyuk bir zirvede: maliyet renderProgressive'in ICINDE ama"
    Write-Host "    OLCULEN hicbir bolumde degil (geometri senkronu / instance taramasi /"
    Write-Host "    descriptor guncellemesi / drenaj). Bilesen tablosu orada CEVAP DEGIL."
}

# -- E) ANA DONGU DAGILIMI (rt.perf loop.*) ----------------------------------
# ★★★ Yukaridaki butun tablolar backend'in `renderProgressive`'ini ve ekran
#     yolunun iki kopyasini olcer. Bu bolum karenin GERI KALANINI olcer, ve
#     olculen sahnede geri kalan karenin %95'iydi.
Write-Host ""
Write-Host "=== E) ANA DONGU DAGILIMI (rt.perf loop.*) ==="
Invoke-RtIpc perf.reset @{} | Out-Null

# Surekli surus: her IPC turu bir kare tetikler, cevap kare islenene kadar
# donmez. Yani bu dongunun HIZI uygulamanin kare hizidir.
$sw = [System.Diagnostics.Stopwatch]::StartNew(); $calls = 0
while ($sw.ElapsedMilliseconds -lt 3000) {
    $d = $NudgeMetres * (($calls % 2) * 2 - 1)
    Invoke-RtIpc camera.set_position @{ position = @(($bx + $d), $by, $bz) } | Out-Null
    $calls++
}
$elapsed = $sw.Elapsed.TotalSeconds
Invoke-RtIpc camera.set_position @{ position = @($bx, $by, $bz) } | Out-Null

$sections = Invoke-RtIpc perf.list @{}
$loop = @($sections | Where-Object { $_.name -like 'loop.*' })
if ($loop.Count -eq 0) {
    Write-Host "  loop.* bolumu YOK -- ana dongu enstrumantasyonu bu binary'de derlenmemis." -ForegroundColor Yellow
    $verdicts += 'loop.* enstrumantasyonu yok'
} else {
    $frame = $loop | Where-Object { $_.name -eq 'loop.frame' }
    Write-Host ("  {0,-28} {1,>8} {2,>9} {3,>9} {4,>7}" -f 'bolum', 'ort ms', 'max ms', 'toplam', 'sayi')
    foreach ($s in ($loop | Sort-Object { -$_.total_ms })) {
        $mean = if ($s.count -gt 0) { $s.total_ms / $s.count } else { 0.0 }
        $mark = if ($s.name -eq 'loop.frame') { ' <= TUM ITERASYON' }
                elseif ($s.name -eq 'loop.throttle_sleep') { ' <= UYKU, IS DEGIL' } else { '' }
        Write-Host ("  {0,-28} {1,8:F3} {2,9:F2} {3,9:F1} {4,7}{5}" -f $s.name, $mean, $s.max_ms, $s.total_ms, $s.count, $mark)
    }
    if ($frame -and $frame.count -gt 0) {
        $frameMean = $frame.total_ms / $frame.count
        $sleep = $loop | Where-Object { $_.name -eq 'loop.throttle_sleep' }
        $sleepMean = if ($sleep -and $sleep.count -gt 0) { $sleep.total_ms / $sleep.count } else { 0.0 }
        $parts = ($loop | Where-Object { $_.name -ne 'loop.frame' } |
                  ForEach-Object { if ($_.count -gt 0) { $_.total_ms / $_.count } else { 0.0 } } |
                  Measure-Object -Sum).Sum
        Write-Host ""
        Write-Host ("  kare (loop.frame)      {0,8:F3} ms   -> {1,5:F1} fps" -f $frameMean, (1000.0 / [math]::Max($frameMean, 0.001)))
        Write-Host ("  bunun UYKUSU           {0,8:F3} ms" -f $sleepMean)
        Write-Host ("  gercek IS              {0,8:F3} ms   -> {1,5:F1} fps tavani" -f ($frameMean - $sleepMean), (1000.0 / [math]::Max($frameMean - $sleepMean, 0.001)))
        Write-Host ("  parcalarin toplami     {0,8:F3} ms   ARTIK={1:F3} ms" -f $parts, ($frameMean - $parts))
        Write-Host ("  IPC turu               {0,8:F3} ms/cagri ({1} cagri / {2:F1} s)" -f (1000.0 * $elapsed / [math]::Max($calls,1)), $calls, $elapsed)
        if ($sleepMean -gt 1.0) {
            Write-Host "  ★★★ Uyku > 1 ms: bu oturum tier0'a GIRMEDI. camera_moved yalnizca fare/klavyeden" -ForegroundColor Yellow
            Write-Host "      set edilir, yani IPC'den olculen kare hizi motorun hizi DEGILDIR." -ForegroundColor Yellow
            $verdicts += 'olcum tier0 disinda (uyku dahil)'
        }
    }
}

Write-Host ""
if ($verdicts.Count -gt 0) { Write-Host ("ISARETLER: " + ($verdicts -join ' | ')) }
