# Probe-GeometrySlope.ps1 -- karenin GEOMETRIYE BAGLI kismi ile SABIT kismini
# ayirir. Cikti: her gecis icin `ms = sabit + egim x Mtri` ve orta noktanin
# olculmeden tahmini.
#
# ★★★★★ NEDEN BOYLE OLCULUYOR. Tek bir kare tablosu "main_pass 56 ms" der ve
#   bu sayi iki tamamen farkli seyin toplamidir: ucgen debisi (LOD ile inen) ve
#   piksel basina golgelendirme (LOD ile INMEYEN). Hangisine yatirim yapilacagi
#   ancak ucgen sayisi DEGISTIRILEREK ayrilir. 2026-09-10 olcumu: main_pass'in
#   33 ms'i sabit, yani LOD calismasi onun tek ms'ine dokunmaz.
#
# ★★★ Kalite preset'i ucgen sayisini degistiren TEK IPC kolu (proxy ikamesi).
#   Preset ayni zamanda golge/PCF butcelerini de degistirir; RT golge acikken
#   shadow_atlas SKIPPED oldugu icin bu karisim main_pass'a bulasmaz. RT golge
#   KAPALI iken bu script'in sonucu GECERSIZDIR -- uyari basar.
#
# ★★ Raster viewport yalnizca kirliyken cizer: kare sayisi = kamera yazmasi
#    sayisi. viewport.render_frames path tracer'i surer, buraya UYMAZ.
#
# Kullanim:
#   .\scripts\ipc\Probe-GeometrySlope.ps1
#   .\scripts\ipc\Probe-GeometrySlope.ps1 -Frames 40 -IncludeFull

param(
    [int]$Frames = 30,
    [int]$WarmupFrames = 10,
    # 'full' proxy ikamesini tamamen kapatir: ucuncu ve en uzak nokta, uydurmayi
    # dogrulayan kol. Yavas sahnelerde kapatilabilir (o zaman iki nokta kalir ve
    # orta nokta dogrulamasi YAPILAMAZ).
    [switch]$SkipFull
)

$ErrorActionPreference = 'Stop'
Import-Module "$PSScriptRoot\RtIpc.psm1" -Force

$restorePreset = (Invoke-RtIpc viewport.quality).preset
$shading = (Invoke-RtIpc viewport.shading).mode
$rt = Invoke-RtIpc viewport.rt_shadow

if ($shading -ne 'material' -and $shading -ne 'rendered') {
    Write-Warning "shading='$shading'. Bu olcum material/rendered icin anlamli; solid modda materyal gecisi yok."
}
if (-not $rt.ready) {
    Write-Warning "RT golge hazir DEGIL (reason='$($rt.reason)'). shadow_atlas kareye girer ve preset onun butcesini de degistirir -> egim KIRLENIR."
}

function Measure-Window([string]$Label, [int]$N) {
    $cam = Invoke-RtIpc camera.get
    $p = @([double]$cam.position[0], [double]$cam.position[1], [double]$cam.position[2])
    Invoke-RtIpc viewport.reset_frame_timings | Out-Null
    for ($i = 0; $i -lt $N; $i++) {
        # Ayni degeri iki kez yazmak bir no-op olarak elenebilir; kucuk bir
        # salinim her yazmanin GERCEKTEN bir kare satin almasini garanti eder.
        $o = 0.02 * ($i % 5)
        Invoke-RtIpc camera.set_position @{ position = @(($p[0] + $o), $p[1], $p[2]) } | Out-Null
    }
    Start-Sleep -Milliseconds 1200
    Invoke-RtIpc camera.set_position @{ position = $p } | Out-Null
    $t = Invoke-RtIpc viewport.frame_timings
    $stage = @{}
    foreach ($s in $t.stages) { $stage[$s.name] = [double]$s.gpu_mean_ms }
    [pscustomobject]@{
        label   = $Label
        frames  = $t.frames
        mtri    = [double]$t.applied.visible_triangles / 1e6
        draws   = $t.applied.draw_calls
        frame   = [double]$t.frame_gpu_mean_ms
        main    = $stage['main_pass']
        prepass = $stage['depth_prepass']
        shadow  = $stage['rt_shadow']
        atlas   = $stage['shadow_atlas']
        warn    = $t.warnings
    }
}

$presets = @('performance', 'balanced')
if (-not $SkipFull) { $presets += 'full' }

$points = @()
foreach ($preset in $presets) {
    Invoke-RtIpc viewport.set_quality @{ preset = $preset } | Out-Null
    Measure-Window "warmup-$preset" $WarmupFrames | Out-Null
    $points += (Measure-Window $preset $Frames)
}
Invoke-RtIpc viewport.set_quality @{ preset = $restorePreset } | Out-Null
Measure-Window 'warmup-restore' $WarmupFrames | Out-Null

"`n=== olculen noktalar ==="
$points | Format-Table label, frames, @{n='Mtri';e={'{0:N2}' -f $_.mtri}}, draws,
    @{n='frame';e={'{0:N2}' -f $_.frame}}, @{n='main';e={'{0:N2}' -f $_.main}},
    @{n='prepass';e={'{0:N2}' -f $_.prepass}}, @{n='rt_shadow';e={'{0:N2}' -f $_.shadow}} -AutoSize

if ($points.Count -lt 2) { Write-Warning 'Iki noktadan az; uydurma yapilamaz.'; return }

# Uydurma UC nokta varken UC NOKTADAN DEGIL, iki UC noktadan yapilir; ortadaki
# nokta boylece bagimsiz bir DOGRULAMA olarak kalir. En kucuk kareler kullanmak
# orta noktayi uydurmaya dahil eder ve testi kendi kendini onaylar hale getirir.
$lo = $points[0]
$hi = $points[-1]
$dT = $hi.mtri - $lo.mtri
if ([math]::Abs($dT) -lt 0.5) { Write-Warning 'Ucgen sayisi kollar arasinda degismedi; proxy ikamesi kapali olabilir.'; return }

"`n=== dogrusal ayristirma  (ms = sabit + egim x Mtri) ==="
foreach ($name in 'main', 'prepass') {
    $slope = ($hi.$name - $lo.$name) / $dT
    $const = $lo.$name - $slope * $lo.mtri
    "{0,-9} sabit = {1,6:N2} ms   egim = {2,5:N3} ms/Mtri" -f $name, $const, $slope
    if ($points.Count -ge 3) {
        $mid = $points[1]
        $pred = $const + $slope * $mid.mtri
        $err = 100.0 * ($pred - $mid.$name) / $mid.$name
        "{0,-9}   orta nokta: tahmin {1,6:N2} / olculen {2,6:N2}  ({3,5:N1}%)" -f '', $pred, $mid.$name, $err
        if ([math]::Abs($err) -gt 5.0) {
            Write-Warning "$name dogrusal DEGIL (%$([math]::Round($err,1))). Preset yalnizca ucgen sayisini degil baska bir seyi de degistirmis olabilir."
        }
    }
}

$b = $points | Where-Object label -eq 'balanced' | Select-Object -First 1
if ($b) {
    $sM = ($hi.main - $lo.main) / $dT
    $sP = ($hi.prepass - $lo.prepass) / $dT
    $geo = ($sM + $sP) * $b.mtri
    $fix = ($b.main + $b.prepass) - $geo
    "`n=== balanced karesinin bolunmesi ({0:N1} M ucgen) ===" -f $b.mtri
    "geometri debisi : {0,6:N1} ms   ({1:P0})" -f $geo, ($geo / $b.frame)
    "sabit golgelend.: {0,6:N1} ms   ({1:P0})" -f $fix, ($fix / $b.frame)
    "RT golge        : {0,6:N1} ms   ({1:P0})" -f $b.shadow, ($b.shadow / $b.frame)
    "kare toplami    : {0,6:N1} ms" -f $b.frame
}

"`nkalite preset'i geri alindi: $restorePreset"
