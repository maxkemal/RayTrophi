<#
.SYNOPSIS
    Physical Sky kadranlarini surmenin kare basina maliyeti. "Slider'i tutunca
    TDR oluyor / her parametre asiri maliyetli" sikayetinin OLCUSU.

.DESCRIPTION
    Bu probe fare surtunmesini taklit eder: LUT'u kirleten bir alani art arda
    degistirir ve telemetriyi okur.

    ★★★ TEK AYIRICI OLCU, LUT'u kirleten bir alan ile KIRLETMEYEN bir alanin
    ayni sayida cagrida ne kadar surdugudur. Ikisi de setWorldData'ya gider,
    ikisi de dunyayi kirletir; TEK fark atmosfer LUT'unun yeniden uretilmesi.

      control  : world.set_sun_size   -> lut_dirty KURMAZ
      subject  : world.set_atmosphere -> her alan lut_dirty KURAR

    ★★ Kontrol icin `sun_intensity` KULLANILMAZ. Denendi ve YANLIS: RtApi
    tarafi onu yonlu isiga da yaziyor (syncDirectionalLightToWorldSun) ve
    `g_lights_dirty` kuruyor, yani isik buffer'i + golge atlasi isi ekliyor.
    Kontrolun subject'ten PAHALI cikmasi olcumu okunamaz yapiyordu.
    `sun_size` yalnizca setNishitaParams'a gider ve LUT listesinde degildir.

    ★★★ ISINMA ZORUNLU. Ilk olcum MaterialPreview'in ilk-kare kaynak
    kurulumunu (pipeline/golge atlasi/raster geometri) yutuyor: olculdu,
    duzenleme basina 500 ms'ti ve ikinci turda 5 ms'ye dustu. Isinmasiz bir
    tek tur, hangi alanin once olculdugune gore sira degistirir.

    ★★ SAYILAR YALNIZCA VIEWPORT CIZERKEN ANLAMLIDIR. Pencere odakta degilse
    uygulama kare gondermez ve maliyet "ucuz" gorunur -- kare dongusune kor
    bir olcum. Script bunu tespit edip UYARIR, sessizce yesil raporlamaz.

    ★★★ NE GORMEN GEREK
      - ms/edit: control ile subject BENZER. Olculen saglikli taban: ikisi de
        ~5 ms (bir kare periyodu), maks bir vsync karesi.
      - Subject birkac kat buyukse: LUT ya CPU'da pisiyor (SkyView 256x128 x
        128 adim x ic ice 40 adim ~ 168M exp) ya da her duzenleme cihazi
        drenaj ediyor. Once log'a bak:
        `[Vulkan] Atmosphere LUT compute pipeline ready (VulkanViewportBackend).`
      - device_lost: backend 'vulkan' kalmali. 'cpu' olduysa cihaz kaybedildi.
      - drains/frame: 2026-09-02'den beri sayac YALNIZCA gercekten bloklaninca
        artiyor. Olculen saglikli taban ~0.06/kare.

.EXAMPLE
    .\Probe-AtmosphereCost.ps1
    .\Probe-AtmosphereCost.ps1 -Iterations 60 -Field temperature
#>
[CmdletBinding()]
param(
    [int]$Iterations = 30,
    [ValidateSet('air_density', 'dust_density', 'ozone_density', 'humidity',
                 'temperature', 'mie_density', 'rayleigh_density',
                 'atmosphere_height', 'mie_anisotropy')]
    [string]$Field = 'air_density',
    # Olcum MaterialPreview'da yapilir (sikayetin geldigi mod). Bittiginde
    # baslangictaki mod geri yazilir.
    [switch]$KeepShading
)

Import-Module "$PSScriptRoot\RtIpc.psm1" -Force

# ★★★ MUTLAK adim, CARPAN DEGIL.
#   Ilk surum `base * (1 + i*0.02)` ile suruyordu ve bu SESSIZ bir yalan
#   uretiyordu: sahnede `dust_density = 0.0` ile karsilasildi, carpan sifiri
#   sifir birakti, setNishitaParams hicbir degisiklik gormedi, `lut_dirty` HIC
#   kurulmadi -- yani "subject" turu LUT'a hic dokunmadan kontrolle ayni cikip
#   YESIL raporlayacakti. Varsayilan bir olcum degildir; sifir bir taban ise
#   hic degildir. Asagida deger GERCEKTEN hareket etti mi diye ayrica bakiyoruz.
$FieldStep = @{
    air_density       = 0.05
    dust_density      = 0.05
    ozone_density     = 0.05
    humidity          = 0.02
    temperature       = 1.5
    mie_density       = 25.0
    rayleigh_density  = 100.0
    atmosphere_height = 500.0
    mie_anisotropy    = 0.01
}
$step = $FieldStep[$Field]

function Get-Telemetry {
    try { return Invoke-RtIpc viewport.frame_telemetry } catch { return $null }
}

function Invoke-Sweep([string]$Label, [scriptblock]$Apply, [int]$N) {
    $before = Get-Telemetry
    $times = New-Object System.Collections.Generic.List[double]
    $failures = 0
    for ($i = 0; $i -lt $N; $i++) {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        try { & $Apply $i } catch { $failures++ }
        $sw.Stop()
        $times.Add($sw.Elapsed.TotalMilliseconds)
    }
    $after = Get-Telemetry
    $submitted = 0; $drains = 0; $stale = 0
    if ($before -and $after) {
        $submitted = $after.frames_submitted - $before.frames_submitted
        $drains    = $after.resource_drains  - $before.resource_drains
        $stale     = $after.stale_presents   - $before.stale_presents
    }
    [pscustomobject]@{
        Label       = $Label
        Fail        = $failures
        MsAvg       = [math]::Round(($times | Measure-Object -Average).Average, 1)
        MsMax       = [math]::Round(($times | Measure-Object -Maximum).Maximum, 1)
        Submitted   = $submitted
        Drains      = $drains
        Stale       = $stale
        DrainsFrame = $(if ($submitted -gt 0) { [math]::Round($drains / $submitted, 2) } else { $null })
    }
}

# ── Onkosullar ──────────────────────────────────────────────────────────────
# Dunya Nishita degilse LUT hic uretilmez ve olcum 0 farkla yesil doner.
$world0 = Invoke-RtIpc world.get
if ($world0.mode -ne 'nishita') {
    throw "world.mode '$($world0.mode)'. Bu probe fiziksel gokyuzunu olcer; once: Invoke-RtIpc world.set_mode @{ mode = 'nishita' }"
}
$shading0 = (Invoke-RtIpc viewport.status).shading
if (-not $KeepShading -and $shading0 -ne 'material') {
    $null = Invoke-RtIpc viewport.set_shading @{ mode = 'material' }
    Start-Sleep -Milliseconds 300
}
$atmo0 = Invoke-RtIpc world.get_atmosphere
$base  = [double]$atmo0.$Field

Write-Host ("mode=nishita  shading={0}  field={1}  base={2}  step={3}  n={4}" -f
            (Invoke-RtIpc viewport.status).shading, $Field, $base, $step, $Iterations) -ForegroundColor Cyan

try {
    # ★ Deger GERCEKTEN hareket ediyor mu? Etmiyorsa olcum LUT'a hic dokunmaz
    #   ve "hizli" cikar -- yani sessizce yesil yalan soyler.
    $null = Invoke-RtIpc world.set_atmosphere @{ $Field = ($base + $step) }
    $moved = [double](Invoke-RtIpc world.get_atmosphere).$Field
    if ([math]::Abs($moved - $base) -lt ($step * 0.5)) {
        throw ("'{0}' {1} -> {2}: deger hareket etmedi. Bu haliyle olcum LUT'a dokunmaz ve YANLIS yesil verir." -f $Field, $base, $moved)
    }
    $null = Invoke-RtIpc world.set_atmosphere @{ $Field = $base }

    # ★★★ ISINMA: sonuc ATILIR. Ilk tur ilk-kare kaynak kurulumunu yutuyor.
    $null = Invoke-Sweep 'warmup' {
        param($i) $null = Invoke-RtIpc world.set_sun_size @{ sun_size = ($world0.sun_size + ($i % 5) * 0.002) }
    } 15

    # A/B/A/B: kalan surukleme etkisini iptal eder.
    $results = @()
    $results += Invoke-Sweep "A1 control sun_size (LUT temiz)" {
        param($i) $null = Invoke-RtIpc world.set_sun_size @{ sun_size = ($world0.sun_size + ($i % 10) * 0.005) }
    } $Iterations
    $results += Invoke-Sweep "B1 subject $Field (LUT kirli)" {
        param($i) $null = Invoke-RtIpc world.set_atmosphere @{ $Field = ($base + ($i % 10) * $step) }
    } $Iterations
    $results += Invoke-Sweep "A2 control sun_size (LUT temiz)" {
        param($i) $null = Invoke-RtIpc world.set_sun_size @{ sun_size = ($world0.sun_size + ($i % 10) * 0.005) }
    } $Iterations
    $results += Invoke-Sweep "B2 subject $Field (LUT kirli)" {
        param($i) $null = Invoke-RtIpc world.set_atmosphere @{ $Field = ($base + ($i % 10) * $step) }
    } $Iterations
}
finally {
    $null = Invoke-RtIpc world.set_atmosphere @{ $Field = $base }
    $null = Invoke-RtIpc world.set_sun_size @{ sun_size = $world0.sun_size }
    if (-not $KeepShading -and $shading0 -ne 'material') {
        $null = Invoke-RtIpc viewport.set_shading @{ mode = $shading0 }
    }
}

$results | Format-Table -AutoSize

$control = (($results | Where-Object Label -like 'A*' | Measure-Object MsAvg -Average).Average)
$subject = (($results | Where-Object Label -like 'B*' | Measure-Object MsAvg -Average).Average)
$submitted = ($results | Measure-Object Submitted -Sum).Sum
$backend = (Invoke-RtIpc viewport.status).backend

if ($backend -ne 'vulkan') {
    Write-Host "FAIL: backend '$backend'. Olcum sirasinda Vulkan birakildi -- cihaz kaybi." -ForegroundColor Red
}
elseif ($submitted -le 0) {
    Write-Host "UYARI: olcum sirasinda HIC kare gonderilmedi. Viewport cizmiyorsa bu" -ForegroundColor Yellow
    Write-Host "       sayilar maliyeti degil yalnizca IPC gidis-donusunu olcer." -ForegroundColor Yellow
    Write-Host "       Uygulama penceresini one al ve tekrar calistir." -ForegroundColor Yellow
}
elseif ($control -gt 0 -and $subject -gt $control * 2.0) {
    Write-Host ("FAIL: LUT'u kirleten duzenleme kontrolden {0}x pahali." -f [math]::Round($subject / $control, 1)) -ForegroundColor Red
    Write-Host "      Log'da su satiri ara -- yoksa CPU LUT'a dusuluyor demektir:" -ForegroundColor Red
    Write-Host "      [Vulkan] Atmosphere LUT compute pipeline ready (VulkanViewportBackend)." -ForegroundColor Red
}
else {
    Write-Host ("OK: control {0:N1} ms vs subject {1:N1} ms -- LUT duzenlemesi bedava." -f $control, $subject) -ForegroundColor Green
}

$drainsFrame = ($results | Where-Object DrainsFrame -ne $null | Measure-Object DrainsFrame -Average).Average
if ($drainsFrame -gt 1.0) {
    Write-Host ("UYARI: kare basina {0:N2} GERCEK drenaj. Bir duzenleme yolu her karede" -f $drainsFrame) -ForegroundColor Yellow
    Write-Host "       GPU'nun okudugu bir kaynagi degistiriyor." -ForegroundColor Yellow
}
