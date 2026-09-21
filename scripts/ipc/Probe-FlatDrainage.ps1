<#
.SYNOPSIS
    Duzlukte akarsu davranisi: Garbrecht-Martz duzlik gradyani gercekten
    calisiyor mu, yoksa kadran cozucuye ULASMIYOR mu?

.DESCRIPTION
    Sikayet: cukur sinirlarinda ve duzluklerde akis duz ve koseli cikiyor,
    birlesme ve kivrilma olmuyor. Kok neden ikiliydi:

      1. Doldurulmus duzlukte gradyan YOK. Eski sema her tasma halkasini
         birkac ulp kaldiriyordu; kardinal ve capraz adimin ayni ucretlendigi
         bir merdiven Chebyshev mesafe alanidir ve onun seviye egrileri
         KAREDIR. Kare bir mesafe alaninda en dik inis 0 ve 45 derecede akar.
      2. Birkac ulp sayisal olarak SIFIR dusustur. terrain_lem_incise
         asinmayi incisionSafety * (alicidan dusus) ile sinirlar, yani duzluk
         HIC oyulamiyordu. Ilk kesim olmayinca A^m geri beslemesi hic
         baslamiyor: ilk drenaj cozumunun cizdigi desen NIHAI desen oluyor.

    Bu script iddiayi tek degiskenle sinar: flat_gradient.

    ★★★ ILK KONTROL BIR KORLUK TESTIDIR. A (flat_gradient=0, eski davranis)
    ile B (2e-4) AYNI sayilari veriyorsa sonuc "iyilesme yok" DEGILDIR --
    kadran cozucuye ulasmiyordur. Ayni dizi = ayni kare dersi; onceki
    partilerde birebir ayni sayi hep bunu isaret etti.

    ★★ ON KOSUL: unresolved_flat_cells = 0 olmali. Sifir degilse geodezik
    cephe haritadaki en genis duzlugu asamamistir; o hucreler yonlendirme
    gradyani tasimaz ve TERMINAL COKUKTUR -- yani ustlerindeki her havzayi
    sessizce kirparlar ve render'da bunun HIC belirtisi olmaz. Once
    flat_resolve_passes'i yukselt, sonra geri kalan sayilari oku.

.EXAMPLE
    .\Probe-FlatDrainage.ps1 -Terrain T

.NOTES
    Uygulama acik ve IPC hazir olmali:
        .\scripts\ipc\Start-RayTrophi.ps1
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Terrain,
    [int]$Iterations = 16,
    [int]$FlatResolvePasses = 512
)

Import-Module "$PSScriptRoot\RtIpc.psm1" -Force

function Reset-Terrain {
    # A ve B AYNI zeminden baslamali. terrain.erode yuzeyi kalici olarak
    # degistirir, yani iki kosuyu ust uste yapmak ikinci kosuya birinciden
    # asinmis bir arazi verir ve fark "ramp'in etkisi" DEGIL "sirasi" olur.
    $null = Invoke-RtIpc terrain.evaluate @{ name = $Terrain }
    for ($i = 0; $i -lt 600; $i++) {
        $s = Invoke-RtIpc terrain.evaluation_status @{ name = $Terrain }
        if ($s.state -ne 'running') { break }
        Start-Sleep -Milliseconds 400
    }
    if ($s.state -ne 'completed') {
        throw "terrain.evaluate tamamlanmadi (state=$($s.state), error='$($s.error)')."
    }
}

function Invoke-Erode([double]$FlatGradient) {
    Reset-Terrain
    # terrain.erode YALNIZCA true/false doner; olcumler ayri bir cagridadir.
    $ok = Invoke-RtIpc terrain.erode @{
        name                 = $Terrain
        type                 = 'hydraulic'
        backend              = 'auto'
        fluvial_cycle        = 1
        fluvial_iterations   = $Iterations
        flat_gradient        = $FlatGradient
        flat_resolve_passes  = $FlatResolvePasses
        undo                 = $false
    }
    if (-not $ok) { throw "terrain.erode basarisiz dondu (flat_gradient=$FlatGradient)." }

    $stats = Invoke-RtIpc terrain.erosion_stats @{ name = $Terrain }

    # ** Bir probe'un ilk gorevi KENDI KORLUGUNU dislamaktir. Bos bir stats
    #    nesnesiyle devam etmek her karsilastirmayi "esit" yapar ve script
    #    olcmedigi bir seyi FAIL diye raporlar -- olcu aleti doluluk
    #    raporlarken tam olarak bu olur.
    if ($null -eq $stats -or $null -eq $stats.eroded) {
        throw "terrain.erosion_stats bos dondu; olculecek bir sey yok."
    }
    return $stats
}

function Show-Row([string]$Label, $Stats) {
    '{0,-14} eroded={1,10:F4}  deposited={2,10:F4}  density={3,7:F4}%  trunk={4,6:F2}%  lakes={5,7:F3}%  unresolvedFlats={6}' -f `
        $Label,
        $Stats.eroded, $Stats.deposited,
        ($Stats.drainage_density * 100.0),
        ($Stats.max_drainage_area_fraction * 100.0),
        ($Stats.lake_area_fraction * 100.0),
        $Stats.unresolved_flat_cells
}

Write-Host ''
Write-Host '=== Duzluk drenaji: flat_gradient A/B ===' -ForegroundColor Cyan
Write-Host ''

# * grad=0 eski BUILD degildir: doldurma artik merdiven de eklemedigi icin
#   duzlukte HIC gradyan kalmaz, yani A bir TABANDIR, bir "oncesi" degil.
#   Gercek oncesi/sonrasi icin eski commit'i derlemek gerekir.
$sa = Invoke-Erode 0.0        # ramp KAPALI = taban
$sb = Invoke-Erode 0.0002     # ramp ACIK   = gercek tasma ovasi egimi

Show-Row 'A grad=0'      $sa
Show-Row 'B grad=2e-4'   $sb
Write-Host ''

$fail = $false

# --- 1. KORLUK TESTI: kadran cozucuye ulasiyor mu? ---------------------
if ($sa.eroded -eq $sb.eroded -and $sa.drainage_density -eq $sb.drainage_density) {
    Write-Host 'FAIL  A ve B BIREBIR ayni. Bu "iyilesme yok" demek degil:' -ForegroundColor Red
    Write-Host '      flat_gradient cozucuye ULASMIYOR. Once applyFluvialCycleSettings' -ForegroundColor Red
    Write-Host '      ve flatStepFor zincirini kontrol et; asagidaki hicbir sayi anlamli degil.' -ForegroundColor Red
    $fail = $true
} else {
    Write-Host 'OK    Kadran cozucuye ulasiyor (A != B).' -ForegroundColor Green
}

# --- 2. ON KOSUL: cozulmemis duzluk kalmamali --------------------------
if ($sb.unresolved_flat_cells -lt 0) {
    Write-Host 'WARN  unresolved_flat_cells = -1 => OLCULMEDI (GPU geri okumasi basarisiz).' -ForegroundColor Yellow
    Write-Host '      Bu bir sifir DEGILDIR. Asagidaki yorumlar bu yuzden temkinli okunmali.' -ForegroundColor Yellow
} elseif ($sb.unresolved_flat_cells -gt 0) {
    Write-Host ("FAIL  {0} hucre cozulmemis duzluk. Bunlar terminal cokuk; ustlerindeki" -f $sb.unresolved_flat_cells) -ForegroundColor Red
    Write-Host '      havzalari kirparlar ve render bunu GOSTERMEZ. flat_resolve_passes yukselt.' -ForegroundColor Red
    $fail = $true
} else {
    Write-Host 'OK    Cozulmemis duzluk yok.' -ForegroundColor Green
}

# --- 3. ASIL IDDIA: duzluk artik oyuluyor ------------------------------
# Duzluk daha once HIC oyulamiyordu (clamp = incisionSafety * 0). Ramp acikken
# ayni sahnede toplam asinma ARTMALI. Tolerans degisime olceklenir, mutlak bir
# esige degil: sahne kucukse fark da kucuk olur ama isaret ayni kalir.
$delta = $sb.eroded - $sa.eroded
$rel = if ($sa.eroded -gt 1e-9) { $delta / $sa.eroded } else { [double]::PositiveInfinity }
if ($delta -le 0) {
    Write-Host ('FAIL  Ramp acikken asinma ARTMADI (delta={0:F6}). Duzluk hala oyulmuyor:' -f $delta) -ForegroundColor Red
    Write-Host '      terrain_lem_incise icindeki routeDrop terimini kontrol et.' -ForegroundColor Red
    $fail = $true
} else {
    Write-Host ('OK    Asinma arttI: delta={0:F6} ({1:P1}). Duzluk artik yatak aciyor.' -f $delta, $rel) -ForegroundColor Green
}

# --- 4. AG BUTUNLUGU: govde nehir kaybolmadi ---------------------------
# ★ En sinsi basarisizlik burasi: ramp yanlis kurulursa duzlukte yeni cokukler
#   uretir, ag parcalanir, ama manzara YINE inandirici gorunur. Govde havzasi
#   haritanin yuzdesi olarak DUSMEMELI.
if ($sb.max_drainage_area_fraction -lt $sa.max_drainage_area_fraction * 0.9) {
    Write-Host ('FAIL  Govde havzasi kucüldu: {0:P2} -> {1:P2}. Ramp cokuk uretiyor olabilir.' -f `
        $sa.max_drainage_area_fraction, $sb.max_drainage_area_fraction) -ForegroundColor Red
    $fail = $true
} else {
    Write-Host ('OK    Govde havzasi korundu: {0:P2} -> {1:P2}.' -f `
        $sa.max_drainage_area_fraction, $sb.max_drainage_area_fraction) -ForegroundColor Green
}

Write-Host ''
if ($fail) {
    Write-Host 'SONUC: FAIL' -ForegroundColor Red
    exit 1
}
Write-Host 'SONUC: PASS' -ForegroundColor Green
Write-Host ''
Write-Host 'Sayilar yesilse bile GOZLE bak: duzlukte kanallar birlesiyor mu,' -ForegroundColor DarkGray
Write-Host 'yoksa hala paralel mi akiyorlar? Bu dort sayi "kose" olcmez.' -ForegroundColor DarkGray
