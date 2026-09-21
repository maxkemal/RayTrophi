<#
.SYNOPSIS
    Avulsiyonun ve alluvial yayilmanin gercekten ates edip etmedigini TEK
    DEGISKENLE olcer. Sekil sayilarina bakar, resme degil.

.DESCRIPTION
    Bu partide erozyonun cokelme yarisi kuruldu: akis yonleri artik canli
    yataktan turetiliyor (avulsiyon) ve taze cokel duraylı bir yelpaze egimine
    gevsiyor (alluvial yayilma).

    Kutle defteri bu iyilesmeyi OLCEMEZ: ayni hacimli bir sirt ile bir yelpaze
    defterde birebir ayni gorunur. Ayiran sayi orandir:

        sirt     -> peak/mean 10+, dar alan
        yelpaze  -> peak/mean 2-4, genis alan

    ★ SINSI DURUM: Covered %100'e yaklasip oran dusuyorsa bu yelpaze DEGIL --
    yayilma gecisi gevsek-malzeme sinirindan kacmis ve anakayayi duzluyordur.
    Manzara bundan daha yumusak ve daha hos cikar; kimse bunu bug diye
    bildirmez. Script bu durumu ayrica uyarir.

    ★★ Tolerans DEGISIME olceklenir, mutlak esige degil: ayni sahnede iki kosu
    karsilastirilir, cunku "iyi bir peak/mean" arazi tipine gore degisir.

    ★★★ BIRIM UYARISI: bu script terrain.erode uzerinden olcer, ve o yolda
    deepest/mean_deposit_meters BRUT TASIMA DEFTERIDIR -- talus bir taneyi
    hucreden hucreye tasirken her adimda yeniden deftere yazilir. Ayni alan
    adlari terrain.evaluate (HydraulicErosion node) yolunda NET YUKSELMEDIR,
    cunku node publishNetAggradation cagirir. Iki yolun sayilarini birbiriyle
    KARSILASTIRMA; oranlar yalnizca ayni yol icinde anlamlidir.

.EXAMPLE
    .\Probe-DepositShape.ps1 -Terrain Terrain
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Terrain,
    [int]$AvulsionInterval = 8,
    [int]$AlluviumSteps = 4
)

Import-Module "$PSScriptRoot\RtIpc.psm1" -Force

function Get-Shape([string]$Label, [hashtable]$Overrides) {
    $req = @{ name = $Terrain; fluvial_cycle = 1 } + $Overrides
    $null = Invoke-RtIpc terrain.erode $req
    $s = Invoke-RtIpc terrain.erosion_stats

    # ★★★ ALETIN KENDISINI ONCE DOGRULA. Bu script bir donem her kosuda sifir
    # okudu: terrain.erode GPU yolunda summarize() `fields` kapisinin arkasinda
    # kaliyordu ve erosion_stats reset() edilmis yapiyi dondururdu. Sifirlar
    # hata gibi gorunmez -- $mean 0 olunca oran NaN olur, NaN -ge NaN yanlistir,
    # ve asagidaki kapi "[OK] avulsiyon sekli degistirdi" diye YESIL basar.
    # Hicbir sey olcmemis bir kosu BASARISIZ olmalidir, sessiz gecmemeli.
    if ([double]$s.eroded -le 0.0 -and [double]$s.deposited -le 0.0) {
        throw ("[ALET BOZUK] '$Label' kosusundan sonra terrain.erosion_stats bos " +
               "(eroded=0, deposited=0, cycle_iterations=$($s.cycle_iterations)). " +
               'Olculen hicbir sey yok; sekil sayilari anlamsiz. Once summarize() ' +
               'yolunun bu backend icin calistigini dogrula.')
    }
    $mean = [double]$s.mean_deposit_meters
    $ratio = if ($mean -gt 1e-4) { [double]$s.deepest_deposit_meters / $mean } else { [double]::NaN }
    [pscustomobject]@{
        Run       = $Label
        CoveredPc = [math]::Round([double]$s.deposited_area_fraction * 100, 2)
        MeanM     = [math]::Round($mean, 3)
        PeakM     = [math]::Round([double]$s.deepest_deposit_meters, 3)
        PeakOverMean = [math]::Round($ratio, 2)
        TrunkPc   = [math]::Round([double]$s.max_drainage_area_fraction * 100, 1)
        LeakPc    = [math]::Round([double]$s.mass_error_fraction * 100, 4)
    }
}

# Referans: bu partiden ONCEKI davranis. avulsion_interval = 0 yonlendirme
# yuzeyini max(filled,bed) yerine filled'a dondurur, alluvium_steps = 0
# yayilmayi tamamen kapatir -- yani ikisi birlikte eski koda birebir esittir.
$off = Get-Shape 'off (eski davranis)' @{ avulsion_interval = 0; alluvium_steps = 0 }
$avOnly = Get-Shape 'yalniz avulsiyon'  @{ avulsion_interval = $AvulsionInterval; alluvium_steps = 0 }
$both = Get-Shape 'ikisi'               @{ avulsion_interval = $AvulsionInterval; alluvium_steps = $AlluviumSteps }

@($off, $avOnly, $both) | Format-Table -AutoSize

Write-Host ''
# NaN her karsilastirmayi yanlis dondurur, yani NaN'i once ELE -- yoksa
# "olculemedi" durumu asagidaki else dalindan yesil cikar.
if ([double]::IsNaN($off.PeakOverMean) -or [double]::IsNaN($avOnly.PeakOverMean)) {
    Write-Host '[OLCULEMEDI] peak/mean orani NaN: ortalama cokel sifir.' -ForegroundColor Red
    Write-Host '  Bu bir sonuc DEGIL, olcum yoklugudur. Yesil sayma.' -ForegroundColor Red
} elseif ($avOnly.PeakOverMean -ge $off.PeakOverMean) {
    Write-Host '[BASARISIZ] Avulsiyon tek basina sekli DEGISTIRMEDI.' -ForegroundColor Red
    Write-Host '  Sekildeki her iyilesme yayilma gecisinden geliyor demektir,' -ForegroundColor Red
    Write-Host '  yani kozmetik bir duzlestirme aldik. Once bunu coz.' -ForegroundColor Red
    Write-Host '  Bak: terrain_lem_weights kayit satiri (3 buffer / 16 bayt) ve' -ForegroundColor DarkGray
    Write-Host '       route dongusundeki avulsion dispatch kosulu.' -ForegroundColor DarkGray
} else {
    Write-Host "[OK] Avulsiyon tek basina peak/mean orani $($off.PeakOverMean) -> $($avOnly.PeakOverMean) dusurdu." -ForegroundColor Green
}

if ($both.CoveredPc -gt 60 -and $both.PeakOverMean -lt 3) {
    Write-Host '[SINSI] Covered cok yuksek ve oran cok dusuk: bu yelpaze degil,' -ForegroundColor Yellow
    Write-Host '  yayilma gecisi anakayayi duzluyor olabilir. alluvium_steps=0 ile' -ForegroundColor Yellow
    Write-Host '  karsilastir; fark manzarayi YUMUSATIYOR ama yelpaze uretmiyorsa' -ForegroundColor Yellow
    Write-Host '  gevsek-malzeme clampi (alluviumIn) kacmis demektir.' -ForegroundColor Yellow
}

if ([math]::Abs($both.LeakPc) -gt 0.5) {
    Write-Host "[KACAK] Sediment defteri kapanmiyor: $($both.LeakPc)% hesapsiz." -ForegroundColor Red
    Write-Host '  Yayilma gecisi cokeli yalnizca YER DEGISTIRMELI, deftere yazmamali.' -ForegroundColor Red
}

if ($both.TrunkPc -lt 10) {
    Write-Host "[ANA KOL YOK] En buyuk havza haritanin %$($both.TrunkPc)'i." -ForegroundColor Red
    Write-Host '  Drenaj grafi parcali; sekil sayilari bu durumda anlamsizdir.' -ForegroundColor Red
}
