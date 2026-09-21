<#
.SYNOPSIS
    maxDepositionMeters gercekten "toplam hucre yukselmesini" bagliyor mu?
    Tek degiskenli merdiven; her satir bir oncekinden TEK kadran farkli.

.DESCRIPTION
    terrain_lem_route.comp'taki sinir kendini "Bound TOTAL build-up, not only
    this pass" diye tanitiyor ve TerrainManager.h maxDepositionMeters'i ayni
    dille anlatiyor. Bu script o iddiayi sinar.

    ★★★ OLCUM NODE YOLUNDAN GELIR, terrain.erode'dan DEGIL. Sebep: node yolu
    (TerrainV2.HydraulicErosion) publishNetAggradation cagirir, yani
    deepest_deposit_meters cikis yuzeyi eksi giris yuzeyi -- yani tam olarak
    "hucre ne kadar yukseldi". terrain.erode ayni alanda BRUT tasima defterini
    dondurur; talus bir taneyi tasirken onu her adimda yeniden deftere yazar,
    dolayisiyla o sayi yukselme DEGILDIR. Ayni isim, iki buyukluk.

    ★ Sinir yalnizca route gecisine uygulanir. Talus (mass wasting) ve alluvial
    yayilma hucreyi serbestce yukseltir. Merdiven bunu ayirir:

        R0 route only  -> sinir neyse tepe odur (sinir CALISIYOR)
        R2 +talus      -> tepe siniri asar
        R5 cap 4->1 m  -> tepe neredeyse HIC dusmez

    Son satir belirleyicidir: kadrani dortte bire indirmek tepeyi kaydirmiyorsa
    kadran o buyuklugu yonetmiyordur.

.EXAMPLE
    .\Probe-BuildCap.ps1 -Terrain T -ErodeNode 5

.NOTES
    Grafik onkosulu: <kaynak relief> -> HydraulicErosion -> Height Output.
    ErodeNode, nodes.list'teki HydraulicErosion node id'sidir.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Terrain,
    [Parameter(Mandatory)][int]$ErodeNode
)

Import-Module "$PSScriptRoot\RtIpc.psm1" -Force

function Set-P([string]$Name, $Value) {
    $null = Invoke-RtIpc nodes.set_property @{
        graph_type = 'terrain'; graph_name = $Terrain; node_id = $ErodeNode
        property = $Name; value = $Value
    }
}

# ★ PowerShell 5.1'in ConvertTo-Json'u 4.0'i "4" diye yazar. setNodeProperty
# float bir alana tam sayi kabul etmiyorsa (eski binary) bir kadrani 0 veya 1
# yapmak imkansizdir -- tam da A/B icin gereken degerler. Sonsuz kucuk sapma
# hem eski hem yeni binary'de calisir.
function Set-F([string]$Name, [double]$Value) {
    $v = $Value
    if ([math]::Floor($v) -eq $v) { $v = $v + 1.0e-6 * [math]::Max(1.0, [math]::Abs($v)) }
    Set-P $Name $v
}

function Invoke-Run([string]$Label) {
    $null = Invoke-RtIpc terrain.evaluate @{ name = $Terrain }
    $s = $null
    for ($i = 0; $i -lt 1200; $i++) {
        $s = Invoke-RtIpc terrain.evaluation_status @{ name = $Terrain }
        if ($s.state -ne 'running') { break }
        Start-Sleep -Milliseconds 500
    }
    if ($s.state -ne 'completed') { throw "evaluate '$Label' bitmedi: $($s.state) $($s.error)" }
    $st = Invoke-RtIpc terrain.erosion_stats
    if ([double]$st.eroded -le 0.0 -and [double]$st.deposited -le 0.0) {
        throw "[ALET BOZUK] '$Label' sonrasi erosion_stats bos; olculen bir sey yok."
    }
    [pscustomobject]@{
        Run        = $Label
        PeakM      = [math]::Round([double]$st.deepest_deposit_meters, 2)
        MeanM      = [math]::Round([double]$st.mean_deposit_meters, 3)
        CoveredPc  = [math]::Round([double]$st.deposited_area_fraction * 100, 1)
        LakePc     = [math]::Round([double]$st.lake_area_fraction * 100, 1)
        DeepLakePc = [math]::Round([double]$st.deep_lake_area_fraction * 100, 2)
        TrunkPc    = [math]::Round([double]$st.max_drainage_area_fraction * 100, 1)
    }
}

try {
    # Ortak taban: dropletler ve butun stabilizasyon post-process'leri KAPALI,
    # tek gecis. Geriye kalan tek yukselme kaynagi LEM'dir; boylece merdivenin
    # her basamagi tek bir gecise atfedilebilir.
    Set-P 'multiPass'               $false
    Set-P 'params.iterations'       0
    Set-P 'params.fillPits'         $false
    Set-P 'params.removeSpikes'     $false
    Set-P 'params.smoothSurface'    $false
    Set-P 'params.channelEvolution' $false
    Set-P 'params.macroDrainage'    $false
    Set-P 'params.fluvialCycle'     $true
    Set-P 'params.fluvialIterations' 16
    Set-F 'params.maxDepositionMeters' 4.0

    $rows = @()

    Set-P 'params.alluviumSteps'      0
    Set-P 'params.massWasting'        $false
    Set-P 'params.avulsionInterval'   0
    Set-F 'params.hillslopeDiffusion' 0.0
    $rows += Invoke-Run 'R0 route only'

    Set-F 'params.hillslopeDiffusion' 0.02
    $rows += Invoke-Run 'R1 +creep'

    Set-P 'params.massWasting'        $true
    $rows += Invoke-Run 'R2 +talus'

    Set-P 'params.alluviumSteps'      4
    $rows += Invoke-Run 'R3 +alluvium'

    Set-P 'params.avulsionInterval'   8
    $rows += Invoke-Run 'R4 +avulsion'

    Set-F 'params.maxDepositionMeters' 1.0
    $rows += Invoke-Run 'R5 cap 4->1m'

    $rows | Format-Table -AutoSize
    Write-Host ''

    $r0 = $rows[0]; $r2 = $rows[2]; $r4 = $rows[4]; $r5 = $rows[5]
    if ([math]::Abs($r0.PeakM - 4.0) -gt 0.25) {
        Write-Host "[BEKLENMEDIK] route-only tepe $($r0.PeakM) m, sinir 4 m." -ForegroundColor Yellow
        Write-Host '  Route gecisinin kendi siniri tutmuyor; once ONU coz.' -ForegroundColor Yellow
    } else {
        Write-Host "[OK] Route gecisi kendi sinirini tutuyor ($($r0.PeakM) m)." -ForegroundColor Green
    }

    $capDrop = $r4.PeakM - $r5.PeakM
    if ($capDrop -lt ($r4.PeakM * 0.25)) {
        Write-Host "[SOZLESME TUTMUYOR] Kadran 4 m -> 1 m ($($r4.PeakM) -> $($r5.PeakM) m)." -ForegroundColor Red
        Write-Host '  Dortte bire inen bir sinir tepeyi neredeyse hic kaydirmiyor:' -ForegroundColor Red
        Write-Host '  yukselmeyi yoneten sey bu kadran DEGIL. Talus ve alluvial' -ForegroundColor Red
        Write-Host '  yayilma hucreyi sinirin disindan yukseltiyor.' -ForegroundColor Red
        Write-Host "  Talus'un tek basina katkisi: $($r0.PeakM) -> $($r2.PeakM) m." -ForegroundColor DarkGray
    } else {
        Write-Host "[OK] Kadran tepeyi yonetiyor: $($r4.PeakM) -> $($r5.PeakM) m." -ForegroundColor Green
    }

    # ★ Sinsi okuma: yukselme buyurken drenaj SAGLIKLI kalabilir. Sinirin
    # savundugu ariza (kapali cukurlar, parcalanmis drenaj) bu sayilarda
    # gorunur; gorunmuyorsa metre sayisi tek basina bir hukum DEGILDIR.
    Write-Host ''
    Write-Host ("Drenaj saglik kontrolu: gol %{0} -> %{1}, ana kol %{2} -> %{3}" -f `
                $r0.LakePc, $r4.LakePc, $r0.TrunkPc, $r4.TrunkPc) -ForegroundColor DarkGray
    Write-Host '  Bunlar sabit kalirken tepe buyuyorsa, yukselme sinirin' -ForegroundColor DarkGray
    Write-Host '  onlemek icin yazildigi arizayi URETMIYOR demektir.' -ForegroundColor DarkGray
}
finally { Disconnect-RtIpc }
