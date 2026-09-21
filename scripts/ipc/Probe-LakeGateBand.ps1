<#
.SYNOPSIS
    Krater kenarindaki keskin seridi, gol maskesi kapilarina baglayip baglamadigini
    olcer. Tek degisken: LakeBasin'in kabul esigi.

.DESCRIPTION
    Hipotez: LakeBasin.out[0] (gol maskesi) iki yerde SERT esikle kesiyor --
    RiverNetwork (>= 0.5f, discharge/order/source sifirlanir) ve RiverBedCarve
    (>= 0.5f, `continue`, yani yatak KAZILMAZ). Ikincisi dogrudan yuksekilige
    yaziyor, birincisi SurfaceComposer uzerinden splat/satmap'e.

    Test: gol maskesini bosalt (minimumAreaSquareMeters'i devasa yap, hicbir
    havza kabul edilmesin), ayni transekti tekrar ornekle. Serit hem kotta hem
    renkte gidiyorsa kok bu zincirdir.

    ★ OLCU ALETI UYARISI: terrain.export_heightmap 8-bit PNG yaziyor
    (TerrainManager.cpp saveMapPNG). heightScale 2000 m ise bir adim ~7.8 m --
    aradigimiz basamak muhtemelen bunun ALTINDA ve export'ta TAM SIFIR gorunur.
    Bu yuzden yukseklik farki export'tan degil, terrain.sample_height ile
    float hassasiyetinde bir TRANSEKT'ten okunuyor.

.PARAMETER CenterX
    Kraterin dunya-uzayi merkezi. Transekt buradan disari dogru cizilir.

.EXAMPLE
    .\Probe-LakeGateBand.ps1 -CenterX 0 -CenterZ 0 -Radius 400 -Render
#>
[CmdletBinding()]
param(
    [string]$Terrain,
    [string]$Graph,
    [Parameter(Mandatory = $true)][float]$CenterX,
    [Parameter(Mandatory = $true)][float]$CenterZ,
    [Parameter(Mandatory = $true)][float]$Radius,
    [int]$Samples = 400,
    [switch]$Render,
    [int]$RenderSpp = 64,
    [string]$OutDir = "$env:TEMP\rt_lakeband"
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'RtIpc.psm1') -Force
Wait-RtIpcReady -TimeoutSeconds 120 | Out-Null
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Wait-Eval {
    param([string]$Name)
    for ($i = 0; $i -lt 3000; $i++) {
        $s = Invoke-RtIpc terrain.evaluation_status @{ name = $Name }
        if ($s.state -ne 'running') { return $s }
        Start-Sleep -Milliseconds 200
    }
    throw "terrain.evaluate 600 sn icinde bitmedi."
}

function Sample-Transect {
    param([string]$Name, [int]$N)
    $out = New-Object double[] $N
    for ($i = 0; $i -lt $N; $i++) {
        $t = $i / [double]($N - 1)
        $out[$i] = [double](Invoke-RtIpc terrain.sample_height @{
            name = $Name; world_x = ($CenterX + $Radius * $t); world_z = $CenterZ })
    }
    return $out
}

# ── 0. Uygulanabilirlik. Bunlar YOKSA test yesil donmemeli, DURMALI. ─────────
if (-not $Terrain) { $Terrain = (Invoke-RtIpc terrain.list)[0].name }
if (-not $Graph)   { $Graph   = (Invoke-RtIpc nodes.graphs @{ graph_type = 'terrain' })[0] }
Write-Host "Terrain='$Terrain' Graph='$Graph'" -ForegroundColor Cyan

$nodes = Invoke-RtIpc nodes.list @{ graph_type = 'terrain'; graph_name = $Graph }
function Find-Node { param([string]$Frag) ,@($nodes | Where-Object { $_.type_id -like "*$Frag*" }) }

$lake    = (Find-Node 'LakeBasin')[0]
$network = (Find-Node 'RiverNetwork')[0]
$carve   = (Find-Node 'RiverBedCarve')[0]

if (-not $lake) { throw "Graph'ta LakeBasin yok. Bu hipotez bu graph icin GECERSIZ - serit baska bir sebepten." }
Write-Host ("LakeBasin id={0}  RiverNetwork={1}  RiverBedCarve={2}" -f `
    $lake.id, $(if($network){$network.id}else{'YOK'}), $(if($carve){$carve.id}else{'YOK'})) -ForegroundColor Cyan

# Maske pinleri gercekten BAGLI mi? Bagli degilse kapi zaten kapali degildir.
$gates = @()
if ($network) {
    $p = Invoke-RtIpc nodes.list_params @{ graph_type='terrain'; graph_name=$Graph; node_id=$network.id }
    $lm = $p | Where-Object { $_.index -eq 3 }
    Write-Host ("  RiverNetwork.in[3] (lakeMask) connected = {0}" -f $lm.connected)
    if ($lm.connected) { $gates += 'RiverNetwork' }
}
if ($carve) {
    $p = Invoke-RtIpc nodes.list_params @{ graph_type='terrain'; graph_name=$Graph; node_id=$carve.id }
    $lm = $p | Where-Object { $_.index -eq 6 }
    Write-Host ("  RiverBedCarve.in[6] (lakeMask) connected = {0}" -f $lm.connected)
    if ($lm.connected) { $gates += 'RiverBedCarve' }
}
if ($gates.Count -eq 0) {
    Write-Host "HIC BIR gol maskesi pini bagli degil. Hipotez BURADA OLDU - serit baska bir sebepten." -ForegroundColor Red
    return
}

# ── 1. Taban ─────────────────────────────────────────────────────────────────
$baseArea = (Invoke-RtIpc nodes.get_property @{
    graph_type='terrain'; graph_name=$Graph; node_id=$lake.id
    property='minimumAreaSquareMeters' })
Write-Host "minimumAreaSquareMeters tabani = $baseArea" -ForegroundColor Cyan

try {
    Invoke-RtIpc terrain.evaluate @{ name = $Terrain } | Out-Null
    Wait-Eval $Terrain | Out-Null
    $A = Sample-Transect $Terrain $Samples
    $statsA = Invoke-RtIpc terrain.erosion_stats @{ name = $Terrain }
    $authA  = Invoke-RtIpc terrain.flow_authority @{ name = $Terrain }
    if ($Render) {
        Invoke-RtIpc render.start @{ output_path = "$OutDir\lakes_on.png"; spp = $RenderSpp } | Out-Null
        for ($i=0; $i -lt 900; $i++) { if ((Invoke-RtIpc render.status).state -ne 'rendering') { break }; Start-Sleep -Milliseconds 500 }
    }

    # ── 2. Golleri bosalt ────────────────────────────────────────────────────
    Invoke-RtIpc nodes.set_property @{
        graph_type='terrain'; graph_name=$Graph; node_id=$lake.id
        property='minimumAreaSquareMeters'; value=1.0e9 } | Out-Null
    Invoke-RtIpc terrain.evaluate @{ name = $Terrain } | Out-Null
    Wait-Eval $Terrain | Out-Null
    $B = Sample-Transect $Terrain $Samples
    $statsB = Invoke-RtIpc terrain.erosion_stats @{ name = $Terrain }
    if ($Render) {
        Invoke-RtIpc render.start @{ output_path = "$OutDir\lakes_off.png"; spp = $RenderSpp } | Out-Null
        for ($i=0; $i -lt 900; $i++) { if ((Invoke-RtIpc render.status).state -ne 'rendering') { break }; Start-Sleep -Milliseconds 500 }
    }

    # ── 3. Rapor ─────────────────────────────────────────────────────────────
    $maxDiff = 0.0; $argMax = -1; $nonZero = 0
    for ($i = 0; $i -lt $Samples; $i++) {
        $d = [Math]::Abs($A[$i] - $B[$i])
        if ($d -gt 1e-4) { $nonZero++ }
        if ($d -gt $maxDiff) { $maxDiff = $d; $argMax = $i }
    }
    $argR = if ($argMax -ge 0) { $Radius * $argMax / ($Samples - 1) } else { -1 }

    Write-Host ""
    Write-Host "===== SONUC =====" -ForegroundColor Green
    Write-Host ("bagli kapilar        : {0}" -f ($gates -join ', '))
    Write-Host ("goller ACIK  lake_cells={0} frac={1:N4} cycle_iters={2}" -f $statsA.lake_cells, $statsA.lake_area_fraction, $statsA.cycle_iterations)
    Write-Host ("goller KAPALI lake_cells={0} frac={1:N4} cycle_iters={2}" -f $statsB.lake_cells, $statsB.lake_area_fraction, $statsB.cycle_iterations)
    Write-Host ("flow authority source: {0}" -f $authA.source)
    Write-Host ("max |dh|             : {0:N4} m   (merkezden {1:N1} m)" -f $maxDiff, $argR)
    Write-Host ("|dh| > 1e-4 ornek    : {0} / {1}" -f $nonZero, $Samples)
    Write-Host ""
    if ($statsB.lake_cells -ne 0) {
        Write-Host "UYARI: goller KAPALI durumda lake_cells hala 0 degil. Tek degiskenli test BOZULDU;" -ForegroundColor Yellow
        Write-Host "       asagidaki dh yorumlanamaz." -ForegroundColor Yellow
    } elseif ($maxDiff -lt 1e-4) {
        Write-Host "KOT DEGISMEDI. Serit geometride DEGIL - carve kapisi masum, mask tarafina bak." -ForegroundColor Yellow
    } elseif ($nonZero -lt ($Samples / 4)) {
        Write-Host "KOT DEGISTI ve fark DAR bir bantta yogun => gol kapisi zinciri DOGRULANDI." -ForegroundColor Green
    } else {
        Write-Host "Kot her yerde degisti. Gol kapisi bir SERIT degil genel bir etki yapiyor;" -ForegroundColor Yellow
        Write-Host "bu tek basina seridi aciklamaz." -ForegroundColor Yellow
    }
    $A | ForEach-Object { $_ } | Out-File "$OutDir\transect_lakes_on.txt"
    $B | ForEach-Object { $_ } | Out-File "$OutDir\transect_lakes_off.txt"
    Write-Host "transektler: $OutDir"
}
finally {
    Write-Host "minimumAreaSquareMeters geri aliniyor ($baseArea)..." -ForegroundColor Cyan
    Invoke-RtIpc nodes.set_property @{
        graph_type='terrain'; graph_name=$Graph; node_id=$lake.id
        property='minimumAreaSquareMeters'; value=$baseArea } | Out-Null
    Invoke-RtIpc terrain.evaluate @{ name = $Terrain } | Out-Null
    Wait-Eval $Terrain | Out-Null
}
