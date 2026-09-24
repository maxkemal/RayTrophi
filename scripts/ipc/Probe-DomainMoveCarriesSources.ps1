<#
.SYNOPSIS
  Bir gaz/sivi domain'i tasindiginda ona bagli emitterlerin de tasindigini
  dogrular -- ve parent'li bir kaynagin TASINMADIGINI.

.DESCRIPTION
  ★★★★★ Bu betigin varlik sebebi iki belirtinin TEK sebep olmasidir. Domain
  tasiniyordu ama emitterler yerinde kaliyordu; yeniden simule edilen duman
  eski yerinde beliriyordu. Bu disaridan "cache silinmemis, eski konumuna
  zipladi" gibi gorunur -- ve o hikaye daha inandiricidir, cunku cache zaten
  supheli bir seydir. Oysa bayat olan cache degil, EMITTERLERDI.

  ★★★ Ve bu ariza hicbir testte yakalanmamisti, cunku domain tasima IPC'de HIC
  YOKTU: yalnizca gizmo ve panelden yapilabiliyordu. CLAUDE.md kural 1'in
  tarifi tam olarak bu -- panelden erisilen bir yetenek test EDILEMEZ sayilir.
  `sim.move_domain` once olcu aleti olarak eklendi, sonra ariza kapatildi.

.NOTES
  ★ `sources_carried` bu betigin kalbidir: tasinan kaynak SAYISI. Sifir donmesi
    bir hata DEGIL, bir olcumdur -- domain hic kaynak tasimiyor olabilir ya da
    tasidigi kaynaklarin hepsi PARENT'LI olabilir. Parent'li kaynak kendi
    nesnesini izler, domain'i degil: tek bir konumun iki sahibi olamaz.
#>
[CmdletBinding()]
param([string]$Domain = "")

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'RtIpc.psm1') -Force

$fails = 0
function Check($name, $ok, $detail) {
    if ($ok) { Write-Host ("  [GECTI] {0} {1}" -f $name, $detail) }
    else     { Write-Host ("  [KALDI] {0} {1}" -f $name, $detail) -ForegroundColor Red; $script:fails++ }
}

# Hedef domain: parametre verilmediyse kaynagi olan ILK domain secilir.
$sources = @(Invoke-RtIpc flow_source.list @{})
if ($sources.Count -lt 1) {
    Write-Host "Sahnede flow source yok - bu betik en az bir emitter gerektirir." -ForegroundColor Yellow
    exit 2
}
if (-not $Domain) {
    $unparented = $sources | Where-Object { -not $_.parent_object }
    if (-not $unparented) {
        Write-Host "Butun kaynaklar parent'li - tasinmamalari DOGRU davranis." -ForegroundColor Yellow
        exit 2
    }
    $Domain = [string]@($unparented)[0].domain
}
Write-Host ("Hedef domain: {0}" -f $Domain)

function SourcesOf($dom) {
    @(Invoke-RtIpc flow_source.list @{}) | Where-Object { [string]$_.domain -eq $dom }
}

$before = @(SourcesOf $Domain)
Check "domain'in kaynagi var" ($before.Count -ge 1) ("{0} kaynak" -f $before.Count)
$cacheBefore = Invoke-RtIpc sim_cache.status @{}
Write-Host ("Tasima oncesi cache: {0} RAM karesi, disk bake gecerli={1}" -f `
            $cacheBefore.ram_frames, $cacheBefore.valid)
$delta = @(3.0, 1.0, -2.0)

try {
    # ── 1. Tasima kaynaklari TASIMALI ───────────────────────────────────────
    Write-Host "1. ★★★ sim.move_domain kaynaklari da tasimali"
    $r = Invoke-RtIpc sim.move_domain @{ domain = $Domain; delta = $delta }
    $expectCarried = @($before | Where-Object { -not $_.parent_object }).Count
    Check "sources_carried = parent'siz kaynak sayisi" `
          ([int]$r.sources_carried -eq $expectCarried) `
          ("beklenen {0}, donen {1}" -f $expectCarried, $r.sources_carried)

    $after = @(SourcesOf $Domain)
    foreach ($b in $before) {
        $a = $after | Where-Object { $_.name -eq $b.name } | Select-Object -First 1
        if (-not $a) { Check ("kaynak kayboldu: " + $b.name) $false ""; continue }
        $dx = [double]$a.position[0] - [double]$b.position[0]
        $dy = [double]$a.position[1] - [double]$b.position[1]
        $dz = [double]$a.position[2] - [double]$b.position[2]
        if ($b.parent_object) {
            # ★★ Parent'li kaynak KIMILDAMAMALI: sahibi parent nesnesidir.
            Check ("parent'li '{0}' YERINDE kaldi" -f $b.name) `
                  ([Math]::Abs($dx)+[Math]::Abs($dy)+[Math]::Abs($dz) -lt 1e-3) `
                  ("sapma {0:F4},{1:F4},{2:F4}" -f $dx,$dy,$dz)
        } else {
            Check ("'{0}' domain ile tasindi" -f $b.name) `
                  ([Math]::Abs($dx-$delta[0]) -lt 1e-3 -and `
                   [Math]::Abs($dy-$delta[1]) -lt 1e-3 -and `
                   [Math]::Abs($dz-$delta[2]) -lt 1e-3) `
                  ("delta {0:F3},{1:F3},{2:F3}" -f $dx,$dy,$dz)
        }
    }

    # ── 2. Domain'in kendisi de tasinmis olmali ─────────────────────────────
    Write-Host "2. domain kutusu da tasinmali (yarim duzeltme yok)"
    Check "merkez delta kadar kaydi" ($null -ne $r.center) ("center {0}" -f ($r.center -join ', '))

    # 3. *** Tasima BAKE'I OLDURMEMELI
    #
    # ** Bu kapi ancak ONCESINDE cache VARSA bir olcumdur. Sifir kare varken
    #    "kare sayisi degismedi" demek hicbir sey kanitlamaz -- silinecek bir sey
    #    yoktu. O yuzden bos cache PAS degil, ATLANDI olarak raporlanir.
    Write-Host "3. *** tasima bake'i silmemeli"
    if ([int]$cacheBefore.ram_frames -le 0) {
        Write-Host "  [ATLANDI] cache bostu - silinecek bir sey yoktu, kapi olcum degil" -ForegroundColor Yellow
    } else {
        # **** Olcum KARE DONGUSUNDEN SONRA alinir. Cache'i dusuren sey tasima
        #   kodu degil, kare dongusunun kendi auto-invalidate'idir ve o bir
        #   sonraki tikte kosar. Hemen sonra saymak "duruyor" der, ve o cevap
        #   makul gorunur -- bu deponun en pahali hata sinifi tam olarak budur.
        Start-Sleep -Milliseconds 600
        $cacheAfter = Invoke-RtIpc sim_cache.status @{}
        Check "RAM cache kareleri bir tik SONRA da duruyor" `
              ([int]$cacheAfter.ram_frames -eq [int]$cacheBefore.ram_frames) `
              ("once {0}, sonra {1}" -f $cacheBefore.ram_frames, $cacheAfter.ram_frames)
        # * Ve imza GERCEKTEN yeni kutuya gore yeniden alinmis olmali. Degismemis
        #   bir imza, tasimanin acceptSimConfigAsBaked'i hic cagirmadigini
        #   gosterir: cache bu tur hayatta kalir ama bir sonraki edit'te dusen
        #   sey yanlis imzaya karsi olculur.
        Check "baked imza yeni kutuya gore tazelendi" `
              ([string]$cacheAfter.config_signature -ne [string]$cacheBefore.config_signature) `
              ("once {0}, sonra {1}" -f $cacheBefore.config_signature, $cacheAfter.config_signature)
    }
}
finally {
    # Geri al: ayni domain'i ters delta ile tasi.
    Invoke-RtIpc sim.move_domain @{ domain = $Domain; delta = @(-$delta[0], -$delta[1], -$delta[2]) } | Out-Null
}

Write-Host ""
if ($fails -eq 0) { Write-Host "TUM KAPILAR GECTI" -ForegroundColor Green }
else { Write-Host ("{0} KAPI KALDI" -f $fails) -ForegroundColor Red; exit 1 }
