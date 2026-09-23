<#
.SYNOPSIS
  Odak kilidi (orbit pivot) sozlesmesini dogrular: pivotun BAYATLAYAMADIGINI,
  ve navigasyon yaricapinin LENS ODAK DUZLEMINDEN bagimsiz kaldigini.

.DESCRIPTION
  ★★★★★ Bu betigin varlik sebebi tek bir satirdi (2026-09-23'e kadar):

      Camera::setLookDirection: lookat = lookfrom + direction * focus_dist

  `focus_dist` bir LENS ozelligidir -- otofokus ve odak halkasi yazar. O satir
  yuzunden her fare rotasyonu navigasyon hedefini odak duzlemine tasidi. Yani
  odagi 0,5 m'ye cekmek orbit yaricapini 0,5 m'ye dusurdu; pan hizi ve dolly
  adimi bu yaricapla ORANTILI oldugu icin viewport cevap vermez oldu.

  ★★★ Belirtinin neden kimse tarafindan bug diye raporlanmadigi onemli: hata
  yok, log yok, ekran goruntusunde de hicbir sey yok. "Kamera bir tuhaf"
  diye gecistirilen sinifta. Tek gorunur hali bir SAYI -- ve `camera.get_pivot`
  var olana kadar okunacak sayi yoktu. Kapi 2 bu betigin kalbidir.

  ★★ Kapi 4 ikinci hata sinifini kapatir: pivotun BAYATLAMASI. Frame Selected
  secimin merkezini yazardi, sonraki orta-tik ise imlecin altindaki yuzeyi
  yazardi, ve birbirlerinden haberleri yoktu. Artik secim modunda pivot
  SAKLANAN degil TURETILEN bir degerdir -- her hareketin basinda yeniden
  hesaplanir, dolayisiyla bayatlamasi yapisal olarak imkansizdir.

.NOTES
  ★ Bu betik GORUNTU olcmez, SOZLESME olcer. Kameranin gercekten dondugunu
    viewport.capture + viewport.get_screenshot ile goz kontrolu yapar;
    ikisi birlikte kosmali.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'RtIpc.psm1') -Force

$fails = 0
function Check($name, $ok, $detail) {
    if ($ok) { Write-Host ("  [GECTI] {0} {1}" -f $name, $detail) }
    else     { Write-Host ("  [KALDI] {0} {1}" -f $name, $detail) -ForegroundColor Red; $script:fails++ }
}

$cam0 = Invoke-RtIpc camera.get @{}
$piv0 = Invoke-RtIpc camera.get_pivot @{}

# Hedef obje: sahnedeki ilk mesh. Secim olmadan kilit ARMED olamaz.
$objects = Invoke-RtIpc scene.list_objects @{}
if (-not $objects -or $objects.Count -lt 1) {
    Write-Host "Sahnede obje yok - bu betik bir secim gerektirir." -ForegroundColor Yellow
    exit 2
}
$target = $objects[0].name
Write-Host ("Hedef obje: {0}" -f $target)

try {
    # ── 1. Frame Selected pivotu KURUYOR mu ─────────────────────────────────
    Write-Host "1. camera.frame_selected secimi cerceveler ve kilidi kurar"
    Invoke-RtIpc select.object @{ name = $target } | Out-Null
    Invoke-RtIpc camera.frame_selected @{ lock_pivot = $true } | Out-Null
    $p = Invoke-RtIpc camera.get_pivot @{}
    Check "mode=selection" ($p.mode -eq 'selection') ("mode={0}" -f $p.mode)
    Check "pivot ARMED" ($p.locked -eq $true) ("locked={0}" -f $p.locked)
    Check "kilitli obje dogru" ($p.selection_name -eq $target) `
          ("selection_name='{0}'" -f $p.selection_name)
    Check "nav yaricapi makul (>0.01)" ([double]$p.nav_distance -gt 0.01) `
          ("nav_distance={0:F3}" -f $p.nav_distance)

    # ── 2. ★★★★★ ASIL KAPI: ODAK NAVIGASYONU SURUKLEMEZ ────────────────────
    # Bu kapi kalirsa `focus_dist` yeniden navigasyon yaricapina baglanmistir
    # ve pan/dolly yakin odakta yine kilitlenecektir. Belirti sessizdir --
    # kapi tam da bu yuzden var.
    Write-Host "2. ★★★ odak mesafesini cekmek nav yaricapini DEGISTIRMEMELI"
    $navBefore = [double]$p.nav_distance
    Invoke-RtIpc camera.set_focus_distance @{ focus_distance = 0.5 } | Out-Null
    $p2 = Invoke-RtIpc camera.get_pivot @{}
    Check "focus_distance YAZILDI" ([Math]::Abs([double]$p2.focus_distance - 0.5) -lt 0.01) `
          ("focus_distance={0:F3}" -f $p2.focus_distance)
    Check "nav_distance DEGISMEDI" `
          ([Math]::Abs([double]$p2.nav_distance - $navBefore) -lt 0.01) `
          ("once {0:F3}, sonra {1:F3}" -f $navBefore, $p2.nav_distance)

    # ── 3. Dolly ODAGI yeniden yazmamali ────────────────────────────────────
    # Zoom eskiden `focus_dist = new_distance` yazardi: dikkatle kurulan bir
    # odak, tek bir tekerlek hareketini yasamazdi.
    Write-Host "3. camera.dolly odak duzlemini KAYDIRMAMALI"
    Invoke-RtIpc camera.dolly @{ factor = 0.5 } | Out-Null
    $p3 = Invoke-RtIpc camera.get_pivot @{}
    Check "nav yaricapi YARILANDI" `
          ([Math]::Abs([double]$p3.nav_distance - $navBefore * 0.5) -lt ($navBefore * 0.05)) `
          ("beklenen {0:F3}, olculen {1:F3}" -f ($navBefore * 0.5), $p3.nav_distance)
    Check "focus_distance HALA 0.5" ([Math]::Abs([double]$p3.focus_distance - 0.5) -lt 0.01) `
          ("focus_distance={0:F3}" -f $p3.focus_distance)

    # ── 4. ★★★ PIVOT BAYATLAYAMAZ: orbit yaricapi KORUR ────────────────────
    # Gercek bir orbit yaricapi degistirmez. Degistiriyorsa donus bir orbit
    # degil, orbit adi takilmis bir fly-look'tur -- Frame Selected'in "tutmuyor"
    # gibi hissettiren tam olarak o davranisti.
    Write-Host "4. ★★ camera.orbit yaricapi korumali (fly-look DEGIL)"
    $navOrbit = [double]$p3.nav_distance
    Invoke-RtIpc camera.orbit @{ yaw = 45.0; pitch = 20.0 } | Out-Null
    $p4 = Invoke-RtIpc camera.get_pivot @{}
    Check "nav_distance korundu" `
          ([Math]::Abs([double]$p4.nav_distance - $navOrbit) -lt ($navOrbit * 0.02)) `
          ("once {0:F3}, sonra {1:F3}" -f $navOrbit, $p4.nav_distance)
    # ★★★ Vec3'ler IPC'den DIZI doner ([x,y,z]). `.x` ile okumak sessizce BOS
    #   uretir, fark 0 cikar ve kapi HER ZAMAN gecer -- yani olcu aleti
    #   dogruladigini sandigi seyi hic olcmemis olur. Bu betik ilk halinde tam
    #   olarak bu hataya dusmustu.
    $c4 = Invoke-RtIpc camera.get @{}
    $dx = [double]$c4.target[0] - [double]$p4.pivot[0]
    $dy = [double]$c4.target[1] - [double]$p4.pivot[1]
    $dz = [double]$c4.target[2] - [double]$p4.pivot[2]
    Check "kamera HALA pivota bakiyor" `
          ([Math]::Sqrt($dx*$dx + $dy*$dy + $dz*$dz) -lt ($navOrbit * 0.02)) `
          ("target-pivot sapmasi {0:F4}" -f [Math]::Sqrt($dx*$dx + $dy*$dy + $dz*$dz))

    # ── 5. ★★★ PAN KILIDI KIRMALI (DCC sozlesmesi) ─────────────────────────
    # Bir objeden UZAGA pan yapmak ile ona KILITLI kalmak celiskili iki
    # niyettir. Blender de ayni sekilde cozer: pan gorus pivotunu tasir, yani
    # eski kilit gitmistir. Ilk tasarim kilidi pan boyunca korumaya calisti ve
    # isinlanma gibi hissettiren bir arizaya yol acti -- pivot TURETILMIS
    # oldugu icin sonraki hareket onu objeye geri cakiyordu.
    Write-Host "5. ★★ camera.pan secim kilidini BIRAKMALI"
    Invoke-RtIpc camera.pan @{ right = 1.0; up = 0.5 } | Out-Null
    $p5 = Invoke-RtIpc camera.get_pivot @{}
    Check "mode free'ye dondu" ($p5.mode -eq 'free') ("mode={0}" -f $p5.mode)
    Check "capa disarm edildi" ($p5.locked -eq $false) ("locked={0}" -f $p5.locked)
    # ★ Kilit birakildi diye capa KAYBOLMAZ: effectivePivot() `lookat`e duser
    #   ve o da pan ile birlikte tasindi. Yani orbit/dolly pan edilen noktanin
    #   etrafinda calismaya devam eder; yaricap makul kalmali.
    Check "nav yaricapi makul kaldi" `
          ([double]$p5.nav_distance -gt 0.01 -and [double]$p5.nav_distance -lt 1e6) `
          ("nav_distance={0:F3}" -f $p5.nav_distance)

    # ── 6. Secim yoksa kilit DURUSTCE unarmed demeli ────────────────────────
    # ★ "Varsayilan bir olcum degildir": secim yokken eski objenin merkezini
    #   tutmak sessizce bayat bir pivot uretirdi. Mod korunur, pivot dusurulur.
    Write-Host "6. secim kalkinca pivot unarmed olmali (mod KORUNUR)"
    # ★ Onkosul: kapi 5 modu bilerek 'free'ye dusurdu, yani kilidi yeniden
    #   kurmadan bu kapi olcecegi seyi olcmez -- 'selection modunda secim
    #   yoksa' durumunu test ediyoruz, 'free modunda' degil.
    Invoke-RtIpc camera.set_pivot_mode @{ mode = 'selection' } | Out-Null
    Invoke-RtIpc select.clear @{} | Out-Null
    $p6 = Invoke-RtIpc camera.get_pivot @{}
    Check "mode HALA selection" ($p6.mode -eq 'selection') ("mode={0}" -f $p6.mode)
    Check "locked=false" ($p6.locked -eq $false) ("locked={0}" -f $p6.locked)

    # ── 7. Serbest mod kilidi gercekten birakmali ───────────────────────────
    Write-Host "7. camera.set_pivot_mode free kilidi birakmali"
    Invoke-RtIpc camera.set_pivot_mode @{ mode = 'free' } | Out-Null
    $p7 = Invoke-RtIpc camera.get_pivot @{}
    Check "mode=free" ($p7.mode -eq 'free') ("mode={0}" -f $p7.mode)
    Check "locked=false" ($p7.locked -eq $false) ("locked={0}" -f $p7.locked)
}
finally {
    Invoke-RtIpc camera.set_pivot_mode @{ mode = [string]$piv0.mode } | Out-Null
    Invoke-RtIpc camera.set_focus_distance @{ focus_distance = [double]$cam0.focus_distance } | Out-Null
    Invoke-RtIpc camera.set_position @{ position = @([double]$cam0.position[0], [double]$cam0.position[1], [double]$cam0.position[2]) } | Out-Null
    Invoke-RtIpc camera.set_target @{ target = @([double]$cam0.target[0], [double]$cam0.target[1], [double]$cam0.target[2]) } | Out-Null
}

Write-Host ""
if ($fails -eq 0) { Write-Host "TUM KAPILAR GECTI" -ForegroundColor Green }
else { Write-Host ("{0} KAPI KALDI" -f $fails) -ForegroundColor Red; exit 1 }
