<#
.SYNOPSIS
    Bulut hacimleri V0+V1 kabul testi (docs/dev/ATMOSPHERE_CLOUD_VOLUMES.md §4).

.DESCRIPTION
    Tamamen SAYISAL; goruntu okumaz. Firtina hucreleri ve kolon acik bir
    sahne ister (storms.enabled + storms.column + en az bir gelismis hucre).

      0. V0: get_clouds.column_levels gecerli; LCL = en alcak acik katman
         tabani, tropopoz = max(storms.column_top_m, LCL + 1500).
      1. Kapaliyken taban cizgisi: 5 noktada yogunluk kaydedilir.
      2. Acilinca: viewport cihazinda generator_ready, atlas_slots >= 1,
         live_slots >= 1, repaints_total > 0.
      3. Sekil (V2 kabarcik kulesi), hucre merkezinden:
         3a. YUKSEKLIGE BAGLI: tabanin %25 ve %60 ustundeki yatay volume_sdf
             profilleri birbirinden > 150 m farkli (kabuk modelinde birebir
             ayniydi: dusey uzatilmis 2B alan).
         3b. TABAN (V3): 0.4 R cevresindeki 4 dusey hattin alt gecisi kutu tabani ile LCL arasinda (loblar, duvar bulutu, scud LCL altina sarkar, ustune cikmaz).
         3c. DETERMINIZM: ayni profil iki kez -> bit bit ayni.
      4. Sabit sonum: derin icteki (sdf < -400 m) iki noktada yogunluk ESIT.
      5. Kapatinca: atlas_slots 0 ve 1'deki 5 nokta BIT BIT ayni.

    ★ En sinsi basarisizlik: 3a gecer ama goruntude kule yine "egg" -> govde
      zinciri (0.6 R dev kabarciklar) turret'leri yutuyor; turret detail 0
      ise yalniz govde kalir. Goz testinde turret'ler ayri tepeler olmali.

    Baslangic bulut durumu sonunda geri yazilir.
#>
[CmdletBinding()]
param([string]$Backend = 'viewport')

Import-Module "$PSScriptRoot\RtIpc.psm1" -Force
# Any error is a FAIL, never a silent PASS (a statement-terminating
# exception used to let the script run on and print PASS).
$ErrorActionPreference = 'Stop'

$fails = 0
function Check([bool]$ok, [string]$what) {
    if ($ok) { Write-Host "  OK   $what" -ForegroundColor Green }
    else     { Write-Host "  FAIL $what" -ForegroundColor Red; $script:fails++ }
}
function Sample([string]$mode, $pts) {
    $list = New-Object System.Collections.ArrayList
    foreach ($p in $pts) { [void]$list.Add(@([double]$p[0], [double]$p[1], [double]$p[2])) }
    (Invoke-RtIpc world.sample_clouds @{ mode = $mode; points = $list.ToArray(); backend = $Backend }).values
}
function VolumeStats { ((Invoke-RtIpc world.cloud_stats @{}).backends | Where-Object role -eq $Backend).volumes }
# Zero crossings of a sampled line: positions where the sign flips (linear interpolation).
function Crossings($xs, $vs) {
    $out = @()
    for ($i = 1; $i -lt $vs.Count; $i++) {
        $a = [double]$vs[$i - 1]; $b = [double]$vs[$i]
        # Leaving the box (1e6) straight into cloud: the box face is the
        # crossing. Skipped, a base only ~60 m above the box bottom (less than
        # a sample step) lost its crossing and the body exit was taken for
        # the base (2026-10-14).
        if ([math]::Abs($a) -ge 1e5 -and $b -lt 0) { $out += $xs[$i]; continue }
        if ($a -lt 0 -and [math]::Abs($b) -ge 1e5) { $out += $xs[$i - 1]; continue }
        if (($a -lt 0) -ne ($b -lt 0) -and [math]::Abs($a) -lt 1e5 -and [math]::Abs($b) -lt 1e5) {
            $t = $a / ($a - $b)
            $out += $xs[$i - 1] + $t * ($xs[$i] - $xs[$i - 1])
        }
    }
    $out
}

$orig = Invoke-RtIpc world.get_clouds @{}
try {
    Write-Host "0. column levels (V0)"
    $lv = $orig.column_levels
    $bases = @($orig.layers | Where-Object { $_.enabled -and $_.extinction_per_m -gt 0 } | ForEach-Object { $_.base_altitude_m })
    Check ($lv.valid -and $bases.Count -gt 0) "column_levels valid"
    if ($lv.valid) {
        $lcl = ($bases | Measure-Object -Minimum).Minimum
        Check ([math]::Abs($lv.lcl_m - $lcl) -lt 0.5) ("LCL {0:0} = lowest base {1:0}" -f $lv.lcl_m, $lcl)
        $tropo = [math]::Max($orig.storms.column_top_m, $lcl + 1500)
        Check ([math]::Abs($lv.tropopause_m - $tropo) -lt 0.5) ("tropopause {0:0} = {1:0}" -f $lv.tropopause_m, $tropo)
    }
    if (-not ($orig.storms.enabled -and $orig.storms.column)) {
        Write-Host "  SKIP 1-5: storms.enabled + storms.column needed" -ForegroundColor Yellow
        return
    }

    Write-Host "1. baseline with volumes off"
    Invoke-RtIpc world.set_clouds @{ volumes = @{ enabled = $false } } | Out-Null
    Start-Sleep -Milliseconds 800
    $cells = (Invoke-RtIpc world.cloud_stats @{}).storms.cells | Sort-Object tower -Descending
    $c0 = $cells | Select-Object -First 1
    $probe = @()
    foreach ($dy in 1000, 2000, 3000) { $probe += , @($c0.position_m[0], $dy, $c0.position_m[2]) }
    $probe += , @(($c0.position_m[0] + 800), 1500, $c0.position_m[2])
    $probe += , @($c0.position_m[0], 1500, ($c0.position_m[2] - 800))
    $before = Sample 'density' $probe

    Write-Host "2. volumes on"
    Invoke-RtIpc world.set_clouds @{ volumes = @{ enabled = $true; slots = 8 } } | Out-Null
    Start-Sleep -Milliseconds 1500
    $vs = VolumeStats
    Check ($vs.generator_ready) "generator_ready"
    Check ($vs.atlas_slots -ge 1) "atlas_slots = $($vs.atlas_slots)"
    Check ($vs.live_slots -ge 1) "live_slots = $($vs.live_slots)"
    Check ($vs.repaints_total -gt 0) "repaints_total = $($vs.repaints_total)"

    if ($vs.live_slots -ge 1) {
        $box = $vs.boxes[0]
        $R = [double]$box.radius_m
        $cx = [double]$box.min[0] + 0.5 * [double]$box.size[0]
        $cz = [double]$box.min[2] + 0.5 * [double]$box.size[2]
        $vox = [double]$box.size[0] / 128.0
        $voxY = [double]$box.size[1] / 128.0
        $step = [math]::Max($voxY / 2, 25)
        $ys = @(); for ($y = -3000; $y -le 16000; $y += $step) { $ys += $y }
        $pts = @(); foreach ($y in $ys) { $pts += , @($cx, $y, $cz) }
        $zc = @(Crossings $ys (Sample 'volume_sdf' $pts))
        Write-Host "3. V2 tower (cell id $($box.cell_id), R $([math]::Round($R)) m)"
        Check ($zc.Count -ge 2) "vertical line through the core: $($zc.Count) crossings"
        if ($zc.Count -ge 2) {
            # First segment = the tower body (a second one is the anvil above it).
            $yb = $zc[0]; $yt = $zc[1]
            $xs = @(); for ($x = ($cx - 1.5 * $R); $x -le ($cx + 1.5 * $R); $x += ($vox / 2)) { $xs += $x }
            $y1 = $yb + 0.25 * ($yt - $yb); $y2 = $yb + 0.6 * ($yt - $yb)
            $p1 = @(); $p2 = @()
            foreach ($x in $xs) { $p1 += , @($x, $y1, $cz); $p2 += , @($x, $y2, $cz) }
            $s1 = Sample 'volume_sdf' $p1
            $s2 = Sample 'volume_sdf' $p2
            $maxDiff = 0.0; $in1 = 0; $in2 = 0
            for ($i = 0; $i -lt $s1.Count; $i++) {
                if ($s1[$i] -lt 0) { $in1++ }
                if ($s2[$i] -lt 0) { $in2++ }
                if ([math]::Abs($s1[$i]) -lt 1e5 -and [math]::Abs($s2[$i]) -lt 1e5) {
                    $maxDiff = [math]::Max($maxDiff, [math]::Abs($s1[$i] - $s2[$i]))
                }
            }
            Check ($in1 -gt 0 -and $in2 -gt 0) "both heights cut the cloud ($in1 / $in2 samples inside)"
            Check ($maxDiff -gt 150) ("3a profiles differ with height: max {0:0} m (> 150)" -f $maxDiff)

            $bottoms = @()
            foreach ($o in @(@(0.4, 0), @(-0.4, 0), @(0, 0.4), @(0, -0.4))) {
                $px = $cx + $o[0] * $R; $pz = $cz + $o[1] * $R
                $yy = @(); for ($y = ($yb - 1500); $y -le ($yb + 2500); $y += ($voxY / 4)) { $yy += $y }
                $pp = @(); foreach ($y in $yy) { $pp += , @($px, $y, $pz) }
                $c = @(Crossings $yy (Sample 'volume_sdf' $pp))
                if ($c.Count -gt 0) { $bottoms += $c[0] }
            }
            # V3: the base hangs in lobes (and scud) below the LCL, never
            # above it, and never below the box's hang room.
            $lcl = [double]$lv.lcl_m
            $boxBottom = [double]$box.min[1]
            $inBand = @($bottoms | Where-Object { $_ -le ($lcl + 1.5 * $voxY) -and $_ -ge ($boxBottom - 1.5 * $voxY) }).Count
            $txt = ($bottoms | ForEach-Object { "{0:0}" -f $_ }) -join ' '
            Check ($bottoms.Count -ge 3 -and $inBand -eq $bottoms.Count) ("3b base between box bottom {0:0} and LCL {1:0}: {2} of {3} ({4})" -f $boxBottom, $lcl, $inBand, $bottoms.Count, $txt)
            $hanging = @($bottoms | Where-Object { $_ -lt ($lcl - 1.5 * $voxY) }).Count
            Write-Host ("       {0} of {1} bottoms hang below the LCL (lobes / wall cloud / scud)" -f $hanging, $bottoms.Count)

            $again = Sample 'volume_sdf' $p1
            $det = $true; for ($i = 0; $i -lt $s1.Count; $i++) { if ($s1[$i] -ne $again[$i]) { $det = $false } }
            Check $det "3c deterministic"

            Write-Host "4. constant extinction inside"
            $deep = @(); for ($i = 0; $i -lt $s1.Count; $i++) { if ($s1[$i] -lt -400) { $deep += , $p1[$i] } }
            if ($deep.Count -ge 2) {
                $d = Sample 'density' @($deep[0], $deep[$deep.Count - 1])
                Check ($d[0] -gt 0 -and [math]::Abs($d[0] - $d[1]) -le 1e-6 * [math]::Max($d[0], 1e-9)) ("{0:0.00000} == {1:0.00000}" -f $d[0], $d[1])
            } else {
                Write-Host "  SKIP no point deeper than 400 m on the profile" -ForegroundColor Yellow
            }
        }
    }

    Write-Host "5. volumes off -> unchanged"
    Invoke-RtIpc world.set_clouds @{ volumes = @{ enabled = $false } } | Out-Null
    Start-Sleep -Milliseconds 1500
    $vs = VolumeStats
    Check ($vs.atlas_slots -eq 0) "atlas released (atlas_slots = $($vs.atlas_slots))"
    $after = Sample 'density' $probe
    $same = $true
    for ($i = 0; $i -lt $before.Count; $i++) { if ($before[$i] -ne $after[$i]) { $same = $false } }
    Check $same ("baseline densities identical: " + (($before | ForEach-Object { '{0:0.0000}' -f $_ }) -join ' '))
}
catch {
    Write-Host "  FAIL exception: $_" -ForegroundColor Red
    $fails++
}
finally {
    Invoke-RtIpc world.set_clouds @{ volumes = @{ enabled = $orig.volumes.enabled; slots = $orig.volumes.slots;
        voxel_m = $orig.volumes.voxel_m; skin_m = $orig.volumes.skin_m; min_tower = $orig.volumes.min_tower } } | Out-Null
}
if ($fails -eq 0) { Write-Host "PASS" -ForegroundColor Green } else { Write-Host "FAIL ($fails)" -ForegroundColor Red }
