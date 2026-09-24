# Probe-SssResponse.ps1 — does each SSS dial actually move the render?
#
# The SSS plan (docs/dev/VULKAN_RT_SSS_ANALIZ_VE_IYILESTIRME_PLANI.md) promised
# RGB radius tinting, an effective IOR and no double Beer-Lambert darkening.
# None of it could be measured while the SSS params were panel-only. This probe
# drives them through material.set and compares mean image colour.
#
# Usage: open a lit scene with a closed mesh (sphere/head) in view, Vulkan RT.
#   .\scripts\ipc\Probe-SssResponse.ps1 -Object Sphere
param(
    [Parameter(Mandatory)] [string]$Object,
    [int]$Spp = 64,
    [string]$OutDir = "$env:TEMP\rt_sss_probe"
)
$ErrorActionPreference = 'Stop'
Import-Module "$PSScriptRoot\RtIpc.psm1" -Force
Add-Type -AssemblyName System.Drawing
New-Item -ItemType Directory -Force $OutDir | Out-Null

function Set-Mat($p, $v) { Invoke-RtIpc material.set @{ object_name = $Object; param = $p; value = $v } | Out-Null }
function Get-Mat($p)     { Invoke-RtIpc material.get @{ object_name = $Object; param = $p } }

function Render-Mean($tag) {
    $path = "$OutDir\$tag.png"
    Invoke-RtIpc render.start @{ output_path = $path; spp = $Spp } | Out-Null
    for ($i = 0; $i -lt 1200; $i++) { if ((Invoke-RtIpc render.status).state -ne 'rendering') { break }; Start-Sleep -Milliseconds 250 }
    $bmp = [System.Drawing.Bitmap]::FromFile($path)
    try {
        $r = 0.0; $g = 0.0; $b = 0.0; $n = 0
        for ($y = 0; $y -lt $bmp.Height; $y += 4) {
            for ($x = 0; $x -lt $bmp.Width; $x += 4) {
                $c = $bmp.GetPixel($x, $y); $r += $c.R; $g += $c.G; $b += $c.B; $n++
            }
        }
        $m = [pscustomobject]@{ tag = $tag; r = $r / $n; g = $g / $n; b = $b / $n }
        '{0,-14} R={1,7:N2} G={2,7:N2} B={3,7:N2}' -f $m.tag, $m.r, $m.g, $m.b | Write-Host
        return $m
    } finally { $bmp.Dispose() }
}

# 1) API round trip — every SSS key must be writable AND read back the same.
$fail = 0
$checks = @(
    @('subsurface', 1.0), @('subsurface_scale', 0.2), @('subsurface_anisotropy', 0.3),
    @('subsurface_ior', 1.4), @('subsurface_color', @(1, 1, 1)), @('subsurface_radius', @(1, 1, 1)),
    @('subsurface_method', 1.0), @('subsurface_method', 0.0), @('subsurface_max_steps', 64.0)
)
foreach ($c in $checks) {
    Set-Mat $c[0] $c[1]
    $got = Get-Mat $c[0]
    $want = ($c[1] | ForEach-Object { [double]$_ }) -join ','
    $have = ($got | ForEach-Object { [math]::Round([double]$_, 4) }) -join ','
    if ($have -ne $want) { Write-Host "FAIL roundtrip $($c[0]): want $want got $have"; $fail++ }
}
if ($fail -eq 0) { Write-Host 'PASS roundtrip: all SSS keys' }

# Neutral baseline: white base colour, white SSS colour.
Set-Mat base_color @(0.8, 0.8, 0.8)
Set-Mat subsurface_color @(0.8, 0.8, 0.8)
Set-Mat subsurface_anisotropy 0.0

# 2) Energy: SSS=1 with the same colour must not be far darker than diffuse.
Set-Mat subsurface 0.0
$diff = Render-Mean 'diffuse'
Set-Mat subsurface 1.0
$neutral = Render-Mean 'sss_neutral'
$ratio = ($neutral.r + $neutral.g + $neutral.b) / [math]::Max(1e-3, $diff.r + $diff.g + $diff.b)
'energy sss/diffuse = {0:N3}  (expect ~0.7..1.1; old code ~0.3 = double darkening)' -f $ratio | Write-Host

# 3) RGB radius must TINT: long red radius => red/green rises vs neutral.
Set-Mat subsurface_radius @(1.0, 0.2, 0.1)
$red = Render-Mean 'sss_red_radius'
$tint0 = $neutral.r / [math]::Max(1e-3, $neutral.g)
$tint1 = $red.r / [math]::Max(1e-3, $red.g)
'R/G neutral={0:N3} red-radius={1:N3}  -> {2}' -f $tint0, $tint1, $(if ($tint1 -gt $tint0 * 1.03) { 'PASS' } else { 'FAIL: radius does not tint' }) | Write-Host
Set-Mat subsurface_radius @(1, 1, 1)

# 4) IOR must change the image.
Set-Mat subsurface_ior 1.01
$i1 = Render-Mean 'sss_ior_1.01'
Set-Mat subsurface_ior 2.5
$i2 = Render-Mean 'sss_ior_2.5'
$d = [math]::Abs($i1.r - $i2.r) + [math]::Abs($i1.g - $i2.g) + [math]::Abs($i1.b - $i2.b)
'IOR delta = {0:N2}  -> {1}' -f $d, $(if ($d -gt 0.5) { 'PASS' } else { 'FAIL/weak: IOR has no visible effect' }) | Write-Host
Set-Mat subsurface_ior 1.4

# 5) Walk cap is a BIAS: energy must rise with steps and flatten by ~64.
$caps = @{}
foreach ($n in 8, 32, 64, 256) {
    Set-Mat subsurface_max_steps $n
    $m = Render-Mean "sss_steps_$n"
    $caps[$n] = $m.r + $m.g + $m.b
}
'steps 8/32/64/256 sum = {0:N1} / {1:N1} / {2:N1} / {3:N1}' -f $caps[8], $caps[32], $caps[64], $caps[256] | Write-Host
$conv = [math]::Abs($caps[256] - $caps[64]) / [math]::Max(1e-3, $caps[256])
'64 vs 256 gap = {0:P1}  -> {1}' -f $conv, $(if ($conv -lt 0.03) { 'PASS (64 converged)' } else { 'WARN: 64 too low for this material' }) | Write-Host
Set-Mat subsurface_max_steps 64

# 6) Fast vs Random Walk: same colour family, fast has no bleed.
Set-Mat subsurface_method 1
$fast = Render-Mean 'sss_fast'
Set-Mat subsurface_method 0
$ratioFast = ($fast.r + $fast.g + $fast.b) / [math]::Max(1e-3, $neutral.r + $neutral.g + $neutral.b)
'fast/walk energy = {0:N3}  (expect ~0.8..1.2; far off = the two methods disagree on colour)' -f $ratioFast | Write-Host

Write-Host "images: $OutDir"
