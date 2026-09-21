# Uzak vs yakin kol: ayni sahne, ayni ayarlar, FARKLI EKRAN KAPLAMASI.
# Ucgen sayisi ile GPU ms'in ayni yone mi ters mi gittigini olcer.
#   ayni yon  -> ucgen/vertex bagli
#   ters yon  -> fragman bagli (overdraw / alpha-test)
param([double]$CloseFactor = 0.12, [int]$Frames = 22)
Import-Module "E:\RayTrophi_projesi\raytracing_Proje_Moduler\scripts\ipc\RtIpc.psm1" -Force
$sig = @'
using System;using System.Runtime.InteropServices;
public class FgWin2 {
 [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
 [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
 [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
}
'@
Add-Type -TypeDefinition $sig -ErrorAction SilentlyContinue
$app = Get-Process RayTrophiStudio -ErrorAction Stop
$prev = [FgWin2]::GetForegroundWindow()
[void][FgWin2]::ShowWindow($app.MainWindowHandle, 9)
[void][FgWin2]::SetForegroundWindow($app.MainWindowHandle)
Start-Sleep -Seconds 2

$cam = Invoke-RtIpc camera.get
$eye = @([double]$cam.position[0], [double]$cam.position[1], [double]$cam.position[2])
$tgt = @([double]$cam.target[0],   [double]$cam.target[1],   [double]$cam.target[2])

function Set-Eye([double[]]$p) {
    Invoke-RtIpc camera.set_position @{ position = @(($p[0]+0.0), ($p[1]+0.0), ($p[2]+0.0)) } | Out-Null
}
function Measure-Arm([string]$label, [double[]]$base) {
    Invoke-RtIpc viewport.reset_frame_timings | Out-Null
    # Kamerayi hedefe dogru kucuk adimlarla titret: her yazi BIR kare satin alir.
    for ($i = 1; $i -le $Frames; $i++) {
        $k = 1.0 + ($i % 2) * 0.0015
        Set-Eye @(($tgt[0] + ($base[0]-$tgt[0])*$k),
                  ($tgt[1] + ($base[1]-$tgt[1])*$k),
                  ($tgt[2] + ($base[2]-$tgt[2])*$k))
    }
    $t = Invoke-RtIpc viewport.frame_timings
    $mp = ($t.stages | Where-Object { $_.name -eq 'main_pass' }).gpu_mean_ms
    $tr = ($t.stages | Where-Object { $_.name -eq 'transmission' }).gpu_mean_ms
    $dp = ($t.stages | Where-Object { $_.name -eq 'depth_prepass' }).gpu_mean_ms
    [pscustomobject]@{
        kol = $label; tris = $t.applied.visible_triangles; draws = $t.applied.draw_calls
        frame_gpu = [math]::Round($t.frame_gpu_mean_ms,1)
        main_pass = [math]::Round($mp,1); transmission = [math]::Round($tr,1)
        depth_prepass = [math]::Round($dp,1); kare = $t.frames
    }
}

$far = Measure-Arm 'UZAK (mevcut)' $eye
$near = @(($tgt[0] + ($eye[0]-$tgt[0])*$CloseFactor),
          ($tgt[1] + ($eye[1]-$tgt[1])*$CloseFactor),
          ($tgt[2] + ($eye[2]-$tgt[2])*$CloseFactor))
$nearArm = Measure-Arm ("YAKIN (x{0})" -f $CloseFactor) $near
$far2 = Measure-Arm 'UZAK (tekrar)' $eye

Set-Eye $eye
Invoke-RtIpc camera.set_target @{ target = @(($tgt[0]+0.0),($tgt[1]+0.0),($tgt[2]+0.0)) } | Out-Null
if ($prev -ne [IntPtr]::Zero) { [void][FgWin2]::SetForegroundWindow($prev) }

@($far, $nearArm, $far2) | Format-Table -AutoSize
$dTri = [double]$nearArm.tris - [double]$far.tris
$dMs  = [double]$nearArm.main_pass - [double]$far.main_pass
"ucgen degisimi : {0:N0}" -f $dTri
"main_pass degisimi: {0:N1} ms" -f $dMs
if ($dTri -lt 0 -and $dMs -gt 0) { "VERDICT: ucgen DUSTU, maliyet ARTTI -> FRAGMAN BAGLI (overdraw / alpha-test)" }
elseif ($dTri -gt 0 -and $dMs -gt 0) { "VERDICT: ikisi de artti -> ayrisamadi, CloseFactor'u kucult" }
else { "VERDICT: maliyet ucgenle ayni yonde -> geometri bagli" }
