# Probe-FrameStages.ps1 -- per-pass CPU/GPU cost of the raster viewport frame,
# and an optional A/B between two viewport settings.
#
# ★★★★★ WHY IT DRIVES THE CAMERA. The raster viewport renders ONLY when it is
#   marked dirty. Measured 2026-09-09: 24 s of a still camera produced ZERO
#   raster frames while the display loop happily re-presented one old image, and
#   a wall-clock FPS A/B over that correctly reported "noise". A camera write
#   buys exactly one frame, so N writes buy N frames. Nothing else in the IPC
#   surface reliably does.
#
# ★★★ WHY THE WINDOW IS IN C++ AND NOT HERE. Every IPC call waits a display-loop
#   tick (~54 ms measured), so pulling one sample per call would price the
#   instrument higher than the thing it measures. reset -> drive -> read means
#   one round trip for the reset, N for the camera, one for the answer.
#
# ★★ viewport.render_frames DOES NOT WORK for this: it drives the path tracer.
#   It reported 225 ms/frame while frames_submitted never moved.
#
# ★ The app draws nothing while its window is unfocused, so the window is
#   brought forward and handed back.

param(
    [int]$Frames = 60,
    [double]$OrbitDegrees = 0.35,
    # Optional A/B. Each arm is a scriptblock-free switch name understood below.
    [ValidateSet('none', 'rt_shadow', 'depth_prepass', 'gpu_culling')]
    [string]$Compare = 'none',
    [switch]$KeepFocus
)

Import-Module "$PSScriptRoot\RtIpc.psm1" -Force

$sig = @'
using System;using System.Runtime.InteropServices;
public class FgWin {
 [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
 [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
 [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
}
'@
Add-Type -TypeDefinition $sig -ErrorAction SilentlyContinue

$app = Get-Process RayTrophiStudio -ErrorAction Stop
$prevFg = [FgWin]::GetForegroundWindow()
[void][FgWin]::ShowWindow($app.MainWindowHandle, 9)
[void][FgWin]::SetForegroundWindow($app.MainWindowHandle)
Start-Sleep -Seconds 2

$cam = Invoke-RtIpc camera.get
$eye = @([double]$cam.position[0], [double]$cam.position[1], [double]$cam.position[2])
$tgt = @([double]$cam.target[0],   [double]$cam.target[1],   [double]$cam.target[2])

$dx = $eye[0] - $tgt[0]
$dz = $eye[2] - $tgt[2]
$radius = [math]::Sqrt($dx * $dx + $dz * $dz)
$angle0 = [math]::Atan2($dz, $dx)

function Invoke-Orbit([int]$n) {
    # A small orbit rather than a re-set to the same value: a backend that skips
    # a no-op write would leave the viewport clean and measure nothing.
    for ($i = 1; $i -le $n; $i++) {
        $a = $angle0 + ($i * $OrbitDegrees * [math]::PI / 180.0)
        # ★ Every element parenthesised: in PowerShell the comma binds TIGHTER
        #   than arithmetic, so `$a + $b * $c, $d` parses as `$a + ($b * ($c, $d))`
        #   and fails with "Object[] does not contain op_Multiply". Measured here.
        $ex = $tgt[0] + ($radius * [math]::Cos($a))
        $ez = $tgt[2] + ($radius * [math]::Sin($a))
        Invoke-RtIpc camera.set_position @{ position = @($ex, $eye[1], $ez) } | Out-Null
    }
}

function Restore-Camera {
    Invoke-RtIpc camera.set_position @{ position = $eye } | Out-Null
    Invoke-RtIpc camera.set_target   @{ target   = $tgt } | Out-Null
}

function Measure-Window([string]$label) {
    Invoke-RtIpc viewport.reset_frame_timings | Out-Null
    Invoke-Orbit $Frames
    $t = Invoke-RtIpc viewport.frame_timings
    [pscustomobject]@{ label = $label; timings = $t }
}

function Show-Window($run) {
    $t = $run.timings
    ""
    "=== $($run.label) ==="
    if (-not $t.available) {
        "  NO FRAMES MEASURED"
        foreach ($w in $t.warnings) { "  ! $w" }
        return
    }
    $a = $t.applied
    ("  applied : {0} / {1} / {2}  {3}x{4}" -f $a.shading, $a.quality_preset, $a.lighting_preset, $a.width, $a.height)
    ("            depth_prepass={0} gpu_culling={1} rt_shadow ready={2} cascades_replaced={3}" -f `
        $a.depth_prepass, $a.gpu_culling, $a.rt_shadow_ready, $a.rt_cascades_replaced)
    ("            lights={0} shadowed={1} volumes={2} tris={3:N0} draws={4}" -f `
        $a.scene_lights, $a.shadowed_lights, $a.volume_count, $a.visible_triangles, $a.draw_calls)
    ("  frames  : {0} ({1} with GPU marks) over {2:N0} ms wall" -f $t.frames, $t.frames_with_gpu, $t.window_wall_ms)
    ("  frame   : cpu {0,8:N3} ms (p95 {1,8:N3})   gpu {2,8:N3} ms (p95 {3,8:N3})" -f `
        $t.frame_cpu_mean_ms, $t.frame_cpu_p95_ms, $t.frame_gpu_mean_ms, $t.frame_gpu_p95_ms)
    ""
    "  {0,-14} {1,10} {2,10} {3,10} {4,10}  {5}" -f 'stage', 'cpu ms', 'cpu p95', 'gpu ms', 'gpu p95', 'ran'
    foreach ($s in $t.stages) {
        # ★ "ran 0" with 0 ms is a RESULT (the stage was skipped every frame),
        #   not missing data. Printed as a word so it cannot be read as a zero.
        $ran = if ($s.frames_ran -eq 0) { 'SKIPPED' } else { "$($s.frames_ran)" }
        "  {0,-14} {1,10:N3} {2,10:N3} {3,10:N3} {4,10:N3}  {5}" -f `
            $s.name, $s.cpu_mean_ms, $s.cpu_p95_ms, $s.gpu_mean_ms, $s.gpu_p95_ms, $ran
    }
    foreach ($w in $t.warnings) { "  ! $w" }
}

$runs = @()
switch ($Compare) {
    'none' { $runs += Measure-Window 'current settings' }
    'rt_shadow' {
        Invoke-RtIpc viewport.set_rt_shadow @{ enabled = $true } | Out-Null
        $runs += Measure-Window 'rt_shadow ON'
        Invoke-RtIpc viewport.set_rt_shadow @{ enabled = $false } | Out-Null
        $runs += Measure-Window 'rt_shadow OFF'
        Invoke-RtIpc viewport.set_rt_shadow @{ enabled = $true } | Out-Null
        # ★ The second ON arm is not redundant: if it disagrees with the first by
        #   as much as the A/B gap, the whole comparison is drift, not an effect.
        $runs += Measure-Window 'rt_shadow ON (repeat)'
    }
    'depth_prepass' {
        Invoke-RtIpc viewport.set_raster_depth_prepass @{ enabled = $true } | Out-Null
        $runs += Measure-Window 'depth prepass ON'
        Invoke-RtIpc viewport.set_raster_depth_prepass @{ enabled = $false } | Out-Null
        $runs += Measure-Window 'depth prepass OFF'
        Invoke-RtIpc viewport.set_raster_depth_prepass @{ enabled = $true } | Out-Null
        $runs += Measure-Window 'depth prepass ON (repeat)'
    }
    'gpu_culling' {
        Invoke-RtIpc viewport.set_raster_gpu_instancing @{ enabled = $true } | Out-Null
        $runs += Measure-Window 'gpu instancing ON'
        Invoke-RtIpc viewport.set_raster_gpu_instancing @{ enabled = $false } | Out-Null
        $runs += Measure-Window 'gpu instancing OFF'
        Invoke-RtIpc viewport.set_raster_gpu_instancing @{ enabled = $true } | Out-Null
        $runs += Measure-Window 'gpu instancing ON (repeat)'
    }
}

Restore-Camera
if (-not $KeepFocus -and $prevFg -ne [IntPtr]::Zero) {
    [void][FgWin]::SetForegroundWindow($prevFg)
}

foreach ($r in $runs) { Show-Window $r }

if ($runs.Count -eq 3) {
    $a = $runs[0].timings; $b = $runs[1].timings; $c = $runs[2].timings
    if ($a.available -and $b.available -and $c.available) {
        $onMean = ($a.frame_gpu_mean_ms + $c.frame_gpu_mean_ms) / 2.0
        $spread = [math]::Abs($a.frame_gpu_mean_ms - $c.frame_gpu_mean_ms)
        $delta  = $b.frame_gpu_mean_ms - $onMean
        ""
        "GPU frame: arm A (mean of two) {0:N3} ms, arm B {1:N3} ms, difference {2:N3} ms" -f `
            $onMean, $b.frame_gpu_mean_ms, $delta
        "repeat-arm spread {0:N3} ms" -f $spread
        if ($spread -ge [math]::Abs($delta)) {
            "VERDICT: NOISE -- the two identical arms differ by as much as the A/B gap."
        } else {
            "VERDICT: gap exceeds the repeat spread by {0:N1}x." -f ([math]::Abs($delta) / [math]::Max($spread, 0.001))
        }
    }
}
