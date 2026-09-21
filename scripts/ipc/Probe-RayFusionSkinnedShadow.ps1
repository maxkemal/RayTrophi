<#
.SYNOPSIS
  Does the RayFusion scene acceleration structure follow a SKINNED mesh, or is
  it still describing the pose of the frame it was built in?
.DESCRIPTION
  The bug this probes is silent by construction. GPU skinning rewrites the
  CONTENTS of the vertex buffer the BLAS borrows; the handle, the device
  address and the vertex count all stay put. Both scene-AS signatures therefore
  match, nothing rebuilds, and every counter reports a healthy "ready" AS while
  the character animates in the raster image and casts a FROZEN shadow.

  So this script does not ask "is it ready". It steps the timeline and asks
  whether skin_refits MOVED -- and, just as importantly, whether it STOPS
  moving when the timeline is parked.
.NOTES
  The user builds and starts the application, and loads a scene containing an
  animated skinned character, before running this. The viewport must be in a
  raster shading mode (Solid / Matcap / Material Preview): the skinned raster
  buffers this measures are not written in any other mode, so a path-traced
  viewport would report a truthful zero for the wrong reason.
  The script moves the playhead and puts it back.
#>
[CmdletBinding()]
param(
    [int]$Frames = 6
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'RtIpc.psm1') -Force
$failures = 0
function Check([string]$Name, [bool]$Ok, [string]$Detail = '') {
    if ($Ok) { Write-Host "[PASS] $Name" }
    else {
        Write-Host "[FAIL] $Name" -ForegroundColor Red
        if ($Detail) { Write-Host "       $Detail" -ForegroundColor DarkYellow }
        $script:failures++
    }
}

# ★ IPC WRITE-THEN-MEASURE. The raster frame is drawn by the application's own
#   display loop, not by this pipe: a write dirties the viewport and the frame
#   happens between calls. Reading a counter in the same breath as the write
#   that should move it measures the PREVIOUS frame. Every read below is
#   separated from its write by at least one other round trip.
function Wait-Frame {
    [void](Invoke-RtIpc viewport.status @{})
    [void](Invoke-RtIpc viewport.status @{})
}

try {
    Connect-RtIpc -TimeoutMs 3000

    $shading = (Invoke-RtIpc viewport.status @{}).shading
    Write-Host "       shading=$shading"

    Wait-Frame
    $before = Invoke-RtIpc rayfusion.scene_as @{}

    Check 'Hardware ray tracing is available' ($before.hardware_rt -eq $true) `
        'Without it no RayFusion ray step runs at all, and this probe proves nothing.'
    Check 'Scene AS is ready' ($before.ready -eq $true) $before.inactive_reason

    # The whole probe rests on this: a scene with no skinned BLAS cannot show
    # the bug, and a PASS on it would be the instrument lying.
    if ($before.blas_skinned -eq 0) {
        Write-Host '[SKIP] No skinned BLAS in this scene -- load an animated character first.' -ForegroundColor Yellow
        Write-Host "       blas_count=$($before.blas_count) indexed=$($before.blas_indexed) flat=$($before.blas_flat)"
        exit 2
    }
    Write-Host "       blas_skinned=$($before.blas_skinned) of blas_count=$($before.blas_count)"

    # timeline.get_frame returns a BARE SCALAR, not an object -- reading
    # `.frame` on it yields $null, every set_frame below lands on frame 1..N
    # regardless of where the playhead was, and the restore at the end silently
    # jumps the scene to frame 0. Measured 2026-09-13.
    $startFrame = [int](Invoke-RtIpc timeline.get_frame @{})
    $refitsBefore = [int64]$before.skin_refits
    $buildsBefore = [int64]$before.builds
    $failsBefore  = [int64]$before.skin_refit_failures

    for ($i = 1; $i -le $Frames; $i++) {
        [void](Invoke-RtIpc timeline.set_frame @{ frame = ($startFrame + $i) })
        Wait-Frame
    }
    $after = Invoke-RtIpc rayfusion.scene_as @{}
    [void](Invoke-RtIpc timeline.set_frame @{ frame = $startFrame })

    $refitDelta = [int64]$after.skin_refits - $refitsBefore
    # THE measurement. Stuck at zero is the stale-shadow bug: the raster image
    # animates and the traced structure does not.
    Check 'Stepping the timeline re-fit the skinned BLASes' ($refitDelta -gt 0) `
        "skin_refits did not move across $Frames frames (still $($after.skin_refits)). The traced shadow is frozen at the build pose."
    Check 'No refit failed' ([int64]$after.skin_refit_failures -eq $failsBefore) `
        "skin_refit_failures rose to $($after.skin_refit_failures) -- a BLAS was not built updatable, which is a programming error, not a transient state."

    # A full teardown per frame would ALSO make the shadow correct, and would
    # hide a cost nobody budgeted behind a passing test. Separate the two.
    Check 'Deformation did NOT force full AS rebuilds' ([int64]$after.builds -eq $buildsBefore) `
        "builds rose from $buildsBefore to $($after.builds): this is paying full-rebuild price for what should be a refit."

    Write-Host "       skin_refits +$refitDelta, last_skin_refit_ms=$($after.last_skin_refit_ms)"

    # ★★★★★ The quietest failure of all is the PAUSED case, and it is not
    #   hypothetical: this check caught exactly that on 2026-09-13. The
    #   generation was bumped whenever the skinning DISPATCH ran, and that
    #   dispatch runs every frame whether or not the pose changed -- so a
    #   timeline parked at frame 0 paid 2.18 ms of refit about 33 times a
    #   second, against a whole rendered frame of 1.20 ms GPU. A counter that
    #   keeps climbing while nothing moves is worse than no counter: it reports
    #   work on a scene that is doing none, and nobody files that as a bug.
    Wait-Frame
    $idleA = Invoke-RtIpc rayfusion.scene_as @{}
    Wait-Frame
    $idleB = Invoke-RtIpc rayfusion.scene_as @{}
    Check 'A parked timeline costs no refit' ([int64]$idleB.skin_refits -eq [int64]$idleA.skin_refits) `
        "skin_refits climbed while the playhead was parked ($($idleA.skin_refits) -> $($idleB.skin_refits)): the gate is firing on something other than deformation."
}
finally {
    Disconnect-RtIpc -ErrorAction SilentlyContinue
}

if ($failures -gt 0) { Write-Host "$failures check(s) failed." -ForegroundColor Red; exit 1 }
Write-Host 'All checks passed.' -ForegroundColor Green
