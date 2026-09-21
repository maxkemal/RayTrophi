<#
.SYNOPSIS
  Measure the RayFusion per-pixel specular reflection pass and the emissive
  triangle NEE table on the running application.
.DESCRIPTION
  Reads the acceptance instruments rather than looking at the picture, and
  separates the THREE ways this pass can leave the image unchanged -- because
  all three look identical on screen and only the counters tell them apart:

    1. gated_pixels == 0  -> no pixel asked for a reflection (weight never
                             written: wrong viewport/lighting, or no prefiltered
                             environment). `reason` says which.
    2. shaded_hits == 0   -> every ray was rejected. The image is byte-identical
                             to reflections being off. `gated_pixels` can NEVER
                             report this: it counts what WANTED a reflection.
    3. sky_misses == rays -> every ray reached the sky, so the subtracted and
                             added terms cancel by design. Not a fault: there is
                             nothing in the scene to reflect.
.NOTES
  The user builds and starts the application; this script neither compiles nor
  launches it. It changes a session-local development toggle and restores it.

  *** A FRAME IS PRODUCED BETWEEN WRITING AND MEASURING, deliberately. The
  counters are zeroed in the frame command buffer, filled by the GPU and read by
  the CPU on the NEXT frame (`counters_lag_one_frame`). Measuring immediately
  after the write reports the PREVIOUS batch -- which reads exactly like the
  pass doing nothing.
#>
[CmdletBinding()]
param(
    [ValidateSet(1, 2, 4)][int]$Samples = 1,
    [double]$RoughnessGate = 0.30,
    [double]$WeightGate = 0.01
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'RtIpc.psm1') -Force

$failures = 0
function Check([string]$Name, [bool]$Ok) {
    if ($Ok) { Write-Host "[PASS] $Name" }
    else { Write-Host "[FAIL] $Name" -ForegroundColor Red; $script:failures++ }
}
function Note([string]$Text) { Write-Host "       $Text" -ForegroundColor DarkGray }

# One frame, so the counters describe work that actually ran.
function Step-Frame {
    try { Invoke-RtIpc viewport.capture @{} | Out-Null } catch { Start-Sleep -Milliseconds 250 }
}

$restore = $null
try {
    Connect-RtIpc -TimeoutMs 3000

    $before = Invoke-RtIpc rayfusion.reflections @{}
    $restore = $before
    Check 'Reflection surface answers over IPC' ($null -ne $before)
    if (-not $before.supported) {
        Write-Host "[SKIP] hardware ray query unavailable: $($before.reason)" -ForegroundColor Yellow
        return
    }

    Invoke-RtIpc rayfusion.set_reflections @{
        enabled        = $true
        samples        = $Samples
        roughness_gate = $RoughnessGate
        weight_gate    = $WeightGate
    } | Out-Null

    Step-Frame
    Step-Frame   # second frame: the first one's counters are still the old batch
    $r = Invoke-RtIpc rayfusion.reflections @{}

    Write-Host ''
    Write-Host "reflection: ready=$($r.ready) $($r.width)x$($r.height)"
    Write-Host ("  gated_pixels {0}  rays {1}  shaded_hits {2}  sky_misses {3}" -f `
        $r.gated_pixels, $r.rays, $r.shaded_hits, $r.sky_misses)
    if ($r.reason) { Note $r.reason }

    Check 'Pass recorded a dispatch' ($r.ready -eq $true)
    Check 'Some pixel asked for a reflection' ($r.gated_pixels -gt 0)
    if ($r.gated_pixels -eq 0) {
        Note 'No weight reached the G-buffer. Needs material viewport + scene'
        Note 'lighting + a prefiltered environment -- the SAME gate the fragment'
        Note 'shader uses to fill the weight.'
    }

    # *** The acceptance number. Reported as a finding, not a failure: a scene
    #   with nothing reflective nearby legitimately shades nothing.
    if ($r.rays -gt 0) {
        if ($r.shaded_hits -eq 0) {
            Write-Host '[FINDING] rays > 0 but shaded_hits == 0: the image is IDENTICAL to' -ForegroundColor Yellow
            Write-Host '          reflections being off. Every ray was rejected.' -ForegroundColor Yellow
        } elseif ($r.sky_misses -eq $r.rays) {
            Note 'Every ray reached the sky: image unchanged by design, nothing to reflect.'
        } else {
            $shadedShare = [math]::Round(100.0 * $r.shaded_hits / $r.rays, 1)
            Note "shaded $shadedShare% of rays -- this is the share that CHANGED pixels."
        }
    }

    # ── Emissive triangle NEE table ──────────────────────────────────────────
    $p = Invoke-RtIpc rayfusion.probe_field @{}
    Write-Host ''
    Write-Host ("emissive NEE: {0} triangles, area {1}" -f $p.emissive_triangles, $p.emissive_area)
    Write-Host ("  excluded by: cap {0} tri | welded mesh {1} mesh | transparent {2} material" -f `
        $p.emissive_dropped, $p.emissive_skipped_indexed, $p.emissive_rejected_transparent)

    if ($p.emissive_rejected_transparent -gt 0) {
        Note '*** EXPECTED, and the surprise worth knowing: lamp shades are usually'
        Note 'TRANSPARENT and fall outside the bounce subset -- so the most likely'
        Note 'emissive object in a scene is exactly the one contributing nothing.'
        Note 'Put an OPAQUE emissive surface in the scene to exercise this path.'
    }
    if ($p.emissive_skipped_indexed -gt 0) {
        Note 'A welded (indexed) mesh keeps its index buffer on the GPU only, so its'
        Note 'triangles cannot be resolved on the CPU. Those materials are excluded'
        Note 'WHOLE -- partial representation would break single counting.'
    }
}
finally {
    if ($null -ne $restore -and $restore.supported) {
        try {
            Invoke-RtIpc rayfusion.set_reflections @{
                enabled        = [bool]$restore.enabled
                samples        = [int]$restore.samples
                roughness_gate = [double]$restore.roughness_gate
                weight_gate    = [double]$restore.weight_gate
                max_distance   = [double]$restore.max_distance
            } | Out-Null
        } catch { Write-Host "[WARN] could not restore reflection settings: $_" -ForegroundColor Yellow }
    }
    if ($failures -gt 0) { Write-Host "$failures check(s) failed" -ForegroundColor Red }
    else { Write-Host 'checks passed' -ForegroundColor Green }
}
