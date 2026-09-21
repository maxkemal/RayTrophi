<#
    Probe-MeshEdit.ps1

    Drives the polygon edit operators that were implemented on SceneUI but had
    no IPC address until now, plus the bounding box that scene.object_info
    gained. Every step MEASURES rather than eyeballs: element counts come back
    from the operator itself and the assembly check reads world bounds.

    Usage:
        .\scripts\ipc\Probe-MeshEdit.ps1
#>

Import-Module "$PSScriptRoot\RtIpc.psm1" -Force

$ErrorActionPreference = 'Stop'
$failures = @()

function Check($label, $condition, $detail) {
    if ($condition) {
        Write-Host "  PASS  $label" -ForegroundColor Green
    } else {
        Write-Host "  FAIL  $label -- $detail" -ForegroundColor Red
        $script:failures += $label
    }
}

Write-Host "`n=== 1. scene.object_info bounds ===" -ForegroundColor Cyan
# ★ A FRESH name every run. Reusing a name this probe deleted on its previous
# run makes scene.add_primitive hand back a name no object actually answers to,
# and every later call fails with "object not found" -- the probe would be
# reporting its own cleanup as a product bug.
$probeName = 'ProbeEditCube_' + (Get-Date -Format 'HHmmssfff')
$box = Invoke-RtIpc scene.add_primitive @{ type = 'cube'; name = $probeName; size = 1.0 }
Invoke-RtIpc scene.set_transform @{ name = $box; translation = @(0, 0.5, 0); scale = @(0.4, 0.1, 0.4) } | Out-Null
$info = Invoke-RtIpc scene.object_info @{ name = $box }

Check "has_bounds is true" ($info.has_bounds -eq $true) "object_info returned no bounds"
# A `size` of 1.0 is a 1-unit cube (edge length), NOT a 1-unit radius, so a
# 0.4 scale gives 0.4 m. The first version of this check expected 0.8 and
# failed against correct geometry -- a wrong test is a false alarm, which is
# worse than no test.
Check "world_size tracks the transform scale" (
    ([math]::Abs($info.world_size[0] - 0.4) -lt 0.02) -and
    ([math]::Abs($info.world_size[1] - 0.1) -lt 0.02)
) ("world_size = " + ($info.world_size -join ','))
Check "world_center sits at the authored position" (
    [math]::Abs($info.world_center[1] - 0.5) -lt 0.02
) ("world_center = " + ($info.world_center -join ','))

Write-Host "`n=== 2. mesh.edit state and selection ===" -ForegroundColor Cyan
$state = Invoke-RtIpc mesh.edit.begin @{ object = $box }
Write-Host ("  faces={0} edges={1} vertices={2} half_edge_valid={3}" -f `
    $state.faces, $state.edges, $state.vertices, $state.half_edge_valid)

# ★ half_edge_valid false means the operators fall back to the triangle-soup
# path where several of them refuse. Without this the later failures would
# look like broken operators rather than a mesh whose topology never built.
Check "half-edge topology built" ($state.half_edge_valid -eq $true) `
    "operators will fall back to the legacy triangle path"
Check "cube has 6 polygon faces" ($state.faces -eq 6) ("faces = " + $state.faces)

$sel = Invoke-RtIpc mesh.edit.select_by_normal @{ object = $box; direction = @(0, 1, 0); max_angle = 20 }
Check "select_by_normal found exactly the top face" ($sel.matched -eq 1) `
    ("matched = " + $sel.matched)
Check "selection is reflected in the state" ($sel.selected_faces -eq 1) `
    ("selected_faces = " + $sel.selected_faces)

Write-Host "`n=== 3. Operators ===" -ForegroundColor Cyan
# ★ Judge extrude by the BOX it produced and by Euler's characteristic, not by
# the polygon-face count: the editable cache regroups polygons after every
# edit, so `faces` can stay 6 on a perfectly good extrude. Measuring the wrong
# number reported a working operator as a silent no-op.
# ★ MEASURED: `distance` is in OBJECT-LOCAL units, not metres. This cube is
# scaled 0.1 in Y, so a request of 0.15 moves the face 0.015 m in world space.
# The first version of this check compared against the world figure and
# reported a correct operator as broken -- the unit, not the operator, was the
# thing that had never been written down.
$scaleY = 0.1
$heightBefore = $info.world_size[1]
$askedLocal = 0.15
$after = Invoke-RtIpc mesh.extrude @{ object = $box; distance = $askedLocal }
$grown = Invoke-RtIpc scene.object_info @{ name = $box }
$grewBy = $grown.world_size[1] - $heightBefore
Check "extrude grew the object by distance x scale" (
    [math]::Abs($grewBy - ($askedLocal * $scaleY)) -lt 0.002
) ("grew {0:N4} m, expected {1:N4} (local {2} x scale {3})" -f $grewBy, ($askedLocal * $scaleY), $askedLocal, $scaleY)

# ★★ The check that actually matters, and the one nobody would notice by
# looking: an extruded box must stay a CLOSED surface. V - E + F = 2 for any
# closed mesh; anything else means the operator left a hole, and the render
# hides it whenever the missing face points away from the camera.
$chi = $after.vertices - $after.edges + $after.faces
Check "extrude left a closed surface (Euler chi = 2)" ($chi -eq 2) `
    ("V={0} E={1} F={2} -> chi={3}; a hole is open" -f $after.vertices, $after.edges, $after.faces, $chi)

# Inset needs a face selection again: the extrude invalidated the old ids.
$sel2 = Invoke-RtIpc mesh.edit.select_by_normal @{ object = $box; direction = @(0, 1, 0); max_angle = 20 }
if ($sel2.selected_faces -gt 0) {
    $inset = Invoke-RtIpc mesh.inset @{ object = $box; amount = 0.03 }
    Check "inset added faces" ($inset.faces -gt $after.faces) `
        ("faces went " + $after.faces + " -> " + $inset.faces)
} else {
    Check "top face reselectable after extrude" $false "select_by_normal matched nothing"
}

# ★ Bevel was registered as "Planned" in the tool catalogue while
# SceneUI::bevelSelectedEdges was fully implemented. This step is the one that
# decides which of the two was telling the truth.
$edgeSel = Invoke-RtIpc mesh.edit.select @{ object = $box; domain = 'edge'; all = $true }
$beveled = Invoke-RtIpc mesh.bevel @{ object = $box; width = 0.01; segments = 2; round = $true }
Check "bevel changed the topology" ($beveled.triangles -gt $after.triangles) `
    ("triangles stayed at " + $beveled.triangles + " -- bevel may be a no-op despite reporting ok")
$bchi = $beveled.vertices - $beveled.edges + $beveled.faces
Check "bevel left a closed surface (Euler chi = 2)" ($bchi -eq 2) `
    ("V={0} E={1} F={2} -> chi={3}" -f $beveled.vertices, $beveled.edges, $beveled.faces, $bchi)

Write-Host "`n=== 3b. Bevel refuses what it cannot build ===" -ForegroundColor Cyan
# ★★ MEASURED: a cylinder's caps are a fan around ONE centre vertex, so that
# vertex has valence 32 and its bevel corner patch cannot be triangulated. It
# used to publish anyway: Euler chi came out -56 with 10 degenerate triangles
# and nothing reported it. Bevel now refuses and leaves the mesh untouched.
# This check fails if that silent-damage path ever comes back.
$cylName = 'ProbeBevelCyl_' + (Get-Date -Format 'HHmmssfff')
$cyl = Invoke-RtIpc scene.add_primitive @{ type = 'cylinder'; name = $cylName; size = 1.0 }
Invoke-RtIpc scene.set_transform @{ name = $cyl; translation = @(0, 4, 0) } | Out-Null
$cylBefore = Invoke-RtIpc mesh.edit.begin @{ object = $cyl }
Invoke-RtIpc mesh.edit.select @{ object = $cyl; domain = 'edge'; all = $true } | Out-Null

$refused = $false
try {
    Invoke-RtIpc mesh.bevel @{ object = $cyl; width = 0.05; segments = 2; round = $true } | Out-Null
} catch {
    $refused = $true
}
$cylAfter = Invoke-RtIpc scene.object_info @{ name = $cyl }
Check "bevel refuses a high-valence (fan-capped) corner" $refused `
    "bevel reported success on a cylinder pole -- the silent-damage path is back"
Check "refused bevel left the mesh untouched" ($cylAfter.triangles -eq $cylBefore.triangles) `
    ("triangles went {0} -> {1}; a refused operation must publish nothing" -f $cylBefore.triangles, $cylAfter.triangles)
Invoke-RtIpc scene.delete @{ name = $cyl } | Out-Null

Write-Host "`n=== 4. Catalogue honesty ===" -ForegroundColor Cyan
$tools = (Invoke-RtIpc mesh.tools.list @{ workspace = 'edit' }).tools
$ids = $tools | ForEach-Object { $_.id }
Check "edge bevel is listed as available" ($ids -contains 'edit.edge_bevel') `
    ("edit tools = " + ($ids -join ','))
Check "vertex operators are listed" (
    ($ids -contains 'edit.weld_vertices') -and ($ids -contains 'edit.merge_vertices')
) ("edit tools = " + ($ids -join ','))

Write-Host "`n=== Cleanup ===" -ForegroundColor Cyan
Invoke-RtIpc scene.delete @{ name = $box } | Out-Null

Write-Host ""
if ($failures.Count -eq 0) {
    Write-Host "ALL CHECKS PASSED" -ForegroundColor Green
    exit 0
}
Write-Host ("FAILED: " + ($failures -join '; ')) -ForegroundColor Red
exit 1
