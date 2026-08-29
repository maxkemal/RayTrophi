param(
    [int]$TimeoutMs = 15000
)

$ErrorActionPreference = 'Stop'
$modulePath = Join-Path $PSScriptRoot '..\ipc\RtIpc.psm1'
Import-Module $modulePath -Force

$hostName = $null
$skinHost = $null
$clearSkinHost = $null
$splineName = $null
$profileName = $null
$graphCreated = $false

function Assert-True([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw "ASSERT: $Message" }
}

try {
    Connect-RtIpc -TimeoutMs $TimeoutMs

    $selfTest = Invoke-RtIpc 'spline.animation.self_test'
    Assert-True ([bool]$selfTest.ok) "spline animation self-test failed"
    $sweepTest = Invoke-RtIpc 'mesh.profile.sweep.self_test'
    Assert-True ([bool]$sweepTest.ok) "profile sweep/UV self-test failed"
    $cacheSelfTest = Invoke-RtIpc 'geometry_cache.self_test'
    Assert-True ([bool]$cacheSelfTest.ok) "geometry cache core self-test failed"

    $types = @(Invoke-RtIpc 'nodes.types')
    foreach ($required in @('GeoV2.SplineObject', 'GeoV2.ResampleCurve',
                             'GeoV2.CurveTaper', 'GeoV2.CurveTwist',
                             'GeoV2.CurveWave', 'GeoV2.CurveToMesh', 'GeoV2.Output')) {
        Assert-True ([bool]($types | Where-Object type_id -eq $required)) "missing node type $required"
    }

    $hostName = Invoke-RtIpc 'scene.add_primitive' @{
        type = 'cube'; name = '__IPC_CurveHost'; size = 1.0
    }
    $splineName = Invoke-RtIpc 'spline.create' @{
        primitive = 'open_line'; name = '__IPC_CurvePath'; plane = 'xy'
    }
    $profileName = Invoke-RtIpc 'spline.create' @{
        primitive = 'rectangle'; name = '__IPC_SkinProfile'; plane = 'xy'
    }

    $baseline = Invoke-RtIpc 'spline.get' @{ name = $splineName }
    $startY = [double]$baseline.points[0].position[1]
    $startRadius = [double]$baseline.points[0].user_data[0]
    Invoke-RtIpc 'spline.keyframe.insert' @{
        name = $splineName; frame = 0; object_transform = $true; points = $true
    } | Out-Null

    $deformed = $baseline | ConvertTo-Json -Depth 30 | ConvertFrom-Json
    $deformed.points[0].position[1] = $startY + 4.0
    $deformed.points[0].user_data[0] = $startRadius + 2.0
    Invoke-RtIpc 'spline.set' @{ name = $splineName; spline = $deformed } | Out-Null
    Invoke-RtIpc 'spline.keyframe.insert' @{
        name = $splineName; frame = 10; object_transform = $true; points = $true
    } | Out-Null

    Invoke-RtIpc 'timeline.set_frame' @{ frame = 5 } | Out-Null
    Start-Sleep -Milliseconds 250
    $midpoint = Invoke-RtIpc 'spline.get' @{ name = $splineName }
    Assert-True ([math]::Abs([double]$midpoint.points[0].position[1] - ($startY + 2.0)) -lt 0.001) `
        'timeline did not interpolate spline point position'
    Assert-True ([math]::Abs([double]$midpoint.points[0].user_data[0] - ($startRadius + 1.0)) -lt 0.001) `
        'timeline did not interpolate curve radius'

    Invoke-RtIpc 'spline.insert_point' @{ name = $splineName; segment = 0; t = 0.5 } | Out-Null
    $topologyKeys = @(Invoke-RtIpc 'spline.keyframe.list' @{ name = $splineName })
    Assert-True ($topologyKeys.Count -eq 2) 'topology edit removed spline keys'
    Assert-True (-not ($topologyKeys | Where-Object point_count -ne 3)) `
        'inserted point was not propagated to every spline key'
    Invoke-RtIpc 'timeline.set_frame' @{ frame = 7 } | Out-Null
    Start-Sleep -Milliseconds 250
    $topologyFrame = Invoke-RtIpc 'spline.get' @{ name = $splineName }
    Assert-True ($topologyFrame.points.Count -eq 3) `
        'spline animation stopped after point insertion'

    Invoke-RtIpc 'nodes.create_graph' @{ graph_type = 'geometry'; graph_name = $hostName } | Out-Null
    $graphCreated = $true
    $source = Invoke-RtIpc 'nodes.add' @{
        graph_type = 'geometry'; graph_name = $hostName; type_id = 'GeoV2.SplineObject'
    }
    $resample = Invoke-RtIpc 'nodes.add' @{
        graph_type = 'geometry'; graph_name = $hostName; type_id = 'GeoV2.ResampleCurve'
    }
    $tube = Invoke-RtIpc 'nodes.add' @{
        graph_type = 'geometry'; graph_name = $hostName; type_id = 'GeoV2.CurveToMesh'
    }
    $output = Invoke-RtIpc 'nodes.add' @{
        graph_type = 'geometry'; graph_name = $hostName; type_id = 'GeoV2.Output'
    }
    Invoke-RtIpc 'nodes.set_property' @{
        graph_type = 'geometry'; graph_name = $hostName; node_id = $source;
        property = 'object'; value = $splineName
    } | Out-Null
    Invoke-RtIpc 'nodes.link' @{
        graph_type = 'geometry'; graph_name = $hostName; from_node = $source;
        from_output = 0; to_node = $resample; to_input = 0
    } | Out-Null
    Invoke-RtIpc 'nodes.link' @{
        graph_type = 'geometry'; graph_name = $hostName; from_node = $resample;
        from_output = 0; to_node = $tube; to_input = 0
    } | Out-Null
    Invoke-RtIpc 'nodes.link' @{
        graph_type = 'geometry'; graph_name = $hostName; from_node = $tube;
        from_output = 0; to_node = $output; to_input = 0
    } | Out-Null

    $apply = Invoke-RtIpc 'nodes.apply' @{ graph_type = 'geometry'; graph_name = $hostName }
    Assert-True ([bool]$apply.ok) 'curve graph apply failed'
    $mesh = Invoke-RtIpc 'scene.object_info' @{ name = $hostName }
    Assert-True ([int]$mesh.triangles -gt 0) 'Curve to Mesh produced no triangles'

    $skin = Invoke-RtIpc 'spline.skin.create' @{
        spline = $splineName; output = '__IPC_QuickSkin'; radius = 0.2;
        path_samples = 24; radial_segments = 8; cap_start = $true;
        cap_end = $true; use_point_radius = $true;
        taper_start = 1.0; taper_end = 0.35; taper_falloff = 1.5;
        twist_start_degrees = 0.0; twist_end_degrees = 180.0;
        wave_amplitude = 0.2; wave_cycles = 1.5; wave_phase_degrees = 20.0;
        wave_noise = 0.05; wave_seed = 42; wave_axis = 2;
        custom_profile = $profileName
    }
    $skinHost = [string]$skin.object_name
    Assert-True ([int]$skin.vertex_count -gt 0) 'one-click Spline Skin produced no vertices'
    Assert-True ([int]$skin.triangle_count -gt 0) 'one-click Spline Skin produced no triangles'
    Assert-True ([int]$skin.taper_node -gt 0) 'quick skin did not create Curve Taper'
    Assert-True ([int]$skin.twist_node -gt 0) 'quick skin did not create Curve Twist'
    Assert-True ([int]$skin.wave_node -gt 0) 'quick skin did not create Curve Wave'
    Assert-True ([int]$skin.profile_node -gt 0) 'custom profile source node was not created'
    $updatedSkin = Invoke-RtIpc 'spline.skin.create' @{
        spline = $splineName; output = '__IPC_ShouldNotCreateAnotherHost'; radius = 0.35;
        path_samples = 24; radial_segments = 8; cap_start = $true;
        cap_end = $true; use_point_radius = $true; taper_start = 0.8;
        taper_end = 0.25; twist_end_degrees = 270.0;
        wave_amplitude = 0.1; wave_cycles = 2.0; custom_profile = $profileName
    }
    Assert-True ([string]$updatedSkin.object_name -eq $skinHost) `
        'skin parameter update created a second preview host'
    $displayState = Invoke-RtIpc 'spline.get' @{ name = $splineName }
    Assert-True ([bool]$displayState.skin_display.enabled) `
        'spline payload did not persist enabled skin display state'
    Assert-True ([string]$displayState.skin_display.host -eq $skinHost) `
        'spline payload did not persist the linked preview host'
    Assert-True ([string]$displayState.skin_display.custom_profile -eq $profileName) `
        'spline payload did not persist the custom profile'
    $skinMesh = Invoke-RtIpc 'scene.object_info' @{ name = $skinHost }

    $cache = Invoke-RtIpc 'geometry_cache.bake' @{
        object_name = $skinHost; start_frame = 0; end_frame = 10; frame_step = 2
    }
    Assert-True ([bool]$cache.enabled) 'baked geometry cache is not enabled'
    Assert-True ([bool]$cache.topology_valid) 'baked geometry cache topology is invalid'
    Assert-True ([int]$cache.sample_count -eq 6) 'geometry cache stored the wrong sample count'
    Assert-True ([int]$cache.vertex_count -eq [int]$skinMesh.vertices) `
        'geometry cache vertex count does not match the flat mesh'
    $cacheStatus = Invoke-RtIpc 'geometry_cache.status' @{ object_name = $skinHost }
    Assert-True ([int64]$cacheStatus.memory_bytes -gt 0) 'geometry cache reports zero memory'
    Invoke-RtIpc 'geometry_cache.set_enabled' @{
        object_name = $skinHost; enabled = $false
    } | Out-Null
    Invoke-RtIpc 'geometry_cache.set_enabled' @{
        object_name = $skinHost; enabled = $true
    } | Out-Null
    Invoke-RtIpc 'geometry_cache.clear' @{ object_name = $skinHost } | Out-Null
    $finalized = Invoke-RtIpc 'spline.skin.finalize' @{ spline = $splineName }
    Assert-True ([string]$finalized.object_name -eq $skinHost) `
        'skin finalize replaced the preview host instead of detaching it'
    Assert-True ([int]$finalized.vertex_count -gt 0) 'finalized skin has no vertices'
    $finalizedState = Invoke-RtIpc 'spline.get' @{ name = $splineName }
    Assert-True (-not [bool]$finalizedState.skin_display.enabled) `
        'skin display remained enabled after finalize'

    $clearSkin = Invoke-RtIpc 'spline.skin.create' @{
        spline = $splineName; output = '__IPC_ClearSkin'; radius = 0.1;
        path_samples = 12; radial_segments = 6
    }
    $clearSkinHost = [string]$clearSkin.object_name
    Invoke-RtIpc 'spline.skin.clear' @{ spline = $splineName } | Out-Null
    $clearSkinHost = $null

    [pscustomobject]@{
        ok = $true
        self_test = $selfTest.details
        sweep_test = $sweepTest.details
        cache_self_test = $cacheSelfTest.details
        spline = $splineName
        midpoint_y = [double]$midpoint.points[0].position[1]
        midpoint_radius = [double]$midpoint.points[0].user_data[0]
        propagated_key_points = [int]$topologyFrame.points.Count
        host = $hostName
        vertices = [int]$mesh.vertices
        triangles = [int]$mesh.triangles
        skin_host = $skinHost
        skin_vertices = [int]$skinMesh.vertices
        skin_triangles = [int]$skinMesh.triangles
        cache_samples = [int]$cache.sample_count
        cache_memory_bytes = [int64]$cacheStatus.memory_bytes
    } | ConvertTo-Json -Depth 10
}
finally {
    if ($skinHost) {
        try { Invoke-RtIpc 'nodes.remove_graph' @{ graph_type = 'geometry'; graph_name = $skinHost } | Out-Null } catch {}
        try { Invoke-RtIpc 'scene.delete' @{ name = $skinHost } | Out-Null } catch {}
    }
    if ($clearSkinHost) {
        try { Invoke-RtIpc 'nodes.remove_graph' @{ graph_type = 'geometry'; graph_name = $clearSkinHost } | Out-Null } catch {}
        try { Invoke-RtIpc 'scene.delete' @{ name = $clearSkinHost } | Out-Null } catch {}
    }
    if ($graphCreated -and $hostName) {
        try { Invoke-RtIpc 'nodes.remove_graph' @{ graph_type = 'geometry'; graph_name = $hostName } | Out-Null } catch {}
    }
    if ($hostName) {
        try { Invoke-RtIpc 'scene.delete' @{ name = $hostName } | Out-Null } catch {}
    }
    if ($splineName) {
        try { Invoke-RtIpc 'scene.delete' @{ name = $splineName } | Out-Null } catch {}
    }
    if ($profileName) {
        try { Invoke-RtIpc 'scene.delete' @{ name = $profileName } | Out-Null } catch {}
    }
    Disconnect-RtIpc
}
