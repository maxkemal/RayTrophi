# Run from the Player package folder. Generates a separate scene beside workshop.rtp.
[CmdletBinding()]
param(
    [string]$PlayerFolder = '.',
    [ValidateSet('Prepare', 'Baseline', 'Reduced')]
    [string]$Mode = 'Prepare'
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $PlayerFolder).Path
$player = Join-Path $root 'RayTrophiPlayer.exe'
$source = Join-Path $root 'scenes/pastel-paint-workshop/workshop.rtp'
$reduced = Join-Path (Split-Path -Parent $source) 'workshop-gpu-reduced.rtp'

foreach ($required in @($player, $source, "$source.bin")) {
    if (!(Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing package file: $required"
    }
}

if ($Mode -eq 'Prepare') {
    $scene = Get-Content -LiteralPath $source -Raw -Encoding UTF8 | ConvertFrom-Json
    $design = $scene.game_design
    $hiddenObjects = @($design.hidden_objects.objects | Where-Object { $_.object })
    if (!$design.balloon_demo -or !$hiddenObjects.Count) {
        throw 'This workshop package has no saved balloon demo or hidden-object design.'
    }

    $names = @($hiddenObjects | ForEach-Object { $_.object })
    $scene.world.mode = 0
    $scene.world.hdri_path = ''
    foreach ($entry in $hiddenObjects) {
        $entry.visibility = 'hidden'
        $entry.blocks_balloon = $false
        $entry.blocks_player = $false
        $entry.paint_collider_mode = 'none'
        $entry.randomize_on_reset = $false
    }
    # The hidden-object loader requires a wet surface for each managed mesh.
    # Keep the smallest valid surface so the hidden visibility rule still applies.
    foreach ($surface in $design.wet_paint.surfaces) {
        if ($surface.object -in $names) {
            $surface.resolution = 32
        }
    }
    foreach ($system in $scene.particle_simulation.systems) {
        $system.colliders = @(
            $system.colliders | Where-Object { $_.source_name -notin $names }
        )
    }

    $suffixes = @('.bin', '.shared', '.aux.json') | Where-Object {
        Test-Path -LiteralPath "$source$_" -PathType Leaf
    }
    foreach ($target in @($reduced) + @($suffixes | ForEach-Object { "$reduced$_" })) {
        if (Test-Path -LiteralPath $target) {
            throw "Comparison file already exists: $target"
        }
    }
    foreach ($suffix in $suffixes) {
        Copy-Item -LiteralPath "$source$suffix" -Destination "$reduced$suffix"
    }
    $json = $scene | ConvertTo-Json -Depth 100 -Compress
    [IO.File]::WriteAllText($reduced, $json, [Text.UTF8Encoding]::new($false))
    Write-Host "Prepared: $reduced"
    Write-Host "Disabled hidden targets: $($names -join ', ')"
    return
}

$project = if ($Mode -eq 'Baseline') { $source } else { $reduced }
if (!(Test-Path -LiteralPath $project -PathType Leaf)) {
    throw 'Run the script with -Mode Prepare first.'
}
Write-Host 'In Player, use F2 to confirm render resolution is 640x480 and F1 to show the HUD.'
Start-Process -FilePath $player -WorkingDirectory $root -WindowStyle Normal -Wait -ArgumentList @(
    '--project', ('"{0}"' -f $project), '--width', '640', '--height', '480',
    '--pacing', '0', '--continuous'
)
