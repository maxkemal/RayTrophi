# Physics reference sweep for the Nuclear Gas domain.
#
# The recorded table in docs/dev/NEXT_BUILD_CHECKS.md was taken with this exact
# protocol: reset, then step the timeline one frame at a time, then measure at
# frames 40 / 80 / 120. Anything else is not comparable - a panel readout only
# describes whatever frame happened to be last, and the plume grows monotonically
# for most of the shot, so "the numbers look different" is meaningless without
# the frame number attached.
param(
    [string]$Domain = 'Nuclear Gas',
    [int[]]$Marks = @(40, 80, 120)
)

Import-Module (Join-Path $PSScriptRoot 'RtIpc.psm1') -Force

Invoke-RtIpc gas.reset @{} | Out-Null
$last = ($Marks | Measure-Object -Maximum).Maximum
$rows = @()
for ($f = 1; $f -le $last; $f++) {
    Invoke-RtIpc timeline.set_frame @{ frame = $f } | Out-Null
    if ($Marks -contains $f) {
        # Read AFTER the stepping that feeds it, never in the same batch.
        $p = Invoke-RtIpc gas.measure_plume @{ domain = $Domain }
        $s = Invoke-RtIpc gas.step_stats   @{ domain = $Domain }
        $rows += [pscustomobject]@{
            frame    = $f
            cells    = $p.active_cells
            fill     = [math]::Round($p.fill_fraction, 5)
            top      = [math]::Round($p.top_above_floor, 3)
            centroid = [math]::Round($p.centroid_above_floor, 3)
            peakT    = [math]::Round($p.peak_temperature, 4)
            meanT    = [math]::Round($p.mean_temperature, 5)
            burning  = $s.burning_cells
            maxSpeed = [math]::Round($s.max_speed, 4)
            pMeasured= $p.pressure_measured
            pMin     = [math]::Round($p.pressure_min, 4)
            pMax     = [math]::Round($p.pressure_max, 4)
            total_ms = [math]::Round($s.total_ms, 2)
        }
        Write-Host ("f{0}: cells={1} fill={2} top={3} peakT={4} meanT={5} burning={6}" -f `
            $f, $p.active_cells, $rows[-1].fill, $rows[-1].top, $rows[-1].peakT, $rows[-1].meanT, $s.burning_cells)
    }
}
$rows | Format-Table -AutoSize | Out-String | Write-Host
