# Pressure-sweep scaling probe for the Vulkan gas solver.
#
# gpu_pressure_ms is one of the few rows that still synchronizes, so it stays
# comparable even though the other stage rows now measure enqueue time only.
# Fit ms(N) = a*N + b: b is the fixed cost that lowering the sweep count can
# never buy back, and it is the number that decides whether MGPCG is worth it.
param(
    [string]$Domain = 'Nuclear Gas',
    [int[]]$Sweeps = @(10, 20, 40, 80),
    [int]$Frames = 30
)

Import-Module 'E:\RayTrophi_projesi\raytracing_Proje_Moduler\scripts\ipc\RtIpc.psm1' -Force

$rows = @()
foreach ($n in $Sweeps) {
    Invoke-RtIpc gas.set_settings @{ domain = $Domain; pressure_iterations = $n } | Out-Null
    Invoke-RtIpc gas.reset @{} | Out-Null
    # The write above must not be read in the same batch: step the frames
    # first, then read the counters that the stepping produced.
    for ($f = 1; $f -le $Frames; $f++) {
        Invoke-RtIpc timeline.set_frame @{ frame = $f } | Out-Null
    }
    $st  = Invoke-RtIpc gas.step_stats @{ domain = $Domain }
    $pl  = Invoke-RtIpc gas.measure_plume @{ domain = $Domain }
    $chk = Invoke-RtIpc gas.get_settings @{ domain = $Domain }
    $rows += [pscustomobject]@{
        N           = $n
        applied     = $chk.pressure_iterations
        measured    = $st.measured
        pressure_ms = $st.gpu_pressure_ms
        total_ms    = $st.total_ms
        burning     = $st.burning_cells
        cells       = $pl.active_cells
        top         = $pl.top_above_floor
        fill        = $pl.fill_fraction
        peakT       = $pl.peak_temperature
        meanT       = $pl.mean_temperature
    }
    Write-Host ("N={0} applied={1} pressure_ms={2} total_ms={3} cells={4} top={5} fill={6} meanT={7}" -f `
        $rows[-1].N, $rows[-1].applied, $rows[-1].pressure_ms, $rows[-1].total_ms, `
        $rows[-1].cells, $rows[-1].top, $rows[-1].fill, $rows[-1].meanT)
}
$rows | Format-Table -AutoSize | Out-String | Write-Host
$rows | ConvertTo-Json -Depth 3
