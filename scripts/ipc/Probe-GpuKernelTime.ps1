# Per-kernel GPU time for one simulation step, from timestamp queries.
#
# ★★★ WHY NOT JUST READ gas.step_stats. Since the gas chain became
# device-resident the solver stages no longer call synchronize(), so the CPU
# timers around them measure ENQUEUE time. The work is not lost - it is billed
# to whichever later stage does synchronize. The rows therefore still look like
# a ranking while no longer being one. These numbers are written by the device
# around each dispatch and stay true wherever the submission boundary falls.
#
# Protocol (the ordering matters):
#   enable -> step the warm-up frames -> reset -> step ONE frame -> read.
# Reading in the same batch as the stepping that feeds it returns the previous
# interval, which is the counter trap this repo keeps re-learning.
param(
    [string]$Domain = 'Nuclear Gas',
    [int]$WarmupFrames = 30,
    [int]$MeasureFrames = 1
)

Import-Module (Join-Path $PSScriptRoot 'RtIpc.psm1') -Force

$probe = Invoke-RtIpc perf.gpu_kernel_timings @{ reset = $false }
if (-not $probe.supported) {
    Write-Host "GPU timestamps are NOT supported on this compute queue."
    Write-Host "That is ABSENCE, not a measured zero - do not read the table below as work that did not happen."
    return
}

Invoke-RtIpc perf.set_gpu_kernel_timing @{ enabled = $true } | Out-Null
Invoke-RtIpc gas.reset @{} | Out-Null
for ($f = 1; $f -le $WarmupFrames; $f++) {
    Invoke-RtIpc timeline.set_frame @{ frame = $f } | Out-Null
}

# Discard everything the warm-up accumulated, then measure a known frame count.
Invoke-RtIpc perf.gpu_kernel_timings @{ reset = $true } | Out-Null
for ($f = $WarmupFrames + 1; $f -le ($WarmupFrames + $MeasureFrames); $f++) {
    Invoke-RtIpc timeline.set_frame @{ frame = $f } | Out-Null
}

$r  = Invoke-RtIpc perf.gpu_kernel_timings @{ reset = $true }
$st = Invoke-RtIpc gas.step_stats @{ domain = $Domain }

Write-Host ("GPU total over {0} frame(s): {1:N2} ms   (step_stats total_ms = {2:N2})" -f `
    $MeasureFrames, $r.total_ms, $st.total_ms)
Write-Host ""
$r.kernels | ForEach-Object {
    [pscustomobject]@{
        kernel      = $_.kernel
        ms          = [math]::Round($_.ms, 3)
        calls       = $_.calls
        ms_per_call = [math]::Round($_.ms / [math]::Max(1, $_.calls), 4)
        pct         = if ($r.total_ms -gt 0) { [math]::Round(100.0 * $_.ms / $r.total_ms, 1) } else { 0 }
    }
} | Format-Table -AutoSize | Out-String | Write-Host

# ★ The gap is the point: GPU total well below step total_ms means the step is
#   NOT GPU-bound, and the remainder is host work, readback stalls or enqueue
#   overhead - none of which a faster kernel can buy back.
Write-Host ("Unaccounted by GPU kernels: {0:N2} ms" -f ($st.total_ms - $r.total_ms))

Invoke-RtIpc perf.set_gpu_kernel_timing @{ enabled = $false } | Out-Null
