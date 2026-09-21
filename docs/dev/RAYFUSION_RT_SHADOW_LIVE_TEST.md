# RT shadow producer live test — 2026-09-08

Tested the user-built application and already open forest scene through local
IPC. No compilation, restart, or scene save was performed. No RT shadow control
exists in the current UI source; the switch is exposed through Python and IPC.

- Material Preview, Vulkan, 1680 x 945; hardware ray query supported.
- Disabled: enabled=false, ready=false, rays=0, reason="disabled".
- Enabled after a viewport frame: enabled=true, ready=true, rays=1587600,
  reason="". This count is a pixel/ray upper bound, not measured traversal count.
- Repeated off/off/on/on/off/on/off with the same camera. No reported device
  loss, NaN pixels, or black pixels. Small image changes occur even in off/off
  and on/on controls; strict pixel identity in the forest was not established.
  Mean luminance across this sequence was 0.57456–0.57467. JPEG comparison cannot
  establish raw framebuffer equality.
- Temporarily aimed the camera upward and repeated off/on/off. All three probe
  histograms and luminance statistics matched exactly (mean 0.3800993).
- Camera target, capture setting, and shadow switch restored after testing.

The first batch used viewport.render_frames; source inspection shows this calls
the progressive renderer rather than explicitly forcing a raster viewport frame.
Its timing values are excluded. The repeat batch instead invalidated the camera
and allowed natural display-loop frames to arrive. Capture makes presentation
synchronous, so these frame timings are not an interactive performance benchmark.

Representative late forest frame_ms samples were 322.11 off, 322.48 on, 326.83
off. Sky samples were 50.52 off, 48.97 on, 50.44 off. These are whole-frame
telemetry snapshots with too few samples to isolate shadow cost. They do NOT
verify the previous approximately 5 ms prediction or prove sky traversal count.

No RT mask readback, GPU timestamp measurement of this pass, validation-layer
log verification, or window resize test was performed. Visible shadows still
come from the cascade atlas; this test validates the producer switch/status and
basic runtime behavior, not RT shadow quality or completion of step 2b.

Raw evidence: `tmp/rt_shadow_live/results.json`, `repeat.json`, `sky.json`,
`camera.json`, and the adjacent captured JPEGs.
