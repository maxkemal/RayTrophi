# Player scene postprocess

Player publishes the loaded project's `color_processor` through Studio's shared
`rtpost::tick` / `syncDisplay` service. The existing raster post pass reads
`g_display_post`: tone mapping, exposure, gamma, saturation, temperature, vignette
and physical camera exposure now use the saved scene settings on the first frame
and subsequent updates. There is no additional GPU pass, CPU color-processing
loop, full-frame copy or shader variant. This is an integration fix, not a new
postprocessing implementation.

Existing Python `rt.post` operations and IPC `post.get`, `post.set_exposure`,
`post.set_gamma`, `post.set_saturation`, `post.set_color_temperature`,
`post.set_tone_mapping`, `post.set_vignette_enabled`,
`post.set_vignette_strength`, `post.get_exposure`, `post.configure_exposure` and
`post.reset_exposure` use the same shared services and validation. Player allows
these IPC methods under the existing capability policy. Python remains a Studio
authoring surface; no Python interpreter is added to Player.

Raster retains Studio raster's existing exposure limitation: it does not produce
the Rendered HDR histogram. Auto-histogram mode therefore has no new raster meter;
do not treat an accepted mode setting as evidence of active metering. An enabled
meter alone does not force an otherwise settled Player to submit frames. Stylize
and other effects outside the existing raster pass are not added by this fix.

## Build and acceptance handoff

Source verification: IPC capability/descriptor audit PASS; Player transitive
include audit PASS (666/666 clean); focused whitespace check PASS. Existing
`scripts/check_post_v2.mjs` passes its color math/adaptation checks but fails at
line 59 because it expects `launchOptixDisplayPost(` in the unchanged
`OptixWrapper.cpp`. That unrelated integration guard was not modified.
No C++ build or live visual acceptance was performed.

User: incremental Player C++ build; no shader compilation is needed.
The agent must not build C++ or launch the application.

Other Player test agent: use the external `scripts/test/rt_ipc.py` client against
the Player pipe after the user builds and opens it:

1. Save distinct tone map / gamma / exposure / vignette settings in Studio, then
   open the project in Player. Compare `post.get` and the raster display fields
   in `viewport.frame_timings` with Studio; capture the first completed frame.
2. On a stationary camera, change tone map and exposure through IPC. Confirm new
   completed pixels, not only successful responses; restore the original values.
3. Check physical mode uses the saved camera exposure. Manual mode must report
   camera exposure 1. Reject invalid tone names / gamma using shared errors.
4. Reopen a second project with different settings; ensure the first frame does
   not retain the old project's grading. Check a settled static scene still
   stops submitting frames, including auto-histogram mode without a raster meter.
5. Compare normal async timing with capture disabled at the same render size.
   No performance claim is made until measured; the structural change adds only
   shared parameter synchronization and existing post-pass work.
