# Towering cumulus: cylindrical silhouette correction

The user verified scene depth, then reported rocket-like clouds in the current
`towering_cumulus` scene. The live IPC screenshot and cloud settings are saved as
`cloud_tower_live_2026_10_11.jpg` and `.json`. No scene settings were changed.
Viewport capture was temporarily enabled and disabled after the screenshot.

The old storm envelope had a narrow constant-width stem and a small pointed crown.
Independent noise frequencies and vertical aspect rules at region boundaries sliced
that envelope into visibly different cylinders. Increasing fine erosion could not
remove this underlying silhouette.

Changes in the existing shared density implementation:

- Four broad, overlapping ellipsoidal lobes replace the stem/crown radius profile.
  Existing 3D noise displaces their boundaries; no additional texture reads or
  textures are introduced. This does add arithmetic inside the existing nine-cell
  neighbourhood, so GPU cost still needs measurement.
- Linked regions interpolate coverage, extinction, erosion and noise frequencies
  between region midpoints. Convective shape transitions use group altitude;
  the vertical aspect no longer resets with each region's role.
- Tracking bounds use the group's maximum coverage and extinction, preserving the
  upper bound despite interpolation. Disabled/zero-extinction regions stay disabled.
  Region-specific albedo, phase and emission remain region-specific; full optical
  interpolation is not part of this correction.
- Preset footprint increases from 6 km to 14 km; tower noise period is 4.2 km.
  Extinctions decrease to .012/.008/.004 per metre to reduce opaque uniform shells.
  The preset core remains shared by UI, scripting and IPC; there are no new fields.

15 non-build checks pass, including the existing lookup budget, linked tracking
bound wiring, and conservative support bounds. The worst-case lobe support fits
inside the enumerated neighbour region and below the per-cell top. These checks
do not establish compilation, realism or runtime performance.

## User build and visual acceptance

Build C++ and all shaders with the normal workflow, then reapply
`towering_cumulus` to receive its new footprint/extinction defaults. Merely opening
the saved scene retains old preset values. The captured scene also has layer-0
coverage .23 rather than the preset's .45; reapplying replaces those edits.

Compare from the captured camera, from a distant side view, and inside the cloud.
Look for connected broad masses, irregular rounded crowns, exposed sidewall
billows, and no horizontal shape restart at 2400/9500 metres. Check both RT modes
and RayFusion, including mountain occlusion. Try `storm` separately. Adjusting
mesh/march steps may change integration artifacts, but does not validate the
silhouette correction. Visual acceptance is pending the user's next build.
