# Original mixed-size scene: MPM / DEM pilot

External IPC against the user's running build, 2026-10-22. No application launch,
compilation or source modification. Initial configuration and samples are in
[dem_mpm_original_scene_20261022.json](dem_mpm_original_scene_20261022.json).
Forced autosave before editing preserves configuration, not live checkpoints.

Original Physics Domain 1: bounds +/-2.5 m, effective voxel .033557 m, capacity
100,000, Gravel point source at (0,.85,0), radius .63 m, velocity (0,-1.1,0).
Original DEM size recipe: .011-.036 m bimodal, large fraction .01, seed 1360.

Pilot uses the same domain geometry, boundaries and source footprint. Fine Sand
MPM Test derives from Sand with granular_transport=mpm, used by domain default
and new Fine Sand MPM Source. Original Point Flow Source remains Gravel DEM,
physical radius fixed at .036 m. Size-based automatic solver reassignment was
not introduced. Transport follows substance, not visual mesh dimensions.

Pilot birth caps: 19,800 MPM parcels and 200 DEM grains. These are computational
counts, not a physical 99:1 grain-count or mass-equivalent conversion of the old
DEM size mixture. MPM parcel mass follows voxel/PPC and material density; the
same source count is not equivalent physical sand volume. Full-scale matched-
mass comparison and original all-DEM control have not been performed.

72 steps, dt=1/60 s, total 1.2 s. All complete; no held step. Final owners remain
19,800 MPM + 200 DEM, zero fluid. Shared geometry/contact clock active, sampled
86 transport substeps per frame. Reciprocal contacts measured throughout; peak
sampled contact momentum residual 2.701e-7 N s. Final frame reports 124,062
contact events across substeps (not unique pairs or the whole-run event total).
Working set reported 4,355,420,904 bytes (~4.06 GiB), excluding total application
usage. Detailed timing is in the JSON summary.

The fine voxel across the large domain is a substantial memory cost even for
this 20,000-carrier pilot. No voxel coarsening was silently applied. These results
establish mixed transport/contact execution, not segregation, correct sinking,
long-term repose or convergence. Those remain physical acceptance work.

Left paused at the measured end state for inspection. Sources retain the pilot
birth caps. Reset and Play rerun the fixture. No explicit user project save was
forced after configuration changes; normal Save retains this version.
