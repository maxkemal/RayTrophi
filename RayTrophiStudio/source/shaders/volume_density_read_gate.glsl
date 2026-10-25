// Optical reads are scaled to zero before any graph is evaluated. Geometry
// callers use applyOpticalControls=false and MUST still read their SDF proxy.
if (applyOpticalControls && vol.density_multiplier == 0.0) return 0.0;
