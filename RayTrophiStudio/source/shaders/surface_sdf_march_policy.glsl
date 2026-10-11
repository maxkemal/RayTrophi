#ifndef RT_SURFACE_SDF_MARCH_POLICY
#define RT_SURFACE_SDF_MARCH_POLICY

// Grow the sample budget before coarsening a resolved surface traversal.
// The hard bound is only a pathological-volume escape hatch.
float surfaceSdfMarchStep(float span, float voxel, float authoredStep,
                          int authoredBudget, out int steps) {
    float fineStep = clamp(authoredStep, max(voxel * 0.1, 1e-6),
                           max(voxel * 0.45, 1e-6));
    float required = ceil(max(span, 0.0) / fineStep);
    int budget = max(clamp(authoredBudget, 1, 16384),
                     int(min(required, 16384.0)));
    float stepSize = max(fineStep, span / float(budget));
    steps = min(int(min(ceil(span / stepSize), 16384.0)) + 1, budget + 1);
    return stepSize;
}
#endif
