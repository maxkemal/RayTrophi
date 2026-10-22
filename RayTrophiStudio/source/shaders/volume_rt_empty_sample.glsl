// Optical density is already rejected by the canonical sampler. With no
// companion or graph, this sample cannot scatter, absorb or emit. In particular
// it must not spend eight more NanoVDB fetches on a temperature that is unused.
if (!hasCompanion && volumeProgram == MATPROG_NONE && density <= 0.0) {
    step++;
    t = tNear + (float(step) + rayJitter) * baseStep;
    continue;
}
