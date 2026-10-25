void volumeRecordSdfSearch(bool arbiter, uint samples, uint queries, uint skipped) {
    if (!volumeInstrumentationEnabled()) return;
    if (arbiter) {
        atomicAdd(volumeInstrumentation.sdfArbiterSamples, samples);
        atomicAdd(volumeInstrumentation.sdfArbiterQueries, queries);
        atomicAdd(volumeInstrumentation.sdfArbiterSkipped, skipped);
    } else {
        atomicAdd(volumeInstrumentation.sdfMainSamples, samples);
        atomicAdd(volumeInstrumentation.sdfMainQueries, queries);
        atomicAdd(volumeInstrumentation.sdfMainSkipped, skipped);
    }
}
