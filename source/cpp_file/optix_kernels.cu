// ... SBT oluþturma kýsmý ...
OptixShaderBindingTable sbt;
// ... SBT'yi konfigüre etme

// Hitgroup'ta sonuçlarý depola
extern "C" __global__ void __closesthit__radiance() {
    const HitGroupData* hitData = reinterpret_cast<HitGroupData*>(optixGetSbtData());
    // ... Diðer hesaplamalar

    // Sonucu SBT'nin payload bölümüne yaz
    OptixHitResult* hitResult = reinterpret_cast<OptixHitResult*>(optixGetSbtDataPayload());
    hitResult->hasHit = true;
    // ... Diðer sonuçlarý ayarla
}

// Host'ta sonuçlarý oku
OptixHitResult* hostResult = new OptixHitResult[numRays];
cudaMemcpy(hostResult, sbt.payloadBase, numRays * sizeof(OptixHitResult), cudaMemcpyDeviceToHost);