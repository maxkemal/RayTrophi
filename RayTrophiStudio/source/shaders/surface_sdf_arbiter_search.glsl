        uint sdfSamples = 1u;
        uint sdfQueries = 0u;
        uint sdfSkipped = 0u;
        SurfaceSdfRegionCache sdfRegion;
        sdfRegion.base = ivec3(0);
        sdfRegion.width = 0u;
        sdfRegion.upper = 1e30;
        bool sdfNano = surface.volume_type == 2 && surfaceBuf.address != uint64_t(0);
        VolumeNanoRay sdfRay;
        if (sdfNano) sdfRay = volumeNanoRay(surface, rayOrigin, rayDir, surfaceBuf, mapH);
        for (int s = 0; s < steps; ++s) {
            int skip = sdfNano ? surfaceSdfEmptySteps(surface, sdfRay, t0s, step, endT,
                steps - s, startedInside, d0, surfaceBuf, acc, sdfRegion, sdfQueries) : 0;
            if (skip >= 2) {
                t0s += float(skip) * step;
                d0 = sampleIsoField(surface, rayOrigin + rayDir * t0s, surfaceBuf, mapH, acc);
                sdfSamples++;
                sdfSkipped += uint(skip - 1);
                s += skip - 1;
                if (t0s >= endT) break;
                continue;
            }
            float t1s = min(t0s + step, endT);
            float d1 = sampleIsoField(
                surface, rayOrigin + rayDir * t1s, surfaceBuf, mapH, acc);
            sdfSamples++;
            bool crossed = startedInside
                ? (d0 >= ISO && d1 < ISO)
                : (d0 < ISO && d1 >= ISO);
            if (crossed) {
                float a = t0s;
                float b = t1s;
                for (int refine = 0; refine < 4; ++refine) {
                    float mid = 0.5 * (a + b);
                    float dm = sampleIsoField(
                        surface, rayOrigin + rayDir * mid, surfaceBuf, mapH, acc);
                    sdfSamples++;
                    if ((dm > ISO) == startedInside) a = mid;
                    else b = mid;
                }
                nearestHit = min(nearestHit, 0.5 * (a + b));
                foundCrossing = true;
                break;
            }
            t0s = t1s;
            d0 = d1;
            if (t0s >= endT) break;
        }
        volumeRecordSdfSearch(true, sdfSamples, sdfQueries, sdfSkipped);

