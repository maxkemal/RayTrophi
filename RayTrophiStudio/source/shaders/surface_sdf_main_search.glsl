        uint sdfSamples = 1u;
        uint sdfQueries = 0u;
        uint sdfSkipped = 0u;
        SurfaceSdfRegionCache sdfRegion;
        sdfRegion.base = ivec3(0);
        sdfRegion.width = 0u;
        sdfRegion.upper = 1e30;
        bool sdfNano = vol.volume_type == 2 && vdbBuf.address != uint64_t(0);
        VolumeNanoRay sdfRay;
        if (sdfNano) sdfRay = volumeNanoRay(vol, rayOrigin, rayDir, vdbBuf, vdbMapH);
        for (int s = 0; s < maxSteps; ++s) {
            int skip = sdfNano ? surfaceSdfEmptySteps(vol, sdfRay, t, step, tFar,
                maxSteps - s, startInside, prevD, vdbBuf, vdbAcc, sdfRegion, sdfQueries) : 0;
            if (skip >= 2) {
                t += float(skip) * step;
                prevD = sampleIsoField(vol, rayOrigin + rayDir * t, vdbBuf, vdbMapH, vdbAcc);
                sdfSamples++;
                sdfSkipped += uint(skip - 1);
                s += skip - 1;
                if (t >= tFar) break;
                continue;
            }
            float nextT = min(t + step, tFar);
            float curD = sampleIsoField(vol, rayOrigin + rayDir * nextT, vdbBuf, vdbMapH, vdbAcc);
            sdfSamples++;
            bool crossed = startInside
                ? (prevD >= ISO_THRESH && curD < ISO_THRESH)   // exit
                : (prevD <  ISO_THRESH && curD >= ISO_THRESH); // enter
            if (crossed) {
                // Preserve the original bisection refinement and crossing bracket.
                float t0 = t;
                float t1 = nextT;
                float d0 = prevD;
                float d1 = curD;
                for (int it = 0; it < 4; ++it) {
                    float tMid = 0.5 * (t0 + t1);
                    float dMid = sampleIsoField(vol, rayOrigin + rayDir * tMid, vdbBuf, vdbMapH, vdbAcc);
                    sdfSamples++;
                    bool midInside = dMid > ISO_THRESH;
                    if (startInside == midInside) {
                        t0 = tMid;
                        d0 = dMid;
                    } else {
                        t1 = tMid;
                        d1 = dMid;
                    }
                }
                float denom = d1 - d0;
                float frac = (ISO_THRESH - d0) / ((abs(denom) > EPSILON) ? denom : EPSILON);
                frac = clamp(frac, 0.0, 1.0);
                hitT = t0 + frac * (t1 - t0);
                break;
            }
            prevD = curD;
            t = nextT;
            if (t >= tFar) break;
        }

        volumeRecordSdfSearch(false, sdfSamples, sdfQueries, sdfSkipped);

