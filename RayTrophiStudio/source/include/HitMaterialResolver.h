#pragma once

#include "Hittable.h"

namespace HitMaterialResolver {

void resolveMaterialPointers(HitRecord& rec);
void applyTerrainBlendIfNeeded(HitRecord& rec);
void resolveSurfaceData(HitRecord& rec);

}
