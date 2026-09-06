#pragma once
#include "Import/ImportedModel.h"
#include "Import/ModelProbe.h"

namespace rtimport {
// Increment 2: static geometry + SKINNING + transform animation.
//
// Still unsupported, and still a LOUD error rather than a silent omission:
// blend shapes (morph targets) and geometry caches. Reading such a file and
// ignoring those deformers would give a character frozen in its neutral
// expression with nothing in the log — the failure shape this repo keeps
// paying for. There is nothing to fall back TO any more: Assimp was removed in
// Faz 3 and ufbx is the only FBX reader, which is exactly why this boundary has
// to fail loudly instead of importing a partial character.
bool readUfbx(const std::string& path, const ImportOptions& options,
              ImportedModel& out, std::string& error);

// Counts and a bounding box only. Embedded image content is NOT decoded, which
// is the whole point: the asset browser wants a number, not a scene.
bool probeFbx(const std::string& path, bool applyNodeTransforms, ModelProbe& out);
}
