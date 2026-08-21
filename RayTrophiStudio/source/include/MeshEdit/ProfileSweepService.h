#pragma once

#include "MeshEdit/ProfileSweep.h"
#include "MeshEdit/SplinePrimitive.h"
#include "MeshEdit/ProfileRevolve.h"

namespace MeshEdit {

struct ProfileSweepPreviewRequest {
    SplinePrimitiveType profile = SplinePrimitiveType::Circle;
    SplinePrimitiveType path = SplinePrimitiveType::OpenLine;
    SplinePrimitiveSettings primitive;
    ProfileSweepSettings sweep;
};

// Shared operation entry point for UI, Python and IPC. It intentionally returns
// generated flat data without publishing into a scene; commit/persistence is the
// next service layer and will consume this same result.
ProfileSweepResult previewProfileSweep(const ProfileSweepPreviewRequest& request);

ProfileRevolveResult previewProfileRevolve(const std::string& preset,
                                           const ProfileRevolveSettings& settings = {});

} // namespace MeshEdit
