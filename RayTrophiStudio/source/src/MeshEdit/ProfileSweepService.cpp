#include "MeshEdit/ProfileSweepService.h"

namespace MeshEdit {

ProfileSweepResult previewProfileSweep(const ProfileSweepPreviewRequest& request) {
    const BezierSpline profile = makeSplinePrimitive(request.profile, request.primitive);
    const BezierSpline path = makeSplinePrimitive(request.path, request.primitive);
    return buildProfileSweep(profile, path, request.sweep);
}

ProfileRevolveResult previewProfileRevolve(const std::string& preset,
                                           const ProfileRevolveSettings& settings) {
    if (preset == "cup") return buildProfileRevolve(makeCupProfile(), settings);
    if (preset == "bottle") return buildProfileRevolve(makeBottleProfile(), settings);
    ProfileRevolveResult result;
    result.report.operation_id = "profile.revolve";
    result.report.addError("unknown_profile_preset", "Revolve preset must be cup or bottle.");
    return result;
}

} // namespace MeshEdit
