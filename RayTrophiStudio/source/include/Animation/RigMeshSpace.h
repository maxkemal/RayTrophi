#pragma once
#include "DNA/GeometryDetail.h"
namespace RigAuthoring {
// Static viewport geometry uses local P_orig/N_orig. P/N may be baked world
// caches; transforming those again would apply the import placement twice.
inline const Vec3* meshSpacePositions(const DNA::GeometryDetail& geometry,bool weighted=false) {
    const auto* original=weighted?nullptr:geometry.get_positions_orig();
    return original?original:geometry.get_positions();
}
inline size_t meshSpacePositionCount(const DNA::GeometryDetail& geometry,bool weighted=false) {
    return weighted?geometry.get_core_attribute_count(DNA::Attr::P):geometry.get_positions_orig_count();
}
inline const Vec3* meshSpaceNormals(const DNA::GeometryDetail& geometry) {
    const auto* original=geometry.get_normals_orig();
    return original?original:geometry.get_normals();
}
inline size_t meshSpaceNormalCount(const DNA::GeometryDetail& geometry) {
    return geometry.get_core_attribute_count(DNA::Attr::N_orig)?geometry.get_core_attribute_count(DNA::Attr::N_orig):geometry.get_core_attribute_count(DNA::Attr::N);
}
}
