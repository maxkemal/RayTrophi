
void HairSystem::updateSkinnedGroom(const std::string& groomName) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    auto it = m_grooms.find(groomName);
    if (it == m_grooms.end()) return;
    
    HairGroom& groom = it->second;
    if (groom.boundTriangles.empty()) return;

    // Helper to get position from barycentrics
    auto getPos = [](const Triangle& tri, float u, float v, bool original) {
        float w = 1.0f - u - v;
        if (original) {
            return tri.getOriginalVertexPosition(0) * w + 
                   tri.getOriginalVertexPosition(1) * u + 
                   tri.getOriginalVertexPosition(2) * v;
        } else {
            return tri.getVertexPosition(0) * w + 
                   tri.getVertexPosition(1) * u + 
                   tri.getVertexPosition(2) * v;
        }
    };
    
    // Helper to get normal from barycentrics
    auto getNorm = [](const Triangle& tri, float u, float v, bool original) {
        float w = 1.0f - u - v;
        if (original) {
            return (tri.getOriginalVertexNormal(0) * w + 
                    tri.getOriginalVertexNormal(1) * u + 
                    tri.getOriginalVertexNormal(2) * v).normalize();
        } else {
            return (tri.getVertexNormal(0) * w + 
                    tri.getVertexNormal(1) * u + 
                    tri.getVertexNormal(2) * v).normalize();
        }
    };

    bool anyChange = false;

    // Update guides based on skin deformation
    for (auto& strand : groom.guides) {
        if (strand.triangleIndex >= groom.boundTriangles.size()) continue;
        
        const auto& tri = *groom.boundTriangles[strand.triangleIndex];
        
        // 1. Calculate Old Frame (Bind Pose)
        Vec3 P0 = getPos(tri, strand.barycentricUV.u, strand.barycentricUV.v, true);
        Vec3 N0 = getNorm(tri, strand.barycentricUV.u, strand.barycentricUV.v, true);
        
        // 2. Calculate New Frame (Current Pose)
        Vec3 P1 = getPos(tri, strand.barycentricUV.u, strand.barycentricUV.v, false);
        Vec3 N1 = getNorm(tri, strand.barycentricUV.u, strand.barycentricUV.v, false);
        
        // 3. Compute Rotation (align N0 to N1)
        Quaternion rot = Quaternion::rotationBetween(N0, N1);
        Matrix4x4 R = rot.toMatrix();
        
        // 4. Update all control points
        // Use restGroomedPositions as the source of truth for the shape
        if (strand.groomedPositions.size() != strand.restGroomedPositions.size()) {
            strand.restGroomedPositions = strand.groomedPositions; // Reset if mismatch
        }

        for (size_t i = 0; i < strand.groomedPositions.size(); ++i) {
            Vec3 oldPt = strand.restGroomedPositions[i];
            Vec3 localOffset = oldPt - P0;
            Vec3 revolvedOffset = R.transform_vector(localOffset); 
            strand.groomedPositions[i] = P1 + revolvedOffset;
        }
        anyChange = true;
    }

    if (anyChange) {
        // Re-interpolate children to follow new guide positions
        interpolateChildren(groom);
        m_bvhDirty = true;
    }
}
