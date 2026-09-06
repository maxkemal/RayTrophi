/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Animation/AnimationData.h
* Author:        Kemal Demirtas
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*
* WHAT A SKINNED, ANIMATED MODEL CARRIES BESIDES GEOMETRY.
*
* ★★★ WHY THIS FILE EXISTS — A PREREQUISITE, NOT A TIDY-UP (Faz 3)
* ---------------------------------------------------------------------------
* Both structs below are CORE engine types. Faz 0 already removed Assimp from
* their contents (aiVectorKey/aiQuatKey became RayTrophi::VectorKey/QuatKey;
* BoneData::boneNameToNode, a map of DANGLING aiNode*, was deleted outright).
* But they kept LIVING in AssimpLoader.h, and that is not a cosmetic problem:
*
*   - AssimpLoader.h is 3400 lines and opens with <assimp/Importer.hpp>,
*     <assimp/scene.h>, <assimp/postprocess.h>. It is included by Renderer.h
*     and scene_data.h, so essentially the whole engine transitively included
*     Assimp — to reach two structs that no longer contain any Assimp type.
*   - Every NEW reader had to include the OLD loader. UfbxReader.cpp and
*     GltfDirectReader.cpp both did exactly that, with an apologetic comment.
*     A reader written to replace Assimp cannot depend on Assimp's header.
*
* So deleting AssimpLoader is impossible while these two live there. This file
* is that door being opened; it changes no behaviour and no layout.
*
* ★ THE TWO ARE TOGETHER ON PURPOSE. A clip and the skeleton it drives are
* always read as a pair — AnimationController, AnimatedObject, OzzRuntime and
* the .rtp serializer each take both — and splitting them into two headers
* would only mean two includes at every one of those call sites.
*
* ★ MeshInstance was NOT moved, it was DELETED. It sat between these two
* structs, described an "aiMesh ID", and had exactly zero readers in the repo.
* Migrating it would have carried a dead Assimp-shaped type into a header meant
* to be Assimp-free (CLAUDE.md rule 5: verified dead, so removed).
* =========================================================================
*/
#pragma once

#include <cmath>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Matrix4x4.h"
#include "Quaternion.h"
#include "Vec3.h"
// VectorKeys / QuatKeys, and decomposeTRS() used by the sampler below.
#include "Animation/AnimationKeys.h"

struct AnimationData {
    std::string name;
    std::string modelName; // Import prefix of the model this animation belongs to
    double duration;
    double ticksPerSecond;
    // ★ RayTrophi's OWN key type, not aiVectorKey/aiQuatKey (Faz 0 of the Assimp
    // import replacement). These maps are read by AnimatedObject, AnimationNodes
    // and the .rtp serializer - i.e. Assimp's types had reached the animation
    // core, and the loader could not be swapped without changing them first.
    std::map<std::string, RayTrophi::VectorKeys> positionKeys;
    std::map<std::string, RayTrophi::QuatKeys>   rotationKeys;
    std::map<std::string, RayTrophi::VectorKeys> scalingKeys;
    
   
    int startFrame = 0;
    int endFrame = 0;

    // Calculates the interpolated transform matrix for a node at a specific time
    // Belirli bir andaki node için enterpolasyonlu dönüşüm matrisini hesaplar
    Matrix4x4 calculateAnimationTransform(const AnimationData& animData, float time, const std::string& nodeName, const Matrix4x4& defaultTransform)
        const {
      
        // Calculate normalized animation time within duration
        // Animasyon süresini normalize et (döngüsel zaman)
        double animationTime = fmod(time * ticksPerSecond, duration);

        // Decompose default transform to retrieve Bind Pose values
        // Bind Pose değerlerini almak için varsayılan transformu ayır
        // ★ Was: copy into an aiMatrix4x4 just to call its Decompose(). That one
        // call is why <assimp/matrix4x4.h> had to be reachable from here.
        Vec3 defScale, defPos;
        Quaternion defRot;
        RayTrophi::decomposeTRS(defaultTransform, defPos, defRot, defScale);

        bool hasPos = positionKeys.count(nodeName) && !positionKeys.at(nodeName).empty();
        bool hasRot = rotationKeys.count(nodeName) && !rotationKeys.at(nodeName).empty();
        bool hasScl = scalingKeys.count(nodeName) && !scalingKeys.at(nodeName).empty();

        if (!hasPos && !hasRot && !hasScl) {
            return defaultTransform;
        }

        Matrix4x4 translationMatrix = Matrix4x4::identity();
        Matrix4x4 rotationMatrix = Matrix4x4::identity();
        Matrix4x4 scaleMatrix = Matrix4x4::identity();

        // --- POSITION ---
        if (hasPos) {
            Vec3 position(0, 0, 0);
            const auto& keys = positionKeys.at(nodeName);

            // Binary search: keys are sorted ascending by time (guaranteed by the loader).
            // upper_bound finds the first key strictly after animationTime; step back one
            // to get the key at-or-before it.  Wraps to 0 if past the last key (loop semantics).
            size_t frameIndex = 0;
            if (keys.size() > 1) {
                auto it = std::upper_bound(keys.begin(), keys.end(), animationTime,
                    [](double t, const RayTrophi::VectorKey& k) { return t < k.time; });
                if (it != keys.begin()) --it;
                frameIndex = static_cast<size_t>(it - keys.begin());
                if (frameIndex >= keys.size() - 1) frameIndex = 0;
            }

            size_t nextFrameIndex = (frameIndex + 1) % keys.size();
            float deltaTime = (float)(keys[nextFrameIndex].time - keys[frameIndex].time);
            if (deltaTime < 0) deltaTime += (float)duration;

            float factor = (deltaTime == 0) ? 0.0f :
                (float)(animationTime - keys[frameIndex].time) / deltaTime;

            const Vec3& start = keys[frameIndex].value;
            const Vec3& end = keys[nextFrameIndex].value;
            position = Vec3(
                start.x + (end.x - start.x) * factor,
                start.y + (end.y - start.y) * factor,
                start.z + (end.z - start.z) * factor
            );
            translationMatrix = Matrix4x4::translation(position);
        } else {
             // Fallback to Bind Pose Position
             translationMatrix = Matrix4x4::translation(defPos);
        }

        // --- ROTATION (Dönme) ---
        if (hasRot) {
            Quaternion rotation(0, 0, 0, 1);
            const auto& keys = rotationKeys.at(nodeName);

            size_t frameIndex = 0;
            if (keys.size() > 1) {
                auto it = std::upper_bound(keys.begin(), keys.end(), animationTime,
                    [](double t, const RayTrophi::QuatKey& k) { return t < k.time; });
                if (it != keys.begin()) --it;
                frameIndex = static_cast<size_t>(it - keys.begin());
                if (frameIndex >= keys.size() - 1) frameIndex = 0;
            }

            size_t nextFrameIndex = (frameIndex + 1) % keys.size();
            double deltaTime = keys[nextFrameIndex].time - keys[frameIndex].time;
            if (deltaTime < 0) deltaTime += duration;

            double factor = (deltaTime == 0) ? 0.0 :
                (animationTime - keys[frameIndex].time) / deltaTime;

            const Quaternion& start = keys[frameIndex].value;
            const Quaternion& end = keys[nextFrameIndex].value;
            rotation = Quaternion::slerp(start, end, (float)factor);
            rotationMatrix = rotation.toMatrix();
        } else {
             // Fallback to Bind Pose Rotation (Fixes FBX flipped camera issues)
             // Bind Pose Rotasyonuna dön (FBX ters kamera sorunlarını düzeltir)
             rotationMatrix = defRot.toMatrix();
        }

        // --- SCALING (Ölçekleme) ---
        if (hasScl) {
            Vec3 scaling(1, 1, 1);
            const auto& keys = scalingKeys.at(nodeName);

            size_t frameIndex = 0;
            if (keys.size() > 1) {
                auto it = std::upper_bound(keys.begin(), keys.end(), animationTime,
                    [](double t, const RayTrophi::VectorKey& k) { return t < k.time; });
                if (it != keys.begin()) --it;
                frameIndex = static_cast<size_t>(it - keys.begin());
                if (frameIndex >= keys.size() - 1) frameIndex = 0;
            }

            size_t nextFrameIndex = (frameIndex + 1) % keys.size();
            float deltaTime = (float)(keys[nextFrameIndex].time - keys[frameIndex].time);
            if (deltaTime < 0) deltaTime += (float)duration;

            float factor = (deltaTime == 0) ? 0.0f :
                (float)(animationTime - keys[frameIndex].time) / deltaTime;

            const Vec3& start = keys[frameIndex].value;
            const Vec3& end = keys[nextFrameIndex].value;
            scaling = Vec3(
                start.x + (end.x - start.x) * factor,
                start.y + (end.y - start.y) * factor,
                start.z + (end.z - start.z) * factor
            );
            scaleMatrix = Matrix4x4::scaling(scaling);
        } else {
            // Fallback to Bind Pose Scale
             scaleMatrix = Matrix4x4::scaling(defScale);
        }

        // Combine matrices: Final = Translation * Rotation * Scale
        // Matrisleri birleştir: Final = Çevirme * Dönme * Ölçekleme
        return translationMatrix * rotationMatrix * scaleMatrix;
    }
};

struct BoneData {
    std::unordered_map<std::string, unsigned int> boneNameToIndex;
    // ★ boneNameToNode (std::unordered_map<std::string, aiNode*>) was REMOVED in
    // Faz 0 of the Assimp import replacement. It was not just an Assimp type in a
    // core struct - it was a DANGLING one: the aiScene those aiNode* pointed into
    // is released when the importer goes out of scope, and BoneData outlives it
    // (Renderer even merged the map across imports). Nothing read it: every value
    // it carried was already copied out beside it into boneDefaultTransforms and
    // boneParents at fill time. Dead pointers with no reader, so: removed, not
    // migrated (CLAUDE.md rule 5).
    std::unordered_map<std::string, Matrix4x4> boneOffsetMatrices;
    std::unordered_map<std::string, Matrix4x4> boneDefaultTransforms; // NEW: Local bind pose (node->mTransformation)
    std::unordered_map<std::string, std::string> boneParents; // NEW: Child name -> Parent name for hierarchy reconstruction
    std::unordered_map<std::string, Matrix4x4> perModelInverses; // NEW: Model prefix -> globalInverseTransform
    std::unordered_set<std::string> weightedBoneNames; // Only bones coming from skinned mesh weights
    Matrix4x4 globalInverseTransform;
    
    // =========================================================================
    // OPTIMIZATION: Reverse lookup table (bone index -> bone name)
    // Eliminates O(n²) complexity in animation updates
    // OPTİMİZASYON: Ters arama tablosu (kemik indeksi -> kemik adı)
    // Animasyon güncellemelerindeki O(n²) karmaşıklığını ortadan kaldırır
    // =========================================================================
    std::vector<std::string> boneIndexToName;
    
    // Rebuild reverse lookup table - call after all bones are added
    // Ters arama tablosunu yeniden oluştur (tüm kemikler eklendikten sonra çağır)
    void rebuildReverseLookup() {
        if (boneNameToIndex.empty()) {
            boneIndexToName.clear();
            return;
        }
        
        // Find max index to size the vector correctly
        unsigned int maxIndex = 0;
        for (const auto& [name, idx] : boneNameToIndex) {
            if (idx > maxIndex) maxIndex = idx;
        }
        
        boneIndexToName.resize(maxIndex + 1);
        for (const auto& [name, idx] : boneNameToIndex) {
            boneIndexToName[idx] = name;
        }
    }
    
    // Get bone name by index (O(1) lookup)
    // İndekse göre kemik adını al (O(1) arama)
    const std::string& getBoneNameByIndex(unsigned int index) const {
        static const std::string empty;
        if (index < boneIndexToName.size()) {
            return boneIndexToName[index];
        }
        return empty;
    }
    
    // Check if bone index is valid
    // Kemik indeksinin geçerli olup olmadığını kontrol et
    bool isValidBoneIndex(unsigned int index) const {
        return index < boneIndexToName.size() && !boneIndexToName[index].empty();
    }
    
    // Get total bone count (Toplam kemik sayısı)
    size_t getBoneCount() const {
        return boneNameToIndex.size();
    }
    
    void clear() {
        boneNameToIndex.clear();
        boneOffsetMatrices.clear();
        boneDefaultTransforms.clear();
        boneParents.clear();
        perModelInverses.clear();
        weightedBoneNames.clear();
        boneIndexToName.clear();
        globalInverseTransform = Matrix4x4::identity();
    }
    
    // Original root transform before any intelligent scaling corrections
    Matrix4x4 originalRootTransform = Matrix4x4::identity();
};
