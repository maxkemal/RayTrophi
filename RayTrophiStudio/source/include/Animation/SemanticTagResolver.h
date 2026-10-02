#pragma once
#include <string>
#include <vector>
#include <unordered_map>

struct SceneData;

namespace RayTrophi {

class SemanticTagResolver {
public:
    static SemanticTagResolver& instance();

    // Mapping registration
    void registerSkeletonMapping(
        const std::string& skeletonSignature,
        const std::unordered_map<std::string, std::string>& tagToBoneMap
    );

    // Resolvers
    bool resolveBoneId(
        const SceneData& scene,
        const std::string& characterId,
        const std::string& semanticTag,
        std::string& outBoneId
    ) const;

    bool resolveChainBoneIds(
        const SceneData& scene,
        const std::string& characterId,
        const std::string& chainSemanticTag,
        std::vector<std::string>& outChainIds
    ) const;

    // Normalizing human readable alias to canonical tag
    std::string normalizeTag(const std::string& rawNameOrTag) const;

private:
    SemanticTagResolver() = default;
    
    // Skeleton signature -> (SemanticTag -> BonePattern/Name)
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> m_mappings;
};

} // namespace RayTrophi
