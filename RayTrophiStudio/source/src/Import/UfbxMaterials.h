#pragma once
#include "../../../external/ufbx/ufbx.h"
#include "Import/ImportedModel.h"
#include "Texture.h"
#include <map>
#include <unordered_map>

namespace rtimport {
struct FbxMaterialBinding {
    uint16_t id = 0;
    size_t uvSet = 0;
};
class UfbxMaterials {
public:
    UfbxMaterials(const std::string& path, const ImportOptions& options, ImportStats& stats);
    void prefetch(const ufbx_scene& scene);
    FbxMaterialBinding bind(const ufbx_material* material, const ufbx_mesh& mesh);
private:
    using TextureKey = std::pair<std::string, int>;
    TextureKey key(const ufbx_texture& texture, TextureType type) const;
    std::shared_ptr<Texture> decode(const ufbx_texture& texture, TextureType type) const;
    std::shared_ptr<Texture> texture(const ufbx_material_map& map, TextureType type);
    std::string directory_;
    const ImportOptions& options_;
    ImportStats& stats_;
    std::map<TextureKey, std::shared_ptr<Texture>> textures_;
    std::map<std::pair<const ufbx_material*, size_t>, uint16_t> materials_;
};
}
