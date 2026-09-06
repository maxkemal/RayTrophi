/*
* =========================================================================
* Project:       RayTrophi Studio
* File:          Import/ModelImport.cpp
* =========================================================================
*/
#include "Import/ModelImport.h"

#include "Import/GltfDirectReader.h"
#include "Import/UfbxReader.h"
#include "Import/ObjReader.h"
#include "Import/ModelProbe.h"

#include <algorithm>
#include <cctype>
#include <filesystem>

namespace rtimport {
namespace {

std::string lowerExtension(const std::string& filepath) {
    std::string extension = std::filesystem::path(filepath).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
        [](unsigned char c) { return char(std::tolower(c)); });
    return extension;
}

} // namespace

bool loadModel(const std::string& filepath,
               const ImportOptions& options,
               ImportedModel& out,
               std::string& error) {
    out = ImportedModel{};
    error.clear();

    if (filepath.empty()) {
        error = "empty path";
        return false;
    }

    // ★ Assimp is gone (Faz 3). Every format now has a direct reader, and there
    // is no fallback path between them by design: "the reader worked" and "the
    // reader failed and something covered for it" must never look alike from
    // the outside. An unsupported extension fails loudly here.
    if (isGltfPath(filepath)) return readGltf(filepath, options, out, error);

    const std::string extension = lowerExtension(filepath);
    if (extension == ".fbx") return readUfbx(filepath, options, out, error);
    if (extension == ".obj") return readObj(filepath, options, out, error);

    error = "unsupported model extension: " + extension;
    return false;
}

// Same dispatch, same rules — counts and a box instead of a scene.
bool probeModel(const std::string& filepath, bool applyNodeTransforms, ModelProbe& out) {
    out = ModelProbe{};
    if (filepath.empty()) return false;

    if (isGltfPath(filepath)) return probeGltf(filepath, applyNodeTransforms, out);

    const std::string extension = lowerExtension(filepath);
    if (extension == ".fbx") return probeFbx(filepath, applyNodeTransforms, out);
    if (extension == ".obj") return probeObj(filepath, applyNodeTransforms, out);
    return false;
}

} // namespace rtimport
