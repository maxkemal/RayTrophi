// AssimpLoader.cpp
#include "AssimpLoader.h"

std::vector<std::shared_ptr<Light>> AssimpLoader::lights;
std::unordered_map<std::string, std::shared_ptr<Texture>> AssimpLoader::textureCache;
std::vector<TextureInfo> AssimpLoader::textureInfos;

