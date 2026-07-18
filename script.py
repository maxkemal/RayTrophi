import sys

with open('e:/RayTrophi_projesi/raytracing_Proje_Moduler/RayTrophiStudio/source/src/Backend/VulkanBackend.cpp', 'r', encoding='utf-8') as f:
    content = f.read()

with open('scratch_method.txt', 'r', encoding='utf-8') as f:
    method_content = f.read()

# Replace lambda definition with call to resolveTextureHandle
lambda_start = content.find('auto getTexID = [this](int64_t key')
if lambda_start == -1:
    print('Lambda not found')
    sys.exit(1)

# Find the end of the lambda
lambda_end = content.find('};', lambda_start) + 2

lambda_replacement = '''auto getTexID = [this](int64_t key, TextureType textureType, bool forceLinear = false, bool preferSingleChannel = false) -> uint32_t {
            return this->resolveTextureHandle(key, static_cast<int>(textureType), forceLinear, preferSingleChannel);
        };'''

content = content[:lambda_start] + lambda_replacement + content[lambda_end:]

# Insert the method right before uploadMaterials
upload_start = content.find('void VulkanBackendAdapter::uploadMaterials(const std::vector<MaterialData>& materials) {')
content = content[:upload_start] + method_content + '\n\n' + content[upload_start:]

with open('e:/RayTrophi_projesi/raytracing_Proje_Moduler/RayTrophiStudio/source/src/Backend/VulkanBackend.cpp', 'w', encoding='utf-8') as f:
    f.write(content)

print('Success')
