# Texture Sızma Sorunu - Analiz ve Çözüm

## 🔴 Problem
Yeni bir model yüklendiğinde önceki modelin texture'ları GPU ve CPU belleğinde kalıyor ve yeni modele "sızıyor".

## 🔍 Kök Neden Analizi

### 1. **MaterialManager Temizlenmiyor**
- `create_scene()` içinde `MaterialManager::getInstance().clear()` çağrılmıyor
- Eski materyaller bellekte kalıyor
- Material ID'ler karışıyor

### 2. **OptiX CUDA Texture Arrays Temizlenmiyor**
- `OptixWrapper::destroyTextureObjects()` sadece `cudaTextureObject_t`'leri yok ediyor
- Ama altındaki `cudaArray_t`'ler bellekte kalıyor (`cudaFreeArray()` çağrılmıyor)
- CUDA bellek sızıntısı oluşuyor

### 3. **AssimpLoader::clearTextureCache() Eksik**
- Sadece `textureCache.clear()` çağrılıyor
- Her `Texture` nesnesinin `cleanup_gpu()` metodu çağrılmalı
- `cudaDestroyTextureObject()` ve `cudaFreeArray()` çağrılmalı

## ✅ Çözüm

### Adım 1: MaterialManager'ı `create_scene`'de temizle
```cpp
// Renderer.cpp - create_scene() başında
void Renderer::create_scene(SceneData& scene, OptixWrapper* optix_gpu_ptr, const std::string& model_path) {
    // Önce sahneyi sıfırla
    scene.world.clear();
    scene.lights.clear();
    scene.animatedObjects.clear();
    scene.animationDataList.clear();
    scene.camera = nullptr;
    scene.bvh = nullptr;
    scene.initialized = false;
    
    // ✅ MaterialManager'ı temizle
    MaterialManager::getInstance().clear();
    SCENE_LOG_INFO("[MATERIAL CLEANUP] MaterialManager cleared.");
    
    // ✅ Texture cache'leri temizle
    assimpLoader.clearTextureCache();
    
    // ✅ OptiX GPU texture'larını temizle (eğer varsa)
    if (g_hasOptix && optix_gpu_ptr) {
        optix_gpu_ptr->destroyTextureObjects();
        SCENE_LOG_INFO("[GPU CLEANUP] OptiX textures destroyed.");
    }
    
    // ... gerisi aynı
}
```

### Adım 2: OptiX'te CUDA Array'leri de temizle
```cpp
// OptixWrapper.h - texture array tracking ekle
class OptixWrapper {
private:
    std::vector<cudaArray_t> texture_arrays; // Her texture'ın array'ini takip et
    // ...
};

// OptixWrapper.cpp - buildFromData içinde array'leri kaydet
void OptixWrapper::buildFromData(const OptixGeometryData& data) {
    // Önce eski texture'ları temizle
    destroyTextureObjects();
    partialCleanup();
    
    // ... (mevcut kod)
    
    // Texture upload ederken array'leri kaydet
    texture_arrays.push_back(cuda_array); // Her texture için
}

// OptixWrapper.cpp - destroyTextureObjects güncelle
void OptixWrapper::destroyTextureObjects() {
    for (const auto& record : hitgroup_records) {
        const HitGroupData& data = record.data;
        
        if (data.albedo_tex) cudaDestroyTextureObject(data.albedo_tex);
        if (data.roughness_tex) cudaDestroyTextureObject(data.roughness_tex);
        if (data.normal_tex) cudaDestroyTextureObject(data.normal_tex);
        if (data.metallic_tex) cudaDestroyTextureObject(data.metallic_tex);
        if (data.transmission_tex) cudaDestroyTextureObject(data.transmission_tex);
        if (data.opacity_tex) cudaDestroyTextureObject(data.opacity_tex);
        if (data.emission_tex) cudaDestroyTextureObject(data.emission_tex);
    }
    
    // ✅ CUDA array'leri de temizle
    for (auto& array : texture_arrays) {
        if (array) {
            cudaFreeArray(array);
            array = nullptr;
        }
    }
    texture_arrays.clear();
    
    hitgroup_records.clear();
    SCENE_LOG_INFO("[GPU CLEANUP] All texture objects and arrays destroyed.");
}
```

### Adım 3: AssimpLoader::clearTextureCache güncelle
```cpp
// AssimpLoader.h - clearTextureCache güncelle
void clearTextureCache() {
    SCENE_LOG_INFO("[TEXTURE CLEANUP] Starting comprehensive texture cleanup...");
    int gpu_cleaned = 0;
    int cpu_cleaned = 0;
    
    // 1. AssimpLoader'ın local cache'ini temizle
    for (auto& [name, tex] : textureCache) {
        if (tex) {
            tex->cleanup_gpu(); // ✅ GPU belleği temizle
            gpu_cleaned++;
        }
    }
    cpu_cleaned = textureCache.size();
    textureCache.clear();
    
    // 2. Global singleton cache'leri de temizle
    size_t global_texture_cache_size = TextureCache::instance().size();
    size_t global_file_cache_size = FileTextureCache::instance().size();
    
    TextureCache::instance().clear();
    FileTextureCache::instance().clear();
    
    SCENE_LOG_INFO("[TEXTURE CLEANUP] Complete! Stats:");
    SCENE_LOG_INFO("  - GPU textures cleaned: " + std::to_string(gpu_cleaned));
    SCENE_LOG_INFO("  - CPU cache entries removed: " + std::to_string(cpu_cleaned));
    SCENE_LOG_INFO("  - Global TextureCache cleared: " + std::to_string(global_texture_cache_size) + " entries");
    SCENE_LOG_INFO("  - Global FileTextureCache cleared: " + std::to_string(global_file_cache_size) + " entries");
}
```

## 📊 Beklenen Sonuç
Bu düzeltmelerden sonra yeni model yüklendiğinde:
1. ✅ Tüm CPU texture cache'leri temizlenir
2. ✅ Tüm GPU texture object'leri yok edilir
3. ✅ Tüm CUDA array'leri serbest bırakılır
4. ✅ MaterialManager sıfırlanır
5. ✅ Önceki modelden HİÇBİR ŞEY kalmaz

## 🧪 Test Senaryosu
1. Model A yükle → render et
2. Model B yükle → render et
3. Kontrol: Model B'de Model A'nın texture'ları görülmemeli
4. CUDA memory leak check: `nvidia-smi` ile bellek kullanımı kontrol edilmeli
