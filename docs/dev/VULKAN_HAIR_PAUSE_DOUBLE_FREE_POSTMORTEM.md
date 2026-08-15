# Vulkan RT Hair: İkinci Pause/Play Double-Free Olay Notu

> **Durum:** ARSIV — Cozuldu ve dogrulandi; kok neden postmortem'i.

## Durum

Çözüldü ve kullanıcı tarafından doğrulandı.

## Belirti

- Yalnızca Vulkan RT Rendered viewport yolunda görülüyordu.
- Hair dynamics ve force field kapalıyken de oluşabiliyordu.
- İlk `Play -> Pause` çoğunlukla çalışıyordu.
- İkinci pause veya takip eden Play sırasında uygulama kapanıyordu.
- Debug çağrı zinciri şu sınıftaydı:

```text
nvoglv64.dll
VulkanRT::VulkanDevice::destroyBuffer()
Backend::VulkanBackendAdapter::rebuildAccelerationStructure()
SDL_main() deferred Vulkan rebuild block
```

Bu imza bir shader, hair fiziği veya timeline zamanlama hatasından çok Vulkan
allocation sahipliği/double-free problemine işaret eder.

## Tetikleme Zinciri

`Main.cpp`, playback'in aktif durumdan pasife geçtiğini algıladığında, refit edilmiş
BVH kalitesini yenilemek için bir defalık `g_vulkan_rebuild_pending` işaretler.
Frame sonundaki deferred rebuild yolu şunları çalıştırır:

1. `VulkanBackendAdapter::rebuildAccelerationStructure()`
2. `updateGeometry()`
3. `Renderer::uploadHairToGPU()`

Dolayısıyla pause yalnızca zamanlayıcı değişimi değildir; tam BLAS/TLAS teardown ve
rebuild tetikleyebilir. Hatanın ikinci pause/play'de görünmesinin nedeni ilk teardown'ın
buffer sahiplik durumunu bozması, sonraki teardown'ın bozuk handle'ı yeniden free etmesiydi.

## Kök Neden

GPU hair yolu, compute shader'ın doldurduğu ortak `m_hairAabbBuffer` üzerinden bir
AABB BLAS oluşturur:

```cpp
blasHandle.vertexBuffer     = aabbBuffer; // borrowed alias
blasHandle.externalGeometry = true;
```

Buradaki `vertexBuffer` yeni veya BLAS'a ait bir allocation değildir. Sadece cihazın
sahip olduğu `m_hairAabbBuffer` handle'ının kopyasıdır.

`clearHairGeometry()` bu kuralı zaten doğru uyguluyordu:

```cpp
if (!blas.externalGeometry)
    destroyBuffer(blas.vertexBuffer);
```

Ancak full rebuild temizliği `externalGeometry` kontrolü yapmadan
`blas.vertexBuffer`'ı yok ediyordu. Sonuç:

1. İlk pause: ortak hair AABB allocation alias üzerinden serbest bırakılır.
2. Asıl sahip `m_hairAabbBuffer`, artık geçersiz olan handle ve size bilgisini tutmaya devam eder.
3. Hair yeniden yüklenirken buffer yeterli boyutta sanılır ve yeniden allocate edilmez.
4. İkinci rebuild/play: geçersiz allocation tekrar `vkFreeMemory` yoluna girer.
5. NVIDIA sürücüsü içinde access violation oluşur.

## Uygulanan Düzeltme

`rebuildAccelerationStructure()` temizliği şu kurallara geçirildi:

- `externalGeometry == true` ise `vertexBuffer` yalnızca sıfırlanır, destroy edilmez.
- Sahip olunan buffer ve memory handle'ları tüm BLAS listesi boyunca set ile takip edilir.
- Aynı `VkBuffer` veya `VkDeviceMemory` en fazla bir kez yok edilir.
- BLAS'a ait kalıcı `skinScratchBuffer` teardown sırasında serbest bırakılır.
- `VulkanDevice` genel teardown yolu da hair scratch buffer'ını temizler.

İlgili kaynaklar:

- `RayTrophiStudio/source/include/Backend/VulkanBackend.h`
  - `AccelStructHandle::externalGeometry` sahiplik sözleşmesi.
- `RayTrophiStudio/source/src/Backend/VulkanBackend.cpp`
  - `createHairAABB_BLAS_Device()`
  - `clearHairGeometry()`
  - `rebuildAccelerationStructure()`

## Gelecekte Benzer Çöküş İçin Kontrol Listesi

Bir Vulkan teardown/rebuild çöküşü `destroyBuffer`, `vkDestroyBuffer` veya
`vkFreeMemory` içinde görünürse:

1. Handle gerçekten bu yapı tarafından mı sahipleniliyor, yoksa yalnızca alias mı?
2. Aynı `VkDeviceMemory` birden fazla `BufferHandle` kopyasında tutuluyor mu?
3. Attribute view'ları tek birleşik geometry buffer'ına mı işaret ediyor?
4. `externalGeometry` kontrolü bütün cleanup yollarında aynı biçimde uygulanıyor mu?
5. Owner buffer yok edildikten sonra size/handle taşıyan başka bir cache canlı kalıyor mu?
6. Normal cleanup, full rebuild ve device destructor aynı sahiplik sözleşmesini uyguluyor mu?
7. İlk işlem state'i bozup çöküşü yalnızca ikinci tekrar üzerinde mi görünür kılıyor?

Özellikle “ilk kullanım çalışıyor, ikinci teardown/rebuild çöküyor” paterni güçlü bir
double-free veya stale-owner-handle göstergesidir.

## Ayrı Tutulması Gereken Konular

Araştırma sırasında iki ek Vulkan RT güvenlik iyileştirmesi de yapıldı:

- Hair BLAS yerinde güncellenirken gereksiz per-frame TLAS rebuild kaldırıldı.
- Timeline pause öncesine in-flight backend completion bariyeri eklendi.

Bunlar geçerli senkronizasyon iyileştirmeleridir; fakat bu olayın doğrulanmış asıl
kök nedeni `externalGeometry` hair AABB alias'ının full rebuild sırasında free edilmesiydi.
