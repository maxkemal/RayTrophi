# OIDN Denoiser Optimizasyon Raporu

**Tarih:** 2026-04-18
**Kapsam:** `Renderer::applyOIDNDenoising` ve çağrı yolları
**Hedef:** CUDA-olmayan makinalarda sorunsuz CPU fallback + CUDA'da büyük hızlanma

---

## 1. Mevcut Mimari

OIDN tek noktada toplanmış, `OptixWrapper`'daki eski kopya kaldırılmış:

- [Renderer.cpp:80-105](raytrac_sdl2/source/src/Render/Renderer.cpp#L80-L105) — `initOIDN()`
  CUDA denemesi → başarısızsa CPU fallback. **Bu güvenlik ağı korunacak.**
- [Renderer.cpp:107-157](raytrac_sdl2/source/src/Render/Renderer.cpp#L107-L157) — `applyOIDNDenoising(SDL_Surface*)` (tek-frame / animasyon yolu)
- [Renderer.cpp:159-272](raytrac_sdl2/source/src/Render/Renderer.cpp#L159-L272) — `applyOIDNDenoising(OIDNFrameData&, …)` merkezi implementasyon
- [Renderer.cpp:274-321](raytrac_sdl2/source/src/Render/Renderer.cpp#L274-L321) — `applyOIDNDenoisingToCPUAccumulation` (CPU render sonrası)
- [OptixBackend.cpp:327-343](raytrac_sdl2/source/src/Backend/OptixBackend.cpp#L327-L343) — `getDenoiserFrame` (OptiX viewport yolu)

OIDN davranışı: CUDA cihazında çalışıyorsa `newBuffer` GPU bellekte, CPU cihazında host bellekte. API aynı — dolayısıyla kod yolu tek.

---

## 2. Tespit Edilen Maliyet Noktaları

### P1 — Gereksiz ikinci kopya (`oidnOriginalData`)
[Renderer.cpp:180](raytrac_sdl2/source/src/Render/Renderer.cpp#L180), [L220-228](raytrac_sdl2/source/src/Render/Renderer.cpp#L220-L228), [L261-263](raytrac_sdl2/source/src/Render/Renderer.cpp#L261-L263)

`frame.color` zaten çağıran tarafta mevcut; blend için ayrıca `oidnOriginalData` tutuluyor.
**Maliyet:** 1 × full-image allocation + 1 × full-image copy her çağrıda.

### P2 — Tek-thread pixel döngüleri
- [L116-129](raytrac_sdl2/source/src/Render/Renderer.cpp#L116-L129) SDL decode
- [L141-156](raytrac_sdl2/source/src/Render/Renderer.cpp#L141-L156) SDL encode + blend
- [L220-228](raytrac_sdl2/source/src/Render/Renderer.cpp#L220-L228) input copy
- [L253-269](raytrac_sdl2/source/src/Render/Renderer.cpp#L253-L269) blend
- [L289-310](raytrac_sdl2/source/src/Render/Renderer.cpp#L289-L310) CPU-accum'dan linearize

`<execution>` include edilmiş ama kullanılmıyor. 1080p'de her döngü ~2M iterasyon, tek thread.

### P3 — OptiX yolunda çift yönlü round-trip
[OptixBackend.cpp:333](raytrac_sdl2/source/src/Backend/OptixBackend.cpp#L333) → `downloadDenoiserBuffers` GPU→host (3 buffer × float4).
Sonrasında OIDN CUDA modundaysa → host→GPU (OIDN buffer'a write) → GPU→host (output read).
**Viewport'ta frame başına ~6 × image-size PCIe transferi.**

### P4 — SDL pack/unpack mecburiyeti
[L116-129](raytrac_sdl2/source/src/Render/Renderer.cpp#L116-L129) ve [L141-156](raytrac_sdl2/source/src/Render/Renderer.cpp#L141-L156)
Viewport yolunda float accumulation zaten var; Uint32 → float → Uint32 çevrimi çoğu durumda gereksiz.

### P5 — Filter cache ince ayarı
[L174-215](raytrac_sdl2/source/src/Render/Renderer.cpp#L174-L215) — layout/size değişiminde yeniden kuruluyor (iyi). Ama `hdr/srgb` gibi statik ayarlar her frame set edilmiyor — bu zaten doğru. İyileştirme yok, mevcut iyi.

---

## 3. Önerilen Değişiklikler ve CPU/CUDA Uyumluluğu

| # | Değişiklik | CPU Modu | CUDA Modu | Tahmini Kazanç |
|---|---|---|---|---|
| C1 | `oidnOriginalData`'yı sil, blend'i `frame.color`'dan oku | ✅ Tamamen uyumlu | ✅ Tamamen uyumlu | ~%10-15 (bellek + döngü) |
| C2 | Tüm pixel döngülerini `std::execution::par_unseq` ile paralelleştir | ✅ CPU'da ana kazanç buradan | ✅ Pre/post-processing hızlanır | 4-8× on 8+ core |
| C3 | SDL yolunda float-path shortcut (viewport'ta pack/unpack bypass) | ✅ Uyumlu | ✅ Uyumlu | ~%20 SDL yolu |
| C4 | OptiX yolunda `oidnNewSharedBuffer` + CUDA device pointer | ⚠️ **Conditional** | ✅ En büyük kazanç | 6× transfer → 0 |

**C4'ün CPU fallback güvenliği:**
OIDN cihazı CPU modundaysa `newSharedBuffer` ile CUDA device pointer geçilemez. Bu yüzden C4 runtime-conditional olacak:

```cpp
const bool canZeroCopy =
    oidnDevice.get<oidn::DeviceType>(/*…*/) == oidn::DeviceType::CUDA
    && backendIsOptix
    && cudaPointersValid;

if (canZeroCopy) {
    // newSharedBuffer(d_accumulation_float4, …)  — zero copy
} else {
    // Mevcut download + newBuffer yolu korunur
}
```

Böylece CUDA'sız makinada **hiçbir kod yolu değişmez**; sadece C1/C2/C3 kazançları devreye girer, C4 pas geçilir.

---

## 4. Uygulama Sırası

**Faz 1 — Risksiz kazanımlar** (CPU + CUDA her ikisine yarar)
1. C1: `oidnOriginalData` siliniyor, blend `frame.color`'dan
2. C2: 5 döngünün hepsi `par_unseq`
3. C3: `applyOIDNDenoisingToCPUAccumulation` zaten float → direkt çalışır; SDL yolunda float buffer varsa kısayol ekle

_Test:_ non-CUDA makinada benchmark, kalite parity (bit-exact beklentisi: evet, C1'de blend matematiği aynı; C2 unseq'de float toplam sırası değişebilir ama her pixel bağımsız olduğundan bit-exact).

**Faz 2 — OptiX zero-copy** (yalnız CUDA)
4. C4: `initOIDN()`'da CUDA context'i OptiX ile aynı device'da aç, `oidnNewSharedBuffer` ile `d_accumulation_float4` / `d_denoiser_albedo` / `d_denoiser_normal` paylaş.
5. `OptixBackend::getDenoiserFrame` yerine Renderer'a CUDA pointer veren yeni yol (`DenoiserFrameDataGPU`). Eski yol CPU fallback için kalır.

_Test matrisi:_
- CUDA'lı makina, OptiX viewport → C4 aktif, beklenen ~6× transfer tasarrufu
- CUDA'lı makina, CPU render yolu → C4 inaktif, C1/C2 aktif
- CUDA-olmayan makina → OIDN CPU, `canZeroCopy=false`, tüm faz1 kazançları aktif, davranış mevcut kodla aynı

---

## 5. Risk Notları

- **C2 (par_unseq):** OIDN filter.execute() tek thread'li çağrılmalı (mevcut `oidnMutex` zaten koruyor); pre/post pixel döngüleri OIDN-dışı olduğu için güvenli.
- **C4 device matching:** OIDN CUDA device ID'si OptiX'inkiyle eşleşmezse runtime error olur → try/catch ile fallback yola düş.
- **C4 buffer lifetime:** OIDN shared buffer, CUDA pointer'ı free edilmeden önce release edilmeli; `resetBuffers`/resize'da OIDN filter'ı invalidate et.
- **Kalite parity:** C1'den sonra bit-exact olmalı. Regresyon için tek-frame render + SSIM ≥ 0.999 check.

---

## 6. İlerleme

- [x] Faz 1 / C1 — `oidnOriginalData` silindi, blend `frame.color`'dan
- [x] Faz 1 / C2 — Paralel döngüler (`std::execution::par_unseq`)
- [x] Faz 1 / C3 — Primary viewport yolu zaten float-path; SDL yalnızca fallback için
- [x] Faz 1 build + çalışma doğrulaması (CUDA makina)
- [x] Faz 2 / C4 — OIDN CUDA direct-pointer path (setImage + device pointer + stride)
- [x] Faz 2 build doğrulaması
- [ ] Faz 2 benchmark (CUDA makina, ms/frame ölçüm)
- [ ] Regresyon: SSIM parity testi (CPU vs GPU path görsel eşitlik)

---

## 7. Yapılan İş — Dosya Bazlı Değişiklik Kaydı

### Faz 1 (CPU + CUDA her ikisine fayda)

**[Renderer.h](raytrac_sdl2/source/include/Renderer.h)**
- `oidnColorData` ve `oidnOriginalData` `std::vector<float>` member'ları silindi (iki full-image kopya kaldırıldı)
- `applyOIDNDenoisingGPU` forward declaration'ı + `Backend::DenoiserFrameDataGPU` fwd-decl eklendi
- OIDN GPU-path binding cache üyeleri: `oidnCudaInitialized`, `oidnCudaOrdinal`, `oidnCudaStream`, `oidnGpuCachedColor/Albedo/Normal`

**[Renderer.cpp](raytrac_sdl2/source/src/Render/Renderer.cpp)**
- `<algorithm>` ve `<cuda_runtime.h>` include'ları eklendi
- `applyOIDNDenoising(SDL_Surface*)`: sRGB→linear decode ve linear→sRGB encode+pack döngüleri `std::for_each_n(std::execution::par_unseq, …)` — pointer-range iterator
- `applyOIDNDenoising(OIDNFrameData&, …)` merkezi implementasyon:
  - Input artık `oidnColorBuffer.write(0, size, frame.color)` ile **doğrudan** yükleniyor (ara staging yok)
  - Output `oidnOutputBuffer.read(0, size, output.data())` ile **doğrudan** indiriliyor
  - Blend in-place, `frame.color`'u original referans alarak paralel
- `applyOIDNDenoisingToCPUAccumulation`: Y-flip + Vec4→float3 linearize döngüsü paralel

### Faz 2 (CUDA zero-copy yolu)

**[OptixWrapper.h](raytrac_sdl2/source/include/OptixWrapper.h)**
- Yeni getter'lar: `getDenoiserAlbedoDevicePtr()`, `getDenoiserNormalDevicePtr()` (mevcut `getAccumulationDevicePtr`'a ek)

**[Backend/IBackend.h](raytrac_sdl2/source/include/Backend/IBackend.h)**
- `DenoiserFrameDataGPU` struct'ı: device pointer'lar (void*), `pixelByteStride`, `rowByteStride`, `cudaStream`, `cudaDeviceOrdinal`
- `virtual bool getDenoiserFrameGPU(DenoiserFrameDataGPU&)` default `false` — non-OptiX backend'ler için güvenli

**[Backend/OptixBackend.h](raytrac_sdl2/source/include/Backend/OptixBackend.h) + [OptixBackend.cpp](raytrac_sdl2/source/src/Backend/OptixBackend.cpp)**
- `getDenoiserFrameGPU` override: OptiX'in `d_accumulation_float4` / `d_denoiser_albedo` / `d_denoiser_normal` device pointer'ları, `pixelByteStride=sizeof(float4)`, `rowByteStride=W*sizeof(float4)`, OptiX CUDA stream

**[Renderer.cpp `applyOIDNDenoisingGPU`](raytrac_sdl2/source/src/Render/Renderer.cpp)** — yeni metod:
- `cudaGetDevice` ile OptiX'in bulunduğu device ordinal'i çözer
- `oidn::newCUDADevice(ord, optixStream)` ile OIDN'i OptiX device + stream'ine bağlar
- Device veya stream değiştiyse OIDN cihazını yeniden kurar, cached filter/buffer'ları invalidate eder
- `setImage("color"/"albedo"/"normal", devPtr, Format::Float3, W, H, 0, 16, W*16)` — OptiX float4 buffer'larını **kopyasız** okur (ilk 3 float RGB, 4'üncü padding)
- Output OIDN-owned CUDA buffer'a yazılır, execute sonrası `oidnOutputBuffer.read` ile host'a inilir
- Pointer mismatch (OptiX resize) algılanınca filter re-commit edilir
- `blend < 0.999` veya herhangi bir başarısızlıkta `false` döner → caller host yoluna düşer

**[Main.cpp viewport denoise dispatcher](raytrac_sdl2/source/src/Core/Main.cpp)**
- GPU-first dispatch: önce `getDenoiserFrameGPU` + `applyOIDNDenoisingGPU`
- Başarısızsa / kısmi blend durumunda mevcut `getDenoiserFrame` + `applyOIDNDenoising` host yolu
- Backend hiçbir denoiser frame sağlamıyorsa SDL direct-surface fallback (eski davranış korundu)

---

## 8. Kazanç Özeti

| Senaryo | Önce | Sonra |
|---|---|---|
| CUDA + OptiX viewport, blend=1.0 | 3×D→H + 3×H→D + 1×D→H = **~6 full-image PCIe** + 4 seri döngü | **1 full-image PCIe (output)** + paralel döngü |
| CUDA + kısmi blend | 6 PCIe + staging copy'ler | 6 PCIe (host yolu) + **0 staging copy** + paralel döngüler |
| CUDA-olmayan, OIDN CPU | 4 seri döngü + 2 staging kopya | Paralel döngüler + **0 staging kopya** |
| CPU render sonrası denoise | Tek-thread linearize + 4 seri döngü | Paralel linearize + paralel döngüler |

---

## 9. Öğrenilenler / Tuzaklar

- **`std::for_each_n(policy, size_t(0), N, f)` çalışmaz** — MSVC C2938/C2794. İlk argüman iterator olmalı. Çözüm: pointer-range üzerinden iterate edip lambda içinde `&elem - base` ile index çıkar.
- **OIDN 2.3 `setImage` overload'u** `void* devPtr` + `pixelByteStride` + `rowByteStride` alıyor → OptiX float4 buffer'ları formatsız paylaşılabiliyor.
- **OIDN CUDA device, OptiX'in CUDA context'iyle uyumlu olmalı**. `oidn::newCUDADevice(deviceId, stream)` ile aynı device + aynı stream → runtime hatası yok.
- **Pointer-bazlı invalidation** yeterli: OptiX resize sonrası `cudaMalloc` yeni pointer verir; `applyOIDNDenoisingGPU` bu değişimi algılayıp filter'ı re-commit eder.
- **CPU fallback korundu**: GPU path runtime'da başarısız olursa `false` döner, caller otomatik host yoluna düşer. CUDA-olmayan makinada `getDenoiserFrameGPU` hiç çağrılmaz (non-OptiX backend default false döner).
- **Backend transition (OptiX→Vulkan) access violation**: GPU path `oidn::newCUDADevice(ord, optixStream)` ile bağlıyken OptiX kapatılınca stream geçersiz oluyor; sonraki host-path `execute()` nvcuda64.dll içinde crash ediyor. Host path girişinde `oidnCudaInitialized==true` ise device+filter+buffer teardown edilip `initOIDN()` ile self-managed device yeniden kuruluyor. Bu teardown geçiş başına bir kez olur; rutin frame'lerde skip edilir.
