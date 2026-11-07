# GPU yüzey level set'i — sıvı/kar yüzeyini simülasyon cihazında kurmak

> **Durum:** AKTİF — 2026-11-04. Y1 + Y2 YAZILDI, derlenmedi. Y3 (raster) ayrı bir iş
> değil: raster modlarında GPU'nun kurduğu ızgara indirilip mevcut NanoVDB yoluna veriliyor.

## Devir notu (2026-11-04 öğleden sonra, composition partisi)

- İlk derlemeden sonra kullanıcının sahnesinde GPU yolu hiç çalışmadı. Domain'de Output
  listesinden bir madde materyali bağlıydı ve GPU yolu bunu reddediyordu. Gerçek sahnelerin
  çoğu böyle, yani ret yolu işe yaramaz kılıyordu.
- `sim_surface_composition.comp` eklendi, ret kaldırıldı. Volume köpük ve anizotropi hâlâ
  CPU'ya düşüyor.
- RayFusion'ın madde materyalini almaması derlemeden sonra da sürüyor; kök bulunamadı.
  `render.volume_slots`'a cihaz başına materyal/composition teşhis alanları eklendi.
  Sıradaki derlemede ilk iş bunları okumak (bkz. NEXT_BUILD_CHECKS).

## Devir notu (2026-11-04, Y1+Y2 partisi)

- **Yazılan:**
  - 10 compute kernel `sim_surface_*.comp` (+ `sim_surface_common.glsl`), kernel tablosunda.
  - Host builder `FluidSurfaceGpu.{h,cpp}`.
  - `scene_data.h` render_sync'te önce GPU'yu dener, olmazsa CPU'ya düşer.
  - RT shader'da canlı yoğun yüzey: `surface_dense_sampling.glsl`; `volume_hit_sampling` iki
    örnekleyici ve `volume_closesthit` SDF hakemi.
  - Backend'de `liveDenseSurface` dalı; `VDBVolumeData::surface_live_uvw_address`.
  - IPC `fluid.surface_parity`, `fluid.set_param surface_gpu`, `fluid.get surface_route`;
    panelde "Build Surface on GPU" ve "Built on".
- **Rota (`LevelSetStats::route`):**
  - 0 CPU.
  - 1 GPU, proxy host'a indirilir: raster/RayFusion, OptiX, CPU, m>1.
  - 2 GPU canlı: Vulkan RT cihaz adresinden okur, hiçbir şey geri gelmez.
- **GPU yolunun reddettikleri** (CPU'ya düşer, `surface_gpu_status` nedenini söyler):
  anizotropik kernel, madde başına materyal (composition), volume köpük, Vulkan compute yok,
  eksik `.spv`.
- **Tasarımdan sapma:** girdiler adımın cihaz tamponlarından DEĞİL, host parçacıklarından
  yükleniyor (~28 B/parçacık, 100 k'da 2.8 MB). Nedeni: adımdan sonra host parçacıkları
  değiştiriyor (açık sınırda silme, gözenek alışverişi, etiketler, sis sürüklenmesi), yani
  cihaz kopyası çizilen durum değil.
- **ABI:** `GpuVDBVolume`'a dokunulmadı (OptiX PTX'i bozulmasın diye). Canlı yüzey gazın
  `dense_*` paket alanlarını kullanıyor; backend onu `source_type == 4` ile ayırıyor.
  Proxy'nin yerleşimi `uvw_*` alanlarından geliyor (canlı rota yalnız m=1).
- **Sıradaki:** derleme → `rt_test_surface_gpu_ipc.py` → gözle Rendered karşılaştırması.
  Kontrol listesi `NEXT_BUILD_CHECKS.md`'de.

## Neden

2026-11-04 sabah ölçüldü. Kare başına değerler; sahne hareketli kar, 100 k parçacık,
kapalı 5 m domain, h 0.1, m=1. Ayrıntı: [MPM_UYUYAN_KAROLAR.md](MPM_UYUYAN_KAROLAR.md).

| Kalem | ms/kare |
|---|---|
| Solver `total_ms` (GPU 66) | ~128 |
| `sim.render.surface.level_set` (CPU, OMP) | **184** |
| — `surface_anisotropy_ms`: merkez yumuşatma (`position_smoothing` 0.9; anizotropi KAPALI) | **124** |
| — `surface_schedule_ms` (iş karoları) | 16 |
| — `surface_gather_ms` (hücre başına SDF) | 9 |
| — `surface_bin_ms` | 3 |
| — `surface_smooth_ms` | 1 |
| `sim.render.surface.material_coordinates` (uvw, CPU) | 51 |
| `loop.frame` | ~420 |

Merkez yumuşatmada her parçacık 2.5 h yarıçapında ~500 komşu tarıyor (8 ppc). Bu
parçacık başına bağımsız bir toplama, yani GPU'nun en rahat yaptığı iş.

Kullanıcı kuralı: **mimari mümkün olduğunca CPU↔GPU transferi yapmamalı.**

## Mevcut yol (değişecek)

```
GPU step → parçacıkları HOST'a indir (18 MB) → render_sync (ana thread):
  CPU level set → CPU uvw → CPU composition → yoğunluk vekili (0.5 − 0.5·φ/w)
  → OpenVDB/NanoVDB kur → İKİ cihaza da yükle (RT + raster)
```

## Mimari gerçekler (kodda doğrulandı)

1. **Simülasyon compute'u ile Rendered (Vulkan RT) aynı VkDevice'ı paylaşıyor.** Gaz bunu
   zaten kullanıyor: `gasGpuFieldView` → `mgr.setLiveDenseGpuFields` → RT shader
   `sampleDenseGasFloat` (buffer_reference). Volume type 4, source type 5. Gaz
   yoğunluğu hiç CPU'ya inmeden çiziliyor. **Emsal var.**
2. **Solid/Material raster viewport AYRI bir VkDevice** (`VulkanViewportBackend`,
   `initializeViewportBackendIfAvailable`). Ayrılık kasıtlı: TDR kurtarma kapısı yalnız
   o cihazı yeniden açıyor. Simülasyon adresleri orada **sıfır okunur.** Gaz için çözüm
   sürüm kapılı bir host aynası (`g_dense_gas_host_mirror_needed`): yalnız raster ikinci
   tüketici olarak gerçekten varken indirir.
3. Raster yüzeyi kendi cihazına yüklenen NanoVDB tablosundan okuyor
   (`material_preview_sdf_surface.frag`, `MaterialPreviewSdfSurface.cpp`).

## Aşamalar

### Y1 — GPU üretici (simülasyon cihazında, kalıcı tamponlar)

Mevcut CPU algoritmasının birebir aynısı. Aynı kernel ağırlıkları, aynı sınırlar, aynı
kütle yarıçapı; parite test edilebilir olsun diye.

| Kernel | İş | CPU karşılığı |
|---|---|---|
| `sim_surface_bin_count` / prefix / `scatter` | seçili, sonlu, kütlesi > 0.02 parçacık → kutu (bin_voxel = max(R, h)) | aşama 1 |
| `sim_surface_centres` | ağırlıklı ortalama merkez, λ = position_smoothing; nmin altında ham merkez. Anizotropi açıksa kovaryans + analitik özdeğer (Sym3) | aşama 1b |
| `sim_surface_rebin` | yumuşamış merkezleri destek (R·stretch) kutularına | `SurfaceParticleBins` |
| `sim_surface_gather` | hücre başına φ, `support_confidence`; doluluk maskeli yoğun ızgara (iş karoları yerine) | aşama 2 |
| `sim_surface_smooth` | Laplace süpürmeleri, güven ağırlıklı (ping-pong) | `smoothSurfaceTiles` |
| `sim_surface_proxy` | yoğunluk vekili 0.5 − 0.5·φ/w, aktif kutu min/max | render_sync içindeki döngü |
| `sim_surface_uvw` | malzeme koordinatı toplama (aynı kernel) | `buildMaterialCoordinateGrid` |
| `sim_surface_composition` | madde/malzeme slot karışımı | `buildCompositionGrid` |

**Girdi — transfer yok (ana yol):**
- Matter adımı bittiğinde parçacık konumları zaten cihazda (`primary.fluid_positions`).
- Yüzey adımın SONUNDA, aynı cihaz tamponundan kurulur. Domain'in kimliği ve sürümü
  eşleşmelidir.
- Durum sürümü tutmuyorsa yedek yola düşülür ve yalnız konumlar yüklenir (100 k × 12 B
  = 1.2 MB). Bu durumlar: zaman çizelgesinde gezinme, önbellekten/diskten geri yükleme,
  duraklatılmış karede malzeme değişikliği.
- Seçim (`FluidViewSelection::keeps`: madde etiketi, etiket rotası, faz) ilk sürümde
  host'ta parçacık başına 1 bayt maske olarak hesaplanıp yüklenir (0.1 MB). Sonra bir
  cihaz tablosuna taşınır.

**Çıktı:** domain başına kalıcı cihaz tamponları (yoğunluk vekili, uvw residual,
composition, aktif kutu, sürüm). CPU'daki `domain_sdf_buffers` bu yolda **dolmaz.**

**Kabul:** aynı parçacıklardan CPU ve GPU φ farkı `max ≤ 1e-3·h`; 50³ ve m=2'de yüzey
hücre sayısı ±%0.5; `level_set` scope'u kaybolur; yeni `sim.render.surface.gpu_ms`
(zaman damgası) < 5 ms.

### Y2 — Rendered (Vulkan RT) doğrudan okur: transfer SIFIR

- Yeni yayın kaydı: `SimulationSurfaceGpuView`. Gazdakinin ikizi; yoğunluk, uvw ve
  composition adresleri, çözünürlük, origin, voxel, sürüm ve aktif kutu taşır.
- `VDBVolumeManager`'a canlı yoğun **yüzey** türü eklenir. RT yüzey shader'ı
  (`volume_surface_shading.glsl`, `volume_hit_sampling.glsl`) NanoVDB yerine
  `sampleDense*` ile okur. Yoğun örnekleyici zaten var.
- Yüzeyde boş alan atlama için gazın majorant blok listesi aynı şekilde kurulur. Aktif
  kutu AABB'yi daraltır, mevcut `sdf_occupied` kuralı korunur.
- OptiX dondurulmuş: NanoVDB yolunda kalır. Bu durumda indirme gerekir, Y3 ile aynı
  host aynası kullanılır.

### Y3 — Raster viewport (ikinci VkDevice)

Seçenekler:
- **Önerilen:** gazdaki host aynası kuralının aynısı. Sürüm kapılı olur ve yalnız raster
  gerçekten tüketici olduğunda çalışır. Çıktı ızgarası küçük: m=1'de vekil 0.5 MB, uvw
  1.5 MB, composition ~1 MB. m=2'de ×8. Bugün solver kare başına 37.7 MB yüklüyor ve
  18 MB indiriyor; bu ayna onun %5'i.
- **Reddedilen:** `VK_KHR_external_memory_win32` ile iki cihaz arasında doğrudan
  paylaşım. Gerçek sıfır kopya olurdu, ama iki cihazın bellek ömrünü birbirine bağlar.
  Raster cihazı TDR'de tek başına yeniden açılabiliyor ve bu ayrılığın nedeni tam olarak
  o. Kuyruklar arası sahiplik devri ve çift tampon da gerekir. Ölçülen ayna maliyeti
  sorun olursa yeniden ele alınır.

### Y4 — Sonra (bu notun kapsamı dışı)

- Solver her kare 37.7 MB yüklüyor, 18 MB indiriyor. Parçacık durumu kare başında
  yeniden yükleniyor. Bu, transfer ilkesinin asıl büyük ihlali, ama ayrı bir iş.
- Seçim maskesinin cihaz tablosuna taşınması.
- `fluid.set_param visible=false` yüzeyi kapatmıyor. Değer domain kutusunun nesnesine
  (`obj->visible`) yazılıyor, render_sync kapısı ise `system.visible`'a bakıyor. Yani
  görünmez bir domain her kare yüzey kurmaya devam ediyor.

## Riskler

- **Üretici ≠ tüketici:** yüzey adımın sonunda kurulursa, render_sync'in sürüm kapısı
  bunun yerine geçmeli. Aksi halde duraklatılmış karede malzeme değişikliği yüzeye
  ulaşmaz (eski "duraklatılmış kare" hatası, `sdf_sig` yorumu).
- Sıra: indirme ile yüzey dispatch'i aynı submission'da olmalı. Ayrı bir
  `synchronize()` eklemek zinciri uzatır.
- ★ **Sessizce makul görünen sonuç:** GPU φ'si yarım voxel kaymış ya da yumuşatma tek
  süpürmede kalmış olabilir. Yüzey "biraz farklı ama güzel" görünür. Bu yüzden Y1'in
  kabulü sayısal paritedir, gözle bakmak yetmez.
