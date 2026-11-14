# Player CUDA/OptiX sınırı — 2026-11-13

Bu parti kalan 12 CUDA/OptiX gören aday çeviri birimini ele alır. Sonuç bir
include/backend kaynak ayrımıdır; çalışan veya linklenmiş Player kanıtı değildir.
C++ proje derlemesi, uygulama açma ve canlı/sayısal kabul yapılmadı.

## Kapsam ve ölçüm

| Başlangıçtaki aday | İşlem |
|---|---|
| `RtApi.cpp` | OptiX tanısı hedefe göre seçilen iki modüle taşındı. |
| `VulkanBackend.cpp` | CUDA/OIDN dış bellek köprüsü ayrıldı; Vulkan kaynakları ortak kaldı. |
| `ProjectManager.cpp` | Kullanılmayan OptiX include'ları kaldırıldı; CUDA beklemesi offline hook'a bağlandı. |
| `GasSimulator.cpp` | CPU hesabı ortak; CUDA/FFT/kaynak işlemleri ayrı modülde. |
| `GasVolume.cpp` | VDB yayıncısı ortak; CUDA doku yükleme/temizlik ayrıldı. |
| `TerrainManager.cpp` | Vulkan/CPU ortak; eski CUDA fallback ve OptiX foliage ayrı modüllerde. |
| `VDBVolumeManager.cpp` | Host NanoVDB ortak; CUDA kopyalama/temizlik ayrıldı. |
| `Renderer.cpp` | OptiX çağrıları SDK'sız `OfflineRenderBridge`, CUDA temizlik çağrıları `OfflineDeviceHooks` üzerinden. |
| `SceneSerializer.cpp` | Kullanılmayan OptiX include'ları kaldırıldı. |
| `OptixAccelManager.cpp` | Gerçek offline implementasyon; Player manifestinden dışlandı. |
| `OptixWrapper.cpp` | Gerçek offline implementasyon; Player manifestinden dışlandı. |
| `OptixWrapper_UvUpdate.cpp` | Gerçek offline implementasyon; Player manifestinden dışlandı. |

`scene_ui_nodeeditor.hpp` içindeki CUDA beklemesi de aynı hook'u kullanır.
`StylizeKernel.h` ortak vektör tiplerini `GpuInteropTypes.h` üzerinden alır.
Erosion skaler parametreleri `TerrainErosionGpuTypes.h` içinde tek kaynaklıdır;
eski `erosion_ops.cuh` CUDA tüketicilerine uyum başlığı olarak kalır.
CPU/CUDA'nın ortak kenar harmanlaması `TerrainErosionBoundary.h` içindedir.

| Ölçüt | Önce | Sonra |
|---|---:|---:|
| Player adayları | 547 | 552 |
| Yasaklı başlığa ulaşmayan adaylar | 514 | 530 |
| CUDA/OptiX gören adaylar | 12 | 0 |
| Embree | 6 | 6 |
| OIDN | 1 | 0 |
| Python | 16 | 16 |

Sekiz yeni Host modülü adaydır; üç gerçek OptiX dosyası dışlandı: net +5 aday.
OIDN 1 → 0, OptixWrapper'ın doğru offline sınıflandırılmasından gelir.
Baseline CUDA/OptiX=0, OIDN=0 olarak sıkılaştırıldı.

## Hedeflerin kaynak seçimi

Ortak kaynaklarla birlikte aşağıdaki her satırın yalnız bir tarafı linklenmeli.
Studio vcxproj/filters soldakileri `ClCompile`, sağdakileri `None` olarak kaydeder.
Gelecekteki Player projesi sağdakileri `ClCompile` seçmelidir.

| Studio | Player |
|---|---|
| `VulkanDenoiserCuda.cpp` | `VulkanDenoiserHost.cpp` |
| `RtApiOptixDiagnostics.cpp` | `RtApiOptixDiagnosticsHost.cpp` |
| `OfflineDeviceCuda.cpp` | `OfflineDeviceHost.cpp` |
| `OfflineRenderOptix.cpp` | `OfflineRenderHost.cpp` |
| `GasSimulatorCuda.cpp` | `GasSimulatorHost.cpp` |
| `GasVolumeCuda.cpp` | `GasVolumeHost.cpp` |
| `VDBVolumeCuda.cpp` | `VDBVolumeHost.cpp` |
| `TerrainManagerCuda.cpp` + `TerrainFoliageOptix.cpp` | `TerrainOfflineHost.cpp` |

Player gaz initialize yolu CUDA kullanılabilirliğini false görür ve mevcut CPU
fallback'ini seçer. Ulaşılamaması gereken doğrudan CUDA gas step'i sessiz başarı
yerine hata üretir. Vulkan terrain yolları aynıdır; Vulkan başarısızsa Player CPU
fallback'ini kullanır. VDB host verisi ve yaygın InstanceGroup yolu korunur; eski
OptiX foliage ve OIDN interop mevcut değildir. OptiX tanı API'si bağlam yoksa aynı
hatayı, bağlı Player bağlamında `available=false` ile başarılı sonuç döndürür.
Yeni kullanıcı işlemi eklenmedi; mevcut IPC/script imzaları değiştirilmedi.

## Yapılan doğrulamalar

- MSVC `/Zs`: 9 değişen ortak kaynak SDK'lı/CUDA'sız, 9 Studio adaptörü ve 8 Player
  adaptörü: **35/35 PASS**. Player tarafındaki **17** gerçek `/showIncludes`
  dökümünde CUDA/OptiX SDK başlığı yok. Kontrol komutlarından bu SDK include yolları
  çıkarıldı. MSBuild'in tlog seçenekleri ve proje ortam include yolları kullanıldı.
- İlk turdaki UI bekleme çağrısı, proxy/null karşılaştırması, taşınan terrain
  yardımcısı ve NanoVDB warning ayarı düzeltildi; etkilenen kontroller tekrar geçti.
- Taşınan lifecycle/hesap/tanı gövdeleri, dört erosion CUDA kuyruğu ve ortak skaler
  yapılar için **31 kaynak eşitliği kontrolü PASS**. Yorum/boşluk ve clang-format'ın
  bitişik string literal bölmeleri normalize edildi. Bu sayısal koşu değildir.
- Stylize ve erosion yapılarında **81 alan offset'i**, sizeof/alignof ve Stylize
  `KernelParams=148 B`: SDK'lı/CUDA'sız **2/2 `/Zs` PASS**.
- XML/proje/filter kaydı: **17 yeni modül** doğru öğe türünde ve tek kayıtlı; dört
  yeni ortak başlık kayıtlı. Include denetimi ve `git diff --check` PASS.

## Etkilenen CUDA shader'ları

Yalnız `erosion_kernels.cu` ve `StylizeKernel.cu` doğrulandı. CUDA 12.6, compute_50,
fast-math; Stylize için mevcut C++17 ayarı kullanıldı. İki PTX ptxas doğrulaması PASS.

- Erosion PTX önce/sonra **bayt bayt aynı**. `RayTrophiStudio/ptx/` ve
  `x64/Release/ptx/` kopyaları hash ile doğrulandı.
  SHA-256: `617a23fc5981421e41c9cb007e064f5a6e14e1bebfe98a457a21e75163f619c8`.
- Stylize PTX kodu kaynak yolundan türeyen anonim-namespace sembol hash'i normalize
  edilince aynı. Bu kernel exe'ye `CudaCompile` nesnesiyle bağlanır; doğrulama PTX'i
  kullanılmayan bir runtime varlığı olarak dağıtılmadı. Native `.cu` nesnesinin
  yeniden oluşturulması kullanıcının sonraki Studio derlemesine dahildir.
- Yeni shader yok; mevcut bulk erosion derleme satırı ve Stylize `CudaCompile`
  kaydı korundu. Shader alanları, binding/push boyutları ve GPU registry değişmedi.

## Sonraki kullanıcı kabulü

1. Önceki başlık/nesne yerleşimi partileriyle birlikte Studio TAM derleme/link;
   proje reload sonrası yeni adaptörlerin tek implementasyon olarak linklenmesi.
2. Vulkan RT + raster görünümü, VDB sequence ve canlı gas; CPU ve CUDA gas yolları,
   backend geçişi, proje yeniden açma ve kapanış.
3. Vulkan erosion ve CPU fallback; mevcut Studio OptiX foliage/saç/malzeme güncelleme.
   OIDN GPU-direct, resize ve backend değişiminde kaynak temizliği; Stylize görünümü.
4. Player hedefi kurulunca gerçek CUDA/OptiX kütüphanesiz link testi. Include=0;
   CPU/offline renderer, Embree, Python, editor servisi ve su bağlantılarının link
   ayrımının tamamlandığı anlamına gelmez. Ortak kare döngüsü/loader ayrımı sürer.
