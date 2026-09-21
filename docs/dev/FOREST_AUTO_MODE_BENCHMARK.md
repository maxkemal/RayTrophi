# Yoğun orman/dağ — Auto mod karşılaştırması

2026-09-08, açık kullanıcı sahnesinde IPC ölçümü. GPU: NVIDIA GeForce RTX 3060;
Windows sürücü sürümü 32.0.15.9186. Viewport 1680×945. Derleme yapılmadı.
Kullanıcı toplam instance-expanded geometriyi yaklaşık 5 milyar üçgen,
flat geometriyi 8,7 milyon üçgen olarak bildirdi; bu iki toplam bu testte
bağımsız yeniden sayılmadı. Canlı sayılar: 50.074 instance, 76 viewport BLAS.

## Yöntem ve ölçüm birimi

Auto açık; camera position/target sabit. Her modda 5 ısınma yenilemesi,
ardından 12 ölçüm yenilemesi. Önbellekteki karenin hızını ölçmemek için
FOV aynı başlangıç değeri ve +0.0001 derece arasında değiştirildi. Capture
kapalı; sahne geometri/materyal verileri değiştirilmedi.

Başlangıç Material + three_point, bounce kapalıydı. RayFusion turu Material
+ scene lighting + traced + bounce açık olarak ölçüldü; gerçek producer,
world IBL ve bounce_active değerleri doğrulandı. RT, Vulkan Rendered yoludur.

Süreler `loop.viewport_render.total_ms` farkıdır: CPU'dan gözlenen backend
işi ve bu kapsam içindeki beklemeler. **GPU timestamp veya interaktif FPS
değildir.** Rasterda 12 yeni gönderim ve 12 ek cached/presentation çağrısı,
RT'de 48 backend çağrısı vardı. Rasterdaki son ~1 ms telemetri cached-frame
işiydi; gerçek çizim süresi olarak kullanılmadı.

| Mod | Yenileme başına ortalama toplam backend işi | Yenilemede yapılan iş | Auto görünür üçgen |
|---|---:|---|---:|
| Solid | 16,15 ms | 1 yeni raster gönderimi + sunum/cached çağrı | 91,11–91,96 milyon |
| RayFusion yolu | 15,17 ms | 1 yeni raster gönderimi + sunum/cached çağrı | 38,51–39,15 milyon |
| Vulkan RT | 154,13 ms | 4 backend çağrısı; sonraki okumada 3 örnek | Raster sayacı bu modda stale; kullanılmadı |

RT'de her yenilemenin son `loop.viewport_render.last_ms` değerlerinin medyanı
60,13 ms idi. Bu, son ilerleme çağrısı maliyetidir; tüm RT kareleri için
sabit süre veya yakınsamış görüntü FPS'i değildir. Auto'nun raster modlarında
farklı geometri yükleri seçmesi nedeniyle Solid/RayFusion sürelerinden shader
maliyeti farkı çıkarılamaz. RT ise aynı miktarda iş/örnek üretmiyor.

Ham toplamlar: Solid 193,8142 ms / 12 yenileme; RayFusion 182,0283 ms / 12;
RT 1849,5580 ms / 12. UI/IPC beklemelerini içeren loop.frame değerleri FPS'e
çevrilmedi. Bu pencerede loop.present/throttle sayaçları ilerlemedi; gerçek
ekrana sunum temposu ölçülmüş sayılmaz.

## RayFusion kapsamı ve AS

Probe alanı hâlâ 4×2×4, spacing=3, minimum=(-2,-1,-2), follow_camera=false.
Kamera yaklaşık (105,30,-111); alan x/z [-6,6), y [-3,3) ile sınırlı.
Dolayısıyla bu test **ormanın tamamının GI ile kaplandığı bir RayFusion kalite
testi değildir**. Alan dışındaki yüzeylerde fallback sürer. Tur sonunda 32/32
valid, pending=0 ve bounce_active=true. Son probe partisi trace_ms=0,2907;
bu bütün alanın veya her karenin tracing maliyeti değildir.

Viewport AS: 116.124.160 bayt (~110,7 MiB), ilk kayıtlı build 232,84 ms.
Ölçüm turlarında build sayısı artmadı. RayFusion turunda son bounce_prepare_ms
5,0771 idi; CPU hazırlığı ayrı bir optimizasyon adayıdır. Sayılar GPU timer
olarak yorumlanmaz.

## Elenen ölçüm ve araç sınırı

Ek doğrulama için `viewport.render_frames(count=4)` denendi. Bu yardımcı,
`Renderer::render_progressive_pass` çağrılarını sayıyor; canlı GPU örnek
ilerlemesini saymıyor. Sonuçlarda 406 ms/geçiş ardından 0,001 ms/geçiş ve
GPU sample sayısında farklı ilerleme görüldü. Bu ek turun **hiçbir zamanı
mod karşılaştırmasında kullanılmadı**. `getMillisecondsPerSample()` Vulkan
backend'de kaynakta sabit 0 döndüğü için o alan da ölçüm değildir.

## Sonuç ve restorasyon

Bu test, cluster/LOD çalışması için somut raster yükünü gösteriyor: Auto'da
bile onlarca milyon görünür üçgen. Piksel-altı RT devrinin daha ucuz olduğunu
kanıtlamıyor. Sonraki deney sabit seçilmiş geometri, aynı görünürlük çıktısı
ve gerçek GPU pass zamanlarıyla raster/RT karşılaştırması olmalı. Alpha ve
transmission maliyetleri bu turda ayrı ayrı kapatılıp ölçülmedi.

Material, Auto, three_point, traced açık, bounce kapalı, capture kapalı ve
özgün FOV geri yüklendi; son salt okunur sorguyla doğrulandı.

Ham kayıtlar: `tmp/forest_benchmark_initial.json`,
`tmp/forest_benchmark_results.json`, `tmp/forest_benchmark_summary.json`,
`tmp/forest_benchmark_restored.json`. Elenen ek tur:
`tmp/forest_benchmark_rt_batches.json`. Tekrar script'i:
`tmp/forest_benchmark.py` (mevcut initial kaydına bağlı oturum testi).
