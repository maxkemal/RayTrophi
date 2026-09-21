# Realtime viewport GPU culling'i HİÇ açmıyordu — ÖLÇÜLDÜ ve düzeltildi

> **Durum:** AKTİF — kök neden canlı IPC ile ölçüldü (2026-09-08), düzeltme
> YAZILDI/DERLENMEDİ. Kaynak denetimi `audit_raster_gpu_instancing.py` geçti.
> Görsel doğruluk ve kare süresi kazancı **doğrulanmadı**.

## Ölçüm — 1000 foliage + terrain + river, canlı sahne

`viewport.frame_telemetry`, kamera hareket ederken 60 kare boyunca:

```
gpu_culling            false        global_instance_buffer  false
cull_mesh_count        0            draw_calls              37
total_instances        1034         full_instances          722 / proxy 312
visible_triangles      45.920.711   full 45.890.759 / proxy 29.952
scatter_triangle_target 29.622.857  -> hedefin %55 ÜZERİNDE
resource_drains        +59 / 60 kare  = kare başına 0,98 TAM drenaj
frame_ms               1,61         present_ms 0,48   cpu_record_ms 0,15
display_loop_period_ms 60,49        (≈16 fps)
```

★ `frame_ms` 1,6 ms ama döngü periyodu 60 ms. Yine
[main loop cost](project_main_loop_cost_is_unmeasured) şekli: backend'in kendi
muhasebesi karenin %3'ü. Kalanın büyük kısmı **drenajda**, çünkü
`drainInteractiveViewportInFlight()` ölçülen bölgenin *dışında* çağrılıyor.

## Kök neden: override, taban sınıfın kuyruğunu KOPYALAMIŞ ve bir çağrıyı düşürmüş

`m_rasterUseGlobalInstBuffer`'ı true yapan tek yer `rebuildRasterInstanceLayout()`,
ve o yalnız taban sınıfın `buildRasterGeometryImpl` kuyruğundan çağrılıyordu.
`VulkanViewportBackend::buildRasterGeometry` o kuyruğun bir **kopyasını**
taşıyordu ve kopyada layout çağrısı yoktu — doğrudan per-mesh yüklemeye gidiyordu.

Tek eksik çağrı **iki maliyeti birden** doğuruyor:

1. **GPU culling ve scatter LOD proxy'leri hiç devreye girmiyor.** 45,9M üçgen
   her kare gönderiliyor; proxy'ler bunun 30 binini taşıyor. Frustum culling
   yok (CPU tarafı zaten `kRasterFrustumCullingEnabled = false` ile derleme
   dışı), LOD üçgen bütçesi uygulanmıyor.
2. **Kare başına tam boru hattı drenajı.** Drenajsız doğrudan yazma
   (`RasterGlobalInstanceBuffer::write`, kaynakta *"NO DRAIN, NO STAGING"*)
   aynı bayrağın arkasında. Bayrak kapalı olunca `setRasterVisibleInstances`
   per-mesh yola düşüyor ve o yolun ilk ifadesi `drainInteractiveViewportInFlight()`.
   Ölçülen: **0,98 drenaj/kare** — CPU/GPU örtüşmesi sıfır.

★★★★ **Tüketici tarafı zaten tamamdı.** Cull compute kurulumu, sıkıştırılmış
buffer bağlama (`cullOutBase`), gölge geçişi — hepsi yazılmış ve bu yol için
tasarlanmış. Eksik olan tek şey **üreticiyi çağırmaktı**.

★★★★ **Bu, aynı tuzağın ÜÇÜNCÜ kurbanı.** `ensureInteractiveViewportResourcesImpl`
içindeki yorum atmosfer LUT'u için aynı şeyi zaten yazıyor: *"TABAN SINIFTAKİ
aynı çağrı burada TEKRARLANMALI çünkü bu fonksiyon onu OVERRIDE ediyor."*
Kural bu partide **koda** çevrildi: iki kuyruk artık tek gövde
(`refreshRasterInstanceLayout()`), ve denetim script'i ikinci bir kopyayı
reddediyor.

★ Ve **görsel ipucu yoktu**: culling kapalıyken sahne DOĞRU çizilir, yalnızca
yavaştır. Bir arızanın resmi yoksa geriye tek kanıt sayılar kalır.

## Düzeltme

| Ne | Nerede |
|---|---|
| Ortak kuyruk | `VulkanBackendAdapter::refreshRasterInstanceLayout()` |
| Taban + override ikisi de onu çağırır | `VulkanBackend_Raster.cpp`, `VulkanViewportBackend.cpp` |
| A/B kolu | `viewport.set_raster_gpu_instancing {enabled}` (+ Python, RtApi) |
| Ölçüm | panelde "Submitted: N.NNM tris in N draws" + culling durumu |

Kol **kapatılabilir bırakıldı**: bu yol bu backend'de hiç koşmadı, ve kapatılamayan
bir düzeltme kendisini yargılayacak ölçümü de öldürür. Kapatmak Scene Log'a
UYARI yazar — yavaş-ama-doğru bir viewport'un aylar sonra başka izi olmaz.

★ Anahtar panele **konmadı**, ölçüm kondu. Arızalı yolu menüden seçilebilir
bırakmak kuralı tercihe çevirirdi; aynı gerekçe `viewport.set_scene_load_guard`
için de yazılmıştı.

## ⚠ Açık risk — bilerek bırakıldı

Global buffer açıkken `uploadVisibleRasterInstances` **erken dönüyor**. Yani GPU
cull kurulumu başarısız olursa sahne culling'siz VE proxy'siz çizilir — yani
bugünkünden **kötü**. Kaynak bunu zaten raporluyor (`gpu_culling=false` +
Scene Log uyarısı), ama kabul testinde ilk bakılacak şey budur.

## Sıradaki

Kontrol listesi: [NEXT_BUILD_CHECKS.md](NEXT_BUILD_CHECKS.md) madde 1-3.
Kazanç ölçüldükten sonra sıradaki aday, karenin geri kalanının **hâlâ**
ölçülmemiş olması: `frame_ms` 1,6 / döngü 60 ms.

## İlgili

- [RAYFUSION_FOLIAGE_FRAME_COST.md](RAYFUSION_FOLIAGE_FRAME_COST.md) — aynı sahnedeki CPU kök nedeni
- [REALTIME_CAMERA_MOTION_PERF.md](REALTIME_CAMERA_MOTION_PERF.md) — aynı hata sınıfının önceki turu
