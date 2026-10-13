# RT shader derleme yükü: kural ve aday listesi

> **Durum:** AKTİF — ilk adım (bulut taşıması → callable) CANLI 2026-10-13: 651 sn → ~3 dk

## Sorun

Vulkan RT pipeline'ı, bir shader değiştikten sonraki ilk Rendered
kullanımda sürücüde baştan derlenir.
- Süre birkaç gün içinde ~3 dk'dan ~11 dk'ya çıktı (651 sn, 2026-10-13).
- Disk önbelleği (`vk_pipeline_cache`) yalnız aynı shader'larla ikinci
  açılışı hızlandırır.
- **Yeni kurulan bir makinede ilk RT geçişi bu süreyi her zaman öder.**
  O sırada RayFusion çalışır; kullanıcı boş beklemez ama RT moduna
  geçemez. Bu bir kullanıcı sorunudur, geliştirici kolaylığı değil.

Neden büyüdü:
- `compile_shaders.ps1` her modülü `glslc -O` ile derler, yani her include
  çağrıldığı her aşamaya tamamen inline edilir.
- Sürücü her aşamayı ayrı derler. Aşamalar paralel derlenir, ama süre
  kod boyutuyla doğrusaldan hızlı büyür ve toplam süreyi EN BÜYÜK aşama
  belirler.
- Atmosfer partileri bulut taşımasını (≈1.4 MB SPIR-V) beş aşamaya
  kopyaladı.

2026-10-13 SPIR-V boyutları (bulut callable'a taşınmadan önce):

| Aşama | Boyut | Not |
|---|---|---|
| volume_closesthit | 5.6 MB | en büyük: toplam süreyi bu belirler |
| closesthit | 3.7 MB | |
| sphere_closesthit | 3.0 MB | **aynı kaynak** (`closesthit.rchit -DSPHERE_HIT=1`) |
| miss | 1.6 MB | neredeyse tamamı bulut |
| hair_closesthit | 1.6 MB | neredeyse tamamı bulut |
| raygen | 0.18 MB | bulut tabanı var, taşıma yok |

## Kural

1. **Ağır ve paylaşılan bir alt sistem tek aşamada derlenir.** Birden çok
   hit/miss aşamasının çağırdığı büyük bir hesap (bulut taşıması gibi)
   callable shader'a gider; çağıran aşamada yalnız ucuz kapı testi kalır.
2. **Taşıma görüntüyü değiştirmez.** Aynı fonksiyonlar, aynı girdiler.
   Daha önce aşamalar arasında farklı olan bir şey (örnek: bulutun gök
   ışığı kaynağı) bayrakla korunur; birleştirmek ayrı karardır.
3. **Önce ölç.** `render.rt_pipeline_status` → `stage_seconds` aşama
   başına sürücü süresini verir. Kod boyutu tahmindir; süre ölçümdür.
4. **Sıcak yolda dikkat.** Callable çağrısı ucuzdur ama bedava değildir.
   Her birincil isabette çalışan bir hesabı taşımadan önce pass süresini
   önce/sonra ölç.
5. **Yeni özellik eklerken:** bir include birden çok RT aşamasına
   giriyorsa ve büyükse, en baştan callable olarak yaz.

## Yapılan

- **Bulut taşıması → `cloud_callable.rcall`** (2026-10-13).
  - Giren kod: marcher, referans yürüyüşü, yağış, cirrus ve yüzeyin güneş
    NEE'sindeki bulut gölgesi.
  - Arayüz: `cloud_call_types.glsl` ve `cloud_call.glsl`.
  - Gök ışığı: miss'in gök kodu `rt_sky.glsl`'de, miss ve callable
    paylaşıyor.
  - Beklenen: miss ve hair_closesthit küçülür, closesthit /
    sphere_closesthit / volume_closesthit her biri ~1.4 MB azalır.
  - Canlı sonuç (kullanıcı, 2026-10-13): derleme 651 sn → ~3 dk;
    görüntü aynı; iki sahnede pass süresi aynı. `stage_seconds` henüz
    okunmadı (uygulama önbellekten açıldı).

- **sphere_closesthit → closesthit** (2026-10-13, yazıldı, derlenmedi).
  - Önce: aynı kaynak `-DSPHERE_HIT=1` ile ikinci kez derleniyordu
    (3.0 MB).
  - Şimdi: köpük küresi hit grubu `closesthit.spv`'yi kullanıyor; ikisini
    `gl_HitKindEXT == FOAM_SPHERE_HIT_KIND` ayırıyor. Normal merkezden
    hesaplanıyor; ortak tanım `foam_sphere.glsl`.

## Adaylar (sıra: kazanç / risk)

1. **volume_closesthit (5.6 MB).** Toplam süreyi bu belirliyor. İçinde
   `bsdf_scatter.glsl` (2172 satır), `material_program.glsl` (815),
   `PNanoVDB.h` (3635) var; aynıları closesthit'te de var. Önce
   `stage_seconds` ile neyin pahalı olduğunu ölç. Malzeme VM'i ya da BSDF
   örneklemesi callable'a giderse iki aşama birden küçülür. Ama bunlar
   sıcak yol (madde 4).
2. **Raygen'deki bulut tabanı + şimşek kanalı.** Küçük (0.18 MB toplam).
   Öncelik düşük; şimşek kodu başka bir ajanın alanında.
