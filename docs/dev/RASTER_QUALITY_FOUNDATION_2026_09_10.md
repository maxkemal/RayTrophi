# Raster kalite temeli — uygulanan ilk parti

Durum: kaynak değişiklikleri ve derlemesiz denetimler tamamlandı. Shader/C++
derlemesi, GPU doğrulaması, görüntü kabulü ve süre ölçümü **yapılmadı**.
Ana plan: [RASTER_FRAME_COST_2026_09_10.md](RASTER_FRAME_COST_2026_09_10.md).

## Uygulananlar

- Probe birincil ve bounce gölge ışınlarında `NoOpaqueEXT`: sahne BLAS'ları
  `OPAQUE` kurulduğundan sadece `OpaqueEXT` bayrağını yazmamak yeterli değildi.
  Alfa aday döngüsü artık BLAS bayrağı tarafından atlanamaz.
- Indexed/flat SoA hit çözümlemesi korunur; materyal ID üst işaret biti temizlenir.
  UV'nin Y dönüşü raster prepass ve RT shadow ile eşlenir. Paylaşılan albedo/opacity
  dokusunda alpha kanalı, açık alpha-channel bayrağı olmasa da kullanılır.
- Bounce BSDF uygunluğu ile coverage ayrılır: desteklenmeyen bir BSDF üzerindeki
  cutout delikleri de ışını geçirir. Hem authored bit25 hem viewport bit26 okunur.
  Eksik opacity dokusu kapalı kabul edilir; descriptor kapasitesi shader'da sınanır.
- Sonlu parametreli cutout materyallerinde SSS düşük frekanslı diffuse yaklaşımıyla
  kabul edilir. Translucency iki yarımküre arasında tek örnekli diffuse karışımıdır;
  clearcoat diffuse enerjisini azaltır. Bu, tam SSS/refraction çözümü değildir.
  Gerçek `transmission > 0.001`, transmission dokusu, terrain katmanları, su ve
  hacim sınıfları desteklenmeye devam **etmez**. Eski 69/70 ret sayısının tümünün
  kalkacağı söylenemez; yeni sahne sayaçları ölçülmelidir.
- Desteklenen ince cutout arka yüzleri gölgelenir ve kapalı geometri içine düşmüş
  probe tespitine katılmaz. Kapalı opak yüzeylerin arka yüz reddi korunur.
- `core_status` artık yayınlanmış alan, materyal görünümü ve scene lighting
  koşullarından hesaplanır. `gi_active` görünür her pikselin alan içinde olduğunu
  veya GPU'nun tamamlandığını kanıtlamaz; uygun, bağlı traced alanı bildirir.
- `probe_field` ve grid setter yanıtlarında `minimum_cell` / `minimum_world`
  eklendi. Eski `minimum` hücre indeksi olarak korunur. Dünya dönüşümü ortak API'de
  double hassasiyetle yapılır; IPC ve Python aynı değerleri döndürür.
- `viewport.raster_depth_prepass`: `enabled` güncel istek; `effective`,
  `forced_by_rt_shadow` son kaydedilen raster karesidir. `observed=false` ise henüz
  örnek yoktur. `effective_scope=last_recorded_raster_frame` zaman ayrımını belirtir.
  Setter'dan hemen sonra eski kareye ait değer okumak mümkündür; yeni kare üretin.

Kontrol yüzeyleri değişmedi: Python `rayfusion.set_probe_producer(traced=...)`,
`rayfusion.set_probe_bounce(enabled=...)` ve eş adlı IPC yöntemleri aynı servisleri
çağırır. Mevcut boolean doğrulaması ve desteklenmeyen backend hata davranışı korunur.
Yeni bir UI-only ayar veya bağımsız geometri temsili eklenmedi.

## Plan değerlendirmesi ve kalan işler

Bu parti planın **2 ve 7** numaralı kalemlerinin temelini işler; bütün yol haritasını
tamamlamaz. Yeni AO, koni/bilateral/PCSS gölge, çok kademeli LOD, opak prepass kovası
ve main-pass özellik maliyet anahtarları henüz uygulanmadı. Ön geçiş korunmuştur.
Yeni AO'nun mevcut traced GI üstünde aynı kapanmayı ikinci kez çarpmaması gerekir.

Ölçüm tablosundaki süreler geçmiş baseline'dır. Aynı ışın sayısı aynı süreyi garanti
etmez: `NoOpaqueEXT` alfa adaylarını gerçekten çalıştırır ve bu iş ücretlidir.
Tek ışın + 5x5 filtre 25 bağımsız örnekle eşdeğer değildir; filtre maliyeti sıfır
değildir. Güneşin 0.53 derece değeri çap olarak kullanılıyorsa koni yarı açısı
0.265 derece olmalıdır. AO çarpanı sıfır olabiliyorsa ambient'i de sıfırlayabilir.
Bu tahminlerden hız veya kalite garantisi çıkarılmamalıdır.

## Derleme ve kabul — kullanıcı tarafından

1. Normal shader derleme akışınızda **rayfusion_probe_trace_beta.spv** dosyasını
   yeniden üretin (`rayfusion_probe_bounce.glsl` dahil edilir), ardından normal C++
   proje derlemesini yapın. Eski SPV yeni push parametresini ve ince yaprak alanlarını
   tüketmez. Bu partide SPV üretilmedi.
2. Aynı orman/kamera, aynı çözünürlük, material + scene lighting, balanced kalite,
   RT shadow açık ve prepass açık baseline alın. Dört iğne materyalinin kaydedilmemiş
   `alpha_cutout` durumunu kontrol edin. Kapsayan probe grid kullanın; `minimum_world`
   ve `counts * spacing` ile kapsamı okuyun.
3. Sabit kadrajda sky-bake → traced+bounce → sky-bake tekrar karşılaştırması yapın.
   Her kolda alanın yeni üreticisi yayınlanana kadar raster kareleri üretin;
   `producer`, `pending`, `valid`, `bounce_shaded_hits`, ret nedenleri ve
   `rejected_inside` kaydedilsin. Kamera sabit görüntülerde açık/koyu yaprak
   bölgelerinin ve zeminin RT referansına yaklaşmasını inceleyin.
4. Ek kontrol: traced+bounce kapalı/açık aynı grid ve kamera. Delik coverage'ı
   iki kolda aynı kalmalı; değişen katkı bounce olmalı. UV'si belirgin asimetrik
   bir alpha kartı, shared albedo-alpha dokusu, indexed mesh, bit25/bit26 ve
   yaprağın iki yüzünü kontrol edin. Opak kutunun içindeki probe hâlâ reddedilmeli;
   cam hâlâ destek dışı raporlanmalı. Vulkan validation hatası olmamalı.
5. Her zamanlama kolundan önce `viewport.reset_frame_timings`, sonra en az 60
   gerçek raster karesi. `frames > 0`, GPU örnekleri, aynı `visible_triangles`,
   çözünürlük, kalite ve kamera koşullarını doğrulayın. GPU kare/main/prepass/RT
   shadow ve probe `trace_ms` / `bounce_prepare_ms` değerlerini kaydedin.
   Önceki ±%0.8 gürültü tahminini yeni koşullarda A/B/A ile yeniden kontrol edin.
6. RT shadow açıkken prepass isteğini kapatıp yeni kare üretin:
   `enabled=false`, `observed=true`, `effective=true`, `forced_by_rt_shadow=true`
   beklenir. RT shadow'u da kapatıp yeni kare üretince effective/forced false
   beklenir. Sonunda iki ayarı da eski değerlerine döndürün.
7. `rayfusion.core_status`: material+scene ve bağlı geçerli traced alanda aktif;
   sky-bake veya solid/rendered görünümünde pasif olmalı. Python/IPC yanıtlarında
   `minimum_world[i] == minimum_cell[i] * spacing` ve eski minimum eşliği doğrulansın.

Derlemesiz kontroller:

```powershell
python scripts/audit_rayfusion_bounce.py
python scripts/audit_rayfusion_probe_grid.py
python scripts/audit_raster_quality_foundation.py
```

Üçü geçti. Genel `audit_material_coverage.py` mevcut IPC descriptor metninde
`alpha_cutout|base_color` beklentisinde durdu; bu parti o alanı değiştirmedi.
Bu hata giderilmiş veya GPU kabulü yapılmış olarak raporlanmamalıdır.
