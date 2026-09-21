# Vulkan RT: fluid kutusu içindeki gas katkısı

## Canlı gözlem — 2026-09-07

Kullanıcının açık uygulamasına IPC ile bağlanıldı; uygulama başlatılmadı,
derleme yapılmadı. Kamera ve simülasyon değiştirilmeden material/RayFusion ve
rendered/Vulkan RT görüntüleri alındı. Frame 250, playing=false, epoch=0;
1680×945; Vulkan RT görüntüsü 128 sample sonrasında yakalandı.

- `Burning Fuel Liquid`: min=(-2.5, 0, -2.5), max=(2.5, 1.8, 2.5), surface.
- `Burning Fuel Gas`: min=(-2.5, 0, -2.5), max=(2.5, 5, 2.5), volume.
- İki backend de iki volume instance ve ayrılmış SSBO raporladı.
- RayFusion'da alev tabana kadar sürüyor; Vulkan RT'de liquid domain üst
  sınırında düz bir bant oluşuyor ve alt aralıktaki gas katkısı kayboluyor.
- Yerel görüntüler: `tmp/gas_rayfusion_before.jpg`,
  `tmp/gas_vulkan_rt_before.jpg`.

İnceleme sonunda shading=material ve capture=false geri yüklendi. Kamera,
timeline, domain ve materyal değerleri değiştirilmedi. Sayaç denemesi
`enabled=false` döndürdü; sıfır değerler ölçüm olarak kullanılmadı.

## Kaynak bulgusu ve düzeltme — YAZILDI

`volume_intersection.rint` sıvının gerçek yüzeyini değil AABB girişini
raporluyor. Küçük `isoBias`, yalnız yakın girişler arasında öncelik sağlar;
sıvı AABB'si önce kazanırsa closest-hit sıvı yüzeyine yürür veya AABB'den çıkar.
Bu arada önündeki/boş domain içindeki gas'ın ışımaya katkısı hesaplanmaz.
Görüntüdeki düz kutu sınırı bu kaynak hatasıyla uyumludur. Kesin düzeltme
doğrulaması yeni shader build'iyle yapılacak.

`volume_overlap_selection.glsl`, sıvı kutusu önce seçildiğinde önündeki aktif
gas aralığını bulur ve mevcut gas yürüyüşüne yönlendirir. Gerçek sıvı sınırı
mevcut `nearestSurfaceSDFCrossing` / `sampleIsoField` ile aranır; gas yürüyüşü
aynı sınırda sıvıya döner. Yeni bir yoğunluk/materyal kopyası oluşturulmaz.
Kamera-içeride durumu ve dejenere aralık kontrolü seçilen hacme göre güncellenir.

Gazdan sıvıya dönüş trace'i gas'ı cull mask ile dışlar. Yeni
`RayPayload.volumeTraversalMask`, bu giriş maskesini yardımcı modüle taşır;
böylece modül gas'ı tekrar seçip sıfır ilerlemeli döngü oluşturamaz. Raygen
her trace öncesi maskeyi yazar; photon raygen kendi 0x01 maskesini yazar.
Ortak payload ABI'si bir uint büyüdü. UI/Python/IPC işlemleri değişmedi;
düzeltme hepsinin kullandığı render yolundadır.

Sınırlar: mevcut 16-volume tarama sınırı korunur. Desteklenmeyen SDF depolaması
için mevcut muhafazakâr AABB handoff davranışı değişmedi. Bu çalışma genel
olarak tüm üst üste gas domain'lerinin sıralı integrasyonunu çözmüş sayılmaz.

## Doğrulama

Shader aralık kontrolünde `gl_RayTmaxEXT` orijinal ray limiti olarak
kullanılmadı: closest-hit aşamasında `gl_HitTEXT` ile eşanlamlıdır
([Khronos GLSL EXT_ray_tracing](https://github.khronos.org/Vulkan-Site/glslext/latest/glslext/ext/GLSL_EXT_ray_tracing.html)).
Burada liquid kutusu girişini gas çıkış limiti saymak gas aralığını tekrar silerdi.

Kaynak denetimleri geçti:

- `python scripts/audit_shader_struct_layout.py`
- `python scripts/audit_realtime_sdf_surface.py`

Bunlar shader derlemesi veya yeni GPU davranışının testi değildir.

Kullanıcı build/kabul sırası:

1. **`rt_payload.glsl` kullanan tüm RT shader'larını birlikte yeniden derle.**
   Yalnız `volume_closesthit` güncellenirse payload ABI'si uyuşmaz.
2. Uygulamayı normal şekilde derleyip yeni shader'larla aç.
3. Aynı 250. kare ve kamerada iki görüntüyü tekrar al: fluid kutusunun 1.8
   yüksekliğinde yapay alev kesilmesi kalmamalı; gerçek liquid yüzeyi görünmeli.
4. Liquid domain'inde boş aralıktan geçen ışın, gerçek sıvıya çarpan ışın,
   kamera domain içindeyken görüntü, gas önündeki katı obje ve gas→sıvı dönüşü
   kontrol edilmeli. Siyah bant, tekrar-giriş döngüsü veya TDR kabul edilmez.
5. Gas-only, fluid-only, ayrık domain ve cached/live geçişlerini kontrol et;
   128 sample görüntü ve kare süresini aynı çözünürlükte karşılaştır.

Başarı kullanıcı build'inden sonra kaydedilecek; şu an yalnız eski build'deki
görsel fark doğrulandı.
