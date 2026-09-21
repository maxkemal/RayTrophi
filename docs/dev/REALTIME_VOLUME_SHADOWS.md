# Realtime hacim gölgeleri ve dolaylı ışık sınırı

> **Durum:** REFERANS — kullanıcı Realtime hacim gölgelerinin güzel çalıştığını doğruladı (2026-09-06). GI ayrı açık iş; uygulanmadı.

**Kabul güncellemesi:** aşağıdaki derlenmedi/GPU kabulü açık ifadeleri ilk
teslimin kaydıdır. Sonraki kullanıcı testi gölgelerin çalıştığını doğruladı.
Bütün ışık türleri/kalite geçişleri için ayrı ölçüm ve GPU zamanlaması
raporlanmadı; performans veya tam regresyon kabulü iddiası eklenmedi.

## Uygulanan yol

Mevcut Scene shadow atlasının ışık seçimi, noktasal ışık yüzleri ve güneş/
directional kademeleri kullanılır. `MaterialPreviewVolumeShadow` compute
modülü her gölge ışınında gaz/VDB yoğunluğunu okuyup birikimli optik kalınlığı
derinlik katmanlarına yazar. Doku yerine mevcut binding 7 shadow SSBO'sunun
kuyruğu kullanılır: 33 × 512 byte kayıt, 16 byte metadata, ardından her texel
için giriş/çıkış derinliği ve katman değerleri. Hacim binding 20 ve compute
binding 0 aynı backend'in `m_volumeBuffer` verisini okur.

Yüzey, SDF alıcısı ve hacim aydınlatması aynı
`material_preview_shadow_data.glsl` yardımcısını kullanır. Opaque atlasın
gölgesi ile hacmin geçirgenliği ilgili ışığın radiance değerinde çarpılır.
Ortam ışığı ve emission bu doğrudan ışık gölgesiyle çarpılmaz.

Hacmin önündeki alıcı gölgelenmez, içindeki alıcı yalnız önündeki yoğunluğu,
arkasındaki alıcı toplam ışık kaybını görür. Perspektif ışıklarda nonlinear
hardware depth yerine doğrusal ışık ekseni derinliği (`clip.w`) kullanılır.
Katmanlar kesişen hacim aralığına sıkıştırılır. Her katmanda hacim aralığına
kırpılmış integrasyon, uzak iki domain arasındaki boşluğun ince domain'i
örneklerden tamamen kaçırmasını önler.

Yoğunluk örnekleme kodu `material_preview_volume_fields.glsl` içine çıkarıldı;
görünüm ve gölge aynı sparse/dense grid, dönüşüm, remap ve cutoff işlemlerini
okur. Yeni bir hacim/material otoritesi yoktur. Mevcut raster hacim shader'ının
malzeme graph kapsamı genişletilmedi.

Derin gölge varken önceki self-shadow ikinci kez çarpılmaz. Gölge atlasına
alınmayan ışıklar mevcut yerel self-shadow yaklaşımına döner. Bu eski yoldaki
ışıklar arası tek cache değeri ayrıldı; ilk dolu örnekte her ışık kendi cache'ini
doldurur. Yerel ışık fallback mesafesi kamera march uzunluğu yerine ışığa
mesafedir.

## Mevcut quality sözleşmesi

| Preset | Opaque gölge tile | Hacim tile | Katman | Azami adım / hacim / ışın |
|---|---:|---:|---:|---:|
| Performance | 256 | 32 | 8 | 16 |
| Auto / Balanced | 512 | 128 | 12 | 48 |
| Quality / Full | 1024 | 256 | 16 | 64 |

Ortak bütçe fonksiyonu: `Backend::volumeShadowBudget()`. Renderer, rtapi
`viewportQuality()`, panel, Python `rt.viewport.quality()` ve IPC
`viewport.quality` aynı değerleri kullanır. Yeni alanlar:
`volume_shadow_tile_resolution`, `volume_shadow_depth_layers`,
`volume_shadow_steps`. Bunlar **yapılandırılmış bütçe**, GPU ölçümü veya
compute pipeline'ın başarıyla kurulduğu iddiası değildir. Mutasyon mevcut
`viewport.set_quality`/Python/panel yoludur; yeni bağımsız ayar yok.

Hacim yokken yalnız küçük kayıt tamponu gerekir. Derin atlas tamponu ilk
hacimde büyür; Performance ~10 MiB, Balanced ~56 MiB, Quality ~72 MiB ek
depolama kullanır. Kalite düşürülünce kapasite en yüksek tahsiste kalır,
aktif veri düzeni/kullanılan adımlar güncel kaliteyi izler. Yeniden tahsis ve
descriptor değişimi öncesi in-flight kareler drain edilir; kayıt güncellemesi,
compute yazımı ve fragment okuması komut tamponunda barrier ile sıralanır.

Bu ilk uygulama, çizilen raster karede ilgili shadow view'larını yeniden
hesaplar. Kareler arası ışık/yoğunluk revision cache'i eklenmedi; raster'ın
mevcut idle kare atlaması korunur. Tek karede sonuç bütün alıcılarca paylaşılır.
Kamera hareketi/simülasyon sırasında maliyet ölçülmeden FPS iddiası yoktur.

## Açık sınırlar

- Scene lighting içindir; Three Point sahne ışığı/gölge atlası kullanmaz.
- Gölge ışık bütçesi dışındaki lambalar yüzeylerde gölge düşürmez.
- Bir ışında en fazla 16 kesişen katılımcı hacim; SDF caster bu integrasyona
  dahil değildir. SDF **alıcı** olarak hacim gölgesini okur.
- Geçirgenlik scalar'dır; renkli absorbsiyonun spektral gölgesi ve geniş alan
  ışığının fiziksel penumbrası tam çözülmez. Bilinear filtre ve mevcut opaque
  PCF kullanılır. İnce detay ve ayrık uzak hacimler sınırlı katmanlarda yumuşar.
- Mevcut camera-centred cascade alanının dışındaki caster kapsamı genişletilmedi.
- Kaynak density cutoff/remap ve shadow strength gölgeye etki eder. Emission
  ayrı ışık yayılımı üretmez; ateşten çevreye GI bu değişikliğin parçası değildir.
- Compute shader kurulamazsa log uyarısı ve devre dışı metadata ile devam eder;
  kalite sayılarının okunması çizim kanıtı değildir.

## Solid/Atmosfer ikincil aydınlatma

Kullanıcı bu gölge işiyle birlikte Solid/Atmosfer'in doğrudan ışık almayan
yüzeylere engellenmemiş ortam katkısı vermesini bildirdi. Tek renkli dünya,
açık ve engelsiz yüzeylerde yön bağımsız ışık verebilir; kapalı yüzeylere aynı
katkının ulaşması için görünürlük hesabı gerekir. HDRI kolunda da çevredeki
geometrinin ortam ışığını kesmesi tam çözülmez; yönlü harita eksikliği daha az
belirgin gösterebilir. Gölge atlasını ambient ile çarpmak bir GI çözümü değildir.

Henüz GI kodu eklenmedi. Hafif ilk seçenek lineer HDR sahne rengi + depth/
normal ile tek sekme screen-space GI, diffuse ortam görünürlüğü ve depth/
normal reddiyle temporal filtrelemedir. Tek sekme ışık, emission ve specular
ile gelişigüzel karıştırılmamalı; ekran dışında kalan oda/engel verisini bu yol
bilemez. Ekran dışı doğruluğu hedeflenirse viewport cihazında flat TriangleMesh/
DNA verisinden traversal ve ray-query/probe altyapısı ayrı kurulmalıdır. Yeni
GI operasyonları UI/script/IPC ortak çekirdeğiyle birlikte teslim edilmelidir.

## Yapılmış kontroller

- 96 analitik homogeneous slab geçirgenlik/receiver derinlik kontrolü;
  perspektif derinlik dönüşümü ve her quality için son tampon indeksi.
- Üç receiver shader'ı, ortak density include, UI/IPC/Python değerleri ve
  vcxproj'da tek `.cpp` kaydı statik kontrolü.
- `audit_realtime_sdf_surface.py`, `audit_shader_struct_layout.py` geçti.
- `audit_ipc_capabilities.py`: 415 metot sınıflandırılmış; üretilmiş
  descriptor güncel. Metot sayısı eşzamanlı diğer çalışma değişikliklerini de içerir.

Derleme/uygulama başlatma yapılmadı. Aynı çalışma alanındaki diğer HDR/DoF
ve pozlama işleri korunarak [NEXT_BUILD_CHECKS.md](NEXT_BUILD_CHECKS.md)'ye
§20 ve sonrası eklendi.
