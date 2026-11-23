# Vulkan malzeme doku dizileri

> **DEVİR 2026-11-23 (Codex) — Lenovo ARM64 Release.** Tek sahneli Boyama Odası
> APK'sı `E:\rt_android_build\RayTrophiPlayer-arm64-release-workshop-2026-11-23.apk`
> kurulu. Player açık, HDRI etkin, iç hedef %100 (1920×1200), tablet doğal çözünürlükte
> ve önceki 204 dpi ayarında. Kamera karşılaştırmalar boyunca aynı kaldı.
> **Açılış:** Debug 163,6 s → Release 63,8 s, iki açılışta da HDRI etkin.
> **GPU:** 1920×1200 HDRI 4,02 s → düz renk 3,20 s → HDRI 3,85 s.
> HDRI ile 960×600 1,06 s, 480×300 0,34 s. Tam hedefte piksel sayısı ana etken.
> **Boya:** PC Player'da renk doğru. Android'de düşük çözünürlük denemesinde damlalar
> ulaştı ve doğru renkle boyadı. Geç doku sayfalama hipotezi doğrulanmadı.
> **24 Kasım:** HDR'siz proje kopyasıyla aynı Release APK'da iki açılışı karşılaştır.
> Ardından 960×600'de aynı kamera ile gölge, ışık ve transmission maliyetini tek
> değişkenli ölç. Ayrıntılar ve açık kontroller NEXT_BUILD_CHECKS başında.

> **DEVİR 2026-11-23 (Claude) — YENİ OTURUM ÖNCE BUNU OKU.** C modeli adım 1+2 DERLENDİ ve
> CANLI: Windows PASS (Material=Rendered dokular, purge, yansıma/GI/RT gölge), Lenovo
> Mali-G52 KABUL PASS (ret yok, workshop çiziliyor, `packed=35 pages=2`). Derlemede bulunan:
> GPU_ONLY `createBuffer` `initialData` YAZMAZ → ref tablosu `uploadBuffer` ile (düzeltildi).
> Derlenmedi bekleyen küçük düzeltme: sıkıştırma önizleme set damgası (STALE ERROR yanlış alarmı).
> **AÇIK — Lenovo PERF:** 1920x1200 kare ≈3,8 s GPU, piksele doğrusal; 32 dallı switch ELENDİ
> (2 dallı varyant aynı); düz duvarda bile 960x600 893 ms → her pikselde temel aydınlatma.
> Sahne 6 ışık/6 gölgeli/3 kaskad. Ayrıştırma shader varyantlarıyla (S gökyüzü, A gölge,
> B ışık döngüsü, C prob GI) sürüyor — sonuç NEXT_BUILD_CHECKS madde 9'da. Player IPC'si
> kalite/ışık/dünya anahtarlarını kapatıyor (§1 açığı). **Yükleme ≈125 s:** game_design 88 s,
> prepare_particles 28 s; kullanıcı: HDR dosyası 26 MB (mobilde küçült/ASTC). IPC
> `launcher.select` yüklemeyi başlatmıyor (dokunma gerekiyor). Diğer AÇIK: texture_arrays IPC,
> geç doku canlı sıkıştırma, binding 25–27 partially-bound, Codex Residency çekirdeği kullanılmıyor.
> **★ KULLANICI GÖZLEMİ (Lenovo, ıslak boya):** balon patlayınca duvarda boyanın NORMAL etkisi
> görünüyor ama ALBEDO yok (duvar kendi renginde kabarıyor). Muhtemel kök: boya albedo dokusu
> sahne batch'inden SONRA oluşuyor/kaydoluyor → sıkıştırılmamış (tekli) → viewport ref
> tablosunda `kNoTexture` (bilinen sınır 8). Doğrulama: log'da boya dokusu yükleme satırı
> `[TexturePages] compaction` satırından sonra mı. Düzeltme: `markMaterialPreviewTexturesChanged`
> bekleyen-sıkıştırma bayrağı koysun, kare başında (kayıt dışı) `compactMaterialTexturesLocked`
> koşsun; ya da boya katmanı yaratıcısı sıkıştırmayı tetiklesin. Windows'ta da aynı olmalı —
> Studio'da boya testi (madde 3) ile önce orada üret.

> **DEVİR (2026-11-22 gece, Claude):** C modeli ADIM 1+2 YAZILDI, derlenmedi (/Zs PASS,
> 10 dosya); viewport SPIR-V DAĞITILDI (8 modül, spirv-val PASS, indexing capability 0).
> Adım 1: `VulkanMaterialTextureStorage` sıkıştırma + `compactMaterialTexturesLocked` (sahne
> batch sonu), `ImageHandle` katmanlı. Adım 2: `VulkanMaterialTextureBinding` = 32 sayfa +
> slot→ref SSBO (preview 1/29, rt_shadow 5/8, prob/yansıma/GI 7/14); ret dalı ve indexing
> kapıları viewport'tan söküldü; taban adapter'ın (g_backend) önizleme set'i de geçirildi.
> AÇIK: (a) `texture_arrays` IPC durumu YOK (CLAUDE.md §1 — sayfa sayısı, packed/skipped,
> bütçe aşımı, paketlenmemiş doku sayısı); (b) sıkıştırmadan sonra gelen doku canlı
> yeniden sıkıştırma tetikleyicisi yok; (c) binding 25–27 Mali'de partially-bound'suz;
> (d) Codex'in Residency/UploadQueue/FramePublication çekirdeği kullanılmıyor — C modelinde
> gerekmezse sök (§5). Kontrol: NEXT_BUILD_CHECKS başı.

> **★ KAPSAM KARARI (2026-11-22, kullanıcı):** Doku dizileri YALNIZ viewport/RayFusion
> backend'inde (`VulkanViewportBackend.cpp` + RayFusion/preview/GI pass'leri; Studio
> viewport, Windows Player, Android Player). **Vulkan RT render backend'i
> (`VulkanBackend.cpp`, RT binding 6, `*.rchit/*.rmiss/*.rahit`) BINDLESS KALIR** —
> RT her çarpışmada her malzemeye erişir, hedef cihazlarda RT zaten yok. Bu §5 ihlali
> değil: her renderer'ın tek yolu var. Sıra: adım 1 viewport doku descriptor
> noktalarını tek dosyada topla (eski ABI, davranış aynı) → adım 2 o dosyayı diziye
> çevir + viewport cihazından indexing zincirini ve ret dalını sök → derleme → Lenovo.
> Aşağıdaki "Vulkan RT 6" ve "RT shader'ları da geçer" ifadeleri bu kararla geçersiz.

> **★★ ADIM 2 ÖNCESİ BULGULAR (2026-11-22, Claude) — host tasarımını değiştirir:**
> 1. **Viewport, RT ile AYNI malzeme buffer'ını okur** (`VulkanViewportBackend.cpp:2081`,
>    `m_externalMaterialBuffer`/`m_materialExtBuffer`; önizleme=Rendered eşitliği için bilerek).
>    Alanlar (`albedo_tex`…) RT **slot ID**'si taşır. Viewport shader'ı ref beklediği için
>    buffer'a ref yazılamaz (RT kırılır) → viewport'a **slot→ref çeviri SSBO'su**
>    (`uint slotToRef[]`), shader `ref = slotToRef[slot]`. Malzeme buffer'ı ve RT değişmez.
> 2. **Islak boya tekli image'a yazıyor:** `copyDeviceBufferToTexture`
>    (`VulkanBackend.cpp` ~15004) her vuruşta `m_uploadedImages` image'ına kopyalar + mip
>    zincirini yeniden üretir. Diziler AYRI KOPYA olursa boya viewport'ta görünmez
>    (sessiz — Player'ın ana demosu). "Player'da tekli image'ı bırak" seçeneği bu yüzden
>    ucuz değil.
> 3. **Önerilen model (C): tek depolama = dizi katmanı, her yerde.** Her doku yüklemede
>    bir sayfanın katmanına yerleşir; `ImageHandle` (image, view, layer) taşır. Bindless
>    tüketiciler (RT binding 6, IBL, gölge, matcap) aynı katmanın **katman başına 2D
>    view'ını** (`VK_IMAGE_VIEW_TYPE_2D`, `baseArrayLayer=layer`) kullanır; viewport
>    sayfanın dizi view'ını. Bellek her platformda 1×, boya bir kez yazar ikisi de görür.
>    Dokunulacak yazıcılar: 3 upload fonksiyonu (`endBatchedTextureUpload`,
>    `registerSceneTextureUpload`, ~14702), `destroyTexture`, boya kopyası, mip üretimi —
>    hepsi `subresourceRange.baseArrayLayer = layer`. Yerleşim artımlı (yükleme anında
>    anahtar bilinir); sayfa dolunca aynı anahtarla yeni sayfa, 32 sayfa tavanı aşılırsa
>    `kNoTexture` + uyarı. Ebat birleştirme (üst kovaya taşıma) bu modelde yok — tam
>    anahtar; kova patlaması `texture_arrays` ile ölçülür, gerekirse sonra.
>    Codex'in async `Residency/UploadQueue/FramePublication` çekirdeği C'de canlı
>    düzenleme yeniden yerleşimi için kalır; ilk sürüm senkron yerleşim.
>
> **★★ C MODELİ — KESİN TASARIM (2026-11-22, envanter sonrası):**
> - **Büyüyen sayfa YOK.** Dizi image'ı büyütülemez; büyütmek view'ları geçersiz kılar ve
>   `ImageHandle` değer olarak kopyalanıyor (matcap, env, IBL kaynağı, bekleyen RT
>   descriptor'ları, `registerSceneTextureUpload` destroy lambda'sı). Bunun yerine
>   **sıkıştırma**: yükleme bugünkü gibi tekli image'a gider; batch/sahne yüklemesi sonunda
>   tekli image'lar anahtara (format, w, h, mip, sampler sınıfı) göre gruplanır, TAM boyutlu
>   sayfalar ayrılır, `vkCmdCopyImage` ile katmanlara kopyalanır, `m_uploadedImages`
>   girdileri (image=sayfa, view=katman 2D view, `arrayLayer`, `sharedImage=true`) ile
>   değiştirilir, tekli image'lar GPU boşaldıktan sonra yok edilir,
>   `m_textureCacheGeneration` artırılır (viewport descriptor'ı buna bakıp yeniden kuruluyor)
>   ve RT binding 6 slotları yeniden yazılır.
> - Sıkıştırmadan sonra gelen tek doku tekli kalır → viewport onu `kNoTexture` görür
>   (sayaçla raporlanır); sonraki sıkıştırma alır. Boya yeni doku üretmez.
> - `ImageHandle` += `uint32_t arrayLayer = 0; bool sharedImage = false;`
>   `destroyImage`: view+sampler her zaman, image+memory yalnız `!sharedImage`. Sayfaları
>   depolama modülü yok eder.
> - Bütün yazıcılar `baseArrayLayer = img.arrayLayer` (VulkanBackend.cpp 6677–7226,
>   14741–15090: mip üretimi, layout geçişi, 4 copyBufferToImage, kısmi mip yenileme,
>   in-place/region güncelleme, boya kopyası). Bariyer katman başına, sayfa geneli değil.
> - Kimlik image ile değil VIEW ile (`removePendingRTTextureDescriptor` :5096).
> - Değer kopyaları (matcap/env) sıkıştırmada id'den yeniden çözülür.
> - `uploadTexture3D` kapsam dışı. Sampler doku başına kalır; sayfa anahtarına sampler sınıfı
>   girer, viewport sayfa descriptor'ı sayfanın ilk dokusunun sampler'ını kullanır.

> **Devir (2026-11-22):** Frame yayını farklı sayfaya taşınan katmanı atomik
> olarak değiştirip eski katmanı son okuyucu tamamlanınca geri alabiliyor.
> Renderer ve durum snapshot'ı entegrasyonu açık. Yeni SPIR-V'yi dağıtmadan
> önce tüm host descriptor ve GPU ref üreticilerini birlikte geçir.

> **Durum:** AKTİF — 2026-11-22, kullanıcı onayladı. Android staging ilk bağımsız parti.

## Mevcut model ve tablet bulgusu

22 Kasım 2026 tarihli kullanıcı `adb vkjson` ölçümünde Lenovo TB330FU (Mali-G52 MC2,
seri HVA751MH) Vulkan 1.3 loader ve cihaz, aşama başına 256 sampled image ve 35 SSBO,
4096 image array layer ve 16384 2D boyut raporluyor. `independentBlend`,
`fragmentStoresAndAtomics`, `bufferDeviceAddress` ve uniform sampled-image array dinamik
indeksleme var. `VK_EXT_descriptor_indexing` ve gereken üç core özelliği yok.
`rayFusionDeviceRejection` ilk olarak `required descriptor indexing features unavailable`
döndürüyor. Çocuk başlatıcısı açılıyor, oyun viewport başlatmada duruyor.

Vulkan backend CPU doku nesnesini yükler, GPU doku kimliğini materyal, terrain ve diğer
tüketicilere yazar. Material Preview set 0 binding 1 ve Vulkan RT set 0 binding 6 birer
sampled-image descriptor tablosu taşır. RT slot güncellemeleri
`updateRTTextureDescriptor` ve `m_pendingTextureDescriptors` ile, viewport güncellemeleri
`updateMaterialPreviewTextureDescriptor` ile yürür. Viewport bağlama kapasitesini
`rayFusionMaterialTextureCapacity` cihaz limitlerine göre kısar. Instance, hit ve materyal
programı farklı invocation'larda farklı doku seçebildiği için uniform dinamik descriptor
indeksleme özelliği bu erişimi karşılamaz.

Kaynak taramasında 66 `nonuniformEXT` satırı 13 dosyada bulundu.
`material_program.glsl`, `material_preview_frag.frag`, `material_preview_opacity.glsl`,
`material_preview_sdf_surface.frag`, `rayfusion_probe_bounce.glsl`,
`rayfusion_rt_shadow.comp`, `bsdf_scatter.glsl`, `rt_sky.glsl`,
`volume_hit_sampling.glsl`, `closesthit.rchit`, `hair_closesthit.rchit`,
`shadow_anyhit.rahit` ve `volume_closesthit.rchit`. Derleme kapsamı include zincirini
takip ederek preview shadow, probe, reflection, screen GI ve RT varyantlarını da kapsar.

## Kova seçimi ve sahne ölçümü

Varsayılan, tam `(width, height, VkFormat, mipLevels, sampler davranışı, image kullanım sınıfı)`
eşleşmesidir. 512/1K/2K/4K doğal kovalardır. Dikdörtgen ve power-of-two dışı boyutlar
kendi tam ebatlarını kullanır. Her katmanın UV ölçeği (1,1). Tekrarlama, clamp, normal
kanal düzeltmesi ve anisotropik filtreleme mevcut tüketicinin semantiğini korur.
`VkFormat` sRGB ile linear ve BC/ASTC ile R/RG/RGBA/float ayrımını taşır. Kaynak codec
ve GPU formatı farklıdır. Cihazda desteklenen sıkıştırılmış payload ve mipler doğrudan
kopyalanır. Desteklenmeyen format için özgün decode verisinden desteklenen kayıpsız
upload biçimi seçilir. Normal yerleşim boyut değiştirmez. Sayfa bütçesi aşılırsa
aşağıdaki birleştirme kuralı devreye girer.

Kaynak `RayTrophiStudio/assets/scenes/<scene>/<project>.rtp.shared` içindeki
`textures[]` kayıtları, bir kayıt/doku ID'si başına bir katman varsayımıyla sayıldı.
İki manifestin `dist/RayTrophiPlayer/scenes` kopyaları kaynakla aynı. SHA-256 önekleri
workshop için `5283ed152e62`, kids-play-room için `d47e16de56da`. Bütün kayıtlar
gömülü JPEG/PNG. HDR ortam dosyaları ve çalışma anında üretilen kaynaklar bu sayımın
dışında.

| Sahne | Boyut | sRGB albedo | Linear normal / roughness / height / opacity | Toplam |
|---|---|---:|---:|---:|
| workshop | 1024×1024 | 8 | 8 / 8 / 8 / 3 | 35 |
| kids-play-room | 512×512 | 70 | 15 / 10 / 16 / 0 | 111 |
| kids-play-room | 1024×1024 | 0 | 3 / 3 / 2 / 0 | 8 |

| Sahne | Tam eşleşme, RGBA8 tam mip yükü | Eksen başına veya kare 512/1K/2K/4K yuvarlama | Boyut dolgusu |
|---|---:|---:|---:|
| workshop | 195 734 140 B (186,67 MiB) | 195 734 140 B | %0 |
| kids-play-room | 199 928 332 B (190,67 MiB) | 199 928 332 B | %0 |

Hesap, her doku için `4 × Σ max(1,w>>m) × max(1,h>>m)` biçiminde tüm mipleri
1×1'e kadar toplar. Dolgu oranı `(kova yükü − özgün yük) / kova yükü`.
Örnek olarak, bu sahnelerde bulunmayan 1536×1024 RGBA8 dokuyu 2048×1024'e
yuvarlamak yaklaşık %25, 2048×2048'e yuvarlamak yaklaşık %62,5 dolgu üretir.
Tam eşleşme ayrıca mip kenarlarında wrap ve filtreleme için özel UV hesabı gerektirmez.
Bedeli, değişken boyut veya sampler profili çok olan sahnelerde daha fazla kova.
Yalnız renk uzayı ve boyut üzerinden workshop 2, kids-play-room 3 kova kullanır.
Gerçek format, mip ve sampler ayrımı bu sayıyı artırabilir.

RGBA8 modeli GPU tahsis ölçümü değildir. Backend bugün BC4/5/7 cache ve kompakt
R/RG seçebiliyor, bazı upload kiplerinde mip sayısını kısıyor ve preview boyut
sınırıyla resize yapabiliyor. Bu kontrolleri uygulama aşamasında envanterle.
Diziye geçiş boyut veya mip kırpması getirmeyecek. Açıkça seçilmiş preview
kalite ayarları varsa aynı ayarı önce/sonra karşılaştırmasında sabitle ve
varsayılan tam kalite yolunda özgün ebatı doğrula. Gerçek maliyeti format
blokları, boş kapasite, Vulkan tahsis hizalaması ve büyüme sırasındaki ek
tahsis üzerinden cihazda raporla.

### Önce Android staging

`scripts/stage_player_android_assets.py --all --max-texture 512` Android
varsayılanını kullanır. Yalnızca staging kopyasındaki gömülü JPEG/PNG dokular
Lanczos ile oran korunarak küçültülür. PNG kayıpsız yazılır, JPEG kalite 97 ve
4:4:4 kullanır. Küçültülen normal vektörleri yeniden normalize edilir. Doku
ID'leri ve manifest şeması sabit kalır. Gömülü doku bölümünün boyut ve ofsetleri
güncellenir, Android asset stamp dönüştürülmüş çıktıdan hesaplanır.

| Sahne | Önce → sonra doku | Yeniden boyutlanan | Önce → sonra gömülü yük |
|---|---:|---:|---:|
| workshop | 35 → 35 | 35 | 1 370 526 → 118 439 B |
| kids-play-room | 119 → 119 | 8 | 1 378 996 → 1 058 553 B |

Android 512 staging sonrası RGBA8 tam mip yükü modeli workshop için 46,67 MiB
ve kids-play-room için 158,67 MiB. Bunlar GPU tahsisi ve APK depolama ölçümü
değildir. **Sonra: Android ASTC yolu.** BC desteklemeyen Mali/Adreno cihazlarında
BC7 cache içeriği RGBA8 olarak yüklenir. Özgün boyutlu iki sahne için bu yaklaşık
190 MiB/sahne demektir. Android ASTC ayrı partidir. Lenovo'da diziye geçişten
sonra gerçek `allocated_bytes` değerini iki sahne için 512 staging ile ölç.

### Sayfa bütçesi aşılırsa

32 sayfaya sığmayan boyut anahtarlarından en az kullanılanları önce birleştir.
Kullanımı referans sayısıyla ölç, eşitlikte tahmini ek bayt ve deterministik
anahtar sırasını kullan. Hedef kova her iki eksende de kaynak kadar büyük olur.
Uyumlu hedef yoksa aday üst boyutları karşılaştır ve en düşük ek yükle birden çok
anahtarı birleştirecek hedefi seç. Gömülü veya kaynak mip seviyelerini düşürme.
Her özgün mipten eşleşen üst boyutlu mip örneklenir, normal vektörleri yeniden
normalize edilir. Sıkıştırılmış kaynak yeniden kayıplı kodlama gerektiriyorsa
uyumlu decode biçimine terfi edilir. Raster ve ray örneklemesinde özgün doku
ebatına göre LOD/ofset semantiği korunur.

Boyut çeşitliliği tek başına sahne yüklemesini reddetmez. Renk uzayı, format
ve image kullanım semantiğini koruyarak birleştir. Fiziksel image limits ve
tahsis hatalarını gerçek cihaz hatası olarak raporla. `merged_keys` özgün ve
hedef boyutları, etkilenen doku sayısını ve ek tahminî payload baytlarını bildirir.
Birleştirme yapıldığında aynı snapshot'a bir uyarı ekle.

## Shader ve descriptor ABI

- Shader kaynak geçişi 32 öğeli bir `sampler2DArray` descriptor dizisi
  kullanıyor. Dallar descriptor dizisine yalnız derleme zamanı sabitleriyle
  erişiyor. Set 0 binding numaraları korunuyor: preview ve raster gölge 1,
  ray-query gölge 5, Vulkan RT 6, probe/reflection/screen GI 7. Önceki
  slot sayısı kontrolleri artık GPU referansının sayfa alanını denetliyor.
  Host tarafı her binding'e 32 array view yazmalı. Kullanılmayan öğeler
  geçerli bir dummy array view almalı. Bu host geçişi bitmeden yeni SPIR-V
  source/runtime dizinlerine dağıtılmaz.
- `VulkanMaterialTextureDescriptors.h` her yazımda 32 öğenin tamamını
  doldurur ve boş sayfalara shader-read durumunda 1×1×1 dummy array view
  bağlanmasını ister. Yayın öncesinde bütün GPU doku referansları sayfa
  kapasitesi ve katman doluluk haritası ile denetlenir. Ayrılmış ve emekli
  katmanlar geçersizdir. Geçersiz referans bütün descriptor yazımını durdurur.
  Ret sonucu ilk hatanın nedenini, varsa sayfa, katman ve referans indeksini
  taşır. Residency bu durumda yeni descriptor ve materyal tamponunu birlikte
  reddeder. Önceki çift ancak bağlı tüm kaynakları hâlâ tutuluyorsa çizilebilir.
  Güvenli bir önceki snapshot yoksa o çizim atlanır. Sayaç kare yayını başına
  bir kez artar ve log aynı yapılandırılmış ret verisini kullanır.
- Önerilen ilk bütçe 32 sabit kova sayfası. Her sayfa bir `sampler2DArray`
  descriptor öğesi ve bir `VK_IMAGE_VIEW_TYPE_2D_ARRAY` view kullanır. Kova anahtarı
  aynıyken katman sınırı dolarsa yeni sayfa açılır. GLSL `switch` dalları sabit
  descriptor öğelerini örnekler, değişken katman `vec3(uv, layer)` koordinatıdır.
  Boş binding'ler geçerli 1×1 dizi view'ına bağlanır. Vulkan RT ve RayFusion
  aynı binding sözleşmesini kullanır. 32 taslak değerdir. Pipeline layout
  envanteri ve cihaz limitleriyle doğrula. [Vulkan descriptor örneği](https://docs.vulkan.org/samples/latest/samples/extensions/descriptor_indexing/README.html)
- Mevcut 32 bit Vulkan GPU doku indekslerini `texture_ref` ile değiştir.
  Öneri üst 16 bit sayfa, alt 16 bit katman. `0xffffffff` yok işareti,
  sayfa 0 katman 0 geçerli. Tüm eski signed/unsigned ve sıfır sentinel
  kontrollerini birlikte çevir. CPU cache kimliği ve proje texture ID'si
  GPU referansından ayrı kalır. Materyal SSBO stride ve push constant
  boyutunun değişmemesi hedeflenir. C++ / SPIR-V offset ve stride karşılaştırması
  bunu doğrular. Materyal programı operand kodlama sınırlarını denetle ve
  gerekiyorsa program ABI sürümünü artır.
- `material_program.glsl` merkezli yardımcılar renk, alfa, explicit LOD,
  gradient, boyut sorgusu ve gerekiyorsa texel fetch işlemlerini karşılar.
  Fragment gradient'lerini kova dalından önce, her çağrının mevcut UV dönüşümü
  ve türev semantiğine göre hazırla. Compute ve RT mevcut explicit veya
  taban LOD davranışını korur. Opacity/cutout, triplanar, terrain ve
  reflection/GI tüketicilerinin bütün varyantlarını denetle.
- Terrain splat/overlay/semantic, instance MSF, FFT height/normal, hacim
  malzemesi ve HDR ortam gibi ortak tablo kullanıcılarını ayrı ayrı izle.
  Düzenli 2D kaynaklar uygun diziye girer. Ortam gibi tekil pipeline
  kaynakları sabit binding kullanabilir. Gerçek 3D/storage/UI/generic image
  handle tüketicileri kendi türlerinde kalır. Bir uploader'ın çağrılması
  kaynağı kendiliğinden malzeme katmanı yapmaz.
- Toplam pipeline bütçesinde sampler, sampled-image, per-stage resource,
  per-set ve bound-set limitlerini diğer binding'lerle birlikte ölç.
  `vkGetPhysicalDeviceImageFormatProperties` ile gerçek format/usage için
  boyut, mip ve katman sınırlarını denetle. Sayfa bütçesi aşılırsa önce boyut
  birleştirme kuralı çalışır. Cihazın veya formatın fiziksel sınırına takılan
  tekil doku için `kNoTexture` kullanılır ve uyarıya doku indeksi, boyut,
  mip sayısı, profil yazılır. Diğer dokular yerleşir. Bu durumda Vulkan çalışma
  anında aşağı örnekleme yapmaz. Tahsis veya transfer işlemi başarısız olursa
  eldeki geçerli snapshot korunur.

## Residency ve async düzenleme

Yeni Vulkan residency modülü payload hazırlama, kova/katman ayırma ve frame
snapshot yayınını yönetir. Aynı kaynak ve uyumlu format/sampler/mip anahtarı
katmanı paylaşabilir. İlk yüklemede bilinen katman sayısı için toplu tahsis
yap. Canlı eklemede boş katmanı kullan veya gerektiğinde yeni image
oluşturarak sayfayı büyüt. Büyüme GPU üstünde mip/katman kopyası yapar.
Farklı biçime geçiş, paint dirty ve silme işlemleri referans revizyonu taşır.

Decode ve hazırlık mevcut async malzeme yolunda kalır. Transfer, mip üretimi,
bariyerler ve GPU kopyası render zamanlayıcısının batch'ine girer. Tek GPU
kuyruğunda hazır olma durumu mevcut fence/timeline veya submit sırasıyla
izlenir. Edit yolunda `vkQueueWaitIdle`, `vkDeviceWaitIdle` veya bloklayan
tek seferlik submit kullanılmaz. Barrier aralıkları dokunulan mip ve katmanları
kapsar. In-flight edit için yeni katman veya nesil kullan. Descriptor set,
materyal buffer ve dizi view aynı nesil olarak güvenli frame sınırında
yayınlanır. Eski katmanlar, view'lar, setler ve staging son okuyucu fence'inden
sonra geri alınır. Yeni revizyon geldiğinde geç biten eski işin yayını engellenir.
Tahsis, transfer ve scene teardown hata yollarında kısmi referans yayınlama.

## Eski yolun sökümü ve cihaz kabulü

1. Ortak Vulkan helper ve residency tamamlanınca, descriptor tablosu kullanan
   bütün shader girişlerini ve layout/pool/write üreticilerini birlikte geçir.
   Windows Studio, Windows Player ve Android aynı modeli kullanır. CUDA ve
   OptiX kapsam dışıdır. Uzun dosyalarda yalnız entegrasyon bağlantıları kalır.
2. Eski unsized/fixed per-texture sampler tablolarını, `nonuniformEXT`
   qualifier'larını, per-texture descriptor pending/deferred yazımlarını,
   dummy slot döngülerini ve descriptor slot dolunca cache purge yolunu sök.
   `VULKAN_TEXTURE_CAPACITY` ve `rayFusionMaterialTextureCapacity` eski
   amaçlarıyla kalmasın. Genel image cache ve bağımsız GPU kaynaklarının
   ömrünü çağıranlarıyla denetle.
3. Kullanıcısı kalmayan descriptor-indexing feature/extension zincirini,
   partially-bound/update-after-bind layout ve pool flag'lerini ve pipeline
   gate'lerini kaldır. SPIR-V'de `RuntimeDescriptorArray`,
   `SampledImageArrayNonUniformIndexing` ve `NonUniform` kalmadığını doğrula.
   Sonra `rayFusionDeviceRejection` indexing ret dalını kaldır.
4. Sırada kaynakta `independentBlend` / `fragmentStoresAndAtomics` ve ortak
   graphics+compute queue kontrolü var. Kullanıcı ölçümüne göre ilk ikisi
   mevcut. Kaydedilen `tablet_vkjson.json` ortak graphics+compute kuyruğunu
   (iki kuyruk, flags 7), per-set 210 SSBO ve aşama başına 128 sampler
   gösteriyor. Gerekli render formatlarının optimal tiling özellikleri
   mevcut. Gerçek image usage kombinasyonları ve pipeline oluşturma
   canlı cihazda doğrulanacak. İlk cihaz koşusunda gerçek ilk hata varsa
   aşama, limit/format veya VkResult ile kaydet. Yeni sampler bütçesini de
   ölç. Mali için bir sonraki eksik özellik şu anda belirlenemiyor.
5. **İkinci hedef cihaz: Omix X700, Mali-G76 MC4, Vulkan 1.1.131** (gpuinfo raporu
   2026-11-22). Bu cihazda da descriptor indexing yok, `maxImageArrayLayers`
   **1024**. Sayfa kapasitesini cihazın `maxImageArrayLayers` sınırı ve
   seçilen format/usage için bildirilen katman sınırının küçüğüyle sınırla.
   Bu partide Vulkan 1.2 cihaz kapısı korunur. Önce Lenovo TB330FU (Vulkan 1.3,
   descriptor indexing yok) üzerinde iki sahnede doku dizilerinin çalıştığını
   doğrula. Bu kabulden sonra ayrı bir partide Android'de `apiVersion < 1.2`
   kapısını 1.1'e indir, buffer device address ve scalar block layout'u KHR/EXT
   eklentileriyle iste (ikisi de raporda var), Player shader'larını
   `vulkan1.1` hedefiyle derle (323/323 deneme derlemesi PASS). Ayrıntı:
   PLAYER_ANDROID_PORT.md "İlk gerçek telefon".

## IPC, build ve kabul sırası

Sayfa yerleşimi için ilk C++ dilimi `VulkanMaterialTexturePages.h` içinde.
Tam eşleşme, cihaz/format katman sınırında bölme, taşmada uyumlu anahtarı
üst ebata taşıma ve birleştirme raporu hesaplanır.
`VulkanMaterialTextureImageLimits.h` format, usage ve flag başına
`vkGetPhysicalDeviceImageFormatProperties` sorgusunu yapar. Planlayıcı
cihaz ve formatın ebat/katman sınırlarını, mip zincirini ve
`maxResourceSize` sınırına göre doku yükünü denetler.
`VulkanMaterialTextureArrayImage.h` tracked image/view tahsisini ve çağıranın
komut tamponuna katman başına tam mip kopyasını kaydeder. Tahsis sırasında
aynı komut tamponuna tüm mip ve katmanlar için tek shader-read bariyeri kaydeder.
Boş katmanların layout'u geçerlidir ve bu katmanlara materyal referansı verilmez.
Upload, staging ofsetlerini formatın texel blok boyutuna göre kontrol eder.
Residency yeni katmanı yayınlamadan önce transferi sıralar ve canlı düzenlemede
eski katmanı son okuyucu fence'inden sonra geri alır.
`VulkanMaterialTextureResidency.h` tek sayfada bu kuralları uygular:
rezervasyon sürümü eski upload işlerini ayırır, transfer tamamlanmadan
yayına izin verilmez ve emekli katman fence seri numarası tamamlanana
kadar boş sayılmaz. `VulkanMaterialTextureUploadQueue.h` transfer batch'lerini
aynı Vulkan kuyruğuna fence ile submit eder, tamamlanana kadar staging ve
hedef image lease'lerini tutar. Kuyruk doluysa beklemek yerine `VK_NOT_READY`
döner. Transfer seri numarası frame okuyucu seri numarasından ayrıdır.
Canlı çağıranın kuyruk ve frame entegrasyonu ile sayfalar arası materyal
yayını henüz bağlanmadı. Residency zamanlayıcısı, descriptor,
shader ve canlı IPC bu yerleşime henüz bağlanmadı. Yerleşim test kaynağı
MSVC x64 ve iki Android NDK hedefinde syntax-only denetiminden geçti.

`UploadQueue::submitLayers` çağıranın açtığı batch içinde katman kopyalarını
ve `TRANSFER_WRITE → SHADER_READ` bariyerlerini kaydeder. Çağıran hedef
shader aşamalarını ve Vulkan kuyruğunun dış senkronizasyonunu sağlar. Bir
katmanın kaydı başarısız olursa batch gönderilmez. Kuyruk durumunun JSON
alanları `pending_uploads`, `completed_upload_serial` ve
`upload_not_ready_rejections`. Bunlar canlı IPC'ye henüz bağlanmadı.
`FramePublication::publish` tek sayfadaki katman geçişini ve materyal/descriptor
çiftini aynı render-thread çağrısında yayınlar. `canPublish` ve descriptor
doğrulaması kaynak hazırlığından önce çalışır. Her hazırlık için yeni bir
descriptor set ayrılır. Etkin veya uçuşta olan bir set aday olarak reddedilir.
Hazırlık başarısızsa etkin çift ve residency korunur. `previousLastReader`,
başarıyla gönderilen son okuyucu frame'in seri numarasıdır. Emekli katman ve
önceki çiftin kaynakları okuyucu fence'i tamamlanana kadar tutulur. Lenovo
renderer frame zamanlayıcısı henüz bu çağrıya bağlı değil.

Uygulamaya başlamadan önce çalışılacak dosyaları `AJAN_KOORDINASYON.md` §1'e
yaz. Descriptor ABI değişikliğini, bağlama sayıları ve varsa push constant
değişikliğiyle birlikte §2'ye kaydet.

Mevcut `render.status` ve `viewport.status` ile karşılık gelen scripting
durumuna aynı çekirdek snapshot'tan `texture_arrays` ekle. Render ve
viewport farklı VkDevice kullanıyorsa her yanıt kendi cihazını gösterir.
Alanlar `bucket_count`, `page_count`, `layers_used`, `layers_capacity`,
`payload_bytes`, `allocated_bytes`, `merged_keys`, `warnings`,
`image_format_queries`, `rejected_textures` ve tahsis
  tepesini görmek için `retired_pending_bytes`. `layers_valid`, doluluğun
  fiziksel kapasite içinde kaldığını ve tüm referansların yaşayan katmanlara
  işaret ettiğini bildirir. `descriptor_publications_rejected` reddedilen
  yayınların sayısını, `last_descriptor_rejection` son ret ayrıntısını taşır.
  İlk ret öncesi son ret alanı null olur. Sayfa ayrıntısında ebat, format, mip sayısı ve
doluluk yer alır. Başlatılmamış veya farklı backend durumu null olur.
`image_format_queries` profil başına sayısal Vulkan formatını, usage ve
create flag maskelerini, gerçek `vk_result` değerini ve sorgu özelliklerini
taşır. Başarısız sorguda özellikler null olur. Doku uyarısının `profile` alanı
bu tablodaki profile, `texture_index` ise yerleşim girdisindeki sıraya işaret
eder. `rejected_textures` reddedilen dokuların kendi listesidir. Bilinen
profillerde format, kullanım maskesi ve sorgu sonucunu da içerir. Örneğin
sorgu başarılı olup doku ebat sınırını aştıysa `vk_result` yine `VK_SUCCESS`
kalır. JSON üretimi hazır, canlı IPC bağlama residency snapshot'ıyla yapılacak.
IPC overlay güncellenir, `scripts/gen_ipc_descriptors.py` ile tablo
üretilir ve `scripts/audit_ipc_capabilities.py` çalıştırılır.

1. **Kabul için tam C++ derlemesi gerekir.** Uygulama partisi çıktığında bu
   uyarıyı `NEXT_BUILD_CHECKS.md` başına, build bekleyen işi koordinasyon
   panosunun §3'üne yaz. Baseline için mevcut Windows Studio sürümünü ve
   sahne dosyalarını sakla. Yeni ABI ile eski EXE/SPIR-V'yi eşleştirme.
2. Değişen shader girişlerini include grafiğinden çıkar, ilgili
   `compile_*.bat` kayıtlarına göre `glslc` ile derle. `spirv-val` ve
   ABI/capability kontrolünden sonra `RayTrophiStudio/source/shaders` ve
   `x64/Release/shaders` kopyalarının hash'lerini eşleştir. Bağlama ve
   push değişikliklerini koordinasyon panosu §2'ye kaydet.
3. Studio/Player MSVC `/Zs` ve NDK x86_64/arm64 syntax-only kontrolü.
   Kullanıcı Windows tam build ve Android paket build'lerini alır. İlgili
   kaynak listelerini ve Android shader staging'ini denetle.
4. Windows Studio'da workshop ve kids-play-room için dış IPC ile
   `render.start` karelerini aynı kamera, çözünürlük, seed/sample,
   ışık, pozlama ve simülasyon durumunda karşılaştır. Piksel eşitliği,
   değişen piksel oranı, kanal başına maksimum mutlak fark ve RMSE yaz.
   Gürültülü render varsa aynı sürüm tekrarıyla gürültü tabanını ölç.
   Yanlış albedo, normal, roughness veya opacity eşleşmesi makul görünebilir.
   Bu yüzden yakın/uzak yüzeyleri, cutout ve canlı paint düzenlemesini ayrıca
   karşılaştır. Scratchpad karelerini iş sonunda sil.
   A6 `raster_main_gpu_samples` ile aynı kare, çözünürlük ve ısınma sonrası
   örnek sayısında Windows önce/sonra raster GPU süresini ve yüzde farkını yaz.
   Probe compute GPU süresini de sabit probe bütçesiyle ölç. %5'i aşan artışı
   burada kaydet ve etkin sayfa sayısına göre ölü dalları kaldıran specialization
   constant varyantını pipeline hazırlama maliyetiyle birlikte değerlendir.
5. x86_64 emülatörde iki sahneyi aç. İlk kare, gerçek katman istatistikleri
   ve async düzenleme sırasında ilerleyen kare sayacı beklenir. Siyah veya
   durağan karede pipeline/format tanısı ile IPC snapshot'ını birlikte al.
6. Lenovo arm64 `adb -s HVA751MH` ile iki sahnede başlatıcıdan ilk kareye süreyi,
   sabit ölçüm aralığında FPS/frame time ve kullanılan çözünürlüğü kaydet.
   Sahne yükleme tanısını, varsa GPU ret nedenini ve `texture_arrays`
   değerlerini topla. Mali-G52'de 32 dallı shader'ın raster fragment ve
   probe compute GPU sürelerini ayrı raporla. Lenovo ölçümü yeni sürüm içindir.
   Windows canlı IPC ayrı süreçten named
   pipe, Android canlı IPC mevcut yerel soket istemcisiyle yürür.
