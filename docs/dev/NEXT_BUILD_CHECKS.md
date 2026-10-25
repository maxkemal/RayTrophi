# NEXT BUILD CHECKS

> **Durum:** CANLI — 2026-10-25 (büyüyen adım söküldü, radyans önbelleği varsayılan açık: derlenmedi; önbellek ve uyarlanan adımın sökülmesi canlı doğrulandı). Altındaki ıslak kum partileri 2026-10-23 kaydı olarak duruyor.









## Gaz march'ı: büyüyen adım söküldü (2026-10-25, derlenmedi; SHADER + C++)

RT_GAS_SECONDARY_MARCH.md §14. Radyans önbelleği canlı doğrulandı (−%53,
geri düşme 0); büyüyen adım örnek azaltmadı ve başlığı kararttı, söküldü.
Değişen shader: volume_closesthit. C++: VulkanBackend(_Volumes), RtApi, IPC,
Python, panel, descriptor. Kaynak: `check_volume_march_lod_contracts.py` → OK.

1. **Açılış değeri.** `render.volume_march_lod` beş alan döner:
   1/4/2/128 ve `secondary_radiance_cache` true. `*_transmittance_growth` yok.
   Panelde büyüme kaydırıcıları yok. Yedi alan dönüyorsa eski exe.
2. **Referans birebir.** 1/1/1/0, önbellek kapalı: önceki referansla aynı
   (nükleer sahne, düzlem açık ~500 ms). Farklıysa geri alma eksik
   (`grep growStride shaders` boş olmalı).
3. **Önbellek hâlâ çalışıyor.** Varsayılanda `secondary_density_samples`
   ≈ 0, `radiance_cache_fallbacks` ≈ 0; başlık referansla aynı tonda (büyüme
   onu −1.8/255 kararttıyordu).

## Gaz: sekme ışınları için radyans önbelleği + geçirgenliğe göre büyüyen adım (2026-10-25, derlenmedi; SHADER + C++)

RT_GAS_SECONDARY_MARCH.md §13. Yeni shader include:
`volume_radiance_cache.glsl` (vcxproj'a eklendi). Değişen shader'lar:
volume_closesthit, volume_light_cache.glsl (kanal 6 → 10),
volume_instrumentation.glsl (+2 sayaç, struct 248 → 256 B). C++: ışık önbelleği
kanal sayısı, VulkanBackend(_Volumes), RtApi, IPC, Python, panel, descriptor.
RT pipeline baştan derlenir. Kaynak kontrolleri:
`check_volume_march_lod_contracts.py` ve `check_volume_light_cache_contracts.py`
→ OK.

1. **Ayar ve ABI.** `render.volume_march_lod` yedi alan döner:
   1/4/2/128, `secondary_radiance_cache` true, büyüme 3/3. `render.volume_stats`
   içinde `radiance_cache_marches` / `radiance_cache_fallbacks` var. Sayaçlar
   saçmaysa shader ile C++ struct'ı hizasız. `render.volume_light_cache`
   `channels: 10` döner.
2. **Referans birebir.** 1/1/1/0, önbellek kapalı, büyüme 1/1: maliyet ve
   görüntü önceki referansla aynı. Farklıysa büyüme ya da önbellek kapalıyken
   de çalışıyor.
3. **Önbellek doluyor.** Varsayılanda, statik karede birkaç örnekten sonra
   `radiance_cache_fallbacks`, `radiance_cache_marches`'ın yanında küçük
   kalmalı; `secondary_density_samples` neredeyse sıfıra inmeli. Geri düşmeler
   yüksek kalıyorsa radyans etiketi her karede değişiyor (ışık durumu hash'i
   kararsız) ya da sahiplik bırakılmıyor.
4. **Işık önbelleği bozulmadı.** `light_cache_hits` / `light_cache_misses`
   eskisi gibi. Sahnede 5'ten fazla ışık varken 6.+ ışığın gölgesi march'tan
   gelmeli. ★ Kanal eşlemesi toplam kanal sayısını izlerse ışık 5–8 radyans
   kanallarına yazar: yanlış gölge, hata yok.
5. **Görüntü.** Nükleer sahne, düzlem açık; referans ile varsayılan aynı
   oturumda, aynı karede:
   - zemindeki dolaylı ateş ışığı;
   - dumanın gölgeli tarafı;
   - yoğun başlığın arkasındaki kenarlar.
   Hafif yumuşama beklenir; zeminin belirgin kararması ya da açılması
   beklenmez.
6. ★ **Sinsi olan: donmuş önbellek.** Işık ya da güneş hareket ettirildiğinde
   sekme ışınlarının gördüğü gaz eski aydınlatmada kalırsa, ışık durumu
   hash'i değişmeyi kaçırıyor demektir. Görüntü inandırıcı görünür ama yanlış.
   Kontrol: güneşi döndür; zemindeki dolaylı ışık birkaç örnek içinde yeni
   yöne uymalı.

## Gaz march'ı: uyarlanan adım söküldü, varsayılan 1/4/2/128 (2026-10-25, derlenmedi; SHADER + C++)

RT_GAS_SECONDARY_MARCH.md §12. Önceki derlemedeki uyarlanan adım canlı
ölçüldü: daha çok örnek aldı ve görüntüyü açtı, söküldü. Değişen shader:
volume_closesthit. C++: VulkanBackend(_Volumes), RtApi, IPC, Python, panel,
descriptor. Kaynak: `python scripts/test/check_volume_march_lod_contracts.py` → OK.

1. **Açılış değeri.** `render.volume_march_lod` dört alan döner:
   1/4/2/128, `adaptive_*` alanı yok. Panelde uyarlanan adım kaydırıcıları
   yok. Yedi alan dönüyorsa eski exe.
2. **Referans birebir.** 1/1/1/0'da maliyet ve görüntü uyarlanan adımdan
   önceki referansla aynı. Fark varsa geri alma eksik
   (`grep segStride shaders` boş olmalı).
3. **Varsayılan.** Nükleer sahne, düzlem açık: referansa göre ~−%30, zemin
   referansla aynı ton (ortalama fark ≈ +1/255). Zemin belirgin açıksa sınır
   hâlâ 32.

## Obje görünürlüğü çekirdekte: `scene.set_visible` / `scene.get_visible` (2026-10-25, derlenmedi)

Outliner gözü artık `rtapi::setObjectVisible` çağırıyor (yeni dosya
`RtApiObjectVisibility.cpp`, vcxproj'a eklendi). IPC `scene.set_visible`
(SceneWrite) + `scene.get_visible` (Read, backend maske okuması dahil),
`scene.object_info.visible`, Python `rt.scene.set_visible/get_visible`.
Test: `python scripts/test/rt_object_visibility_ipc.py --object Plane_1`.

1. **Derleme + `agent.discover`**: `scene.set_visible` ve `scene.get_visible`
   listede, ikisi de documented. Yoksa eski exe ya da üretilmiş tablo bayat
   (`python scripts/audit_ipc_capabilities.py` yeşil olmalı).
2. **Bilinmeyen ad reddedilir**: `scene.set_visible {name:"yok", visible:false}`
   hata döner. Başarı dönüyorsa `forEachEntry` boş eşleşmeyi saymıyor —
   script'ler yanlış adla "gizledim" sanır.
3. **Panel gözü hâlâ çalışıyor** (tek obje, grup, Characters/Models): göz
   kapanınca obje viewport'tan kalkar, açınca geri gelir. Çalışmıyorsa UI →
   rtapi yönlendirmesi kırık (`setHierarchyObjectVisibility` artık yalnızca
   `rtapi::setObjectVisible` çağırıyor; `rtapi::g_ctx` bağlı mı?).
   Characters/Models gözü eskiden yalnız `member->visible` çeviriyordu ve
   backend maskelerine hiç ulaşmıyordu — artık ulaşmalı.
4. **Test script'i PASS** (Plane_1, Vulkan RT açık): `hidden:` satırlarında
   `viewport_raster`, `viewport_rt`, `render_rt` maskeleri 0/N, `pick_gpu`
   objeyi döndürmez; `restored:` satırlarında N/N. `SKIP` = o backend'de
   read-back yok (-1) ya da obje hiç kurulmamış (0 instance) — PASS değil.
5. **★ EN SİNSİ: viewport'ta gizli, RT render'da görünür (ya da tersi).**
   Sessizce makul görünür çünkü baktığın pencere doğru davranır. Ölç:
   `scene.get_visible` → `viewport_raster_visible` ile `render_rt_visible`
   gizlenmiş objede İKİSİ DE 0 olmalı. Gözle: objeyi gizle, final render al
   (`render.start`), objenin gölgesi/yansıması da kalkmış olmalı. Biri 0
   diğeri değilse iki rol (`g_viewport_backend` / `g_backend`) tek backend
   sanılıp ikincisine `setVisibilityByNodeName` gitmemiştir
   (`render_is_viewport` alanına bak).
6. **★★ Alan raporu (2026-10-25, Nuclear Detonation / Plane_1): "göster"
   ilk seferde RT'ye ulaşmıyordu.** Kök: Vulkan `updateGeometry`, raster
   rebuild ve OptiX snapshot GİZLİ objeyi tamamen ATLAR — gizliyken herhangi
   bir rebuild olduysa objenin instance'ı yoktur ve `setVisibilityByNodeName
   (true)` maskeleyecek bir şey bulamayıp sessizce hiçbir şey yapmaz; obje
   alakasız bir sonraki rebuild'e kadar yoktur. Düzeltme: `setObjectVisible`
   gösterirken backend read-back'te ad için 0 instance (backend'de başka
   instance varken) görürse rebuild planlar; OptiX'te read-back yok, her
   göstermede OptiX rebuild planlanır. Ölç: test script'inin `rebuilt:` ve
   `restored:` fazları (`--rt-counters` ile RT sonucu da: yüzey sekmesi/yol
   gizlide değişmeli, restore'da baseline'a dönmeli). `restored: ... 0
   instances, baseline had N` FAIL'i tam bu hatadır. Log'da
   `[Visibility] ... geometry rebuild scheduled` görülmeli.
   Aynı testte gizliyken outliner gözü kapalı kalmalı: `rebuildMeshCache`
   rep facade'a `tmesh->visible` kopyalıyor; göz açık görünüyorsa o satır
   çalışmıyor. **H tuşu (hepsini göster) aynı hataya açık** — `showAllInstances`
   da düşmüş instance'ı geri getiremez; bu partide dokunulmadı.
7. **Pending-delete**: silinmiş (undo bekleyen) objeye `visible:true` → hata,
   obje gizli kalır.
8. **Bilinen sınır**: `scene.raycast` görünürlüğe bakmaz (snap yüzeyi
   sorgusu, eskisi gibi); gizli objeye hâlâ isabet etmesi bug değil. Proje
   kaydında görünürlük yazılıyor mu bu partide değişmedi.

## Kamera ışık adımı varsayılanı 1 (2026-10-25, derlenmedi; yalnız C++ varsayılanları)

1. **Açılışta `render.volume_march_lod` 1/4/4/32 döner**, panelde Camera Light
   Stride 1. 2 dönüyorsa eski exe. Kutunun gölgeli alt yüzünde yatay bant
   yok. Bant varsa kamera adımı hâlâ 2, ya da başka bir şey bant çiziyor
   (ayar 1/1/1/0 ile kontrol et).

## Gaz march LOD'u: ışık adımı + ikincil adım (2026-10-25, CANLI: splat+gaz −%47, tam sahne −%39; kamera varsayılanı 1 derleme bekliyor)

RT_GAS_SECONDARY_MARCH.md §8. Değişen shader'lar: volume_closesthit (march),
closesthit, hair_closesthit, raygen, rt_sky.glsl (yalnız dünya aynası
yeniden adlandırıldı). C++: vulkan_world_data.h, VulkanBackend(_Volumes),
RtApi, IPC, Python, yetki, panel, descriptor. RT pipeline baştan derlenir.
Kaynak: `python scripts/test/check_volume_march_lod_contracts.py` → OK.

1. **Ayar okunuyor.** `render.volume_march_lod` 2/4/4/32 döner; panelde dört
   kaydırıcı aynı değerleri gösterir. `render.set_volume_march_lod
   camera_light_stride=17` reddedilir. Okunmuyorsa ya da 0'lar dönüyorsa
   IPC/yetki zinciri eksik.
2. **Referans birebir.** 1/1/1/0'da görüntü ve maliyet bu partiden önceki
   gibi: splat_gas ~286, gas_only ~93 ms. Farklıysa kelime yanlış çözülüyor
   (0 = eski march olmalı) ya da `volumeShadowDue` k = 1'de eski koşul değil.
3. **Kazanç.** `python scripts/test/rt_volume_march_lod_ab_ipc.py`: splat_gas
   belirgin düşer (hedef ~150–180 ms); gas_only bir miktar düşer; splat_only
   aynı kalır. splat_gas'ta `secondary_density_samples` ~1/3'e iner,
   `camera_density_samples` değişmez.
4. **Görüntü.** Referans ile varsayılanı aynı kamerada karşılaştır:
   - doğrudan görünen gazın yumuşak gölgeleri ve ışık konisi kenarı;
   - gaz içindeki splat'ların dolaylı aydınlanması.
   Kamera tarafında hafif yumuşama beklenir; bant ya da basamak beklenmez.
5. ★ **Sinsi olan: donmuş gölge.** Gazın içinde, ışığa göre gölgede kalan
   bölge 2/4/4/32'de referanstan belirgin AYDINLIK ya da bütün ışın boyunca
   tek tonda görünüyorsa, önbelleklenen gölge ilk değerinde kalmış demektir
   (`volumeShadowDue` yenilemiyor). Bu "LOD biraz yumuşattı" gibi okunur ama
   yanlış ışıktır. Kontrol: `camera_light_stride` 1 ile 4 arasında değiştir;
   gölgeli bölge yalnız yumuşamalı, kaybolmamalı.
6. **Panel.** Kaydırıcı bırakılınca uygulanır ve birikim sıfırlanır; sürükleme
   sırasında her karede yeniden yükleme olmamalı.

## Gaz march'ı: kamera / ikincil sayaç ayrımı (2026-10-25, CANLI: toplam tam, ikincil pay %45.5 → ölçüt kaldı)

RT_GAS_SECONDARY_MARCH.md §4 adım 0. Yalnız ölçü aleti; görüntü ve maliyet
değişmemeli. Değişen shader'lar: volume_instrumentation.glsl,
volume_closesthit, closesthit (küre varyantı dahil).
C++: VulkanVolumeInstrumentation.h (224 → 248 B), RtApi, IPC, Python, panel,
descriptor.

1. **ABI tutuyor.** `render.volume_counters enabled=true`, sonra
   `render.volume_stats`. Eski alanlar önceki ölçümle aynı mertebede:
   splat + gazda yol başına `density_samples` ~33. Hepsi sıfır ya da saçma
   büyük sayılarsa shader ile C++ struct'ı hizasız; ikisi birlikte derlenmedi.
2. **Sınıflar toplamı kapatıyor.** `camera_density_samples +
   secondary_density_samples == density_samples`; aynısı rays ve shadow için.
   Eksik kalıyorsa sınıflanmamış örnek var: kayıt çağıran bir shader main()'de
   `volumeInstrumentationSetSegment` çağırmıyor.
3. **Yalnız gazda kamera ağır basıyor.** gas_only'de kamera payı büyük olmalı.
   Kamera bitinin anlamı doğru mu, buna bakılır.
4. ★ **Sinsi olan: ikincil sıfır.** splat_gas'ta `secondary_density_samples`
   0 ise toplam yine tutar (madde 2 geçer), ama kamera biti sekmelerde
   temizlenmiyor demektir. Bu durumda "ikincil pay yok, LOD işe yaramaz" diye
   yanlış karar çıkar. Splat'lardan seken ışınlar gaza giriyor: ikincil sayı
   sıfır olamaz.
5. **Karar ölçütü.** splat_gas'ta ikincil örnek payı > %60 ise LOD (adım 1)
   doğru hedefte.

## Gaz içindeki splat'lar: iç içe katı izi GERİ ALINDI (2026-10-25, CANLI DOĞRULANDI: 27 / 97 / 295 ms)

RT_GAS_OPTICAL_PERFORMANCE.md §"Seçenek A ölçüldü: gerileme, geri alındı".
Geri dönen shader'lar: volume_closesthit, closesthit (küre varyantı dahil),
hair_closesthit, miss, rt_payload.glsl. C++: özyineleme derinliği yeniden 2.
RT pipeline baştan derlenir.

1. **Maliyet eski yerine döndü.** Aynı sahne, kare 28, `fluid.set_phase_output`:
   gas_only ~94 ms, splat_gas ~311 ms, splat_only ~25 ms. A'lı derlemede
   265 / 406 / 25 ms ölçüldü. gas_only hâlâ ~265 ise sebep A değil; aynı ağaçtaki
   başka bir değişiklik (paralel ajan) ya da eski .spv. Önce .spv zaman damgasına bak.
2. **Görüntü.** Gaz içindeki splat'lar görünür, sonraki sekmenin gökyüzü yerinde.
   Bu yol A'dan önce canlı doğrulanmıştı; bozuksa geri alma eksik kalmış demektir
   (`grep PL_NESTED_SOLID shaders` boş olmalı).

## Hacim kutuları dolu hücrelere daralır: sıvı SurfaceSDF + host gaz (2026-10-25, derlenmedi; yalnız C++)

Matter Fog akışı daraltması önceki build'de canlı doğrulandı (kutular dar geldi).
Bu parti aynısını sıvı SurfaceSDF / host gaz yayınına ekler.
RT_GAS_OPTICAL_PERFORMANCE.md §"Devam: ilk sahne". Ölçüm: aynı sahne, kare 28.

1. **SDF kutusu daraldı.** Katı splat, gaz hidden, sıvı sdf iken
   `render.volume_slots`: "[Fluid NanoVDB]" slotu artık [0,5]³ değil, su ve
   köpüğün etrafında küçük bir kutu. Daralmadıysa: yükleme bu karede olmadı ve
   kutu hiç hesaplanmadı (bir sim adımı ilerlet), ya da canlı GPU yolu.
2. **Maliyet.** Aynı rota (splat / sdf / hidden) 135 ms'den ~26 ms'ye (boş
   sahne) yaklaşır; tam inmez, çünkü suyun kendisi gerçek bir yüzey.
3. **Su görünümü aynı.** Kenarda kesilmiş su yüzeyi ya da köpük = kutu fazla dar.
4. ★ **Sinsi olan:** sahne kaydedilip yeniden açılınca, ya da sim geri sarılınca
   su kutusu yeniden hesaplanmadan eski dar kutu kalırsa, yeni su o kutunun
   dışında GÖRÜNMEZ ve hiçbir hata vermez. Geri sar → oynat → suyun yeni
   konumunda göründüğünü kontrol et.
5. **Gaz (host yolu).** Host'tan yüklenen gaz domain'lerinde de kutu daralır;
   canlı GPU gazında değişmez.

## Hacim ışık önbelleği — A/B anahtarı söküldü (2026-10-25, derlenmedi; yalnız C++)

V1 canlı kabul edildi (docs/dev/VOLUME_LIGHT_CACHE.md §11–12). Bu parti yalnız
`enabled`'ı söker; shader değişmedi, RT pipeline yeniden derlenmez.
Kaynak kontrolü: `python scripts/test/check_volume_light_cache_contracts.py` → OK.

1. **Derleme temiz.** `VolumeLightCacheConfig` hiçbir yerde kalmamış olmalı;
   kalmışsa bağlayıcı/derleyici hatası verir.
2. **Panel.** Performance'ta onay kutusu yok, "Light Cache Cells" var; değiştirince
   `render.volume_light_cache.max_cells` aynı değeri gösterir.
3. **IPC.** `render.set_volume_light_cache {max_cells: 48}` çalışır,
   `{enabled: false}` "max_cells" eksik diye reddedilir. `render.volume_light_cache`
   çıktısında `enabled` alanı yok.
4. ★ **Sinsi olan:** önbellek gerçekten açık mı? Counters açıkken duran karede
   `light_cache_hits` > 0 ve misses ≈ 0. Anahtar sökülürken `wanted` koşulu
   yanlış kalsaydı hiçbir hata olmaz, yalnızca eski hıza dönülürdü.

## Islak kum 4. parti — tane hash kovası 16 → 32 (2026-10-23, derlenmedi; C++ + shader)

Canlı (teğet seed build'i): 50 000 ıslak Sand tanesi (ıslaklık 1.0) düşerken
"grain neighbour bucket overflow (more than 16 grains hashed to one bucket)"
ile tutuldu — havada, küp collider'ın 35 cm üstünde (çarpma değil; sertlik ×4
de aynı). Sebep: ıslak hash hücresi 3r, yoğun seed kafes adımı 1.474r → kötü
hizada hücrede ~14 merkez (meşru); 16'da tek hash çakışması taşırıyor.
Değişiklik: `kMatterGrainBucketCapacity` 32, shader `BUCKET` 32, revizyon 31
(32-bit indeks 2^31'e sığar; 50k tanede kova belleği ~18 → ~35 MB).

1. **Shader derlendi:** `sim_matter_grain` revizyon 31; eski .spv ile adım
   "shader revision mismatch" der.
2. **Aynı sahne:** 50k ıslak Sand bloğu düşüp yere çarpar, tutulma yok.
3. ★ **Sinsi olan:** "neighbour list full" (bit 8) — 32'lik Verlet listesi
   yoğun ıslak paketlemede dolabilir (3r içinde FCC ~18 komşu, iç içe
   geçmede daha fazla); retry var ama taşarsa yine tutar.

## Islak kum 3. parti — teğet seed (2026-10-23, derlenmedi; yalnız C++)

FCC aralığı 2.2002R → 2.0851R (jitter 0.07R → 0.03R): boşluk 0.2R → 0.085R,
doluluk 0.556 → 0.653. Not: ISLAK_KUM_DOGRULAMA.md 3. parti + DEVİR bölümü.

1. **C++ testi** `matter_grain_birth_test.cpp`: 172 tane, ≥ 2R ayrılık geçmeli.
2. **Seed sayısı:** 0.3×0.5×0.3 m Sand, r 12.5 mm → ~3300 (önce 2813).
3. **İlk kare sakin:** seed sonrası 5 adımda ortalama hız 0.52 m/s değil, küçük
   (blok bulut gibi düşmüyor); `fluid.get settle_converged` true olmalı.
   ★ **En sinsi sonuç:** kuru blok ayakta kalıyor — FCC kilitlenmesi; gerçek
   kuru kum çöker. Öyleyse jitter'ı ölçerek büyüt (d formülüyle birlikte).
4. **Islak kısa kolon** (0.2 m, ıslaklık 0.3, Sand r 0.25 mm): önceki 0.029'dan
   yüksek olmalı; hâlâ çöküyorsa yapışma ölçeği (DEVİR maddesi) sırada.

## Islak kum 2. parti — FCC seed + köşeli kum (2026-10-23, derlenmedi; yalnız C++)

Canlı (18:09): seed ıslaklığı çalışıyor (wet_grains, 8834 köprü), ama seed'li
blok kuru da ıslak da tek tane kalınlığına yayıldı. Kökler (ISLAK_KUM notu):
Sand μ_r 0.02 (kuru döküm ~8°; 0.2 ile 30°, ıslak 47°) ve basit kübik seed
(0.39, kayma direnci yok). Değişiklik: seed FCC (0.556), Sand μ_r 0.2 + su
kapasitesi 0.1, Gravel μ_r 0.25; `matter_grain_birth_test.cpp` güncellendi.

Sıra:

1. **C++ testi:** `scripts/test/matter_grain_birth_test.cpp` geçmeli (172 tane,
   ≥2R ayrılık, voxel bağımsızlığı, bütçe öneki).
2. **Seed yoğunluğu:** Sand seed 0.3×0.5×0.3 m, r 12.5 mm → ~2600–2900 tane
   (önce 1800). *Bozuksa:* hâlâ 1800 = eski exe.
3. **Kolon:** `.\scripts\ipc\Probe-WetSandBlock.ps1 -Case column -Wetness 0,0.3`
   — artık türetme gerekmez ama probe WetSandProbe kullanır (Sand + kapasite
   0.1; μ_r Sand'den 0.2 gelir). Kuru: yığına çöker (merkez ~0.06–0.08);
   ıslak 0.3: belirgin yüksek (ayakta ya da dik). ★ **En sinsi sonuç:** kuru
   kolon da ayakta kalır → FCC kristal kilitlenmesi (gerçek kumda yok); o zaman
   jitter'ı ölçerek büyütmek gerekir.
4. **Kaya:** `-Case rock` — ıslak blok topak halinde çatlamalı.
5. **Regresyon:** kuru Sand dökümü artık ~30° (önce düz yayılıyordu); Sand +
   Water sahnelerinde taneler ıslanır (köprü) — tasarım gereği.

## Islak kum 1. parti — seed'e başlangıç ıslaklığı (2026-10-23, derlenmedi; yalnız C++)

★ Önce: kullanıcı "derledim" dedi ama `x64/Release/RayTrophiStudio.exe` ve
`RayTrophiStudio.log` 16:10 — kar 6. parti (tek Snow, substep paneli) bu
exe'de YOK (madde listesinde Packed Snow duruyor). Bu build ikisini birden getirir.

Bulgular (canlı, 16:10 exe):
- Sand = DEM (`granular_transport dem`), `grain_water_capacity_fraction = 0` →
  Sand hiç ıslanamaz; ıslak kum için türetip kapasite vermek gerekiyor.
- Islak tane yalnız akış kaynağından (`grain_birth_saturation`) doğuyordu;
  `fluid.seed` hep KURU → ıslak, sıkıştırılmış blok yazılamıyordu.
- Kalıp testi: 3600 ıslak tane döküldü (wet_grains true, merkez 0.277 m), ama
  kalıp collider'larını kapatmak simülasyonu 0. kareye sardı → 0 parçacık.
  Sim sürerken collider/kaynak düzenlemek = geri sarma; test tek kurulum olmalı.
- Köprü kuvveti ölçeklemesi doğru: F = 2πγcosθ·R·(R/r) (2026-10-21).

Değişiklik: domain seed tarifine `fluid_seed_grain_saturation` (0..1 kapasite);
seed DEM tanelerine `initMatterGrainBirthWater` ile su verir; ownership bunu
ıslak doğum sayar. Kayıt (proje + sahne), imza hash'i, `fluid.seed
grain_saturation`, `rt.fluid.seed(grain_saturation=)`, `fluid.get
seed_grain_saturation`, panelde "Seed Grain Wetness" (yalnız tane domain'inde).

Sıra:

1. **Exe yeni mi?** `RayTrophiStudio.exe` tarihi > 16:10, madde listesinde
   Packed Snow YOK. Değilse aşağıdakilerin hiçbiri koşamaz.
2. **Kolon testi:** `.\scripts\ipc\Probe-WetSandBlock.ps1 -Case column`
   (kuru 0 ve ıslak 0.3). Kuru: merkez 0.25'ten belirgin düşer (yayılır);
   ıslak: ~0.25'te kalır (ayakta). `wet_grains` ıslakta True, kuruda False.
   ★ **En sinsi sonuç:** ikisi de aynı düşüyor ve wet_grains True — seed suyu
   veriliyor ama köprü kuvveti tutmuyor (yapışma bu ölçekte zayıf); ikisi de
   ayakta → kuru kum da yapışıyor (sürtünme/uyku tutuyor), o da yanlış.
3. **Kaya testi:** `-Case rock`: ıslak blok kayanın üstünde topak halinde
   kalmalı/çatlamalı (merkez ~0.45), kuru blok kayanın etrafına akmalı.
4. **Panel:** tane domain'inde seed bölümünde "Seed Grain Wetness" kaydırıcısı;
   değiştirip bırakınca yeniden seed eder; proje kaydedip açınca değer kalır.

## Kar 6. parti — tek Snow, substep yalanı (2026-10-23, derlenmedi; yalnız C++)

Kullanıcı: "iki kar ön ayarı kafa karıştırır" ve "UI alt adımı 12 gösterse de
200 çıkıyor". Ölçüm: panelde görünen tek sayı **Max Advection Substeps = 12**
(`APICSolverParams::max_substeps`) — Matter granular yolunda hiçbir şeyi
sınırlamıyor (her ortak alt adımın içindeki taşımayı sınırlar, orada zaten 1).
Gerçek sayıyı elastik dalga CFL'si veriyor: 1/24 s ÷ (0.35 × 0.025 m ÷
√(1.4e5·5 / 400) = 41.8 m/s) = 200. Panel yalnız "adaptive" diyordu.

Değişiklik: `Packed Snow` kaldırıldı (kimse türetmemişti; ad hash etiketi ⇒
kayıtlı projeye etkisi yok). `Granular::planGranularSubsteps` (planlayıcıyla
aynı aritmetik) → panel "Granular Skeleton" başlığında: dinlenmede substep,
son kare, formül (kare dt / (0.35·voxel / dalga hızı)), dalga hızının
bileşenleri. Advection kaydırıcısının altında granular için açıklama. IPC/Python:
`granular_predicted_substeps`, `granular_wave_speed_m_s`,
`granular_sizing_young_modulus`, `granular_frame_dt`.

Sıra:

1. **Madde listesi:** yalnız `Snow`; `Packed Snow` yok.
2. **Panel = koşu.** Snow domain'i (voxel 0.025, 24 fps): "Solver substeps per
   frame: 200 at rest | last frame: 200". `fluid.get granular_predicted_substeps`
   200. *Bozuksa:* tahmin ≠ son kare (dinlenmede) ⇒ planlayıcı ile UI farklı
   formül kullanıyor (ikisi de `planGranularSubsteps`/`elasticStepInfo` olmalı).
3. **Advection kaydırıcısı** altında "Granular material: the solver substep
   count is set by its stiffness…" satırı (yalnız granular domain'de).
4. ★ **Sinsi olan:** çarpma anında "last frame" tahminden BÜYÜK olabilir
   (gerinme hızı terimi) — bu doğru; dinlenmede eşit olmalı.

## Kar 5. parti — sertleşme tabanı (2026-10-23, derlenmedi)

Build: **C++ VE `compile_shaders.bat`** (stress_update; push constant boyutu aynı,
son pad → `snow_min_hardening`). Yeni madde alanı `granular_snow_min_hardening`
(Snow 0.1). `detached` artık "tabana inmiş" demek (eskiden %99 kayıp).

4. parti canlı (15:43 exe): topak yapısı atandı (J_P 0.952..1.105), kaya sonucu
DEĞİŞMEDİ. Hız taraması 2.3 / 3.3 / 4.6 m/s → hepsinde ~%70 yumuşama, %26 toz
(enerjiden bağımsız). Kayaya ÇARPMADAN konan blok 1.5 s'de dağıldı (varsayılan
E); E 7e5 ile aynı blok sağlam durdu. Görsel (ekran görüntüsü, silindi): kaya
bloğu DELİP içine girmiş, blok kayayı sarıp zemine inmiş, dışarıdan sağlam küp
gibi; sayaçlardaki "toz" kayanın çevresindeki kesme bölgesiydi. Kütle merkezi
y 0.305 (kaya üstünde dururken beklenen ~0.39).

Kök: exp(ξ(1−J_P)) bütün modülleri (sıkışma dahil) çarpıyor ve tabanı yoktu;
gevşeyen kar basınçsız toz oldu, engelin etrafından aktı.

Sıra:

1. **Shader derlendi** (`.spv` > `.glsl` tarihi).
2. **Delip geçme bitti.** `Probe-SnowImpact.ps1 -Target rock`: `centroid y`
   0.305'ten belirgin yüksek (0.35+). `powder` sütunu artık tabandaki
   parçacıklar; çatlak yüzeyleri kadar olmalı, bloğun yarısı değil.
   ★ **En sinsi sonuç:** merkez yükselir ama blok tek parça kayanın üstünde
   asılı kalır (çatlamaz). Görsele bak: kenarlar aşağı kopmalı.
3. **Durgun kaya testi** (varsayılan E): kayanın üstüne konan blok dağılmamalı
   ya da yalnız sarkan kenarları kopmalı.
4. **Zemin regresyonu:** `-Target floor` önceki gibi (ortalama J_P ~0.68).
5. **Packed Snow (karar verildi: Snow toz kar kalır).** Yeni yerleşik madde
   `Packed Snow`: aynı yasa, ρ 550, E 1e6, iletkenlik 0.45; katalogda Snow'un
   yanında, görünümü Snow ile aynı. Test: `fluid.set_param SnowTest
   default_substance="Packed Snow"` → `Probe-SnowImpact.ps1 -Target rock`:
   kaya üstünde konan blok sağlam durmalı (E 7e5 ölçümü gibi), 4.6 m/s'de
   kenarlar kopmalı, `centroid y` ~0.39'a yakın. Substep Snow'un ~2.3 katı
   (voxel 0.025: ~180).
   *Bozuksa:* madde listede yoksa eski exe; Snow ile aynı davranıyorsa E
   okunmuyor (`fluid.get granular_young_modulus` 1e6 olmalı).

## Kar 4. parti — Voronoi topak yapısı (2026-10-23, derlenmedi; yalnız C++)

Build: yalnız C++ (shader değişmedi). Madde alanı yeni: `granular_snow_clump_size`
(m, varsayılan 0.12); Snow dağılımı 0.05 → 0.1.

3. parti canlı (15:14 exe, 15:11 .spv; `Probe-SnowImpact.ps1`, voxel 0.025):
buz tabanı PASS, `hcap` 0. Zemin: 5.6 → 0.5 m/s iki adımda (FLIP'siz 0.17),
ortalama J_P 0.686. Kaya: %65 parçacık %50+ yumuşamış, %20–45 toz — FLIP öncesiyle
aynı. Mandal yalnız ~12 basınçsız yüzey parçacığında (sönüm basınç kapılı).
Homojenlik testi: bağımsız (beyaz) dağılım 0.05 → 0.3 hiçbir şeyi değiştirmedi →
sorun genlik değil İLİŞKİ UZUNLUĞU (P2G ~50 parçacığı ortalıyor).

Sıra:

1. **Yapı atandı.** `Probe-SnowImpact.ps1 -Target floor` ilk satırında J_P aralığı
   ≈ e^±0.1 (0.905 .. 1.105) ve ortalaması 1'den biraz büyük (sınırlar gevşek).
   *Bozuksa:* 0.951..1.051 = eski exe (beyaz gürültü, 0.05).
2. **Kaya: topaklar kalıyor (asıl hedef).** `-Target rock`: `powder` ve `d90`
   3. partiden (yaklaşık 8 000 / 10 000) belirgin düşük, `d50` ise yüksek kalabilir
   (çatlak yüzeyleri). ★ **En sinsi sonuç:** sayılar düşer ama kar hâlâ tek
   kütle — çatlak açılmamış, yalnız kırılganlık azalmış. Ayırt etmek için
   görsel bak: sınırlar boyunca ayrılan 10–15 cm'lik parçalar olmalı.
3. **Zemin:** blok yine sıkışır (beklenen); kenarlardan topak kopması varsa artı.
4. **Topak boyutu duyarlılığı:** SnowProbe gibi türetilmiş bir maddede
   `granular_snow_clump_size` 0.06 / 0.2 dene; parça boyu onunla değişmeli.
   0.1 m voxelde 0.12 → 0.3 m'ye (3 voxel) yükseltilir; bu bilinçli.

## Kar 3. parti — FLIP + yavaş parçacık sönümü (2026-10-23, derlenmedi)

Build: **C++ VE `compile_shaders.bat`**. ★ Önce şunu bil: 15:03 exe'si kar
2. partinin C++'ını içeriyor ama `.spv`'ler 13:15 tarihli — tavan ve buz tabanı
kernel'de YOK. Canlıda J_P 0.303'e indi (taban 0.436) ve durgun blokta %76
parçacık `hcap` bayrağı taşıdı. Bu partinin değişen shader'ları:
`sim_fluid_granular_stress_update.glsl` (2. parti), `sim_fluid_granular_settle.glsl`
(push constant boyutu aynı, pad0 → `damping_max_speed`).

Canlı ölçüm (`scripts/ipc/Probe-SnowImpact.ps1`, voxel 0.025, 17 576 parçacık,
0.3 m blok ~2 m'den, ~6 m/s; adım 49 ms, 80 substep):
düz zemin → ortalama J_P 0.68, 13 parçacık %50+ sertlik kaybı, dinlenmede J_P max
1.06 → 1.17 tırmanıyor (titreşim mandalı); küre kaya → ortalama J_P 1.73, %67
parçacık %50+ kayıp, %49 toz. Model kırılıyor (geometri ayrımı PASS), ama düz
zeminde ölü durma ve kayada topaksız toza dönme.

Değişiklik: Matter yolunda kar için FLIP (Snow `solver_flip_blend` 0.95; P2G
gerilme impulsundan ÖNCE taban kopyası, aynı ağırlıklarla normalize;
`granularFlipSnapshotWanted`), settle sönümü kar için yalnız < 0.5 m/s parçacıklara.

Sıra:

1. **Shader'lar derlendi.** `.spv` tarihleri `.glsl`'lerden yeni. Bozuksa her
   şeyi maskeler.
2. **Kar 2. parti kernel'i.** `Probe-SnowImpact.ps1 -Target floor`: J_P floor
   PASS (≥ 0.436), `hcap` ~0.
3. **Titreşim mandalı yok.** Aynı koşuda "rest ratchet" PASS: dinlenmede J_P max
   sabit. *Bozuksa:* settle `damping_max_speed` okunmuyor (eski settle .spv).
4. **FLIP etkin.** Kaya vakasında çarpmadan sonra `v_max` 3. partiden yüksek
   kalmalı (parçalar uçuyor); düz zeminde blok "ölü durmamalı" (çarpma sonrası
   ilk örnekte v_mean önceki 0.17'den belirgin büyük).
   ★ **En sinsi sonuç:** FLIP açık görünür ama taban alanı boş/sıfırsa G2P
   v_p + v_new − 0 yapar → kar ivmelenip patlar ya da `invalid` > 0. Patlama
   görürsen önce scratch alanın dolup dolmadığına bak.
5. **Kayada topak kalıyor mu.** `d90`/`powder` 3. partiden (8 662 / 17 576)
   belirgin düşük olmalı; hâlâ yarısı tozsa θs (0.0075) bu çözünürlük için
   fazla kırılgan — ayarı ölçerek değiştir.
6. **Kum regresyonu.** Sand/Gravel (DP) domain'inde FLIP KAPALI kalmalı
   (`flip_blend` yalnız snow'da okunur): yığın davranışı değişmemeli.

## Ayrı parti — ölçeklenmiş küp + collider takibi (2026-10-23, derlenmedi)

Build: yalnız C++ (shader yok, yeni .cpp yok). Dosyalar: `scene_data.h`,
`SceneSelection.h/.cpp`, `scene_ui_hierarchy.cpp`, `RtApi.h`, `RtApiFluid.cpp`,
`RtIpc.cpp`, `RtPython.cpp`.

Kökler:
- **Küp eski boyuta dönüyor** = inspector'ın Pos/Rot/Scl alanları. Herhangi bir
  alan değişince `fromTRS(pos, rot, scale)` TÜMÜYLE handle'a yazılıyor; bu üçlü
  seçim anında alınmış bir KOPYA (`sel.selected`). IPC `scene.set_transform`,
  undo, script onu hiç tazelemiyor; gizmo da bir kare geriden tazeliyor
  (`getPivotMatrix()` yeni matris uygulanmadan ÖNCE okunuyor). Sonra Pos'a
  dokunmak eski ölçeği geri yazıyordu. IPC yolu kendisi temiz: canlıda
  ölçek 2 → taşı → ölçek 2 kaldı. Düzeltme: alanlar çizilmeden önce
  `syncObjectTransformFromHandle()` canlı handle'dan tazeler (önbellek hâlâ
  aynı matrisi veriyorsa Euler korunur).
- **Collider eski pozda kalıyor** = `getSurfaceMeshCacheForObject` epoch memo'su.
  Epoch yalnız geometri jenerasyonu ya da keyframe re-pose ile bitiyordu; düz
  transform düzenlemesi (inspector, gizmo, `TransformCommand`=IPC/undo, script)
  bitirmiyordu. Tek istisna seçili obje + duraklatılmış timeline (gizmo
  çiziminde). Düzeltme: memo, handle'ın build anındaki matrisiyle eşleştikçe
  geçerli (tüketici tazeliği kontrol eder; tüm yollar kapsandı).
- **IPC körlüğü**: `collider.list/get` yalnız yazılmış `bounds_min/max`'ı
  (obb modunda KULLANILMAYAN -1..1) gösteriyordu. Yeni salt-okunur alanlar:
  `resolved_valid`, `resolved_world_min/max`, `resolved_size` — çözücülerin
  kullandığı resolver'dan.

Sıralı kontrol (bağımsız/hızlı önce):

1. **IPC readback var mı** — `Invoke-RtIpc collider.list @{}` →
   `Default_Cube Collider` satırında `resolved_valid: true`, `resolved_size`
   küpün ölçeğine eşit (ölçek 1 küp için ~2,2,2).
   Yoksa: eski exe (zaman damgasına bak).
2. **Collider IPC ile takip ediyor** — `scene.set_transform @{name='Default_Cube';
   scale=@(2,1,1)}` → hemen `collider.list`: `resolved_size` ≈ (4,2,2);
   ardından `translation=@(0.5,…)` → `resolved_world_min/max` x'te +0.5 kaydı,
   boyut aynı. Timeline OYNARKEN de aynı sonuç gelmeli.
   Eski boyut/konum dönüyorsa: memo kontrolü devreye girmiyor (matris
   karşılaştırması); `resolveObjectTransformForSimulation(…, false)` bulamıyor olabilir.
3. **Panel: ölçek korunuyor** — küpü seç, IPC ile ölçekle (2,1,1), sonra
   inspector'da Pos'u sürükle. Ölçek 2 KALMALI, Scl alanı 2 göstermeli.
   Aynısını gizmo ile ölçekleyip inspector'dan taşı. Bozuksa: sync çağrısı
   alanlardan önce çalışmıyor.
4. **Panel: döndürülmüş obje Rot sürüklemesi** — Rot.y'yi 0→120 sürükle;
   değer sıçramamalı/titrememeli. Titriyorsa: sync her kare tetikleniyor
   (karşılaştırma toleransı / konvansiyon uyumsuzluğu).
5. **Undo** — ölçek + taşıma sonrası Ctrl+Z: obje ve collider birlikte döner.
6. ★ **Sinsi olan** — kar MPM sim'i (reset + oynat) ölçeklenmiş/taşınmış küpe
   çarpmalı. Collider overlay doğru görünüp sim eski yere çarpıyorsa: sim kare
   önbelleği eski kareleri oynatıyor (bu parti önbelleği temizlemiyor —
   reset gerekir). Bu sessizce makul görünür; kar havada eski küpe oturur.
7. **Perf** — `rt.perf`'te `sim.collider.surface_cache_rebuild` statik sahnede
   kare başına 0 kalmalı (memo hâlâ tutuyor). Artıyorsa: handle matrisi her
   kare yeniden yazılıyor (bit düzeyinde değişen bir yazıcı var).

Bilinen sınır: inspector Pos/Rot/Scl düzenlemesi hâlâ undo kaydı ATMIYOR ve
`fromTRS`'in dönüş inşası `getPivotMatrix`'ten (Rz·Ry·Rx) farklı görünüyor —
doğrulanmadı; döndürülmüş objede ayrı iş, bu partide dokunulmadı.

## Kar 2. parti — sertleşme tavanı, buz tabanı, yoğunluk dağılımı (C++ derlendi, SHADER DERLENMEDİ)

Build: **C++ VE shader** (`compile_shaders.bat`; push constant boyutu değişmedi, iki pad alan oldu).
Not: `docs/dev/KAR_STOMAKHIN_MPM.md` §6.

Canlı ölçüm (1. parti build'i, 87 552 parçacık, 0.1 m voxel, ~3 m düşüş, 7 m/s çarpma):
serbest düşüşte J_P = 1.000 sabit (kayma yok), 9 substep / 92 ms. Çarpma karesinde
7646 parçacığın modülü dalga sınırına kırpıldı (substep bir kare geç kaldı), blok
eşit sıkıştı (ortalama J_P 0.74, min 0.395 = buzdan yoğun), hiçbir parçacık
gevşemedi (max J_P 0.88), sonra 184 substep / 430 ms'de KALICI kaldı.

Ek ölçüm (1. parti exe, 13:26): çarpmada adım 131 → 528/420 ms (dt 1/60);
GPU kernel toplamı substep başına ~2.3 ms, bunun ~1.7 ms'si `sim_matter_p2g`.
Substep başına maliyet makul; patlayan şey substep SAYISI (9 → 184). 24 fps
zaman çizelgesinde dt 2.5× → "kare başına saniyeler". Domain DEM yolundan
geçiyor: Sand → Surface madde bağı Sand'in `dem` taşımasıyla tane çözücüsünü
açıyor (0 tane varken) ve backend'i Vulkan'a kilitliyor.

Sıra: bağımsız ve hızlı görülen önce.

1. **Shader yüklendi.** Log'da stress_update push-constant hatası yok.
   *Bozuksa:* .spv eski; aşağıdakilerin hepsini maskeler.
2. **Maliyet sabit.** `Probe-SnowStomakhin.ps1` çıktısında `sub` serbest düşüşte
   VE çarpmadan sonra aynı (~9·√5 ≈ 20). Adım süresi ~200 ms civarı, 430 değil.
   *Bozuksa:* çarpmadan sonra artıyorsa boyutlandırma hâlâ ölçülen J_P'ye bakıyor
   (eski exe). Daha ucuz istersen Snow'da `granular_snow_max_hardening` = 1 →
   ~9 substep (sıkışınca sertleşme yok, gevşeme/kırılma yine çalışır).
3. **Kırpma sıfıra yakın.** `hcap` (hardening_capped) çarpma karesinde ~0.
   *Bozuksa:* yüzlerce = substep tavana göre boyutlanmıyor.
4. **J_P buz tabanı.** Probe "J_P floor respected" PASS (≥ 0.436).
5. **Başlangıç dağılımı.** Seed sonrası ilk satırda J_P ≈ 0.951 .. 1.051.
   *Bozuksa:* 1.000/1.000 = `assignSnowHeterogeneity` çağrılmıyor.
6. **Gevşeme (asıl hedef).** Probe "loosening" PASS: J_P max, başlangıç
   dağılımının üstüne çıkıyor. ★ **En sinsi sonuç:** blok yine eşit sıkışıp
   esniyor, J_P max başlangıçtaki 1.05'te kalıyor. Bu, düz levhanın düz zemine
   çarpmasının geometrisi olabilir (yanal kuşatma, çekme yok) — 7. madde ayırır.
7. **Geometri ayrımı (kullanıcı/ajan):** aynı karı küre olarak ya da eğik bir
   collider'a düşür. Kürede/eğimde kırılıp levhada kırılmıyorsa model doğru,
   levha testi çekme üretmiyor. İkisinde de kırılmıyorsa granular hatta FLIP
   eksikliği (KAR notu §6) sıradaki iş.
8. **Görsel:** kenarlardan topak kopması, açık kalan çatlaklar.
9. **Backend kilidinin sebebi görünüyor.** Domain sekmesinde kilitli backend
   altında "DEM is on because Sand (substance binding) uses granular transport
   'dem'…" yazmalı; `fluid.matter_models` → `grain_ownership.origin` aynı şeyi
   döner. Sand bağını kaldırınca (ya da reset sonrası) CPU seçilebilmeli.
   *Bozuksa:* yalnız "Grains need the Vulkan backend." = eski exe.

---

## 1. parti (2026-10-23) — Stomakhin dalı + sayaç körlüğü + gerilme yoğunluğu

Canlı sonuç: sayaçlar ölçülüyor (measured true, counted = n), snow yasası seçili,
serbest düşüşte J_P sabit — 1–3. maddeler PASS sayılabilir. 4–7 yukarıdaki partiye
devredildi.

## Önceki parti (2026-10-22, 3. parti) — henüz doğrulanmadı

Önceki partinin açık deneyi (RT gaz phase-cache var/yok iki build) bu partiyle
çakışmıyor; ayrıntısı `RT_GAS_OPTICAL_PERFORMANCE.md` sonunda.

Build: C++ yeterli (shader değişmedi). Değişen: `MaterialNodesV2.h` (derleyicide
`inPin`: değeri pinde yaşayan 25 giriş artık pin değerini derler),
`ParticleRenderBridge.cpp` (yakın mesh havuzu kapasite kuralı
`nearPoolCapacityFor`). Gerçek shard + mineral ölçümü bu build'den önce canlı
koşuldu (KUM_TANE…md §8b).

Sıra: bağımsız ve hızlı görülen önce.

1. **Mix Color rengi render'a ulaşıyor.** Yeni principled materyal → node
   grafiği: Mix Color (Fac 0, Color1 kırmızı, bağlantısız) → Base Color;
   bir küpe ata, Rendered (Vulkan RT). Küp **kırmızı** olmalı.
   Ajan yolu: `rt_grain_mineral_material.py` yerine aynı grafiği
   `nodes.set_param` ile kurup `render.probe`/ekran görüntüsü.
   *Bozuksa:* gri (0.5) = derleyici hâlâ literal basıyor (eski exe — zaman damgası).
2. **Eski sahneler.** Mix Color / Hue-Sat / Combine Color kullanan kayıtlı bir
   materyali aç. ★ **En sinsi sonuç:** render'ı DEĞİŞEBİLİR — bu düzeltmenin
   kendisi: artık panelde görünen pin değeri çiziliyor, eskiden 0.5/1.0 literal.
   Değişen sahne bug değil; değişmeyen ama panelde farklı renk gösteren sahne bug.
   *Bozuksa:* bağlı girişli bir node'un rengi değiştiyse `inPin` bağlantıyı
   atlıyor (olmamalı — ilk satırı `inOr` ile aynı).
3. **Küre yolu değişmedi.** Near-mesh kapalı bir kum sahnesinde
   `render.instance_stats` ve ms/örnek önceki build'le aynı.
   *Bozuksa:* değişiklik ana havuza sızmış (yalnız yakın havuz kuralı değişti).
4. **Boşluk azaldı.** Ajan fixture'ı kurar ve tam K0'ı koşar:
   `python scripts	est
t_k0_grain_mesh_near_ipc.py --domain K0_Sand --apply --distance 1.5 --library fracture`.
   Beklenen hepsi-mesh vakası (4591): `tlas_instances` ≈ 6.1 bin (önce 12321);
   1.5 m (780): ≈ 1.0–1.6 bin (önce 3105). Script artık kullanıcının near
   `sources` listesini de geri yükler (`near_restored` true).
   *Bozuksa:* hâlâ ~12 bin = eski exe ya da kapasite yolu eski kuralı kullanıyor.
5. **Kamera hareketi.** Fixture'da kamerayı birkaç kez az oynat (aynı mesafe
   bandında): her harekette yapısal yeniden kurma olmamalı (`perf.list`
   `accel.vulkan_rt.rebuild` sayısı hareket başına artmamalı, yalnız kapasite
   aşılınca artar).
   ★ ms/örnek boşluk azaldıktan sonra hiç değişmezse "boşluk bedava" diye
   okunur — opak shard ölçümü (§8b) zaten bunu söylüyor; ama yalnızca
   `tlas_instances` gerçekten düştüyse geçerli. Önce slot sayısına bak.

6. **Grafik silme + Color Ramp durakları + pin değeri (tek script).**
   `python scripts	est
t_material_node_parity_ipc.py` → `PASS`
   (pin_value, remove, ramp_stops). Kendi küpünü 60 m yukarıda kurar, `render.probe`
   ile ölçer, kamerayı geri yükler.
   ★ **ESKİ exe'de bu script uygulamayı ÇÖKERTTİ** (2026-10-22): `nodes.set_property`
   iç içe dizi değeri (`[[...],...]`) alınca pipe koptu, süreç kapandı. Eski
   `nodeParamFromJson` 3 elemanlı diziyi Vector3 sanıp `get<float>()`'u bir dizi
   üzerinde çağırıyordu; atılan `json::type_error` yakalanmıyor gibi. Yeni build'de
   bu yol `Rows` olarak ayrıştırılıyor. Yine de kontrol et: IPC dispatch'i
   `std::exception` (yalnız `runtime_error` değil) yakalıyor mu? Yakalamıyorsa
   herhangi bir metoda bozuk tipli değer göndermek uygulamayı düşürür.

Doğrulama (ajan): değişiklikler derlenmedi. Derleyici hatası canlıda bulundu
(kırmızı Mix Color gri çizildi), düzeltme sonrası koşulmadı.

---

## Ayrı parti (aynı build): fracture shard'ları flat SoA

Kaynakta, derlenmedi. Değişen: `scene_ui_menu.hpp` (shard üretimi + silme),
`scene_data.h` (`nodeWorldCenter`, `getRigidBodySourcePivot`, kayıt öncesi
rest snapshot'ı, yeni `forgetBodyMeshCaches`), `ThermalFractureBridge.cpp`
(`nodeMeshVolume` artık `indices` üzerinden). Yeni .cpp yok, IPC yüzeyi değişmedi.
Her shard artık **tek** `TriangleMesh` (ardışık üçlü, `indices[i]==i`); facade üretilmiyor.

Sıra: bağımsız ve hızlı görülen önce.

1. **Shard tek flat mesh.** `scene.add_primitive cube` → `physics.fracture_object`
   (site_count 24) → bir `shard_objects` adına `scene.object_info`.
   Görmen gereken: `meshes: 1`, `triangles` > 0, `has_bounds: true`.
   *Bozuksa:* `meshes: 0` = hâlâ facade üretiliyor (eski exe — zaman damgasına bak).
2. **Görünüm/malzeme aynı.** Viewport'ta preview gap'li shard'lar; exact modda dış
   yüz kaynak dokusunu taşır, kesik yüzler `<obje>_Interior` malzemesinde.
   *Bozuksa:* hepsi tek renk/doku (0,0) = materialID/uv özniteliği yazılmamış.
3. **Seçim/picking.** Bir shard'a viewport'ta tıkla → hiyerarşide o shard seçilir;
   Rigid Body listesinde tıklamak da viewport'ta seçer.
   *Bozuksa:* `rebuildMeshCache` flat shard'ı `direct_mesh_nodes`'a almıyor.
4. **Re-fracture + unfracture.** Aynı objeyi farklı seed'le ikinci kez kır, sonra
   `physics.unfracture_object`. Beklenen: ikinci kesimde eski shard'lardan **hiçbiri**
   kalmaz (obje sayısı = yeni shard sayısı + diğerleri); unfracture orijinali geri
   koyar, `__shard_` adlı obje kalmaz.
   *Bozuksa:* çift shard = `fractureEraseNodes_` flat'i silmiyor.
5. **Make Breakable → kır → simüle et.** `physics.make_fracture_group` (veya panel),
   `physics.break_fracture_group`, birkaç kare oynat. Shard'lar ayrılıp düşmeli,
   grup kütlesi hacme göre (24 shard'lık 2 m küp ≈ 8 m³ × yoğunluk), **24 kg değil**.
   Patlama merkezden dışa doğru olmalı.
   *Bozuksa:* hepsi aynı yöne/orijinden uçuyorsa `nodeWorldCenter` flat'i okumuyor;
   kütle = shard sayısı kg ise `nodeMeshVolume` 0 dönüyor.
6. **Re-fracture SONRASI simülasyon.** 5'i yap, sonra objeyi re-fracture et ve tekrar
   kır/oynat. Çökme veya eski kesimin pozunda donmuş shard olmamalı.
   *Bozuksa:* rigid bake cache'i ölü `TriangleMesh*` tutuyor (`forgetBodyMeshCaches`
   çağrılmamış) — bu UB, bazen "çalışır" görünür.
7. **Kayıt/açma.** `scripts/material_phase_tests/phase_10_fracture_roundtrip.py`
   (script değişmedi, Release kopyası aynı). PASS + yeniden açılışta shard adları ve
   cluster'lar aynı.
8. **★ Simülasyon ortasında kayıt (en sinsi).** 5'teki sahnede shard'lar havadayken
   (kare ~30) `project.save`, sonra `project.open`. Açılışta shard'lar **dinlenik
   (kesilmiş) pozda** olmalı, havadaki pozda değil; kayıttan sonra da canlı sahne
   havadaki pozda kalmalı.
   *Bozuksa:* açılışta shard'lar dağınık duruyorsa kayıt öncesi rest snapshot'ı flat
   mesh'i atlıyor — hata vermez, "kaydettiğim yer" gibi makul görünür. Bu
   add_primitive ile eklenen her flat rigid body için de geçerliydi (eski hata).
9. **Termal kırılma.** Yanmış bir objede `pattern=2` ile kır; panel mesajında
   "no recorded burn damage" yalnız gerçekten hasar yoksa çıkmalı; kırılma bölgesi
   yanık bölgeye yoğunlaşmalı.
10. **K0 tane örneği olarak shard.** (Diğer oturumun scripti; o oturum facade reddini
    kaldırınca) `rt_k0_grain_mesh_near_ipc.py --library fracture`: `exemplars` = shard
    sayısı, `exemplar_triangles` > 0, `status: active`.
    *Bozuksa:* `missing source` = `gatherNormalizedSceneFaces` shard'ı bulamıyor.


## 2026-10-22: prosedürel tane kütüphanesi ve DEM/MPM/fluid görünümü

Derlemeyi kullanıcı yapar; ajan build veya uygulama başlatmadı. Yeni
`UI/GrainMeshLibraryUI.cpp` proje/filters listelerine kaydedildi.

1. Üstteki materyal/K0 maddeleri 1–6'yı yeni executable ile tamamla.
2. Domain'i Splat/Particles görünümünde, geometry=icosphere olarak hazırla;
   birkaç carrier doğduktan sonra timeline'ı duraklat, kamerayı Near Distance
   içine getir. Ayrı terminalde:
   `python scripts/test/rt_grain_library_options_ipc.py --domain K0_Sand --render`
   Beklenen PASS: rock ve voronoi için near_instances>0; yanlış string/sayı/
   iç içe dizi sonrası IPC hâlâ cevap verir. Probe ayarları geri yükler.
3. 2'yi küçük **DEM, MPM ve fluid** sahnelerinde ayrı çalıştır. Virtual Grains
   8 iken mesh açık → primary_visual_children=1; kapalı → eski ayar geri
   gelmeli (MPM Physical Carriers açıksa primary=1 kalması doğrudur).
   Whitewater sayısı etkilenmemeli. Rendered RT ve Solid/Matcap görünümünde
   aynı carrier hem küre hem mesh olarak çift çizilmemeli.
4. Near Grain Meshes → Grain Shape: Procedural Rock / Rounded ve Procedural
   Voronoi / Angular kaynak obje seçmeden çalışmalı. Scene obje sayısı sabit
   kalmalı. Near Distance/bütçe dışında kalan carrier'lar küre kalmalı.
5. Scene objects seç; iki farklı materyalli kaynak (mümkünse bir kaynakta iki
   yüz materyali) ekle. Use Exemplar Face Materials açıkken yüz materyalleri
   ve UV'ler korunmalı; kapalıyken domain/substance/ıslak bantlar kullanılmalı.
   Kaynakta materyal atamasını değiştir: paused sahnede yakın mesh güncellenmeli.
   Uzak küreler bu seçenekten etkilenmez; yakın/uzak renk farkı bu modda beklenir.
6. Kaydet/aç: shape ve source_materials aynı kalmalı. Eski projede scene/false.
   K0 A/B ölç: near_restored=true; benchmark kendi shape/materyal seçimlerini
   geri yüklemeli. K0 fixture önce yeniden kurulmalı, önceki sahne kaydedilmemişti.
7. `scripts/test/grain_mesh_library_test.cpp` kaynak testi kullanıcı test
   derlemesinde çalıştırılmalı; ajan proje kuralı gereği derlemedi.
8. Fiziksel boyut dağılımı sonraki iştir. Önceki emitter durması kullanıcı
   tarafından sahne kurulumuna bağlandı; çözücü bug'ı olarak açık tutulmuyor.

Ajanın build gerektirmeyen kontrolleri: Python AST, JSON/XML okuma,
IPC descriptor üretimi/--check, ilgili dosyalarda diff whitespace kontrolü.
Canlı render, kayıt round-trip ve C++ kabul testleri çalıştırılmış sayılmaz.


## 2026-10-22: Add Mesh / Grains–Crystals üreticisi

Ayrıntılı sözleşme: `docs/dev/GRAIN_CRYSTAL_GENERATOR.md`.

1. Kullanıcı derlemesinde yeni `Scene/GrainGenerator.cpp`, `UI/GrainGeneratorUI.cpp`
   ve `Api/Rt{Api,Ipc,Python}GrainGenerator.cpp` modülleri alınmalı (vcxproj/filters
   kaydedildi; CMake source glob'ları kapsıyor). Ajan derlemedi/uygulamayı başlatmadı.
2. Add Mesh → Procedural Generator → Grains / Crystals: Rock/Sand, Voronoi,
   Snow Crystal ve Mineral Crystal. Seed, adet, biçim, çap aralığı ve ağırlıklı
   materyaller kullanıcı kontrolünde olmalı. Yeni kütüphane gerçek flat objeler olmalı.
3. Ayrı terminal: `python scripts/test/rt_grain_generator_ipc.py` → PASS.
   `--keep` geometri/materyal/kayıt incelemesi için örnekleri bırakır.
4. Mesh Grains → Create Grain / Crystal Library… aynı pencereyi doğru hedefle
   açmalı; Generate seçili domain'e kaynakları ve kendi materyallerini bağlamalı.
   Tek Undo/Redo objeleri ve önceki domain kaynak ayarlarını birlikte geri getirmeli.
5. Karı alt/üst/yandan incele: kapalı, kalınlıklı dallı plaka; uç/kenar delikleri
   ve üst üste iç dal yüzleri olmamalı. Salt: 4 kenar/0 uç; Quartz/Ice: 6 kenarlı
   prizma; Sugar: seçilebilir uzama/uç. Aspect ve seed sonuçta görülmeli.
6. Üretilen örnekleri içeren proje kopyasını kaydet/aç. İsimler, geometri,
   yerleşim, UV ve materyaller korunmalı. Küçük çaplı kar örnekleri de Mesh Grains
   kaynağı olmalı (SoA yüz okumasında mutlak minik-alan elemesi kaldırıldı).
7. Eski near_mesh_shape=rock/voronoi sahneleri açılmalı; yeni UI'de ayrı sabit
   üretici dropdown'ı görünmemeli. Scene object listesi tasarımı ertelendi.
8. Kullanıcı kaynak testlerinde `grain_generator_test.cpp` + mevcut
   `grain_mesh_library_test.cpp` çalıştırılmalı. Fiziksel boyut dağılımı hâlâ ayrı iş.

## Voronoi edge rounding acceptance (2026-10-22)

- User build: GrainGenerator.cpp and GrainGeneratorUI.cpp integration.
- Generate Voronoi with Examples=1: an irregular cut cell, not an uncut cube.
- Change seed: different geometry; repeat seed: deterministic geometry.
- Compare Roundness=0/0.6 with Edge Detail=6 under lighting and wireframe.
- Compare Edge Detail=2/8: real triangle density must change when rounded.
- Aspect=(0.7,1.6,1): verify smooth normals and closed surface.
- Repeat via Python and scene.generate_grains IPC with identical parameters.
- User compile/run scripts/test/grain_generator_test.cpp: new single-cell,
  seed, closed-surface, smooth-normal and detail regressions.
- Recheck Snow Crystal, materials, Undo/Redo and save/load.

Codex did not build or launch the application; visual acceptance remains pending.

## DEM physical size distribution (2026-10-22, user build/live checks pending)

- Build C++ and DEM / grain-fluid / grain-MPM shaders together (DEM revision 28).
- Build/run scripts/test/matter_grain_size_test.cpp and existing birth/stages tests.
- In a paused empty scene, external terminal: python scripts/test/rt_grain_size_ipc.py.
- UI: min radius .01 m, max .02 m, Sand + large stones, large fraction .10.
  Verify Sphere and Near Mesh agree, larger stone resting heights, impacts,
  sleep/wake, source spacing, reset replay, project save/load and cache scrub.
- Recheck uniform/log-uniform, max=0 legacy fixed size, invalid patches unchanged,
  and all five size fields locked while particles are live.
- Recheck wet/liquid/gas/thermal/MPM momentum/heat regressions for mixed sizes.
- Dense 1:2 and 1:4 piles: inspect explicit bucket/list/contact capacity holds;
  existing 16 bucket / 32 list / 24 history limits remain, no silent truncation.
- See docs/dev/DEM_SIZE_DISTRIBUTION.md for settings and physical semantics.

No project/shader compilation or application/live IPC verification performed by Codex.

## DEM size controls and visible refusal feedback follow-up

- Pending user build: size edits are staged together, applied atomically with
  Apply size settings; invalid ranges are shown beside controls; reset required
  blocks applying sizes while preserving the draft. Discard restores saved sizes.
- Large stone share now uses percent with small/large count labels and estimated
  volume/mass dominance; optional 95% sand / 5% stones preset. Physics RNG unchanged.
- Rejected settings and held solver steps display prominent orange/red wrapped text.
- Verify two radius edits do not apply individually; Apply/Discard; external IPC
  settings refresh the draft; pending size edits do not alter other solver changes.
- Live IPC diagnosis in Physics Domain 1: damping bound requested 529 substeps,
  cap 512. Raised cap to 1024; steps dt=1/60 and dt=1/24 succeeded, held=false.
  Same 1143 grains retained; center Y .828615 -> .808913 -> .747721 m.
  Timeline remained paused at frame 22; sizes and source unchanged, no reset/save.
- This verifies the current executable; pending UI source changes need a user build.

## DEM mixed-size contact pile / runtime neighbour lists (revision 29)

Live reproduction at paused frame 17, Physics Domain 1: minimum radius .026 m,
maximum .083 m, bimodal large fraction .07, seed 1360, max_substeps 1024.
One external dt=1/24 step held with "grain neighbour list full (>32 ...)" at
4289 grains. This is a separate capacity failure from the previous CFL limit.
No reset, size/source edits or save were performed during this reproduction.

Pending user build:
- Rebuild C++ and all sim_matter_grain shader variants together (revision 29).
- Fixed-size domains retain 32-list baseline; mixed sizes start with 128 slots.
- Standalone DEM retries list-only overflow with a doubled budgeted buffer from
  unchanged host state; shared MPM/liquid clocks refuse replay to avoid double steps.
- Allocation includes old+new peak memory and checks shader index width.
- IPC diagnostics: runtime.neighbour_list_capacity / neighbour_list_retries.
- With the failing pile open and paused, external terminal:
  python scripts/test/rt_grain_neighbour_capacity_ipc.py --steps 8
- Check actual floor contact, dense mixed pile, fixed-size baseline, storage/working
  budget failures, and shared-clock mixed cases. Actual contact history remains 24;
  this change removes the 32-candidate ceiling, not all other solver budgets.

Codex performed source checks only for revision 29; no compilation or live run of
new code. The open executable still has revision 28 and cannot use this fix yet.

## DEM contact history overflow at floor impact (revision 30)

Live read-only diagnosis: paused frame 29, Physics Domain 1, 94,915 grains,
minimum radius .011 m, maximum .036 m, large count 995 (1% authored), seed 1360,
max_substeps 2048. Status: contact count/history-in-flight exceeds legacy 24.
No reset, setting mutation, extra tick or save performed during this diagnosis.

Pending user build and acceptance:
- Rebuild C++ plus all sim_matter_grain shader variants together, revision 30.
- Mixed-size history starts at 64 records, fixed-size remains 24. Four 32-bit
  mask words support records across boundaries 31/32, 63/64 and 95/96 up to 127.
- Block metadata is now owner + four masks + rest + compensation (7 words),
  following the capacity-sized identity->block map. Sleep/audit offsets changed.
- Standalone DEM history overflow retries from unchanged canonical host state
  with capacity doubled to at most 128, charged to actual/peak memory budget.
- Shared MPM/liquid clocks cannot replay and continue to refuse overflow.
- Adaptive CFL retries use observed maximum contacts + 4, bounded by allocated
  history; shared clocks use the full allocated history bound.
- runtime.contact_history_capacity / contact_history_retries expose allocation.
- External terminal with the failed pile paused:
  python scripts/test/rt_grain_neighbour_capacity_ipc.py --steps 8
- Verify floor impact and sustained dense packing beyond 24 contacts, sleep/wake,
  changing neighbour identities, population reorder, GPU-resident fixed-size baseline,
  friction/rolling persistence, conservative memory refusal, and mixed-owner safety.
- This raises mixed-size memory use (64 records for every grain initially);
  it is not sparse per-grain contact storage and does not establish million-grain
  cinematic performance. At 128 records a million-grain history alone is ~3.34 GiB,
  potentially above a device's single-buffer limit. The checked budget must refuse it.

Revision-30 work received source checks only. No build/shader compilation or live
physics acceptance of the new code was performed by Codex.

## Size-aware contact CFL bound (host-only follow-up, shader ABI stays revision 30)

Live diagnosis at frame 26: .011/.036 m bimodal 1%, 86,462 grains, cap 2048,
held at damping bound 2081. Existing bound combined globally smallest mass with
largest radius stiffness. This pair does not describe a physical grain.

Source change: for each inertial mass mi and radius ri, bound incident stiffness
by alpha*max(ri,2*ri*Rmax/(ri+Rmax)); reduce maximum ki/mi across particles.
Then omega^2 <= 2*contact_budget*max(ki/mi), retaining conservative damping and
accuracy algebra. Wet bridge bound remains conservative. Mixed-size rolling
uses the conservative unequal-radius rotational factor. Equal dry sizes retain
the old normal bound. No stiffness, restitution, size or substep cap is modified.

Validation:
- Analytical wall/pair inequalities checked for .011/.016/.026/.036 m; new
  normal-rate bound is .468085 of the old fictitious min-mass/max-k bound.
- Added C++ regression for unequal pairs, walls and equal-size parity in
  scripts/test/matter_grain_size_test.cpp (not compiled/run by Codex).
- Live revision-30 executable, unchanged authored settings: 12 external ticks
  at dt=1/48 succeeded, 36..42 peak contacts, history 64, neighbour list 128,
  no hold. This validates shorter dt under old bound, NOT the new host code.
- User rebuild C++, no shader ABI change. Repeat original dt=1/24 dense impact
  and sustained pile, inspect cfl_stiffness_mass_rate_s2 and substeps.
- Repeat fixed-size convergence, high rolling friction and wet/shared-clock
  cases. Lower substep count is accepted only with stable/converged physics.
- No reset, material/size edits, substep cap changes or scene save by Codex;
  12 manual half ticks advanced the live simulation, timeline left paused.

## Explicit reset clears stale DEM CFL hint

Live check confirmed contact_resolution=8 was applied; a NEW dt=1/24 step still
held on damping (2129 > 2048), not accuracy. Lowering accuracy cannot remove a
damping limit. The new size-aware bound remains pending a user C++ build.

Source audit additionally found resetGridDomainStates cleared host particles but
only invalidated generic field copies; retained matter_runtime->grain kept the
previous dense pile's cfl_contact_hint, publication identities and sleep context.
Added invalidateMatterGrainAfterReset in the focused DEM module and minimal calls
in both explicit reset buffer loops. Reusable allocations/growth requests remain;
CFL hint resets to baseline, history/residency/sleep/motion publication is cold.
Do NOT call this on per-step invalidateDeviceCopies (would erase valid history).

User build and regression:
- C++ rebuild only, shader ABI stays revision 30.
- After a dense pile/error, reset to sparse births and verify first-step
  cfl_contacts_per_grain starts at baseline instead of the old peak.
- Test fixed/mixed sizes, empty reset, repeated reset, and births restarting IDs.
- C++ reset-state regression added to matter_grain_size_test.cpp, not executed.
- Source checks passed. No reset or authored setting changes made in the live
  scene by Codex in this follow-up; one failed probe tick was attempted.

## Live close-out 2026-10-22: DEM continuation and MPM/DEM fixture

User-built executable: existing 100,000-grain DEM scene completed six dt=1/24
steps without holds, 1162..1260 substeps, max contacts 33, mean step_total 1358 ms.
This closes the reported short continuation failure, not long-run/scale acceptance.

Created MPM Sand + DEM Stones in the same scene. Previous domain/source retained
but disabled; editing setup reset previous live particles. New fixture completed
86 dt=1/120 ticks with 12,288 MPM carriers and 10 DEM stones, nonzero bidirectional
contact events and sampled momentum residual ~1e-8 N*s. It remains paused at frame 0.
Actual penetration/support/convergence acceptance remains next work.

Details: docs/dev/DEM_MPM_ACCEPTANCE_20261022.md
Raw measurements and previous configuration: docs/dev/dem_mpm_acceptance_20261022.json
No compilation or application launch by Codex; live tests used external named pipe.

## Seed material UX follow-up

- User C++ build: existing MatterModelControls.cpp now provides the seed panel's
  shared domain-default substance picker and granular MPM/DEM/fluid model label.
- Minimal integration call in scene_ui_simulation_domains.cpp; no feature body
  added to the over-2000-line panel file. No shader changes.
- Verify Seed/Matter pickers remain synchronized; liquid/MPM/DEM labels match
  substance.get; errors are visible; reset+seed and save/load preserve material.
- UI, Python and IPC all use existing updateFluidDomain; no UI-only material state.
- Current live bed is granular MPM, NOT fluid (0 fluid / 12288 MPM / 10 DEM).
  Fluid-like appearance is not dismissed; pile/slope/penetration acceptance remains.
- Source checks passed, no compilation and no live acceptance of the new UI yet.


### DEM seed ownership and physical packing (2026-10-22)

- First Sand seed regression: resolve substance ownership and material before
  birth, after replacement clears old particles. Previously a first seed could
  become MPM and lock that owner until reset.
- UI, Python `rt.fluid.seed`, IPC `fluid.seed` share this core. DEM seeds use
  sphere-volume mass and a maximum-radius lattice independent of voxel/PPC.
  Variable radii keep deterministic IDs; smaller grains begin with gaps and settle.
  This is conservative initial packing, not dense equilibrium packing.
- DEM requested count is a lower bound when truncated (emitted + 1). Candidate
  work is bounded; capacity/scan cutoff has a separate DEM warning.
- User: C++ build only, no shader changes. Codex did not compile or run C++ tests.
- Run updated `scripts/test/matter_grain_birth_test.cpp`: physical spacing/mass,
  granular model, voxel independence, append collision rejection, budget, small box.
- Live: empty domain, builtin Sand, seed replace=true BEFORE any step.
  `fluid.matter_models`: DEM > 0, MPM = 0, reset_pending=false. Repeat after
  reset/reseed. Replace existing incorrectly born MPM particles.
- Identical box/radius with different voxel/PPC must give identical DEM count.
  Custom granular_transport=mpm and Water retain voxel/PPC sampling and owners.
  MPM default + DEM source must seed MPM, even with domain grains enabled.
- Drop Sand above floor to test impact scattering. A zero-velocity resting seed
  is not a bounce test. Restitution and friction were not changed.


### DEM seed column symmetry regression (2026-10-22)

Live read-only evidence after the ownership fix: Physics Domain 1 has 34,512
DEM grains, zero MPM/fluid owners, reset_pending=false, dry grains, zero liquid
bridges, restitution=0.5. Horizontal bounds remain the seed footprint; horizontal
momentum and spin energy are zero, maximum contacts per grain=2. This matches
perfectly aligned columns compressing/rebounding without lateral excitation,
not an adhesive material or MPM ownership fallback.

- Seed now uses deterministic bounded position jitter (10% of maximum radius
  per coordinate) with spacing 2.2002 R. The worst adjacent jitter extremes
  retain at least 2.0002 R separation along their separating axis. Bounds include
  the jitter margin. No velocity kicks or changes to material forces.
- Existing size_seed controls reproducibility through the same UI/script/IPC
  core. Seed positions remain independent of voxel/PPC and particle budget.
- Initial count is lower than the old perfectly touching lattice; this remains
  conservative packing, not a dense packing generator. The box needs room for
  the maximum diameter plus its jitter margin.
- Updated C++ regression assertions cover pair separation, containment, layer
  misalignment, seed repeatability/change, append rejection, budget-prefix stability.
  Codex did not compile/run C++ tests or launch the application.
- User build: ordinary C++, no shader change. Clear/replace and seed again: old
  live positions cannot acquire the new birth jitter. Run with settle-on-seed
  disabled for an observable drop. Verify horizontal spreading and nonzero spin
  after impact, no coherent column bounce, no held step or capacity overflow.
  Repeat with settle enabled and with radius variation; check a final stable pile.
