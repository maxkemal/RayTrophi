# Fotogerçekçi gökyüzü ve bulut sistemi — tasarım (Atmosfer Faz 3)

> **Durum:** AKTİF — 2026-09-30 açıldı, §9 kararları kullanıcı tarafından ONAYLANDI (hepsi öneri yönünde). `ATMOSPHERE_SYSTEM.md` §7 Faz 3'ün ayrıntılı tasarımı.

Hedef: film/oyun üretim kalitesinde, **fiziksel birimlerle tanımlı**, iki Vulkan
backend'de aynı veriden beslenen bulutlu gökyüzü. Vulkan RT **referans**
(yansız, zamanla yakınsayan), RayFusion **ona ölçülerek kalibre edilmiş
gerçek zamanlı yaklaşım**. Froxel işinde kurduğumuz parite disiplini aynen
geçerli: aynı compute shader her cihazda, tek parametre paketi, sayıyla parite.

---

## ▶ DEVİR NOTU — başka bir ajan buradan devam edebilir (2026-09-30)

**Faz 3e (2026-10-01, yazıldı, derlenmedi):** dikey yapı — weather A = hücre gelişme
yüksekliği (`topFrac` çarpanı + taban ±%4), `cloudShear` (rüzgâr kesmesi, tüm katman
okumaları + majorant), konvektif gürültü dikey esnetme, ışık march'ı mutlak 40 m'den +
uzak örnek, powder, bulut içi adım ×0.6. Kabul: `Probe-CloudVertical.ps1`. Oktav
kalibrasyonu hâlâ açık (ışık değiştiği için şimdi anlamlı).

**Durum:** 3a (otorite + GPU alanı + IPC) ve 3b (RT) derlendi, kullanıcı
canlı gördü: "çok hızlandı". Son değişiklik (ambient örtme) YALNIZ shader.
⚠ Kullanıcı son turda yalnız shader derledi: C++ derlenmeden `weather[3]`
eski anlamını (katman sayısı) taşır → `cloudRtSteps()` 16'ya kelepçelenir =
kaba adım + düz görünüm. İlk iş: **C++ de derlensin**, sonra göz.

**Kod haritası (tek tanımlar):**
- Otorite: `include/Atmosphere/AtmosphereClouds.h/.cpp` (CloudState, JSON, preset,
  doğrulama). World: `World::setClouds/cloudWindOffset/syncCloudPacket`.
- GPU paketi: `include/Backend/CloudParams.h` (208 B; `weather.w`=rt_steps,
  `misc.z`=rt_max_bounces, `misc.w` bayrak: +1 render, +2 ikincil tam, +4 referans).
- Yoğunluk + faz: `shaders/cloud_common.glsl` (`cloudLayerDensity`, `cloudPhase`,
  `cloudPhaseFit`). Doku üretimi: `cloud_noise.comp` (mode 0–4; 4 = majorant).
- RT: `shaders/cloud_rt.glsl` — `cloudMarch` (varsayılan), `cloudRender`
  (referans yol izleyici), `cloudSunTransmittance` (closesthit yüzey gölgesi),
  `cloudIntervals` (küresel kabuk, float-güvenli). miss.rmiss çağırır.
- Cihaz: `src/Backend/VulkanDeviceClouds.cpp` (dokular, RT binding 30–35,
  `updateCloudRtParams`). Adapter: `VulkanBackendAdapter::setCloudState`
  (render bayrağı, accumulation reset). IPC: `src/Api/RtApiClouds.cpp`
  (`world.get/set_clouds`, `apply_cloud_preset`, `cloud_stats`, `sample_clouds`).
- Kabul: `scripts/ipc/Probe-Clouds.ps1` (3a, sayısal).

**3c YAZILDI (derlenmedi, 2026-09-30):** `shaders/cloud_raster.comp` (aynı
`cloudMarch`, düşük çöz. + zamansal), `VulkanDeviceClouds.cpp` raster bölümü,
`VulkanBackendAdapter::recordCloudRasterPass` (render pass'ten önce, iki çağrı
yeri: VulkanBackend.cpp + VulkanViewportBackend.cpp), sky pass binding 25 +
`push.materialMeta[0]` bayrağı. CANLI. 3c-2 YAZILDI: titreme düzeltmesi (varyans kırpma) + RayFusion
bulut gölgesi (`cloud_shadow.comp`: y=0 düzleminde 512²/32 km, kameraya kilitli;
`material_preview_frag.frag previewCloudShadow`, preview binding 26/27, sceneFlags 64).
Beer haritası yerine doğrudan geçirgenlik (alıcılar bulutun altında → yeterli).
Parite IPC'si (march modu) hâlâ yok. SIRADAKİ (kullanıcı isteği): **katmanlı
bulut şekilleri** — madde 2.

**Şekil 2. adım + cirrus YAZILDI (2026-09-30)** — NEXT_BUILD_CHECKS "Bulut şekli + cirrus".
**ERTELENDİ (kullanıcı: "en sona"):** RayFusion'da ana TAA kapalıyken kamera
ileri/geri hareketinde ve parametre değişiminde bulut titremesi. İleri/geri
harekette piksel hareketi küçük ama derinlik değişiyor → varyans kırpması 3×3'ü
grup içiyle sınırlı ve gürültülü; aday çözüm: kırpmayı ayrı bir geçişte tam 3×3
(ya da 5×5) komşulukla yapmak + parametre değişiminde geçmişi atmak yerine hızlı
yakınsama (alpha 0.5, birkaç kare).

**Sıradaki işler (öncelik sırası):**
1. **Görsel ayar (küçük):** kalın bulut hâlâ düz ise → (a) oktav sabitleri
   `CLOUD_MS_A/B/C` referansa (`rt_reference_path_trace`) karşı kalibre et —
   aynı kare, iki mod, ortalama parlaklık oranı ~1 olmalı; (b) Nubis "powder"
   yerine fiziksel yol: ışık taramasında ilk 2 örnek detaylı zaten, gerekirse
   6→8 örnek. Sanatsal düğme EKLEME (ilke §1).
2. **Şekil — İLK ADIM YAPILDI (2026-09-30, yalnız shader):** yerel örtü = bulut boyu (küme merkezi kule, kenarı alçak kubbe; stratiform tiplerde kapalı) + yükseklikle yoğunlaşan çekirdek. Kalan: `cloudLayerDensity` — tip başına dikey profil
   (stratus/cumulus/Cb) + tabanda sert, tepede kabarık erozyon; weather G
   kanalı tip varyasyonu zaten var. Bu, "fotoğraf gibi" algısının ana kaynağı.
3. **3c RayFusion:** aynı `cloudMarch`'ı compute'a taşı (¼ çözünürlük, 4×4 Bayer
   zamansal, ağırlıklı bulut derinliğiyle yeniden izdüşüm, bilateral büyütme),
   raster gökyüzü birleşiminde uygula. `cloudMarch` `worldData`/`atmosphereLUTs`
   isimlerine bağlı: compute'ta aynı isimlerle bağla ya da makroyla soyutla.
   Parite ölçümü: `world.sample_clouds` modu ekle (march inscatter/T) → RT ile
   aynı sorgu aynı değer.
4. **Beer gölge haritası** (§3.3): RayFusion güneşi + froxel (ışık huzmeleri, 3d).
   RT yüzey gölgesi şimdilik doğrudan optik derinlik (24 adım) — yeterli.
5. **Bulutlu panorama → ortam ışığı** (§3.4): RayFusion ambient + RT ikincil ışınlar
   (şu an ikincil = yarım adımlı march).
6. **3d:** froxel'e gölge haritası, preset cilası, `rt.perf` ile bütçe.
7. Cirrus (CloudState'te var, render'ı yok) — 3c/3d.

**Tuzaklar:** RT world struct'ı 5 shader'da kopya (miss/closesthit/hair/volume/
raygen) — alan eklenirse hepsi. Bulut **geometriden sonra** gelmez: dağın
arkasındaki bulut doğru, içindeki bulut yok (bilinçli). OptiX eski hacmi
paketten çizer, dokunma (dondurulmuş).

## ▶ Kaldığımız yer (2026-09-30)

**KARAR DEĞİŞTİ (2026-09-30, kullanıcı onayı):** §9-1 tersine döndü — RT varsayılanı
yol izleme değil, RayFusion ile ORTAK olacak **marcher** (`cloudMarch`; Nubis +
Hillaire: uyarlamalı adım, 6 örnekli güneş, 3 Wrenninge oktavı, enerji korunumlu
integrasyon). Gerekçe ölçülen: sahnedeki katman (sönüm 0.07/m, 2 km) yol izlemede
örnek başına on binlerce doku okuması + bütçe kesilmesiyle eksik ufuk. Yol
izleyici `quality.rt_reference_path_trace` ile referans olarak kaldı; oktav
sabitleri ona karşı kalibre edilecek. 3c: aynı `cloudMarch` ¼ çözünürlük + zamansal.

**3b YAZILDI** (derlendi ilk hali; marcher güncellemesi derlenmedi). Kontrol listesi:
NEXT_BUILD_CHECKS "Faz 3b". Uygulama kararları:
- Bulut **miss shader'ında** çözülür (`cloud_rt.glsl`): yalnız sahneden kaçan
  ışınlar bulut görür. Bulutun içindeki/arkasındaki geometri bilinçli eksik.
- Boş gök atlaması: weather map'ten türetilen 256² majorant (`cloud_noise.comp`
  mode 4, döşeme başına 3×3 komşuluk maksimumu). Segment ≤ 1 döşeme → tek
  fetch muhafazakâr sınır. Katmanlar çakışmadığı için sınır katman başına.
- Kamera/ikincil ayrımı: `PL_CAMERA_SEGMENT` payload biti (raygen bounce 0).
  İkincil ışınlarda 3 sekme sınırı = adı konmuş yanlılık (§9-2; panorama 3c).
- Beer shadow map **3b'de yok**: RT yüzey NEE'si doğrudan ratio tracking ile
  (referans kalite). Gölge haritası RayFusion (3c) ve froxel (3d) için gelecek.
- Eski prosedürel hacim Vulkan'dan çıktı, OptiX/CPU için yaşıyor (karar a).
- Bulut dokuları artık bulut kapalıyken de ayrılır (~25 MB, RT binding'leri
  her zaman geçerli olsun diye); **üretim** hâlâ tembel.

**3a (aynı derleme):** Kontrol listesi: NEXT_BUILD_CHECKS "Faz 3a";
kabul betiği `scripts/ipc/Probe-Clouds.ps1`. Uygulama kararları:
- Eski Vulkan bulut hacmi **3b'ye kadar yaşıyor** (paketten besleniyor); 3a'da
  sökmek iki derleme boyunca bulutsuz Vulkan demekti. 3b'nin İLK işi söküm.
- Bulut durumu backend'e `WorldData` ile gitmez (CUDA ile paylaşılan POD);
  `VulkanBackendAdapter::setCloudState` aynı dünya senkron hunisinden
  (`Main.cpp syncVulkanWorldToBackend`) çağrılır. IPC örneklemesi durumu önce
  kendisi iter (script aynı karede set+sample yapabilsin).
- Rüzgâr sürüklenmesi = iklim rüzgârı × zaman (sabit rüzgârda kesin; anahtarlı
  rüzgârda anlık hız — yine kareye bağlı, deterministik).
- Weather map örtü kanalı normal-CDF ile düzleştirilir (σ≈0,11) → "coverage"
  gökyüzü kesrine yaklaşık eşit. Görsel doğrulama 3b'de.

**3b'nin başlangıç noktası:** `cloud_common.glsl` (yoğunluk + faz) hazır; RT
descriptor setine 5 binding (4 doku + CloudParams buffer) eklenecek; raygen/miss
atmosfer geçişinde delta/ratio tracking. Önce eski hacim + `proceduralCloudDensity`
+ `VkVolumeInstance` bulut alanları sökülür.

## 0. Bugün (kod okuması, 2026-09-30)

| Konu | Durum | Dosya |
|---|---|---|
| Bulut | Tek prosedürel VDB hacmi, ±50–500 km dilim, **`world.objects` + TLAS'a giriyor**, genel hacim shader'ından geçiyor | `Render/VolumetricRenderer.cpp:176` `ensureInternalSkyCloudVolume` |
| Yoğunluk | Hash-gradient Perlin fBm, her örnekte ALU ile (doku yok), 8 adım varsayılan | `volume_closesthit.rchit:683-760` |
| İki katman | İkisi açıksa yoğunluğu büyük olan seçilir, diğeri **sessizce düşer** | `VolumetricRenderer.cpp:211` |
| Işık | Sanatsal düğmeler: `silver_intensity`, `shadow_strength`, `ambient_strength`, dual-lobe HG, sabit `multi_scatter` | `World.h:93-111` |
| RayFusion | **Bulut yok** | — |
| Zamansal | `raster_taa.comp` derinlikten yeniden izdüşüm yapar, hareket vektörü yok; gökyüzü (derinlik = uzak) için bulut derinliği bilinmez | `raster_taa.comp:57,131` |
| 3D doku | Yükleme yolu var (`VK_IMAGE_TYPE_3D`) | `VulkanBackend.cpp:14483` |
| Hazır olan | Transmittance / sky-view / multi-scatter LUT, aerial froxel (0–64 km), iklim anlık görüntüsü (rüzgâr, T, RH), güneş = directional kuralı | Faz 1a/1b/2 |

Sonuç: bulut için neredeyse hiçbir şey yeniden kullanılmıyor; **atmosfer
altyapısının tamamı** yeniden kullanılıyor. Eski bulut yolu sökülür (§8).

---

## 1. İlkeler

1. **Fiziksel birim, sanatsal düğme değil.** Sönüm katsayısı 1/m, damlacık
   çapı µm, taban/kalınlık m, rüzgâr m/s. "Silver intensity", "shadow
   strength" gibi düğmeler kalkar; o görünümler fizikten çıkar. Sanatçı
   kontrolü **preset + birkaç anlamlı parametre** ile verilir.
2. **RT = referans, RayFusion = ölçülmüş yaklaşım.** RayFusion'daki her
   yaklaşım sabiti (çoklu saçılma oktavları, ambient ölçeği) RT'ye karşı
   **sayıyla** ayarlanır ve sabitin yanına ölçümü yazılır.
3. **Tek veri, iki cihaz.** Noise dokuları, weather map, gölge haritası ve
   bulutlu gökyüzü panoraması her cihazda aynı compute shader'la üretilir.
4. **Bulut ışığı dünyaya da düşer.** Zemin gölgesi, ışık huzmeleri (froxel),
   kapalı havada kararan ortam ışığı. Bulutu yalnız "gökyüzüne boyanan"
   bir şey yapmak, fotoğrafta en çok göze batan hatadır: güneşli zemin +
   kapalı gök.
5. **Deterministik.** Aynı kare = aynı bulut; rüzgâr sürüklemesi timeline
   zamanının fonksiyonu, kare sayısının değil. Önbellekli render tekrar
   oynatılınca bulut kaymaz.
6. **Her yetenek IPC'de** (CLAUDE.md kural 1) ve ölçülebilir: yoğunluk ve
   geçirgenlik örnekleme uçları testin temelidir.

---

## 2. Fiziksel model

### 2.1 Ortam

- Bulut damlacıkları: tek saçılma albedosu ~0,9999 (görünür bantta
  soğurma ihmal edilebilir); **sönüm σt** bulut tipine göre ~0,02–0,15 1/m
  (stratus ince, cumulus orta, cumulonimbus çekirdeği yüksek). 1 km
  kalınlığındaki cumulus'un optik derinliği ~50 → pratikte opak. Bu sayı
  kabul testidir.
- **Faz fonksiyonu: Jendersie & d'Eon 2023** — damlacık çapından
  (5–50 µm) türetilen HG + Draine karışımı. Tek bir fiziksel parametre
  (`droplet_diameter_um`) hem ileri saçılma zirvesini (gümüş kenar) hem
  geri saçılma halesini (glory) verir. Eski `anisotropy / anisotropy_back /
  lobe_mix` üçlüsünün yerini alır. Aynı fonksiyon iki backend'de
  (`cloud_phase.glsl`).
- Bulut ile hava aynı ışın boyunca birlikte integre edilir: bulut
  örneğine gelen güneş, **bulut yüksekliğindeki** transmittance LUT'tan
  okunur (gün batımında bulutun alt yüzü kızarır, üstü beyaz kalır).

### 2.2 Geometri

- Katmanlar **küresel kabuk** (gezegen yarıçapı atmosferden). Ufukta
  bulut tabakasının incelmesi ve eğriliği buradan gelir; düz dilim
  ufukta yanlıştır.
- Katman tipleri:
  - **Alçak/orta hacimsel katman** (cumulus, stratocumulus, altostratus,
    cumulonimbus): 3D march. Tip, weather map'ten gelen sürekli bir
    değerdir (0 stratus → 0,5 cumulus → 1 cumulonimbus); dikey yoğunluk
    profili **tip × yükseklik oranı** 2D LUT'undan okunur (Schneider).
  - **Yüksek ince katman** (cirrus, cirrostratus): ~8–11 km'de ince kabuk.
    2D doku + analitik aydınlatma (tek saçılma + ince tabaka çoklu saçılma
    yaklaşımı). Buz kristali fazı ayrı (halo sonraya, §10).

### 2.3 Yoğunluk alanı

`density(p) = coverage_remap(base_shape(p), weather.coverage, profile(type, h))
             − erosion(detail(p + curl(p)), h) ` × `σt(type)`

- `base_shape`: 128³ Perlin-Worley (düşük frekans topaklar), tile'lanabilir.
- `detail`: 32³ Worley fBm (kenar erozyonu, alt kısımda kabarık, üstte
  tüylü — Schneider'ın yükseklikle erozyon ters çevirmesi).
- `curl`: 128² curl-noise, detail'i büker (türbülanslı kenarlar).
- Dokular **başlangıçta compute ile üretilir**, dosyadan okunmaz (sürüm
  bağımlılığı ve lisans yok); iki cihazda aynı shader, aynı seed.

### 2.4 Weather map

- Dünyaya sabitli 2D alan, varsayılan 2048² texel × ~150 km kapsama
  (~73 m/texel; tek cumulus 0,5–2 km → 7–27 texel). Kanallar:
  **R** örtü, **G** tip, **B** yağış (Faz 4), **A** yoğunluk çarpanı.
- Kaynaklar: (a) prosedürel üretici — compute, parametreli (örtü,
  tip dağılımı, "hücre" ölçeği, cephe çizgisi), (b) kullanıcı dokusu,
  (c) Faz 4+: zamanla değişen üreticiler (fırtına hücresi büyür/söner).
- **Rüzgâr:** Faz 2 iklim anlık görüntüsünden. Sürükleme ofseti =
  `∫ wind(t) dt` timeline zamanı üzerinden (anahtarlı rüzgâr için kapalı
  formda örneklenen integral) → deterministik. Katman yüksekliğinde
  rüzgâr: bugün tek zemin rüzgârı; kayma (shear) gelince katman başına
  farklı hız (§10).
- Bulutun **evrimi** (topakların şekil değiştirmesi): noise koordinatına
  yavaş bir zaman ekseni (4D benzeri kaydırma), rüzgârdan bağımsız.

---

## 3. Işık taşınımı

### 3.1 Vulkan RT — referans

Bulut, atmosfer geçişinde (raygen'in birincil ışını ve miss) **hacimsel yol
izleme** ile çözülür:

- **Delta tracking** ile serbest yol, **ratio tracking** ile gölge
  geçirgenliği. Majorant: weather map'ten türetilen kaba bir 3D
  maksimum-yoğunluk ızgarası (boş gökyüzü hızlı atlanır).
- Her saçılma olayında **NEE**: güneşe (transmittance LUT × bulut ratio
  tracking) ve gökyüzüne (sky-view LUT'tan önem örneklemeli yön).
- **Gerçek çoklu saçılma:** yol birkaç yüz sekmeye kadar sürebilir; Russian
  roulette. Gümüş kenar, koyu taban, kapalı havanın gri yumuşaklığı
  **buradan kendiliğinden** çıkar — yaklaşım yok.
- Progressive birikim zaten var; bulut gürültüsü örnek sayısıyla yakınsar.
- **İkincil ışınlar** (yüzeyden yansıyıp göğe giden): tam march pahalı →
  §3.4'teki bulutlu gökyüzü panoraması. Bu bilinçli bir yanlılık, adı
  konur ve ölçülür (§7). Karar: §9-2.

### 3.2 RayFusion — gerçek zamanlı yaklaşım

- **¼ çözünürlük march + zamansal birikim** (4×4 Bayer: her karede 16
  pikselin 1'i), 64–128 adım, mavi gürültü jitter.
- Çoklu saçılma: **Wrenninge oktav yaklaşımı** (σ, faz ve katkı her
  oktavda ölçeklenir), oktav sabitleri RT referansına karşı ayarlanır.
- Işığa doğru: 6 örneklik koni (ya da gölge haritasından optik derinlik —
  hangisinin RT'ye daha yakın olduğu ölçülür).
- Ortam ışığı: bulut yüksekliğinde üst yarıküre gökyüzü ışınımı (sky-view
  LUT'tan, froxel'deki gibi) + alt yarıküre zemin albedosu; yükseklikle
  karışım.
- Enerji-korunumlu adım integrasyonu (Hillaire 2016): adım uzunluğundan
  bağımsız sonuç — adım sayısını değiştirmek parlaklığı değiştirmemeli
  (kalite preset'i pozlamayı değiştirmez; bkz. 1/π dersi).
- **Zamansal yeniden izdüşüm:** TAA'nın derinliği gökyüzü için anlamsız, o
  yüzden bulut geçişi kendi **ağırlıklı bulut derinliğini** (geçirgenlik
  ağırlıklı ortalama mesafe) yazar ve kendi geçmişini onunla izdüşürür;
  geçmiş komşuluk kutusuna kırpılır. Sonuç tam çözünürlüğe bilateral
  büyütmeyle çıkar ve aerial perspective ile birleşir.

### 3.3 Bulut gölge haritası (dünyaya etki)

- Güneş yönünde, dünyaya sabitli ortografik 2D harita (varsayılan 1024²,
  kamera etrafında ~30 km, texel'e kilitli kayma). **Beer Shadow Map**
  (Frostbite): texel başına ön yüz derinliği + ortalama sönüm + maks optik
  derinlik → herhangi bir yükseklikte geçirgenlik tek dokudan.
- Tüketiciler:
  - RT: güneş NEE'si (yüzey + hacim + froxel)
  - RayFusion: directional güneş ışığına çarpan (gölge atlasıyla çarpılır)
  - **Froxel:** hava + sisin güneş terimi → bulut aralarından **ışık
    huzmeleri** (crepuscular rays) bedavaya gelir.
- Güneş = directional kuralı (2026-09-30) ile uyumlu: gölge haritası
  güneş ışığının bir **çarpanıdır**, ayrı bir güneş değil.

### 3.4 Bulutlu gökyüzü panoraması (ortam ışığı)

- Düşük çözünürlüklü (ör. 256×128) sky-view + bulut panoraması; güneş,
  bulut veya kamera yüksekliği değişince yeniden üretilir (zamanla
  dağıtılmış).
- Tüketiciler: RayFusion'ın ortam/IBL okuması (**hafızadaki karar: RayFusion
  dikişi = ambient okuması** — bulut tam buraya girer), RT'nin ikincil
  ışınları (§3.1), probe alanı.
- Kapalı havada zeminin kararması ve rengin griye dönmesi buradan gelir.

### 3.5 Aerial perspective

- 0–64 km froxel'in içindeki bulutlar froxel'den; ötesi (ufuk bulutları)
  transmittance LUT + sky-view'dan analitik. Bulut başına ağırlıklı derinlik
  kullanılır (Hillaire): uzak bulutlar maviye/pusa gömülür.

---

## 4. Veri modeli ve yetki

```text
CloudState                          (World'ün parçası, serileşir, anahtarlanır)
├── weather : source (procedural|texture), procedural params, texture path,
│             extent_m, evolution_speed
├── layers[] (hacimsel, en fazla 3):
│     base_altitude_m, thickness_m, coverage_scale, type_bias,
│     extinction_per_m, droplet_diameter_um, erosion, detail_scale_m,
│     enabled
├── cirrus  : altitude_m, coverage, opacity, scale_m, enabled
└── quality : rt (max_bounces, ...), realtime (steps, resolution_divisor)
```

- Rüzgâr **CloudState'te yok**: iklimden okunur (Faz 2 ilkesi — atmosfer
  ortak veri üretir, tüketiciler okur).
- Eski düz `cloud_*` / `cloud2_*` alanları ve sanatsal düğmeler **sökülür**
  (kural 5); eski projeler için bir kerelik çeviri: örtü/yükseklik/yoğunluk
  anlamlı eşlenir, sanatsal düğmeler atılır ve bu log'a yazılır.
- **Preset'ler** (tek tıkla, sonra düzenlenebilir): açık, güzel hava
  cumulus'u, dağınık, kırık, kapalı stratus, fırtına öncesi (cumulonimbus),
  yüksek cirrus, gün batımı altocumulus.

---

## 5. Performans bütçesi (hedef, 1080p, orta sınıf GPU)

| Geçiş | RayFusion | RT |
|---|---|---|
| Noise dokuları | bir kez (başlangıç) | bir kez |
| Weather map | değişince / rüzgârla kaydırma bedava | aynı |
| Gölge haritası | ≤ 0,5 ms, gerekince | aynı doku |
| Bulut march | ≤ 2 ms (¼ çöz.) | örnek başına; birikim |
| Panorama | ≤ 0,3 ms dağıtılmış | aynı doku |

RT için hedef "ilk saniyede okunur, 5–10 sn'de temiz". Bütçeler
`world.clouds.stats` ile ölçülür (sayaç + ms), tahmin edilmez.

---

## 6. IPC yüzeyi (kural 1)

| Metot | Ne |
|---|---|
| `world.clouds.get` / `set` | CloudState (katmanlar, cirrus, weather, kalite) |
| `world.clouds.apply_preset` | preset adıyla |
| `world.clouds.sample_density` | noktada σt (GPU'da küçük bir dispatch + geri okuma — CPU aynası yazılmaz, tek kaynak GPU) |
| `world.clouds.sample_transmittance` | iki nokta arası (ör. zemin→güneş) geçirgenlik |
| `world.clouds.shadow_at` | zemin noktasında gölge haritası geçirgenliği |
| `world.clouds.stats` | cihaz başına: doku hazır mı, march ms, dispatch sayacı, adım, katman başına march sayısı |
| `world.clouds.set_weather_texture` | kullanıcı dokusu |

---

## 7. Doğrulama (kabul)

Sayısal, sonra göz:

1. **Fiziksel tutarlılık:** 1 km cumulus sütununda `sample_transmittance`
   ≈ e^−(σt·1000) (tip sönümüyle), `sample_density` boş gökte 0.
2. **Beyaz fırın:** albedo 1, düzgün gökyüzü ışığı, güneş yok → bulut
   içi/dışı radyans = gelen radyans (RT). Enerji kaçağı/üretimi yakalar.
3. **Adım bağımsızlığı:** RayFusion adım sayısı 64 → 128: ortalama
   parlaklık farkı < %2 (enerji-korunumlu integrasyon).
4. **Parite:** aynı kamera, aynı gökyüzü şeridi → RayFusion / RT ortalama
   ve histogram; hedef ±%5. Tutmazsa oktav sabitleri ayarlanır ve ölçüm
   sabitin yanına yazılır.
5. **Gölge konumu:** `shadow_at` ile iki backend'de zemin gölgesi aynı
   noktada; güneş hareket edince gölge doğru yönde kayar.
6. **İki katman aynı anda** görünür, `stats` ikisini de sayar (bugünkü
   sessiz düşme kapanır).
7. **TLAS temiz:** bulut açılıp kapanınca `world.objects` sayısı değişmez.
8. **Determinizm:** aynı timeline karesi iki kez → bayt aynı bulut maskesi.
9. **Göz:** preset'ler referans fotoğraflarla yan yana (güzel hava cumulus
   öğle, gün batımı, kapalı, ufuk). ★ Sinsi olan: "güzel ama düz" bulut —
   çoklu saçılma eksikse bulutlar kirli gri, taban çok koyu olur ve bu
   "stil" sanılır; 2 ve 4 bunu sayıyla yakalar.

---

## 8. Söküm listesi

- `ensureInternalSkyCloudVolume` / `removeInternalSkyCloudVolume` ve
  `__RayTrophi_Internal_SkyCloudVolume` (hacim + `world.objects` girişi).
- `proceduralCloudDensity` ve volume shader'daki bulut dalı; `VkVolumeInstance`
  bulut alanları.
- `NishitaSkyParams` düz `cloud_*`, `cloud2_*`, `cloud_fft_map`,
  `cloud_use_fft` (CUDA FFT bulut — `CloudManager` zaten söküldü).
- miss.rmiss / closesthit / hair / volume world struct'larındaki bulut
  alanları (yeni pakete taşınır).
- OptiX (dondurulmuş): karar (a) — CloudState'ten eski OptiX alanlarına
  tek yönlü eşleme fonksiyonu, yalnız OptiX başlığına yazar.

---

## 9. Kararlar (2026-09-30, kullanıcı: "onaylıyorum" — dördü de öneri yönünde)

1. **RT bulut: yol izleme referans mı, RayFusion'la aynı yaklaşım mı?**
   Öneri: **yol izleme** (§3.1). Pahalı ama tek gerçek referans; RayFusion
   onun sayesinde kalibre edilir. Aynı yaklaşım iki yerde olursa parite
   bedava gelir ama "doğru" olan bilinmez.
2. **RT ikincil ışınlarda bulut:** panorama (hızlı, ölçülmüş yanlılık) mı
   tam march mı? Öneri: panorama varsayılan, kalite ayarıyla tam march.
3. **Weather map kapsaması:** 150 km / 2048² varsayılan; uçuş sahneleri
   için daha büyük kademeli (clipmap) sonraya.
4. **Cirrus:** bu fazda mı, sonraki alt partide mi? Öneri: 3c.

---

## 10. Alt partiler (her biri bir derleme)

| Parti | İçerik | Kabul |
|---|---|---|
| **3a Temel** | CloudState + JSON + anahtar + IPC; noise dokuları (compute, iki cihaz); weather map üretici + rüzgâr sürüklemesi; `sample_density/transmittance`; **eski bulut yolu söküldü** | §7-1, 6 (veri), 7, 8 |
| **3b RT referans** | atmosfer geçişinde delta/ratio tracking, NEE güneş+gök, çoklu saçılma, Jendersie-d'Eon faz; gölge haritası + RT tüketimi; aerial perspective | §7-2, 5 (RT), göz |
| **3c RayFusion** | ¼ çözünürlük march + zamansal; oktav çoklu saçılma (RT'ye kalibre); gölge haritası tüketimi; bulutlu panorama → ortam ışığı (iki backend); cirrus | §7-3, 4, 5 |
| **3d Dünya etkisi** | froxel'e gölge haritası (ışık huzmeleri); preset'ler; panel; performans ölçümü | §7-9, bütçe |

Sonra (Faz 4 ile): yağış perdeleri cumulonimbus altında, ıslaklık
birikimi; rüzgâr kayması ile katman başına farklı hız; buz kristali halo;
gece gökyüzü (ay, yıldızlar).

---

## Kaynaklar

- S. Hillaire, *Physically Based Sky, Atmosphere and Cloud Rendering in
  Frostbite* (SIGGRAPH 2016) — enerji-korunumlu integrasyon, Beer Shadow Map.
- S. Hillaire, *A Scalable and Production Ready Sky and Atmosphere Rendering
  Technique* (EGSR 2020) — mevcut LUT'larımız.
- A. Schneider, *The Real-time Volumetric Cloudscapes of Horizon Zero Dawn*
  (SIGGRAPH 2015) + *Nubis* (2017/2022) — weather map, tip×yükseklik
  profili, erozyon, zamansal ¼ çözünürlük.
- M. Wrenninge et al., *Oz: The Great and Volumetric* (SIGGRAPH 2013) —
  çoklu saçılma oktav yaklaşımı.
- J. Jendersie, E. d'Eon, *An Approximate Mie Scattering Function for Fog
  and Cloud Rendering* (SIGGRAPH 2023) — damlacık çapından faz fonksiyonu.
- J. Novák et al., *Monte Carlo Methods for Volumetric Light Transport
  Simulation* (EG STAR 2018) — delta / ratio tracking.
