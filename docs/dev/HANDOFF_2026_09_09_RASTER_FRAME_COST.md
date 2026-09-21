# Devir notu — raster kare maliyeti, 2026-09-09

> **Durum:** ARŞİV — **iki açık hedefi de kapandı ve ölçüldü (2026-09-10).**
> §3 (erken derinlik testi kapalı) doğrulandı ve düzeltildi: `early_fragment_tests`
> beyanı eklendikten sonra ön geçiş **219,8 → 57,9 ms** kazandırdı, yani §3'ün
> teşhisi doğruydu. §4 (transmission replay) da kapandı: **103,2 → 1,6 ms**.
> Karenin bugünkü hâli ve sıradaki hedefler:
> [RASTER_FRAME_COST_2026_09_10.md](RASTER_FRAME_COST_2026_09_10.md).
> ★ Bu not kayıt için duruyor — §7'deki üç ölçüm tuzağı ve §8'deki araç notları
> hâlâ geçerli referanstır.

---

## 0. Tek cümlelik durum

**Foliage sahnesinde çakılmanın sebebi RayFusion değil.** RT gölge dispatch'i
karenin **%0,4'ü** (1,17 ms). Kare `main_pass` + `transmission` içinde;
transmission ise **72 bin üçgenlik su yüzeyi için 27,3 milyon üçgen** çizdiriyor.

Ve en olası kök: **alpha-test `discard` erken derinlik testini kapatıyor**, yani
gizli foliage katmanları TAM materyal shader'ından geçiyor (§3). Kullanıcının
"kameraya birkaç ağaç girse çakılıyor" gözlemi tam olarak bunun imzasıdır.

---

## 1. Ölçülen kare

Alet: `viewport.reset_frame_timings` → kamerayı sür → `viewport.frame_timings`.
Sürücü: `scripts/ipc/Probe-FrameStages.ps1`. Preset **performance**,
1680x945, 27,3 M görünür üçgen, 34 draw, 27 kare, hacim yok, 1 sahne ışığı.

| aşama | GPU ms | pay |
|---|---|---|
| main_pass | 147,8 | %52 |
| **transmission** | **103,2** | **%36** |
| depth_prepass | 20,4 | %7,2 |
| rt_shadow | 1,17 | %0,4 |
| overlay | 0,85 | %0,3 |
| sky + gpu_cull + post + transmission_prep | ~0,4 | %0,1 |
| shadow_atlas | — | **SKIPPED** |
| volume_sdf | — | SKIPPED |

CPU tarafı kare başına **0,69 ms**. Yani tamamen **GPU-bound**; kullanıcının
gözlemi doğru.

★ Bu tablo aynı sahnede kameraya göre oynar (27,3 M üçgen bu kadrajda). İki
tabloyu karşılaştırırken `applied` satırındaki `tris`/`draws` eşit değilse
karşılaştırma geçersizdir.

---

## 2. Bu oturumda ÇÖZÜLENLER (hepsi canlı doğrulandı)

**1. Derinlik ön geçişi ana geçişle INVARIANT değildi.**
Ön geçiş `VP*M*p`, ana geçiş `VP*(M*p)` yazıyordu — matematiksel olarak aynı,
sayısal olarak değil. Ekran uzayı maskesinin 2 ULP'lik derinlik kapısı
tutmuyor, fragman cascade'e düşüyor; cascade devredildiği için **tamamen
aydınlık** kalıyordu.
*Kanıt:* HEAD'deki `.spv` — ön geçişte 1 `OpMatrixTimesMatrix`, 0 `Invariant`.
Düzeltme sonrası iki shader da aynı `OpMatrixTimesVector` dizisi ve
`Invariant`; `-O` altında da korunuyor.

**2. Ekran ışın gölgesinde `tmin` sabit 0,02 m idi.**
Işın başlangıcı ölçülmüş değil geri kurulmuş bir noktadır; belirsizliği z² ile
büyür (1 km'de ~0,5 m), ışın kendi zeminini vuruyordu.
*Kanıt:* ekran görüntüsü A/B — RT kolunda zeminde 1 piksel yüksekliğinde yatay
çizgiler, cascade kolunda temiz; ön planda temiz, ufka doğru yoğun.
*Düzeltme:* belirsizlik tahmin edilmiyor, projeksiyondan türetiliyor
(`tmin = max(0,02, slack*3)`). Kullanıcı onayladı.

**3. Maske tek ışığı kapsıyordu, oysa dünya güneşi ile senkron directional ışık
aynı yöndeydi.** Raster tek fiziksel güneş için iki cascade kümesi çiziyordu.
*Düzeltme:* maske başlığına kapsama kümesi eklendi (16 → 32 bayt).
*Kanıt:* canlı — `shadow_atlas` **38 ms → SKIPPED**, `cascades_replaced=4`,
`shadowed=0`.
Kapsama kararı **UI senkron anahtarına değil ölçülen yöne** bakar
(`dot >= 0,9999`). Bayrak bir niyet, yön bir ölçümdür.

**4. Ölçüm aletinin kendi iki kusuru.** Prepass kapalıyken `depth_prepass`
191,6 ms rapor edip aynı satırda `SKIPPED` diyordu; `applied.depth_prepass`
istenen kolu okuyordu.
*Kanıt:* canlı tabloda `depth_prepass` 20,4 ms ve `applied.depth_prepass=True`
tutarlı. `transmission_prep` ayrı aşama olarak ayrıldı (11 aşama).

---

## 3. ★★★★★ AÇIK BİRİNCİ HEDEF — erken derinlik testi KAPALI (alpha-test overdraw)

Kullanıcı gözlemi (2026-09-09): **"kameraya birkaç ağaç bile girse performans
çakılıyor."** Bu, üçgen sayısının değil **ekran kaplaması × overdraw**'ın
bağlayıcı olduğunu söyler. Ölçülen iki kare bunu destekliyor:

| kadraj | görünür üçgen | main_pass |
|---|---|---|
| A | 21,2 M | 215,6 ms |
| B | 27,3 M | 147,8 ms |

**Üçgen %29 ARTARKEN main_pass %31 DÜŞTÜ.** (Uyarı: iki farklı kamera, kontrollü
bir A/B değil — bir gözlem. Kontrollü ölçüm için §8'deki script hazır.)

### Koddaki kanıt — üç olgu

1. `material_preview_frag.frag` içinde **4 adet `discard`** var (satır 623, 872,
   895, 898).
2. Depoda **hiçbir shader `layout(early_fragment_tests) in;` beyan etmiyor**
   (`grep -r early_fragment_tests shaders/` → 0).
3. Ana material pipeline (`mpDS`, VulkanViewportBackend.cpp:1816):
   `depthTestEnable=TRUE`, **`depthWriteEnable=TRUE`**,
   **`depthCompareOp=LESS_OR_EQUAL`**.

`discard` içeren bir fragment shader'da derinlik testi **varsayılan olarak
shader'dan SONRAYA ertelenir** — shader fragmanı öldürebileceği için. Yani ana
geçişte, önündeki yapraklarca tamamen örtülmüş her foliage kartı fragmanı
**bütün materyal shader'ını** (doku fetch'leri, ışıklandırma, RT gölge maskesi
okuması, probe okumaları) koşturur ve ancak ondan sonra derinlik testiyle
atılır. Kaplama arttıkça maliyet katman sayısıyla çarpılır.

### Neden şimdi çözülebilir

Derinlik ön geçişi **zaten koşuyor** (20,4 ms) ve hangi fragmanın kazandığını
**alpha cutout dahil** çözmüş durumda. Ana geçiş şuna çevrilebilir:

- `depthCompareOp = VK_COMPARE_OP_EQUAL`
- `depthWriteEnable = VK_FALSE`
- fragment shader'da `layout(early_fragment_tests) in;`

Bu, overdraw'ı piksel başına **1 gölgelenen fragmana** indirir. Aynı pipeline'ı
transmission replay de kullandığı için kazanç **iki aşamada birden** görünür.

### Ön koşullar ve tuzaklar — sırayla

- ★★★★★ **Ön koşul ZATEN SAĞLANDI:** `EQUAL` ancak ön geçiş ile ana geçişin
  `gl_Position`'ı bit bazında aynıysa çalışır. Bu oturumda düzeltilen §2/1
  numaralı hata tam olarak buydu; o düzeltme olmadan `EQUAL` sahnenin çoğunu
  siler. Bu iş o yüzden ancak şimdi yapılabilir.
- ★★★ **Ön geçiş bazı materyalleri BİLEREK eliyor.** `material_preview_shadow_frag.frag:40`
  `transmission_tex != 0u` ve bazı flag'leri `discard` ediyor — yani cam/su ön
  geçişte YOK. O materyaller `EQUAL` altında ana geçişte de kaybolur. Çözüm:
  ana geçiş için iki pipeline (ön geçişin kapsadıkları `EQUAL`, kalanlar
  `LESS_OR_EQUAL`), ya da eleme kümesini iki yerde aynen aynalamak. Bu depoda
  "iki yerde tutulan kural" defalarca sessiz arızaya döndü; tercih birincisi.
- ★★ `early_fragment_tests` ile `discard` artık derinlik yazmayı engellemez —
  ana geçişte derinlik yazma KAPALI olacağı için sorun değil, ama ikisi
  birlikte değiştirilmeli.
- ★ Ön geçiş kapalıyken (RT gölge yokken) bu yol geçersizdir: `EQUAL` pipeline'ı
  yalnızca `depthPrepassActive` iken bağlanmalı.

**Kabul testi:** kamerayı ağaçların arasına sok. `main_pass` ms'i, ekran
kaplaması artarken artmamalı. §8'deki `Probe-Overdraw.ps1` bunu doğrudan verir.

---

## 4. AÇIK İKİNCİ HEDEF — transmission replay'i mesh başına ayıkla

### Ne yapıldı

`recordMaterialPreviewTransmissionPass` bütün sahneyi tam materyal
pipeline'ından ikinci kez çiziyor. Kapısı yalnızca
`if (m_materialPreviewTransmission)` idi — yani "kaynak kuruldu mu", ki init'te
koşulsuz kurulur. **Sahnede geçirgen materyal olup olmadığına hiç bakmıyordu.**

Sahne seviyesinde doğru kapı kuruldu (`materialPreviewTransmissionHasWork()`):
geçirgen materyal VAR **veya** hacim VAR (SDF yüzeyi ve hacim geçişleri bu
replay'in içinde kaydediliyor ve ikisi de `m_volumeCount > 0`'a bakıyor).
Önbelleklenmiyor — bayat bir `false` **camın sessizce kaybolması** demekti.

### Neden performans DEĞİŞMEDİ (ölçüldü)

Sahnede gerçekten geçirgen materyal var, yani kapı **doğru** davranıyor:

- 131 materyalden **40'ı `transmission = 1.0`** — hepsi terrain'in ürettiği su:
  `Water_Mat_AutoLake_*` ve `RiverWater_AutoRiver_*`.
- 30 objeden **29'u** bu materyalleri taşıyor (5 göl + 24 nehir segmenti).
- Bu su yüzeylerinin **toplam üçgeni: 72.030**.
- Replay ise **27.275.617 üçgen** çiziyor.

**Yaklaşık 379 kat fazla çizim.** Sahne seviyesindeki kapı bu sahne için yanlış
granülerlik.

### Sıradaki adım

Replay döngüsü `m_rasterMeshes` üzerinde dönüyor. **Mesh başına ayıklama:** bir
mesh ancak geçirgen bir materyal içeriyorsa yeniden çizilsin. Veri mevcut:
`RasterMeshBuffer::cpuMatIds` + `m_cachedGpuMaterials`.

★★★ **Tehlike ve neden şimdi yapılmadı:** bu bir önbellek gerektirir
(`cpuMatIds` taraması O(vertex), her kare yapılamaz) ve **bayatlama yönü "cam
kaybolur"dur** — sessiz görsel regresyon, kimse bunu bug diye raporlamaz. Bu
depoda tam olarak bu sınıf tekrar tekrar geri geldi. Önbelleğin geçersizleme
noktaları: (a) `m_cachedGpuMaterials` güncellenince, (b) `cpuMatIds` yazan HER
yer — ki `matIdsHashValid` için zaten aynı sözleşme var, ona bağlanabilir.

**Kabul testi:** sulu/camlı sahnede su hâlâ kırılma yapıyor VE `transmission`
aşaması ~0,1 ms'ye düşüyor.

★ İlgili ve HENÜZ DERLENMEMİŞ başka bir fikir var:
[REALTIME_PERF_FIRST_FIXES.md](REALTIME_PERF_FIRST_FIXES.md) "provably opaque
transmission replay'i doku işinden önce ele" diyor — o **fragman seviyesinde**
bir erken çıkış, buradaki ise **çizim seviyesinde**. İkisi çakışmaz, toplanır.

---

## 5. `main_pass`'e DİĞER açı — LOD ve overdraw kaynağı

§3 `main_pass`'e fragman tarafından saldırıyor. Geometri tarafı ayrı ve bu
oturumda ölçülmedi; zaten yazılmış analizler var, önce onlar okunmalı:

- [RASTER_MICROTRIANGLE_WALL.md](RASTER_MICROTRIANGLE_WALL.md) — LOD 98.500'e
  karşı 96 üçgen, 1:1026 uçurum; 292 tam instance üçgenlerin %98,4'ünü taşıyor.
- [GENERAL_MESH_LOD_DESIGN.md](GENERAL_MESH_LOD_DESIGN.md) — yapıya göre
  sınıflandırma tasarımı, kod yok.
- [RASTER_SCENE_LIGHTING_FRAGMENT_COST.md](RASTER_SCENE_LIGHTING_FRAGMENT_COST.md)
  — maliyet piksel değil ÜÇGEN sayısıyla doğrusal; foliage kartlarında overdraw.

---

## 6. ÖLÇÜLMEMİŞ — `depth_prepass` 20,4 ms neyi kazandırıyor

Ön geçişin ana geçişi ne kadar ucuzlattığı **hiç ölçülmedi**.
`-Compare depth_prepass` doğrudan cevaplar, **ama RT gölge prepass'i zorla
açar**: kolu kapatmak için önce `viewport.set_rt_shadow @{enabled=$false}`.
Bu sırayı atlarsan A/B'nin iki kolu da prepass'li olur ve fark "gürültü" çıkar.

---

## 7. ★★★ ÖLÇÜM TUZAKLARI — bu oturumda üç kez yanlış "0" üretildi

Bir sonraki ajan bunları bilmeden ölçerse aynı yanlış sonuca varır.

1. **IPC parametre adları alandan alana değişiyor.** `material.info` → `name`;
   `material.get_param` ve `material.textures` → `material_name`;
   `material.of_object` → `object_name`; `scene.object_info` → `object_name`.
   Yanlış adla çağrı **hata fırlatır**; PowerShell'de `catch {}` ile yutulursa
   sonuç sessizce boş küme olur. "0 geçirgen materyal" diye rapor edilmek
   üzereydi — gerçek 40. **Döngüde hatayı SAY ve raporla.**
2. **`material.info` özetinin söylediğini döndürmüyor.** Özeti "Return a
   material's full parameter set" diyor; döndürdüğü `{id, name, type}`.
   Parametre için `material.get_param` kullan. Bu, "documented_coverage VARLIK
   ölçer DOĞRULUK değil" kuralının canlı bir örneği.
3. **`scene.list_objects` düz string dizisi döndürür**, nesne değil.

Kalıcı olarak geçerli olanlar: raster viewport **yalnız dirty olunca** çizer
(kamerayı sür), **pencere önde değilse hiç çizmez**, ve bir IPC yazısı ile
ölçüsü arasına bir kare koymak gerekir.

---

## 8. ARAÇLAR

```powershell
Invoke-RtIpc viewport.capture @{ enabled = $true }
# kamerayı sür
$shot = Invoke-RtIpc viewport.get_screenshot     # alan: image_base64
[IO.File]::WriteAllBytes("shot.jpg", [Convert]::FromBase64String($shot.image_base64))
Invoke-RtIpc viewport.capture @{ enabled = $false }
```

**Raster viewport'u** verir — `viewport.render_frames`'in aksine, o path
tracer'ı sürer. §2'deki 2 numaralı kök neden bununla, iki kolun görüntüsünü
yan yana koyarak tek turda bulundu; sayısal ölçüm turları aynı soruyu ayırt
edememişti. **Artefaktın ekrandaki yönü ve dağılımı, sayıların taşımadığı
bilgiyi taşıyor.**

### `scripts/ipc/Probe-Overdraw.ps1` — YAZILDI, HENÜZ KOŞMADI

§3'ün kararını veren ölçüm. Aynı sahneyi aynı ayarlarla **uzak** ve **yakın**
kadrajda ölçer, üçgen sayısı ile `main_pass` ms'ini karşılaştırır:

- ikisi **aynı yönde** → geometri/vertex bağlı,
- **ters yönde** (üçgen düşer, maliyet artar) → **fragman bağlı: overdraw /
  alpha-test**.

Üç kollu (uzak / yakın / uzak-tekrar), yani tekrar kolu sapmayı fiyatlar.
Bu oturumda yazıldı ama uygulama kapandığı için **koşturulamadı** — bir sonraki
ajanın ilk işi bu olmalı, çünkü §3'ün önceliğini bu sayı doğrular ya da çürütür.

```powershell
.\scripts\ipc\Probe-Overdraw.ps1 -CloseFactor 0.12 -Frames 22
```

---

## 9. Bu partide değişen dosyalar

**Shader:** `material_preview.vert`, `material_preview_shadow.vert`,
`material_preview_shadow_opaque.vert`, `material_preview_rt_shadow.glsl`,
`rayfusion_rt_shadow.comp`.

**C++:** `include/Backend/VulkanBackend.h`,
`src/Viewport/MaterialPreviewRtShadow.cpp`,
`src/Viewport/MaterialPreviewShadow.cpp`,
`src/Viewport/MaterialPreviewTransmission.cpp`,
`src/Backend/VulkanViewportBackend.cpp`.

Maske başlığı 16 → **32 bayt** oldu. Eski shader + yeni C++ (veya tersi)
çökmez, pikselleri kaymış okur — ikisi birlikte derlenmeli.


## Live follow-up (2026-09-09)

2026-09-10 closure: see
[MATERIAL_COVERAGE_PIPELINE_2026_09_10.md](MATERIAL_COVERAGE_PIPELINE_2026_09_10.md)
for repeated live cutout A/B (Balanced + RT: 324 ms to 82 ms), user-confirmed
foliage/transmission visual tests, final RT-default/import-MASK source changes
awaiting build, and the remaining main-pass/depth-prepass bottlenecks.

See [RASTER_FRAME_COST_AB_2026_09_09.md](RASTER_FRAME_COST_AB_2026_09_09.md) for same-camera world/opacity/transmission measurements. Balanced preset: Nishita 372 ms vs solid world 201 ms; foliage opacity zero 177 ms vs restored 378 ms. Authored water transmission zero did not stop replay. All 44 modified scalar values were read back restored; sky IBL reports ready with no fallback. These measurements supersede the unqualified geometry-bound verdict of Probe-Overdraw.ps1.
