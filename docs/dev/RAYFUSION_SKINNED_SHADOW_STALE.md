# RayFusion: skinli mesh'in gölgesi ilk karenin pozunda donuyordu

> **Durum:** AKTİF — belirti CANLI DOĞRULANDI, düzeltme DERLENDİ ve ÇALIŞIYOR.
> Maliyet ÖLÇÜLDÜ: **2,18 ms refit / 1,20 ms render karesi.** Bedeli kırmak
> için ölçüm üçe bölündü (YAZILDI/DERLENMEDİ). (2026-09-13)

## Belirti (DOĞRULANDI)

Skin animasyonlu bir nesnenin RayFusion ışın gölgesi **ilk karedeki konumda
kalıyor.** Kullanıcı bir önceki gecenin build'inde tam testle doğruladı.

Hiçbir sayaç bunu göstermez: `rayfusion.scene_as` `ready=true`, `blas_count`
dolu, `builds` sabit — ki **`builds`'in sabit olması DOĞRUDUR**, kamera hareketi
rebuild etmemeli. Aletin tamamı yeşilken izlenen sahne yanlıştı.

★ Kayda değer: bu iş bir şüphe olarak başladı, kod okunarak doğrulandı, sonra
**ekranda** doğrulandı. Üç aşama ayrıdır ve ilk ikisinde "ölçüldü" demek yanlış
olurdu. Hangi yolun gölgeyi çizdiğini ayırmanın yolu: `viewport.status`
(shading) + `viewport.rt_shadow` (`ready`, `cascades_replaced`) — Solid/Matcap'te
gölgeyi cascade shadow map çizer ve o her kare canlı skinli tampondan rasterize
edildiği için her zaman doğrudur, yani hata orada görünmez.

## ★★★★★ CANLI MALİYET ÖLÇÜMÜ (2026-09-13, IPC ile)

Düzeltme derlendi, gölge doğru çalışıyor. Sonra maliyet ölçüldü. Sahne:
9 BLAS (8'i skinli), `material` shading, 1680x945, RT gölge gerçekten devrede
(`ready=true`, `cascades_replaced=3`, 1.587.600 ışın/kare), **realtime
animasyon oynuyor** (timeline'dan sürülmüyor).

| ölçüm | değer |
|---|---|
| `blas_skinned` / `blas_count` | **8 / 9** |
| `skin_refit_failures` | **0** |
| `builds` | **1** (refit çalışıyor, tam kurulum yok) |
| `signature_ms` | 0,0027 ms (kapının kendi bedeli — ihmal edilebilir) |
| kare GPU / CPU ortalaması | **1,20 ms / 0,083 ms** (256 kare, ~84 fps) |
| `rt_shadow` geçişi | 0,13 ms GPU |
| refit hızı | 261,6 `skin_refits`/s ÷ 8 BLAS = **~33 parti/s** |
| **`last_skin_refit_ms`** | **2,18 ms** |

★★★★★ **Refit, animasyonlu bir karede 2,18 ms — bütün render karesinin
(1,20 ms GPU) yaklaşık 1,8 KATI.** Bu israf değil, gerekli iş; ama ölçüldüğü
yerde kabul edilemez ve "Açık kalan"daki maddeleri ertelenebilir olmaktan
çıkarır.

### ★★★★ Bu ölçümde YAPILAN OKUMA HATASI (dersin kendisi)

İlk okumada `timeline.get_frame` 0'da sabit görülüp **"sahne duruyor, refit
boşa koşuyor"** sonucuna varıldı. Yanlış: sahne timeline'dan sürülmüyordu,
**realtime animasyon oynuyordu.** O sayaç realtime oynatımı takip etmez.

★★★ Sayılar zaten doğruyu söylüyordu ve gözden kaçtı: render 84 fps iken refit
**33/s**. Kare başına olsaydı 84 olurdu. 33, ~30 fps'lik bir animasyon
oynatımının poz başına TAM BİR refit'idir — yani hedeflenen davranış.
**Bir sayacın HIZI, neyi saydığının kanıtıdır.** Onu göz ardı edip başka bir
sayacın birimini varsaymak bu deponun klasik hatası (CLAUDE.md: "Tüketicinin
BİRİMİNİ oku", "Varsayılan bir ölçüm değildir").

Sonuç: `noteSkinnedPose` (poz hash'i) **bu senaryoda kazanç sağlamaz** —
çağıran zaten yalnızca poz ilerlediğinde çağırıyor. Yine de doğrudur ve
animasyon durdurulup sahne incelenirken bedeli sıfırlar; ucuz bir sigorta
olarak kalır, performans düzeltmesi olarak değil.

### Sıradaki ölçüm: 2,18 ms'nin NERESİ?

Tek sayının üzerine optimize edilemez: içinde dört ayrı şey var ve dördünün
çözümü ayrı. `last_skin_refit_ms` üçe bölündü ve IPC'ye açıldı:

| alan | ne | çözümü |
|---|---|---|
| `last_skin_drain_ms` | kuyruktaki karenin bitmesini bekleme | bizim işimiz değil; refit'i kare komut tamponuna taşımak gerekir |
| `last_skin_blas_ms` | submit + fence + 8 BLAS refit | `dispatchSkinning` gibi skinning tamponuna gömmek |
| `last_skin_tlas_ms` | tam TLAS rebuild (**ikinci kez drain eder**) | MODE_UPDATE (`recordGpuTLASUpdate` gibi) |

★ `last_skin_tlas_ms`'e özellikle bak: `rebuildRayFusionTLAS()` kendi içinde
bir drain daha yapıyor, yani kare başına İKİ drain ödeniyor olabilir.

★ Yan bulgu: `timeline.get_frame` **çıplak skaler** döndürüyor, nesne değil.
Probe script'i `.frame` okuyup `$null` alıyordu — "başladığın kareye geri dön"
adımı sahneyi sessizce 0. kareye atıyordu. Düzeltildi.

## Kök neden

★★★★★ **Kapı, tanımı gereği göremeyeceği bir şeye bakıyordu.**

`ensureRayFusionSceneAS()` iki imza tutuyor:

| imza | neyi hash'liyor |
|---|---|
| `geometrySignature` | meshKey, `vertexCount`, **vertexBuffer'ın VkBuffer handle'ı**, `indexCount`, indexBuffer handle'ı |
| `instanceSignature` | meshKey, instance transformu, mask |

GPU skinning (`VulkanViewportBackend::syncRasterSkinnedVertices` →
`dispatchSkinningToBuffers`) çıktısını **`mesh.vertexBuffer`'ın içine, yerinde**
yazıyor. Yani:

- handle aynı,
- device address aynı,
- vertex sayısı aynı,
- **sadece içerik değişiyor.**

Her iki imza da eşleşiyor → `return true` → hiçbir şey yeniden kurulmuyor.
BLAS (`createTriangleBLAS_Device`) o tamponu **ödünç** aldığı için gölge ışını
yeni pozisyonları okuyor gibi görünebilir, ama **BVH kurulduğu karenin pozuna
göre kurulmuş durumda** ve `allowUpdate=false` ile kurulduğu için refit bile
edilemiyordu.

★★★ Bu, deponun en pahalı hata sınıfının yeni bir örneği: **ölçü aleti sağlık
raporluyordu.** `blas_indexed` tam olarak aynı gerekçeyle eklenmişti (welded
mesh'i index'siz kurmak da "ready" bir AS üretir); bu da onun kardeşi.

★★ İkinci, daha sinsi katman: **TLAS de bayat.** TLAS her instance için
BLAS'ın kurulma anındaki dünya AABB'sini taşır ve ışını alt seviyeye inmeden
o kutuda eler. Yalnızca BLAS'ı refit etmek, kolunu eski kutusunun dışına
savuran bir karakterde gölgenin **tamamen kaybolduğu** bir bölge bırakır —
"özellik yok" gibi değil, "gölgeden ısırık alınmış" gibi görünür.

## Düzeltme

1. **`m_rasterSkinGeneration`, tek kapısı `noteSkinnedPose(boneMatrices)`** —
   üç skinning yolu da (viewport GPU compute, viewport CPU fallback, adapter
   CPU) onu çağırır ve generation yalnızca **POZ** değiştiğinde artar.
   ★ Dürüstçe: ölçülen sahnede kazanç sağlamaz, çünkü çağıran zaten yalnızca
   poz ilerlediğinde çağırıyor. Doğru ve ucuz bir sigortadır (durdurulmuş
   animasyonda bedeli sıfırlar), performans düzeltmesi değil.
2. **`createTriangleBLAS_Device(..., bool allowUpdate)`** — `ALLOW_UPDATE`
   yalnızca `mesh.hasSkinning` olan mesh'e verilir (bütün sahneye vermek,
   hiç refit edilmeyen statik %99'un traversal'ını yavaşlatmak olurdu) ve o
   BLAS'ın scratch tamponu `skinScratchBuffer`'da **saklanır** (build boyunda —
   periyodik MODE_BUILD sıfırlaması onu gerektiriyor).
3. **`recordTriangleBLASRefit(cmd, blasIndex)`** — CPU'dan yükleme YOK;
   skinning compute'un yazdığı tampon üzerinde MODE_UPDATE. `updateBLAS()` bu
   iş için kullanılamaz: CPU pointer'ı alır ve skinli sonucun üstüne base pose
   yazardı. 32 refit'te bir MODE_BUILD ile ağacın dejenere olması engellenir
   (`refitHairAABB_BLAS` ile aynı ritim ve aynı gerekçe). **Kaydeder, submit
   etmez:** bütün skinli BLAS'lar tek komut tamponuna girer. Mesh başına bir
   submit+fence olsaydı bir karakter birkaç stall ederdi — gövde, saç kapağı ve
   kıyafet ayrı skinli mesh'lerdir.
4. **`refreshRayFusionSkinnedBLAS()`** — kapının içinde, *instance imzasından
   ÖNCE*. Sadece kemikle animasyon edilen bir karakterin node transformu hiç
   değişmez; bu kolu instance dalının içine koymak en yaygın skinli sahnenin
   hiç refit edilmemesi demek olurdu. Refit sonrası **TLAS yenilenir**.
5. **Sayaçlar** — `blas_skinned`, `skin_refits`, `skin_refit_failures`,
   `last_skin_refit_ms`; `rayfusion.scene_as` üzerinden IPC + Python'a açık.
   `blas_skinned` refit'in **mümkün** olduğunu, `skin_refits` **olduğunu**
   söyler. İkisi ayrı olmak zorunda: birincisi tek başına yine varlık ölçer.

## Refit gerekli mi, ve RT render için mi yazıldı?

İki ayrı soru; ikisinin de cevabı **Vulkan RT render backend'inde zaten yazılı.**

**1. RayFusion raster, ama gölge ışınla üretiliyor.** Viewport rasterize eder;
RT gölge maskesi `rayfusion_rt_shadow.comp` içinde **ray query** ile gerçek bir
TLAS/BLAS üzerinden çizilir. Bir ışının görebildiği tek şey AS'tir — raster
tarafta yapılacak hiçbir şey bunu düzeltmez. Skinli bir mesh'in ışın gölgesini
doğrultmanın üç yolu var: BLAS'ı refit et, her kare yeniden kur (daha pahalı),
ya da skinli mesh'i ışın gölgesinden çıkarıp cascade'e bırak (kalite dikişi).

**2. Bu, RT render backend'i için YAZILMADI — orada zaten vardı.** İki ayrı AS,
iki ayrı BLAS listesi:

| | BLAS'ı kuran | skinli BLAS `allowUpdate` | kare başına refit |
|---|---|---|---|
| RT render backend | `createBLAS` (`VulkanBackend.cpp:8241`) | **evet** (`hasSkinning` veya dinamik ad öneki) | **evet** — `dispatchSkinning`, skinning komut tamponunun İÇİNDE |
| RayFusion viewport AS | `createTriangleBLAS_Device` | **hayır** (düzeltmeden önce) | **hiç yoktu** |

Nihai render'da gölgenin doğru olmasının sebebi tam olarak budur: o yol kare
başına refit ediyor ve **hep ediyordu.** Bu partide o dosyalara dokunulmadı;
eksik olan RayFusion'ın kendi AS'iydi.

**3. Ağırlık sorusunun cevabı da orada.** Kare başına refit bu kod tabanında
kanıtlanmış, hâlihazırda yürürlükte olan tasarım. MODE_UPDATE tam kurulumun
küçük bir kesridir ve yalnızca skinli mesh'e uygulanır — statik geometri
hiçbir şey ödemez. Ağır olan refit değil, **etrafındaki tesisattır**: submit +
fence bekleme, drain, TLAS rebuild. `dispatchSkinning` refit'i skinning'in
komut tamponuna gömerek ekstra submit'i sıfırlar; bu partide refitler en azından
tek tampona toplandı, kalanı "Açık kalan"da.

## Dokunulan dosyalar

- `source/src/Viewport/RayFusionSceneAS.cpp` — refit, kapı, sayaçlar
- `source/include/Backend/VulkanBackend.h` — imzalar, `m_rasterSkinGeneration`
- `source/include/Backend/IBackend.h` — status alanları
- `source/src/Backend/VulkanViewportBackend.cpp`, `.../VulkanBackend_Raster.cpp` — generation bump
- `source/include/Api/RtApiRayFusion.h`, `source/src/Api/RtApiRayFusion.cpp`,
  `.../RtIpcRayFusion.cpp`, `.../RtPythonRayFusion.cpp` — API/IPC/Python
- `scripts/ipc_descriptor_overlay.json` + üretilmiş `RtIpcMethodDescriptors.cpp`
- `scripts/ipc/Probe-RayFusionSkinnedShadow.ps1` (+ `x64/Release/` kopyası)

Yeni `.cpp` yok, `.vcxproj` dokunulmadı.

## Açık kalan

- **Kalan tek stall: refit partisinin kendi submit'i.** Artık bütün skinli
  BLAS'lar TEK komut tamponuna kaydediliyor, ama o tampon yine kendi
  `beginSingleTimeCommands`/`endSingleTimeCommands` çiftiyle gidiyor — kare
  başına bir fence bekleme. Kanıtlanmış hedef `dispatchSkinning`: refit'i
  **skinning'in komut tamponuna** kaydeder, ekstra submit sıfırdır. Viewport'ta
  bunu yapmak `dispatchSkinningToBuffers`'a BLAS indeksini geçirmeyi gerektirir
  (mesh→BLAS eşlemesi `RayFusionSceneASResources`'ta duruyor).

  **Bunu A/B ile ÖLÇME.** Koşulsuz gereksiz bir submit+fence'i kaldırmak bir
  takas değildir — "belki yavaşlar" dalı yoktur, o yüzden gerekçelendirmek için
  ölçüm istemez. (Aynı muhakemenin bu depodaki emsali: `VulkanBackend.cpp`
  ~10183, hair prepass zaten TLAS kuracakken ikinci `updateTLAS`'ı "redundant
  full submit+wait that only stalls the CPU" diye kaldırıyor.) Gereken tek sayı
  `last_skin_refit_ms`'in **tek bir okuması**, ve o da "doğru mu" değil
  **"şimdi yapmaya değer mi"** sorusunu cevaplar: küçükse bu madde gürültüdür,
  drain ve TLAS rebuild baskındır.

  ★★★★ **ASIL RİSK HIZDA DEĞİL: merge bir garantiyi FENCE'ten BARRIER'a
  taşır.** Bugün skinning submit edilip fence'le beklendiği için vertex
  yazmalarının görünürlüğü tartışmasız. Aynı tampona girdiklerinde garantiyi
  barrier verir — ve mevcut barrier YETERSİZ:

  | | `dstAccessMask` | hedef stage |
  |---|---|---|
  | `dispatchSkinningToBuffers` (bugün) | `VERTEX_ATTRIBUTE_READ` | `VERTEX_INPUT` |
  | `dispatchSkinning` (kanıtlanmış) | `ACCELERATION_STRUCTURE_READ_KHR` | `ACCELERATION_STRUCTURE_BUILD_KHR` |

  Refit'i o tampona koyup barrier'ı genişletmezsen AS build, skinning'in
  yazmalarını görmeden okuyabilir. Belirtisi hata değil: ara sıra yanlış
  geometri ya da sürücü reset'i — yani **bu maddenin doğrulaması süre değil,
  `Probe-RayFusionSkinnedShadow.ps1`'in hâlâ geçmesidir.**
- **TLAS tam rebuild.** RT render yolu da tam olarak öyle yapıyor (`updateTLAS`
  aslında `createTLAS(allowUpdate=true)`), yani bu bir gerileme değil — ama 65k
  instance'lı scatter ormanı + animasyonlu karakter aynı sahnedeyse iki yol da
  aynı duvara çarpar. Gerçek çözüm `recordGpuTLASUpdate` gibi MODE_UPDATE.
- Viewport'un **kısmi** GPU skinning başarısızlığı (bazı mesh GPU, bazısı CPU)
  durumunda generation yalnızca CPU kolundan bump ediliyor. Nadir yol; GPU
  descriptor havuzu tükenmediği sürece görülmez.
