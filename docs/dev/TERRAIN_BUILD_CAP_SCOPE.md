# maxDepositionMeters neyi bağlıyor — ölçüldü

> **Durum:** AKTİF — 2026-08-27. Ölçüm yapıldı ve iki ölçü aleti arızası
> düzeltildi (derlenmedi). Kadranın kapsamı hakkındaki karar KULLANICIDA.

Bu not, "çökelme 4 m sınırını aşıyor" soruşturmasının devamıdır
(bkz. `TERRAIN_DEPOSITION_MODEL.md`). Devralınan iz şuydu:

> Route aşamasındaki 4 m sınırı çalışıyor, fakat sonraki alluvial
> spreading/talus hareketi "toplam hücre yükselmesi" sözleşmesini aşıyor.
> Sınırı LEM'in yayılan nihai yüzeyine de uygulayalım.

İz **doğru ama eksik**, ve önerilen çözüm ölçüm yapılınca yanlış çıktı.

---

## ★★★ Önce alet: iki ölçüm arızası

Devralınan sayıya güvenmeden önce onu üreten yol denendi, ve iki bağımsız
arıza çıktı. İkisi de bu partide düzeltildi.

### 1. `terrain.erode` GPU yolunda istatistikler HİÇ dolmuyordu

`rtapi::erode` → `hydraulicErosionGPU(terrain, params)` çağırıyor; o imzada
`fields` **varsayılan olarak nullptr**. `hydraulicErosionGpuVulkan` içindeki
defter özeti `if (ok && fields && lemState.valid)` kapısının arkasındaydı, yani
script'ten sürülen her GPU erozyonu `summarize()`'ı **atlıyordu**.

Ölçülen:

```
terrain.erode backend=gpu → erosion_stats:  hepsi 0, gpu_path=false, cycle_iterations=0
terrain.erode backend=cpu → erosion_stats:  eroded=651, deepest_deposit=4.11 m  (dolu)
```

★★★ Sıfırlar hata gibi görünmez. `scripts/ipc/Probe-DepositShape.ps1` — tam da
bu çökelme işi için yazılmış A/B probe'u — `mean = 0` görünce oranı `NaN`
hesaplıyor, `NaN -ge NaN` yanlış dönüyor, ve script **yeşil** basıyor:
"[OK] Avulsiyon tek başına peak/mean oranını düşürdü." Hiçbir şey ölçmemiş bir
koşu başarı raporluyordu. Aynı sınıf: `project_physics_validation_suite`
(rig kendi hatasına düştü, 0==0 yeşil).

Düzeltme: özet, döngü koştuysa **her zaman** üretilir; `fields` artık yalnızca
uzamsal haritaların da geri verilip verilmeyeceğine karar verir. Kaynak
`fields` yokken iki geçici harita ayrılıyor (on bir değil). Probe'a da
"ölçüm yoksa FIRLAT" kapısı ve NaN dalı eklendi.

### 2. `nodes.set_property` float alanlara tam sayı kabul etmiyordu

`RtApiNodes.cpp::setNodeProperty` yalnızca `is_number_float + Kind::Float`
eşleşmesini kabul ediyordu. JSON'un tek sayı tipi var ve çoğu kodlayıcı tam
değeri `4` diye yazar — **PowerShell 5.1 her zaman**. Sonuç: deponun kendi IPC
istemcisinden hiçbir float kadranı `0`, `1` veya `4` yapılamıyordu — yani tam
olarak bir A/B'nin ihtiyaç duyduğu değerler (bir aşamayı **kapatmak** için 0).

Düzeltme: float hedefe Int değer genişletilerek yazılır. Tersi (int alana
float) hâlâ reddedilir — o sessizce budardı.

---

## Ölçüm: tek değişkenli merdiven

`scripts/ipc/Probe-BuildCap.ps1`. Sahne: `MountainRange → HydraulicErosion →
Height Output`, 512², 1 km, 100 m yükseklik ölçeği. Dropletler ve bütün
stabilizasyon post-process'leri kapalı, tek geçiş — geriye kalan tek yükselme
kaynağı LEM.

★★★ Ölçüm **node yolundan** alınır, `terrain.erode`'dan değil: node
`publishNetAggradation` çağırır, yani `deepest_deposit_meters` çıkış yüzeyi
eksi giriş yüzeyidir — *hücre ne kadar yükseldi*. `terrain.erode` aynı alanda
brüt taşıma defterini döndürür. **Aynı isim, iki büyüklük** — karşılaştırma.

| Koşu | Tepe (m) | Ort. (m) | Kaplama % | Göl % | Ana kol % |
|---|---|---|---|---|---|
| R0 yalnız route | **4.00** | 0.777 | 4.5 | 8.8 | 14.8 |
| R1 + creep | 4.00 | 0.770 | 4.5 | 8.7 | 14.8 |
| R2 + talus | **8.34** | 1.012 | 14.7 | 8.3 | 14.2 |
| R3 + alluvium | 9.72 | 0.950 | 16.9 | 8.0 | 13.8 |
| R4 + avulsiyon | 9.41 | 0.947 | 17.5 | 7.0 | 14.0 |
| R5 kadran 4→1 m | **8.14** | 0.784 | 17.7 | 7.7 | 14.3 |

Okunuşu:

- **R0 = 4.00 m, tam.** Route geçişinin kendi sınırı kusursuz tutuyor.
  Devralınan izin bu yarısı doğru.
- **Sıçrama talus'ta** (+4.34 m), alluviumda değil (+1.38 m). Devralınan not
  ikisini birlikte suçluyordu; ağırlık talus'ta.
- **★★★ R5 belirleyici satır.** Kadranı dörtte bire indirmek tepeyi 9.41 →
  8.14 m yapıyor. Sözleşme geçerli olsaydı tepe ~1 m civarına inerdi.
  **Kadran bu büyüklüğü yönetmiyor.**

## ★★★ Talus DOYMUYOR — bu bir gevşeme değil

Ayırt edici ikinci ölçüm: gerçek bir gevşeme, duraylı açıya inince durur;
adım eklemek yüksekliği büyütmez.

| `massWastingSteps` | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| Tepe (m) | 5.93 | 6.59 | 8.34 | 10.85 | 13.54 | 15.38 |

32 kata kadar hiçbir doyma yok — her ikilemede ~2.5 m. `fluvialIterations`
ekseni de aynı (8→6.66, 16→8.34, 32→10.88). Yani `massWastingSteps` bir
**yakınsama** kadranı değil, sınırsız bir **zaman adımı**; ortada duraylı bir
yüzey yok, sadece "ne kadar çok geçiş, o kadar çok taşınmış malzeme".

Deponun kendi tripwire'ının kardeşi: *iterasyonla BÜYÜYEN şey fizik değildir.*
Muhtemel mekanizma — talus bir **gather**: kaynak tarafındaki kararlılık sınırı
(`rate*0.5*maxExcess`) hücrenin kendi `maxExcess` komşusuyla sırasını
çevirmemesini garantiler, ama **varış tarafında hiçbir sınır yok**: bir
çukurluk aynı geçişte 8 komşudan birden alabilir.

## ★★ Ama arıza belirtisi YOK

Sınırın önlemek için yazıldığı şey kapalı çukurlar ve parçalanmış drenajdı
(ölçülmüştü: 38.5 m dolgu, haritanın dörtte biri kapalı havzada). Merdivende
tepe 4 → 15 m giderken **göl oranı 8.8 → 8.2, ana kol 14.8 → 13.8** — ikisi de
sabit. Yani bu yükselme, kadranın savunduğu arızayı **üretmiyor**.

Metre sayısı tek başına bir hüküm değildir.

---

## ★★★ AÇIK KARAR — kullanıcıya ait

`maxDepositionMeters` route geçişinin defterini bağlar; `TerrainManager.h` ve
`terrain_lem_route.comp` yorumu ("Bound TOTAL build-up, not only this pass")
ise toplam hücre yükselmesini vaat eder. İkisi aynı şey değil. İki dürüst
çıkış var:

**A. Adı davranışa uydur** (ucuz, hiçbir arazi değişmez).
`maxRouteDepositionMeters`, ve route'taki "TOTAL build-up" yorumu düzeltilir.
CLAUDE.md kural 5: sessizce anlam değiştirme, **alanın adı da değişmeli**.
Bedeli: proje dosyalarındaki serileştirilmiş alan adı göçü.

**B. Sözleşmeyi gerçek yap** — route + talus + alluvium için hücre başına tek
bir *monoton* yükselme bütçesi. Bedeli ağır: her arazinin görüntüsü değişir,
ve kütle korunumu için "yerleşemeyen malzeme kaynakta kalır" semantiği
gerekir (yüzeyi sonradan kırpmak **defterli olmayan kütle imhasıdır**).

★ Devralınan plan — *"sınırı LEM'in yayılan nihai yüzeyine uygula"* — B'nin
kırpma varyantıdır ve **önerilmez**: kütleyi deftersiz yok eder, ve ölçüm
gösteriyor ki savunduğu arıza ortada yok.

★ Ayrıca not: alluvium geçişi route'un bütçe tamponunu (`depositionField`)
**azaltıyor**, yani route bir kez yaydığı hücreye tekrar çökeltebiliyor.
Bu bir gözden kaçma değil, `terrain_lem_alluvium.comp` yorumunda **bilinçli**
("the total-build cap continues to describe where the material actually is").
Ama sonucu şu: sınır *anlık*, *toplam* değil — B seçilirse bu da çözülmeli.

## Açık kalan (bu partide dokunulmadı)

- `deepest_deposit_meters` iki yolda iki farklı büyüklük. Uzlaştırmak yayımlanan
  API alanının anlamını değiştirir; ayrı bir karar.
- Talus'un varış tarafı sınırsız (yukarıdaki doyma ölçümü). Bu, kadran kararı
  ne olursa olsun kendi başına bir soru.

[[TERRAIN_DEPOSITION_MODEL.md]] · [[TERRAIN_FLOW_AUTHORITY]]
