# Terrain erozyonu: fluvial döngü (LEM)

> **Durum:** REFERANS — kullanıcı Parti 6'yı canlı test etti ve kapattı (2026-08-23). Detaylı ölçüm sonucu kayıtlı değil; yeni bir arıza bulunursa buraya postmortem olarak eklenir, aksi halde bu not artık bağlayıcı mimari referanstır.

## Neden

Kullanıcının bildirdiği üç belirti aslında tek bir eksikliğin görünümüydü:

1. Erozyon **çok simetrik** çıkıyor; sertlik maskesi bunu ancak maskeliyor.
2. Dağın yükseklerinde **göller** kalıyor; heyelan ve taşma ile boşalmıyor.
3. Küçük dereler büyük çaylara/nehirlere birleşip **delta** üretmiyor.

Damla (droplet) çözücüsü Monte-Carlo bir rastgele yürüyüştür. Her damla aynı
suyla doğar, yalnızca **yerel eğimi** görür ve sabit bir ömürden sonra ölür.
Bundan üç yapısal sonuç çıkar ve hiçbiri parametre ayarıyla kaybolmaz:

| Eksik | Sonuç |
|---|---|
| **Drenaj alanı (A) geri beslemesi yok** | Bir vadi tabanı ile bir yamaç, *damla başına* aynı fiziği görür; aradaki fark yalnızca ziyaret sayısıyla **lineerdir**. Fluvial hiyerarşi `E ~ A^m S^n` (m≈0.5) süperlineer geri beslemesinden doğar. O olmayınca erozyon istatistiksel olarak **izotropiktir**. |
| **Depresyon işlemi yok** | Çukura giren damla orada ölür ve yükünü bırakır. Hiçbir şey taşmaz; dağdaki kapalı havza kalıcıdır. Gerçekte göl **geçici bir baz seviyesidir**: dolar, en alçak eşikten taşar, boşalım eşiği kazar, göl boşalır. |
| **Uzun menzilli sediment taşınımı yok** | ~64 adımlık damla ömrü 4K haritanın **%1.5'i**. Hiçbir tane sırttan kıyıya ulaşamaz. Delta tanımı gereği "bir havzanın yükünün duran suya varması" olduğundan, delta yalnızca yok değil, **erişilemezdi**. |

Buna ek olarak kütle hareketi (heyelan) döngünün *dışındaydı* — ayrı bir node,
önce veya sonra bir kez. Kanal kazar → duvar dikleşir → duvar çöker → moloz
kanala düşer geri beslemesi hiç kurulmadığı için vadi duvarları düzgün ve
simetrik kalıyordu, vadi başı geriye yürümüyordu.

## Ne yapıldı

`TerrainErosionLem.h` / `TerrainErosionLem.cpp` — bir Landscape Evolution Model
döngüsü. Damla çözücüsü **atılmadı**: mikro dokuyu (rill, yüzey karakteri) iyi
veriyor ve pipe modeline karşı görsel olarak zaten kazanmıştı. Eksik olan onun
*üstündeki* makro katmandı.

### Çalışma sırası

```
LEM ana döngüsü (fluvialIterations)
   → damla aşamaları (tek veya 3 pass)
      → LEM cilası (fluvialIterations/6, en az 2)
         → alan yayını (finalize)
```

LEM **önce** koşar ki damla yürüyüşü gürültüye değil, düzenlenmiş bir drenaj
ağına doku eklesin. **Sonra** tekrar koşar çünkü damlalar döngünün çözdüğü
çukurları ve dikenleri geri getirir; ciladan geçmeyen bir göl alanı damla
gürültüsünü göl diye raporlar.

### Bir iterasyonun geçişleri

| Geçiş | Shader | İş |
|---|---|---|
| Drenaj çözümü | `terrain_lem_restrict`, `terrain_lem_prolongate`, `terrain_flow_fill`, `terrain_lem_rain`, `terrain_lem_accumulate` | Depresyon doldurma + yağış ağırlıklı MFD alan birikimi. `drainageRefreshInterval` iterasyonda bir. |
| Kazma | `terrain_lem_incise` | `E = K·A^m·S^n`. Göl hücresi kazmaz; **eşiği kazar** — göl bu yüzden boşalır. |
| Taşınım | `terrain_lem_route` | `sedimentRouteSteps` hücre boyu advection + çökelme. |
| Kütle hareketi | `terrain_lem_talus` | Repose açısını aşan fazlayı gather biçiminde aşağı aktarır. |
| Sürünme | `terrain_lem_diffuse` | `D·∇²z`, kanalda kısılmış, gölde kapalı. |
| Yayın | `terrain_lem_finalize` | Discharge, kanal genişlik/derinlik, su seviyesi, akış yönü, göl derinliği. |

### Delta neden kendiliğinden çıkıyor

`terrain_lem_route` her hücrede yükün bir kesrini bırakır:

```
fdep = 1 - exp(-Vs · L / q),   q = A · rainRate / cellSize
```

- Dik dar kanalda `q` büyük → `fdep ≈ 0` → malzeme geçer.
- Geniş düz vadi tabanında `q` küçük → taşkın ovası / alüvyon yelpazesi.
- Duran suda `q ≈ 0` → `fdep = 1`, **yükün tamamı su hattında düşer**. Düştükçe
  taban yükselir, su hattı dışa kayar, ağız ilerler.

Delta *modellenmiyor*; progradasyon bu üç satırdan çıkıyor. Aynı sebeple göl de
kendiliğinden dolar.

## Sayısal sınırlar (bunlar süs değil)

Kullanıcının açıkça istediği yer burası. Bir manzara ile diken tarlası
arasındaki fark bu maddeler:

| Sınır | Nerede | Ne engelliyor |
|---|---|---|
| `incisionSafety` | incise | Bir hücre **kendi alıcısının altına asla kesilmez**. Alıcıyı geçerek kesmek, bir kazma şemasının yeni kapalı çukur *üretme* yoludur; o çukurlar sonra göl diye okunur ve bir sonraki drenaj çözümü onları tekrar doldurmak zorunda kalır. |
| `depositionSafety` | route | Çökelme, kendisini besleyen komşudan daha yüksek bir set/diken **inşa edemez**. Gölde su yüzeyinin üstüne çıkamaz. |
| `maxStepMeters` | incise, talus | Tek adımın mutlak tavanı (0 = yarım hücre). Tek bir patolojik eğim diken derinliğinde delik açamaz. |
| `slopeMin` / `slopeMax` | incise | `pow(S, n)` öncesi kelepçe. Kelepçesiz `S` dikey bir hücre çiftinde inf/NaN'a giden klasik yoldur. Bunlar **fiziksel taban değil**, `pow()` koruması. |
| Talus `rate·0.5·maxExcess` | talus | Bir hücre `maxExcess`'i tanımlayan komşusuyla sırasını **ters çeviremez**; salınım yok. |
| `D·dt/dx² ≤ 0.25` | diffuse | Host **alt adımlar**, sonucu kelepçelemez. Kararlılık sınırını aşan bir difüzyon şeması "biraz yanlış" görünmez; dama tahtası dikenlerine ıraksar. |
| Self-retention | route | Çıkış ağırlığı sıfır olan hücrenin yükü iki geçiş arasında **yok olmaz**. Bu olmadan tek belirti "geldiği dağdan küçük bir delta" olurdu. |
| Talus sınır hücresi | talus | Sınır hücresi dökmez ama **almalıdır**. Almayınca her geçişte kütle sızar ve tek belirti defterin kılpayı kapanmaması olur — yani "float gürültüsü" diye yazılıp geçilecek bir bulgu. |

## Ölçüm: sediment defteri

`HydraulicErosionStats` / `rt.terrain.erosion_stats`. Bu dosyadaki her arıza
sessizdir: taşınımdaki bir sızıntı, yakınsamamış bir doldurma, hiçbir şey
yapmayan bir talus geçişi — hepsi makul görünen bir heightfield üretir.

```
eroded == deposited + exported + carried
```

Defter **kapanmak zorundadır**: kütle hareketi eşit erozyon ve çökelme yazar,
sürünme ise bilerek defterin dışındadır (yüzeyi taşır, sediment üretmez).
Kalıntı ⇒ taşınım sızıntısı. Eşik %0.5.

Diğerleri: `lake_area_fraction` (eğimli arazide yüksekse doldurma
yakınsamamıştır), `drainage_density` ve `max_drainage_area_km2` (sıfıra
yakınsa hiyerarşi hiç oluşmamış).

## GPU / CPU sözleşmesi

Morfoloji geçişleri (kazma, taşınım, kütle hareketi, sürünme) iki yolda **aynı
açık şema ve aynı sınırlar**dır; yakın sonuç verir.

**Drenaj çözümü vermez, ve bu işaretlenmiştir:**

| | CPU | GPU |
|---|---|---|
| Depresyon doldurma | Priority-flood (**tam**) | Cascadic piramit + Planchon-Darboux gevşemesi (**yaklaşık**) |
| Alan birikimi | Topolojik tek süpürme (**tam**) | Cascadic piramit + Jacobi gevşemesi (**yaklaşık**) |
| Sediment yönlendirme | Aynı Jacobi, aynı geçiş sayısı | Aynı |

CPU referanstır, GPU onu yaklaşıklar. Yaklaşıklık reddedilmedi, **işaretlendi**:
yakınsamamış bir doldurma sahte göl, yakınsamamış bir birikim ise hak ettiğinden
zayıf ana nehir olarak görünür — ikisi de `erosion_stats`'ta okunur.

Piramidin iki tasarım kararı kritiktir:

- **Height için max-pooling.** Max-pooling dar bir geçidi ancak kapatabilir, hiç
  açamaz; dolayısıyla kaba doldurulmuş yüzey, incenin geçerli bir **üst
  sınırıdır**. Planchon-Darboux yalnızca azaldığı için bu tohumdan doğru sonuca
  yakınsar. Ortalama ile tohumlamak (bir sırtın altına düşebilir) doldurmayı
  kalıcı olarak eksik bırakırdı — belirtisi yalnızca hiç boşalmayan çukurlar.
- **Ağırlık tablosu yok.** Hücre başına 8 float, 4K'da yarım GB'ın üstü. Geçiş
  zaten bellek-bağımlı, tablo yalnızca küçük kartlarda tahsis hatası satın alır.

## Geriye uyum

`macroDrainage` ile `fluvialCycle` **birbirini dışlar**. Makro aşama tek seferlik
kaba bir havza çözümünden sabit derinlikte vadi oyar ve **oyduğu vadi, onu oyan
akışı hiç değiştirmez** — bir dekorasyondur, geri besleme değil. İkisini birden
koşturmak sonucu atfedilemez hale getirirdi. `fluvialCycle` açıkken makro aşama,
droplet discharge yönlendirmesi ve channel maturation atlanır.

`macroDrainage` ve CUDA erozyon yolu **DEPRECATED** işaretlendi (bkz.
`TerrainManager.h` ve `erosion_kernels.cu` başlığındaki kaldırma planı). LEM
gerçek projelerde doğrulandıktan sonra sökülecekler.

## 1. tur geri bildirimi ve düzeltmeleri (2026-08-23)

İlk derleme sorunsuz çalıştı: **diken veya hatalı çukur üretmedi** ve sonuç eski
yapıdan daha doğal. İki gerçek sorun çıktı.

### A. Maliyet

⚠ **Atıf ÖLÇÜLMEDİ.** İlk bildirim "yeni yapı aşırı maliyet üretti" idi;
sonradan kullanıcı maliyetin ayrı `Fluvial` node'undan geldiğini, onun zaten
pahalı bir node olduğunu bildirdi. Yani aşağıdaki iyileştirme **gerçek bir
israfı kaldırıyor ama gözlenen yavaşlığın sebebi olduğu kanıtlanmadı.** Bu
depoda tam da bu sıra yanlış yapılıyor: önce ölç, teşhis sonra. Ölçüm borcu
`NEXT_BUILD_CHECKS.md` §2'de duruyor — `lastSolveMs`, döngü açık/kapalı.

İyileştirmenin kendisi atıftan bağımsız olarak doğru:
`accumulate` ve `route` geçişleri MFD ağırlıklarını **uçuşta** hesaplıyordu:
bir komşunun ağırlık satırı için o komşunun *kendi* sekiz komşusunu okumak
gerekir ⇒ hücre başına 64–72 dağınık okuma, ve bu geçişler binlerce kez koşuyor.
Yalnız routing tek başına 1K haritada ~166 **milyar** okuma demekti.

Tabloyu ilk turda reddetme gerekçem doğruydu ama çözümü eksikti: **float** tablo
hücre başına 32 bayt, 4K'da yarım GB'ın üzeri. Çözüm tabloyu yön başına **bir
bayta** paketlemek — hücre başına 8 bayt, 4K'da 134 MB, 1K'da 8 MB
([terrain_lem_weights.comp](../../RayTrophiStudio/source/shaders/terrain_lem_weights.comp)).
Ağırlıklar yalnızca akışın komşular arasında nasıl *dağıldığına* karar veriyor;
bayt çözünürlüğü orada hiçbir şey kaybettirmiyor.

★★ **Kuantizasyon "yaklaşık" değil, TAM olmak zorunda.** `route`, kendinde
tutma oranını `1 - sum(ağırlıklar)` olarak türetiyor. Baytlar 255 yerine 253'e
toplansaydı her hücre geçiş başına **%0.8 yükünü sessizce alıkoyardı**; yüz
geçişte bu sedimentin çoğu eder. Bu yüzden yuvarlama artığı en büyük ağırlığa
katlanıyor: çıkışı olan hücrede bayt toplamı **tam 255**, terminal çukurda
**tam 0**.

Yanında iki şey daha:

- **Piramitte geçiş bütçesi seviyeye göre yarılanıyor.** Her ince seviye bir
  üstünden başlıyor, yani yalnız ince ölçekli düzeltme yapması gerekiyor — ama
  geçiş başına 4× pahalı. Düz bütçe, solve'un çoğunu en az ihtiyacı olan
  seviyede harcıyordu.
- **CPU'da yayın yapılmayan turdaki son drenaj çözümü kaldırıldı** (tam
  priority-flood + topolojik süpürme, tamamen boşa gidiyordu).
- Varsayılanlar: `fluvialIterations` 24 → 16, `drainageRefreshInterval` 4 → 6.

### B. ★★★ Vulkan TDR — gönderim sayıyla sınırlandı, İŞLE değil

Hazır kurulumlarda (satmap/river/snow) Vulkan TDR ile çöküyordu; son log satırı
render tarafındaydı, yani suçlu görünen yer **arıza yeri değildi**.

`Batch` her 384 **dispatch**'te bir `synchronize()` ediyordu. Bu descriptor
havuzu (512 set) için doğru bir sınır ama **iş için hiçbir sınır değil**: 4K
ızgarada 384 geçiş, tek submit'te milyarlarca hücre güncellemesi — saniyelerce
GPU, hem render'ı aç bırakır hem Windows watchdog'unu tetikler.

Damla çözücüsü bu dersi **zaten öğrenmişti** ve kodun hemen üstünde yazıyordu
("A fixed 256K batch monopolized the compute queue... made viewport/render work
appear hung"); ben aynı dosyada okuyup uygulamamışım. Artık `Batch` iki bağımsız
sınıra birden bakıyor: dispatch sayısı (descriptor havuzu) **ve** hücre
güncellemesi (watchdog). Talus geçişi 8× ağırlıkla sayılıyor, çünkü gather biçimi
komşunun dağılımını yeniden hesaplıyor.

★ Genel kural: **aynı geçiş sayısı 1K'da ucuz, 4K'da öldürücüdür.** Bir submit
sınırı çözünürlükten bağımsız olamaz.

### C. Panel

Node iki çözücülük kadran biriktirmişti ve okunmaz hale gelmişti — okunmaz bir
panel kozmetik sorun değil, yanlış varsayılanın saklandığı yerdir. Yeni düzen:

- **Üst seviye** yalnız sonucu değiştirenler: Use GPU, Fluvial Cycle, Quality,
  Strength, Incision, Landslides + Repose. Altı kadran.
- **Quality** (Draft/Balanced/High/Custom) yalnız **maliyet** kadranlarını
  belirler (iterasyon, gevşeme bütçeleri, taşınım adımı, piramit derinliği) —
  şekil kadranlarına asla dokunmaz. Yani kalite değiştirmek *aynı manzarayı*
  daha yakınsak yapar, farklı bir manzara üretmez. Bütçelerden birini elle
  değiştirmek Quality'yi otomatik `Custom`'a düşürür (`detectFluvialQuality`,
  serileştirilen bir alan yok — panel her kare gerçek değerlerden türetiyor).
- Damla kadranlarının **tamamı** "Droplet Detail" altında, çünkü ağı artık
  döngü kuruyor; damlalar yüzey dokusu ekliyor.
- Eski `Channel Evolution` + `Macro Valleys` "Legacy Channel Stages" altında ve
  döngü açıkken **devre dışı** görünüyor.

Hiçbir şey silinmedi: her değer bir açılır katman altında duruyor ve hepsi hâlâ
script'ten erişilebilir (`fluvial_quality` dahil).

## Karşılaştırma: elle kurulan zincir

Geniş dereler bugüne kadar **elle** üretilebiliyordu:

```
Hydraulic (varsayılan)  →  Fluvial (flow inertia ~0.4)  →  Hydraulic (varsayılan)
```

Yani: bir kaba geçiş, sonra kanalları açan bir stream-power geçişi, sonra yeni
yamaçlara ince dereler ekleyen bir geçiş daha. Bu zincirin **yaptığı iş doğru**;
sorunu üç ayrı yerdeydi:

- Ortadaki `Fluvial` node'u **detachment-limited**: yalnızca kazır, sediment
  taşımaz, çökeltmez. Kanal açar ama delta veya alüvyon üretemez.
- Aradaki hiçbir adım depresyonu koşullandırmaz, dolayısıyla göller yine kalır.
- Zincirin **kendisi manuel bir adımdır**: hangi node, hangi sırada, hangi
  inertia — bu bilgi grafikte değil, kullanıcının kafasında. Otomatik bir testin
  ortasındaki insan adımı budur.

LEM döngüsü aynı katmanlamayı tek node içinde ve **geri beslemeli** yapar:
LEM ana döngüsü (kaba ağ + göl boşaltma + taşınım) → damla aşamaları (doku) →
LEM cilası (damlaların açtığı çukurları kapatır, hidrolojiyi son yüzeyden yayar).

**Eski zincir bozulmadı.** `Fluvial` node'una dokunulmadı; Hydraulic node'unda
`fluvialCycle` kapatıldığında davranış birebir eskisidir. Karşılaştırmak için:

```python
rt.terrain.erode("Land", "hydraulic", "gpu", fluvial_cycle=0)   # eski zincirin 1. halkası
rt.terrain.erode("Land", "fluvial",   "gpu")                    # 2. halka, değişmedi
rt.terrain.erode("Land", "hydraulic", "gpu", fluvial_cycle=0)   # 3. halka
```

⚠ **Varsayılan artık AÇIK.** Kaydedilmiş bir projede `fluvialCycle` anahtarı
yoktur, dolayısıyla varsayılanı alır ve döngü çalışır — yani mevcut üç node'lu
zincirlerin *her Hydraulic halkası* birer LEM döngüsü koşar. Bu bilinçli: eski
davranış hatanın kendisiydi. Eski sonucu birebir istiyorsan yukarıdaki gibi
`fluvial_cycle=0` yaz (ya da panelden Enable'ı kapat).

## River-to-lake transfer contract

An accepted Lake Basin is a hydrological super-node. River Network keeps the
visible channel outside the lake footprint, adds one shoreline overlap sample,
and transfers stream hierarchy from inlet to the dry spill/outlet. The easy
River Network setup wires both `Lake Mask` and `Lake Spill Points`; a flat
vector direction inside standing water is therefore not allowed to turn the
outlet into a new headwater. `terrain.flow_authority` reports both connections
for script and IPC diagnostics.

Depressions below Lake Basin's acceptance thresholds remain ordinary drainage
cells. LEM sediment routing does not treat the first depth epsilon as a full
settling basin: capture blends from the normal through-flow settling fraction
to complete capture over roughly one physical metre. Small hollows can retain
some load, fill over drainage refreshes, and continue passing sediment; an
established lake still captures the load needed for shoreline progradation.

## Gross transport versus visible alluvium

The conservative erosion/deposition counters are a mass ledger. A talus or
alluvium pass may book the same parcel as leaving one cell and entering the
next many times, so gross deposition is useful for closure but is not terrain
thickness. Hydraulic Erosion keeps the gross `eroded`, `deposited`, `exported`,
and `carried` totals for mass balance, while its public `Deposition` image and
the deposited-area/depth diagnostics publish net positive height change from
the node input. Soil Depth and material masks consume this net aggradation
field; otherwise repeated local transport can incorrectly paint most of the
map as deep alluvium.

## Bilerek kapsam dışı

- **Regolit/toprak katmanı.** Heyelan molozu şu an gevşek malzeme olarak
  işaretlenmiyor; sertlik haritası direnç vekili olarak kullanılıyor. Ayrı bir
  `regolith` alanı (talus ve çökelmenin beslediği, kazımanın öncelikle
  tükettiği) yatak kayası/toprak ayrımını verir ve sıradaki adaydır.
- **Deniz seviyesi baz seviyesi olarak.** Şu an tek baz seviyesi harita kenarı
  ve göl yüzeyleri. `ErosionBoundaryMode::SeaLevel` yalnızca kenar harmanlaması;
  çözücüye girdi olan bir deniz kotu, kıyı deltalarını göl deltaları kadar
  doğru yapar.
- **Katmanlı litoloji (strata).** Sertlik hâlâ tek katmanlı skaler.
