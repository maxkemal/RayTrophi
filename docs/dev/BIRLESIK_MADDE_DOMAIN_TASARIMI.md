# Birleşik madde domain'i — gaz/sıvı ayrımı olmayan simülasyon ve render

> **Durum: 2026-10-05.** C5 kontrollü su bilançosu canlıda korundu. C6 ilk
> kaynak partisi kullanıcı derlemesiyle canlı probları geçti; tam C6/C7 kabulü açık.
> Hedef tek Matter domaininde madde durumundan türeyen gaz/sıvı/granül çözümü
> ve fiziksel alanlarla tutarlı görünüm. Ampirik local-head modeli tam basınç
> çözümü olarak sunulmaz.

---

## Güncel kısa devir — 2026-10-05

| Konu | Durum |
|---|---|
| C5 canlı ölçüm | Sonlu Water/Sand sahnesi; emisyon sonrası su farkı -0.71 mg / 180 adım, kuru Sand sabit. Drenaj/contact/kimlik probları PASS; tüm yakınsama matrisi kapanmadı. |
| Yeni kaynak partisi | C6 wet/drainage/mixed GPU probları PASS; emisyon sonrası 180 adımda su +0.32 mg, kuru Sand sabit. 8-band wet materyaller üretildi; tam fiziksel kabul açık. |
| Sonraki iş | C6 dry/damp/saturated ve spatial/cache/dt/resolution kabulü, ardından C7. Sağ başlangıç kuru kontrolüne uzun koşuda su ulaşır. Tam pressure PDE/drag/buoyancy açık. |
| Kanonik not | [MATTER_C5_HANDOFF.md](MATTER_C5_HANDOFF.md), model: [MATTER_WET_RESPONSE.md](MATTER_WET_RESPONSE.md). |
| Kapanış ve sonrası | [Kalan fizik kabulü ve profile/emitter/domain/UI veri sahipliği](MATTER_ACCEPTANCE_AND_AUTHORING.md). Yeni spatial kabul ölçümleri kaynakta hazır; kullanıcı derlemesi bekler. |
| H1 parti sırası | [MATTER_H1_GRAIN_ROADMAP.md](MATTER_H1_GRAIN_ROADMAP.md) (2026-10-06): B2 kova/repose + B3 taşıyıcı başına sahiplik ve su–tane sürükleme/kaldırma kaynakta, tek build bekliyor; sonra B4 GPU yerleşikliği (H1-C7), B5 hacim dışlama, B6 ıslak tane, B7 DEM–XPBD kararı. |
| Granül önceliği | G2 MPM yerleşme/dt açık. H1 CPU tane referansı 21/21 canlı PASS; production emitter/collider/GPU bağlantısı yok. Öncelik domain içi GPU granül çözümü; DEM ile PBD/XPBD seçimi ölçümle, otomatik hibrit geçiş sonraki kapı. Aşağıdaki H1 üretim güncellemesi kanonik yön; [referans ve sınırlar](MATTER_H1_GRAIN_CONTACT.md). |
| Tarihsel devir | [Önceki kayıt](archive/MATTER_ROADMAP_HANDOFF_2026-10-05_HISTORY.md). Ayrıntılı tasarım ve geçmiş ölçümler aşağıda korunur. |

---

## H1 üretim hedefi — ortak GPU madde, tek domain (2026-10-05)

**Bu bölüm, aşağıdaki 2026-10-04 H1 tasarımının uygulama önceliğini günceller.**
Ana hedef değişmez: tek Matter domaininde birbirini etkileyen gaz, sıvı, granül
ve deformasyon; yerel ıslanma/kuruma, yanma ve erime. Domain sınır/zaman/collider/
bütçe/cache yöneticisidir; tek domain, tek fizik algoritması anlamına gelmez.
Mevcut particle altyapısı yeniden çoğaltılmaz. Emitter aynı fiziksel maddeyi ve
başlangıç durumunu üretir; granül çözücü bu ortak taşıyıcıları ilerletir.

### Veri ve GPU sahipliği

- Kanonik GPU SoA/phase grid durumu üretim çözümünün ana durumudur. Kararlı
  kimlik, substance/model/sahiplik, konum-hız-kütle, sıcaklık/enerji ve pore
  water/capacity yerel olarak korunur. Granül temas geçmişi, bağ/hasar,
  radius/inertia/angular state ilgili modelin gerekli sidecar'larıdır.
- Ortak veri, bütün fazları tek hız/density alanında ortalamak değildir. Model
  başına gerekli grid/scratch ve farklı faz temsilleri korunur; aynı kütlenin
  veya enerjinin ikinci otoriter kopyası oluşturulmaz. Buffer yaşam döngüsü,
  revision ve okuyucu/yazıcı aşamaları tek scheduler tarafından yönetilir.
- Solver → solver ve solver → render normal üretim yolu GPU buffer/aktif listeleri
  üzerinden ilerler. CPU'ya bütün konum/hız/state indirip sonraki çözücü için
  yeniden upload etmek normal taşıma yolu olmaz. Barriers ve dispatch sırası
  açık sözleşmedir; mümkün olan occupancy/hash/stencil/aktif metadata paylaşılır.
- CPU readback yalnız açık sorgu, küçük/asenkron kabul sayaçları, gereken cache/
  save kaydı veya debug referansı içindir. Cache kayıt maliyeti ayrı ölçülür;
  GPU'da kalmak tek başına hız garantisi sayılmaz.
- Aynı domain/grupta her taşıyıcının su, sıcaklık, hasar ve bağ durumu farklı
  olabilir. Akışkanlarda da heterojen durum mümkündür. Domain ortalaması yerel
  kuvvet/bağ/kopma hesabının yerine geçmez. Saturation su/capacity'den türetilir.
- Islanma/kuruma mevcut modelin sürtünme/bağ/dayanımını değiştirebilir; sırf
  saturation değişti diye otomatik model geçişi gerekmez. Yanma/erime/evaporation
  ve faz transferleri dry/free/pore/gas mass, enerji ve momentum ledger'ıyla
  kapanır; tüketilen kaynak iki modele birden yazılmaz.

### Granül çözücü kararı ve sınır

CPU kuvvet-tork tabanlı DEM çekirdeği doğrulama referansıdır; ayrı bir ürün
particle sistemi veya kalıcı CPU sahne solver'ı değildir. Yüksek yoğunluklu
üretim taneleri için force-based DEM ve PBD/XPBD contact adayları; paketleme,
yerleşme, dt/iteration hassasiyeti, runout, temas/rolling ve GPU maliyetiyle
karşılaştırılır. PBD/XPBD üretim için henüz seçilmiş/uygulanmış değildir; görsel
benzerlik fiziksel eşdeğerlik veya açısal momentum kabulü yerine geçmez.
PBD nokta temasının kendiliğinden fiziksel spin/rolling sağladığı varsayılmaz;
gerekli açısal state ve torque/constraint karşılığı ayrıca tasarlanıp ölçülür.
Sürtünme sönümü enerji yok etmek diye saklanmaz; disipasyon/ısı aktarım politikası
ve ölçülen sayısal enerji farkı termal bilanço ile ayrı raporlanır.

Önce ortak Matter runtime içinde açık bir granül taşıma yolu kurulur. Her
parçacığın bir alt adımda tek transport owner'ı vardır; model/solver seçimi
render materyalinden veya virtual sphere sayısından türemez. MPM kuru kumu da
çözebilir; G2 dt/yerleşme RED kapanmadan MPM yanlış yöntem ilan edilmez.
Render çözünürlüğü ile fizik tane/taşıyıcı çözünürlüğü açıkça ayrılır.

2026-10-04 tasarımındaki yüzey tanesi seçimi ve MPM↔DEM otomatik geçişi ikinci
bir hibrit kapıdır, ilk yüksek yoğunluklu granül paketinin ön şartı değildir.
Eski `transport_owner=mpm|dem` sözleşmesi üretimde `mpm|grain` sahipliğine ve
ayrı `grain_solver_kind` seçimine genellenir; bunlar hedef şema, henüz kodda
uygulanmış alanlar değildir. Sonradan geçiş uygulanırsa aşağıdaki tek kimlik,
çift impuls, orbital/spin momentum, hysteresis ve cache koşulları aynen geçerlidir.

### Uygulama ve kabul sırası

2026-10-05 kaynak: [dry grain GPU runtime adayı](MATTER_GRAIN_GPU_RUNTIME.md)
yazıldı; tek kullanıcı C++/shader build bekliyor. Hash/contact/spin, gerçek emitter
ve flat collider bağlantısı var; GPU sayısal kabul, yoğunluk kıyası ve üretim solver
kararı henüz açık. Statik friction history/wet/thermal coupling ve frame sonu
host render köprüsünün kaldırılması bu ilk adayda tamamlanmadı.

| Kapı | Teslim | Kabul / kapanış |
|---|---|---|
| H1-R | Mevcut bounded CPU sphere/contact/integratör referansı | 21/21 canlı PASS; kısa .4 s / 27 tane dt farkı .285 mm. Grid karşı-impuls, production collider ve GPU parity henüz yok; eski H1a bütünü tamamlanmadı. |
| H1-G0 | Ortak GPU taşıyıcı/aktif liste/komşuluk/lifecycle sahipliği; DEM–PBD/XPBD aday kararı | Kimlik/mass/local state korunur, çift sahipliği deterministik; yoğunluk ve bütçe artınca görünüm fizik kimliğini değiştirmez. |
| H1-G1 | Matter emitter → GPU granül step → mevcut flat TriangleMesh collider → GPU render; ortak scheduler | Kuru dökülme, rampaya çarpma/sekme/kayma, yüksek yoğunluklu yığın, iki yük yüksekliği; dt ve tane çözünürlüğü yakınsaması, repose/runout. Sahne gerçek runtime'da çalışır; IPC snapshot replay bu kapıyı kapatmaz. |
| H1-G2 | Aynı state üzerinde yerel wet/thermal/bond/damage ve sıvı–tane karşılıklı contact | Aynı grupta dry/damp/saturated alanlar ayrılır; kapalı kütle/enerji/momentum, kuruma/yanma/erime mevcut ledger ile bağlanır. Her dönüşüm kendi kabul matrisiyle kapanır. |
| H1-P | Production servis/authoring, cache/serializer ve UI/Python/IPC parity | Yeni operations aynı core; strict validation/transactional error, save/load/bake/scrub/resume aynı kimlik ve sidecar durumunu kurar. Authoring capability ilgili fizik paketiyle birlikte gelir; UI-only veya binding-only yol yok. |
| H1-H | Gerekli olduğu kanıtlanırsa MPM↔grain hibrit geçiş ve yerel kırılan bağlar | Önce/sonra mass, linear/angular momentum ve sayısal enerji; bütçe/hysteresis; kar/çığ, toprak/heyelan için yerel bağ kopması. Otomatik geçiş ilk paket için zorunlu değildir. |
| H1-C7 | Saf su/saf kum/karma GPU resident maliyet ve regresyon | Dispatch, GPU süre/bellek, CPU sync ve upload/download byte sayıları; normal solver aşamalarında tam-state CPU roundtrip yok. Cache/query maliyetleri ayrı; hız yüzdesi ölçümsüz ilan edilmez. |

H1-P, H1-G1/G2 ile birlikte teslim edilen ilgili operasyonların şartıdır; genel
editable profile/emitter/contextual UI göçü kendi sonraki yol haritasını korur.
Bu liste eski açık C5/C6/C7 veya G2 kabulünü tamamlandıya çevirmez. Mevcut
27-grain görsel tekrar CPU referansın sampled pozlarını oynatır; üretim GPU
state paylaşımı, emitter/collider bağlantısı veya timeline DEM kanıtı değildir.

---

## Tek cümle

> **Neyin nasıl çözüleceğine ve çizileceğine maddenin fiziksel durumu karar
> verir; kullanıcının seçtiği "domain tipi" ve "render modu" karar vermez.**

Bugün iki seçim var ve ikisi de fizikten önce geliyor: domain tipi (Gas /
Fluid) ve sıvının tek render modu (Splat / SDF / Fog). Gerçek bir şelalede
aynı su aynı anda gövde, sprey ve sistir; yanan bir sıvı aynı anda sıvı ve
gazdır. Tek seçim bunu anlatamaz.

---

## 1. Neden şimdi — bu oturumun kanıtları

2026-09-27'deki fog/splat partilerinin hataları aynı kökten çıktı: **"ne
çiziliyor" sorusu dört yerde ayrı ayrı cevaplanıyor.**

| Hata | Kök |
|---|---|
| VDB panelinde Fog seçimi kare değişince geri döndü | Panel türetilmiş hacme yazdı, köprü domain'den geri yazdı |
| SurfaceSDF override'ı sisi sessizce yuttu | Domain başına TEK hacim; iki görünüm aynı slotu istiyor |
| Sıvı yoğunluk sayacı hep 0 → fog hiç çizilmedi | Analiz geçişi "sıvı yoğunluğa dokunmaz" varsayımıyla gaz-only |
| Sıvıda blackbody "parlıyordu" | Sıcaklık kanalı yoktu; shader yoğunluğu sıcaklık sandı |

Rota `scene_data.h`'de, proxy kararı `ParticleRenderBridge`'de, panel metni
UI'da, `effective_representation` API'de hesaplanıyor. Birleşik model bu
dağınıklığı büyütmek yerine ortadan kaldırmalı.

---

## 2. Bugün elimizde olan (koddan okundu, 2026-09-27)

- **Ortak konteyner zaten var.** Gaz ve sıvı domain'i aynı
  `SimulationGridDomainState`'i ve aynı `grid`'i kullanıyor (hız yüzleri,
  `density`, `temperature`, `fuel`, `interaction`, `solid`). Ayrılan şey
  hangi alanı kimin doldurduğu. ⚠ Ve **ne anlama geldiği**: sıvıda `density`
  hacim oranıdır, gaz kanalları hiç ayrılmaz, hız alanı APIC'in çalışma
  alanıdır (§8b 1. tur, madde 2).
- **İki çözücü de grid üzerinde basınç çözüyor.** APIC sıvısı parçacıktır ama
  P2G → basınç → G2P ile aynı tür MAC grid'ini kullanır; gaz Euler grid'idir.
- **Render katmanlaması zaten var** —
  [VULKAN_GAS_FLUID_LAYERING.md](VULKAN_GAS_FLUID_LAYERING.md) (REFERANS):
  gaz/fog ve SurfaceSDF **ayrı hacimler, ayrı TLAS maske bitleri**; gaz ışını
  gerçek `density = 0.5` geçişinde SDF'ye devreder. Çakışık kutu arızası
  [VOLUME_BOX_REENTRY_POSTMORTEM.md](VOLUME_BOX_REENTRY_POSTMORTEM.md)'de
  çözüldü. **Yani aynı hacimde gaz + sıvı yüzeyi çizmek çözülmüş bir problem.**
- **Sıvının üç görünümü tek parçacık verisinden:** splat havuzu, SDF, fog
  (fog 2026-09-27'de gerçek yoğunluk + Gauss yayma + parçacık Kelvin'i ile).
- **Köpük/whitewater** ayrı bir parçacık kümesi; Ihmsen kriterleriyle (sprey /
  köpük / kabarcık) üretiliyor. Volume modunda SDF hacminin sıcaklık kanalına
  biniyor — pratikte yalnız RT'de görünüyor.
- **Faz geçişlerinin ilk örnekleri çalışıyor:** yanan sıvı gaza dönüşüyor
  (yanma → gaz yoğunluğu/ısı), termal sıvı zinciri donuyor
  (`kParticleFlagFrozen`, katı maskesine damgalanıyor).
- **Dış üreticiler:** Flow Source'lar (sıcaklık override'ı dahil), sahne
  objesi emitter'ları, partikül sisteminin gaz emisyonu (
  [PARTICLE_SYSTEM_GPU_ROADMAP.md](PARTICLE_SYSTEM_GPU_ROADMAP.md): `Domain
  Deposit` çıktısı).
- **Yüzey durumu havuzu:** MSF (Material State Field) — char, sıcaklık, nem,
  kütle kaybı; RT'de yanma/char maskesi olarak okunuyor. Partikül yol haritası
  buna `Surface State Deposit` ile yazıyor.

---

## 3. İlkeler (bağlayıcı)

1. **Fizik karar verir, render okur.** Bir parçacığın/hücrenin ne olduğu
   (gövde, sprey, sis, köpük, donmuş) SİMÜLASYONDA etiketlenir. Render, UI ve
   API bu etiketi okur; kendi sınıflandırmasını yapmaz.
2. **Tek karar noktası.** Etiket → görünüm eşlemesi tek fonksiyondur
   (`resolveDomainRepresentations` gibi). Köprü, panelin "Now drawing" satırı
   ve `fluid.get` aynı sonucu gösterir — ayrı hesaplarsa panel yalan söyler.
3. **★★★ Kimlik ve depolama birleşir, DEĞERLENDİRME birleşmez.**
   [SIMULATION_NODE_CONCEPTUAL_MODEL.md](SIMULATION_NODE_CONCEPTUAL_MODEL.md)
   §2.3 burada da geçerli. "Tek domain" **tek çözücü değildir**: sıvı fazı
   APIC ile, gaz fazı Euler ile çözülmeye devam eder; birleşen şey konteyner,
   grid, UI ve fazlar arası alışveriştir.
4. **Üreticiler domain'e yazar, domain'in sahibi olmaz.** Partikül sistemi,
   Flow Source ve sahne emitter'ları `Domain Deposit` sözleşmesiyle madde
   ekler (hangi faza, hangi sıcaklıkta). Partikül yol haritasının açık
   non-goal'ü korunur: domain çözücüleri partikül sistemine taşınmaz.
5. **Domain yüzeylere yazar, yüzeyin sahibi olmaz.** Sıcak sıvının bıraktığı
   char, ıslaklık, is: `Surface State Deposit` → MSF. RT char maskesi MSF'den
   okunmaya devam eder.
6. **Her etiket ve alışveriş ölçülebilir.** Etiket başına parçacık sayısı,
   fazlar arası aktarılan kütle/ısı, görünüm başına hacim/instance sayısı IPC'den
   okunur. Kütle korunumu bir sayaçtır, bir umut değil.
7. **Vulkan birincil, OptiX dondurulmuş.** Yeni görünümler/katmanlar yalnız
   Vulkan yolunda eklenir; OptiX mevcut davranışını korur.
8. **Eski sahne anlamı sessizce değişmez.** Gas/Fluid domain'i yüklenince
   ilgili fazı taşıyan birleşik domain'e göçer; alan adları değişir
   (CLAUDE.md §5).

---

## 4. Model

### 4.1 Domain = fazların konteyneri

```
Domain
  grid (seyrek, ortak MAC)       ← gaz fazı + sıvının basınç/transfer grid'i
  liquid phase  (APIC parçacıkları, etiketli)
  gas phase     (grid alanları: density/temperature/fuel/flame)
  solid phase   (donmuş parçacıklar + collider'lar → grid.solid)
  exchange      (fazlar arası kütle / ısı / momentum terimleri)
```

### 4.2 Sıvı parçacık etiketleri

Etiketi simülasyon her adımda yazar; kriterler fizikseldir, render ayarı değil.

| Etiket | Kriter (ilk öneri) | Varsayılan görünüm |
|---|---|---|
| `body` | Yüksek komşu sayısı / level-set içi | SDF yüzeyi |
| `spray` | Düşük komşu sayısı, gövdeden kopmuş, balistik | Splat |
| `foam` | Yüzeye yakın, düşük hız, hava karışmış (Ihmsen) | Splat veya ince fog, HER modda |
| `bubble` | Gövde içinde, hava cebi (Ihmsen) | Splat (kırılmalı) |
| `mist` | Çok küçük kütle, havayla taşınan | Fog — ve Faz 3'te gaz fazına aktarılır |
| `frozen` | `kParticleFlagFrozen` | SDF'de ikinci materyal (buz/mum), fog'dan hariç |

Köpük sistemi bu tabloya **katlanır**: ayrı parçacık kümesi değil, etiket olur.
"Köpük yalnız RT'de görünüyor" sorunu böylece kendiliğinden kapanır.

### 4.3 Görünüm çözücü ve görünüm başına kaynak

Domain başına tek hacim slotu **kalkar**. Çözücü her domain için bir liste
döndürür; her girdi kendi kaynağını taşır:

| Görünüm | Kaynak | TLAS sınıfı |
|---|---|---|
| Yüzey | SDF hacmi (`body` + `frozen`, madde materyalleri) | SurfaceSDF maskesi |
| Sprey/köpük | Splat instance havuzu | geometri |
| Sis/gaz | Fog hacmi (`mist` yoğunluğu + gaz fazı alanları) | gaz/fog maskesi |

Katmanlama sözleşmesi zaten "gaz/fog ile SurfaceSDF ayrı hacim" varsayıyor;
bu model o sözleşmenin üzerine oturur, yeni bir render problemi açmaz.

★ Bugünkü `FluidRenderMode` (tek seçim) bu çözücünün **varsayılanı** olur:
kullanıcı bir etiketin görünümünü değiştirebilir (ör. `spray` → gizli), ama
seçim etiket başınadır, domain başına değil.

### 4.4 Fazlar arası alışveriş

| Yön | Terim | Bugün |
|---|---|---|
| sıvı → gaz | buharlaşma, yanma ürünü, `mist` aktarımı | yanma ve mist→gaz 27C canlı PASS |
| gaz → sıvı | rüzgar/gaz hızıyla sürükleme (sprey, mist) | rüzgar + düşük kütleli mist gaz sürüklemesi var |
| sıvı → gaz | sıvı hücreleri gaz için hareketli sınır | 27D kodda; canlı kabul bekliyor |
| ısı | sıvı ↔ gaz ↔ katı iletim | termal zincir kısmen |
| sıvı ↔ katı | donma / erime | var (donma), erime MSF'de |
| dış → domain | `Domain Deposit` (partikül sistemi, Flow Source, sahne emitter'ı) | kısmen |
| domain → yüzey | `Surface State Deposit` → MSF (char, ıslaklık, is) | MSF var, sıvıdan yazım yok |

### 4.5 ★★★ Madde kimyası: tek özellik tablosu, tek dönüşüm defteri

Hedef bir **fizik-kimya sistemi**: granül, sıvı, gaz ve yanan katı, maddenin
kendi özelliklerine göre erir, donar, buharlaşır, yanar, kütle kaybeder — ve
bunu yaparken çözücüler birbirini besler. Dönüşümlerin çoğunun ilk hâli
bugün var; eksik olan onları **birbirinden haberdar eden** ortak katman.

**Bugünkü durum (koddan):** her çözücü madde özelliğini kendi tutuyor.
`MoltenMassTransfer.cpp` erimiş maddenin viskozitesini adında `"Plastic"`,
`"Wax"`, `"Iron"` arayarak seçiyor; MSF'nin kendi `char_rate` / erime
parametreleri, termal sıvı zincirinin kendi `thermal_freeze_kelvin`'i, gazın
kendi yanma ısısı var. Aynı madde (ör. mum) iki çözücüde iki ayrı sayıyla
tanımlı olabilir ve bunu hiçbir şey fark etmez.

**Önerilen iki parça:**

1. **Madde özellik tablosu (tek doğruluk kaynağı).** Madde kimliği
   (`substanceTag`) çözücüden bağımsızdır; faz ve görünüm onun *durumudur*.
   Tablo madde başına, fiziksel birimlerle:
   - faz sınırları: erime / donma / kaynama sıcaklığı (K)
   - gizli ısılar: erime, buharlaşma (J/kg)
   - faz başına yoğunluk (kg/m³), viskozite (m²/s), ısıl iletkenlik
   - yanma: tutuşma sıcaklığı, yanma ısısı, ürün oranları (duman, char, kül)
   - granül parametreleri (sürtünme, kohezyon) katı/granül fazı için

   Her çözücü bu tabloyu OKUR; kendi kopyasını tutmaz. Ad üzerinden özellik
   tahmini (`find("Plastic")`) kaldırılır.

2. **Dönüşüm defteri (exchange bus).** Her adımda:
   çözücüler kendi fazını çözer → durumdan (T, yük, yanma hızı) dönüşümler
   hesaplanır → aktarımlar ortak birimlerde (kg, J, kg·m/s) deftere yazılır →
   hedef çözücü kendi temsiline çevirir → **korunum kontrolü** (kütle ve
   enerji; giren = çıkan + birikim).

   **27A uygulama notu (canlı PASS):** APIC'in `mass_fraction` alanı tek
   başına kg değildir. Her parsele yaşamı boyunca sabit kalan `rest_mass_kg`
   eklendi; mevcut kütle `rest_mass_kg * mass_fraction` olarak tanımlandı.
   Normal tohum/emitter parselleri kanonik sıvı yoğunluğundan `ρ·h³/ppc` ile,
   MSF eriyik parselleri ise aktarımın kesin `spawn_mass/spawn_count` değeriyle
   başlar. İki alan SimCache v7'de birlikte saklanır. 27B bu kg değerini APIC
   kaynak kaybı, gaz hedef envanteri ve defter olayı arasında birebir bağladı;
   cache biçimi fiziksel gaz yan alanları için v8'e çıktı.

| Dönüşüm | Kaynak → hedef | Bugün |
|---|---|---|
| Erime | katı mesh / MSF → sıvı parçacık | var (`MoltenMassTransfer`, MSF 6b/6c) |
| Donma | sıvı → katı (`frozen`) | var (termal sıvı) |
| Buharlaşma / kaynama | sıvı → gaz | yok (yanma dışında) |
| Yanma | sıvı → gaz (+ısı) | var |
| Yanma / piroliz | katı → gaz + char (MSF) | kısmen (yangın → yapı, MSF char) |
| Yumuşama / sinterleme | granül ↔ katı / sıvı | kısmen (granül termal zinciri) |
| Yoğuşma | gaz → sıvı | yok |
| Sis aktarımı | sıvı `mist` → gaz | 27C canlı PASS; fiziksel kg/J + defter + uzamsal taşıma |

★ Enerji de defterdedir: erime ve buharlaşma gizli ısıyı kaynaktan çeker.
Bugün erime ısı tüketmiyorsa, erime sonsuz hızda ve bedava olur — sonuç
"makul görünür" ve kimse bug diye raporlamaz.

### 4.6 ★★★ Materyal sözleşmesi: veri hangi materyalle sürülür

Görünüm çözücü "ne çizilir"i söyler; bu bölüm "neyle boyanır"ı. Geniş kurulmalı,
çünkü aynı madde üç görünümde görünebilir ve **aynı madde gibi görünmek
zorundadır**: şelaleden kopan sprey, gövdeyle aynı suyun damlasıdır.

**Bugün (koddan, 2026-09-27) — domain'e gömülü, görünüm başına ayrı:**

| Görünüm | Materyal | Alan |
|---|---|---|
| SDF yüzeyi | Principled BSDF veya yerleşik dielektrik (-1) | `fluid_surface_material_id` |
| SDF, madde başına | kompozisyon alanı: hücre başına materyal indeksi + karışabilirlik rampası | `fluid_substance_materials[].material_id`, `miscibility` |
| SDF iç ortamı | yerleşik dielektriği domain `VolumeShader`'ı boyar; bağlı BSDF kendi Transmission/Interior'unu kullanır | `shader` |
| Splat | sahne materyali (bilerek yüzeyden ayrı); scene-object splat'te yüz materyalleri | `fluid_particle_material_id` |
| Fog / gaz | `VolumeShader` (yoğunluk, saçılma, emilim, emisyon: blackbody / rampa) | `shader` |
| Köpük (Volume) | ayrı `VolumeShader`, SDF hacminin sıcaklık kanalına biner | `foam_shader` |

★ `SubstanceMaterial` bugün materyal kimliğini VE fiziği (viskozite, faz,
karışabilirlik) aynı struct'ta, **domain başına** tutuyor. §4.5'teki madde
özellik tablosunun ilk hâli budur — ama global değil, domain'e gömülü.

**Önerilen model — madde görünüm profili:**

Madde (`substanceTag`) başına tek bir **görünüm profili**, görünüm başına slot:

```
SubstanceAppearance
  surface   : BSDF materyali          → SDF yüzeyi (+ faz varyantı: frozen)
  interior  : hacim ortamı            → SDF içi emilim/saçılma (dielektrik yolu)
  splat     : BSDF materyali | inherit → varsayılan: surface
  medium    : VolumeShader | derived  → fog/gaz/mist
  phase_overrides: { frozen: surface2, molten: emission ... }
```

Kurallar:

1. **Miras varsayılandır.** `splat` boşsa `surface`'i kullanır; `medium`
   boşsa `interior`'dan türetilir (su sisi: emilimi düşük, saçılması yüksek
   beyaz ortam). Kullanıcı üç görünümü ayrı ayrı boyamak ZORUNDA kalmaz;
   isterse geçersiz kılar.
2. **Faz materyali ayırır, madde kimliğini ayırmaz.** Donmuş mum, sıvı mumla
   aynı maddedir; `phase_overrides.frozen` ikinci yüzey materyalini verir.
   SDF'de faz oranı, kompozisyon alanı gibi hücre başına karışım olur.
3. **Kanal sözleşmesi (HEDEF, bugünkü durum değil):** materyali süren
   veriler adlandırılmış kanallardır ve her görünüm hangi kanalları
   taşıdığını ilan eder:

   | Kanal | SDF hacmi | Splat instance | Fog hacmi |
   |---|---|---|---|
   | yoğunluk / level set | ✔ | — | ✔ |
   | kompozisyon (madde karışımı) | ✔ | madde etiketi | (Faz 3+) |
   | sıcaklık (K) | ✔ (erimiş parıltı) | ✔ (instance verisi) | ✔ (blackbody) |
   | faz oranı (frozen) | ✔ | etiket | — |
   | UVW (doku koordinatı) | ✔ | ✔ | — |
   | köpük oranı | ✔ | etiket | ✔ |
   | alev / reaksiyon hızı | — | — | ✔ (gaz) |

   Koddan doğrulandı (§8b 1. tur): splat instance'ında instance başına veri
   YOK (transform + kaynak indeksi); madde başına materyal, materyal başına
   ayrı geometri kaynağıyla yapılıyor. Sıcaklıkla boyanan splat yeni bir
   instance veri yolu ister (RT ve raster ikisinde). SDF'de sıcaklık kanalı
   bugün köpüğe ait — tablodaki SDF sıcaklık hücresinin ön koşulu köpüğün
   kendi kanalı.
4. **Kanallar birbirini ödünç ALMAZ.** Köpüğün SDF `temperature` kanalına
   binmesi (`FOAM_TEMP_SCALE`) tarihsel bir kısayoldur; sıcaklık gerçek veri
   taşıdığı anda iki anlam çakışır. Birleşik modelde köpük kendi kanalını alır.
5. **Yüzey durumu ayrı katmandır.** Char, ıslaklık, is MSF'den gelir ve sahne
   objesinin materyalinin ÜSTÜNE uygulanır; madde profiline karışmaz.
6. **Profil tek kaynaktır, panel onu düzenler.** Aynı profil splat,
   SDF ve fog panelinde görünür; panelde değiştirmek hepsini değiştirir.
   IPC: profil okunur/yazılır, görünüm başına efektif materyal raporlanır
   (hangi slot, miras mı, override mı).

**Açık:** partikül sisteminin `Appearance Profile`'ları (tek GPU LUT,
commit 7dbd72c) ile madde görünüm profili aynı kavram mı, yoksa biri
diğerine mi başvurur? Faz 0'da ikisi okunup karar verilmeli — iki ayrı
"görünüm profili" kavramı, paneli yalancı yapmanın en kısa yolu.

---

## 5. Çözücü birleştirme: iki seviye

### A) Zayıf bağlama, tek mantıksal domain ve faz gridleri (hedef)

İki çözücü tek Matter kimliği altında kendi fiziksel grid'inde sırayla adım
atar; dönüşüm defteri, kaynak eşleme ve hareketli sınır ile birbirini görür
(§4.4). Şelale sisi, lav dumanı, yanan sıvı ve rüzgarda sprey için fiziksel
olarak yeterli. Gazın geniş/kaba, sıvının dar/ince çalışma alanı korunur.
Collider kaynağı ortaktır ama her fazın grid'ine kendi koordinatlarında
damgalanır. Mevcut çözücülerin kararlılığı riske girmez.

### B) Güçlü iki fazlı projeksiyon (isteğe bağlı, sonra)

Tek hız alanı, tek basınç denklemi, değişken yoğunluk (su ≈ 1000, hava ≈ 1).
Kabarcık, hava karışması, lıkırdama kendiliğinden çıkar. Hipotez: açık
"mühürlü basınç cebi" savrulması hava gerçek faz olunca kaybolabilir —
DOĞRULANMADI. Engeli: 1000:1 oran basınç sistemini kötü koşullar; güçlü
önkoşullayıcı ister ve GPU MGPCG taşıması duraklatılmış durumda. A'nın fazlar
arası eşleme ve korunum sözleşmesi B'nin de temelidir, iş boşa gitmez.

### Çözünürlük politikası

Gaz geniş/kaba, sıvı dar/ince ister. Tek mantıksal domain iki fiziksel faz
grid'ini taşır; bounds ve voxel boyutu faz başına belirlenir. Her grid ayrıca
yalnız etkin hücre penceresinde çalışır. Sıvının yüzey detayı fizik grid'inden
ayrılmış kalır (SDF çözünürlük çarpanı zaten böyle).

---

## 6. Fazlar ve kapılar

Her faz kendi başına değer üretir ve bir sonrakine bir şey yıkmadan zemin
hazırlar. Her kapı IPC'den ölçülür.

| Faz | İş | Kapı (ölçüm) |
|---|---|---|
| **0** | Bu not + ölçü aletleri: etiket sayaçları, görünüm başına kaynak listesi IPC'de | `fluid.get` görünüm listesini ve etiket sayılarını döndürür |
| **1** | Görünüm çözücü + görünüm başına kaynak (§4.3). Tek karar noktası. Etiket henüz yoksa bugünkü mod tek etiket gibi davranır | Aynı domain'de SDF + fog **aynı anda**; panel, köprü ve API aynı listeyi raporlar |
| **2** | Parçacık etiketleri simülasyonda (§4.2); köpük etikete katlanır | Etiket sayıları; şelale sahnesinde gövde SDF + sprey splat + köpük Solid/Material/RT'de |
| **1-UI** | Domain paneli altı sekmeye taşınır (§8b UI KARARI): Domain · Matter · Environment · Solvers · Output · Measure; Gas/Fluid düğmeleri → salt okunur Contents; "Unified Volume Shader Properties" ve gömülü köpük editörü kalkar; `SelectableType::Material` + Output'ta Edit… (Material Properties) / Volume… (hacim nesnesi). Tüm UI metni İngilizce, genel tema, renkli buton yok. Faz 1 ile birlikte | Her alan tek sekmede düzenlenir; bir BSDF nesneye atanmadan Material Properties'te açılır; panelde materyal editörü kalmaz |
| **K** | Madde özellik tablosu (§4.5.1); çözücüler tablodan okur, ad tahmini kalkar. Faz 1–2 ile PARALEL yürür | Aynı maddenin her çözücüde aynı sayıları kullandığı IPC'den okunur; eski sahneler aynı davranır |
| **3** | Zayıf bağlama A + dönüşüm defteri (§4.5.2): faz gridleri arasında sıvı hareketli sınır, gaz sürüklemesi, `mist` → gaz aktarımı; mevcut dönüşümler (erime, donma, yanma) deftere taşınır | Kütle VE enerji korunum sayaçları: kaynaktan çıkan = hedefe giren (tolerans içinde) |
| **4 ✔** | Domain birleşmesi: Gas/Fluid tipleri fazlara dönüşür; eski sahne göçü | PASS: tek Matter domain, `phases=[gas, liquid]`, 120 sıvı parçacığı + 67 aktif gaz hücresi |
| **5** | (İsteğe bağlı) Güçlü iki fazlı projeksiyon B | Kabarcık sahnesi; basınç iterasyon sayısı ve süre ölçülür |

### 1. adım neden ayrı bir ön-refactor DEĞİL

Birleştirme hedefi varken bugünkü `FluidRenderMode`'u temizlemek boşa iş olur —
o enum gidecek. Ama 1. adımın iki parçası birleşik modelde de gerekli:
**görünüm başına kaynak** (birleşik domain de gövdeyi SDF, sisi fog olarak
aynı anda çizer; render sözleşmesi bunları ayrı hacim ister) ve **tek karar
noktası** (etiket → görünüm eşlemesi). Bu yüzden Faz 1 doğrudan birleşik
modelin görünüm çözücüsü olarak kurulur, mevcut modun etrafına değil.

---

## 7. Aktif planlarla ilişki

- [GAZ_DERSLERI_VE_FLUID_DEVRI.md](GAZ_DERSLERI_VE_FLUID_DEVRI.md) (AKTİF
  yazıyor, ama güncel değil): 2026-09-21 sonrası commit'ler performans
  tarafının (§7b) büyük ölçüde ilerlediğini gösteriyor. 2026-09-27'de kodda
  bakıldı: `SimulationComputeVulkan.cpp` tahsislerini VRAM muhasebesine
  kaydetmiyor, yani §7a (simülasyon tamponlarının `perf.get_gpu_memory`'de
  görünmesi) muhtemelen AÇIK — canlı ölçülemedi. Faz gridlerinin gerçek VRAM
  maliyeti C3 kabulünde faz başına görünür olmalı.
- [PARTICLE_SYSTEM_GPU_ROADMAP.md](PARTICLE_SYSTEM_GPU_ROADMAP.md) (AKTİF):
  `Domain Deposit` bu modelin dış üretici sözleşmesidir; `Surface State
  Deposit` MSF'ye yazımdır. Çelişki yok; partikül sistemi domain'e dönüşmez.
- [KINEMATIC_COLLIDER_SOURCES.md](KINEMATIC_COLLIDER_SOURCES.md): collider
  kaynağı ortaktır; C3'ten sonra her faz grid'ine kendi koordinatlarında
  damgalanır.
- [VULKAN_GAS_FLUID_LAYERING.md](VULKAN_GAS_FLUID_LAYERING.md): render
  değişmezleri bu modelde de bağlayıcı (ayrı maske bitleri, gerçek geçişte
  devir, devrin GI sekmesi yememesi).
- ★★ [ATMOSPHERE_SYSTEM.md](ATMOSPHERE_SYSTEM.md) §3.4 (karar 2026-09-30):
  birleşik domain'in **tek bir ORTAM AŞAMASI** olur. Atmosfer yalnız ortak
  veri üretir (`atmosphere::sampleAmbient(pos)`: sıcaklık, nem, basınç,
  rüzgâr); domain başına tek kapı (atmosfer / yerel override), domain içindeki
  bütün çözücüler (APIC, Euler, MSF) çözülmüş ortamı buradan alır — açık
  sınırda rüzgâr girişi, havada sürüklemenin hava hızı, ortam sıcaklığı,
  kuruma için nem. Faz 2'nin çözücü başına `inherit_atmosphere` bayrakları
  (APIC params, partikül ayarları, gaz stratification) bu aşama gelince
  **sökülür**. Kalibrasyon sıfırı (`WorldThermalState.reference_kelvin`)
  ortamdan ayrı kalır — domain'e taşınırken de ayrı kalmalı.

---

## 8. Çözülen veya kapsam dışına ayrılan sorular

Bu başlık artık karar bekleyen bir liste değildir. Son durum:

1. **Etiket kriterlerinin eşikleri:** fiziksel hız/kütle eşikleri mutlak
   birimde; komşuluk yarıçapı voxel biriminde ve IPC'de raporlanır. GPU/CPU
   sınıflandırma aynı sözleşmeyi kullanır.
2. **`mist` aktarımında kütle ölçeği:** fiziksel kg/J sidecar ile görsel gaz
   tracer'ı ayrıldı; dönüşüm tek transfer servisinde ve ledger'da kayıtlıdır.
3. **Kristalleşme verisi:** `frozen` bugün ikili kalır. Kristal boyutu, donma
   süresi ve soğuma hızı ayrı madde-fiziği işidir; çekirdek kapanışına dahil değil.
4. **Domain hareketi:** tek mantıksal domain referans çerçevesi ortaktır; C3'te
   faz gridleri bu çerçeveden kendi bounds/origin'ini türetir.
5. **CPU referansı:** faz alışverişinin otoritesi host/CPU servisleridir; GPU
   yolları aynı sonuçları hızlandırır ve başarısızlıkta açık fallback uygular.

---

## 8b. Yeni oturum için başlangıç — ne incelendi, ne incelenmedi

> 2026-09-27 oturumu bu notu fog/splat işinin YANINDA yazdı. İncelenen yalnız
> fog'un geçtiği yol: render köprüsünün domain hacmi rotası, analiz geçişi,
> katmanlama notu, birkaç yapı. **Tek domain'e geçişin güvenli olduğunu
> söyleyecek kadar inceleme YAPILMADI.** Yeni oturum Faz 0'ı bir ENVANTER
> olarak başlatmalı, koda dokunmadan.

**Ölçülenler (2026-09-27):**

- `SimulationDomainType::Gas|Fluid` doğrudan dalı: **85 yer, 18 dosya** (alt
  sınır — `is_gas_state` gibi bayraklara çevrilmiş dalları saymaz). En yoğun:
  `ParticleSimulation.cpp` 26, `RtApiFluid.cpp` 15, `scene_data.h` 10,
  `scene_ui_simulation_domains.cpp` 8.
- ★ İyi haber: **depolama zaten tek yapı.** `SimulationGridDomainDesc` tek
  struct (~550 satır), içinde 16 `gas_*` ve 50 `fluid_*` alanı yan yana;
  runtime durumu da ortak `SimulationGridDomainState`. Tip enum'u bugün
  "hangi alan grubu anlamlı" seçicisi gibi çalışıyor — birleşme depolamada
  değil, **dallarda** yapılacak.
  ⚠ 1. tur bunu DARALTTI — aşağıya bak: tanım tek struct, ama **alanların
  anlamı** faza göre değişiyor.

### Faz 0 envanteri — 1. tur bulguları (2026-09-27, koddan okundu, ölçülmedi)

**1. 85 dal sınıflandırıldı** (`SimulationDomainType::` doğrudan karşılaştırma):

| Sınıf | Adet | Nerede | Birleşik modelde |
|---|---|---|---|
| (b) çözücü / "faz var mı" | 43 | `ParticleSimulation.cpp` 23, `RtApiFluid.cpp` 9 (7'si "gas domain not found" araması), `scene_data.h` 4, Molten/GasPulse/GasStructural/ParticleAuthoring 6, `RtApiParticle` 1 | Çoğu mekanik: `type == Gas` → "gaz fazı var mı" yüklemi. Zor olanlar aşağıda |
| (e) kimlik / depolama / serileştirme | 18 | `ProjectManager` 4, presetler 4, `RtApiFluid` oluşturma+tip dizgisi 5, `ParticleSimulation.h` 2, kanal düzeni 2, 1 test grid'i | Faz 4 göçü |
| (d) UI | 13 | `scene_ui_simulation_domains` 8, gizmo 3, force field 1, molten transfer 1 | Yeniden tasarım (mock) |
| (c) render | 11 | `scene_data.h` 6, köprü 2, lifecycle 1, `RtApiFluid` 1, splat materyali 1 | Görünüm çözücüsü (Faz 1) |

Bayrağa çevrilmiş kullanım (`is_fluid_domain`, `is_gas_state`,
`wants_gas_channels` …) ayrıca **~60**: UI 21, `scene_data.h` 10 (render
rotası), `ParticleSimulation.cpp` 9, `RtApiFluid.cpp` 8, köprü 4.
`APICFluidSolver.cpp` ve basınç shader'larındaki `is_fluid` **hücre** bayrağıdır,
domain tipi değil — sayılmadı. `ForceField`'ın `is_gas`'ı tüketici filtresi
(`affects_gas`); birleşik domain'de faz başına sorulmalı.

(b)'nin **zor** olanları — mekanik değil, anlam değiştiren dallar:
- `ParticleSimulation.cpp` ~9861: partikül → grid birikimi gaz-only, çünkü
  sıvı domain'de `density` başka bir şey (aşağıda ★★★).
- ~9040: `allocate_gas_channels` sıvıda **kapalı** — `temperature`, `fuel`,
  `interaction` sıvı domain'de BOŞ vektör. Birleşik domain gaz kanallarını
  "gaz fazı var mı"ya bağlamalı; aksi halde her sıvı 3 float/hücre öder.
- ~12451: analiz geçişi (4. parti fog hatasının yeri — sayaç kapsamı).
- ~6194 ve ~11560: sıvı → gaz yanması bugün **iki ayrı domain** arasında,
  kutuların kesişimi üzerinden. Birleşik modelde bu domain içi alışveriş olur
  (§4.4) — en doğal Faz 3 adayı.

**★★★ 2. Aynı alan adı, iki fiziksel anlam** (§2'deki "ortak konteyner"
iddiasının sınırı):

| Alan | Gaz domain'inde | Sıvı domain'inde |
|---|---|---|
| `grid.density` | duman yoğunluğu (birimsiz) | parçacık sayısı / ppc = **hacim oranı** (`sim_fluid_density_splat`) |
| sıcaklık | `grid.temperature`, 0 tabanlı ısı, render'da ×3000 → K | **parçacık üzerinde** Kelvin; grid kanalı yok |
| `vel_x/y/z` | gazın hız alanı | APIC'in P2G/basınç çalışma alanı |

Sonuç: **§5.A'daki "ortak grid" ortak ALAN demek olamaz.** İki faz aynı
seyrek topolojiyi/adreslemeyi paylaşabilir, ama her faz kendi `density` /
sıcaklık / hız alanını taşımalı — yoksa ikinci çözücü ilkinin alanını
sessizce ezer. Bu, §8.2'deki birim sorusunun somut hâli: `mist` aktarımı
"hacim oranı"nı "duman yoğunluğu"na çevirir ve dönüşüm tek yerde yaşamalı.

**3. Hacim kimliği — ölçü aleti YOK.** `render.volume_tables` backend başına
**sayı** raporlar (`instance_count`, `dense_gas_mirror_buffers`), hacim başına
**satır** değil. Kimlik değişmezlerinin testi (slot sabit kalır mı, invalidate
TLAS slotunu kaybeder mi, alakasız rebuild SDF'yi atar mı) ancak şu okuma
eklenince yazılabilir: hacim başına `{domain, rol (sdf/fog/foam), vdb_id,
ssbo_slot, maske sınıfı, is_active, kayıt/yeniden kurulum sayacı}`. Bu Faz 0'ın
**ilk kod işi** — §6 Faz 0 kapısıyla aynı yer (`fluid.get` görünüm listesi).

- ★ Tarihçe: domain başına **ikinci hacim slotu zaten vardı**
  (`domain_foam_volumes`, `domain_foam_vdb_ids`). 2026-06-25'te köpüğü SDF
  hacmiyle çakıştığı için (siyah küp) sökülüp sıcaklık kanalına bindirildi;
  kod bugün yalnız eski oturumdan kalanları söküyor. Çakışık kutu kökü
  **sonra**, 2026-08-16'da çözüldü. Yani söküm gerekçesi muhtemelen artık
  geçersiz — ama DOĞRULANMADI. Faz 1'in ilk canlı ölçümü bu olmalı.
- Katmanlama sözleşmesi ([VULKAN_GAS_FLUID_LAYERING.md](VULKAN_GAS_FLUID_LAYERING.md))
  **iki ayrı domain** için doğrulandı (Burning Fuel Spill: sıvı domain + gaz
  domain). Aynı domain içinde SDF + fog aynı kutuda HİÇ denenmedi.
- **Fog + fog** (iki katılımcı ortam aynı kutuda) için Vulkan'da hakem yok;
  sözleşme "iki tam çakışık AABB yürüyüşü" yasaklıyor. §4.3'ün "mist + gaz
  tek fog hacmi" kararı bunu zaten önlüyor — ama o zaman iki kaynağın
  yoğunluğu tek birimde birleşmeli (madde 2).

**★★ 4. Köpük kanalı: bugün çakışma yok, ama tuzak kurulu.** SDF rotasında
sıcaklık kanalı **hiç yüklenmiyor** (rafine SDF grid'i sim grid'inden farklı
çözünürlükte; `up_temp = nullptr`). Köpük o boş kanala biniyor ve shader
köpüğü `vol.vdb_temp_address != 0` ile TANIYOR (`volume_closesthit.rchit`
~2403, `ray_color.cuh` ~647). Yani SDF'ye erimiş-madde parıltısı için Kelvin
yüklendiği gün, shader onu **beyaz köpük** olarak çizer (1400 K / 10000 =
0.14 köpük yoğunluğu) — hatasız, makul görünen yanlış. Fog rotası Kelvin'i
alır ama köpüğü almaz; ikisi bugün rotayla birbirini dışladığı için
karşılaşmıyor. §4.6'daki "SDF hacmi: sıcaklık ✔" hücresi HEDEF'tir; ön koşulu
köpüğün kendi kanalı ve rafine çözünürlükte sıcaklık.

**5. Tek karar noktası — bugün DÖRT kopya** (§1'deki iddia koddan
doğrulandı): `scene_data.h` ~2663 (`fluid_surface_route` /
`fluid_fog_route`), köprü ~1400 (parçacık başına splat mı), `RtApiFluid.cpp`
~196 (`effective_representation`, yorumu "exactly the way the render bridge
resolves it" diyor — yani elle kopya), UI ~2651 ("Now drawing", combo
indeksinden). ✔ Faz 1 / 1. parti bunları `resolveFluidViews`'a indirdi (köprünün
köpük bölümü ve `refreshFluidSurfaceMaterial` da aynı soruyu ayrıca soruyordu).
**Cevaplandı (koddan):** fog + SDF override'da SDF etiketsiz fog parçacıklarını da
İÇERİYORDU (yalnız Splat bağlamaları dışlanıyordu) — çözücüyle kapandı.

**6. Materyal envanteri (§4.6):**
- Splat instance'ı yalnız `InstanceTransform` + `source_index` taşır; instance
  başına renk/sıcaklık/madde verisi YOK. Madde başına materyal, **materyal
  başına ayrı geometri kaynağı** ile yapılıyor (instance kaynak seçer).
  Sıcaklıkla boyanan splat yeni bir instance veri yolu ister (RT + raster).
- Köpük/sprey/kabarcık üç ayrı materyal (`fparams.*_material_id`), ayrı
  instance grubu.
- **`ParticleAppearanceProfile` ≠ madde görünüm profili.** Partikülünkü
  normalize YAŞ ekseninde bir billboard eğrisi (renk rampası, opaklık, boyut,
  emisyon; sistem başına id, tek GPU LUT). Madde profilinin ekseni faz ve
  sıcaklık. Öneri: birleştirme; adları ayır ("Appearance Profile" partikülde
  kalır, madde tarafı `SubstanceAppearance`). İlişki tek yönlü: bir partikül
  emitter'ının `Domain Deposit`'i hedef maddeyi (`substanceTag`) söyler,
  domain o maddenin profilini kullanır.

**7. Domain'in sahibi.** Grid domain'leri bir `ParticleSimulationSystem`
runtime'ının içinde yaşıyor (`ParticleSystemObject::runtime`, hacim adı
`"<sistem> Domain <d>"`); partikül yol haritası da "grid domain'i olan sistem
device-resident yola girmez" diyor. §3.4 ("üretici domain'in sahibi olmaz")
kavramsal olarak doğru, ama **konteyner olarak** bugün sahip sistemdir. Faz
4'ün kararı: domain sistemde mi kalır, üst seviyeye mi çıkar. Karar verilmedi.

### Faz 0 envanteri — 2. tur: canlı ölçüm (2026-09-27, exe 13:34)

Probe: `scripts/ipc/Probe-FluidRenderMatrix.ps1`. Boş sahne, 2 m'lik kutuda
38.720 parçacıklık su bloğu; ardından `burning_fuel_spill` preset'i.

**Render matrisi — tek başına sıvı (Vulkan):**

| Görünüm | Solid | Material (RayFusion) | Rendered (Vulkan RT) |
|---|---|---|---|
| splat | ✔ impostor küreler (38.720 yüklendi = çizildi) | ✔ instance (38.720), soluk | ✔ |
| SDF | ✔ gri yüzey | ⚠ soluk gri-beyaz kütle — Rendered'daki mavi camla aynı madde gibi görünmüyor | ✔ mavi dielektrik |
| fog | ✔ düz açık mavi önizleme | ⚠ **neredeyse siyah** | ✔ gri-kahve sis |

**Gaz + SDF aynı XZ kutusunda (iki ayrı domain, Burning Fuel Spill):** üç
yolda da iki katman birlikte çiziliyor; katmanlama sözleşmesi canlıda da
tutuyor. Yanma zinciri çalışıyor (60 karede 24.192 → 21.995 parçacık yandı, gaz
26k hücre).

**OptiX hücreleri ölçülemedi:** IPC'den OptiX'e geçiş yok. Dondurulmuş
yolda bu kabul edilebilir, ama §8b madde 7'nin "fark nasıl gösterilir"
sorusunu otomatik testle cevaplamak da mümkün değil.

**★★★ Ölçüm sırasında bulunan hata — IPC mod değişikliği raster'da bir mod
GERİDE kalıyordu.** `fluid.set_param render_mode` yalnız alanı yazıyordu;
panel combo'su ise `requestSimulationTimelineRenderResync()` + `start_render`
de çağırıyor. Duraklatılmış timeline'da hacim rotası yeniden koşmadığı için
Solid/Material, bir **önceki** modun temsilini çizdi: particles'ta eski SDF
splat'lerle üst üste, surface'te boş ekran, fog'da keskin yüzey. Rendered her
karede kendi senkronladığı için hep doğruydu, bu da hatayı gizledi.
`fluid.step` de senkronu tetiklemiyor; yalnız `timeline.set_frame` ya da bir
Rendered karesi tetikliyor. Kanıt: aynı exe'de mod değişikliğinin ardından
resync isteyen `fluid.set_fog` çağrılınca üç mod da Solid'de anında doğru
çizildi. Düzeltme `RtApiFluid.cpp` render_mode dalına yazıldı (panelle
birebir aynı); ✔ exe 22:21'de doğrulandı. ★ Bu tür, IPC'den sürülen raster testlerinin
**yanlış kareyi ölçmesi** demek — 3. partideki fog testinin raster adımları da
bundan etkilenmiş olabilir.

**Hacim tabloları (sayı):** particles modunda render backend tablosu `1`,
viewport `0` kaldı; sökülen SDF render tarafında slotunu tutuyor olabilir
(tasarım: "slot korunur, içerik pasif") — sayıdan ayırt edilemiyor. Madde
3'teki satır aracı tam olarak bunu ölçecek.

**VRAM (madde 4):** iki domain'li sahnede kullanım 1,10 GB; bunun **519 MB'ı
`untracked_bytes`**. Kategoride simülasyon yok. §7'deki "sim tamponları VRAM
muhasebesine kaydolmuyor" tahmini **canlıda doğrulandı** — Faz 3'ün ön koşulu
açık. 16 hacim tavanı ölçülmedi (bu sahnede en çok 2 hacim).

**★ UI KARARI (2026-09-27, kullanıcı): ayrı DCC ekranı YOK; mevcut Domain
alt panelinin SEKMELERİ yeniden düzenlenir.** Fizik paneli (field / particle /
domain / collider / bodies) ve node graph olduğu gibi kalır. Bugünkü üç sekme
(Setup & Grid · Solver & Physics · Shading & Rendering) ve üstteki Gas/Fluid
tip düğmeleri yerine altı sekme:

| Sekme | Soru | İçerik |
|---|---|---|
| **Domain** | Nerede, hangi çözünürlükte, nereden madde giriyor? | Contents (salt okunur, faz başına sayı) · sınırlar ve davranış · çözünürlük/voksel · cihaz/backend · Kaynaklar (seeding, Flow Sources, emisyon sınırları) |
| **Matter** | Madde ne, nasıl davranır, neye dönüşür? | madde listesi; madde başına faz, yoğunluk, reoloji, granül, termal sınırlar (K), gizli ısı, yanma; faz dönüşüm anahtarları (§4.5 tablosu) |
| **Environment** | Maddeyi çevreleyen koşullar? | ortam sıcaklığı (`world.get_thermal`), yerçekimi, ortam hava yoğunluğu/kaldırma referansı; domain'i etkileyen force field'lar (salt okunur liste, düzenleme Fields panelinde) |
| **Solvers** | Hangi çözücü, hangi ayarla? YALNIZ var olan faz için | Sıvı: APIC/FLIP, dissipation, coupling, reseed, etiket üretimi (Ihmsen eşikleri) · Gaz: kanallar, hareket, türbülans |
| **Output** | Hangi veri hangi materyalle çıkar? (§4.3 + §4.6) | etiket → görünüm → materyal tablosu; Now drawing (görünüm çözücüsünden); görünüm başına ayarlar: SDF yüzeyi, splat geometrisi, fog/medium shader, köpük görünümü |
| **Measure** | Ne oldu, kanıtı ne? | istatistikler, step stats, etiket sayaçları, dönüşüm defteri + korunum, önbellek/bake, VDB export |

Bugünkü bölüm → yeni yer:

| Bugün (sekme / bölüm) | Yeni sekme |
|---|---|
| Setup: Compute Device & Backend · Grid Resolution & Scaling · Domain Bounds & Behaviors | Domain |
| Setup: Simulation & Collision Statistics | Measure |
| Solver: Simulation Solver Channels · Turbulence | Solvers (gaz) |
| Solver: Buoyancy & Gas Motion | Solvers (gaz); ortam referansı → Environment |
| Solver: Combustion & Fire Physics | Matter (yanma madde özelliğidir) |
| Solver: Fluid Seeding & Capacity | Domain (Kaynaklar) |
| Solver: APIC/FLIP → Dissipation, Coupling | Solvers (sıvı) |
| Solver: APIC/FLIP → Rheology · Granular Material (+Damage, Thermal Softening) | Matter |
| Solver: Dynamic Particle Reseeding | Solvers (sıvı) |
| Shading: Flow Sources Registry · Flow Control & Emission Limits | Domain (Kaynaklar) — bugün yanlış sekmede |
| Shading: Liquid Display → "Physics (re-bakes the sim)" | Matter — bugün görüntü bölümünün içinde |
| Shading: Liquid Display → "Look" · Splat Geometry · Surface SDF · Unified Volume Shader | Output |
| Shading: Whitewater | üretim eşikleri → Solvers (sıvı, etiketler); görünüm → Output |
| sekme dışı: Fluid Step Stats · VDB Export · VDB Cache & Baking | Measure |

Kurallar: (1) Bir alan TEK sekmede düzenlenir; başka sekmede görünüyorsa salt
okunurdur. (2) Faz yoksa o fazın bölümü gizlenir, sekme boş kalırsa "bu
domain'de gaz yok" yazar. (3) Her alanın IPC karşılığı vardır (CLAUDE.md §1);
sekme adı IPC namespace'i DEĞİLDİR, alanın sahibi olan veri modelidir.
(4) Taşıma Faz 1 ile yapılır, ayrı bir UI turu açılmaz. (5) **Domain paneli
materyal DÜZENLEMEZ, yalnız BAĞLAR** (kullanıcı, 2026-09-27): BSDF'ler (SDF
yüzeyi, splat) Material Properties'te, fog `VolumeShader`'ı sahibi olan hacim
nesnesinin panelinde düzenlenir — kod zaten böyle: `VolumeShader` materyal
kütüphanesinde değil, sahibinde yaşar (domain / GasVolume / VDBVolume). Domain
panelindeki "Unified Volume Shader Properties" kalkar (aynı shader'ın ikinci
editörü). Yönlendirme (koddan, 2026-09-27): Material Properties, Properties panelinin
bir bölümüdür ve yalnız `SelectableType::Object` seçiliyken çizilir
(`scene_ui_hierarchy.cpp` ~3284). Domain panelindeki gömülü köpük editörü
(`drawInlineMatEditor`, `scene_ui_simulation_domains.cpp` ~3644) bu kısıtın
YAMASI olarak yazılmış. Karar: **"Edit…" → seçim `SelectableType::Material`
(yeni; seçim = materyal id) → Properties > Material Properties aynı
`drawPrincipledBSDFEditor` ile açılır.** **"Volume…" → domain'in `VDBVolume`'u
seçilir** (mevcut seçim tipi; hacim paneli zaten "Owned by liquid domain"
diyor). Kök çözülünce gömülü köpük editörü kalkar. Terrain paneli de aynı yamayı
taşıyor (`scene_ui_terrain.hpp` ~721) — fırsatçı temizlik, bu işin parçası değil. (6) Tema:
genel ImGui teması, standart combo/buton — renkli (durum kodlu) buton YOK. Mock (sekmeler):
https://claude.ai/artifact/XQUkjRKwhdYX2UyM7fAVmG

**★★ Faz 1 riski — raster viewport'ta hacim KİMLİĞİ yok (2026-09-27, ölçüldü).**
Viewport backend'i TLAS kurmaz; hacim SSBO'sunu paket sırasıyla yayımlar ve
slotun tek tanıtıcısı `vdb_id`'dir. `vdb_id` tasarım gereği döner: kimlik
testinde particles → surface turu onu 0 → 1 yaptı. Bugün domain başına tek
hacim olduğu için sorun yok. Faz 1'de bir domain SDF + fog iki hacim taşıdığında
viewport tarafında "hangi slot hangi görünüm" sorusu yalnız paket sırasına
bağlı kalır. Render backend bu sorunu `stable_key` ile çözmüştü
(`m_volumeStableKeys`); Faz 1 aynı kimliği paket yoluna da vermeli ya da
paketin kendisine görünüm rolünü (sdf/fog) taşıtmalı.

**Faz 0 envanteri (yeni oturumun ilk işi):**

1. **85+ dalı sınıflandır:** her biri (a) kimlik/depolama → birleşir,
   (b) çözücü seçimi → "faz var mı" sorusuna döner, (c) render → görünüm
   çözücüsüne gider, (d) UI → yeniden tasarlanır, (e) göç/serileştirme.
   Sayılar bu tabloya göre fazlanır.
2. **Render matrisi, ÖLÇEREK:** görünüm (SDF / splat / fog / gaz) × yol
   (Vulkan RT, RayFusion, raster Solid/Material, OptiX). Hangi hücre bugün
   çalışıyor? `viewport.capture` + `render.start` ile görüntü alınır.
3. **Hacim kimliği riski:** domain başına tek hacimden ikiye geçmek, bu
   depoda en çok kırılan bölgeye dokunur (hacim kimliği churn'ü → siyah bant,
   invalidate'te TLAS slotu kaybı, alakasız rebuild'de SDF'nin atılması).
   Önce bu değişmezlerin testleri yazılır.
4. **Sınırlar:** katmanlama ışın başına 16 aktif SDF hacmi tarıyor; domain
   başına hacim sayısı artınca bu tavan ve VRAM ölçülmeli.
5. **Köpüğün kanal ödünç alması:** Volume köpüğü SDF hacminin `temperature`
   kanalına biniyor (`FOAM_TEMP_SCALE`). Sıcaklık artık gerçek fizik verisi
   taşımaya başladı; iki anlam aynı kanalda çakışmamalı.
6. **Materyal envanteri (§4.6):** görünüm başına bugün hangi materyal alanı
   okunuyor, miras var mı, splat'te instance başına veri var mı; partikül
   sisteminin `Appearance Profile`'ı ile madde profilinin ilişkisi.
7. **OptiX dondurulmuş:** yeni katman kombinasyonları yalnız Vulkan'da. OptiX
   bugünkü davranışını korur; farkın kullanıcıya nasıl gösterileceği karar
   ister.

**UI yaklaşımı (öneri, tasarlanmadı):** panel modelden türer, kendi durumu
yoktur (CLAUDE.md §1). Domain paneli bir **içerik listesi** olur: domain'de
hangi fazlar ve maddeler var, kaçar parçacık/hücre. Her madde satırında faz,
materyal ve etiket başına görünüm. Çözücü ayarları yalnız o faz VARSA açılır.
"Now drawing" satırı görünüm çözücüsünün listesini okur. Kod yazmadan önce
tıklanabilir bir mock (artifact) ile üzerinde anlaşılır; her panel alanının
IPC karşılığı mock'ta işaretlenir.

---

## 8c. Faz 2-W: whitewater — görünümde TEK otorite, depoda AYRI (2026-09-28)

**Kullanıcı kararı (revize, 2026-09-28):** Hedef tam akışkan fiziği (gaz, sıvı,
granül). Whitewater bu hedefin parçası DEĞİL, çözülemeyen ölçeğin yer tutucusu.
Bu yüzden birleşme **görünüm katmanında** yapılır, depoda yapılmaz. İlk plan
("tam birleşme": kütlesiz parçacıkları `FluidParticles`'a taşımak) iptal.

**Neden (tartışmanın özeti):**
- "spray" iki farklı şeyin adıydı:
  - **Etiketli birincil parcel:** kütleli, çözücünün taşıdığı su. Etiket fiziği
    sınıflar, üretmez.
  - **Whitewater (Ihmsen):** kütlesiz, tek yönlü. Izgaranın çözemediği ölçeğin
    istatistiksel vekili (hapsolmuş hava, dalga tepesi, kinetik enerji).
- Ölçüm bunu doğruladı: kırılan dalgada birincil spray 0–3 parçacık. İnce jet
  voksel altına inince basınç alanı yok (bkz. "su sıçramıyor" bulgusu), beyaz
  suyu whitewater üretiyor. İkisi aynı işi değil, farklı ölçekleri yapıyor.
- Depo birleşmesi "spray" etiketini bazen kütleli bazen kütlesiz yapardı. P2G,
  basınç, kütle sayımı, kota ve cache'in her biri "bu gerçek mi?" diye sormak
  zorunda kalırdı. Bunun için bir determinizm kapısı tasarlanmış olması,
  tasarımın yanlış yöne gittiğinin işaretiydi.

**Gerçek fizik yolu (uzun vade, ayrı iş):**
- **Kabarcık** = sıvı içindeki hava = iki fazlı akış. Doğal evi birleşik madde
  domain'i (gaz + sıvı aynı alanda, §4.4 fazlar arası alışveriş). O gelince
  whitewater bubble'ın yerini alır.
- **Spray** = çözünürlük / ince jet çözümü.
- O güne kadar whitewater dürüst bir yer tutucu: panel ve API onu "massless,
  subgrid stand-in" diye anlatır, fizik gibi sunmaz.

**Fazlar:**
- **W0 — ön koşullar (19. parti, derlendi ve IPC'de doğrulandı).** `fluid.get/set_whitewater`,
  `set_param max_particles`, `fluid.state_digest`, determinizm probu
  (`rt_probe_whitewater_determinism_ipc.py`). Depo birleşmesi iptal olsa da prob
  geçerli: whitewater ana suyu etkiliyorsa bu bir sızıntıdır.
  - GPU etiket fizibilitesi: EVET. `sim_foam_bin_scatter` bin'leri tamsayı
    atomikleriyle kuruyor; etiket yarıçapı 1,5 voksel = 27 hücre stencil. Yeni
    `sim_fluid_label_count.comp` + n baytlık geri okuma (100k ≈ 100 KB);
    histerezis host'ta. Ön koşul: bin koşulsuz kurulmalı. (Ayrı iş, park.)
- **W1 — tek otorite (20. parti, derlendi ve IPC'de doğrulandı).** Yapıldı:
  - `FoamRenderMode` ve `FoamParams::render_mode` söküldü. Whitewater tipi
    (`secondaryParticleLabel`) aynı label routes ile çözülüyor:
    `FluidViewPlan::viewForWhitewater`, `whitewaterTypesIn`,
    `countWhitewaterPerView`.
  - Görünüm anlamı: **splat** = tip başına küre instance (havuz yalnız splat'e
    yönlenen partikülleri tutar); **sdf** = yüzey hacminin sıcaklık kanalında
    beyaz ortam (eski "Volume"); **fog** = domain fog yoğunluğuna eklenir
    (1 partikül = `volume_density` parcel); **hidden** = çizilmez. Follow =
    untagged girişin görünümü (whitewater'ın maddesi yok).
  - Panel: Render combo yerine "Spray: …, foam: …, bubbles: …" satırı; görünüm
    kontrolleri kullanılan görünüme göre. "Now drawing" satırları
    "N particles + M whitewater", Hidden satırı ikisini ayrı sayar.
  - API: `fluid.get_whitewater` → `views {spray, foam, bubble}` +
    `stats.in_sdf/in_splat/in_fog/hidden`; `fluid.get` → `views[].whitewater`;
    `set_whitewater render_mode` reddedilir.
  - Ölü metaball köpük yüzeyi (`FoamSurface.cpp/.h`, `surface_*` alanları)
    söküldü; çağıranı yoktu.
  - ProjectManager `volume_color/opacity/bubble/spray_strength` kaydetmiyordu
    (yalnız SceneSerializer); eklendi.
  - **Eski sahne:** `render_mode` artık okunmuyor. Varsayılan tablo whitewater'ı
    splat'e gönderir. Eski "Volume" görünümü isteyen sahne foam/bubble'ı "sdf"e
    yönlendirir. Göç yok (geriye uyum yükü yok kuralı), alan adı gitti, anlam
    sessizce değişmedi.
  - Test: `rt_test_whitewater_label_routes_ipc.py`.
- **W2 — İPTAL** (depo birleşmesi).
- **W3 — küçüldü:** yalnız ölü yol sökümü; W1 içinde yapıldı. Kalan: yok.

**Zorunlu yol haritası tamamlandı:** 27D ve Faz 4 domain birleşmesi canlı
kabulden geçti. Tek Matter domain aynı adım zincirinde 120 sıvı parçacığı ve
67 aktif gaz hücresi üretti. Ortak madde özellik tablosu, dönüşüm defteri, yanıcı APIC→gaz korunumu
ve gerçek mist→gaz aktarımı 27C’ye kadar tamamlandı ve canlı kabulden geçti.
GPU etiketleme performans kapısı 23. partide geçti. Kapalı tank korunumu 21.
partide geçti.
Determinizm 22. partide CPU/Vulkan olarak ayrıldı; strict tekrar için CPU yolu
var, Vulkan float-atomik P2G toleranslıdır. RT bloklaşması kullanıcı kararıyla
ertelendi. Faz 5 güçlü iki-fazlı projeksiyon isteğe bağlıdır.

## 8d. Tek-domain çekirdeği kapanış planı (2026-10-01)

Tek kimlik ve gaz/sıvı dönüşüm yolu çalışıyor; fakat bu henüz tam madde sistemi
değildir. Mevcut `granular_enabled` domain'in bütün APIC parçacıklarını aynı
constitutive modele geçirir. Bu nedenle aynı domain'de su ve kum doğru biçimde
birlikte çözülemez. `WetSand` de su emmiş kum değil, sabit cohesion değerli bir
presettir. Kapanış hem birleşmenin yapısal maliyetini kaldırmalı hem de serbest
su ile granül iskeleti aynı Matter adımında korunumlu bağlamalıdır.

### C4 için bağlayıcı veri modeli

`Substance`, fizik ile görünümü bir araya getiren otoriter **madde kimliğidir**;
çözücü sınıfının kendisi değildir. Tanım yoğunluk, viskozite, yüzey gerilimi,
porosity/permeability, granular parametreler, termal ve kimyasal özellikler ile
varsayılan görünüm bağını taşır. Anlık parçacık/hücre durumu ayrıca tutulur:
termodinamik faz (`gas/liquid/solid`), constitutive rejim
(`fluid/granular/elastic/...`), sıcaklık, bileşim ve doygunluk.

Emitter'ın kanonik çıktısı bir `MatterDeposit` paketidir:

- `substance_id` ve gerekirse bileşim/species oranları,
- kg cinsinden kütle, hız ve sıcaklık/enerji,
- substance varsayılanından farklıysa başlangıç durum override'ı,
- hedef Matter domain kimliği ve uzamsal dağılım.

Emitter doğrudan “APIC”, “Euler” veya “MPM” seçmez. Domain her adımda anlık
durumu gruplar ve solver registry üzerinden sıvıyı APIC/FLIP'e, gazı Euler'e,
granül iskeleti Drucker–Prager MPM'e yollar. Aynı substance ısı veya reaksiyonla
durum değiştirdiğinde emitter'a dönmeden başka çözücü yoluna geçebilir. Kimya
da emitter'ın “chemistry” modu değildir: emitter bileşim yatırır; domain'in
reaction/phase-transition kuralları ürün, ısı ve yeni durumu ledger üzerinden
üretir.

UI bunun doğrudan görünümüdür:

1. Emitter satırında önce **Substance**, sonra **Initial State** bulunur. Initial
   State varsayılan olarak `From Substance`tır; ileri kullanıcı gas/liquid veya
   constitutive başlangıcını yalnız geçerli seçenekler arasından override eder.
   Ayrıntılı kimya gerektiğinde emitter **Composition/Mixture**, saflık veya
   konsantrasyon ve başlangıç sıcaklığı taşır. Reaksiyon sabitleri emitter'a
   kopyalanmaz; **Edit Substance…** merkezi tanımı, **Reaction Rules** ise domain
   içindeki reaksiyon/ürün sözleşmesini açar.
2. Matter Domain paneli tek bir gas/liquid/granular modu göstermez. **Active
   Matter** tablosu substance, durum, kütle ve çalışan solver'ı salt-okunur
   raporlar; **Solver Policy** faza/rejime özgü kalite ve grid ayarlarını taşır.
3. Substance görünüm materyalini ve fizik özelliklerini aynı kimlikle bağlar,
   fakat görünüm physics state'i değiştirmez. Islak kum görünümü doygunluğu okur.
4. Node sistemi aynı `MatterDeposit` ve reaction sözleşmesini kullanır. Gelecekte
   birleşik domain'e taşınırken UI, IPC ve node için ikinci bir iş mantığı yazılmaz.

Bu ayrım, “Sand” adlı substance'ın kuru granül başlayıp su emerek nemli/doygun
granüle dönüşmesini; “H2O”nun liquid başlayıp mist/gas durumuna geçmesini aynı
domain kimliğinde ifade eder. `granular_enabled` yalnız eski sahne/preset göçü
için geçici uyumluluk girdisi olabilir; yeni sahnelerde fizik otoritesi olamaz.

### Kapanış kapsamı

Tek-domain çekirdeği şu koşullarda **tamamlandı** sayılır:

1. Tek mantıksal Matter kimliği gaz ve sıvıyı aynı dönüşüm defteri içinde taşır;
   her fazın alanı, GPU buffer'ı, fiziksel bounds'u ve voxel boyutu bağımsızdır.
2. Olmayan faz çözücü, tahsis, tam-grid tarama veya render slotu maliyeti üretmez.
3. Yalnız gaz taşıyan Matter, eşdeğer Gas domain'inden; yalnız sıvı taşıyan
   Matter, eşdeğer Fluid domain'inden 12 ısınmış karenin medyanında %10'dan
   fazla yavaş olmaz.
4. İki aktif fazlı Matter'ın sim süresi, aynı sahnenin ayrı Gas + Fluid toplamını
   %10'dan fazla aşmaz. Bu kapı fizik sonucunu ucuzlatmak için kalite düşürerek
   geçilemez; voxel, basınç iterasyonu, parçacık ve kaynak sayıları eşit tutulur.
5. Kütle/enerji ledger hatası mevcut toleranslarda kalır; kapalı tank korunumu,
   CPU/Vulkan determinizm politikası, SDF + gaz slot kimliği ve cache geri dönüşü
   mevcut testlerden geçer.
6. Burning Fuel Spill ve Ignited Fuel Jet tek Matter reçetesiyle çalışır.
   Flamethrower saf gaz reçetesi tek gaz fazı olarak kalabilir; yardımcı sıvı
   domain üretmez. Eski sahne yükleme yolu korunur, yeni karma presetler ayrık
   Gas + Fluid çifti üretmez.
7. Aynı Matter domain'de sıvı ve granül parçacıklar kendi constitutive modeliyle
   aynı zaman adımında çözülür. Domain-geneli `granular_enabled`, parçacığın su
   mu kum mu olduğuna karar veren fizik otoritesi değildir.
8. Kuma emilen su, serbest sıvıdan kg cinsinden düşülüp granül gözenek suyuna
   eklenir. Drenaj ters kaydı üretir; `MatterExchangeLedger` ve toplam envanter
   kapalı sahnede tolerans içinde kütle korur.
9. Doygunluk kumun efektif gerilme, sürtünme, kohezyon, dilatasyon, yoğunluk ve
   geçirgenliğini sürer. Kısmi doygunlukta kapiler kohezyon artabilir; yüksek
   doygunlukta gözenek basıncı iskeleti zayıflatabilir. Tek yönlü, her koşulda
   “daha ıslak = daha yapışkan” eğrisi kabul edilmez.
10. Görünüm aynı otoriter doygunluk alanını okur; ıslak renk/roughness karışımı
    render tarafında yeniden tahmin edilmez. Cache, serializer, UI, scripting ve
    IPC aynı su-kum durumunu taşıyıp raporlar.

### Kapanış partileri ve güncel durum

| Parti | Kod işi | Ölçülebilir kapı |
|---|---|---|
| **C0 ✔ — sayaç kabulü** | Lazy gaz kg/J sidecar, aktif-support advection ve gerçek sıvı-domain `total_ms` | Canlı PASS; sayaçlar sonraki partilerin performans temelidir |
| **C1 ✔ — sıvı residency** | G2P velocity+affine ile hemen sonraki device advect-tail position+velocity geri okumalarını tek submit/readback sınırında birleştir; device-tail başarısızlığında doğru host fallback | Canlı PASS: 100k sıvıda 1.200.000 byte tekrar velocity indirmesi kaldırıldı; 9.430.584 byte, 12 batch; digest farkı 1e-9 altı |
| **G1 ✔ — granüler residency/occupancy ara turu** | Granüler konum/velocity/affine/state'i alt adımlar arasında cihazda tut; weighted fluid mask'i resident konumdan GPU'da üret; UVW yenileme ve kare sonu host yayınını koru; uzun zinciri descriptor kapasitesinde güvenli gönder | 991.666 parçacıkta canlı PASS: 48/48, Young 381300 Pa, invalid=0; toplam −%33,20, upload −%68,83, download −%72,93. Ölçüm [GRANULAR_GPU_OCCUPANCY.md](GRANULAR_GPU_OCCUPANCY.md)'de |
| **G2 — granüler tane temsili ve yük davranışı** | Granülerde kanonik görünümü bağımsız sphere/splat olarak koru. Tam küp olmayan PPC değerlerinde tekrar eden alt-kafes desenini kır; fizik konumuna render jitter ekleme. Sphere temel yarıçapını voxel, PPC ve hedef görünür paketleme oranından türet; mevcut radius/size değeri bu fiziksel tabanın sanatçı çarpanı olur. Görsel sphere'in DEM temas küresi olmadığını açık tut: MPM malzeme noktalarının fiziksel doluluğu grid ağırlıkları ve constitutive stress'tir. Sand iskeleti kendi overburden yükünü küçük-strain aralığında taşımalı. SDF/isosurface sıvı ve gerçekten bütünleşik yüzey isteyen maddelerde kalır | 4/8 PPC seed'lerinde eksen/yarım-hücre yanlılığı yoktur. PPC veya voxel değişince görünür paketleme oranı sabit kalır; durgun ve dökülen kumda büyük yapay iç boşluk, voxel kafesi ve molekül zinciri görülmez. Solid/Material/Rendered aynı sphere çapını gösterir. İki yığın yüksekliğinde kolon çökmesi, sıkışma ve angle-of-repose çözünürlük yakınsaması geçer; `stiffness_below_load` normal Sand ölçeğinde false olur ve kazanım aşırı settle damping ile üretilmez |
| **C2 — aktif sıvı çalışma penceresi** | Parçacık AABB + CFL/pressure halo üret; P2G clear/normalize, fluid mask ve MGPCG yalnız bu hücre aralığında çalışır; tam grid fallback kalır | `active_liquid_cells/full_cells` IPC'de görünür; yerel sıvıda dispatch edilen hücre sayısı küçülür; sınır/çarpışma ve açık outflow testleri aynı sonucu verir |
| **C3 — faz grid sözleşmesi** | Tek descriptor altında `gas_bounds/voxel` ve `liquid_bounds/voxel`; core + UI + scripting + IPC + serializer/cache aynı servisi kullanır. Varsayılan mantıksal bounds'a düşer; presetler eski gaz/sıvı kutularını faz alanlarına taşır | Geniş/kaba gaz ve dar/ince sıvı aynı Matter domain'de raporlanır; kaynak, hareketli sınır ve render koordinatları doğru eşlenir; bütçe hesabı faz başına gerçek hücre sayısını kullanır |
| **C4 — çok malzemeli MPM çekirdeği** | `substance_tag`/madde özelliklerinden parçacık başına liquid veya granular constitutive model seç; ortak zaman adımında faz başına kütle/momentum P2G ve iki yönlü grid contact/drag çöz. Granülü sıvı pressure projection'a, suyu Drucker–Prager gerilmesine sokma | Aynı kutudaki su ve kuru kum eşzamanlı ilerler; her iki sınıfın parçacık/kütle/momentum sayaçları IPC'de ayrıdır; ayrı çalıştırılan su ve kum referansından sapma tanımlı toleranstadır |
| **C5 — emilim ve drenaj** | Granül parçacığa `pore_water_mass_kg`, porosity/capacity ve saturation sidecar'ı ekle. Temas alanı, geçirgenlik, basınç farkı ve dt ile sınırlı emilim; kapasite üst sınırı; yerçekimi/basınçla drenaj ve serbest suya geri dönüş. Her transfer tek ledger kaydıdır | Kapalı su+kum testinde serbest su kaybı = gözenek suyu kazancı; doygunluk 0..1; kuru kum uzakta kuru kalır; doygun kum drenajla ölçülebilir su geri verir |
| **C6 — ıslak kum fiziği ve görünümü** | Doygunluktan efektif stress/pore pressure, friction, cohesion, dilatancy ve kütle türet; aynı alanı kuru/ıslak materyal karışımına bağla. Core servisini UI/API/IPC/Python, serializer ve cache yüzeylerine taşı | Kuru, nemli ve doygun kumun angle-of-repose/çökme sonucu farklıdır; nemli bölgede ıslak görünüm mekânsal olarak fizik ile çakışır; cache dönüşünde doygunluk ve görüntü korunur |
| **C7 — son kabul ve kesim** | Üçlü performans A/B matrisi ile su-kum fizik matrisini çalıştır; preset göçü, cache/ledger/render regresyonu; geçici uyumluluk dallarını ve bayat plan metnini temizle | %10 tek-faz/iki-faz kapıları geçer. Su jeti/kum yatağı, ıslanma cephesi, drenaj, kuru-nemli-doygun yığın ve kapalı kütle testleri PASS; belge `TAMAMLANDI` olur |

C0, C1 ve G1 tamamlandı. Kalan sıra G2 fizik ölçümleri → C2 → C3 → C4 → H1 → C5 → C6 → C7'dir.
C2–C3'ten bağımsız olan G2 önce kapatılır; granüler görünümün fizik davranışıyla
karıştırılmasını önleyen kısa bir doğruluk partisidir. Granüler için SDF üretmez.
C2–C3 ayrı derlenebilir altyapı partileridir; C4–C6 tam madde fiziğinin uygulama
partileridir. C7 kabul ve temizlik turudur; yalnız kabulün gösterdiği zorunlu
düzeltmeler dışında yeni özellik eklemez.

### Kalan işlerin uygulama sırası (2026-10-02)

1. **G2 — sphere tane dağılımı, görünüm eşliği ve yük desteği.** `seedBox` içindeki alt-kafes
   yuvaları tam küp olmayan PPC değerlerinde hücreden hücreye aynı öneki
   kullanmaz; deterministik hücre anahtarıyla dengeli seçilir. Rastgelelik yalnız
   seed anında fizik parçacık konumuna uygulanır, renderer parçacığı her kare
   oynatmaz. Taneler bağımsız sphere kalır; otomatik SDF, bağ veya birleşik bulk
   mesh üretilmez. Aynı yarıçap/konum sözleşmesi Solid, Material Preview ve RT'de
   kullanılır; çekirdek ayar veya telemetri eklenirse UI, scripting ve IPC birlikte
   teslim edilir. Ardından aynı Sand malzemesi iki kolon yüksekliği ve iki voxel
   çözünürlüğünde ölçülür. Young modülü yalnız görünüş için yükseltilmez: pile
   overburden, elastik strain, gerekli substep, kalıcı sıkışma, runout ve repose
   birlikte kaydedilir. Ölçekten bağımsız doğru sonuç veriyorsa preset yükseltilir;
   sahne derinliğine bağlı destek gerekiyorsa açık bir load-support politikası
   olur ve requested/effective değerleri UI, scripting ve IPC'de ayrı raporlanır.
2. **C2 — aktif sıvı çalışma penceresi.** Önce parçacık AABB'sinden deterministik
   hücre aralığı ve halo servisi çıkarılır. Clear/normalize, occupancy/mask ve
   basınç grid dispatch'leri bu pencereyi kullanır; sınır, collider veya sayısal
   doğrulama başarısızsa tam-grid fallback açıkça raporlanır. Core operasyonu
   scripting ve IPC'de aynı sayaçları verir. Kabul edilmeden C3'e geçilmez.
3. **C3 — faz grid sözleşmesi.** Mantıksal domain bounds'u ile gaz/sıvı fizik
   gridlerinin bounds/voxel/origin'i ayrılır. Tahsis, kaynak koordinat dönüşümü,
   collider damgası, hareketli sınır, render ve cache aynı faz-grid servisini
   kullanır. Olmayan faz hiçbir buffer veya tam-grid işi üretmez. C2 penceresi
   sıvı faz gridinin koordinatlarında tanımlanır; domain-geneli eski indeksler
   otorite olarak kalmaz.
4. **C4 — karma constitutive çekirdek ve ortak transfer tasarımı.** Parçacık
   başına constitutive kimlik, aktif indeks/range listeleri ve ayrı kütle/momentum
   akümülatörleri kurulur. Liquid P2G ile granular stress-P2G'nin aynı parçacık
   komşuluğunu tekrar tekrar gezmesi burada ele alınır: ortak stencil/bin verisi
   paylaşılır veya ölçüm daha iyi olduğunu gösterirse model başına kompakt liste
   dispatch edilir. Su pressure projection'a, kum Drucker–Prager güncellemesine
   gider; grid contact/drag iki yönlü momentum alışverişini çözer.
5. **C5 — korunumlu emilim/drenaj.** Serbest su ile `pore_water_mass_kg`
   birbirinden ayrı tutulur. Transfer miktarı porosity/capacity, permeability,
   temas, basınç ve dt ile sınırlanır; her yön ledger'a tek olay yazar. Önce kapalı
   küçük testte kg eşitliği geçer, sonra su jeti/kum yatağı senaryosuna geçilir.
6. **C6 — doygunluğun fizik ve görünüm otoritesi olması.** Saturation'dan pore
   pressure/effective stress, sürtünme, kohezyon, dilatasyon ve kütle türetilir.
   Aynı alan materyal karışımını sürer. Core servis, serializer/cache, UI,
   scripting ve IPC birlikte teslim edilir; render kendi ıslaklık tahminini yapmaz.
7. **C7 — kabul, performans ve söküm.** Tek su, tek kum ve karma su-kum için aynı
   çözünürlük/parçacık/alt-adım ayarlarında A/B matrisi çalıştırılır. C4 sonrası
   GPU timestamps ile P2G scatter, stress-P2G, G2P, contact ve occupancy yeniden
   ölçülür. `%10` kapanış kapıları, korunum, cache/scrub-resume, eski sahne/preset
   göçü ve render regresyonları geçince `granular_enabled` fizik otoritesi ve
   geçici uyumluluk dalları sökülür.

### Scatter maliyeti için karar kaydı

G1 canlı ölçümünde toplam GPU süresinin büyük bölümü `sim_fluid_p2g_scatter`
(979,39 ms) ve `sim_fluid_granular_stress_p2g` (360,40 ms) oldu. Constitutive
stress update 26,33 ms, occupancy 13,57 ms idi. Bu yüzden sonraki performans
hedefi stress matematiğini gevşetmek veya fizik alt adımını azaltmak değildir.

Scatter'a bugünkü iki ayrı shader üzerinde açık uçlu bir tur yapılmaz. C2 önce
grid kapsamını küçültür; C4 ise karma su-kum için hangi komşuluk, akümülatör ve
contact verisinin ortak olduğunu belirler. Optimizasyon C4'ün kanonik transfer
çekirdeğinde yapılır ve C7'de şu kapılarla kabul edilir:

- 48/48 alt adım ve authored stiffness korunur; kalite azaltımı hızlanma sayılmaz.
- Tek-kum sonucu G1 fizik referansından, tek-su sonucu C1 referansından tanımlı
  tolerans dışında sapmaz.
- Ortak stencil/bin belleği parçacık ve aktif hücre sayısıyla raporlanır; her
  kare kontrolsüz büyümez ve olmayan model için ayrılmaz.
- GPU timestamp tablosu gerçek shader sürelerini, transfer sayaçları host/device
  trafiğini, toplam süre ise submit/fence maliyetini birlikte gösterir.
- Birleştirilmiş traversal ölçümde daha yavaşsa iki kompakt model listesine
  dönülür; ortak servis ve contact sözleşmesi korunur.

Bu sıra, pahalı döngüyü C4'te yeniden yazmadan önce geçici olarak hızlandırıp
sonra ikinci kez taşıma riskini önler. G1 ölçümleri C2/C4/C7 için sabit referanstır.

### Bilerek kapanış dışında

- Gaz-sıvı güçlü tek basınç projeksiyonu ve fiziksel kabarcık çözümü. Granül-sıvı
  contact, emilim ve gözenek basıncı bunun dışında değildir; C4–C6'da zorunludur.
- RT SDF bloklaşması, splat materyal editörü ve Solid mod splat/gaz sırası.
- OptiX'e yeni Matter katmanlarının taşınması.
- Whitewater deposunu ana APIC parçacıklarıyla birleştirmek; W2 iptal kararı sürer.

Bu işler tek-domain çekirdeğini yeniden açmaz; kendi plan ve kabul kapılarıyla
ilerler.

## 9. ★ Sinsi başarısızlıklar (şimdiden işaretli)

- **Etiket render'da hesaplanırsa** her şey çalışıyor görünür, ama panel,
  API ve görüntü farklı sınıflandırmalar gösterir. Kural §3.1.
- **Kütle sessizce kaybolur:** `mist` sıvıdan silinir ama gaza eklenmezse sis
  yine görünür (yayılmış fog), sadece sıvı yavaş yavaş azalır. Korunum sayacı
  olmadan kimse fark etmez.
- **Sayaç sıfırlama/yeniden doldurma kapsamı** (2026-09-27 fog hatası): bir
  faz için sıfırlanan her istatistik aynı faz için yeniden doldurulmalı.
- **★★ SDF'ye sıcaklık yüklemek köpük çizer:** shader SDF hacminde
  `vdb_temp_address != 0` gördüğünde onu köpük sayar. Erimiş madde Kelvin'i o
  kanala girerse ekranda soluk beyaz bir örtü çıkar, parıltı çıkmaz — "köpük
  açık kalmış" gibi görünür, kimse bug demez (§8b 1. tur, madde 4).
- **★★★ Ortak alan, iki çözücü:** gaz ve sıvı aynı `density`/`vel_*`
  vektörüne yazarsa hiçbir şey çökmez; ikinci çözücü birincinin alanını
  "başlangıç koşulu" sanıp devam eder. Belirti fizik gibi görünür (garip
  türbülans, kaybolan duman). Her faz kendi alanını taşır.
- **Varsayılan ölçüm değildir:** etiket sayısı 0 dönerse "hiç sprey yok" mu,
  "sınıflandırma çalışmadı" mı — `*_measured` bayrağı şart.

## Prosedürel küre yolu — açık işler (2026-10-02)

- Küre bulutu tek AABB BLAS; hit shader artık `closesthit.rchit -DSPHERE_HIT=1` (tam materyal yolu). UV/tangent YOK (normal map etkisiz) — bilinçli kabul.
- Beklenen bellek/CPU kazancı gelmedi: `group.instances` (InstanceTransform, 40 B/parçacık) hâlâ tam boyutta; köprü yazıyor, `gatherSphereCloud` bir kez daha okuyor; raster + RayFusion ikosfer kopyasını kullanıyor; OptiX da `group.instances`'ı okuyor.
- Sıra: (1) ölçüm — iki yol (point_sphere_mode açık/kapalı) kare süresi + VRAM, (2) köprüden doğrudan kompakt (konum, yarıçap, materyal) dizisi — Vulkan RT + OptiX + raster üç tüketiciyi birlikte taşımadan `group.instances`'ı silme, (3) RayFusion/raster'ı kompakt diziye bağla.
- Ölçüm bayrağı henüz IPC'de yok (4 dokunuş kuralı).

### Ölçüm (2026-10-02, 87.552 parçacık, Rendered, Vulkan)

`fluid.set_splat_geometry geometry=icosphere_mesh` ↔ `icosphere`, `perf.get_gpu_memory`, tekrarlanabilir (2 tur):
render cihazı device-local 403 → 373 MB (−31 MB ≈ 350 B/parçacık); host aynı (215 → 217 MB); süreç private 3.64 → 3.61 GB.
Sonuç: prosedürel yol doğru ama kazanç parçacıkla doğrusal ve küçük (1M parçacıkta ≈ 350 MB). Süreç belleğinin büyük kısmı
splat'tan bağımsız (render target ~360 MB, host "other" ~217 MB, işlem private 3.6 GB). Köprü refaktörü (group.instances
kaldırma) bellek için DEĞMEZ: 40 B × 87k = 3.5 MB. Bellek avı için splat dışına bakılmalı.

## Devir notu — sim frame cache sıkıştırma (2026-10-02)

Yapıldı + ölçüldü (991k granüler parçacık, 250 kare): cache 24.3 GB (sıvı) / 17.4 GB (granüler) → 8.1 GB (33 MB/kare);
süreç ~10-12 GB. Bütçe tavanı artık makineden: min(RAM%70, RAM-2GiB), ayrıca boş RAM<2 GiB iken yakalama durur
(`simFrameCacheBudgetFromHardware`, scene_ui.cpp). Ara karelerde (25'in katı olmayan): afin atılır; konum/hız/uvw 16-bit
kuantize; düzgün skalerler tek değer; granülerde çözücü-içi tensörler atılır (kod: scene_data.h compress/decompress +
SimFrameCompress.h QuantVec3/ConstScalar). Anahtar kareler (25 kat) tam hassasiyet. Doğrulama: damage/detached
anahtar (50) ↔ ara kare (49,51) aynı (granular_max_damage .9442).

Sıradaki ajan için (öncelik sırasıyla):
1. Ara kareye scrub + OYNATMA testi: çözücü ara kareden değil anahtar kareden devam etmeli (`simLiveVelocityValid` false).
   Henüz kimse ara kareden ileri oynatıp yığının davranışını karşılaştırmadı — sessiz hata adayı (kimlik deformasyonla devam).
2. UI'da "Frame cache RAM sınırı" yüzdesi (şu an otomatik %70/-2GB); sim_cache IPC alanı + panel (4 dokunuş kuralı).
3. Kalan ~33 B/parçacık: konum/hız/uvw 18 B + değişen skalerler. Sonraki adım kareler arası delta — riskli, gerek yoksa yapma.
4. RayFusion'da aynı-cihaz Vulkan granular/splat domain artık konum buffer'ından
   procedural sphere impostor çeker. CPU/CUDA ve karışık label/substance
   fallback'i hâlâ üçgen ikosfer havuzunu kullanır; gölge/prepass eşliği açıktır.
5. Prosedürel kürede UV/normal map yok (bilinçli). A/B: fluid.set_splat_geometry icosphere ↔ icosphere_mesh.


## Granular CFL / particle residency follow-up (2026-10-02)

Source changes complete; build/live acceptance pending. The 32-substep truncation
no longer lowers authored Young modulus: the CPU/GPU planner grants full wave/strain
CFL requests. Closed Vulkan granular intermediate steps keep velocity/affine on
device and reuse P2G particle streams; positions still return for occupancy and
material-coordinate refresh. The legacy max-substep field remains serialized/API
compatible but no longer limits physical resolution. G2P timing can shift into
Advect because its GPU work is flushed there; judge total time and transfer bytes.

Exact baseline, changes and acceptance commands:
[GRANULAR_SUBSTEP_RESIDENCY.md](GRANULAR_SUBSTEP_RESIDENCY.md).
The frame-cache resume test and cache RAM percentage work above remain open.

## GPU render proxy başlangıcı — Vulkan RT + RayFusion (2026-10-03)

Fizik taşıyıcısı ile görünen örnek ayrıldı. Prosedürel sphere kullanan bir APIC
taşıyıcısı kullanıcı ayarlı sayıda kararlı görsel örneğe açılır; çocuk yarıçapı
`r / cbrt(N)` olduğu için
toplam küre hacmi taşıyıcının temsil ettiği hacmi korur. Çocuklar çözücüye,
komşuluk aramasına, P2G/G2P'ye veya çarpışmaya geri yazılmaz.

- RayFusion/raster aynı Vulkan cihazındaki `fluid_positions` buffer'ını storage
  buffer olarak doğrudan çeker. Çocuk merkezi vertex shader'da
  `gl_InstanceIndex` ile türetilir; CPU child listesi ve kare başına konum
  upload'u yoktur.
- Vulkan RT aynı düzeni birleşik procedural-sphere BLAS içinde kullanır. Çocuklar
  ayrı TLAS instance'ları değildir; yalnız sphere record + AABB sayısı artar.
  Domain başına 1..32 milyon kullanıcı bütçesi ve `Max Particles`, kareler
  boyunca değişmeyen etkin çocuk sayısını belirler. Panel gerçek canlı sayı için
  RT proxy girdilerinin yaklaşık belleğini ayrıca gösterir.
- Aynı çoğaltma bütün prosedürel fluid particle sphere yollarında kullanılabilir.
  Granüler hacim tabanı yalnız granular malzemeye özgüdür. Surface SDF/fog
  kaynakları kendi görünüm yolunda kalır.
- Karışık substance/label görünümünde RayFusion'ın doğrudan pull yolu devreye
  girmez; parcel filtrelemesinin doğruluğu için mevcut CPU filtreli köprü
  korunur.
- OptiX bu partinin dışında bırakıldı. Önce aynı yerleşimi doğrulayacak CPU
  referansı ve karşılaştırma testi kurulacak; ardından OptiX GAS üretimi bu
  referansa bağlanacak.

Mevcut proxy çocukları yaklaşık 0,72 voxel desteğine yayar ve küçük, kararlı bir
yakın taşıyıcı aday penceresini uzaklık/yön çekirdeğiyle ortalar. Aynı yol spray,
foam ve bubble sphere gruplarına da uygulanır. Son kalite aşaması bu aday
penceresini gerçek hücre hash'i ve APIC grid hızlarıyla değiştirecek; böylece
örnek kimliği ve hız taşınması tam grid komşuluğuna bağlanacaktır.

Yüksek çözünürlüklü granular görünümde varsayılan yol fizik taşıyıcılarını bire
bir sphere olarak çizer. Böylece birkaç görsel çocuğun tek MPM taşıyıcısına bağlı
hareket ederek moleküler küme gibi görünmesi engellenir. Sanal çocuk modu düşük
çözünürlükte boşluk doldurmak için isteğe bağlı kalır. Bu seçim ayrık eleman
teması veya tane açısal hızı eklemez; gerçek tane-tane yuvarlanma fiziği ayrı bir
DEM/hybrid solver aşamasıdır.

### G2 canlı teşhis ve hibrit tane sınırı (2026-10-03)

Kullanıcının 298³, 0,01678 m voxel sahnesinde 191.666 fizik taşıyıcısı vardı.
Eski çalıştırılabilir dosya, 8 istenen görsel çocuğu 32 milyonluk kapasite
bütçesi nedeniyle 3'e indiriyor ve 574.998 sphere çiziyordu. Çocuklar aynı MPM
taşıyıcısının konum/hızını paylaştığından çarpışmadan sonra üçlü molekül kümeleri
gibi hareket ediyordu. Canlı A/B'de çocuk sayısını 1 yapmak bu kümeyi kaldırdı;
bu nedenle granular ana gövdede `Physical Granular Carriers` varsayılanı açık
kalır. Whitewater ve düşük çözünürlüklü önizleme sanal çocukları kullanabilir.

Aynı sahnenin mekanik teşhisi ayrıca başarısızdır: Sand'in authored Young değeri
200.000 Pa iken 2,84 m malzeme kolonunun küçük-strain sınırında istediği değer
446.058 Pa'dır. `granular_stiffness_below_load=true`, 190.158/191.666 parçacık
yielded ve 112.956 parçacık detached ölçülmüştür. Bu, yalnız render kusuru
değildir; yığın kendi yükünü mevcut constitutive aralıkta taşıyamaz. Değeri bu
sahne için yaklaşık 500 kPa'a çıkarmak yük kapısını kapatır, fakat aynı voxel ve
1/24 s adımda wave substep sayısını yaklaşık 80'den 127'ye taşır. Preset bu
ölçüm uğruna küresel olarak sertleştirilmez; G2 kabulü farklı kolon yükseklikleri
ve toplam süre ile birlikte yapılır.

MPM taşıyıcısı bağımsız bir kum tanesi değildir. Konum, doğrusal hız, affine hız
alanı ve gerilme taşır; tane orientasyonu, açısal hız, tane-tane temas manifoldu
ve rolling resistance taşımaz. Bu yüzden yalnız sphere sayısını artırmak gerçek
tekil yuvarlanma üretemez. Gerçek tekil yuvarlanma için C4 sonrasındaki H1
partisinin sınırlı hibrit yolu aşağıdadır; G2'nin MPM doğruluk kabulünden ayrıdır:

1. Grid occupancy'den serbest yüzey ve kopmuş granular taşıyıcıları GPU'da kompakt
   bir aktif tane listesine çıkar. İç yığın MPM olarak kalır.
2. Aktif taneler için yarıçap, orientasyon, açısal hız ve atalet tut; hücre hash'i
   üzerinden normal temas, Coulomb sürtünmesi ve rolling resistance çöz.
3. DEM temas impulsunu aynı adımda MPM grid momentumuna iki yönlü geri yaz;
   MPM↔DEM geçişinde kütle, doğrusal ve açısal momentum sayaçlarını raporla.
4. Uyuyan/gömülen taneleri tekrar MPM iç yığınına al. Kullanıcı bütçesi aktif DEM
   tanelerini sınırlar; bütçe dolunca iç taneler sanal küreye çevrilmez.
5. Kabulte dökülen kuru kumda tekil yüzey yuvarlanması, angle of repose, iki kolon
   yüksekliği, collider çevresinde kayma ve kapalı momentum bütçesi ölçülür.

Bu katman C4'teki çok malzemeli ortak transfer/contact tasarımıyla aynı hücre
hash'ini paylaşmalıdır. Ayrı bir milyon-taneli tam DEM yolu kurmak, hem G1'de
ölçülen P2G maliyetini ikiye katlar hem de birleşik domain çekirdeğini tekrar
yazdırır.

## C4–H1 veri ve momentum sözleşmesi (2026-10-04)

**C2 kod güncellemesi:** ilk parti `FluidActiveWindow.h` ve Vulkan normalize
pencere shader'ıdır. UI/Python/IPC aynı hücre sayaçlarını sunar. Clear,
occupancy ve MGPCG henüz tam-grid'dir; C2 tamamlandı sayılmaz. Kabul ve sonraki
kod sınırı: [FLUID_ACTIVE_WINDOW.md](FLUID_ACTIVE_WINDOW.md).

**C2b kod partisi:** sıvı occupancy maskesi GPU'da üretilip viscosity/pressure
arasında tekrar kullanılıyor; 17 ortak-kaynak shader varyantı CG hücre işlerini
ve reduction bloklarını aynı aktif pencereye bağlıyor. Basınç çekirdeği yeni
`FluidGpuPressure.inl` modülüne çıkarıldı. Statik sözleşme kontrolü PASS;
Kullanıcı C2b derlemesinin başarılı olduğunu doğruladı; canlı performans kabulü
isteği doğrultusunda daha büyük parti sonunda yapılacak.
Tam-grid cold clear ve yüz sınır temizliği korunuyor; C2 tamamen kapatılmadı.

**C3a faz erişim partisi (tarihsel):** seed, kaynak voxel'i, particle/SDF render, UVW,
fluid istatistikleri ve termal/mist/yanma hesapları ortak faz seçicisine bağlı.
Solver CPU/GPU grid ve aktif metadata geçişi kapsam korumalı; cache ötelemesi
iki grid'i birlikte taşıyor. Bağımsız faz authoring/serializer/bütçe allocation
paketi bu aşamada açık kaldı; aşağıdaki C3b bunları tamamlar. Ayrıntı: [MATTER_PHASE_GRID.md](MATTER_PHASE_GRID.md).

**C3b kaynak paketi (2026-10-04):** kalan bağımsız faz ayarları, ortak doğrulama,
faz başına tahsis ve kombine bütçe, UI/Python/IPC, iki serializer ve cache hash/
restore uyumu uygulanmıştır. Primary SDF/NanoVDB ve secondary gaz/fog doğru faz
gridlerini okur; iki karma preset eski ayrı kutularını tek Matter faz ayarlarına
taşır. C3a ve önceki testler kullanıcı tarafından PASS bildirildi. C3b ile splat
taşıma düzeltmesi tek toplu derleme/canlı kabul bekler; bu kabul gelmeden C3
tablosuna tamamlandı işareti konulmaz. Checklist: [MATTER_PHASE_GRID.md](MATTER_PHASE_GRID.md).

Bu bölüm 2026-10-04 hibrit aday sözleşmesidir; yukarıdaki H1 üretim güncellemesi
uygulama sırası ve solver seçimi için önceliklidir. CPU referansı mevcut, üretim
H1 henüz uygulanmış değildir. Kullanıcı önceki
değişikliklerin derleme ve çalıştırmasının sorunsuz olduğunu doğruladı.
G2'nin MPM yük/repose ölçümleri açık kalır. Gerçek tane yuvarlanması H1'de,
C4 ortak komşuluk ve contact altyapısı üzerine uygulanır; C2/C3'ü engellemez.

### Tek fizik sahibi ve kararlı kimlik

- Her fizik taşıyıcısı kalıcı `particle_id` taşır. Kompakt GPU listelerinin
  indeksleri kimlik değildir; silme, sıralama ve cache dönüşünde kimlik korunur.
  Render çocukları fizik kimliği, kütlesi veya contact girdisi üretmez.
- Granüler taşıyıcı için `transport_owner = mpm | dem` tek otoritedir. DEM
  sahibi taşıyıcı MPM kütle/momentum ve stress scatter'ına tekrar katılmaz.
  Contact'a sunulan DEM desteği ayrı coupling alanıdır; ikinci kütle değildir.
- DEM sidecar yalnız aktif taneler için ayrılır: fiziksel yarıçap, quaternion,
  açısal hız, atalet ve uyku/geçiş durumu. Temas yarıçapı render radius/size
  ayarından türemez; temsil edilen katı hacim ve fizik paketleme politikasıyla
  tanımlanır. Bir MPM taşıyıcısı gerçek mikroskobik kum tanesi sayılmaz.
- C5 gözenek suyu ve C6 doygunluk verisi taşıyıcı kimliğine bağlı kalır; sahiplik
  değişimi substance, kuru iskelet kütlesi veya gözenek suyunu yeniden yaratmaz.

### Ortak alt adım ve contact

1. Ortak alt adım başında sahiplik listeleri sabitlenir. Hücre hash'i mevcut
   konumlardan kurulur; MPM stencil'i ve DEM komşuluk sorgusu aynı origin/voxel
   sözleşmesini kullanır. Contact arama yarıçapı büyükse gerekli komşu hücre
   sayısı artırılır; yalnız bitişik 27 hücre varsayımı yapılmaz.
2. MPM sahipleri model başına ayrı kütle/momentum alanlarına scatter edilir.
   Liquid pressure ve granular stress güncellemeleri kendi alanlarında çalışır.
   DEM serbest hareketi aynı dt ile ilerler; contact kararlılığı daha küçük dt
   istiyorsa ortak scheduler bunu sağlar, authored stiffness düşürülmez.
3. DEM–DEM, DEM–MPM ve collider contact çiftleri tek sahiplik kuralıyla çözülür.
   Her impuls çifti bir kez üretilir; iki tarafa eşit ve zıt uygulanır.
   DEM–MPM karşı impulsu normalize stencil ile MPM momentumuna yazılır.
   Coulomb sürtünmesi ve rolling resistance açısal güncellemeyi de kapsar.
4. Grid ve parçacık güncellemelerinden sonra geçişler commit edilir. Açık/kapalı
   sınır ile collider impulsları ayrı dış-momentum hesabına kaydedilir.

### MPM↔DEM geçiş kapısı

Yüzey/kopma seçimi occupancy, malzeme ve komşuluk verisinden fizik servisi
tarafından yapılır; render etiketinden yapılmaz. Giriş/çıkış eşikleri ve minimum
bekleme süresi farklıdır; yüzeyde her adım sahiplik titreşimi önlenir. Bütçe
dolduğunda aday MPM'de kalır ve ertelenen aday sayısı raporlanır.

MPM→DEM başlangıç dönüşü affine alanın dönel bileşeninden türetilir; ancak
yalnız vorticity kopyalamak açısal momentum korunumunun kanıtı değildir.
Transferin orbital ve iç açısal momentumu ortak dünya referans noktasında
hesaplanır. DEM→MPM dönüşünde spin uygun affine/grid katkısıyla taşınır.
Bu katkı henüz temsil edilemiyorsa geçiş reddedilir; spin sessizce atılmaz.
Her iki yönde önce/sonra kütle, doğrusal momentum, açısal momentum ve kinetik
enerji farkı kaydedilir. Enerji korunumu sürtünmeli contact için zorunlu
değildir; geçişin eklediği sayısal enerji ayrı raporlanır.

### Servis, kalıcılık ve kabul

Yeni kod odaklı `MatterParticleIdentity`, `MatterSpatialIndex`,
`MatterContactService` ve `GranularHybridService` modüllerinde uygulanır.
Bunlar önerilen modül adlarıdır; mevcut eşdeğer core varsa genişletilir.
2000 satır üstü dosyalara yalnız entegrasyon çağrıları eklenir.

H1 authoring sözleşmesi enable, aktif tane bütçesi, fizik yarıçap politikası,
contact friction/rolling resistance ve geçiş eşiklerini kapsar. UI, Python ve
IPC aynı servis üzerinden okur/yazar. Geçersiz veya sonlu olmayan değerler
mutasyondan önce reddedilir; yarım güncelleme yapılmaz. İstatistikler aktif,
uyuyan ve ertelenen tane sayısını, çift sayısını, transfer hatalarını ve GPU
sürelerini içerir. Serializer ve frame cache kimlik, sahiplik ve DEM sidecar'ı
saklar; anahtar kareden resume aynı fizik durumunu kurar.

Uygulama kapıları:

- **C4a:** kimlik/model listeleri ve ayrı akümülatörler; tek su ve tek kum
  referansları. **C4b:** ortak hash/stencil ve iki yönlü su-kum contact.
- **H1a:** CPU küçük-sahne referansı; iki küre contact, Coulomb kayma,
  rolling resistance ve grid karşı-impuls testi. Toleranslar sahne ölçeği,
  dt ve hassasiyetle testte açıkça tanımlanır; sonuçtan sonra gevşetilmez.
- **H1b:** Vulkan aktif liste/contact yolu; CPU referansı, bütçe dolması,
  deterministik çift sahipliği ve toplam süre/bellek karşılaştırması.
- **H1c:** iki yönlü geçiş, kapalı momentum, cache/scrub/resume, iki kolon
  yüksekliği ve iki çözünürlükte repose/runout. Fizik taşıyıcısı sayısının render
  çocuk ayarından etkilenmediği UI/Python/IPC üzerinden doğrulanır.

H1 kapalıyken mevcut MPM sonucu korunur ve DEM buffer/dispatch maliyeti oluşmaz.
İlk uygulama işi C2'nin aktif pencere kapsam denetimidir; H1 bu sözleşme
üzerinden C4 tamamlandıktan sonra kodlanır. Derleme kullanıcı tarafından yapılır.

## C4a kaynak checkpoint — kimlik ve transfer/contact temeli

2026-10-04 büyük kaynak partisi: domain-local 64-bit particle_id, SoA
remove/compact ve cache korunumu; disk sim-cache v10; ayrı fluid/granular
kg ve momentum accumulator'ları, ortak quadratic cell support ve stable-ID
occupant listesi; transaction-safe eşit/zıt normal+Coulomb contact referansı.
Active Matter tablosu ve `fluid.matter_models` Python/IPC sorgusu ortak core
servisini kullanır. Standalone C++ ve dış IPC kabul test kaynakları eklendi.
Statik sözleşme kontrolleri geçti; C++/canlı yeni-parti kabulü henüz çalışmadı.

Bu C4a altyapı/referans partisidir; contact henüz canlı solver'a bağlı değil.
C4 tamamlandı işareti konulmadı. C4b ayrı-model canlı transfer/solve/contact
ve GPU yolunu tamamlayacak; H1, C5/C6 ve C7 açık kalır. Ayrıntı ve kesin
kullanıcı checklist'i [MATTER_TRANSFER_CORE.md](MATTER_TRANSFER_CORE.md).

C3 checkpoint `da93cec` gönderildi. Sonraki RT fog visibility/TLAS düzeltmesi
kullanıcı derlemesinde gas+SDF birlikte render kabulünü geçti. Liquid Body
parametrelerinin gas varlığında bazı karelerde etkisizleşmesi ve taşıma/splat
regresyonu son kalite turuna ertelendi; suyun ateşi söndürmesi kabulü kullanıcı
tarafından geri çekildi. Güncel görsel kayıt [MATTER_PHASE_GRID.md](MATTER_PHASE_GRID.md).

### C4a kısa canlı kabul / C4b CPU stage sınırı

Kullanıcı C4a derlemesini tamamladı. Boş sahnede geçici domainle model/kimlik
ve transfer IPC smoke PASS (216 parçacık, kimlik byte sayısı ve kütle toplamı);
geçici sistem temizlendi. C++ contact ve disk identity roundtrip kabulü açık.

C4b CPU projection→contact→G2P sınırı kaynakta hazır: APIC step odaklı inl
modülüne çıkarıldı, model-local FLIP snapshot, çift finish ve aradaki topology/
layout değişimi kapıları eklendi. Statik kontroller PASS. Bu ilk C4b kaynak
adımı henüz canlı coordinator/MAC contact/GPU karma transport'u bağlamaz.
Sonraki büyük blok bu bağlamaları tamamlayacak; ara derleme talep edilmiyor.
Ayrıntı [MATTER_TRANSFER_CORE.md](MATTER_TRANSFER_CORE.md).

2026-10-04 C4b: atomik iki-model CPU batch koordinatörü kaynakta eklendi.
Canlı MAC temas/CFL/bellek bütçesi bağlantısı bekliyor; karma transport tamamlanmadı.
İki emitter için domain kimyası ve ayrı splat materyali gözlemleri
`MATTER_PHASE_GRID.md` son kalite notlarına alındı.

C4b üretim hedefi kullanıcı yönlendirmesiyle açıkça GPU compute: CPU karma
koordinatör doğrulama referansı olarak kalacak, GPU domain'i otomatik CPU'ya
yönlendirmeyecek. GPU model partition kernel/ABI/buffer guard kaynakta eklendi;
indexed transfer/contact ve canlı GPU coordinator tamamlanmadan tam karma
test veya yeni derleme döngüsü istenmeyecek.


2026-10-04 güncel C4b test noktası: `MATTER_MIXED_GPU_TEST_POINT.md`.
Karma GPU canlı bağlantısı, ortak alt adım, fiziksel P2G kütlesi ve GPU temas
kaynakta eklendi; önceki “bağlı değil” kayıtları tarihsel ilerleme notlarıdır.
Derleme/shader/sahne kabulü henüz kullanıcı tarafından yapılmadı. Sonraki adım
tek derleme ile karma GPU kabulü; C4 tamamlandı etiketi henüz verilmedi.

## 2026-10-04 Output ve ortak emitter havuzu partisi

Karma GPU ilk canlı probu geçti; kullanıcı su/kum rejimlerini görsel ayırdı.
Output yeniden Liquid Display ile başlar; Matter madde bazlı SDF/Splat/Fog ve
mevcut sahne materyallerini aynı ortak API üzerinden sunar. Katalog maddesi
görsel preset'ten düzenlenebilir sahne materyali üretebilir. Serbest canlı
kapasite kaynaklara `particle_pool_weight` ile paylaştırılır; kaynakların
ömür boyu üretim limitinden ayrıdır. Bu yeni parti kullanıcı derlemesi/kabulü
bekliyor; madde başına fog shader ve C5/C6 hâlâ açık. Ayrıntı/kabul komutu:
`docs/dev/MATTER_MIXED_GPU_TEST_POINT.md`.


### 2026-10-05 — C5 büyük kaynak partisi, kabul bekliyor
GPU emilim/drenaj, kanonik pore mass/capacity/porosity/thermal energy sidecar'ları,
ıslak taşıyıcı transport kütlesi, korunum kapılı yayın ve ledger olayları yazıldı.
UI, Python ve IPC ortak authoring servisine bağlı; serializer ve cache v11 hazır.
Bu bir kaynak teslimidir: C++/shader derlenmedi, canlı C5 kabulü yapılmadı.
C5 fiziksel kabulü açık; hücre-local ilk adımın dt/çözünürlük, havuz doluluğu ve
cache round-trip ölçümleri gerekir. C6 wet friction/cohesion/pore-pressure ve
ıslak görünüm henüz uygulanmadı; C7 kalite kabulü açık. İlk scope Closed Vulkan
Water + Sand/Gravel/Soil. Eski cache v10 yeniden bake edilir.
Kesin devir, dosyalar, sınırlamalar ve sonraki kabul adımları:
[MATTER_C5_HANDOFF.md](MATTER_C5_HANDOFF.md).


## 2026-10-05 canlı C5 incelemesi — kullanıcı derlemesinden sonra
Kullanıcı açık ikili emitter sahnesini ölçmeye izin verdi. Uygulama açılmadı,
derleme yapılmadı; dış scripts/test/rt_ipc.py ile named-pipe escalation kullanıldı.
Sahne C4_Mixed_GPU, Closed Vulkan, 32^3, 50.000 particle limit. İlk durumda
frame 0/boş runtime, C5 disabled; C4_Sand emitter aslında Soil (Granular),
C4_Water emitter küçük harf water (Fluid). Soil için domain custom cohesion
1807 Pa, Young 706400 Pa; bu kuru Sand kabul sahnesi değildir.

Mevcut ayarlar ile ilerletmede GPU üç stage etkin, held=false; temas sayaçları
199,529,1562 oldu. Contact pair sayısı kuvvet transferinin tüm doğruluğunu,
batmayı veya hidrostatik buoyancy/drag kabulünü kanıtlamaz.

Test sırasında Water emitter etiketi Water'a düzeltildi; aynı Obj_1_Material /
SDF binding Water için eklendi (eski water binding korunuyor). Soil ve kaynak
konumları değiştirilmedi. C5 enabled, permeability=1e-8, drainage_scale=0 ile
emilim; sonra drainage_scale=1 ile drenaj denendi. Bu ayarlar sahnede kaldı,
dosyaya save yapılmadı. Son playhead 68, playing=false.

Kaydedilmiş canlı örneklerde pore water 0.0134 -> 0.0354 -> 0.2959 -> 3.3068 kg,
max saturation 0.3204; absorption last-step 0.4237 kg. Mass residual yaklaşık
-7.2e-8 kg, momentum residual 4.6e-7 kg m/s. Drenaj örneği: 0.00049856 kg /
66 yeni Water parcel, mass residual -8.1e-9 kg. Bunlar anlık solver örnekleridir,
aynı durdurulmuş sistemin before/after bilançosu veya çözünürlük kabulü değildir.
Timeline set_frame asenkron resync yapar; config değişimi cache/resync tetikler.
JSON'daki frame requested playhead'dir; her örneğin solver yaşını garanti etmez.
Örnekler: matter_c5_live_2026-10-05.json.

Önemli açık sorun: sonraki ölçümde 50.000 limit doldu: 45.425 Fluid + 4.575
Granular. 3.524 pore taşıyıcısında drainage slotu yok, drained_kg=0; su tutuldu.
Kararlı son query öncesi/sonrası sim.control_state epoch=0, frame=68,
playing=false idi. Önceki query ile bu query arasında çok sayıda drainage birth
oluştu; timeline-resync/step sayımı sonraki kontrollü testte izlenmeli.

Harici mixed GPU --contact probe PASS (78.098 contacts). C5 --expect drainage
probe son karede FAIL: havuz dolu olduğundan drained_kg=0, bu aşamada beklenen
budget koşulu; önceki kayıtlı örnekte drenaj doğumu görüldü. C5 genel finite /
saturation / GPU publication probe ayrıca çalıştırıldı. C5 tam kabul açık.

## Öncelikli sonraki düzeltmeler
1. FluidPhysicalMass.cpp densityForParticle yalnız liquid_density kullanıyor.
   Sand/Gravel/Soil bu alanı override etmiyor (default 1000); dry density alanı
   Sand 1600, Gravel 1750, Soil 1450. Gözlenen Soil 0.125 kg/parcel, Water
   ~0.124625 kg/parcel. Rejim-aware mass initialization yapılmalı: Granular dry
   density, Fluid liquid density; exact phase-transfer masses korunmalı. CPU/
   GPU ortak helper, legacy Auto fallback ve yeni-emission yolları denetlenmeli.
   Bu turda fizik kaynak kodu değiştirilmedi; mevcut exe bu kusuru taşıyor.
2. C5 tiny drainage parcel her taşıyıcı/her step üretimi havuzu hızlı dolduruyor.
   GPU hücre/model bazlı birleştirme veya kontrollü birth batching/reuse tasarla;
   mass/momentum/thermal energy/ID/cache/ledger korunumu birlikte sürmeli.
3. Fluid-only pressure ile unilateral Coulomb grid-contact tam hidrodinamik
   drag/buoyancy doğrulaması değildir. C4 roadmap iki yönlü contact/drag kabulünü
   (ayrı referans, yoğunluk kontrastı, submerged bed, pressure/drag transfer)
   açık tut. C6 wet effective stress/cohesion uygulanmadan batma kabulü verme.
4. Yukarıdakilerden sonra gerçek canonical Water+Sand (zero cohesion) paused
   isolated test; emitters stop, epoch/time checked balances ve dt/resolution.


### 2026-10-05 C6 ilk kaynak partisi (temel canlı problar geçti)
Canonical pore saturation -> wet strength/local head ve 8-band granular splat
appearance; shared UI/Python/IPC authoring, serializer/cache hash ve test kaynakları
bağlı. C6 fiziksel kabul kapanmadı; local head pressure PDE değildir. Toplu shader
ABI: stress_update 18/68, stress_p2g 9/52. Güncel kısa durum/test listesi:
[MATTER_C5_HANDOFF.md](MATTER_C5_HANDOFF.md); model sınırları:
[MATTER_WET_RESPONSE.md](MATTER_WET_RESPONSE.md).
