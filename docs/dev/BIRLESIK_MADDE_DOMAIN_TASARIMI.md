# Birleşik madde domain'i — gaz/sıvı ayrımı olmayan simülasyon ve render

> **Durum:** TASLAK — 2026-09-27. Uygulanmadı; tasarım değişebilir. Hedef: tek
> domain bir fizik-kimya sistemi gibi davranır — madde (granül, sıvı, gaz,
> yanan katı) kendi özelliklerine göre erir, donar, buharlaşır, yanar, kütle
> kaybeder; çözücüler bu dönüşümlerde birbirini besler. İlk somut senaryo
> şelale (gövde SDF + kopan sprey splat + yükselen sis fog).

---

## ★ DEVİR NOTU — her oturum sonunda güncellenir

> Bu mimari tek oturumda bitmeyecek. Bu blok, işi devralan ajanın OKUDUĞU
> İLK şeydir; her iş partisinin sonunda güncellenir. Bayatsa önce onu düzelt.

| | |
|---|---|
| **Son güncelleme** | 2026-09-27 |
| **Kodda ne var** | Hiçbir faz başlamadı. Ön hazırlık olarak sıvıda fog modu (gerçek yoğunluk + Gauss yayma + parçacık Kelvin'i ile blackbody) yazıldı; son parti DERLENMEDİ — `docs/dev/NEXT_BUILD_CHECKS.md` en üst bölüm |
| **Sıradaki iş** | Faz 0 envanteri (§8b) — koda dokunmadan: 85+ gaz/sıvı dalının sınıflandırması, render matrisi ÖLÇÜMÜ, hacim kimliği değişmez testleri, materyal envanteri (§4.6), UI mock'u |
| **Karar verilmiş** | §3 ilkeleri; kimlik+depolama birleşir, çözücü değerlendirmesi birleşmez; önce zayıf bağlama; ayrı "1. adım refactor"u YOK (Faz 1 doğrudan görünüm çözücüsü) |
| **Açık karar** | §8 açık sorular; §4.6 materyal sözleşmesinin ayrıntısı; OptiX farkının kullanıcıya gösterimi |
| **Kullanıcıyla konuşulan** | Hedef fizik-kimya sistemi (§4.5); şelale ilk senaryo; UI modelden türemeli, kod öncesi mock |
| **Dikkat** | `GAZ_DERSLERI_VE_FLUID_DEVRI.md` koddan geride; bir AKTİF planın "sıradaki"sine güvenmeden commit'lere ve koda bak |

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
  hangi alanı kimin doldurduğu.
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
| sıvı → gaz | buharlaşma, yanma ürünü, `mist` aktarımı | yanma var; mist yok |
| gaz → sıvı | rüzgar/gaz hızıyla sürükleme (sprey, mist) | rüzgar sürüklemesi var |
| sıvı → gaz | sıvı hücreleri gaz için hareketli sınır | yok |
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

| Dönüşüm | Kaynak → hedef | Bugün |
|---|---|---|
| Erime | katı mesh / MSF → sıvı parçacık | var (`MoltenMassTransfer`, MSF 6b/6c) |
| Donma | sıvı → katı (`frozen`) | var (termal sıvı) |
| Buharlaşma / kaynama | sıvı → gaz | yok (yanma dışında) |
| Yanma | sıvı → gaz (+ısı) | var |
| Yanma / piroliz | katı → gaz + char (MSF) | kısmen (yangın → yapı, MSF char) |
| Yumuşama / sinterleme | granül ↔ katı / sıvı | kısmen (granül termal zinciri) |
| Yoğuşma | gaz → sıvı | yok |
| Sis aktarımı | sıvı `mist` → gaz | yok (Faz 3) |

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

   Bugün splat havuzu domain başına tek materyal taşıyor gibi görünüyor
   (scene-object splat'te yüz materyalleri); instance başına veri yolu olup
   olmadığı Faz 0'da DOĞRULANMALI. Yoksa sıcaklık/madde başına splat rengi
   yeni bir instance veri yolu ister (RT ve raster ikisinde).
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

### A) Zayıf bağlama, ortak grid (hedef)

İki çözücü aynı seyrek grid'de sırayla adım atar, birbirini sınır koşulu ve
kaynak terimi olarak görür (§4.4). Şelale sisi, lav dumanı, yanan sıvı,
rüzgarda sprey için fiziksel olarak yeterli. Maliyet bugünkü iki
simülasyonun toplamından DÜŞÜK olmalı: grid, transferler ve collider
voksellemesi bir kez yapılır. Mevcut çözücülerin kararlılığı riske girmez.

### B) Güçlü iki fazlı projeksiyon (isteğe bağlı, sonra)

Tek hız alanı, tek basınç denklemi, değişken yoğunluk (su ≈ 1000, hava ≈ 1).
Kabarcık, hava karışması, lıkırdama kendiliğinden çıkar. Hipotez: açık
"mühürlü basınç cebi" savrulması hava gerçek faz olunca kaybolabilir —
DOĞRULANMADI. Engeli: 1000:1 oran basınç sistemini kötü koşullar; güçlü
önkoşullayıcı ister ve GPU MGPCG taşıması duraklatılmış durumda. A'nın ortak
grid'i B'nin de temelidir, iş boşa gitmez.

### Çözünürlük politikası

Gaz geniş/kaba, sıvı dar/ince ister. Ortak grid **seyrek** olmalı (seyrek gaz
çözücü var): yalnız bir fazın bulunduğu bölgeler aktif. Sıvının yüzey detayı
grid'den ayrılmış kalır (SDF çözünürlük çarpanı zaten böyle).

---

## 6. Fazlar ve kapılar

Her faz kendi başına değer üretir ve bir sonrakine bir şey yıkmadan zemin
hazırlar. Her kapı IPC'den ölçülür.

| Faz | İş | Kapı (ölçüm) |
|---|---|---|
| **0** | Bu not + ölçü aletleri: etiket sayaçları, görünüm başına kaynak listesi IPC'de | `fluid.get` görünüm listesini ve etiket sayılarını döndürür |
| **1** | Görünüm çözücü + görünüm başına kaynak (§4.3). Tek karar noktası. Etiket henüz yoksa bugünkü mod tek etiket gibi davranır | Aynı domain'de SDF + fog **aynı anda**; panel, köprü ve API aynı listeyi raporlar |
| **2** | Parçacık etiketleri simülasyonda (§4.2); köpük etikete katlanır | Etiket sayıları; şelale sahnesinde gövde SDF + sprey splat + köpük Solid/Material/RT'de |
| **K** | Madde özellik tablosu (§4.5.1); çözücüler tablodan okur, ad tahmini kalkar. Faz 1–2 ile PARALEL yürür | Aynı maddenin her çözücüde aynı sayıları kullandığı IPC'den okunur; eski sahneler aynı davranır |
| **3** | Zayıf bağlama A + dönüşüm defteri (§4.5.2): sıvı hareketli sınır, gaz sürüklemesi, `mist` → gaz aktarımı; mevcut dönüşümler (erime, donma, yanma) deftere taşınır | Kütle VE enerji korunum sayaçları: kaynaktan çıkan = hedefe giren (tolerans içinde) |
| **4** | Domain birleşmesi: Gas/Fluid tipleri fazlara dönüşür; eski sahne göçü | Eski gaz ve sıvı sahneleri aynı görüntüyle açılır; faz listesi IPC'de |
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
  görünmesi) muhtemelen AÇIK — canlı ölçülemedi. Faz 3'ün ön koşulu yalnız
  bu: ortak grid'e iki çözücü koyarken VRAM'in nereye gittiği görünmeli.
- [PARTICLE_SYSTEM_GPU_ROADMAP.md](PARTICLE_SYSTEM_GPU_ROADMAP.md) (AKTİF):
  `Domain Deposit` bu modelin dış üretici sözleşmesidir; `Surface State
  Deposit` MSF'ye yazımdır. Çelişki yok; partikül sistemi domain'e dönüşmez.
- [KINEMATIC_COLLIDER_SOURCES.md](KINEMATIC_COLLIDER_SOURCES.md): collider
  kaynağı her faza aynı `grid.solid`'i verir.
- [VULKAN_GAS_FLUID_LAYERING.md](VULKAN_GAS_FLUID_LAYERING.md): render
  değişmezleri bu modelde de bağlayıcı (ayrı maske bitleri, gerçek geçişte
  devir, devrin GI sekmesi yememesi).

---

## 8. Açık sorular

1. **Etiket kriterlerinin eşikleri** mutlak birimde mi (m, m/s) voksel
   biriminde mi? Kural: fiziksel eşik mutlak birimde — ama komşu sayısı
   çözünürlüğe bağlı. Faz 2'de ölçülerek seçilecek.
2. **`mist` aktarımında kütle ölçeği:** sıvı parçacığı (kg) → gaz yoğunluğu
   (birimsiz alan). Tüketicinin BİRİMİ okunmalı; dönüşüm tek yerde.
3. **Kristalleşme verisi:** `frozen` bugün ikili. Kristal boyutu için
   donma süresi ve soğuma hızı parçacığa yazılmalı — Faz 2 sonrası ayrı iş.
4. **Ortak grid'in hareketi:** domain hareketi/çalkalanma iki faz için aynı
   referans çerçevesini kullanmalı.
5. **CPU referansı:** her alışveriş teriminin CPU karşılığı olacak mı, yoksa
   yalnız GPU mu? Partikül yol haritasının "açık CPU referansı" ilkesiyle
   hizalanmalı.

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

## 9. ★ Sinsi başarısızlıklar (şimdiden işaretli)

- **Etiket render'da hesaplanırsa** her şey çalışıyor görünür, ama panel,
  API ve görüntü farklı sınıflandırmalar gösterir. Kural §3.1.
- **Kütle sessizce kaybolur:** `mist` sıvıdan silinir ama gaza eklenmezse sis
  yine görünür (yayılmış fog), sadece sıvı yavaş yavaş azalır. Korunum sayacı
  olmadan kimse fark etmez.
- **Sayaç sıfırlama/yeniden doldurma kapsamı** (2026-09-27 fog hatası): bir
  faz için sıfırlanan her istatistik aynı faz için yeniden doldurulmalı.
- **Varsayılan ölçüm değildir:** etiket sayısı 0 dönerse "hiç sprey yok" mu,
  "sınıflandırma çalışmadı" mı — `*_measured` bayrağı şart.
