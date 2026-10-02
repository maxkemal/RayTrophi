# Birleşik madde domain'i — gaz/sıvı ayrımı olmayan simülasyon ve render

> **Durum:** AŞAMALI UYGULAMA — 2026-09-28. Faz 0 ve ilk görünüm/UI adımları
> mevcut; simülasyon etiketleri, dönüşüm defteri ve gerçek mist→gaz faz
> aktarımı canlı çalışıyor. Faz 4 domain birleşmesi canlı kabulden geçti.
> İleri aşamaların
> tasarımı değişebilir. Hedef: tek
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
| **Son güncelleme** | 2026-10-01 — C0 ve C1 canlı PASS. C1, 100k sıvıda yinelenen 1.200.000 byte velocity indirmesini ve bir batch fence'ini kaldırdı (9.430.584 byte, 12 batch); fiziksel digest farkı 1e-9 altı. C4 parçacık constitutive kimlik temeli de canlı kabul edildi. |
| **Kodda ne var** | `SimulationDomainType::Matter`, tek descriptor/kimlik altında ayrı gaz alanı (`state.grid`) + APIC alanı (`matter_liquid_grid`) taşıyor; iki çözücü sıralı çalışıyor. Mist/yanma aktarımı, hareketli sıvı sınırı ve ayrı SDF/gaz render slotları bağlı. Burning Fuel Spill'de gaz görünümü, SDF sürekliliği ve yüzeyden kademeli yanma canlı doğrulandı. Boş gaz fazı uyuyor. Matter faz başına ayrı kalıcı GPU buffer takımı taşıyor. Burning Fuel Spill 12 karede 65.664 sıvı parçacığı + 9.693 aktif gaz hücresiyle geçti; sıvı kare başına ~17 MB yükleyip ~11 MB indirirken gaz toplamı 98,45 ms idi. Bu toplamın görünmeyen ortak maliyetlerinden biri bulundu: fiziksel gaz kütle/enerji sidecar'ları tüm gaz domainlerinde boşken bile tam grid üzerinde CPU'da iki kez taşınıyordu. Sidecar artık ilk gerçek faz aktarımına kadar ayrılmıyor; dolu alan yalnız pozitif destek AABB'si ve güvenlik bandında advect ediliyor. `gas.step_stats.inventory_advection_ms` ve `fluid.step_stats.total_ms` eklendi. Splat havuzu iki spray için 256 slot açtı ve RT geçişi TDR'siz geçti. Son optimizasyonlar henüz derlenmedi; shader değişmedi; commit edilmedi. |
| **Faz 0 durumu** | 1. tur (koddan) + 2. tur (canlı) BİTTİ — §8b. Kısıt: `density`/sıcaklık/hız faza göre farklı ANLAM taşıyor → ortak grid ≠ ortak alan. Canlı: Vulkan matrisi 9/9 çiziyor (fog Material ≈ siyah); gaz+SDF katmanı 3 yolda tutuyor; sim VRAM'i izlenmiyor (519 MB). IPC render_mode resync düzeltmesi ✔ derlendi ve doğrulandı (exe 22:21) |
| **Sıradaki iş** | C2 aktif sıvı çalışma penceresi, sonra C3 faza özgü grid. Ardından C4 parçacık başına constitutive model ve çok malzemeli P2G/contact, C5 korunumlu su emilim-drenajı, C6 doygunluğun kum fiziği ve materyalini sürmesi, C7 performans/fizik kabulü. |
| **Karar verilmiş** | UI: ayrı DCC ekranı yok, mevcut Domain paneli altı sekme (§8b); panel materyal bağlar, düzenlemez; ad "Physics Domain" (✔ doğrulandı); tüm UI metni İngilizce, genel tema. §3 ilkeleri; kimlik+depolama birleşir, çözücü değerlendirmesi birleşmez. Emitter çözücü seçmez: `substance + başlangıç durumu + kütle/enerji/bileşim` yatırır; domain anlık durumdan constitutive model/çözücü seçer. Granular bir termodinamik faz adı değil, constitutive rejimdir. Kimya emitter özelliği değil, substance bileşimi ile domain reaksiyon kurallarının sonucudur. |
| **Açık karar** | Su+kumun aynı Matter adımında doğru etkileşmesi artık kapanış şartıdır. Temel sözleşme §8d'de sabit: serbest su kütlesi ile granül gözenek suyu ayrı ve korunumludur; domain-geneli `granular_enabled` fizik otoritesi olmaktan çıkar; görünüm fiziksel doygunluğu okur. Gaz-sıvı ortak basınç, materyal editörü, OptiX ve RT/SDF kusurları ayrı kalır. |
| **Kullanıcıyla konuşulan** | Hedef TAM akışkan fiziği: gaz, sıvı, granül (2026-09-28). Whitewater kütlesiz alt-ızgara yer tutucusu, fizik gibi sunulmaz; etiketle yalnız GÖRÜNÜMDE birleşir (§8c). Fizik-kimya sistemi (§4.5); şelale ilk senaryo; UI modelden türemeli, kod öncesi mock. |
| **Dikkat** | `GAZ_DERSLERI_VE_FLUID_DEVRI.md` koddan geride; bir AKTİF planın "sıradaki"sine güvenmeden koda bak. Eski sahnelerde whitewater artık küre çizilir (varsayılan tablo splat; eski "Volume" = foam/bubble → Isosurface; göç yok). Canlı test pahalı: sayıyla doğrula, en fazla bir görüntü. Testler: scripts/test/rt_test_whitewater_label_routes_ipc.py, rt_probe_whitewater_determinism_ipc.py, rt_test_fluid_label_views_ipc.py, rt_test_fluid_labels_cache_ipc.py. |

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

### Kalan sekiz kapanış partisi

| Parti | Kod işi | Ölçülebilir kapı |
|---|---|---|
| **C0 — sayaç kabulü** | Lazy gaz kg/J sidecar, aktif-support advection ve gerçek sıvı-domain `total_ms` | Flamethrower hızlanması korunur; düz gazda `inventory_advection_ms≈0`; 100k/40³ tek sıvıda `total_ms`, `sim.timeline.step` ile aynı maliyet sınıfında |
| **C1 — sıvı residency** | G2P velocity+affine ile hemen sonraki device advect-tail position+velocity geri okumalarını tek submit/readback sınırında birleştir; device-tail başarısızlığında doğru host fallback | Referansta batch sonu 13'ten en az 12'ye iner; 100k parçacıkta tekrar indirilen velocity kaldırıldığı için download en az 1,2 MB/kare azalır; parçacık digest ve kapalı-tank sonucu değişmez |
| **C2 — aktif sıvı çalışma penceresi** | Parçacık AABB + CFL/pressure halo üret; P2G clear/normalize, fluid mask ve MGPCG yalnız bu hücre aralığında çalışır; tam grid fallback kalır | `active_liquid_cells/full_cells` IPC'de görünür; yerel sıvıda dispatch edilen hücre sayısı küçülür; sınır/çarpışma ve açık outflow testleri aynı sonucu verir |
| **C3 — faz grid sözleşmesi** | Tek descriptor altında `gas_bounds/voxel` ve `liquid_bounds/voxel`; core + UI + scripting + IPC + serializer/cache aynı servisi kullanır. Varsayılan mantıksal bounds'a düşer; presetler eski gaz/sıvı kutularını faz alanlarına taşır | Geniş/kaba gaz ve dar/ince sıvı aynı Matter domain'de raporlanır; kaynak, hareketli sınır ve render koordinatları doğru eşlenir; bütçe hesabı faz başına gerçek hücre sayısını kullanır |
| **C4 — çok malzemeli MPM çekirdeği** | `substance_tag`/madde özelliklerinden parçacık başına liquid veya granular constitutive model seç; ortak zaman adımında faz başına kütle/momentum P2G ve iki yönlü grid contact/drag çöz. Granülü sıvı pressure projection'a, suyu Drucker–Prager gerilmesine sokma | Aynı kutudaki su ve kuru kum eşzamanlı ilerler; her iki sınıfın parçacık/kütle/momentum sayaçları IPC'de ayrıdır; ayrı çalıştırılan su ve kum referansından sapma tanımlı toleranstadır |
| **C5 — emilim ve drenaj** | Granül parçacığa `pore_water_mass_kg`, porosity/capacity ve saturation sidecar'ı ekle. Temas alanı, geçirgenlik, basınç farkı ve dt ile sınırlı emilim; kapasite üst sınırı; yerçekimi/basınçla drenaj ve serbest suya geri dönüş. Her transfer tek ledger kaydıdır | Kapalı su+kum testinde serbest su kaybı = gözenek suyu kazancı; doygunluk 0..1; kuru kum uzakta kuru kalır; doygun kum drenajla ölçülebilir su geri verir |
| **C6 — ıslak kum fiziği ve görünümü** | Doygunluktan efektif stress/pore pressure, friction, cohesion, dilatancy ve kütle türet; aynı alanı kuru/ıslak materyal karışımına bağla. Core servisini UI/API/IPC/Python, serializer ve cache yüzeylerine taşı | Kuru, nemli ve doygun kumun angle-of-repose/çökme sonucu farklıdır; nemli bölgede ıslak görünüm mekânsal olarak fizik ile çakışır; cache dönüşünde doygunluk ve görüntü korunur |
| **C7 — son kabul ve kesim** | Üçlü performans A/B matrisi ile su-kum fizik matrisini çalıştır; preset göçü, cache/ledger/render regresyonu; geçici uyumluluk dallarını ve bayat plan metnini temizle | %10 tek-faz/iki-faz kapıları geçer. Su jeti/kum yatağı, ıslanma cephesi, drenaj, kuru-nemli-doygun yığın ve kapalı kütle testleri PASS; belge `TAMAMLANDI` olur |

C0 tek derleme ve canlı ölçüm turudur. C1–C3 ayrı derlenebilir partilerdir;
C4–C6 tam madde fiziğinin uygulama partileridir. C7 kod ekleme turu değil,
kabul ve temizlik turudur. Performans temeli önce kurulur; aksi halde su-kum
teması zaten pahalı olan tam-grid ve readback yolunun maliyetini katlar.

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
