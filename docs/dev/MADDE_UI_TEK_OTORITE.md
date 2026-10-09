# Madde UI temizliği — her ayar tek yerde, çözücü maddeden

> **Durum:** TASLAK — 2026-10-09. §9 kararları verildi (kullanıcı ajana bıraktı);
> U0–U5 kaynakta, tek build (kullanıcı isteği); derlenmedi. Kontratlar 21/21 PASS.

Bu not [MADDE_TIPLERI_TASARIMI.md](MADDE_TIPLERI_TASARIMI.md)'nin **devamıdır** ve onun
kararlarını değiştirmez. Orada model kuruldu: madde seçilir, hal sıcaklıktan, çözücü hal ve
maddeden türer; preset bir tariftir; uzman domain'ler T0 ölçümü olmadan kalkmaz. Bu not o
modelin **panelde ve tane/MPM alanlarında** henüz uygulanmamış yarısıdır. Bir de, bugün
kurulamayan **kar** davranışının neden kurulamadığını ve ne gerektiğini yazar.

---

## 1. Kullanıcı bulguları (2026-10-09, canlı)

1. Madde düzenleyicide sürüklenen değer geri dönüyordu. Değer her kare kütüphaneden okunuyor,
   yalnız bırakınca yazılıyordu. **Düzeltildi** (U0).
2. Sand kaynağı "DEM" görünüyordu ama bağlı bir MPM akışı üretti. Neden: domain'deki
   "Enable discrete grains" anahtarı (varsayılan kapalı) sessizce MPM'e düşürüyordu.
   **Anahtar söküldü** (U0).
3. Aynı malzemenin ayarları üç yerde: Domain → Matter (tane malzemesi), Domain → Solvers
   (tane çözücüsü), madde düzenleyici (MPM granüler alanları). Bir de kaynak paneli.
4. Panelde çok fazla mühendislik ayarı var: sertlik, alt adım, kontak çözünürlüğü,
   doluluk oranı, uyku eşikleri.
5. "Kimya ve fizik preset'leri aynı listede." Yerleşik madde listesinde yanan katılar,
   dökülen sıvılar ve taneler tek düz listede.
6. **Kar gibi davranan bir malzeme hâlâ kurulamıyor.**

## 2. Dört soru modeli

Her ayar şu dört sorudan **birine** cevap verir ve yalnız o sorunun yerinde durur:

| Soru | Yeri | İçerdiği |
|---|---|---|
| **Nerede, ne büyüklükte, ne kalitede?** | Domain | Sınırlar, sınır tipi, voxel / simülasyon tane boyu, **Kalite**, bütçe, backend |
| **Ne?** | Madde | Fizik (davranış ailesine göre), termal, kimya, görünüm |
| **Ne kadar, nereden, hangi durumda?** | Kaynak | Konum, hız, debi, süre, madde, başlangıç sıcaklığı, başlangıç nemi |
| **Nasıl çözülür?** | **Hiçbir yerde ayar değil** | Maddeden türer; panelde ve IPC'de yalnız etiket |

Kural: bir ayarın hangi soruya cevap verdiği belirsizse, ayar ya iki ayara bölünür ya da
türetilir. "Panel kendi durumunu tutmaz" (CLAUDE.md §1) bu modelin panel tarafıdır.

★ Sayısal ayar ile fizik ayrımı ([MADDE_TIPLERI §6a](MADDE_TIPLERI_TASARIMI.md)) korunur.
FLIP karışımı, sönümler ve alt adım tavanı domain'in sayısal ayarlarıdır, maddenin değil.
Bu not onları kullanıcıdan **gizler** (Kalite altında), taşımaz.

## 3. Envanter: tane ayarları bugün nerede, nereye gitmeli

Kaynak: `MatterGrainParams` (domain, `fluid.set_grain_settings`) ve `SubstanceProfile`.

| Alan (bugün domain) | Hangi soru | Hedef | Not |
|---|---|---|---|
| `friction`, `rolling_friction`, `twisting_friction`, `restitution` | Ne? | **Madde**, "Taneler (DEM)" bölümü | Kumun özelliği. Bugün aynı kum iki domain'de farklı sürtünebilir. |
| `tangential_stiffness_ratio` | Ne? (ileri) | Madde, ileri | 2/7 küre sabiti; nadiren değişir. |
| `packing_fraction` | Ne? | Madde | Tane kütlesi = yoğunluk / doluluk × hacim. Kütle salt okunur gösterilir. |
| `represented_grain_radius_m` (gerçek tane boyu) | Ne? | **Madde** (`grain_diameter_m`) | 1 mm kum maddenin özelliği; simülasyon tane boyu domain'de kalır. |
| `water_capacity_fraction`, `absorption_rate_per_s`, `drying_rate_per_s` | Ne? | Madde (tane) | Gözenekliliğin özelliği. |
| `surface_tension_n_m`, `contact_angle_deg` | Ne? (sıvının) | **Sıvı maddesi** + tane maddesi | Köprü kuvveti sıvının yüzey gerilimini okumalı; bugün domain'de ikinci kopya. |
| `birth_saturation` | Hangi durumda? | **Kaynak** | "Nemli kum döken kaynak." |
| `radius_m` (simülasyon tane boyu) | Ne büyüklükte? | Domain | Voxel boyu gibi bir çözünürlük; maliyeti belirler. |
| `stiffness_n_m` | — | **Türetilir** | Bugün tooltip kullanıcıya "yarıçapla birlikte düşür" diyor. Bu otomatik olmalı (U3). |
| `contact_resolution`, `max_substeps` | Ne kalitede? | Domain → **Kalite** | Kullanıcı tek kontrol görür. |
| `sleep*` | Ne kalitede? | Domain, ileri | Performans ayarı. |
| `fluid_coupling`, `volume_exclusion`, `wet_grains` | Nasıl? (model seçimi) | Domain, ileri / türetilir | `wet_grains` domain'de sıvı + ıslanabilir tane varsa türetilebilir (§9-c). |
| ~~`enabled`~~ | Nasıl? | **Türetildi (U0)** | `matterGrainOwnership`. |

★ **İki sürtünme sorunu.** Madde MPM için `granular_friction_degrees` (iç sürtünme açısı φ)
taşıyor, DEM ise domain'deki kontak sürtünmesi μ'yü. Bunlar **aynı sayı değil**: DEM'de
yığın açısı μ ve μ_r birlikte belirler (kar DEM kalibrasyonunda μ 0.3 + μ_r 0.2 → 34°,
[BUZ_MODEL_KARARI](BUZ_MODEL_KARARI.md)). Bu yüzden `μ = tan φ` gibi bir eşleme **yapılmaz**;
o, sessiz yanlış kalibrasyon olur. Öneri: iki alan ayrı kalır ama **aynı maddede, yan yana**
durur, ve madde panelinde ölçülmüş yığın açısı (repose süiti) bilgi olarak gösterilir.

## 4. Panel düzeni

**Domain** (dört bölüm):

1. **Kap:** sınırlar, sınır tipi, voxel / tane boyu, backend.
2. **Kalite:** tek seçim *Taslak / Normal / Yüksek* (`contact_resolution` 12/24/48, CFL ve
   sweep sayıları). Altında kapalı **İleri**: alt adım tavanı, sertlik çarpanı, uyku, FLIP
   karışımı, sönümler.
3. **İçerik** (bugünkü Matter sekmesinin yerine): domain'deki maddelerin **salt okunur**
   listesi. Her satırda madde, ailesi, türetilmiş çözücüsü ("DEM taneler", "MPM süreklilik",
   "FLIP sıvı", "Gaz") ve varsa neden MPM'e düştüğü. Her satırda "Maddeyi düzenle"
   (madde düzenleyiciyi açar). Sayısal bir tane ayarı burada yoktur.
4. **Kaynaklar:** mevcut kaynak paneli. Madde seçicinin altında "bu domain'de: DEM taneler"
   satırı (U0'da yazıldı).

**Madde düzenleyici** (her yerde aynı bileşen, `SubstanceEditorUI`):

- Seçici **aileye göre gruplu** ve aranabilir: Sıvı / Granüler / Katı / Gaz / Yakıt.
- Bölümler: **Davranış** (yalnız ailenin alanları; granülerde "Taneler (DEM)" ve "Süreklilik
  (MPM)" alt başlıkları ve `granular_transport` seçimi), **Termal**, **Kimya** (yalnız
  yanıcı/tepkimeli maddede açılır; ayrı bir "kimya preset'i" yoktur), **Görünüm**.
- Her bölümde kapalı **İleri**.
- Yerleşik madde salt okunur; "Düzenle" = "Türet" (mevcut).

Gizlenen hiçbir alan script'ten kaybolmaz. IPC ve Python aynı alanları taşır; panel
sadeleşir (CLAUDE.md §1: script'ten yazılan alan panelde de düzenlenebilir, ileri bölümde).

## 5. Kütüphane: kaç madde, hangileri

[MADDE_TIPLERI karar 3](MADDE_TIPLERI_TASARIMI.md): yerleşik olan **temel** malzemelerdir,
gerisini kullanıcı türetir. Bu not o kararı korur ve bir ölçüt ekler:

> **Bir yerleşik madde, ancak başka bir parametre rejimini gösteriyorsa vardır.** Aynı rejimin
> varyasyonu (Steel ↔ Iron, Paper ↔ Cloth) yerleşik değil, örnek türetmedir, ya da kalıyorsa
> yanma/termal rejimi farklı olduğu için kalır.

Bugünkü 21 yerleşik (`MaterialStateField.cpp`), aileye göre:

| Aile | Bugün | Eksik rejim |
|---|---|---|
| Sıvı | Water, Oil, Honey, Chocolate, Mud, Gasoline, Alcohol | — |
| Granüler | Sand (DEM), Gravel (DEM), Ice (DEM, [BUZ_MODEL_KARARI](BUZ_MODEL_KARARI.md) Seçenek A), Soil (MPM) | **Kar** (sıkışan, kohezif, kırılan; §6) |
| Katı | Wood (Oak), Iron, Steel, Copper, Paper, Cloth, Plastic (PE), Wax, Stone, Flesh | — |
| Gaz | yok | Hava / duman / yakıt gazı (`gas_channel`, MADDE_TIPLERI T5) |

Silme önerilmiyor: katıların çoğu MSF yanma/erime rejimi için var. Önerilen yalnız
**gruplama** (§4) ve iki eksik rejim. Sahne tarifleri (Campfire, Explosion) madde listesinde
değil, ayrı bir "Tarifler" listesindedir ([MADDE_TIPLERI §3.4](MADDE_TIPLERI_TASARIMI.md)).

## 6. Kar: neden kurulamıyor, ne gerekir

Hedef davranış: basınca sıkışır ve sertleşir (ayak izi kalır), hafif kohezyonla topaklanır
(kartopu), gerilince kırılır (çığ kopması), ısınınca suya döner.

Referans model: Stomakhin et al. 2013, "A material point method for snow simulation".
Elastoplastik; deformasyon gradyanı elastik ve plastik parçalara ayrılır. Plastik hacim oranı
`J_p` kalıcıdır ve sertlik `exp(ξ (1 − J_p))` ile ölçeklenir. Yani **sıkışan kar sertleşir,
gerilen kar zayıflar**. Kritik sıkışma/gerilme sınırları θc ≈ 2.5e-2, θs ≈ 7.5e-3;
E₀ ≈ 1.4e5 Pa, ν ≈ 0.2, ξ ≈ 10, ρ₀ ≈ 400 kg/m³.

Bugünkü MPM granüler yolu (kodu okuyarak; **ölçülmedi**):

- Drucker–Prager; sertleşme `exp(hardening_coefficient × accumulated_plastic)`
  (`sim_fluid_granular_stress_update.glsl`). Birikmiş **kayma** plastisitesiyle büyüyor.
  **Hacimsel sıkışmaya (J_p) bağlı değil.**
- Kohezyon, çekme kesmesi, kırılma gerinimi, hasar/iyileşme, yeniden bağlanma ve sıcaklıkla
  yumuşama alanları var. Kar için gereken parçaların çoğu mevcut.
- **Hipotez:** kar davranışının eksik halkası sıkışmayla sertleşme (J_p durumu ve
  yoğunlaşma). Kayma ile sertleşen bir malzeme ezildiğinde yoğunlaşmaz, akar. Bu yüzden
  "toprak gibi" ya da "un gibi" görünür ama "kar gibi" görünmez.

Önce ölçüm, sonra kod (U5):

1. Soil'den türetilmiş bir "Snow (deneme)" maddesiyle üç kabul sahnesi: tek eksenli sıkıştırma
   (yoğunluk artıyor mu, sertlik artıyor mu), kartopu (topak kalıyor mu), eğimde kopma.
2. Sayıları yaz. Hipotez doğrulanırsa: parçacık başına `J_p` durumu, `exp(ξ(1−J_p))`
   sertleşmesi, θc/θs sınırları, yoğunluğun `J_p` ile güncellenmesi (görünür sıkışma).
   Mevcut kayma sertleşmesi **ayrı bir alan** olarak kalır (toprak bunu kullanıyor); anlam
   değiştirilmez, yeni alanın adı ayrı olur (CLAUDE.md §5).
3. Kar yerleşik maddesi ancak bu kabul geçince eklenir (§5 ölçütü: yeni rejim).

Açık model sorusu: kar, "Water'ın katı hali" mi, ayrı bir madde mi? MADDE_TIPLERI T2
"Ice → Water katı hal" diyor, ama kar ile buz aynı hal ve farklı katı davranış (granüler MPM
↔ elastik/engel). Öneri: kar **ayrı madde** olsun (`based_on` Water'ın termal alanları), erime
ürünü olarak Water etiketini taşısın. Bunun için yeni bir `melt_product` alanı gerekir (§9-d).

## 7. Aşamalar

Her aşama kendi build'i ve IPC kabulüyle kapanır ve CLAUDE.md §1'in dört dokunuşunu taşır.

| Aşama | İş | Kabul |
|---|---|---|
| **U0** ◐ kaynakta 2026-10-09 | Taneler maddeyi izler (`matterGrainOwnership`, anahtar söküldü, `grain_ownership` IPC); Solvers ve kaynak panelinde durum satırı; madde düzenleyicide sürükleme düzeltmesi. | `NEXT_BUILD_CHECKS` "Taneler maddeyi izler". |
| **U1** ◐ kaynakta 2026-10-09 | Madde düzenleyici: aileye göre bölümler + İleri; seçici gruplu + arama. Alan taşımaz. | Panel alan denetimi (`check_domain_panel_fields.py`) + madde IPC testi değişmeden PASS. |
| **U2** ◐ kaynakta 2026-10-09 | §3 tablosundaki malzeme alanları domain'den maddeye. Köprü fiziği sıvı maddesinden. `birth_saturation` kaynağa. Göç: eski domain değerleri tabandan farklıysa `<Domain> grains` türetilmiş madde ([MADDE_TIPLERI §6a](MADDE_TIPLERI_TASARIMI.md) göç kuralı). `set_grain_settings` malzeme anahtarlarını yeni yerini söyleyen hatayla reddeder. | Grain süiti aynı sayılar; eski proje aynı fizikle açılır. |
| **U3** ◐ kaynakta 2026-10-09 | Kalite kontrolü (zaten vardı: Draft/Production/Reference); sertlik tane yarıçapından türetilir (varsayılan k = 2e4 N/m @ r = .025 m, süit kolları 1e5 kullanıyor → k ∝ r) + ileri çarpan. | Süit; yarıçap yarıya inince çakışma oranı aynı kalır. |
| **U4** ◐ kısmi 2026-10-09 | Domain paneli: Kap / Kalite / İçerik / Kaynaklar; Matter sekmesi salt okunur İçerik olur. | Panel alan denetimi: her kaybolan alan CHANGES'ta, gerekçeli. |
| **U5** ◐ kaynakta 2026-10-09 | Kar: §6 ölçümü, sonra hacimsel sertleşme. | §6 üç sahne. |

U1 ve U4 yalnız panel; U2 ve U3 fizik beslemesine dokunur. Diğer ajanın DEM uyku işi
(`sim_matter_grain.glsl`, `MatterGrainGpu.cpp`) U2 ve U3 ile aynı dosyaları besler; sıralama
kullanıcının kararı.

## 8. Riskler

- **Sessiz kalibrasyon kaybı (U2):** domain'de elle ayarlanmış sürtünme maddeye taşınırken
  kaybolmamalı. Göç kuralı farkı türetilmiş maddeye yazar; yazmazsa yığın açısı değişir ve
  bunu kimse bug diye bildirmez. Kabul: göçten önce ve sonra repose süiti.
- **Aynı madde, farklı domain, farklı tane boyu:** tane boyu domain'de kaldığı için doğru.
  Ama sertlik türetilmezse (U3 öncesi) aynı malzeme iki domain'de farklı sekme gösterir.
  U2 ile U3'ün arası kısa tutulmalı.
- **Panelde gizlenen alan:** İleri'ye inen alan script'ten yazılmaya devam eder. Panel alan
  denetimi "taşındı" ile "kayboldu"yu ayırır.

## 9. Kararlar (2026-10-09)

Kullanıcı kararları ajana bıraktı ("sorularının en iyi cevabı sende"). Gerekçeleriyle:

a. **Sıra: U1 → U2 + U3 → U4 → U5.** U1 yalnız panel, risksiz ve hemen görünür. U2 ile U3
   aynı partide: arada kalırsa aynı madde iki domain'de farklı sekme gösterir (§8). Kar en
   sonda, çünkü yeni fizik ve önce ölçüm istiyor.
b. **Sürtünme: iki ayrı alan, aynı maddede yan yana.** DEM kontak μ ile MPM iç sürtünme açısı φ
   aynı sayı değil (§3 ★); birini ötekinden türetmek sessiz yanlış kalibrasyon olur. Madde
   panelinde ölçülmüş yığın açısı bilgi olarak gösterilir.
c. **`wet_grains` türetilir:** domain'de sıvı ve su kapasitesi > 0 olan bir tane maddesi varsa
   açık. Islanabilirlik maddenin `water_capacity_fraction`'ından gelir (U2 ile birlikte).
d. **Kar ayrı madde:** termal alanları Water'dan (`based_on`), erime ürünü Water etiketi
   (`melt_product`, tüketicisiyle aynı partide, U5).
e. **Gazlar MADDE_TIPLERI T5'te kalır.** Gaz kanalı (`gas_channel`) ayrı mimari soru.

Durum değişince başlıktaki etiket güncellenir; şimdilik TASLAK (kod U0 dışında yazılmadı).

## 10. Uygulama notları (2026-10-09, tek parti)

Kullanıcı U1–U5'i tek build'de istedi. Kararlar ve sapmalar:

- **Sahiplik sırası: kaynak > bağlı madde > Default Substance.** Taneler kaynağın madde
  etiketini taşır; Default Substance yalnız etiketsiz parçacıklar içindir. İlk sürüm
  Default Substance'ı öne alıyordu ve kaynakta türetilmiş bir madde varken Sand'in malzemesini
  koşturuyordu.
- **Bir domain'de bir tane malzemesi.** GPU tek bir malzeme seti taşır; ikinci DEM maddesi
  `grain_ownership.notes`'ta söylenir. Tane başına malzeme ayrı bir shader işi.
- **`grain_water_capacity_fraction` varsayılanı 0** (hiç ıslanmaz). Eski `wet_grains`
  varsayılanı kapalıydı; ıslak davranışı korumak için kapasite açıkça verilir.
- **Doğum nemi kaynakta** (`flow_source.grain_birth_saturation`), maddenin değil.
- **U4 kısmi:** Matter sekmesindeki tane bloğu salt okunur özet oldu, kaynak panelinde çözücü
  satırı ve "Wet at birth" var. Domain panelinin Kap/Kalite/İçerik/Kaynaklar düzeni (4000+
  satırlık dosya) yapılmadı: CLAUDE.md'nin büyük dosya kuralı gereği ayrı parti.
- **U5:** sıkışma sertleşmesi mevcut `pv` (sıkılık) durumunu kullanır; yeni buffer yok, push
  constant'ın iki boş alanı kullanıldı. `melt_product` yazılmadı: Ice de bugün kendi etiketiyle
  eriyor, kar da öyle; ayrı karar.
- **Testler:** `rt_grain_material.py` eski grain anahtarlarını türetilmiş test maddesine çevirir;
  test mantığı değişmedi. Malzemeyi doğrudan sınayan geçersiz anahtar döngüsü ham çağrıyla
  sunucunun reddini doğrular.
