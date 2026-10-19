# Birleşik madde domain'i — gaz/sıvı ayrımı olmayan simülasyon ve render

> **Durum:** AKTİF — 2026-10-19 2. build: P0–P4 CANLI, `rt_matter_smoke` 10/10 PASS. 2026-10-19: Ek 11 kökü düzeltildi (yüzey↔hava filmi, `thermal_response` emekli) ve taneler termal zincire katıldı (tane+sıvı+hava+ateş tek domain); kaynakta, DERLENMEDİ — devir tablosu aşağıda. 2026-10-18: FG1 kaynakta (yanma hava sıcaklığı + buhar oksijen seyreltmesi, eski söndürme söküldü; derlenmedi). Hızlı tur: regresyonlar PASS, Ek 11 3/6, MSF alan sızıntısı düzeltildi (derlenmedi). 2026-10-11: T2 gizli ısı + yüzey yoğuşması (§4.7, MADDE_TIPLERI §6f), sis suyu/doygunluk (§4.8) ve yağmur: otokonversiyon, sis tavanı, düşüş, iniş (§4.8b) CANLI PASS; commit `01284ef`. Açık: sis/yağmur render'a bağlı değil (bulut işiyle), yağmur parsel doğumlarının yeri ölçülmedi. 2026-10-08: Madde T1/T3 kullanıcı build'i ve hızlı
> dış IPC kabulü PASS; legacy/save-open/görsel kabul ve T2 hal/model birleşmesi açık.
> C5/C6/C7 fiziksel yakınsama ve H1 maliyet kapıları bu testlerle kapanmaz.
> Uzman Gas/Liquid domain'leri T0 eşitlik ölçümü olmadan kaldırılmaz.

## Devir — 2026-10-19 sonu (P0–P4 YAZILDI, tek derleme + `rt_matter_smoke.py` bekliyor)

Kapsam ve kalanlar: [MATTER_ROADMAP_2026_10.md](MATTER_ROADMAP_2026_10.md),
test sırası: [NEXT_BUILD_CHECKS.md](NEXT_BUILD_CHECKS.md) en üst. Ayrıntı
aşağıda "Parti 2026-10-19 P1–P4".

## Devir — 2026-10-19 (1. build CANLI; 2. build'e üç düzeltme)

**Canlı (1. build, 2026-10-19 10:56):** madde 1/1, gaz↔tane 6/6, Ek 11 6/7
(fizik tamamı PASS), yüzey suyu 6/6, FG1 8/8, tane ısısı 7/7 (ama oran yanlış,
aşağıda). Üç düzeltme yazıldı, derlenmedi: `msf.fields` → `temperature_fresh`,
`scene.add_primitive` silinmiş ad hatası, taneler için kapasitesiz duvar
soğutması. Kısa tur: [NEXT_BUILD_CHECKS.md](NEXT_BUILD_CHECKS.md) en üst.

(1. build öncesi tablo:)

| Konu | Durum |
|---|---|
| Ek 11 kökü | `sim_msf_gather` kabuğu gövdeden bağımsız olarak hava sıcaklığına çiviliyordu (`thermal_response` 1–8 1/s, havaya enerji yazılmıyordu). Domain içindeki yüzler 294 K'de kalıyor, yalnız zeminin altındaki yarı ısınıyordu: "ısınan yüzler suya değmiyor" ölçümünün kökü buydu. Yerine iki cisim film geldi (aşağıda "Parti 2026-10-19"). |
| Taneler termal zincirde | `thermal_liquid` tane engeli kalktı. Tane↔sıvı, tane↔hava (gaz şeridi), tane↔tane ısısı `MatterGrainThermal`'da; hücreler arası iletim, MSF kabuğu ve Ice erimesi mevcut yollarda. |
| Gaz↔tane testi | Kod değil sahne: rüzgâr düşüş yolunun üstüne alındı, x kapısı düşüş penceresinde. |
| Test sırası | [NEXT_BUILD_CHECKS.md](NEXT_BUILD_CHECKS.md) 0–8. H1 paketi en sonda. |
| Açık | Tane erimesinin (Ice) canlı kanıtı yok. Tane domain'inde sıvı donmuyor, aşırı soğuyor (raporlanıyor). Ortam alan gevşemesi (`cooling_rate`) hâlâ kapasitesiz. `fluid_extinguishing` kararı açık. |

## 2026-10-08 ortak saat kaynak devamı

2026-10-10 gaz/termal performans: termal-açık sahnede CPU kg/J taşınımı
713/823 ms. Kg/J advection, korunum ve sahiplik GPU residency yoluna taşındı;
sürekli tam-grid host mirror kaldırıldı. Surface heat/evaporation GPU envanterini
doğrudan kullanıyor. Kaynak hazır; tek kullanıcı build/canlı kabul bekliyor.
Mevcut parçacık yayını ve diğer termal yolların tam residency işi açık. Kapsam, kalanlar ve kesin checklist: [GAS_INVENTORY_GPU.md](GAS_INVENTORY_GPU.md).

MPM/DEM geometri ve teması cihazda ortak mikro saat; continuum grid transferi
kendi CFL/elastic sıklığında. Sıvı hız/topak tepkisi GPU'da her tick güncel.
Keyfi contact 1M/64-neighbour kapıları kaldırıldı, measured degree relaxation ve
2-D dispatch eklendi. Yeni kaynak henüz derlenmedi; önceki contact smoke PASS.
Frame CFD-DEM support, porous owner izolasyonu ve cinematic yakınsama/maliyet
kapıları açık. Ayrıntı: [son devir](MADDE_T2_T3_HANDOFF.md).

## 2026-10-08 ortak madde state / sahiplik kaynak checkpoint

Toplu kullanıcı onayıyla küçük adım onayları kaldırıldı; kaynak planı önce,
kabul testleri en sonda ve build kullanıcıda. `MatterSubstanceState` ortak
model/obstacle/ν(T) çekirdeği ve `granular_transport` sahipliği kaynakta.
**Ana plan kapanmadı:** MPM↔grain için DEM alt adımı başına iki yönlü Vulkan
contact kaynakta; önceki kullanıcı build PASS, yeni shader/C++ kabulü açık.
Ortak zamanlayıcı/porous owner izolasyonu, karma elastik MPM, gaz↔grain ve T6 kapıları açık. Güncel kaynak/kabul
ayrıntısı [MADDE_T2_T3_HANDOFF.md](MADDE_T2_T3_HANDOFF.md)'nin ilk checkpoint'idir.

## 2026-10-18 hızlı canlı tur (exe 10-18 11:16, boş sahne)

| Konu | Sonuç |
|---|---|
| Regresyon | Madde kütüphanesi, domain maddesi, ısı alışverişi 8/8, sis+yağmur 10/10, kaynama 6/6, değişim defteri PASS (kütle/enerji hatası 0). Füzyon/yoğuşma 8/8 (500 sn zaman aşımını aşıyor; sınırsız koşuldu). |
| Defter testi bayattı | `rt_test_matter_exchange_ledger_ipc.py` 10-02'den beri 7 türü tanıyordu; kod 11 tür üretiyor (`absorption`, `drainage`, `autoconversion`, `precipitation` eksikti). Test güncellendi — kod hatası değil. |
| Ek 11 sıvı↔collider ısısı | 12:32 build'inde yine **3/6**. `thermal_depth_m` = 0,100 m, `surface_area_m2` = 2,16: kabuk modeli DOĞRU, aşağıdaki "1 mm sac" hipotezi YANLIŞ çıktı. Kalan şüpheli: ıslak hücrelerin gördüğü kabuk sıcaklığı (host kopyası/readback gecikmesi ya da ısınan yüzlerin suya değmemesi) — ölçen sayaç yok. Eski metin: **3/6** (10-10: 4/6). Kabuk 349,3 K (eşik 350), su→katı ısı −17,8 J'den −1,8 J'ye sönüyor; beklenen ~−400 J/adım. Sönüm eğrisi 1 mm sac modeline uyuyor; `thermal_depth_m` dışarıdan okunamadığı için kök kesin değil → `msf.fields`'e `thermal_depth_m` + `surface_area_m2` eklendi (derlenmedi). |
| ★ MSF alan sızıntısı | **CANLI PASS (12:32 build):** silinen küpün alanı domain'li ilk adımda budandı. Domain yokken `fluid.step` senkron koduna ulaşmaz, alan o zamana kadar listede kalır (beklenen). Kök: `MaterialStateFieldSystem::endSyncPass` hiç çağrılmıyordu (git geçmişinde de hiç): silinen collider'ın alanı sahnede sıcak kalıyor, `msf.fields`'te listeleniyor ve sıvıyla ısı alışverişine girebiliyordu. Kanıt: test temizliğinden sonra `T_ContactCube` 324 K ile duruyordu. Çağrı eklendi (derlenmedi). |
| FG1 | Kaynakta yazıldı, derlenmedi: aşağıdaki "FG1 kaynak teslimi". Eski fenomenolojik söndürme söküldü. |
| Not düzeltmesi | "Öncelikli sonraki düzeltmeler" 1. madde (rejime göre yoğunluk) kodda yapılmış: `densityForParticle` Granular→`density`, Fluid→`liquid_density`. |

## Güncel kısa devir — 2026-10-11

| Konu | Güncel durum |
|---|---|
| Son partiler | T2 gizli ısılı erime/donma + yüzey yoğuşması (§4.7); sis suyu: buhar/sis iki havuz, gizli ısılı doygunluk ayarı (§4.8); yağmur: Kessler otokonversiyonu, sis tavanı, uç hızla düşüş, sıvıya iniş/emilim (§4.8b). Hepsi derlendi, canlı PASS, `main`'de (`01284ef`; `.spv` artık izlenmiyor, `6b6c643`). Parti 4 (§4.8c: sanal sıcaklık dönüşümü, yoğuşmayan gaz havuzu, rüzgârla yağmur) CANLI PASS (rüzgâr kanıtı zayıf). |
| Canlı kanıt | `fusion_condensation_live.json`, `fog_live.json` (9/9: sis ortalaması ≤ 0,0031 kg/m³, su korunumu 6e-6 kg), kaynama, GPU envanter, ısı alışverişi regresyonları. |
| Açık | (1) Sis ve yağmur render'a bağlı değil — bulut ajanının `CloudPrecipitation` render tarafıyla ortak karar. (2) Yağmurun ~%76'sı emilim yerine parsel doğumuyla sıvıya katılıyor; doğum YERİ ölçülmedi (kaynak üstü mü, tank dibi mi). (3)–(5) parti 4'te kapandı (§4.8c); rüzgâr ölçümü yan duvarları açık sahnede tekrarlanmalı. (6) Ek 11 sıvı↔collider ısısı: son kanıt 2026-10-10, 4/6 (`shell_is_a_body`, `water_warms_through_cube` FAIL) — yeniden koşulmalı. |
| Sonraki | Doğum yeri ölçümü (küçük) veya sis/yağmur görünürlüğü (bulut işiyle). |

## Önceki kısa devir — 2026-10-08 (tarihsel)

| Konu | Durum (o tarihte) |
|---|---|
| Son kaynak partisi | T1 madde kalıtımı + T3 Varsayılan Madde, fizik/hints ayrımı, preset/chemistry sökümü; kullanıcı derledi. |
| Hızlı canlı kabul | Madde library testi (21 yerleşik) ve domain substance sözleşmesi PASS; test nesneleri temizlendi. Simülasyon adımı/görsel kabul yapılmadı. |
| Kaldığımız kaynak noktası | T2: Initial Model ve binding Phase/Constitutive sökümü, madde+sıcaklıktan model/hal, Frozen↔solid mask tek yolu, Ice→Water göçü. |
| Açık T1/T3 kabulü | Save/open, legacy domain göçü, numerical tuning korunumu, Wax/MSF viskozite görseli ve node geri alma. |
| Kanonik madde planı | [MADDE_TIPLERI_TASARIMI.md](MADDE_TIPLERI_TASARIMI.md) §6a–§6d; T4/H1 B11 ve T5/T6 sonraki kapılar. |
| Ayrıntılı devam/devir | [MADDE_T2_T3_HANDOFF.md](MADDE_T2_T3_HANDOFF.md), [NEXT_BUILD_CHECKS.md](NEXT_BUILD_CHECKS.md), [canlı kanıt](madde_t1_t3_live_2026-10-08.json). |

Önceki kısa devir ve aşağıdaki tarihli kaynak checkpoint'leri tarihsel kayıttır;
güncel madde partisinin durumunu bu tablo ve bağlantılı devir belirler.

---

## Önceki kısa devir — 2026-10-05 (tarihsel)

| Konu | Durum |
|---|---|
| C5 canlı ölçüm | Sonlu Water/Sand sahnesi; emisyon sonrası su farkı -0.71 mg / 180 adım, kuru Sand sabit. Drenaj/contact/kimlik probları PASS; tüm yakınsama matrisi kapanmadı. |
| Yeni kaynak partisi | C6 wet/drainage/mixed GPU probları PASS; emisyon sonrası 180 adımda su +0.32 mg, kuru Sand sabit. 8-band wet materyaller üretildi; tam fiziksel kabul açık. |
| Sonraki iş | C6 dry/damp/saturated ve spatial/cache/dt/resolution kabulü, ardından C7. Sağ başlangıç kuru kontrolüne uzun koşuda su ulaşır. Tam pressure PDE/drag/buoyancy açık. |
| Kanonik not | [MATTER_C5_HANDOFF.md](MATTER_C5_HANDOFF.md), model: [MATTER_WET_RESPONSE.md](MATTER_WET_RESPONSE.md). |
| Kapanış ve sonrası | [Kalan fizik kabulü ve profile/emitter/domain/UI veri sahipliği](MATTER_ACCEPTANCE_AND_AUTHORING.md). Yeni spatial kabul ölçümleri kaynakta hazır; kullanıcı derlemesi bekler. |
| H1 parti sırası | [MATTER_H1_GRAIN_ROADMAP.md](MATTER_H1_GRAIN_ROADMAP.md) (2026-10-06): B2 kova/repose + B3 taşıyıcı başına sahiplik ve su–tane sürükleme/kaldırma kaynakta, tek build bekliyor; sonra B4 GPU yerleşikliği (H1-C7), B5 hacim dışlama, B6 ıslak tane, B7 DEM–XPBD kararı (2026-10-06: XPBD söküldü, DEM tek tane çözücüsü). |
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
| H1-M | Tek domain'de üç sahip: sıvı + grain + MPM (madde başına sahip, MPM↔tane temas). Ayrıntı: [H1 yol haritası B11](MATTER_H1_GRAIN_ROADMAP.md#b11--h1-m-tek-domainde-üç-sahip-sıvı--grain--mpm) | Kum çamura çarpar, içinden geçmez; kütle/momentum kapalı; grain'siz domain ve saf grain domain regresyonsuz. Sessiz-geçme (taneler MPM'in içinden geçiyor) ayrı kol. |
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
| sıvı → gaz | kaynama, kaynama altı buharlaşma, yanma ürünü, `mist` | hepsi var (2026-10-10, §4.7) |
| ısı | sıvı ↔ gaz yüzey filmi, sıvı ↔ collider kabuğu (MSF), sıvı içi iletim + konveksiyon | var (2026-10-10, §4.7) |
| gaz → sıvı | rüzgar/gaz hızıyla sürükleme (sprey, mist) | rüzgar + düşük kütleli mist gaz sürüklemesi var |
| sıvı → gaz | sıvı hücreleri gaz için hareketli sınır | 27D kodda; canlı kabul bekliyor |
| sıvı ↔ katı | donma / erime | var (donma: yüzey + temas çekirdeği; erime MSF ve madde) |
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
| Donma / erime | sıvı ↔ katı (`frozen`, iskelet modeli) | var; 2026-10-11'den beri gizli ısı öder (`fusion_solid_fraction`, §4.7) |
| Buharlaşma / kaynama | sıvı → gaz | kaynama kaynakta (2026-10-10, `transferBoilingToGas`: aşırı ısı → gizli ısı, buhar aynı domain'in gaz fazına, `vaporization` defter kaydı); kaynama altı buharlaşma yok |
| Yanma | sıvı → gaz (+ısı) | var |
| Yanma / piroliz | katı → gaz + char (MSF) | kısmen (yangın → yapı, MSF char) |
| Yumuşama / sinterleme | granül ↔ katı / sıvı | kısmen (granül termal zinciri) |
| Yoğuşma | gaz → sıvı | yüzeyde var (2026-10-11, ters Hertz–Knudsen, CPU+GPU); havada sis suyu §4.8, otokonversiyon + yağmur düşüşü/inişi §4.8b (CANLI PASS) |
| Sis aktarımı | sıvı `mist` → gaz | 27C canlı PASS; 2026-10-11: mist sis suyu havuzuna gider, gizli ısı doygunlukta ödenir (§4.8) |

★ Enerji de defterdedir: erime ve buharlaşma gizli ısıyı kaynaktan çeker.
Bugün erime ısı tüketmiyorsa, erime sonsuz hızda ve bedava olur — sonuç
"makul görünür" ve kimse bug diye raporlamaz.

### 4.7 ★★★ Isı ve faz geçişleri — 2026-10-10 (CANLI, Ek 1–11; Ek 11 kabulü açık: 2026-10-10 kanıtı 4/6)

Her şey parçacığın KENDİ maddesinden okunur: ısı sığası `m·c`, iletkenlik `k`,
yoğunluk `ρ`, gizli ısı `L`, buhar molar kütlesi `M` (`vapor_molar_mass_kg_mol`),
sıvı genleşmesi `β` (`thermal_expansion_per_k`), `Tb`, erime noktası. Karışık bir
domain'de su, yağ, erimiş metal kendi değerleriyle alışveriş eder. Enerji her
alışverişte iki cisim kesin çözümüyle korunur (taşma yok).

| Süreç | Uygulama | Kanıt |
|---|---|---|
| Doğum fazı | Kaynak/Seed/Fill: erir katı (Snow, Ice) ortam sıcaksa erime−10 K'de, sıvı kaynama noktası üstünde doğmaz (Tb'de doğar) — `substanceAmbientBirthKelvin` | canlı PASS |
| Termal alan | Tek `ThermalSource` (96 B float, yerel matris + şekil) MSF ve madde parçacığı için; Box/Cylinder/Cone şekle uyar; eski int→float okuma hatası (MSF falloff hep None) düzeldi | canlı PASS |
| Sıvı içi iletim | `conductThermalLiquid`: hücre ısı sığası + harmonik k, yüz başına kesin çözüm | heat_exchange PASS |
| Konveksiyon | `applyThermalConvection`: Boussinesq `dv=−g·β·(T−Tort)·dt` | boiling `convects` PASS |
| Kaynama | `transferBoilingToGas`: `dm=min(m, m·c·(T−Tb)/L)`, Tb'ye sabitler, buhar sütun yüzeyinin ÜSTÜNE bırakılır (sıvı hücresine değil — RT'de süt bloğu buydu), sanal sıcaklıkla kaldırma `Tv=Tb·M_hava/M`, yanıcı buhar gaz yakıtına; çekirdekli kaynama: boşluk oranıyla fokurdama + seyrek whitewater Bubble | boiling PASS |
| Kaynama altı buharlaşma | `exchangeLiquidGasSurface`: Hertz–Knudsen + Clausius–Clapeyron, nem = atmosfer + hücredeki buhar, gizli ısı parçacıktan; etkin katsayı 4e-5 (açık tava 1e-4 kg/m²s) | heat_exchange PASS |
| Sıvı ↔ hava | Madde domain'inde hava = gaz ızgarası: yüzey parçacığı ↔ üst gaz hücresi, film 20 W/m²K, C_hava=ρ_hava·V·1005; ortama sonsuz soğuma kapalı. Ayrı Fluid+Gas üst üste binince de aynı (Ek 11) | heat_exchange PASS |
| Sıvı ↔ collider | Ek 11: MSF kabuğu (alan·1 mm·ρ·c, MSF'nin kendi kütle kuralı) ↔ temas parçacığı, film 500 W/m²K; sıvı tarafı bir readback geriden okur, kabuğa giden ısı `sim_msf_ambient` binding 4 ile cihaza EKLENİR (durum ezilmez) | `rt_test_contact_heat_ipc.py` 2026-10-10: 4/6 — ısı katıdan sıvıya akıyor, ama `shell_is_a_body` ve `water_warms_through_cube` FAIL; yeniden koşulmadı |
| Donma | Kapı: thermal liquid VEYA ısı alanı (domain `granular_enabled` artık kapatmaz); serbest yüzey de çekirdek (göl yüzeyden donar) | elle doğrulandı (zemin); yüzey Ek 11 |
| Gizli ısı (füzyon) | 2026-10-11: `fusion_solid_fraction` F; Tm'de parsel bekler, `dF = c·ΔT/L`; hal = F ≥ 0.5 (Water/Wax çivili, Snow/Ice model değişir); defter gerçek ödenen ısıyı yazar. Domain donma aynası söküldü | CANLI PASS 2026-10-11: 997 kg tank tam donar, gizli ısı 332 997 993,59 J = defter 332 997 993,53 J |
| Yüzey yoğuşması | 2026-10-11: hava hücresindeki envanter buharı (pv ≤ 1 atm) yüzey sıcaklığındaki doygunluğu aşınca ters HK akısı; buharın kendi enerjisi (e/m) parsele; önce kaybettiği kütleyi doldurur, sonra `rest_mass_kg` büyür, 2× kanonikte ikiye bölünür (`mass_fraction ≤ 1` değişmezi korunur). Defter `condensation` (kaynak gaz domain'i). CPU referansı + `sim_inventory_surface_exchange.comp` | CANLI PASS 2026-10-11 ama ★ölçek yanlış: 68 kg kaynadı, 0,010 kg yoğuştu (aşağıda) |
| Yanan sıvı | Görsel ısı eki söküldü; buharlaşan yakıtın gizli ısısı parçacıktan, alev ısısı gaz filmiyle geri | Ek 11 |
| Canlı düzenleme | Kuvvet alanı değişikliği koşuyu sıfırlamaz: `invalidateSimCacheAfterCurrentFrame` | Ek 11 |

IPC (`fluid.list_domains`): `thermal_field_heated_particles`, `thermal_convecting_particles`,
`thermal_conducting_cells`, `thermal_gas_exchange_particles`, `thermal_heat_to_gas_j`,
`thermal_evaporated_kg`, `thermal_solid_contact_particles`, `thermal_heat_to_solid_j`;
`flow_source.list`: `fluid_birth_kelvin`, `fluid_born_solid`; `fluid.seed fill_level`;
defter `matter.exchanges` (`vaporization`). Testler: `rt_test_snow_ipc.py`,
`rt_test_boiling_ipc.py`, `rt_test_heat_exchange_ipc.py`, `rt_test_contact_heat_ipc.py`.

**★ Canlı ölçümün gösterdiği (2026-10-11, `fusion_condensation_live.json`):**
(1) envanter basınçla sınırlı değil — 8 m³ kutuda 68 kg buhar ≈ 8,5 kg/m³,
1 atm doymuş buharın (~0,6 kg/m³) 14 katı; (2) yoğuşma katsayısı 4e-5 havaya
difüzyon sınırlı buharlaşmaya kalibre, saf buharın soğuk yüzeye yoğuşması
yüzlerce kat hızlı olmalı (sınır gizli ısının uzaklaştırılması). Test yalnız
">0" sınadığı için PASS verdi.

**Açık (bilinçli):** hacimsel yoğuşma (sis/yağmur) — ölçek engeli: bir sıvı
parseli (ρh³/ppc ≈ 8 g @ 4 cm) bir gaz hücresinin fiziksel olarak taşıyabileceği
buhardan (doygunlukta ~mg) binlerce kat büyük; "aşırı doymuş hücre → damla"
hiç tetiklenmez. Doğru yol bulut modellerindeki doygunluk ayarı + hücre başına
taşınan bir **kondensat (sis suyu) alanı**; bu, GPU envanter bankasına (aşağıda)
üçüncü taşınan alan demek, derlenmemiş koda kör eklenmedi. Ayrıca
`transferMistToGas` mist parselini buhara çevirirken gizli ısıyı envantere
**ekliyor ama hiçbir yerden almıyor** (defter "ödendi" diyor) — aynı ölçek
sorunu yüzünden basit düzeltmesi yok, kondensat alanıyla birlikte ele alınmalı.
Gaz içi iletim; gaz havasının fiziksel enerji
envanteri (bugün sıcaklık alanı + hücre ısı sığası); DEM tanelerinin sıcaklığı
([HANDOFF_DEM_THERMAL.md](HANDOFF_DEM_THERMAL.md)); erimiş/donmuş "buz" rijit gövde değil,
sabitlenmiş parçacık ([BUZ_MODEL_KARARI.md](BUZ_MODEL_KARARI.md)); sıvı↔hava filmi
madde başına değil sabit (20 W/m²K). Kaynama sırasında 702 K'de kalan parçacıklar
gözlendi (kaynama kapısından düşen bir alt küme) — ölçülmedi.

### 4.7b Plana yazılmamış yapılar (diğer ajan, 2026-10-10, kodda var)

§4.7 tablosu yüzey ısısı/buharlaşmayı CPU `exchangeLiquidGasSurface` olarak
anlatıyordu; Vulkan yolunda o artık çalışmıyor. Gerçek yapı:

- **Gaz kg/J envanteri GPU'da sahiplidir** (`GasInventoryGpu.h`,
  `GasInventoryDevice.cpp`, `GasInventoryGpu.cpp`, shader'lar
  `sim_inventory_{advect_scalar,maccormack_scalar,conserve,clear,deposit,momentum,summary}`).
  Fizik adımında tam grid upload/download 0 byte; CPU üreticileri (kaynama,
  mist, yanma) `depositGasInventory` ile küçük benzersiz-hücre kayıtları kuyruğa
  koyar. Hata sessiz CPU'ya düşmez: `inventory_step_held`, reset gerekir.
  Belge: [GAS_INVENTORY_GPU.md](GAS_INVENTORY_GPU.md).
- **Yüzey ısı/buharlaşma (ve artık yoğuşma)**: `GasSurfaceExchangePlan.{h,cpp}`
  (plan/doğrulama/yayın, CPU-GPU ortak), `GasSurfaceExchangeCpu.cpp` (CPU
  referansı), `GasSurfaceExchangeGpu.cpp` + `sim_inventory_surface_exchange` /
  `_commit`. Hava hücresi seçimi `GasSurfaceGeometry` (açık yüzleri olan sütun).
  Nem: buhar havuzu tek tür (yoğuşabilir madde); yakıt buharı ve diğerleri
  2026-10-11'den beri ayrı yoğuşmayan havuzda (§4.8c).
- **Gaz faz momentumu**: `GasPhaseMomentum.{h,cpp}` — buhar/mist kütlesinin
  momentumu gaz hızına `sim_inventory_momentum` ile eklenir.
- **Whitewater çalışma zamanı**: `WhitewaterRuntime.{h,cpp}`.
- **Kare önbelleği**: `MatterGridFrameCache.h`, `LosslessFrameArray.h`,
  `GasInventoryFrameCapture` (kg/J kayıpsız RAM payload).
- Yüzey/UVW: `SurfaceParticleBins.h`, `SurfaceWorkTiles.h`
  ([SDF_SPARSE_WORK_OPTIMIZATION_2026_10_10.md](SDF_SPARSE_WORK_OPTIMIZATION_2026_10_10.md)).

Hepsi 2026-10-10 itibarıyla derlenmemiş/canlı kabul bekliyor (NEXT_BUILD_CHECKS).

### 4.8 Sis suyu: doygunluk ayarı ve kondensat — 2026-10-11 (CANLI PASS)

Kullanıcı kararı (2026-10-11): sis suyu **domain içinde**, atmosfer katmanıyla
**aynı birimlerle** (kg/m³ = kg / hücre hacmi, K). Bulut sistemiyle birleşme
ileride bir eşleme meselesi; şimdi bağlanmadı.

- **İki havuz.** `gas_phase_mass_kg` yalnız BUHAR; yeni `gas_phase_condensate_kg`
  havanın taşıdığı sıvı su (sis/bulut). `gas_phase_energy_j` hücredeki TÜM suyun
  enerjisi (buhar c·T + L, sis c·T). GPU bankası 3 havuz × 3 yuva; kondensat
  buharla aynı çekirdeklerle taşınır ve korunur (conserve ikinci çağrı), kare
  önbelleği + disk v13 + backend geçişi + anlık görüntü taşır.
- **Doygunluk ayarı** (`GasSaturation.{h,cpp}`, `sim_inventory_saturation.comp`):
  her gaz adımında hücre, hava sıcaklığıyla dengeye getirilir. Yoğuşan buharın gizli
  ısısı havayı ısıtır, buharlaşan sis havayı soğutur; miktar ısının doyma noktasını
  kaydırmasıyla birlikte ÇÖZÜLÜR (ikiye bölme), aşma yok. C = hava + suyun ısı
  sığası. 1 atm tavanı YOK (tavan kaçak ısınma yaratıyordu; aşırı basınçlı hücre
  basınçlı kap gibi dengeye gelir). Dünya nemi kütlesiz arka plan buharı.
  Python oracle: nemli hava 0,03 kg/m³ @293 K → 5,3 g/m³ yoğuşur, +9 K; kuru
  havada 5 g/m³ sis → 4,5 g/m³ buharlaşır, −8 K; 8,5 kg/m³ buhar ~470 K'de durur.
- **Yağış:** bir hücrenin sis suyu bir sıvı parseline (ρh³/ppc) ulaşınca, Matter
  domain'inin sıvı gridi içindeyse parsel olarak doğar (`precipitation` defter
  türü). Fiziksel sisle tetiklenmez; yoğun buharda döngüyü kapatır.
- **Mist düzeltmesi:** mist parselleri artık buhara değil sis havuzuna gider
  (faz değişimi yok); gizli ısıyı sonra doygunluk ayarında havadan öder. Eskiden
  L·m enerji hiçbir yerden alınmadan envantere ekleniyordu.
- **Yüzey HK katsayısı:** 4e-5 (açık tava, hava içinden difüzyon) hava payıyla
  ölçeklenir, saf buharda arayüz değeri 0,04'e çıkar (iki yön, CPU + shader).
  Yoğuşan buharın enerjisi c·T_hava + L (havuz artık tüm suyu tuttuğu için e/m değil).
- **Kapı:** Matter + thermal chain, ya da sis suyu taşıyan herhangi gaz domain'i.
  Yoğuşan madde: Default Substance (buhar verisi var ve yanıcı değilse), yoksa Water.
- **Açık:** sis suyu RENDER'a bağlı değil (görünür buhar hâlâ `gas.density`
  izleyicisi; bulut işiyle birlikte). ~~Tek tür; ≈600 K sanal sıcaklık~~ →
  §4.8c. Sis çökelmesi (damla düşüşü)
  ve sisin yüzeye çökmesi yok.
- **Canlı kanıt (2026-10-11, `rt_test_fog_ipc.py` 6/6 PASS, `docs/dev/fog_live.json`):**
  240 adımda 130,05 kg kaynadı + 2,47 kg mist; yüzey yoğuşması 3,09 kg (%2,4;
  önceki 0,010/68 kg), yağış 94,09 kg, havada 1,54 kg buhar + 33,81 kg sis.
  Su korunumu: havada 35,343248 kg, beklenen 35,343257 kg (fark 9e-6 kg).
  Regresyonlar (T2, mist, kaynama, ısı, GPU envanter) PASS.
- **★ Sinsi — sis yoğunluğu fiziksel değil:** sis tutan hücrelerde ortalama
  **41,7 kg/m³** (gerçek sis/bulut 1e-4–1e-3 kg/m³). Yağış eşiği bir sıvı parseli
  (ρh³/ppc ≈ 250 kg/m³ bu kurulumda) olduğu için asılı damla sınırı yok; sıvı
  havada %4 hacimle süzülüyor. Test korunumu ölçüyor, bunu ölçmüyor. Çözüm adayı:
  hücre başına asılı su tavanı (otokonversiyon, Kessler tipi, ~1e-3 kg/m³ üstü
  yağmur suyuna) + yağmur suyunun çökme hızıyla düşmesi; kaynaktaki +400 K alan
  zorlaması ölçeği büyütüyor ama sınırın yokluğu ölçekten bağımsız.
  → **Parti 3'te çözüldü (§4.8b, CANLI PASS): sis ortalaması ≤ 0,0031 kg/m³.**

#### 4.8b Yağmur: otokonversiyon, sis tavanı, düşüş, iniş — 2026-10-11 (CANLI PASS)

- **Üç domain parametresi** (`APICSolverParams`, kg/m³ — atmosfer birimi; API
  `fluid.set_param`, panel Thermal Liquid altında, proje + sahne kaydı + hash):
  `fog_autoconversion_kg_m3` = 1e-3 (Kessler eşiği), `fog_autoconversion_rate`
  = 1e-3 1/s (Kessler k1, örtük: (w−eşik)(1−e^(−k dt))), `fog_ceiling_kg_m3` = 5e-3
  (yoğun kümülüs; üstü anında yağmura). Yalnız sıvı içeren domain'de
  (`setFluidThermal`) yazılabilir; saf gaz domain'i varsayılanı kullanır.
- **Yağmur havuzu:** envanterde rüzgârla TAŞINMAYAN iki yuva — yağmur kütlesi [9],
  yağmur enerjisi [10] (kendi duyulur enerjisi; paylaşılan enerji havuzu rüzgârla
  gider). Banka 11 yuva; kalıcı yuvalar `kPersistentSlots = {0,3,6,9,10}` tek
  listeden snapshot/upload/arka uç değişimi. Disk önbelleği **v14** (v11–13 okunur).
- **Düşüş** (`sim_inventory_rain_fall`, CPU `fallRain`): sütun başına bir iş
  parçacığı, Kessler/Klemp–Wilhelmson uç hızı 36,34·(ρ_r[g/cm³])^0,1364 m/s
  (1 g/m³'te 5,5 m/s; 10 m/s tavan), alt adımla hücre başına en çok bir hücre,
  aşağıdan yukarı tarama. Durduğu yerler: katı hücrenin üstü (`gas_solid_mask`),
  ızgara tabanı, yüzey değişiminin **iniş damgası** vurduğu hücre.
- **İniş:** yüzey değişimi her yüzey parselinin hava hücresini o adımın
  damgasıyla işaretler (`gas_rain_landing`, adımda bir ilerleyen
  `gas_rain_stamp`) ve hücredeki yağmuru parsellere yüzey payıyla verir
  (duyulur, faz değişimi yok; `GasSurfaceResult::rain_kg/rain_energy_j`,
  commit yağmuru sıfırlar). Sıvı olmayan yerde (taban) yağmur birikir, bir sıvı
  parseli kütlesine ulaşınca parsel doğar (eski sis yağışı artık yağmur
  havuzundan). Defter: `autoconversion` `:fog→:rain`, `precipitation`
  `:rain→domain` (doğum ve emilim).
- **Ölçüm:** `fog_autoconverted_kg`, `rained_kg` (doğum+emilim),
  `rain_absorbed_kg`, `gas_rain_kg`, `gas_rain_cells`.
- **Açık:** ~~yağmur rüzgârla sürüklenmez~~ (§4.8c); açık tabanlı domain'de de tabanda
  birikir (dışarı akmaz); dar bir yüzey hücresini atlayan yağmur sıvının içinden
  tabana iner ve orada parsel olur (kütle korunur, konum kaba). Bulut ajanının
  `CloudPrecipitation` (`rate_mm_h`) render tarafı; bu fizik tarafı — bağlantı
  bulut işiyle.
- **Canlı kanıt (2026-10-11, `rt_test_fog_ipc.py` 9/9 PASS + regresyonlar):** 96,19 kg
  kaynadı; sis ortalaması en çok **0,0031 kg/m³** (tavan 0,005; önceki 41,7);
  otokonversiyon 94,08 kg, yağmur→sıvı 85,82 kg (20,83 kg yüzeyde emildi, ~65 kg
  parsel doğumu), havada 1,13 kg buhar + 0,0007 kg sis + 8,26 kg düşen yağmur
  (2713 hücre; ~23 kg/s üretim × 2 m düşüş süresiyle tutarlı). Korunum: 9,383396 /
  9,383403 kg (fark 6e-6). **Ölçülmedi:** parsel doğumlarının YERİ — kaynak üstünde
  yoğun yoğuşmanın havada doğurduğu yağmur parseli mi (beklenen), yoksa yüzeyi atlayıp
  tank dibinde biriken yağmur mu (sinsi durum).

#### 4.8c Sanal sıcaklık, yoğuşmayan gaz, rüzgârla yağmur (2026-10-11, CANLI PASS)

Madde planının açık listesindeki 3, 4 ve 5. maddeler. Kontrol listesi:
[NEXT_BUILD_CHECKS](NEXT_BUILD_CHECKS.md).

- **Sanal sıcaklık (3).** Gaz sıcaklık alanı kaldırmayı sürüyor; kaynama buharı
  oraya Tv = T·Ma/M_karışım olarak yazılıyor (saf buhar 373 K → ~600 K). Bu
  bilinçli bir seçim: kaldırma üç ayrı yolda (CPU yoğun, sparse, shader) aynı
  formülle hesaplanıyor ve bileşime dayalı bir kaldırma terimi yok. Gerçek
  sıcaklığa ihtiyaç duyan okuyucular, yani doygunluk ayarı ve yüzey
  alışverişi (CPU ve shader), hücrenin buhar kütlesiyle T'ye çeviriyor:
  - Dönüşüm: `airKelvinFromVirtual`, kapalı form; x ≥ 1 ise saf buhar.
  - Alt sınır: sonuç `min(Tv, ortam)` altına inmez. Alanın ortama soğuması
    (dissipation) karışıma giren ortam havasını temsil ediyor; yoksa soğumuş
    buhar 182 K gibi okunurdu.
  - Geri yazma: değişiklik alana bir Tv FARKI olarak yazılır. Böylece alt
    sınırdan doğan sapma alanda kalır ve sıçrama olmaz.
  - Sonuç: yoğuşan buhar kaldırmasını kaybeder, açığa çıkan gizli ısı havayı
    ısıtır. Kaynamanın pay ile karıştırması doğrusal bir yaklaşım; x = 0,5'te
    gerçek Tv'den yaklaşık %5 sapar.
- **Yoğuşmayan gaz (4).** Envantere 12. yuva eklendi: `gas_phase_foreign_kg`.
  Enerjisi ortak enerji havuzunda tutuluyor.
  - Bu havuza girenler: yanma alışverişinin yakıt buharı ve yoğuşabilir madde
    dışında kaynayan her şey. Doygunluk ayarı bu havuza hiç dokunmaz.
  - Yüzeyde Hertz–Knudsen buharlaşması ve yoğuşması artık yalnızca yoğuşabilir
    maddenin parselleri için çalışıyor (planda diğerlerinin buhar verisi
    sıfırlanıyor). Önceden her madde su buharı havuzuna buharlaşıyordu.
  - Bankanın taşıdığı yerler: kare önbelleği, disk **v15**, arka uç geçişi,
    özet. IPC alanları `gas_foreign_kg` ve `gas_foreign_cells`.
  - Mist damlası: yalnızca yoğuşabilir maddeninki sis suyu; başka sıvının
    damlası (yanan yağ spreyi) yoğuşmayan havuza, yalnız duyulur enerjiyle.
    İlk canlı koşu bunu yakaladı: 0,016 kg yağ damlası suya dönüşmüştü.
  - Bilinen sınırlar:
    - Yoğuşabilir olmayan, yanıcı da olmayan bir sıvı yüzeyden buharlaşmıyor.
    - Yabancı gazın Tv izi dönüşümde sayılmıyor.
- **Rüzgârla yağmur (5).** Bir damla, havanın hızı artı uç hızla düşüşü kadar
  hareket eder. Yağmur kütlesi [9], enerjisi [10] ve yoğuşmayan gaz [11], ana
  havuzlardan sonra semi-Lagrangian olarak taşınıyor. Hedef olarak ana
  havuzların artık boşalmış scratch yuvaları (1/4/7) kullanılıyor. Korunum
  çekirdeği yağmuru (kütle, enerji) çifti olarak normalize ediyor. MacCormack
  kullanılmıyor, çünkü yağmur sütununun keskin kenarında taşma yapar. Düşüş
  geçişi değişmedi. Yukarı akım yağmuru tutar; bu fiziksel bir davranış. CPU
  referansı aynı. Ölçüm alanı `gas_rain_centroid` (özet indirgemesi ağırlıklı
  konumu zaten hesaplıyordu).
- **Ölçülmedi (2):** yağmur parsellerinin doğum yeri. Hâlâ açık.
- **Canlı kanıt (2026-10-11):**
  - `rt_test_fog_ipc.py` 10/10 PASS. Havada kalan buhar 1,13 kg'dan 0,65 kg'a
    indi (kaynayanın %1,2'sinden %0,7'sine): Tv dönüşümü beklenen yönde.
  - `rt_test_gas_species_ipc.py` 4/4 PASS: 2065,528 kg yakıt buharı ve damlası
    yoğuşmayan havuzda (fark 1e-4 kg); su buharı ve sis 0.
  - Rüzgâr, `--wind 4`: PASS ama ZAYIF. Kapalı kutuda tekdüze cisim kuvveti saf
    bir gradyandır, basınç projeksiyonu onu tamamen siler (ilk koşu: kayma
    0,0001 m). Açık sınırla merkez +0,53 m kaydı, ama taban da açık olduğu için
    tank boşaldı ve yağmur kütlesi çok az kaldı. Temiz ölçüm için yalnızca yan
    duvarları açık bir sahne gerekiyor (açık).

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

## 2026-10-10 karar — 2026-10-11 öncelikleri

Kullanıcı aşağıdaki iki açık işi yarın ele almak üzere ana plana bağladı.
Bu kayıt uygulama veya canlı kabul değildir; diğer açık kapılar korunur.

### 1. DEM uyku doğruluğu — mevcut revizyon 23/24 işinin devamı

Yeni bir uyku projesi açılmaz. Başlangıç kaydı
[Revizyon 23 — uyku canlı A/B (2026-10-09)](grain_sleep_ab_rev23/REPORT.md),
güncel kaynak devamı [DEM_UYUYAN_TANELER.md](DEM_UYUYAN_TANELER.md) içindeki
revizyon 24 ve [DEM_SLEEP_TRANSFER_IMPLEMENTATION.md](DEM_SLEEP_TRANSFER_IMPLEMENTATION.md).
Rev23 kısa sleep ve açı paritesi PASS olsa da açık repose tane boyutu kapısı
FAIL olduğundan tam kabul açık; rev24 kaynak durumu canlı kabul değildir.

Kullanıcının hareketsiz parçacıkların uyumasıyla ilgili bugünkü sorunu problemli
sahnede yeniden üretilir; kök neden henüz kesinleşmedi. Durgun/desteklenen tane
uyumalı, desteksiz düşüş uyumamalı; darbe, destek kaybı, hareketli collider ve
dış kuvvet doğru uyanmayı sağlamalı. Sıvı/MPM birlikteliğinde uyku politikası
etkileşim kuvvetlerini kaybetmemeli. Mevcut sleep/transfer, repose ve büyük-N
A/B kapıları ile kullanıcı sahnesi geçmeden bu iş kapatılmaz. Uyuyan/uyanık
sayıları, wake nedenleri, enerji, temas ve maliyet aynı core üzerinden
UI/Python/IPC'de izlenebilir olmalı; cache/reset davranışı doğrulanmalı.

### 2. FG1 — Matter içi sıvı–gaz etkileşimi ve söndürme kabulü

2026-10-10 salt okunur denetimde aynı Matter domaininde
`liquid_moving_boundary` çalışırken eski `runGpuFluidSurfaceQuench` aynı domaini
atlıyor ve yalnız ayrı Fluid/Gas Vulkan çiftini kabul ediyor. Okuma anındaki
emitter Soil/MPM idi; bu gözlem Water ile canlı A/B kabulü değildir.

- Hareketli hacim sınırı, termal alışveriş, kimyasal söndürme ve faz/yanma
  aktarımı ayrı doğrulanır. Gazın kaplanan hücrelerden dışlanması gerçek
  soğutma veya alev kaynağının söndürülmesi kabulü yerine kullanılamaz.
- Water/APIC söndürmesi aynı Matter domainine bağlanır; seçim parçacığın
  substance_tag ve madde özelliklerinden yapılır. Domain varsayılan maddesi
  bütün karışımın kimyasını belirlememeli. Fazların farklı bounds/voxel
  eşlemesi ve GPU alan sahipliği korunmalı.
- Gaz→sıvı ısı ve sıvı→gaz soğutma, mevcut kaynama/yanma aktarımıyla ortak
  core/ledger sözleşmesinde tutarlı olmalı; kg/J iki kez yazılmamalı.
  DEM/MPM mekanik engellemesi ile maddeye bağlı termal/kimyasal etkiler ayrılır;
  çözücü adı tek başına kimya seçmez. Gaz/granülün eksik bağlantıları ayrıca
  raporlanır; mevcut hareketli sınır tam bağlantı kabulü sayılmaz.
- Sürekli gaz flow source ve yanıcı yüzey/MSF ayrı kabul kollarıdır: sıcaklık,
  fuel/flame azalması, su kaldırılınca sürekli kaynağın yeniden beslemesi ve
  yüzey moisture/ısı/yeniden tutuşma ölçülür. Kaynağın enabled alanının
  otomatik kapatılması söndürme tanımı değildir.
- Kabul matrisi: aynı Matter içinde ateş+Water/APIC, ateş+Sand/DEM,
  ateş+Soil/MPM, water+granül karışımı, temassız kontrol ve ayrı-domain referansı.
  Desteklenen backendler doğrulanır; desteklenmeyen yol sessiz no-op olmaz.
  sim_graph.couplings gerçekten çalışan işlemi gösterir. UI/Python/IPC aynı
  core servis, validation ve hata semantiğini paylaşır. Cache/rewind,
  ledger/termal ve render regresyonları geçmeden FG1 ve C7 kapatılmaz.

Uygulama odaklı modüllerde yapılır; 2000 satır üstü kaynaklarda yalnız küçük
entegrasyon olur. Build/shader derlemesi kullanıcıda; canlı testler uygulama
açıkken ayrı süreçte scripts/test/rt_ipc.py ile yürütülür.

### FG1 kaynak teslimi — 2026-10-18 (DERLENMEDİ)

**Bulgu 1 — yanma sanal sıcaklığı okuyordu.** Gaz sıcaklık alanı Tv tutuyor
(§4.8c); 373 K saf buhar orada ~600 K. Üç yanma yolu da (GPU `sim_gas_combustion`,
CPU yedeği, `GridFluid::processCombustion`) tutuşmayı bu alandan karşılaştırıyordu:
kaynayan suyun buharı yanmaya "sıcak hava" gibi görünüyordu. Hiçbir yol oksijen
de okumuyordu.

**Bulgu 2 — eski söndürme fizik değildi.** `runGpuFluidSurfaceQuench` gaz ısısını,
yakıtı ve alevi `(1 − fuel_capacity·dt)` ile çarpıyordu: oran sıvının *yüzey yakıt
kapasitesinden* geliyordu, çekilen ısı hiçbir deftere yazılmıyordu ve yalnız ayrı
Fluid+Gas çiftinde çalışıyordu. Aynı shader'ın yanıcı yarısı zaten ölüydü
(`fluid_extinguishing` kapısı olmadan dispatch edilmiyordu).

**Yeni model (`GridFluid::CombustionAir`, CPU referansı `GridFluid::burnFuel`, GPU
aynı matematik):**
- Tutuşma ve ısı salımı **hava sıcaklığında**: Tv, hücrenin buhar kütlesiyle
  `airKelvinFromVirtual` ile T'ye çevrilir; açığa çıkan ısı alana Tv FARKI olarak
  yazılır (doygunluk/yüzey alışverişiyle aynı sözleşme).
- **Buhar oksijeni seyreltir:** 1 atm'de hücrenin molü sabit, x_O2 = 0,2095 ·
  oxygen · (1 − x_buhar). Yanma hızı x_O2 = 0,13'te (buharla seyreltilmiş
  hidrokarbon LOC'u, N2 0,12 ile CO2 0,145 arası) sıfıra doğrusal iner.
- **Oksijen tek otorite:** MSF pirolizinin okuduğu dünya/domain oksijeni artık gaz
  yanmasına da uygulanıyor. Varsayılan 1,0 → mevcut sahneler değişmez.
- Isı çekme işi zaten defterde olan fiziklerde: yüzey ısı alışverişi ve kaynama
  (gizli ısı dahil). Söndürme seçimi maddenin buhar verisinden geliyor; domain
  bayrağı yok. Buhar yoksa (vapour_molar 0 / depo yok) davranış FG1 öncesiyle
  bit bit aynı.
- Sayaçlar `gas.step_stats` (+ Python, panel): `combustion_measured`,
  `combustion_burning_cells`, `combustion_inerted_cells`,
  `combustion_steam_masked_cells`, `combustion_min_oxygen_fraction`. GPU 16 bayt,
  aşamanın zaten beklediği fence'ten sonra.

**Söküldü:** `runGpuFluidSurfaceQuench`, `sim_fluid_surface_combustion.comp`,
`fluid_combustion_state` tamponu ve sıfırlama bayrakları, `gpu_fluid_combustion_ms`
/ `fluid_combustion_on_gpu` (API/IPC/Python/UI). Paneldeki "Extinguishing liquid
active" ve "extinguishes fire" yazıları kaldırıldı/düzeltildi (artık termal zincir
kapalıysa "inert to fire").

**Bilinen sınırlar:**
- Yoğuşmayan havuz (`gas_phase_foreign_kg`) seyreltmeye sayılmıyor: yakıt buharı
  ile yanma ürünü aynı havuzda, ayrılamıyor.
- Ortam nemi (envantersiz arka plan buharı) sayılmıyor (20 °C'de x ≈ %2).
- Oksijen tüketimi izlenmiyor (O2 alanı yok); seyreltme yalnız buhardan.
- Termal zincir kapalı bir sıvı artık ateşe hiçbir şey yapmıyor — eski ayrı
  Fluid+Gas sahnelerinde su ateşi söndürmeyi bırakabilir. Bilinçli: eski yol
  enerji yok ediyordu.
- GPU envanteri cihazdayken yanma CPU'ya düşerse kuru hava sayar (log'a bir kez).

**Açık karar (kullanıcı):** `fluid_extinguishing` madde bayrağı artık hiçbir
söndürmeyi sürmüyor; kalan tek etkisi `transferFluidCombustionToGas` içinde
"yanıcı değil" kapısı (`fluid_flammable` zaten bunu söylüyor). Tam söküm ~15
dosya + serileştirici + madde override doğrulaması; kayıtlı override'ı olan
projeleri reddetmemesi için derlemeyle birlikte yapılmalı.

Kabul: `scripts/test/rt_test_fire_steam_quench_ipc.py` (iki kol: kuru kontrol /
kaynayan tank). Kontrol listesi: [NEXT_BUILD_CHECKS.md](NEXT_BUILD_CHECKS.md).

**Canlı ilk koşu (2026-10-18, 12:32 build) — 6/7.**
- Kuru kol: yanıyor (340→852 hücre/adım), en düşük O2 = 0,2095 (tam hava).
- Islak kol: buhar 4–9 kg, en düşük O2 = **0,0** (saf buhar hücreleri), oksijen
  sınırlı hücre 854–1942/adım, **2423** hücre-adım "yalnız buharın Tv'si yüzünden
  sıcak" — FG1 öncesi bunlar yanıyordu. Defter dengede. Kaynama/sis/ısı/defter
  regresyonları PASS.
- Tek FAIL bir ölçü hatasıydı: `dry_not_oxygen_limited` — kuru havada her yanan
  hücre "sınırlı" sayıldı. GLSL bölmesi doğru yuvarlanmıyor, a/a = 0,99999994 <
  1. Yanma hızına etkisi ~1e-7; sayaç eşiği 0,9999'a alındı (shader + CPU),
  **yeniden derleme bekliyor**.
- ★ Dürüst sınır: ıslak kolda yanan HÜCRE sayısı kurudan fazla (oran 0,53).
  Kaynayan su gazı ısıtıyor, ateş daha çok hücreye yayılıyor; test hücre sayıyor,
  yakılan yakıt MİKTARINI saymıyor. "Ateş söndü" iddiası bu ölçüyle kanıtlanmış
  değil; yalnızca "buhar oksijeni düşürüp yanmayı kısıyor" kanıtlandı. Yakılan
  yakıt toplamı sayacı ve pilotu buhar sütununun içine alan bir sahne açık.

### Parti 2026-10-18b — gaz↔tane sürüklemesi + FG1/Ek 11 ölçü aletleri (DERLENDİ, canlı 2026-10-18)

**Canlı sonuç (2026-10-18, 2. build):**
- FG1 **8/8**: yakılan yakıt kuru 41,97 → ıslak 2,21 (×0,053); 303 hücre-adım "yalnız buharın Tv'si yüzünden sıcak"; kuru kolda oksijen sınırı 0.
- Yüzey suyu **6/6**: emilen 1,206 kg, buharlaşan 0,023 kg (sıcak üst yüz 636 K), tutulan 1,181 kg = defter 1,183 kg (fark 0,0018 < tolerans 0,014); bayat kare 0.
- Gaz↔tane **4/6**: durgun hava açmıyor, rüzgâr açıyor, çift tam iptal (tepki = −sürükleme). FAIL iki x kolu SAHNE: dürtü çoğunlukla +y (düşen taneye hava direnci, 0,147 N·s), x −0,007 — kapalı kutuda rüzgâr taneye ulaşmıyor (dönüş akışı). Test sahnesi açık yan duvarla ya da rüzgârı tanenin içinden geçirerek yeniden kurulmalı.
- Ek 11 hâlâ 4/6; **alet kökü gösterdi**: ıslak hücrelerin kabuğu 294,2 K (kabuk ortalaması ~320 K). Isınan yüzler suya değmiyor ya da host kopyası bayat — film/alan hesabı değil.


Kullanıcı kararı (2026-10-18): kod büyük adımlarla ilerler, testler partinin
sonunda topluca koşulur.

**T5a gaz↔tane (`Fluid/MatterGrainGasCoupling.cpp`).** Tane şeridi bugüne kadar
yalnız sıvıyı örnekliyordu; rüzgâr/duman/ateş yükselişi kumu taşımıyor, kum havayı
itmiyordu.
- Di Felice sürüklemesi (sıvıyla aynı `matterGrainDragCoefficient`, boşluk 1),
  hava yoğunluğu doğrudan Tv'den (ρ = P·Ma/(R·Tv), buhar defteri gerekmez),
  Sutherland viskozitesi.
- Kare üzerinde tam çözüm `Δv = (u − v)(1 − e^{−k·dt/m})`, DEM alt adımlarına
  **kuvvet** olarak (`external_acceleration`): hız dürtmesi değil, böylece
  duran yığını statik sürtünme tutabilir.
- Hücre başına topak sınırlayıcı: tanelerin toplam dürtüsü gazı karışım denge
  hızının ötesine itemez; aynı çarpan tanelere de uygulanır → çift tam zıt.
- Tepki −J, gaz MAC yüzlerine trilineer örneklemenin eşleniğiyle (momentum/ρV).
  Kare sırası: gaz geçen kare adım attı + host'a senkronlandı → taneler → gaz
  bu kare (host düzenlemesini yükler).
- ★ Kapı: yalnız gaz şeridi önceki karede adım attıysa VE hareket ediyorsa
  (`max_speed > 1e-4`). Durgun/boşta hava her tane sahnesine sürükleme ekler ve
  vakumda kalibre edilmiş bütün kabulleri (serbest düşüş, şev, CPU referansı)
  kaydırırdı. "Hava, domain'in onu simüle ettiği yerde vardır."
- Raporda (`grain_diagnostics.liquid`, panel): `gas_coupled_grains`,
  `gas_drag_impulse_n_s`, `gas_reaction_n_s`, `gas_momentum_residual_n_s`,
  `gas_limited_cells`, `gas_max_relative_speed_m_s`.
- Sınırlar: havada kaldırma (ρ_hava/ρ_tane ~ 5e-4), tanelerin gaz
  projeksiyonundaki hacmi, sürükleme ısısı yok. **Tane domain'i termal sıvı
  zincirini reddediyor**, yani tane + ateş + kaynayan su tek domain'de bugün
  kurulamaz (FG1 matrisinin ateş+Sand kolu bu sınırda).
- Kabul: `scripts/test/rt_test_grain_gas_drag_ipc.py` (durgun/rüzgâr kolu).

**FG1 yakıt miktarı.** `combustion_burned_fuel` (GPU 1e-4 birim sabit nokta,
CPU double). İlk koşu hücre saydığı için "ateş söndü"yü kanıtlayamamıştı; test
artık `steam_reduces_burned_fuel` (ıslak < 0,9 × kuru) ile miktarı kapılıyor ve
pilot buhar sütununun içinde (y 0,42, r 0,12).

**Ek 11 ölçü aleti.** `fluid.list_domains` → `thermal_contact_shell_kelvin`,
`thermal_contact_parcel_kelvin`, `thermal_contact_shell_cells`: temas eden
parsellerin GÖRDÜĞÜ kabuk/parsel sıcaklığı (temas ağırlıklı). Kabuk ≈ su ise
sorun ısınan yüzlerin suya değmemesi ya da host kopyasının bayat olması; kabuk
sıcak ama ısı küçükse film/alan hesabı.

**MSF yüzey suyu artık KÜTLE (aynı parti, kullanıcı isteği).** Eskiden
`sim_msf_wet` 0..1 nemi sıvı doluluğundan yoktan yükseltiyor, `sim_msf_ambient`
gizli ısı ödemeden siliyor, gather ıslak yüzeyi kaynamada kelepçeleyip gelen
ısıyı atıyordu. Şimdi:
- Durum `[3]` = **kg/m² su**; madde alanı `water_capacity_kg_m2` (metal/plastik
  film 0,02–0,05, meşe 1, kumaş 0,3, taş/et 0,5, kum 1,5, toprak 2).
  `absorbency` doygunluğa yaklaşma hızı olarak kaldı.
- **Islanma** (`Fluid/FluidSurfaceWater.cpp`, sıvı dalının sonunda): emici
  yüzeyin hücresine/komşusuna değen *Water* parselleri kütle verir; hücre en çok
  min(eksik, emicilik·eksik·dt) alır, parsel adım başına en çok yarısını verir ve
  canlı/ölü sınırında bırakılmaz. Defter: `absorption` Water → `surface`.
  Getirdiği duyulur ısı yüzeye `heat_input` olarak gider.
- **Kuruma** (`stepSurfaceWater`, kare başına, `stepAmbient`'ten önce, TAZE host
  kopyasında): 373 K üstündeki ısının tamamı buharlaşmaya (`C·(T−Tb)/L`), altında
  açık tava Hertz–Knudsen akısı (katsayı 4e-5, dünya nemi). Her kg L'yi yüzeyden
  öder. Buhar, konumunu içeren gaz domain'inin envanterine (vapour havuzu Water
  değilse yoğuşmayan havuza) gider; domain dışında `atmosphere`'e vent edilir.
  Defter: `vaporization` `surface` → domain/atmosphere, gizli ısı iki tarafta.
- Cihaz yalnız kuyruğu ekler (`water_input`, ambient binding 5); GPU kuruma,
  kaynama kelepçesi ve `sim_msf_wet` SÖKÜLDÜ. Piroliz doygunlukla (su/kapasite)
  kısılıyor.
- ★ Bulunan hata: `host_state_fresh` yalnız gather dispatch'inde sıfırlanıyordu;
  gaz domain'i olmayan sahnede host kopyası sonsuza dek "taze" sayılırdı ve bütçe
  aynı suyu her kare yeniden buharlaştırıp yoktan buhar yapardı. Ambient
  dispatch'i de sıfırlıyor.
- Readback: yalnız yüzeyde su varken ya da sıvı emici yüzeye değerken her kare
  istenir (kuru sahne 0,2 sn kısmasında kalır); bütçe son taze kopyadan beri geçen
  süreyle çalışır, kısma kuruma kaybettirmez.
- Yeniden adlandırma (kural 5): semantik `moisture` → `surface_water_kg_m2`
  (sim node kanalı, `msf.fields` semantics, panel listesi); önbellek **v16**
  (v11–v15'teki 0..1 nem okunur ve atılır, yüzey kuru oynatılır).
- `dry_rate` madde alanı SÖKÜLDÜ: yüklemede override'dan loglanıp düşülür
  (proje reddedilmez, geri yazılmaz), düzenlemede açıklamalı hata.
- IPC: `msf.water_budget` (yeni; dört katman + yetki + overlay),
  `msf.fields` → `surface_water_kg`, `max_water_saturation`.
- Sınırlar: tutulan suyun kendi ısı kapasitesi yüzey C'sine katılmıyor; yalnız
  Water emiliyor (yağ emmiyor); damlama/akma yok (doygunluk üstü su kalmaz çünkü
  alım eksikle sınırlı); kar/buz `absorbency` 0.
- Kabul: `scripts/test/rt_test_surface_water_ipc.py`.

### Parti 2026-10-19 — yüzey↔hava filmi + taneler termal zincirde (DERLENMEDİ)

**Ek 11 kökü (aleti 2026-10-18'de gösterdi).** Ölçülen: ıslak hücrelerin kabuğu
294,2 K, kabuk ortalaması 348 K, en düşük 299 K, en yüksek 396 K. Sahnede küp
orijinde, yarısı domain zemininin altında. `sim_msf_gather` her elemanı
`mix(T, T_gaz, 1−e^{−thermal_response·dt})` ile gaza çekiyordu (demir 3/s):
- Gövdeden bağımsızdı. 0,1 m demir blok ile 1 mm sac aynı hızla havayı izliyordu.
- Havaya enerji yazılmıyordu. Sıcak plaka havayı ısıtmıyor, soğuk hava plakayı
  bedavaya soğutuyordu.
Domain içindeki (ıslanan) yüzler oda havasına çivilendi. Alan ise yalnız domain
dışındaki yarıyı ısıttı.

**Yeni model (`sim_msf_gather`).** Eleman ile hücre havası arasında iki cisim:
- C_e = ρ·c·`thermal_depth_m`·alan. Sıvı temasıyla aynı gövde.
- C_g = ρ_hava(T)·c_hava·h³.
- G = (20 W/(m²K) + alev·ε·σ(T_g²+T_s²)(T_g+T_s))·alan.
- Eleman kesin iki cisim gevşemesiyle hareket eder. Hücre tam −C_e·ΔT_s alır:
  yeni işaretli `msf_accum_convect` (int, 2²⁸ sabit nokta). `sim_msf_resolve`
  onu katlar. Taban artık 1 K (eskiden normalize 0'dı: soğuk yüzeyin soğuttuğu
  hava geri ısıtılıyordu).
- Eleman yalnız hücre havasının PAYINI görür (alanı / 3h²). Aynı hücredeki
  çok sayıda eleman havayı aşırı itemez.
- Alev ışınımı: alev seviyesi 0..1 yayıcılık çarpanı. Kağıt alevde hâlâ tutuşur
  (ince gövde + ışınım), demir esintiyi neredeyse fark etmez.
- Emekli: `thermal_response` madde alanı (kural 5). Yüklemede override log'lanıp
  düşülür, düzenlemede gerekçeli hata verilir. Hiçbir script/test kullanmıyordu.

**Bilinen sınırlar.**
- Hava sıcaklığı alanın Tv'si olarak okunuyor. Buhar hücresinde film biraz
  sıcak çalışır (FG1'deki gibi buhar bağlanmadı).
- Ortam/alan gevşemesi (`sim_msf_ambient`, `cooling_rate`) hâlâ kapasitesiz bir
  hız. Domain içinde bu, ikinci bir hava soğutması demek: demir için ~×1000
  güçlü, ince ahşap için fiziğe yakın. Sonraki adım.
- Yüzey suyu testinde (meşe üstü) sayılar düşecek: eskiden gaz çivilemesiyle
  636 K'e çıkıyordu. Bu fiziksel düzeltmedir.

**Taneler termal zincirde (`Fluid/MatterGrainThermal.cpp`, HANDOFF_DEM_THERMAL 1–3, 5).**
- DEM sıcaklığa ihtiyaç duymaz. Taneler sıcaklıklarını host'ta, kanonik
  `particles.temperature`'da taşır. Seçme/birleştirme bütün dizileri kopyalıyor.
  Termal blok tane adımından önce, tanelerin de bulunduğu `state.particles`
  üzerinde koşuyor.
- Önceden de taneleri taşıyanlar:
  - hücreler arası iletim (`conductThermalLiquid`, taşıyıcının kendi c ve k'sı;
    Sand k 0,27 yatak değeri);
  - MSF kabuk teması (tane için film 200 W/(m²K), sıvı 500);
  - Ice (iskelet) erimesi gizli ısıyla, model geçişiyle sıvı şeridine.
- Hücre içinde yeni olanlar:
  - tane↔sıvı: 500 W/(m²K) film, ıslak payda, hücrenin sıvısına tek havuz
    olarak;
  - tane↔hava: 20 W/(m²K) + alev ışınımı, kuru payda, gaz şeridinin hücresine
    (ΔTv host'a yazılır; gaz adımı başta cihaz kopyalarını geçersiz sayıp
    yükler);
  - tane↔tane: G = k·4πr, hücre tane ortalamasına doğru, kapasiteyle tam
    korunumlu.
- Filmler tanenin kendi iç direnciyle seri: h_eff = 1/(1/h + r/(5k)). 1 cm
  çakıl suda τ ≈ 36 sn, 2 cm'de ≈ 125 sn. İç direnç olmasa ~4× hızlı olurdu.
- Tane alanı 3m/(ρr): süper tane kütlesinin yüzeyi.
- Hava kapısı sürüklemeyle aynı: yalnız gaz şeridi adım attıysa ("hava,
  domain'in onu simüle ettiği yerde vardır").
- Tane domain'inde sıvı DONMAZ, aşırı soğur. Tane sahipliği donmuş taşıyıcı
  kabul etmiyor, adım dururdu. `freeze_held_by_grains` raporlanıyor.
- IPC/Python/panel: `fluid.list_domains` → `thermal_grain_measured/count/wetted/
  min/mean/max_kelvin/heat_to_liquid_j/heat_to_gas_j/energy_residual_j/ms`,
  `thermal_freeze_held_by_grains`. Panelde Thermal Liquid bölümünde iki satır.
- Sınırlar:
  - Tane döngüsü seri (OpenMP yok); 1e5 tanede maliyeti ölç.
  - Gözenek suyu tane ısı kapasitesine katılmıyor (bütün geçişlerle tutarlı
    kalsın diye).
  - Sand/Gravel'ın tek iletkenliği (yatak değeri) iç direnç için de kullanılıyor:
    mineral değerinden yavaş, yani muhafazakâr.
  - Tane↔kabuk alanı hücre yüzü payı (sıvıyla aynı).
  - Hava hücresi hacmi yalnız tane hacmiyle azaltılıyor (sıvı hacmi değil).
- Kabul: `scripts/test/rt_test_grain_heat_ipc.py` (7 kapı).

**Canlı sonuç (1. build, 2026-10-19):**
- Gaz↔tane **6/6**: düşüş penceresinde x dürtüsü +0,043 N·s (y 0,195); toplam
  da artık +0,035. Çift tam zıt.
- Ek 11 **6/7, fizik tamamı PASS**: ıslak kabuk 392,6 K görülüyor (eskiden
  294,2), ilk örnekte su→katı −1486 J/adım (eskiden −17,8), suyun en sıcağı
  295,4 K. Dolumdan önce kabuk düzgün: en düşük 396,67, en yüksek 396,67 K
  (eskiden 299…396).
  - Tek FAIL bir ölçüydü. `msf.fields` host kopyasını okuyor; boş domain
    readback istemiyor, yani kopya alanın yaratıldığı ana ait olabiliyor ve
    `temperature_valid` yine true. Okunan 293,7 K idi, kabuk 390 K'deydi.
    Ek adımla okununca 396,7 K. Düzeltme `temperature_fresh` (derlenmedi).
- ★ **Yan bulgu, hatasız yanlış:** `scene.add_primitive` silinmiş bir adla
  (sıkıştırma bekleyen silme) başarı döndürüyor ama nesneyi görünmez/silinmiş
  yaratıyordu. Her ekle-sil testinin ikinci koşusu bundan bozuldu: MSF alanı
  yok, 6 kapı FAIL. Ad seçimi artık bekleyen silmeleri dolu sayıyor
  (`RtApi.cpp addPrimitive`, derlenmedi). UI'nin kendi ekleme yolu
  denetlenmedi.
- Yüzey suyu **6/6**: meşe üstü 373,6 K'de kaynamada tutunuyor (eskiden gaz
  çivilemesiyle 636 K), buharlaşan 0,0032 kg (eskiden 0,023). Beklenen
  fiziksel düşüş.
- FG1 **8/8**. Madde testi PASS (`thermal_response` reddediliyor).
- Tane ısısı **7/7, ama oran YANLIŞ** (kontrol listesindeki sinsi durum):
  360 → 299 K 1,5 sn'de, sonu 293,0 K, yani suyun altında. Suya 4,66 kJ,
  havaya 68 J gitti; ~157 kJ'lük fazlanın %97'sini `coolThermalLiquid`'in
  kapalı duvar "temas" soğutması yuttu (sonsuz ortam, kapasitesiz hız; tabana
  oturan her tane bir duvara değiyor). Taneler o yoldan çıkarıldı; duvar artık
  tane geçişinde 200 W/(m²K) film, yüzün altıda biri,
  `thermal_grain_heat_to_walls_j` (derlenmedi). Teste `cools_at_physical_rate`
  ve `water_takes_most` kapıları eklendi.
  ★ Aynı kapasitesiz duvar yolu SIVI için hâlâ duruyor (kapalı duvar = ortam
  termostatı). Bu daha önceki bir model kararı; tane bulgusu onun ölçeğini
  gösterdi. Sıvıda da film modeline geçmek sonraki iş.
- Tane geçişinin maliyeti 400 tanede 0,26–0,46 ms. Enerji artığı ≤ 5e-10 J.
- H1 paketi **15/21**.
  - PASS: coexist, porous, wet, motion, dense, repose, base, history,
    convergence, extended, settle, panel.
  - `readiness` FAIL gerçek bir hataydı (bu partiden önce): taneleri yalnız
    İSTEYEN (henüz adım atmamış) bir domain CPU'ya ya da açık sınıra
    alınabiliyordu. Koruma artık doğrudan `matterGrainBlockers` (derlenmedi).
  - İlgisiz/bayat: üç statik sözleşme (REVISION 24→27, editör metni),
    `g2_dry_wet` (Sand artık DEM), `sleep_transfer` (mermi doğmuyor; ayrı bakılmalı).

**Gaz↔tane sahnesi.** Kod değişmedi. Rüzgâr kaynağı düşüş yolunun üstünde
(0, 1,1, 0; r 0,35) ve x kapısı düşüş penceresinde (ilk 45 adım) ölçülüyor.
Kapalı kutuda dönüş akışı içindeki yığın artık kapıyı sulandırmıyor; toplamlar
yine raporlanıyor.

### Parti 2026-10-19 P1–P4 (CANLI — duman 10/10 PASS)

**P1, kapasitesiz ısı yutakları söküldü.**
- Sıvı (`coolThermalLiquid`): kapalı duvar ve çıplak collider yüzü
  500 W/(m²K), gaz ızgarası olmayan domainde açık hava yüzü 20 W/(m²K) +
  doğrusallaştırılmış ışınım. Hepsi parselin kendi ısı kapasitesine karşı ve
  hücre yüzünün payı üzerinden. Duvarın aldığı `heat_to_ambient_j` olarak
  raporlanıyor (dört katman, panel).
- MSF (`sim_msf_ambient`): gaz domain'i dışında oda, aynı hava filmi
  (dünya `convection_coefficient` çarpanı kaldı) + ışınım, gövde
  ρ·c·`thermal_depth_m`. Gaz domain'i içinde hava yalnız gather'ın; her gaz
  domain'i artık `gas_air` bayraklı bir bölge.
- Termal alan açık bir DIŞ KAYNAK olarak kaldı. Kaynama, FG1, sis, kar ve
  füzyon testleri onunla ısıtıyor; hızını filme bağlamak hepsini fiziksel
  ama dakikalar süren sahnelere çevirirdi. Hızı artık alanın
  `thermal_rate_per_s`'i (varsayılan 2/s = parsellerin eski temas hızı;
  IPC/Python/panel/serileştirici/hash). MSF yüzeyleri de aynı hızla ısınır;
  eskiden maddenin `cooling_rate`'iyle ısınıyorlardı (demir 0,05/s).
- Emekli (kural 5): `cooling_rate`, `solver_air/contact_cooling_rate`
  (madde: yüklemede log'lanıp düşülür, düzenlemede gerekçeli hata) ve
  `thermal_air/contact_cooling_rate` (domain: IPC ve Python gerekçeyle
  reddeder; serileştirici ve proje artık yazmıyor, okurken yok sayıyor).
- ★ Ölçek uyarısı: 5 cm'lik vokselde duvarda τ ≈ ρch/500 ≈ 420 sn (su).
  Mum/wax sahneleri artık dakikalarca donar; eskiden 0,5 sn'de donuyordu.
  Bilinçli. Bu, vokselin fiziği.

**P2, filmler Tv değil T okur.**
- `sim_msf_gather` buhar envanterini 7. bağlama olarak alıyor (yoksa kuru).
  T = airFromVirtual(Tv, m); hava kapasitesi ρ = P·Ma/(R·Tv) (kesin); gaz
  alanına ΔT·Tv/T yazılıyor.
- Tane↔hava aynı çevrimi host'taki buharla yapıyor (`gasCombustionAir`).
  Envanter cihazdaysa host işaretçisi boş ve kuru varsayılır (sınır).

**P3, tane domain'inde donma.**
- Donmuş sıvı artık adımı reddettirmiyor. `partitionMatterGrainOwners` onu
  "tutulan" kümeye koyuyor; hiçbir şerit taşımıyor, birleştirmede geri
  geliyor.
- Sıvı buzu katı faz kaplaması olarak görüyor (o yol tane engelinden
  bağımsız). `freeze_held_by_grains` sökuldü.
- Sınır: DEM taneleri yalnız üçgenlerle çarpışıyor, buzun içinden geçer.

**P4, ölçek.**
- Tane ısı geçişi iki fazda paralel: sıvı hücresi grupları (sıvı havuzu,
  duvar, tane-tane) ve hava hücresi grupları. Her grup kararlı sırada,
  sonuç iş parçacığı sayısından bağımsız.
- Gözenek suyu taşıyıcının ısı kapasitesinde (`parcelHeatCapacity`,
  4186 J/(kg K)): bütün geçişlerde tek tanım.
- Sınır: emilen su taşıyıcıya karışma ısısı ödemeden katılıyor.

**Test:** `scripts/test/rt_test_thermal_films_ipc.py` (5 kapı) yeni.
`rt_test_grain_heat_ipc.py` 8 kapı. `rt_test_fluid_phase3_boundary_freeze_ipc.py`
emekli hızları göndermiyor. Hepsi `rt_matter_smoke.py`'de.

## Öncelikli sonraki düzeltmeler
1. ✔ (kodda yapılmış, 2026-10-18'de doğrulandı) FluidPhysicalMass.cpp densityForParticle yalnız liquid_density kullanıyor.
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

### 2026-10-10 sıvı/gaz yüzeyi ve whitewater kaynak güncellemesi

Altı açık yüz üzerinden ortak CPU/GPU ısı-buharlaşma planı, yerel kaynama
çıkışı, faz momentumunun gaz MAC hızına aktarılması ve enerji bütçesi düzeltmesi
kaynakta hazır. WW kapalı hesap kapısı, güncel criterion/komşu girdileri, dar bin
penceresi ve taşmada eksiksiz CPU fallback aynı partidedir. Script/IPC maliyet
ve doğruluk sayaçları eklendi. Dokuz kaynak/matematik kontrolü PASS; build
yapılmadı ve canlı fizik/performans kabulü açık. Güncel kapsam ve tek kullanıcı
derlemesi checklist'i:
[GAS_WHITEWATER_PHYSICS_COST_2026_10_10.md](GAS_WHITEWATER_PHYSICS_COST_2026_10_10.md).
DEM Revizyon 23/24 uyku kabulü bu kaynak kontrolüyle kapanmış sayılmaz.


### 2026-10-10 SDF yüzey çalışma maliyeti

Fiziksel ölçekte kalite çarpanından bağımsız CSR, aktif surface tile planı ve
halo yumuşatma kaynakta hazır. Anisotropik kernel merkez kaymasının eksik
sorgu desteği düzeltildi; ortak native/script/IPC aşama sayaçları eklendi.
Final dense SDF/NanoVDB consumer storage korunuyor. Kaynak/matematik kontrolleri
PASS; C++ derlemesi, canlı hız/kalite/rewind kabulü açık. Kapsam ve checklist:
[SDF_SPARSE_WORK_OPTIMIZATION_2026_10_10.md](SDF_SPARSE_WORK_OPTIMIZATION_2026_10_10.md).


### 2026-10-10 UVW maliyeti ve yüzey merkezi filtresi

UVW gather/extrapolation/smoothing aktif tile planına taşındı; output CSR
filtrelenmiş render merkezlerine kurulur. Position smoothing spherical yolda
da uygulanır; düşük komşulukta isolated droplet korunur, fizik SoA değişmez.
Mevcut UI/script/IPC parametreleri aynı core servise bağlıdır. Kaynak/math
kontrolleri PASS, C++ build ve canlı hız/kalite/UVW/rewind kabulü açık.
[UVW_SURFACE_FILTER_OPTIMIZATION_2026_10_10.md](UVW_SURFACE_FILTER_OPTIMIZATION_2026_10_10.md).
