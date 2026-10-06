# H1 tane yol haritası — tek Matter domain'inde kuru/ıslak granül çözücü

> **Durum:** AKTİF — 2026-10-06 bulut oturumu. B2, B3, B4 (ilk adım), B5, B6, B7, B8 ve B9a kaynakta, **hepsi tek build + tek uzun test geçişi** bekliyor (NEXT_BUILD_CHECKS en üst). Kalan kod: B9'un geri kalanı ve koşullu B10 — canlı sonuçlardan sonra.

Bu belge H1'in (granül çözücü) **parti sırasıdır**: her partinin ne yazdığı, neyi
ölçerek kapandığı ve kapanmazsa neyi ifade ettiği. Tasarım gerekçesi ve kapı tablosu
[BIRLESIK_MADDE_DOMAIN_TASARIMI.md](BIRLESIK_MADDE_DOMAIN_TASARIMI.md) "H1 üretim hedefi"
bölümündedir; çözücünün kendisinin teknik kaydı ve canlı sonuç tabloları
[MATTER_GRAIN_GPU_RUNTIME.md](MATTER_GRAIN_GPU_RUNTIME.md)'dedir. Burası ikisinin arasındaki
**sıra ve kabul** belgesi.

Çalışma kipi (kullanıcı kararı, 2026-10-06): **uzun partiler, az test döngüsü.** Bulutta kod +
statik sözleşme + C++ sözleşme testleri yazılır ve buradaki Linux araçlarıyla derlenebilenler
koşturulur; uygulama derlemesi ve canlı IPC testleri kullanıcının makinesinde tek geçişte yapılır.

---

## 0. Nerede duruyoruz

| Konu | Durum | Kanıt |
|---|---|---|
| Fused alt adım (1 dispatch/alt adım), 24-temas CFL, Cundall–Strack + EPSD2, tane kütlesi | CANLI PASS | `matter_h1_grain_fused_*_2026-10-06.json` |
| Yoğun maliyet 1024/4096/16384 | 25.5/24.4/41.1 ms (eski 173.6/213.8/272.2) | `matter_h1_dense_grain_fused_2026-10-06.json` |
| Diğer Vulkan çözücüleri (descriptor yeniden kullanımı) | G2 PASS | `matter_g2_dry_wet_after_descriptor_reuse*` |
| **B2** kova komşuluğu (rev 6) + `grain_diagnostics.pile` + `--repose-only` | Kaynak, build yok | NEXT_BUILD_CHECKS |
| **B3** parçacık başına sahiplik + su–tane bağlama + `--coexist-only` | Kaynak, build yok; bağlama matematiği bulutta C++ ile koşturuldu (PASS) | `scripts/test/matter_grain_coupling_test.cpp` |
| **B5** hacim dışlama (porous projeksiyon) + basınç kuvveti + `--porous-only` | Kaynak; gözenek ağırlığı ve hidrostatik basınç = Arşimet C++ ile PASS | aynı test |
| **B6** ıslak tane (emilim/kuruma, Willett köprüsü, doğum ıslaklığı) + `--wet-only` | Kaynak; su bilançosu 7e-9 kg, momentum birebir (C++ PASS) | aynı test |
| **B7** XPBD adayı + `--xpbd-compare` | Kaynak; shader doğrulandı | — |
| **B8** hücre sırası + kimlikle taşınan temas geçmişi (rev 10) | Kaynak; shader doğrulandı | — |
| **B4** yerleşik durum yeniden kullanımı + transfer sayaçları | Kaynak (ilk adım) | — |
| **B9a** geçmiş geçersiz kılma + CPU referansı EPSD2 | Kaynak; eğim parity testi C++ PASS (μr=1 kayma 0, μr=0 3.588 m = analitik) | `matter_grain_reference_slope_test.cpp` |
| Emitter tohumu oturumdan bağımsız | Ayrı yerel görev (kullanıcı başlattı) | — |
| H1-R CPU referansı | EPSD2'ye güncellendi (B9a); küçük sahne GPU parity canlıda koşulmadı | B9 |

Kullanıcı kararları (bağlayıcı):
- Su + tane aynı domain'de "adım tutuldu" **uyarısı yazılmaz**; çözüm gerçek birlikte yaşam (B3).
- OptiX'e yeni özellik yok (CLAUDE.md §6); tane render'ı Vulkan RT/raster.

---

## 1. Bağlayıcı ilkeler (her partide geçerli)

1. **Taşıyıcı başına tek taşıma sahibi.** Bir alt adımda bir parçacığı yalnız bir çözücü
   ilerletir. Sahiplik render materyalinden, görsel ayardan, sanal çocuk sayısından türemez;
   bugün `constitutive_model`'dan türer (Granular → grain, Fluid → sıvı şeridi).
2. **Sessiz CPU geri dönüşü yok.** Başaramayan adım tutulur ve nedenini
   `fluid.matter_models → mixed_execution.status` söyler. Yarım yayın yok (transaksiyonel).
3. **Her impuls bir kez, eşit ve zıt.** Bağlama ölçümü `momentum_residual_n_s` ile
   raporlanır; "makul görünüyor" kabul değildir.
4. **Her yetenek IPC/Python'dan sürülebilir** (CLAUDE.md §1): RtApi → IPC → Python →
   capability → descriptor overlay + `gen_ipc_descriptors.py`. Panel yalnız aynı servisi çağırır.
5. **Ölçü aletini önce sına.** Her yeni kapının bir A/B kolu vardır ve A/B'nin ayırt edebildiği
   kapı, makullük kapısından önce okunur (repose'ta μr duyarlılığı, bağlamada açık/kapalı).

---

## 2. Partiler

### B2 — Kova komşuluğu + yığılma açısı (kaynakta, build bekliyor)

- **Kod:** bağlı liste hash yerine üç dönen sabit kapasiteli kova tablosu (kova başına 16);
  `grain_diagnostics.pile` halka profili; `rt_h1_grain_runtime_ipc.py --repose-only`.
- **Kapanış:** step kernel µs/çağrı 243/289'dan düşer; regresyon (temel/statik/yakınsama/
  extended/settle) değişmez; repose: μr .05 vs .3 farkı ≥ 3°, μr .1 açısı 15–45°,
  r .025 vs .0175 farkı ≤ 4°.
- **Kapanmazsa:** µs düşmediyse darboğaz hash zinciri değil (history taraması/BVH/register)
  → B8'in ilk işi hücre sıralı düzen olur. Repose duyarsızsa ölçüm yanlış yeri ölçüyor.

### B3 — Parçacık başına sahiplik + su–tane bağlama (H1-G2a; kaynakta, build bekliyor)

Kullanıcının bildirdiği arıza: aynı domain'de su + dry grain varken parçacıklar emitter'da
doğup kalıyordu. Neden: `grain.enabled` domain çapındaydı; tane adımı akışkan taşıyıcı
görünce tüm adımı reddediyordu (`mixed_step_held`). Yüksek çözünürlükte tek tanenin de
kalması büyük olasılıkla aynı kökten: tane adımı domain'in grid GPU tamponlarına
(`fluid_positions`, `fluid_uploaded_particle_count`) bağımlıydı; grid tamponu kurulamazsa
tane adımı da reddediliyordu. B3'te tane kendi tamponlarına sahip — grid'e bağımlılık yok.

- **Sahiplik:** `partitionMatterGrainOwners`: Granular → grain (kimliğe göre sıralı, temas
  geçmişi yuvaları başka parçacık silinince kaymaz), Fluid → sıvı şeridi. Frozen/elastic/ıslak
  tane ve legacy-granular Auto reddedilir (sahibi olmayan taşıyıcı).
- **Kare sırası (staggered):** (1) sıvı alt kümesi: kuvvet + karma Vulkan sıvı şeridi
  (P2G/MGPCG/G2P), (2) sıvının son hali + tane hacimleri domain grid'ine bin'lenir,
  (3) tane DEM alt adımları: her tane kendi özel "sıvı topağı"na karşı **örtük** Di Felice
  sürüklenmesi + hidrostatik Arşimet kaldırması, (4) zıt impuls aynı hücrelerin sıvı
  parsellerine. Kanonik dizi yalnız her aşama başarılıysa `[sıvı…, taneler…]` olarak yazılır.
- **Topak (lump):** hücre sıvı kütlesi taneler arasında hacim ağırlığıyla bölünür; bir
  hücrenin topakları hücrenin sıvısını aşamaz → örtük çift taşamaz (overshoot yok), sertlik
  sınırı yok, momentum kesin (shader'da `m v + M u` değişmez).
- **Render:** tane domain'inde yalnız granüler taşıyıcı fizik yarıçapıyla, sıvı parsel kendi
  parsel yarıçapıyla çizilir.
- **Ayarlar:** `fluid_coupling` (vars. açık), `drag_viscosity_pa_s` (vars. 1e-3); panel +
  `fluid.set_grain_settings` + Python + save/load.
- **Tanı:** `grain_diagnostics.liquid`: grains, liquid_parcels, coupled_grains,
  drag/buoyancy/liquid_reaction impulsları, `momentum_residual_n_s`, `unmatched_impulse_n_s`,
  `max_submerged_fraction`.
- **Kapanış (`--coexist-only`):** suya gömülü doğan tanenin ilk ivmesi bağlama açıkken
  g' = g(1 − ρ_w V/m) = 6.13 m/s² ±%15, kapalıyken 9.81 ±%5 (A/B); 256 tanelik yığına su
  dökülürken hiçbir adım tutulmaz, su düşer (momentum_y < 0), su ve tane kütlesi sabit,
  momentum artığı ≤ 1e-3 × alışveriş, eşleşmemiş impuls 0.
- **Bilerek dışarıda:** sıvının taneyi **hacim** olarak görmesi (projeksiyonda gözeneklilik)
  — yığın sıvıya göre gözenekli ortamdır; −V∇p dinamik basınç kuvveti; dönme sürüklenmesi;
  eklenik kütle; ıslanma. Hepsi B5/B6.

### B4 — GPU yerleşikliği (H1-C7): kare sonu tam-durum yayınını kaldır

> **Yazılan (ilk adım):** host durumu son yayınla bit-bit aynıysa bank 0 yeniden kullanılır
> (yalnız kare satırları yüklenir); `runtime.state_resident/upload_bytes/download_bytes/
> transfer_batches` sayaçları. Render'ın cihaz tamponundan beslenmesi ve sıvı bin'lemesinin
> cihaza taşınması **ölçümden sonra**: sayaçlar transferin kare süresinde önemli olduğunu
> gösterirse. Aşağıdaki plan o adımın tasarımıdır.

Bugün her kare: tane durumu host'a iner, sıvı alt kümesi ve bağlama alanı host'ta kurulur,
render köprüsü host dizisinden instance yazar. B3'ün bağlaması da host'ta (doğru ama O(N)
CPU + 2 yükleme/indirme). B4 bunu GPU'ya taşır.

- **Kod:** (a) tane durumu kareler arasında cihazda kalır; host kopyası yalnız sorgu/cache/
  kayıt anında, asenkron okuma ile tazelenir ("host stale" bayrağı, `FluidGpuParticleUpload`
  sözleşmesiyle aynı model); (b) sıvı bin'leme cihazda: karma şeridin P2G kütle/momentum
  alanı zaten var — tane kernel'i bunu doğrudan örnekler; tepki ayrı bir yüz-impuls ızgarasına
  sabit-nokta atomik ile yazılır ve sonraki sıvı P2G'sinde gövde kuvveti olarak tüketilir;
  (c) render: Vulkan raster/RT instance tamponu tane konum tamponundan bir compute geçişiyle
  doldurulur (OptiX yolu donmuş: host köprüsünü korur, yalnız istendiğinde okur);
  (d) sayaçlar: kare başına upload/download bayt, CPU senkron sayısı, dispatch.
- **Kapanış:** sorgu yokken kare başına tam-durum indirme 0 bayt; B3 kapıları aynen geçer;
  `--coexist-only` momentum artığı aynı düzeyde; 16384 yoğun sahne ≤ 41 ms (gerilemez).
- **Sinsi risk:** host kopyası bayatken okuyan bir tüketici (cache, emitter doğum filtresi,
  render köprüsü) eski konumu okur ve hata vermez. Her host okuyucusu listelenir ve
  "stale ise önce indir" kapısından geçer; liste B4 notunda tutulur.

### B5 — Hacim dışlama ve basınç kuvveti (unresolved CFD-DEM tam hali)

> **Yazılan:** tanelerin hacim kesri sıvı şeridinin variational yüz ağırlıklarına (ε, en az
> .3) bir adım için yazılır, katı hızı olarak tanelerin ortalama hızı; yeni
> `sim_fluid_divergence_porous` ∇·(ε u_s + (1−ε) u_t) = 0'ı uygular. Taneler sıvının
> basınç kuvvetini alır, sıvı geri alır (model A). Eklenik kütle yazılmadı.
> ★ **2026-10-06 canlı düzeltme:** kuvvet önce çözücünün basınç tamponundan (−V ρ ∇p)
> okunuyordu; canlıda Arşimet'in ≈ %10'u, işareti kare kare dönen gürültü çıktı. Sıvı
> şeridi yerçekimini kare başında bir kez uygulayıp alt adımlarda projekte ettiği için tek
> bir basınç alanı karenin yükünü taşımıyor (alt adım ortalaması da yetmedi: 9.43 → 9.08).
> Artık kuvvet sıvının **ölçülen** ivmesinden: F = batma · ρV (Du/Dt − g), Du/Dt tanenin 8
> hücresindeki sıvı parsellerinin kare boyu kütle ağırlıklı hız değişimi (kimlikle,
> Lagrange). Durgun suda tam Arşimet; GPU basınç okuması yok.
> Kabul: `--porous-only` (yer değiştirme A/B); Richardson–Zaki/Ergun kıyasları sonraki tur.

- **Kod:** sıvı süreklilik denklemi gözeneklilikle (ε = 1 − φ_s): P2G'de katı hacim kesri,
  projeksiyonda ε-ağırlıklı diverjans; taneye −V∇p (hidrostatik kaldırmanın yerine, dinamik
  basıncı da taşır); ölçeklenmiş eklenik kütle (C_m = 0.5) — gerekirse.
- **Kapanış (fiziksel kıyas, analitik):** (1) tek küre son hızı Di Felice/Schiller–Naumann
  ±%10 iki çapta; (2) çökelme kolonu: engellenmiş çökelme hızı Richardson–Zaki
  u = u_t ε^n (n Re'ye göre 2.4–4.65) ±%20; (3) akışkan yatak: minimum akışkanlaşma hızı
  Ergun ±%25; (4) yığına dökülen su yığının **içinden** ölçülebilir bir hızla süzülür
  (Darcy geçirgenliği ölçülür: ne sıfır ne sonsuz).
- **Sinsi risk:** ε'nin P2G ile tane bin'lemesinin farklı çekirdekle hesaplanması — sıvı,
  kendi görmediği bir katı yüzünden kütle kaybediyor gibi görünür. Tek çekirdek, tek alan.

### B6 — Islak tane (H1-G2 wet)

> **Yazılan:** kanonik `pore_water_mass_kg` sidecar'ı, hücre payı ile emilim (kare başına
> hücre sıvısının en fazla yarısı, momentum kesin), kuruma (domain dışına, raporlanır),
> `birth_saturation`, Willett (2000) sarkaç köprüsü (kopma V^(1/3), r/2 ile sınırlı),
> `represented_grain_radius_m` ile Bond sayısı koruma. Islak sürtünme değişmedi; köprüler
> yalnız tane–tane. Kabul: `--wet-only`.

- **Kod:** tane başına gözenek/köprü suyu (mevcut pore ledger anlamıyla: kütle iki yerde
  yaşamaz); sıvı köprüsü kılcal kohezyonu (Willett/Lian köprü kuvveti, kopma mesafesi
  hacimden), ıslak sürtünme; kuruma ledger'a bağlı; ıslak görünüm mevcut 8-bant materyalden.
  Sıvı parselinden taneye emilim/drenaj B3'ün bin'leme alanını kullanır.
- **Kapanış:** kuru limitte (su 0) B2/B3 sonuçları birebir; ıslak kum yığını kuru açıdan
  dik durur (kum kalesi), doygun kum yine akar (kılcal köprüler kapanır); su kütlesi ledger'ı
  kapalı; kuruma süresi çözünürlükten bağımsız ±%20.
- **Sinsi risk:** kohezyon "yapışkanlık" olarak çalışıp yer çekimini yener — taneler duvara
  yapışır ve bu "ıslak kum" gibi görünür. Kopma mesafesi ve kuvvet üst sınırı ölçülür.

### B7 — H1-G0 kararı: DEM mi PBD/XPBD mi (adil maliyetle)

> **Yazılan:** `solver_kind = xpbd` aynı runtime'da (aynı hash + .2 r tahmin payı, collider,
> sahiplik, bağlama, köprü): küçük adımlı tek Jacobi projeksiyonu, uyumluluk 1/k, esnek
> olmayan normal, kol üzerinden konumsal Coulomb, yuvarlanma sınırı. `--xpbd-compare`
> doğruluk kapıları + karşılaştırma tablosu. **Karar canlı tablodan sonra verilecek.**

- **Kod:** aynı runtime içinde (aynı kova hash, aynı collider BVH, aynı sahiplik) XPBD temas
  adayı: pozisyon kısıtı + sürtünme kısıtı + açısal güncelleme (spin için ayrıca tasarlanır;
  PBD nokta teması spin üretmez). Ayar `grain_solver_kind = dem | xpbd`.
- **Karşılaştırma:** repose açısı, runout mesafesi, dt/iterasyon duyarlılığı, enerji
  sönümü, 16k/64k/100k maliyet — aynı sahne, aynı tane sayısı, aynı GPU.
- **Karar kaydı:** kazanan varsayılan olur, kaybeden **sökülür** (CLAUDE.md §5).
- **Sinsi risk:** XPBD'nin iterasyonla "sertleşmesi" — CLAUDE.md teşhis dersi: yakınsadıkça
  sertleşen belirti fizik değildir; iterasyon taraması zorunlu.

### B8 — Ölçek

> **Yazılan:** her kare Morton hücre sırası; runtime yayımladığı kimlik/konumları tutar, sıra
> değişince geçmiş iki GPU dispatch'iyle yeni indekslere taşınır (doğum/silme/sıralama
> statik sürtünmeyi kaybettirmez). **Bilerek yazılmadı:** uyuyan taneler — GPU'da aktif liste
> sıkıştırması olmadan kazanç yok ve "hareket etmesi gereken donmuş yığın" kimsenin
> raporlamadığı hatadır; önce hücre sırasının ölçümü.

- Hücre sıralı parçacık düzeni (alt adım başına değil, N alt adımda bir counting sort),
  uyuyan taneler / adalar, 100k+ tane, temas bütçesinin (24) ölçüme göre gözden geçirilmesi.
- **Kapanış:** 65536 tane yoğun sahne ≤ 60 ms; 100000 sınırı ölçüme göre kaldırılır ya da
  gerekçesiyle tutulur.

### B9 — Üretim (H1-P)

> **Yazılan (B9a):** cihaz geçmişi yalnız yayımlandığı durum için geçerli — reset/scrub/
> restore/düzenleme sonrası bırakılır (B8 ile kimlik eşlemesine dönüştü); CPU referansı
> EPSD2. **Kalan:** geçmişin cache/kayıtta saklanması (bugün resume'da sıfırlanır — ölçülecek
> fark), tane preset'leri, madde başına tane materyali.

- Temas geçmişi cache/scrub/resume'da saklanır (bugün cihazda, resume'da sıfırlanır →
  statik sürtünme ilk karede kaybolur: **ölçülmesi gereken bir sinsi fark**).
- H1-R CPU referansı Cundall–Strack + EPSD2'ye güncellenir; küçük sahnede GPU parity testi.
- Tane substance preset'leri (kum/çakıl/kar/toprak) — `MatterMaterialPreset` üzerinden;
  bağlamsal panel; tane render materyali madde başına.
- **Kapanış:** kaydet/yükle/bake/scrub/resume aynı kimlik ve geçmişle aynı sonucu verir
  (iki koşu farkı float gürültüsü düzeyinde); emitter tohumu oturumdan bağımsız (ayrı görev)
  olduktan sonra build'ler arası birebir A/B mümkün.

### B10 — H1-H (yalnız gerekli olduğu kanıtlanırsa)

- MPM↔grain hibrit geçiş, yerel bağ kopması (kar/çığ, heyelan), yanma/erime ledger kancaları.
  Ön şart: B7 kararı ve B5/B6'nın kapanması.

---

## 3. Test matrisi (hangi kol hangi partiyi kapatır)

| Komut | Parti | Ne ölçer |
|---|---|---|
| `python scripts/test/check_matter_grain_contracts.py` | hepsi | ABI/sıra/bağlantı (build gerekmez) |
| `matter_grain_coupling_test.cpp` (C++) | B3 | Di Felice limitleri, kaldırma, örtük çift momentumu, tepki dağıtımı, sahiplik bölme/birleştirme |
| `rt_h1_grain_runtime_ipc.py` (tam) | regresyon | serbest düşüş, zemin/spin, pile dt |
| `--static-only`, `--convergence-only`, `--extended-only`, `--settle-only` | regresyon | statik tutunma, yakınsama, rampa/köşe/yoğunluk, yerleşme |
| `--repose-only` | B2 | yığılma açısı, μr duyarlılığı, tane boyu yakınsaması |
| `--coexist-only` | B3 | kaldırma A/B, su dökme, korunum |
| `rt_h1_grain_kernel_profile_ipc.py`, `rt_h1_dense_grain_scene_ipc.py` | B2/B4/B8 | maliyet |
| `rt_g2_dry_wet_compare_ipc.py --expect-empty-fluid-skipped` | regresyon | karma MPM/su yolu |

---

## 4. Açık riskler ve sinsi başarısızlıklar

- **B3 topak örneklemesi kare başına donuk:** sıvı hızı kare başında örneklenir; kare içinde
  tane çok yol alırsa (hızlı düşüş) sürüklenme bir kare gecikmeli. Belirti: yüksek hızda
  dalışta dt'ye duyarlı yavaşlama. Ölçüm: `--coexist-only`'yi 60 ve 120 Hz koşturmak (B4'te
  alan cihaza taşınınca alt adım başına örnekleme mümkün).
- **Sıvı şeridi tane kümesini hiç görmüyor** (B5'e kadar): su bir yığının içinden geçer ve
  yalnız yavaşlar. Görsel olarak "su kumun içine girdi" — bu B3'te beklenen, hata değil.
- **Parçacık sırası her karede [sıvı…, tane…]**: kimliğe bağlı her şey (cache, kimlik hash'i)
  etkilenmez; indekse bağlı bir yan önbellek varsa ilk karede bir kez yanlış eşleşir.
  Bilinen tüketici yok; görülürse kökü burası.
- **Tane domain'inde sıvı için kuvvet alanı / hareketli collider** hâlâ adımı tutar
  (tane tarafı desteklemiyor). Ayrıştırma B4 ile.

## 5. Karar kayıtları

- 2026-10-06 (2. bulut turu): kullanıcı "kodlamayı bulutta bitirelim" dedi; B4–B9a yazıldı.
  Bulutta glslangValidator + spirv-val ile shader, taklit başlıklarla g++ sözdizimi ve
  bağımsız C++ testleri koşturuldu. B10 (hibrit) koşullu kaldı: B7 kararı ve canlı B5/B6
  sonuçları olmadan yazılması kanıtsız bir ikinci yol olurdu (CLAUDE.md §5).

- 2026-10-06: Su+tane için uyarı değil gerçek birlikte yaşam (kullanıcı).
- 2026-10-06: İlk bağlama host'ta ve kare başına (B3); doğruluk ve korunum önce, GPU'ya
  taşıma B4'te. Gerekçe: canlı test döngüsü pahalı; host matematiği burada C++ ile
  koşturulabiliyor, shader tarafı yalnız örtük çift güncellemesi.
- 2026-10-06: Sürükleme yasası Di Felice (gözeneklilik üstelliği içerir, Re→0'da Stokes'a
  yakın sonlu limit) — tek tane ve yoğun yığın aynı formülle.
