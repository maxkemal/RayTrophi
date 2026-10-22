# Birleşik madde — kalan yol haritası (2026-10-19)

> **Durum:** AKTİF — kalan iş üç derlemede (D1–D3, aşağıda).  Kod partiler halinde ilerler. Test yalnız birkaç partiden sonra, tek bir toplu duman koşusuyla yapılır (`scripts/test/rt_matter_smoke.py`). Ayrıntılı tarih [BIRLESIK_MADDE_DOMAIN_TASARIMI.md](BIRLESIK_MADDE_DOMAIN_TASARIMI.md) dosyasında; bu dosya yalnız sırayı ve kuralı tutar.

## Çalışma kuralı (2026-10-19 kararı)

- **Kod önce, test en sonda.** P1–P4 arka arkaya yazılır. Her parti kendi
  devir satırını bırakır. Kullanıcı tek derleme alır, ardından tek bir duman
  koşusu yapılır.
- **Duman koşusu özet basar.** Test başına tek satır (`7/7`, ya da FAIL olan
  kapıların adları). Uzun çıktı diske, `%TEMP%`'e gider. Ajan yalnız FAIL
  olanın logunu açar.
- **Test sırasında ayıklama yok.** FAIL nota yazılır, düzeltmesi bir sonraki
  kod partisine girer. Tek istisna, testin kendisini koşulamaz yapan bir hata
  (2026-10-19'daki silinmiş-ad tuzağı gibi). O zaman en fazla bir prob yazılır.
- **Her yeni yetenek için en fazla bir kısa test.** 3–7 sayısal kapı, uzun
  süpürme yok. Görüntü ancak sayılar karar veremiyorsa, tek mozaik.
- H1 paketi gibi uzun süitler yalnız ilgili kod değiştiyse koşulur.

## ★★ Seyrek derleme yol haritası (2026-10-20, kullanıcı isteği: "derleme çok uzun")

**Kural:** kalan iş **üç derlemede** biter. Her derlemenin kodu tamamen yazılır,
statik kontroller geçer; kullanıcı tek derleme alır; ardından ≤10 dk hızlı
duman koşar (`rt_matter_smoke.py --quick`). Derlemeler arasında yeni bulgu
çıkarsa plana girmez, aşağıdaki "Sonra" listesine yazılır. Bir derleme
shader'a da dokunuyorsa bu başlıkta yazar, kullanıcı shader'ı da derler.

### D0 — derlemesiz (şimdi, mevcut binary ile)
- ✔ ÖLÇÜLDÜ (2026-10-20): P6d hacim sıkışması. 1 cm voksel, 20 cm levha,
  625k parsel. Seed seviyesi 0.210 m (beklenen 0.217; seed doğru, kullanıcı
  hipotezi ana neden değil).

  | Tavan | Kullanılan iter. | Kalıntı | 30 kare | 150 kare | Adım |
  |---|---|---|---|---|---|
  | 24 | 24 | 0.50 (yakınsamıyor) | 0.168 | 0.161 (−23%) | 241 ms |
  | 100 | 100 | 4e-3 | 0.210 | 0.200 (−5%) | 266 ms |
  | 400 | 208 | 7e-6 | 0.210 | 0.200 (−5%) | 264 ms |

  Kök: sabit 24 iterasyon tavanı ince ızgarada basıncı kuramıyor. Daha çok
  iterasyonun maliyeti küçük (+%10). D2'de tavan sabit değil, yakınsamaya
  bağlı olacak: kalıntı hedefi ve ızgara boyuyla ölçeklenen üst sınır.
  Kalan −5% yavaş çökme çözücüden değil (400'de de aynı).
- ✔ ÖLÇÜLDÜ: ölü bant probu (2 cm, 400 iterasyon, yakınsak, 600 kare;
  seviye/beklenen, 100 karede bir):
  - 8 parsel/hücre: 0.937 → 0.889, hâlâ düşüyor (çökme);
  - 27 parsel/hücre: 0.937 → 1.044 → 1.032 (şişme).
  Kök: yoğunluk düzeltmesi **tek yönlü ve ölü bantlı**. Yalnız fazla dolu
  hücreyi itiyor, 1 standart hata altını görmüyor. Düşük ppc'de bant geniş
  (~%13): sütun bandın kenarına kadar sessizce çöküyor. Yüksek ppc'de bant
  dar: tek yönlü düzeltme sayım gürültüsünü doğrultup sütunu şişiriyor.
  Zemine çarpan suyun yassı taban yapması ve zayıf sıçraması aynı kök
  (kullanıcı gözlemi, 2026-10-20). Düzeltme D1'de.
- FLIP (P6c): öneri 0.3, kullanıcı gerekçe sordu (2026-10-20). Maliyet farkı
  yok; aynı çekirdekler, yalnız karışım katsayısı değişiyor. Taban APIC,
  zaten neredeyse enerji koruyor. 0.9'un dinlenik tanktaki kıpırtısı sayısal
  gürültü; gerçek suda yok. Dökülmedeki %13–17 kaybın ne kadarının gerçek
  olduğu bilinmiyor: referans ölçülmedi. D2'de 0.3 yazılır ve **Ritter baraj
  yıkılması cephe hızı** (2√(g·h0)) ile 0.3 ve 0.9 karşılaştırılır; referansa
  yakın olan kalır.
- ✔ T6: kullanıcı EVET (2026-10-20) → D3.

### D1 — ısı, faz, okumalar ve sıvı yoğunluğu (C++ + shader)

> **D1 ve D1b DERLENDİ.** D1b (τ 0.05 s + hızlı düşüşte taşıma kendi hızını korur) kullanıcı sahnesinde 24 fps Play'de görsel olarak doğrulandı (2026-10-20): tabaka havada çökmüyor, çarpınca sıçrama doğru. D1 test sırası (boş sahne) henüz koşulmadı. Canlıda: dinlenik tank hacmi korunuyor; çarpmada ezilme τ'dan geliyordu (tablo: NEXT_BUILD_CHECKS D1b). Dinlenik tanktaki ~0.2 m/s kıpırtı FLIP'ten, D2'de.
> Önceki not: 4. madde (boş okuma) teşhis
> testiyle bırakıldı (`rt_test_run_readback_ipc.py`); kökü bulunursa
> düzeltmesi D2'de. 3. madde gereksizdi: sayaç iskeleti zaten sayıyor.
1. Isı alanı temsil edilen gerçek tane boyunu kullanır
   (`represented_grain_radius_m`). Sıcak kum suda saniyeler içinde söner,
   şimdiki gibi ~200 s sürmez.
2. Temas anında kaynama: sıcak tanenin değdiği su, havuz ortalaması 373 K'ye
   çıkmadan buhar üretir (hücre altı film). Enerji defteri kapanır.
3. Erime sayacı iskelet taşıyıcıyı da sayar (kar → su). Bugün yalnız donmuş
   sıvının çözülmesini sayıyor.
4. `fluid.run`'dan hemen sonraki ilk okuma boş dönüyor (iki kez görüldü).
5. Kaynak okuması: `born_solid` ile "kaynamaya kırpıldı" ayrılır; gaz
   kaynağı doğum sıcaklığı raporlamaz.
6. Panel etiketi: "Enable Cooling & Freezing" → bütün ısı zincirini anlatan ad.
7. P9 gözenek ısısı kolu koşulabilir hale gelir.
8. ★ **İki yönlü yoğunluk düzeltmesi** (D0 ölü bant probu). Hedef dinlenme
   yoğunluğu; fazla da eksik de düzeltilir. Eksik tarafı yalnız iç
   hücrelerde (bütün komşuları sıvı), serbest yüzey hiç çekilmez. Ölü bant
   yok; gürültüyü daha geniş komşuluk ortalaması bastırır. Beş yolda aynı
   formül: CPU PCG, `sim_fluid_cg_residual_init`, `sim_sparse_pressure`,
   `sim_fluid_divergence_porous`, CUDA. **D1 bu yüzden shader derlemesi de
   ister.**
9. Basınç iterasyon tavanı yakınsamaya bağlı (P6d, D2'den taşındı).
Kabul ekleri: ppc 8 ve 27'de 600 karede seviye ±%2;
`rt_test_liquid_column_ipc.py` 1 cm'de ±%3; dinlenik tank tabanı bozulmaz
(`seed_settle` liquid_at_rest).
Kabul: `rt_test_snow_heat_ipc.py` + yeni kısa söndürme/kaynama testi + hızlı duman.

### D2 — taneler ve sıvı (C++, belki shader)
1. Taneler buzun içinden geçiyor (statik collider önbelleği buz yüzlerini
   görmüyor).
2. Ortak saat maliyeti: 130 alt adımı DEM'in sönüm sınırı belirliyor ve
   MPM/sıvı şeridi her alt adımda tam iş yapıyor (4 emitterli sahne 60–108 ms).
   Hedef: continuum şeridi kendi CFL'iyle, DEM alt adımları yalnız temasta.
3. DEM yatağının oturtmadan sonra 1.5 mm/kare titremesi;
   sleep_transfer'ın sınırdaki enerji kapısı.
4. FLIP ön ayarları (0.3, Ritter cephe hızıyla doğrulanır).
5. CPU PCG son kalıntıyı raporlar.
6. ★ Play / Live Update IPC'de yok (kural 1 ihlali): kullanıcı Play'de seed
   tabakasının düşerken çöktüğünü görüyor, `fluid.run` ile (dt 1/60, 1/24,
   0.1) üretilemiyor. `timeline.play` / `pause` / oynatma modu ve önbellek
   durumu IPC'ye açılır; kare döngüsü yolu testle sürülebilir olur.
7. CPU APIC taşıması aynı riski taşıyor: RK2, ızgara hızını uzatma bandı
   dışında 0 okuyor. Vulkan'da D1b'de düzeldi (sıvı hücresi dışında
   parçacığın kendi hızı); CPU'ya aynı kural.
8. **D1/D1b canlı sonucu (2026-10-21) — en önemli açık.** 1 cm dinlenik
   tankta (60×50×60 hücre, ppc 8) ortalama hız 160 karede 0.05 → 0.31 m/s
   büyüyor (τ 0.05; τ 0.5'te 0.05 → 0.16). Seviye seed'in %86'sına iniyor.
   CG yakınsıyor (residual 7e-6), yani sorun tavan değil. Dinlenik bir
   sıvıda büyüyen hız enerji pompası demek: yoğunluk düzeltmesinin hızı
   FLIP 0.9 ile parçacıkta birikiyor, her kare gürültüyü kovalıyor. 2 cm
   ppc 8 geçiyor (+%1.8), ppc 27 +%2.8 şişiyor (kapı %2). Öneri: düzeltmeyi
   hızdan çıkar. Ayrı bir yoğunluk çözümüyle yalnız konumu kaydır
   (Kugelstadt 2019, "implicit density projection"), ve FLIP 0.3 maddesiyle
   birlikte ölç. Bedeli adım başına ikinci bir basınç çözümü.
9. Aynı koşu: `seed_settle` liquid_at_rest / grains_at_rest FAIL (su p95
   4.5 mm/kare, settle açık ya da kapalı fark yok: 8. maddeyle aynı kök
   olabilir). `p9_melt` melt_conserves_mass FAIL: `mass_after_pour_kg` 0.0
   okundu (okuma boş dönmüş olabilir, ölçüm değil). `snow_heat` kaynama
   kolu FAIL: sıcak kum suya 432 kJ veriyor ama buharlaşma kontrolle aynı
   (0.45 vs 0.43 g).
Kabul: p9_floor, seed_settle, mpm_dem_clock, liquid_column + hızlı duman.

### D3 — T6 (yalnız D0'da evet çıkarsa)
Şerit anahtarları (`gas_lane_enabled` / `liquid_lane_enabled`), Gas/Liquid
ön ayarları, yüklemede göç (kaynak fazı, gölgelendirici yuvası), eski
önbelleklerin gerekçeli reddi, ölü sanal granüler yolun sökümü, 25 script.
Kabul: T0 yeniden + hızlı duman + iki domain'li eski sahne yükleme testi.

### Bu planın dışında (ayrı plan ister)
Elastik/elastoplastik MPM (sakız, jöle), XPBD ince yapılar, P7 sis/yağmur
render bağlantısı (bulut ajanı), uzun oturum yavaşlaması, yalnız-panel ayar
denetimi.

## ★ Kapanış kapsamı (2026-10-20, kullanıcı onayı) — DONDURULDU, 4/4 CANLI

Plan bu dört maddeyle kapanır. Başka hiçbir şey eklenmez. Yeni bulgular
aşağıdaki "Sonra" listesine yazılır, plana girmez.

1. ✔ CANLI: 10× tane testi yavaşlaması: kodda değil, uzun oturum
   birikimi (taze uygulamada 120 adım 13 s); koşucu temizliği + `--restart-app`.
2. ✔ CANLI: `fluid.run` (T0 onunla ölçüldü).
3. ✔ CANLI: donmuş domain'de tane adımı koşuyor.
4. ✔ CANLI: T0 EQUAL (gaz 1.06×, sıvı 0.97×, sonuçlar eşit). Karar 1'in kapısı geçti; T6 kullanıcı kararı.

Test kuralı: her partide en fazla 10 dakikalık hızlı duman. Uzun süitler
plan sonunda bir kez koşulur.

## Sonra (plan dışı, sırasız)

- ✔ YAZILDI (10. build): MPM + DEM ortak saatinde MPM parselleri (Snow) emitterde kalıyordu (alt adım başına host yüklemesi). Test: `rt_test_mpm_dem_shared_clock_ipc.py`.
- Ortak saat maliyeti (kullanıcı sahnesi 2026-10-20: gaz + kum + su + kar, 6000 parsel): adım medyanı 60 ms (40 → 79 ms, parsel arttıkça). Gaz yalnız 7 ms. 130 alt adımı DEM'in sönüm sınırı (`substep_limit damping`) belirliyor; MPM/sıvı şeridi kuvvet, temas ve taşımayı her alt adımda yapıyor.
- Kaynak okuması: gaz kaynağı `fluid_birth_kelvin` ve `fluid_born_solid` raporluyor (anlamsız). `fluid_born_solid`, "kaynama noktasına kırpıldı" ile "katı doğdu"yu aynı bayrakta birleştiriyor.
- ✔ YAZILDI (11. build), kullanıcı isteği: DEM↔MPM ısı havuzu + hücre içi parsel takası + MPM karın duvar/hava filmi. Test: `rt_test_snow_heat_ipc.py`.
- Sıcak tane suda çok yavaş sönüyor: ısı alanı simülasyon tane yarıçapından hesaplanıyor (2.5 cm → τ ~200 s). Temsil edilen gerçek tane boyu (`represented_grain_radius_m`) ısı alanına girmiyor; gerçek kum ms'de söner.
- Temas anında kaynama yok: 600 K tanenin ısısı hücrenin su havuzuna ortalanıyor; havuz 373 K'ye çıkmadan buhar çıkmıyor (hücre altı film/çekirdek kaynaması modelde yok).
- (eski bulgu) DEM↔MPM ısı kanalı yok: `MatterGrainThermal.cpp` `roleOf`, MPM sahipli granüler parseli (Snow) `Role::None` sayıyor. Kullanıcı sahnesinde (600 K kum + 263 K kar, ısı zinciri açık, 3 s) kum 213 kJ verdi (gaz 66, sıvı 70, duvar 78 kJ; artık 0 J), kar hiç ısınmadı (en düşük 263.13 K), erime 0. Su↔kar iletimi de ölçülmedi.
- Isı zinciri açıkken maliyet: medyan 89 ms (65 → 108 ms, 1500 → 9000 parsel); kapalıyken aynı sayıda ~%20–25 daha az. Karışık adım 72 ms, bunun 68 ms'si GPU beklemesi (130 ortak alt adım). Gaz 12.8 ms, yüzey takası 4.6 ms.
- Panel etiketi yanlış yönlendiriyor: "Enable Cooling & Freezing" bütün ısı zincirini açıyor (erime, kaynama, iletim).
- Elastik/elastoplastik MPM şeridi yok (sakız/jöle/hamur); sonra ince yapılar için XPBD.

- FLIP oranı kararı (P6c; ölçüm tablosu NEXT_BUILD_CHECKS 6. build).
- İnce ızgarada sıvı hacim sıkışması (P6d; `rt_test_liquid_column_ipc.py` hazır).
- sleep_transfer strong A/B enerji kapısı sınırda.
- P9 gözenek ısısı kolu (zaman aşımı yüzünden koşmadı) ve buz çarpışmasının
  canlı kanıtı (madde 3 düzelince).
- CPU PCG son kalıntıyı raporlamıyor.
- P7 sis/yağmur render bağlantısı (bulut ajanıyla).
- DEM yatağının oturtmadan sonra 1.5 mm/kare titremesi.
- Taneler buzun içinden geçiyor (868 buz yüzü var, kum zemine iniyor); aday:
  statik collider önbelleği parmak izi buz yüzlerini görmüyor.
- `fluid.run`'dan hemen sonraki ilk `fluid.matter_models` okuması 0 kg dönüyor.
- Uzun oturumda simülasyon adımının ~10× yavaşlaması (taze uygulamada yok;
  sızıntı/birikim kökü ölçülmedi).
- Yalnız panelde kalan ayarlar için IPC denetimi (sıvı basınç iterasyonu
  böyle çıktı).

- **Kullanıcı sahnesi (4 emitter: gaz, kum, su, kar; 2026-10-21):** D1b
  derlemesiyle sıcak kum suyu kaynatmadı, karı da büyük ihtimalle eritmedi.
  İzole testte (`rt_test_snow_heat_ipc.py`) kum↔kar erimesi geçiyor, kum→su
  kaynaması geçmiyor (432 kJ gidiyor, buhar kontrolle aynı). Sahne farkı
  ölçülmedi. İlk iş: aynı sahnede ısı defterini (`thermal_grain_heat_*`,
  `thermal_fusion_absorbed_j`, `thermal_in_cell_heat_j`) okuyup testle
  karşılaştırmak.
  ★ **ÖLÇÜLDÜ (2026-10-21, derlemesiz):** kök `represented_grain_radius_m = 0`.
  Yerleşik Sand'in `grain_real_radius_m` değeri 0, bu yüzden D1'in film
  düzeltmesi hiçbir hazır maddede devrede değil. Filmler 2.5 cm simülasyon
  yarıçapıyla çalışıyor. Aynı sahnede 5 s (dt 1/24), tek fark türetilmiş
  Sand'de 0.25 mm:
  kum ortalaması 598.7 → 467.6 K; suya giden ısı 0.72 → 63.6 MJ; buharlaşma
  0.025 → 10.2 kg; kar gizli ısısı 1.2 kJ → 1.35 MJ; adım medyanı aynı (~400 ms).
  İzole `snow_heat` boil kolu da yerleşik Sand'le koştuğu için aynı sebeple
  FAIL. Varsayılanı vermek ıslak kohezyonu da değiştirir: köprü kuvveti
  `2πγcosθ·R·(R/r)²` idi (`sim_matter_grain.glsl` bridge, `MatterGrainGpu.cpp`
  cohesion_scale). Bu H1-G2 B6'nın bilinçli seçimiydi: tane başına Bond
  sayısını koruyordu. Ama toplu kohezyon gerilimini α = R/r kat büyütüyordu.
  Gerilmeyi koruyan kaba tanecikleme `α²·F_gerçek(r) = 2πγR·α` ister.
  ✔ **YAZILDI (2026-10-21, kullanıcı kararı), derlenmedi:** Sand 0.25 mm,
  Gravel 4 mm, ölçek R/r. Kontrol listesi NEXT_BUILD_CHECKS'in başında.
  H1 `--wet-only` kum kalesi kapısı ölçek 2500 ile yazılmıştı; FAIL ederse
  fiziksel bir yükseklik kapısıyla yeniden yazılır.
  ✔ **DERLENDİ, CANLI (2026-10-21):** okuma PASS (0.00025). Kullanıcı
  sahnesi 5 s (artık bir collider küp ve ısısız bir gaz emitteri de var;
  derlemesiz denemeyle birebir karşılaştırılamaz): kum 600 → 480 K
  (eskiden 598.7), suya 71 MJ, buharlaşma 14.7 kg, yoğuşma 2.0 kg, gaz
  fazı 1.97 kg, tane defteri artığı 5e-8 J. Yeni bulgular:
  - ★ **Bir parsel 629.8 K** (`thermal_max_kelvin`). En sıcak kaynak 600 K
    ve gaz emitteri ısısız. Isıyı alan parsel veren taneden sıcak olamaz.
    Aday: D1'in temas dağıtımı (1/(d²+(h/4)²) ağırlığı) havuz enerjisini
    tek parsele yığıyor, parseli tanenin sıcaklığıyla sınırlamıyor. Defter
    kapalı olduğu için hiçbir toplam bunu göstermez.
    ✔ YAZILDI (2026-10-21), derlenmedi: dağıtım hücrenin tane sıcaklığıyla
    sınırlı (su doldurma). `snow_heat`'e iki üst sıcaklık kapısı eklendi.
    ⚠ Eski exe'de iki izole kol (su + kum, kar + kum) 629.8 K'yi üretmedi:
    en sıcak parçacık tam 900.0 K. Kaynak kanıtlanmadı. Diğer adaylar:
    yoğuşmanın gizli ısısı (sahnede 2.0 kg), gaz emitteri ya da collider küp.
    Kanıt kullanıcı sahnesinde parçacık bazında okuma gerektirir (hangi
    madde, hangi rol).
    ✔ **KÖK BULUNDU (ikiye bölme, eski exe):** üçlü gerekiyor. Kar + su
    havuz ortalamasını düşürüyor, taneler havuzu ısıtıyor, yakınlık payı
    tanenin yanındaki parseli tanenin ötesine taşıyor. Kum 293 K'de bile
    296 K. Kar ya da su yokken temiz. Gaz ve yoğuşma sebep değil. Yazılan
    sınır bunu kesiyor; `snow_heat` `mix` kolu üçlüyü kuruyor (tablo:
    NEXT_BUILD_CHECKS). Eski exe'de `mix` FAIL: 300.7 K, kaynaklar 293 K.
    İzole test hatayı yeniden üretiyor.
    ✔ **DERLENDİ, CANLI PASS (2026-10-21):** `snow_heat` boil + sand + mix
    11/11. Mix en yüksek 293.0003 K (önce 300.7). Boil buharlaşması 1.08 kg
    (sınırdan önce 1.03–1.08), temas kaynaması korunuyor. Hızlı duman 8/11,
    aynı bilinen FAIL'ler.
    ⚠ **Kullanıcı sahnesi (yeni exe, 5 s): taneler artık ≤ 600.0 K** (önce
    601.9, yayılma kapandı). Ama **ikinci, ayrı bir kaynak var:** bir
    parçacık geçici olarak 683.0 K'ye çıkıyor (gaz kapalı 693.8). 12 karelik
    örneklerin hiçbirinde görünmüyor, yalnız adım başı max'ta. Yine yalnız
    kar + su + 600 K kum birlikteyken.
    Elenen: (a) grain↔gas drag Tv okuyor ama yalnız yoğunluk için, ısı yolu
    değil; (b) GPU yüzey değişimi (`sim_inventory_surface_exchange`) Tv → T
    dönüşümü yapıyor; (c) parsel duvar/hava filmi ortama sınırlı.
    Kalan adaylar: kar parselinin erimesinde ısı kapasitesi geçişi (2100 →
    4186 J/kg K, mass_fraction); `measureTemperatures` sıvı ısı geçişinin
    sonunda ölçtüğü için kaynama/erime geçişinden önceki ara bir durum.
    ★ Kör ikiye bölme pahalı (koşu başı ~1 dk, kanıt yok). Önce kural 1
    açığını kapat: en sıcak N parçacığın madde, rol, kütle oranı ve konumu.
- **Kural 1 açığı (2026-10-21):** IPC'de tek tek parçacık okuması yok
  (sıcaklık / madde / rol / konum). "Bu 734 K parçacık ne?" sorusu yalnız
  ikiye bölmeyle (6 koşu) cevaplanabildi.
  ✔ YAZILDI, derlenmedi: `fluid.particle_extremes` (API + IPC + Python +
  Read yetkisi + descriptor), yeni `RtApiFluidParticles.cpp` (state_digest de
  oraya taşındı). Probe: `rt_probe_hot_particle_ipc.py`.
  ✔ DERLENDİ, CANLI (2026-10-21): link, okuma, hata mesajları, state_digest
  çalışıyor. Probe tek koşuda cevap verdi (aşağıda).
- ★ **683 K parçacık = YAĞMUR PARSELİ (2026-10-21, `particle_extremes`):**
  Kullanıcı sahnesinde 600 K'yi geçenler etiketsiz `model=fluid` parseller,
  hepsi 0.02449 kg (yağmur kuantumu), yeni kimlikler (51001–54005), zeminde
  (y ≈ −2.45). 102–108. adımlarda 614–648 K. Yanlarında bir Sand tanesi
  601.5 K (o parselin hücresinden ısınmış). Kaynak
  `sim_inventory_saturation.comp`: yağmur, otomatik dönüşümde `c·T_hava`
  enerjisiyle havuza giriyor ve parsel `re/(c·r)` sıcaklığında doğuyor.
  Elenen: Tv okuması (shader `airFromVirtual` kullanıyor), yağmur düşüşünde
  kütle/enerji ayrışması (`rain_fall` ikisini aynı oranla taşıyor). O halde
  zemin hücrelerinde **hava 600 K'yi geçiyor**.
  Aday (ölçülmedi): yoğuşmanın gizli ısısı havaya gidiyor
  (`air = t0 + d·L/C`). Kaynama yüzünden hücrede 1 atm saf buhar yoğunluğunun
  üstüne buhar birikirse (600 K'de ~0.37 kg/m³, hücre başına ~72 mg), denge
  sıcaklığı kaynama noktasının çok üstüne çıkar. Mol kesri > 1 olunca
  `airFromVirtual` da anlamsız sonuç verir. Aynı koşudaki 14.7 kg
  buharlaşma / 1.97 kg gaz fazı bu adayla uyumlu.
  İlk ölçüm: zemin hücrelerinde hava sıcaklığı ve hücre başına buhar
  kütlesi. Gaz hücresi düzeyinde okuma yoksa, en sıcak N gaz hücresi okuması
  (kural 1). Düzeltme yağmur/sis yoluna aittir (FOG/T2 notları).
  ✔ YAZILDI, derlenmedi: `fluid.gas_cell_extremes` (en sıcak / en buharlı N
  hücre, Tv, T, buharın 1 atm saf buhara oranı, sis, yağmur sıcaklığı).
  Probe her olayda bunu da basıyor.
  - Kar erimesi 1.35 MJ → 25 kJ. Küp kumla karın temasını kesmiş olabilir.
    Ölçülmedi.
  - Buhar defteri: 14.7 − 2.0 = 12.7 kg net buhar var, gaz fazında 1.97 kg.
    ~10.7 kg'ın nereye gittiği hâlâ belli değil.
  - Kullanıcı gözlemi (aynı build): su ve kar, kumla temas noktalarında
    buhar üretiyor. D1 temas kaynaması görsel olarak da çalışıyor.
  - `snow_heat` boil + sand **7/7 PASS**. Boil: sıcak kum 1.03 kg buharlaştırıyor,
    kontrol 0.5 g (önce 0.45 / 0.43 g). Sand: erimeye giden ısı 0.38 → 1.0 MJ.
    263 K'lik kontrol kumu da karı biraz eritiyor (63 kJ). İnce tane artık
    293 K'lik havadan hızla ısınıp ısıyı kara aktarıyor; fiziksel, kapı 3×
    farkla geçiyor.
  - H1 `--wet-only` **FAIL, beklenen**: 2854 köprü var, ama rms ıslak 0.324 /
    kuru 0.323 m. Yeni ölçekte tane başına köprü ~0.53 N, tane ağırlığı
    ~1.7 N, toplu kohezyon ~1 kPa (gerçek ıslak kum mertebesi). Islak kumun
    ayakta kalabildiği yükseklik ~4c/(ρg) ≈ 25 cm; 0.45 m'den düşen top bu
    sınırda. Eski kapı 50× fazla yapışkan kumu ölçüyordu. Sonra: kapıyı
    fiziksel bir yüksekliğe çevir (ör. H < 4c/(ρg) olan bir sütun
    dağılmaz, çok daha yükseği çöker).
  - Hızlı duman 8/11. Bilinen açıklar: `p9_melt` (D2-9), `p9_floor`
    sand_rests_on_ice (D2-1, buzun içinden geçiş), `seed_settle`
    liquid/grains_at_rest (D2-8/9). Yeni: `seed_settle` grains_control_moves
    (açık 1.49 / kapalı 1.24 mm p95). İkisi de D2-3'ün ~1.5 mm/kare titreme
    tabanında; kapı iki gürültüyü karşılaştırıyor. Kuru kuma bu partinin
    hiçbir yolu dokunmuyor.
  Yeni açık: 10.2 kg buharlaşma var ama `gas_phase_mass_kg` 0.66 kg
  (yoğuşma toplanmadı; buhar defteri ayrıca okunmalı).
- **Collider yüzeyi: DEM yakalıyor, sıvı/kar içine giriyor (kullanıcı,
  2026-10-21):** 4 emitterli sahneye bir collider küp eklendi. DEM taneleri
  küpün yüzeyinde duruyor; su ve kar (ızgara/continuum parselleri) yüzeyin
  içine girip orada etkileşiyor. Ölçülmedi. Aday: continuum şeridi
  collider'ı voksel hücresi olarak görüyor (bu sahnede 5.8 cm), DEM ise
  üçgenleri. Önce derinliği ölç: içerideki parselin yüzeye uzaklığı < 1
  voksel ise çözünürlük, daha derinse SDF/hücre işaretlemesi ya da
  parsel itme yolu hatalı.
- **Gaz render'ı tek malzemeyle düzenleniyor (kullanıcı, 2026-10-21):**
  domain'in gaz verisi tek bir hacim malzemesiyle çiziliyor. Gaz emitter
  nasıl görünüyorsa, kaynayan sudan çıkan buhar da öyle görünüyor (duman
  ile su buharı ayırt edilemiyor). Hedef: malzeme kaynak başına ya da veri
  (tür/kanal) başına bağlansın. Buhar kendi malzemesini alsın, emitter
  dumanı kendi malzemesini. Panel ve IPC tarafı da (kural 1). Ayrı bir
  render partisi; madde planını kapatan derlemelere girmez.

## Partiler (sıra)

| # | Parti | Neden şimdi | Durum |
|---|---|---|---|
| P0 | 2026-10-19'un dört düzeltmesi: `temperature_fresh`, `add_primitive` silinmiş ad, tane duvar filmi, readiness koruması | Yazıldı | CANLI (duman 10/10) |
| P1 | **Kapasitesiz ısı yutaklarının sökümü.** Sıvının kapalı duvar "temas" soğutması (`coolThermalLiquid`, ortam termostatı) ve MSF ortam gevşemesi (`cooling_rate`) film + kapasite modeline geçer. Termal alan "yerel ortam"a film üzerinden ısı veren bir kaynak olur. | Tane testi bu yolun %97 enerji yuttuğunu gösterdi. Sıvı ve kabuk aynı sınıf. Kural 5 gereği `cooling_rate` emekliye ayrılır. | CANLI, duman 10/10 (alan dış kaynak olarak kaldı, hızı alanın `thermal_rate_per_s`'i) |
| P2 | **Buhar hücresinde gerçek hava sıcaklığı.** Yüzey filmleri (MSF gather, tane↔hava) Tv yerine T okur: buhar envanteri bağlanır, `airKelvinFromVirtual`. | FG1'de aynı sınıf düzeltildi; kaynayan suyun üstündeki film sıcak çalışıyor. | CANLI, duman 10/10 (envanter host'taysa tane↔hava kuru varsayar) |
| P3 | **Tane domain'inde donma.** Donmuş sıvı için bir sahip (statik engel taşıyıcı) ya da donan suyun Ice tanesine dönmesi. `freeze_held_by_grains` sıfıra iner. Ice tane erimesinin canlı kanıtı aynı testte. | Tane + soğuk = bugün aşırı soğuma; "tek domain" iddiasının son termal deliği. | CANLI, duman 10/10: donmuş sıvı iki şeridin dışında tutulan buz; taneler buza çarpmıyor (açık) |
| P4 | **Tane ısı geçişinin ölçeği.** Hücre başına OpenMP; 1e5 tanede `thermal_grain_ms` hedefi < 5 ms. Gözenek suyu tane kapasitesine katılır, bütün geçişlerde tutarlı olarak. | Yoğun sahneler (H1 dense 16k) termal açılınca seri döngüye takılır. | CANLI, duman 10/10 (iki faz paralel; gözenek suyu kapasitede) |
| P5 | **`fluid_extinguishing` söküldü** (ajan kararı, 2026-10-19). Bayrakla birlikte `cooling_power` ve `oxygen_dilution` da ölüydü (FG1 eski söndürmeyi sökmüştü; buhar O2'yi fiziksel seyreltiyor). Madde alanı, domain aynaları, proje/sahne serileştirme, panel ve hash'ten çıktı; `substance.set` emekli gerekçesiyle reddeder. `fluid.set_combustion` artık yakıt olmayan sıvıyı tutuşturamaz (`enabled && flammable`). | Kural 5 | CANLI (madde 1/1, FG1 8/8) |
| P6 | **Bayat test temizliği.** 11 statik `check_matter_*` PASS (revizyon ve önbellek sürüm pinleri "en az" kuralına çevrildi, grain kilidi yokluğu, frame cache `std::move`). `g2_dry_wet` Soil'e (MPM) geçti. `sleep_transfer`: "projectile not born" kökü yatak çöküşü (COM y 0.030, beklenen ~0.075), doğum yeri dolu; test artık bu sebebi adıyla raporlar. Çöküşün regresyon olup olmadığı açık. | H1 süitini yeniden güvenilir yapmak | CANLI: statik 11/11, g2_dry_wet PASS; sleep_transfer yatak çöküşüyle FAIL (açık: kararlı yatak mı, regresyon mu) |
| P6b | **Seed oturtma.** Seed alt-kafesi tane yarıçapını ve temas yükünü bilmiyor; bu yüzden ilk karede çöküş ve çalkantı oluyor. Çözüm dinamik gevşeme (`MatterSeedSettle.inl`): alanın kendi adımları zaman, kaynak, ısı ve gaz dururken koşar, her adımdan sonra hız ve spin sıfırlanır. API: `fluid.settle`, panel ve `settle_*` okuması. sleep_transfer tek katlı yatağa ve `fluid.settle`'a geçti. | Kullanıcı gözlemi (Fill Domain çalkantısı); sleep_transfer aynı sınıfta | CANLI, kısmi: sleep_transfer 5/5; seed_settle 2/5, durma ölçütü (hız) çözücü tabanında takılıyor, sıvıda 0.2 m/s dinlenme tabanı şüpheli. Yer değiştirme ölçütü + düzlük algısı + kare başı konum okuması YAZILDI (5. build). Tank tabanı kökü: FLIP 0.90 gürültüsü (FLIP 0 → 0.004 m/s) + boyutsuz tam sayı yoğunluk düzeltmesi; çözüm kararı açık (P6c) |
| P6c | **Sıvı dinlenme tabanı (karar).** Dinlenik tankta 0.16–0.21 m/s FLIP gürültüsü. Seçenekler: suyun FLIP oranını düşürmek / FLIP oranını h ve dt'ye bağlamak / yoğunluk düzeltmesini boyutlu gevşeme süresi + yumuşak yoğunlukla yazmak. | Kullanıcının gördüğü "su durgun başlamıyor"un asıl kökü | Yoğunluk kısmı CANLI (sıvı testleri PASS). Su konumla da p95 4.35 mm/kare çalkalanıyor (FLIP payı). DEM yatağı oturtmadan sonra da 1.5 mm/kare titriyor (kök açık). Oturtma eşleşme kusuru düzeltildi (6. build). FLIP ölçümü: 0.9→0.3 hareketi 5–6× azaltıyor (1.8→0.33 mm/kare ortalama); ince ızgara azaltmıyor; FLIP 0.3'te 0.02 m/s taban kalıyor (FLIP dışı kaynak, sınır adayı). Dökülme: 0.3 tepe hızları korur, duvara tırmanma/sallanmanın %13–17'sini kaybeder, geç evrede %45 daha sakin. Karar: kullanıcı. FLIP kısmı: 5. build sonrası ölçümle (0.9/0.3 × 5/2.5 cm) |
| P6d | **Sıvı hacim sıkışması ince ızgarada.** 1.25 cm'de −15%, 1 cm'de −26/−32% (duran ve düşen levha); 5 ve 2.5 cm temiz. Baş aday basınç iterasyon tavanı 24. İterasyon/kalıntı IPC'ye açıldı + seed kesilme uyarısı/okuması YAZILDI (7. build); ölçüm `rt_test_liquid_column_ipc.py` 24/100/400. Kullanıcı hipotezi (2026-10-20): seed hücrelere eşit dağıtıyor, sim başlayınca parçacıklar fiziksel konumlarına çökerken yükseklik düşüyor; aynı test bunu iterasyon etkisinden ayırır. PARK EDİLDİ (kullanıcı kararı: ana plana devam). | Kullanıcı gözlemi; 5. build yoğunluk düzeltmesi maskeyi kaldırdı | açık |
| P9 | **Birleşik maddenin delikleri.** Taneler buza çarpıyor (donmuş hücre yüzleri statik üçgen). Ice tane erimesinin canlı testi. Cihazdaki buhar envanteri tane↔hava filmine indiriliyor. Emilen gözenek suyu ısısını taşıyor. | P8 ölçümünden önce Matter eksiksiz olmalı | CANLI kısmi: Ice erimesi PASS (kütle kapısı okuma artefaktıyla düştü); donmuş domain'de tane adımı artık koşuyor ama taneler buzun içinden geçiyor (Sonra); gözenek kolu koşmadı (Sonra) |
| P7 | Sis/yağmur render bağlantısı, yağmur doğum yeri ölçümü | Bulut ajanıyla ortak karar | ertelendi |
| P8 | T0 eşitlik ölçümü, sonra uzman Gas/Liquid domain'lerinin geleceği | Plan kuralı: ölçülmeden kaldırılmaz | CANLI: T0 EQUAL (gaz 1.06×, sıvı 0.97×). T6 (uzman tiplerin sökümü) kullanıcı kararı bekliyor |

Test turu: P0–P4 yazıldıktan sonra **tek derleme ve tek duman koşusu**. P6
araya, kod gerektirmeyen bir anda girebilir.

## Duman koşusu (en sonda)

`python scripts/test/rt_matter_smoke.py` koşuyu yapar. Varsayılan kollar hızlı
olanlardır: madde, gaz↔tane, Ek 11, yüzey suyu, FG1, tane ısısı.
`--with-h1` uzun H1 süitini de ekler.
