# Sıvı ↔ MPM bağlaşımı: temas itmelerinden basınç + sürüklemeye

> **Durum:** AKTİF — 2026-11-10: adım 1–2e CANLI (2c'nin kesir tamponu Vulkan'da yoktu → 2e). ★ Tarama 2e'de hâlâ kayıyor (2.63/2.86/1.33 vs 0.72): kök §12 — sürükleme sıvı saatindeydi, kar serbest kütle sayılıyordu. 2f YAZILDI (derlenmedi): sürükleme + basınç itişi ızgara saatinde, etkin hareketlilik.

## 1. Neden

Bugün sıvı ile MPM (kar, çamur, elastik) arasındaki TEK bağ `sim_matter_contact`:
her MPM ızgara adımında, birbirine yaklaşan her yüzde iki hız alanı arasında **tam esnek olmayan
normal itme** + Coulomb sürtünmesi (μ = domain'in çözülmüş `granular_friction_angle`'ı — kar+su
domain'inde Water'ın 35°'si, karın değil).

Ölçülen sonuçları (h 2.5 cm, kar bloğu + çöken su; sıvı |p| kg·m/s, 1 s):

| | |
|---|---|
| saf su (kar yok) | 5.0 |
| senkron, μ 0.7 | 0.08 |
| senkron, μ 0 | 0.41 |
| sıvı kendi saatinde | 1.8–4.9, sıvı adımına bağlı (yakınsamıyor) |

1. **Fizik yanlış:** gerçek suda duvara doğru hız sıkıştırılamazlık basıncıyla teğete yönlendirilir,
   yok edilmez. Her alt adımda (saniyede 180–4800 kez) yaklaşma hızını silen bir itme, alt adım
   sayısıyla büyüyen yapay bir sürüklemedir: su kara değince durur (kullanıcı gözlemi 2026-11-09).
2. **Zaman adımında tutarsız:** itme oran değil, çağrı başına olay. Bu yüzden sıvının kendi saati
   yakınsamıyor ve maliyet işi (`matter_liquid_multirate`) kabul edilemiyor.
3. **Arşimet yok:** kar suya basınçtan kuvvet almıyor; batma/yüzme yalnız temas itmelerinin yan ürünü.

## 2. Mevcut altyapı (DEM için yazılmış, MPM hariç tutulmuş)

`MATTER_H1_GRAIN_ROADMAP.md` B3/B5 — taneler (DEM) için unresolved CFD-DEM tam hali zaten canlı:
- `sim_fluid_divergence_porous`: ∇·(ε u_sıvı + (1−ε) u_tane) = 0 (karışım sürekliliği); ε yüz
  ağırlıklarında, tane hızı sv*'de. ε ≥ 0.3 tabanı (sıkı yatak ~0.36–0.4; daha kapalı yüz CG'yi durdurur).
- Örtük Di Felice çift sürüklemesi (gözeneklilik üstelli, Re→0'da Stokes) — eşit-zıt, oran tabanlı.
- Taneye basınç kuvveti: sıvının kendi ivmesinden, ρ V (Du/Dt − g) (hidrostatik kaldırma yerine).
- Yol haritası satır 35: "MPM varken sıvı porous projection kapalı" — MPM bilerek dışarıda; neden
  `MatterGrainStep.inl` notunda: "porous ağırlıkların sahip başına ayrı görünümü yok".

## 3. Tasarım: MPM ızgarasını gözenekli, hareketli katı olarak sıvının basıncına sok

Sıvı adımında (senkron yolda her adım, kendi saatinde `liquid_stride`'da bir), sıvı P2G ile MPM
P2G (gerilme itmesi dahil) bittikten sonra:

**3.1 Gözeneklilik ve katı hızı (yeni çekirdek, ızgarada):** her MAC yüzünde MPM yüz kütlesinden
katı hacim kesri φ_s = m_mpm / (ρ_bileşen · V_yüz), ε = clamp(1 − φ_s, 0.3, 1).
- ρ_bileşen: kar → buz 917 kg/m³ (`kIceDensity`; J_p de buza doğru sıkışıyor), elastik → kendi
  yoğunluğu, MPM kum → tane yoğunluğu. ★ Karar: buz yoğunluğu = su kar gözeneklerine **sızar**
  (sulu kar). Toplu kar yoğunluğu (400) seçilirse kar su geçirmez hafif blok olur — ayrı test.
- Collider ile birleşim: açık kesir = min(collider açıklığı, ε); katı hızı = collider ve MPM
  hızlarının kesir ağırlıklı ortalaması (`sim_matter_liquid_wall` sv*'yi zaten yazıyor).
- Mevcut `porous_solid_velocity` + `*_divergence_porous` çekirdekleri kullanılır: yeni projeksiyon
  çekirdeği YOK.

**3.2 Basınç tepkisi MPM'e (yeni çekirdek):** projeksiyondan sonra her yüzde
Δ(m u)_mpm = −dt (1−ε) V_yüz ∇p. Arşimet ve basınç itmesi buradan gelir. Kendi saatinde: aralık
boyunca **sabit kuvvet** olarak her MPM alt adımında uygulanır (CFD-DEM'de akışkan kuvvetinin
akışkan adımı boyunca sabit tutulması gibi) — darbe değil, kar zil sesi gibi çınlamaz.

**3.3 Sürükleme (sim_matter_contact'ın yerine):** sıvı–MPM yüz çiftinde örtük, oran tabanlı
Darcy/Di Felice: göreli hız u_rel ← u_rel · exp(−dt/τ), τ = ρ_s k / (μ ε),
Kozeny–Carman k = d² ε³ / (180 (1−ε)²). İtme kütle ağırlıklı eşit-zıt (momentum birebir).
d: maddenin tane çapı (kar ~0.5–1 mm). Örnek: ε 0.58, d 1 mm → τ ≈ 10 ms; kar içindeki göreli
akış ~10 ms'de söner, ama bu bir ORAN: alt adım sayısından bağımsız. Kendi saatinde her MPM alt
adımında donmuş sıvı alanına karşı uygulanır, sıvı payı aralık sonu delta G2P'siyle verilir
(zaten var).

**3.4 Yüzey (kar yüzeyinde akan su):** ε geçişi yüzeyde doğal sınır tabakası verir; ayrıca
Coulomb sürtünmesi YOK. Su kar üzerinde basınçla yönlenir, sürüklemeyle yavaşlar.

**3.5 Eklenmiş kütle:** kar (buz 917 + gözenek suyu) sudan hafif değil ama kuru kar bloğu öyle.
3.2 bölümlenmiş (projeksiyon önce, tepki sonra). Titreşim görülürse (§5 test 1/4): MPM yüz
hareketliliğini matrise ekle — yüz katsayısı β = ε + (1−ε)·ρ_su/ρ_bileşen (Robinson-Mosher 2008
tarzı monolitik, simetrik kalır). `*_var` matris çekirdeklerine bir hareketlilik tamponu.

## 4. Ne silinir, ne kalır

- `sim_matter_contact` sıvı–MPM için sökülür (kabulden sonra; kural: iki yol yaşatma).
- DEM yolu değişmez; üçlü (sıvı+DEM+MPM) için ε = 1 − φ_DEM − φ_MPM, ayrı sahip görünümleri.
- `granular_friction_angle`'ın varsayılan maddeden (Water) gelmesi ayrı bir hata: sıvı–MPM
  bağında artık kullanılmayacak, ama DEM/MPM kendi içinde parsel maddesini okumalı (ayrı iş).

## 5. Kabul (fizik referansı — senkron yolla "aynı olmak" DEĞİL)

| # | Test | Referans | Eşik |
|---|---|---|---|
| 1 | Yüzen elastik blok (ρ 500) | Arşimet: batan kesir 0.5 | ±%5, 2 s'de durulur, titreşim yok |
| 2 | Kar yatağına su sızması | Darcy: q = k g / ν (seçilen d ile) | ıslanma cephesi hızı ±%30 |
| 3 | Kar bloğuna çarpan su (dam-break) | aynı şekilde collider (doğrulanmış sıvı duvarı) | tırmanma yüksekliği ±%15; |p| 1 s'de saf suyun ≥%30'u |
| 4 | Sıvı adımı yarıya (kendi saati) | kendisi | 0.5–1 s zaman ortalamalı |p| ±%10 (tek an gürültülü: aynı kurulum 1.75/4.87) |
| 5 | Yerçekimsiz karışım | korunum | toplam momentum (sıvı+MPM) ±%1 |
| 6 | Kuru kar | regresyon | `rt_matter_smoke.py --quick` 16/16 |
| 7 | Kullanıcı kar+su sahnesi | maliyet | kendi saatinde kare ~0.7 s; sınırda takılma yok |

★ Sessiz başarısızlık: ε tabanı (0.3) yüzünden yoğun karda sıvı "olması gerekenden" hızlı sızar
ve makul görünür. Test 2'yi yatak gözenekliliği 0.3'ün üstünde ve altında iki kez koş.

## 6. Derleme maliyeti

Hedef: **yalnız .inl/.cpp + yeni shader'lar → artımlı derleme.** Yeni durum
`MatterGpuStep.inl` yerel çalışma zamanı yapısında; tane çapı ve bileşen yoğunluğu çağrı yerinde
maddeden okunur (header'a alan eklenmez). A/B anahtarı EKLENMEZ: kabul senkron yola değil fizik
referanslarına; eski davranışın sayıları bu notta ve `MATTER_COKLU_ZAMAN_ADIMI.md`'de kayıtlı,
geri dönüş git'ten. Header değişikliği gerekirse işin başında söylenir ve tek tam derlemede toplanır.

## 7. Sıra

1. 3.1 + mevcut porous projeksiyonu MPM'e aç; 3.2 tepki. Test 1, 5, 6.
2. 3.3 sürükleme, `sim_matter_contact` sıvı–MPM'den çıkar. Test 2, 3.
3. Kendi saatiyle: sabit kuvvet + alt adım sürüklemesi. Test 4, 7; sonra `matter_liquid_multirate`
   varsayılan olur ve anahtar sökülür.
4. Gerekirse 3.5 monolitik hareketlilik.

## 8. Canlı sonuçlar 2026-11-10 (adım 2 + 2b)

**2b:** sürükleme yalnız sıvı adımında, sıvının dt'siyle (yarı-örtük). Adım 2'de her MPM alt adımında
donmuş sıvı alanına karşı çalışıyordu: aralık başına bir tam eşitlenme.

Sıvı |p| 0.5–1.5 s ortalaması (kg·m/s), kar bloğu + çöken su, h 2.5 cm:

| sıvı dt | saf su | kar+su (2b) | oran |
|---|---|---|---|
| 1/24 | 5.19 | 2.12–2.14 (iki koşu) | 0.41 |
| 1/48 | — | 2.92 | — |
| 1/96 | 6.58 | 2.48 | 0.38 |
| 1/192 | 6.61 | 1.62 | 0.25 |
| 1/384 | — | 1.27 | — |

- Eski temas yolunda oran ~0.02 idi: **su artık akıyor.** Kar yerinde (yana kayma ≤ 0.5 cm).
- **Saf su 1/96–1/192'de yakınsıyor; kar+su yakınsamıyor**, adım küçüldükçe sönüm artıyor ve
  düzleşmiyor. Sürüklemenin kendisi artık oran tutarlı; adım sayısıyla büyüyen başka bir karla ilgili
  terim var. Adaylar: (a) her sıvı adımında yeniden kurulan gözenekli kısıt (sv = o anki MPM hızı)
  projeksiyonun enerji almasına yol açıyor olabilir; (b) gözenek hücrelerinde FLIP'in PIC payı; (c)
  sürükleme ve basınç tepkisinin sıralaması (split hatası). Ayırmak için sürükleme ölçeği ve gözenekli
  projeksiyonun ayrı ayrı kapatılabildiği bir tanı turu gerekiyor.
- `rt_test_matter_liquid_drag_ipc.py` 5/5 geçti (1/24↔1/96 %14) — ★ ama tarama yukarıdaki gibi:
  kapı sınırda geçti, yakınsama kanıtlanmadı. Anahtar varsayılan yapılmamalı.
- Sulu kar: 6 s'de yüzeye çıkıp dengede (tepe, gözeneklerin aldığı suyla düşen yüzeyin ~1 voksel
  altında, ~%96 batık beklentisiyle uyumlu). Blok 962 kg/m³, sudan %3.5 hafif: yavaş çıkması fiziksel.
  Test 6 s ve gözenek dolumlu yüzeyle düzeltildi: 6/6.
## 9. Literatür taraması (2026-11-10)

- **Xie, Navas, López-Querol, arXiv 2608.00578 (Ağu 2026)** — yarı-örtük çift noktalı MPM, gözenekli
  deforme ortamda serbest yüzey + sızma. Bizim tasarıma en yakın: karışım sürekliliği
  (1−n)∇·v_s + n∇·v_w = 0; Darcy sürüklemesi p̂ = n²ρ_w g/k (v_w − v_s) **tahmin adımında örtük**
  (eski katı hızına karşı); basınç denklemi **iki fazın hareketliliğiyle**:
  Δt((1−n)/ρ_s + n/ρ_w)∇²δp = −[(1−n)∇·v*_s + n∇·v*_w]; düzeltme her iki faza (su n∇δp, karışım ∇δp).
  ★ Bizim adım 1 katıyı projeksiyonda **sabit hızlı** (Dirichlet) görüyor, tepkiyi sonra veriyor:
  matristeki (1−n)/ρ_s terimi yok. Bu tam da §3.5'te ertelenen monolitik hareketlilik terimi ve kalan
  adım bağımlılığının (her projeksiyonda katıya "duvar" muamelesi) güçlü adayı.
  Sınır: deneylerde dt ~1e-4 s (su CFL'i); alt adım farkı çözmüyor; Forchheimer terimi gecikmeli.
- **Jiang, "Substepping the Material Point Method", arXiv 2508.11722 (Ağu 2025)** — S açık MPM alt
  adımını tek büyük adım gibi sunmak: alt adımlardan sonra parçacık sekant hedefleri
  (v*_p = Δx_p/Δt, C*_p = (F^{n+1}F^{n,-1} − I)/Δt) APIC ile **büyük adımın başındaki ızgaraya**
  geri saçılır; büyük adım çözücüsü (sıvı) katıyı bu tek hızla görür. Çözüm yok, kapalı form.
  ★ Bizim sıvı adımı MPM ızgarasını o anki alt adımda okuyor (anlık hız); doğrusu aralık boyunca
  ortalama/sekant hareket. Hafif sönüm getirir; deney ve yakınsama analizi yok.
- **Li ve ark., "A Dynamic Duo of Finite Elements and Material Points", SIGGRAPH 2024** — örtük FEM ile
  açık MPM'i çok farklı adımlarla (IMEX, asenkron) iki yönlü bağlıyor; bağ IPC temas potansiyeli.
  Asenkron bağın üretimde kabul gördüğünün örneği.
- **Gao ve ark., "Animating fluid sediment mixture in particle-laden flows", SIGGRAPH 2018** — iki MPM
  ızgarası arasında momentum alışverişi (sürükleme) ile sıvı–tortu; adım 2b'deki örtük çift
  sürüklemesinin grafik tarafındaki karşılığı.
- **Fang ve ark. 2018 (bölgesel zaman adımı) / Hu & Fang 2017 AsyncMPM** — blok başına 2^k adım,
  senkron MPM'e göre ~6×; MPM'in kendi içindeki asenkronluk için (dinlenik kar).
- **Goswami 2024, Computer Graphics Forum** — kar/buz yöntemleri taraması; örtük ve öğrenilmiş MPM
  hızlandırma yönleri.

**Uygulanabilir sonuçlar (öncelik sırası):**
1. Basınç matrisine katı hareketliliği: yüz katsayısı ε/ρ_l + (1−ε)/ρ_c (Xie Eq. 21) ve düzeltmenin
   iki faza birlikte uygulanması — §3.5'i şimdi yap; kalan adım bağımlılığının ilk şüphelisi.
2. Sürüklemeyi tahmin adımında örtük, katının eski hızına karşı (zaten 2b'de benzer).
3. Sıvının gördüğü kar hızı: anlık alt adım ızgarası yerine aralık sekant hızı (Jiang 2025).
## 10. Senkron limite tarama (2026-11-10, 2b kodu) — kalan hata BÖLME hatası

Kar+su ort. |p| (0.5–1.5 s), sıvı adımı = çağrı adımı, MPM alt adımı sabit (2.08e-4 s):

| sıvı dt | 1/24 | 1/48 | 1/96 | 1/192 | 1/384 | 1/768 | 1/1536 | 1/4800 (= MPM dt, senkron) |
|---|---|---|---|---|---|---|---|---|
| |p| | 2.13 | 2.92 | 2.48 | 1.62 | 1.27 | 1.00 | **0.70** | **0.73** |

- **Yakınsıyor:** sıvı dt ≤ ~0.65 ms'de ~0.72'ye oturuyor. Bu modelin (gözenekli projeksiyon + Darcy,
  d 1 mm) doğru cevabı bu; saf su 6.6 → su kar gözeneklerine girip Darcy'yle duruyor, ~%11 kalıyor
  (kar 9.6 L, gözenek 5.4 L; su 22.8 L).
- **Hata türü:** kendi saatin bölme hatası, sertlik kaynaklı: sürükleme zaman sabiti τ ≈ 10 ms; λ·dt_sıvı
  ≪ 1 olmadıkça (dt ≪ 10 ms) sonuç kaymış. Basınç projeksiyonu ile sürükleme ayrı ayrı uygulandığı
  için büyük adımda projeksiyonun ürettiği akı sürüklemeyle bozuluyor.
- **Düzeltme (2d):** sürüklemeyi basınç adımının İÇİNDE çöz. Yüz başına örtük 2×2 sistem
  (m_l + K dt, −K dt; −K dt, m_s + K dt), K = βV: basınca karışım akısının tepkisi M_eff ve ön akı F0
  kapalı formda → projeksiyon matrisi M_eff ile, sonra her iki fazın yüz hızı aynı 2×2'den. Ayrı
  sürükleme ve tepki çekirdeklerinin yerini alır; büyük sıvı adımında da sertlikten bağımsız (Xie 2026
  tahmin adımındaki örtük sürüklemenin projeksiyonla birleşik hali). Kabul: aynı tarama 1/24…1/1536
  hepsi 0.72 ± %15.
## 11. 2d türetmesi: sürükleme projeksiyonun içinde = sürükleme projeksiyondan önce

Yüz başına (G = dt·∇P/ρ_l, a = εVρ_l/m_l ≈ 1, b = (1−ε)Vρ_l/m_s ≈ ρ_l/ρ_c, k_l = K dt/m_l,
k_s = K dt/m_s, k = k_l + k_s, d* = u_l* − u_s*, c = ε k_l − (1−ε) k_s):
- Birlikte örtük çözüm: F = F* − c d*/(1+k) − G [εa + (1−ε)b − c(a−b)/(1+k)].
- Önce örtük sürükleme (d → d*/(1+k)), sonra projeksiyon: F = F* − c d*/(1+k) − G [εa + (1−ε)b]
  ve fazlar ayrı ayrı a G, b G alır.
- Fark yalnız c(a−b)/(1+k)·G terimi: basıncın yarattığı göreli hızın aynı adımdaki sönümü. Kar için
  a − b ≈ −0.09 → ihmal edilebilir (sonraki adımda sönüyor). Ağır katıda (kum, a − b ≈ 0.6) bu terim
  büyür: o zaman M = εa + (1−ε)b − c(a−b)/(1+k) matrise ve tepkiye eklenir (§3.5'in tam hali).
- Xie ve ark. 2026 da sürüklemeyi tahmin adımında örtük alıp projeksiyonu (1−n)/ρ_s + n/ρ_w ile
  kuruyor: 2c + 2d bu şemanın kendisi.

## 12. Zaman ölçekleri: neden yakınsamadı, 2f (2026-11-10)

2c+2d canlı (2e ile kesir gerçekten bağlıyken): tarama 1/24 **2.63**, 1/96 **2.86**, 1/384 **1.33**
(hedef ~0.72); sulu kar yalnız 2.6 cm yükseliyor. Ölçüm yerine ölçek analizi:

| Olay | Süre |
|---|---|
| Kar iskeleti elastik dalgası | ~0.2 ms (0.35 h/c → kare başına 200 alt adım) |
| Gözenek sürüklemesi (d 1 mm, φ 0.44) | τ ≈ 9 ms (β ≈ 6.2e4, β/(ρ_su ε) ≈ 110 s⁻¹) |
| Sıvı CFL | 10–40 ms |
| Basınç | kısıt, anlık |

Kural: her bağ kendi en hızlı ölçeğinin saatinde. Basınç sıvı saatinde (doğru, maliyet bu);
sürükleme de sıvı saatine kondu (2b/2d) — yanlış.

**Kayma:** sıvı adımında tam örtük çift gevşemesi ortak hıza gider:
v = (m_l u_l + m_s u_s)/(m_l + m_s); gözenek hücresinde m_l ≈ 560, m_s ≈ 403 → zemindeki kar için
**v ≈ 0.58 u_l**: su adım başına momentumunun yalnız %42'sini kaybeder, karın aldığı pay sonraki
200 elastik alt adımda zemine gider. Etkin sönüm ln(1/0.58)/Δt: 1/24'te ~14 s⁻¹, 1/96'da ~54,
1/384'te ~90, çözünmüş ~110 s⁻¹. Yön ve büyüklük taramayla uyumlu; 2c/2d'nin bir şey
değiştirmemesi de: ikisi de sürüklemeyi sıvı saatinde tuttu.

**Yüzme (2c hatası):** projeksiyon karı serbest kütle (r = ρ_l/ρ_c ≈ 1) sayıyor ama karın tahmin
hızına aralık boyunca yerçekimini koymuyor → karışım akısının (1−ε) payı "zaten duruyor" sayılır,
hidrostatik basınç ~ε kadar kurulur, kar az kaldırılır. Uzun adımda sert iskelet kendi ağırlığını
taşır: katı bloğu M + Δt²K; yerel köşegen 1 + 6(cΔt/h)², cΔt/h = 0.35·n (n = aralıktaki alt adım,
alt adım sayısı elastik CFL'nin kendisi) → **r_etkin = r / (1 + 6(0.35 n)²)**: n = 1'de 0.58r,
n = 4'te 0.08r, n = 200'de 1e-4r = hareketli gözenekli engel (adım 1'in yüzen formülasyonu).

**2f (yalnız `MatterGpuStep.inl`):**
- Sürükleme her MPM ızgara adımında, ızgara dt'siyle, iki tarafı birlikte (sıvı MAC + MPM yüzü,
  eşit-zıt); sıvı payı aralık sonu delta G2P ile parçacıklara (zaten vardı). Adım 2'nin ilk
  hâlinden farkı: o sürüme göre projeksiyon artık r_etkin ile karı engel görüyor ve basınç itişi
  de alt adımlara yayılıyor.
- Basınç itişi (tepki) son projeksiyonun gradyanıyla her ızgara adımında, ızgara dt'siyle —
  §3.2'nin "sabit kuvvet, darbe değil" tarifi; aralık toplamı eskisiyle aynı.
- Maliyet: kare başına ~200 yüz geçişi (tahmin ~10 ms); basınç yine 1–4 kez.
- **Yanlışlanabilir tahmin:** tarama 1/24'te de 1/384 ile ±%15 içinde (≈0.7). Olmazsa analiz yanlış.

**2f canlı (17:20 derlemesi): tahmin YANLIŞLANDI.** Tarama 2.32 / 2.55 / 1.27 — 2b/2d/2e ile aynı.
Sulu kar 5.1 cm tekdüze yükseliyor (etkin hareketlilik yüzmeyi düzeltti, kapı 8 cm 2b'ye göre).
Maliyet değişmedi (172 ms/kare).
**Çıkarım:** altı bağ formülasyonu (yalnız sürükleme; sürükleme projeksiyondan sonra / önce / ızgara
saatinde; gözenekli kapalı / açık; serbest / etkin hareketlilik) taramayı ±%10 içinde değiştirmedi.
Adım bağımlılığı bu terimlerde DEĞİL; hiç dokunmadığımız sıvı aktarım döngüsünde (sıvı adımları
arasında parçacıklar eski hızlarıyla ilerliyor, P2G/G2P/FLIP sıklığı). Hata ~dt orantılı
(1/384→1/768 yarıya), ~5 ms üstünde doyuyor. Saf su aynı belirtiyi göstermiyor.

