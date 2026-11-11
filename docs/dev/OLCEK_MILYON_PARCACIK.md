# Ölçek hedefi: milyonlarca parçacık (sinematik madde simülasyonu)

> **Durum:** AKTİF — 2026-11-11: hedef + maliyet modeli + ölçülmüş darboğazlar. Bugün 100k
> parçacıkta çalışıyoruz; hedef 1M rahat, 10M çevrimdışı mümkün. Kod yazılmadı; adımlar sırayla.

## 0. Hedef (kabul sayıları)

Tek bir RTX sınıfı GPU'da, timestamps KAPALI, `fluid.step` başına (1/24 s):

| Senaryo | Hedef | Bugün (tahmin, §2) |
|---|---|---|
| 1M kar (MPM, ince sahne) | **≤ 1.5 s/kare** | ~8 s + CPU ~0.5 s |
| 10M kar | **≤ 20 s/kare** (çevrimdışı) | ~160 s + CPU ~5 s |
| 10M sıvı (FLIP) | **≤ 5 s/kare** | ölçülmedi |
| 1M karma (kar + su + kum) | karışım cezası ≤ %25 (§3) | kum+kar'da +%70 |
| Her senaryo | kare başına parçacık indirme **0 B** (önbellek yazımı hariç) | ~160 B/parçacık indir + yükle |

★ Ölçüt "kare süresi düştü" değil: **hangi kalemin N ile ölçeklendiği**. 100k'da gizli olan
doğrusal CPU kalemleri 10M'de baskın olur; her adım N=100k ve N=1M'de iki kez ölçülür.

## 1. Maliyet modeli (fizikten)

- **Elastik MPM (kar, jöle):** alt adım `dt ≤ C·Δx/c`, `c = √(E/ρ)`. Aynı sahnede çözünürlüğü
  artırmak (parçacık/hücre sabit) Δx'i küçültür → **maliyet ∝ N^(4/3)**. Daha büyük sahne aynı
  Δx'te ∝ N.
- **Sıvı (FLIP):** adım sayısı akış hızıyla (CFL), kare başına 4–10; basınç hücre sayısıyla.
  Milyonlarda ucuz taraf.
- **DEM:** alt adım tane temas sertliği/kütlesi ve sönümle (§3.2); tane boyutu küçüldükçe artar.
- Bugünkü ölçü: MPM ≈ **18 ns / parçacık·alt adım** (62k kar, 190 alt adım, P2G 158 ms +
  ~60 ms diğer). Parçacık başına ~243 float atomic: P2G atomik verim sınırında.

## 2. Bellek ve trafik tahmini

| | 1M | 10M | 50M |
|---|---|---|---|
| GPU parçacık (~160–200 B) | ~0.2 GB | ~2 GB | ~10 GB |
| Yoğun MPM ızgarası (ince sahne) | 155³ ≈ 0.6 GB | 334³ ≈ 6 GB | sığmaz |
| Seyrek ızgara (dolu + kenar) | ~0.1 GB | ~0.5–1 GB | ~3–5 GB |
| Bugünkü CPU aynası + kopya | ~0.5 GB RAM | ~5 GB RAM | ~25 GB RAM |
| PCIe indir + yükle (~320 B/parçacık) | 0.3 GB/kare | 3 GB/kare | — |
| Seri CPU yayınlama + hazırlık (bugün 51 ms @100k) | ~0.5 s | **~5 s** | — |

## 3. Ölçülenler (2026-11-11, boş sahne, geçici domain'ler)

Kutu 0.5×0.4×0.5 m, h = 2.5 cm, kum = varsayılan Sand (r = 25 mm), kar 8 ppc. Kare 1/24 s.

| Kol | Parçacık | Kare (ms, host) | GPU (ms) | En büyük çekirdek |
|---|---|---|---|---|
| kum | 60 tane | 55 | 69 | `grain_step` 65 (192 çağrı, **0.34 ms/çağrı**) |
| kum r=12.5 mm | 630 tane | 55 | 79 | `grain_step` 74 (366 çağrı, 0.20 ms/çağrı) |
| kum + su | 60 + 11.7k | 68 | 29 | `grain_step` 9 (192 çağrı, 0.05 ms/çağrı) |
| **kum + kar** | 60 + 3.9k | **133** | **158** | **`grain_mpm_gather` 57 + `grain_mpm_count` 54** |
| kar | 4.9k | 92 | 94 | `p2g_fused` 49 (600 çağrı) |
| su | 11.7k | 48 | 32 | CG + GPU yüzey |

### 3.1 Karışımın cezası: EVET, maddeye göre değişiyor

- **DEM + MPM, en pahalısı.** Ortak saatte DEM her sürekli ortam alt adımına zorlanıyor (kendi
  174'ü yerine 200). Asıl ceza ise **tane↔MPM temasının her DEM alt adımında sıfırdan
  aranması**: `sim_grain_mpm_{clear,hash,count,gather}`.
  - Bağlı liste hash'i ve 27 komşu hücre yürüyüşü, hücre boyu ≈ tane çapı (5 cm).
  - Bir tane iş parçacığı ~27 × 64 kar parseli yürüyor; MPM parselleri de taneleri arıyor.
  - **60 tane için kare GPU'sunun %70'i (110 ms).** Maliyeti tane değil parsel yoğunluğu
    belirliyor, ve en uzun zincirin gecikmesi.
- **DEM + sıvı:** DEM ucuzladı (aşağıya bak). Sıvı karma yola giriyor (seyrek MAC + sıvı adımı
  başına basınç senkronu, host pressure 24–58 ms), ve CPU tarafında kare başına
  `buildMatterGrainLiquidField` + sahip ayır/birleştir (`mergeMatterGrainOwners`) çalışıyor.
  Küçükte önemsiz (<3 ms), ama **N ile doğrusal CPU**.
- **MPM + sıvı:** sıvı kendi saatinde (CANLI, `MATTER_COKLU_ZAMAN_ADIMI.md`); kalan karışım
  maliyeti gözenekli bağ (yüz başına geçiş, ucuz).
- Kural: **iki maddenin bağı, ikisinden de pahalı olmamalı.** Bugün DEM+MPM'de ihlal ediliyor.

### 3.2 DEM neden pahalı (kendi başına)

1. **Alt adım sayısı (kum: 174–322 / kare; sınır = `damping`).** `cfl_terms`'ten:
   - k = 20 000 N/m, m = 0.175 kg, **24 temas varsayımı** → ω = √(24k/m) ≈ 1660 rad/s.
     √24 ≈ 4.9× bedel.
   - **Toplam sönüm oranı ζ = 1.49 (aşırı sönüm):** açık şemada kararlılık
     `dt ≤ (2/ω)(√(1+ζ²) − ζ)`. ζ=1.49'da çarpan 0.30, yani sönüm adımı **~3.3× kısaltıyor**.
   - Sonuç stabilite dt = 0.13 ms. Doğruluk dt'si (0.27 ms) ve yol dt'si (6 ms) daha gevşek:
     **sınırlayan fizik değil, açık şemanın sönüm kararlılığı.**
   - Tane yarıçapı yarıya inince alt adım 2× (174 → 348). Gerçek kum (≤ 1 mm) açık şemayla
     sinematik ölçekte olanaksız → Codex'in örtük temas planı (`DEM_CORE_INTEGRATION_ROADMAP.md`
     Faz 2–3) doğru yön.
2. **Küçük N'de gecikme sınırı:** 60 tanede alt adım başına 0.34 ms. Bu, aynı tanelerin kum+su
   kolunda ölçülen 0.05 ms'nin 7 katı. Aday: GPU yük azken düşük saat durumu ve/veya füzyonlu
   çekirdeğin seri iç döngüsü. **Ölçülmedi;** kaynak ayrımı sıradaki iş (aynı kolu arka planda
   yük varken koş).
3. **Bağlı buz (Codex ölçümü):** 59k tane / 336k bağ 971 ms (576–582 alt adım); tüm bağlar
   kırılınca 486 ms (262). Bağ sertliği alt adımı ikiye katlıyor.

### 3.3 S0 ölçek rig'i: 100k → 1M (2026-11-11, `scripts/test/rt_probe_scale_ipc.py`)

Aynı h = 2.5 cm, 8 ppc, 0.3 m yüksek blok; sahne N ile büyüyor, yani alt adım sayısı sabit
(kar 200/kare). RTX 3060 12 GB, timestamps kapalı (GPU ayrı kareler).

| Kol | N | Kare | GPU | Yayınlama + hazırlık (CPU) | Baskın GPU | VRAM artışı |
|---|---|---|---|---|---|---|
| kar | 100k | 245 ms | 308 | 50 + 12 | p2g_fused 211 | 63 MB |
| kar | **1M** | **2.10 s** | 1.89 s | **446 + 107** | p2g_fused 1395 (%74) | 567 MB |
| kar+su | 100k | 208 ms | 355 | 22 + 11 | p2g 143, sparse_mac_mark 73 | 51 MB |
| kar+su | **1M** | **1.52 s** | 1.71 s | **252 + 106** | p2g 720, **sparse_mac_mark 308** | 438 MB |
| su | 100k | 91 ms | 41 | — | GPU yüzey 17 | 37 MB |
| su | **1M** | **316 ms** | 145 | — | **GPU yüzey 88** (`surface_centres` 51) | 254 MB |

Eğimler, d log / d log N, 100k → 1M:
- kar: kare 0.93, GPU 0.79;
- kar+su: kare 0.87, GPU 0.68;
- yayınlama 0.95–1.06, hazırlık 0.95–0.99.

Okuma:
- **Tahminden iyi.** 1M kar 2.1 s; §0'da 3.5 s demiştim. Parsel·alt adım başına GPU maliyeti
  100k'da 15 ns, 1M'de **9.5 ns**: küçük N'de GPU doymamıştı.
- **CPU kenarı tahmin edildiği gibi doğrusal.** Yayınlama + hazırlık 1M karde 0.55 s, yani
  karenin %26'sı. S1 bunu kaldırır.
- **`sparse_mac_mark` karma yolda 19 ms/çağrı, kare başına 16 çağrı.** Kar+su GPU'sunun %18'i.
  Sıvı adımı başına bütün işaretlemeyi baştan yapıyor; S1'den bağımsız, ucuz bir hedef.
- **Saf suda GPU yüzeyi solver'dan büyük:** 88 / 145 ms. `surface_centres` tek çağrıda 51 ms.
- **Bellek** (ölçülen VRAM): kar ~595 B/parçacık (ızgara dahil), kar+su ~460, su ~255.
  10M kar ≈ 6 GB: 12 GB kartta sığar, ama yoğun ızgara payı S4'ü 10M'de gerekli kılıyor.

### 3.3b ★ DÜZELTME (aynı gün): "yayınlama + hazırlık" CPU DEĞİL, çoğu GPU beklemesi

Yayınlama ve hazırlık içine geçici alt zamanlayıcılar kondu (`SCALE-PROBE`, ölçümden sonra
söküldü). 1M kar, zamanlayıcı kapalı:

| Kalem | ms |
|---|---|
| indirme (`endTransferBatch` dahil) | **130–210**; zamanlayıcı açık karelerde 33–47 |
| NaN taraması / commit / uvw / durum sayımı | 3–11 / 0.2–7 / ~0 / 12–23 |
| hazırlık iç kalemleri (yük ölçümü, uyku, bölüm, ıslak, kuru hacim, yükleme) | toplam ~30 |

- Gönderim halkası asenkron. Zamanlayıcı kapalıyken indirme, halkadaki son GPU işinin
  bitmesini bekliyor; `step_stats` aşamaları host duvar saati olduğu için GPU zamanı
  "publication"a yazılıyor.
- **Gerçek CPU kenarı 1M'de ~60–100 ms**, yani karenin ~%5'i; %26 değil. Kare 1.98 s, GPU
  1.77 s: **1M'de darboğaz GPU** ve onun %74'ü `sim_matter_p2g_fused`.
- §2'deki "10M'de CPU ~5 s" tahmini bu yanlış okumadan geliyordu. Doğrusal ölçekle 10M'de
  ~0.6–1 s. S1 hâlâ bellek/PCIe için gerekli (10M'de ~1.7 GB/kare indirme), ama süre
  önceliği değil.
- Kural: host aşamasını CPU sanmadan önce zamanlayıcı AÇIK/KAPALI farkına bak. Fark büyükse
  ölçtüğün şey GPU beklemesidir.

Dilim A yine de kaldı (doğru ve ucuz): tam kopya yerine kalıcı ara tampon + swap, paralel NaN
taraması, tüm-MPM'de sütun swap'ı, tag başına madde araması, ıslak tepki kapalıyken tek
yükleme. 1M kar: yayınlama 446 → 396, hazırlık 107 → 92 ms. Smoke 30/30, multirate 7/7.

### 3.3c P2G'nin kümelenmiş indirgemesi neden nadiren tetikleniyor

Kümelenmiş yol, 8 şerit bileşen başına **aynı taban düğümünü** paylaştığında devreye girer.
- MAC ızgarası kaydırmalı: bileşen ekseninde taban `floor(x − 0.5)`, diğer iki eksende
  `floor(y − 1)`. Bir hücredeki parçacıklar alt hücre konumuna göre farklı tabanlara düşüyor.
- 8 ppc tohumlama her sekizde bire bir parçacık koyuyor. Sonuç: bir hücrenin 8 parçacığı
  çoğunlukla 8 farklı tabana sahip ve yol global atomiklere düşüyor (parçacık başına ~243).
- Çözüm **S5 = paylaşımlı bellek blok P2G** (Gao 2018): GPU'da kare başına blok sıralaması
  (permütasyon, kanonik sıra değişmez), iş grubu başına 4³ hücre + kenar payı karo, karo
  başına bir global yazım.

### 3.3d S5 ilk dilim CANLI: paylaşımlı bellek blok P2G (2026-11-11, ajan derledi)

- `sim_matter_p2g_block_sort.comp` (5 bağlama / 64 B) sıralar: sayma, tarama ve dağıtım ile
  4³ hücrelik bloklara permütasyon. Her alt adımda 4 dispatch. Kanonik sıra değişmez.
- `sim_matter_p2g_fused.glsl` `#ifdef P2G_BLOCK` (21 bağlama / 64 B): iş grubu başına bir
  blok, 9³ düğümlük karo × 3 bileşen × 3 alan (26 KB paylaşımlı bellek). Karo dışına düşen
  katkı global atomiğe gider.
- Temizleme ve normalizasyon füzyonlu geçişlerde kaldı.
- Cihazda `shaderSharedFloat32AtomicAdd` açılıyor (`VulkanBackend.cpp`). Desteklemeyen cihazda
  çekirdek ilk dispatch'te reddedilir ve füzyonlu saçılım kullanılır. `RT_P2G_BLOCK_OFF`
  ortam değişkeni aynı geri dönüşü A/B için zorlar.

**Teşhis (1M kar, saçılım):**

| Varyant | ms/kare |
|---|---|
| Füzyonlu global atomik (önce) | ~1350 |
| Blok + paylaşımlı atomik | 1258 |
| **Blok + şerit başına döndürülmüş şablon sırası** | **~1000** |
| Blok, atomiksiz (YANLIŞ sonuç, yalnız taban ölçümü) | 396 |

Paylaşımlı atomikler ~56 G/s, yani donanım kapasitesinin çok altında. Darboğaz sayı değil,
**aynı adrese çakışma**: tohum sırası uzayda tutarlı olduğu için komşu şeritler aynı tabanı
paylaşıyor. Döndürme bir kısmını açtı; kalan ~600 ms warp'lar arası çakışma.

**Sonuç (GPU, zamanlayıcı açık, aynı sahneler):**
- 1M kar 1797 → **1523 ms**; kare 2.05 → **1.76 s**.
- 1M kar+su 1319 → **1095 ms**; kare 1.38 → 1.33 s.
- 100k kar 194 → 183 ms.

**Doğrulama:**
- smoke MPM 30/30, multirate 7/7.
- **A/B paritesi** (multirate testi, `RT_P2G_BLOCK_OFF` ile): kar kütle merkezi 0.1 mm, p05/p95
  ≤ 0.3 mm, en yüksek hız birebir. Momentum farkı float atomik gürültüsü düzeyinde.

**Hücre kümeli P2G DENENDİ ve SÖKÜLDÜ (2026-11-11, olumsuz sonuç):**

- **Fikir:** MAC ızgarasında bir hücrenin bütün parçacıkları, bir bileşen için aynı 4×3×3 = 36
  düğümlük kutuya düşer (bileşen ekseninde taban `cell−1` ya da `cell`, diğer iki eksende hep
  `cell−1`).
- **Uygulama:** blok çekirdeği içinde paylaşımlı bellekte yerel hücre sıralaması. 8 şeritlik
  küme aynı hücreden 8 parçacık alır; her şerit 36 düğümden ~4.5'ine sahip olur ve 8 parçacığı
  `subgroupShuffle` ile gezer. Atomik sayısı 216 → 36 (6×), hesap ~1.3×.
- **Ölçüm:**

  | Kol | Önce | Küme yolu |
  |---|---|---|
  | 100k kar saçılım | 118 ms | **248 ms** |
  | 1M kar saçılım | ~1000 ms | **1589 ms** |
  | 1M, 256'lık parça (paylaşımlı bellek 39 → 30 KB) | ~1000 ms | **1994 ms** |

  Doluluk değil: belleği küçültmek kötüleştirdi.
- **Aday kökler** (ölçülmedi): bileşen başına 8 × 13 shuffle, dinamik indeksli
  `wx[i]`/`acc[o]` dizilerinin yerel belleğe taşması, parça başına 4 bariyer ve tek iş
  parçacığıyla tarama.
- **Ders:** atomik sayısını azaltmak, ek hesap ve shuffle yükünü karşılamadı. Atomiksiz taban
  (396 ms) da tek başına atomik maliyetini ölçmüyordu: o deneme aynı adrese yazımın çakışma
  serileşmesini de kaldırıyordu.
- Kod, shader ve cihaz kapısı (`supportsClusterShuffle`) silindi. Statik kontrol yokluğunu
  şart koşuyor.

### 3.3f Ölçüm oynaması ÇÖZÜLDÜ + P2G permütasyon denemesi + G2P/stres değerlendirmesi (2026-11-11)

**Kök:**
- IPC isteği ana iş parçacığı kuyruğuna giriyordu (`rtapi::enqueue`) ama ana döngüyü kimse
  uyandırmıyordu.
- Pencere odakta değilken döngü `SDL_WaitEventTimeout(…, 500 ms)` ile uyuyor
  (`Viewport::backgroundWaitMilliseconds`). Yani her IPC çağrısı bu uykuyu bekliyordu.
- Uygulamayı ajan açınca pencere çoğu zaman arka planda kalıyor: testler 10–16× yavaş, GPU
  istekler arasında saat düşürüyor, GPU süreleri de ±%20 oynuyor.

**Düzeltme:** `enqueue` artık `SDL_PushEvent` ile döngüyü uyandırıyor (`RtApi.cpp`).

**Doğrulama, iki ayrı açılış:**

| | Açılış 1 | Açılış 2 |
|---|---|---|
| smoke elastic | 4 s | 5 s (önce 7 ↔ 166 s) |
| 1M P2G | 1101 ms | 1107 ms (fark < %1) |

Bundan sonra açılışlar arası A/B güvenilir.

**P2G asal adımlı permütasyon** (blok içinde `k = m·P mod n`, şeritleri bloğun farklı
yerlerine dağıtmak için): aynı koşulda yalnız döndürme 1011 ms, permütasyon 1101–1107 ms.
**%9 YAVAŞ, söküldü.** Global parçacık okumasını dağıtması, paylaşımlı adres çakışmasında
kazandırdığından pahalı.

**G2P / stres / parçacık geçişleri** (1M kar, ms/kare): G2P 171, stres 116, adveksiyon 35,
oturtma 34, kuvvet 23, parsel teması 19, doluluk 17, yani toplam ~415.
- Basit dördü (adveksiyon, oturtma, kuvvet, temas) birlikte ~111 ms. Bu, tahmini bellek trafiği
  süresine denk: **bant genişliği sınırlı**.
- Birleştirme kazancı ~70–90 ms (~%5). Kuvvet entegrasyonu alt adım başında, adveksiyon ve
  temas sonunda; aynı geçişe almak sıra semantiğine dokunuyor.
- Stres güncellemesi ızgarayı okumuyor (yalnız afin C), yani G2P ile birleşmesi yalnız C'nin
  bir yaz/oku turunu kurtarır (~10–20 ms).
- **Karar:** bu partide yazılmadı; sıradaki "birleşik parçacık sonu geçişi" adayı olarak
  duruyor.

**Son durum (1M kar, GPU):** 1653 ms/kare. Payı: P2G 1079 + sıralama 57 (%69), G2P + stres 287,
geri kalan ~230.

Doğrulama: smoke MPM 30/30, multirate 7/7, yüzey 8/8.

### 3.3g Alt adım sayısı ve parçacık geçişi birleştirmesi: ANALİZ, kod yazılmadı (2026-11-11)

**Kar neden 200 alt adım?** Kod: `snowSizingYoungModulus` + `elasticStepInfo`.
- Boyutlandırma sertleşme TAVANINA göre: E = 1.4·10⁵ × `snow_max_hardening` 5 = 7·10⁵ Pa,
  ρ = 400 kg/m³.
- c = √(E/ρ) ≈ 42 m/s; CFL sınırı 0.35·h/Δt_kare = 0.35·0.025·24 = 0.21 m/s.
- c / 0.21 ≈ **199 → 200**. Tavan çarpanı tek başına √5 = 2.24×: başlangıç sertliğinde
  ~89 adım.

**Neden ucuz bir düzeltme yok:**
1. *Öngörülü sınır* (bir sonraki karede ulaşılabilecek en büyük sertleşme): sertleşme
   h = exp(ξ(1 − J_p)), kar için ξ = 10.
   - Dinlenik kar yerçekimiyle bir karede ~0.4 m/s hızlanabilir. Sıkışma hızı sınırı
     ~2v/h = 33 s⁻¹, yani Δln J_p ≤ 1.4.
   - Bu J_p tabanına (400/917) iner ve h ≫ 5 eder. Güvenli sınır tek karede tavana doyuyor:
     **kazanç sıfır.**
2. *Başlangıç sertliğiyle koş, kelepçe olursa kareyi tavanla yeniden koş* (host durumu yayına
   kadar dokunulmadığı için geri alma ucuz).
   - Doğru, ama boyutlandırma EN sert parsele baktığı için yalnız **hiç sıkışmamış** karda
     kazandırır.
   - İlk temas ya da çarpmadan sonra bir parsel bile tavandaysa bütün domain 200 adıma
     döner. Gerçek sahnelerde kazanç kısa süreli.
3. CFL 0.35'i büyütmek: kodda belgelenmiş gerekçe var (Dough 0.35'te salınıyordu). Ölçmeden
   dokunulmaz.

**Asıl kaldıraç: BÖLGESEL zaman adımı (S6, TASLAK).**
- Kaynak: Fang ve ark., "A Temporally Adaptive Material Point Method with Regional Time
  Stepping", Computer Graphics Forum (SCA 2018).
- Fikir: blok başına seviye ℓ (2'nin kuvveti). Bloğun adım sayısı en sert ve en hızlı parsele
  göre seçilir; gevşek ve sert bölge aynı karede farklı adım atar. Bölge sınırında düğüm
  zamanları eşitlenir.
- Bizim altyapıda blok sıralaması (S5) ve 4³ bloklar zaten var.
- Beklenen kazanç, sertleşmiş parsel oranına bağlı: yalnız çarpma bölgesi sertse kalan hacim
  √5 kat daha az adım atar. Kar sahnesinin %20'si sertse ~%60–70 tasarruf.
- Büyük iş: ızgara düğümü zaman tutarlılığı, P2G/G2P'nin seviye başına koşması, sıvı ve DEM
  bağının alt adım eşlemesi. Ayrı bir tasarım notu gerektirir.

**Parçacık geçişi birleştirmesi: değer düşük, risk ortak kodda.**

| Aday | Kazanç | Engel |
|---|---|---|
| G2P → stres | ~40 ms (%2.4), afin C'nin yaz/oku turu (72 B × 1M × 200) | — |
| Oturtma + stres | yalnız dispatch yükü | farklı verilere dokunuyor |
| Adveksiyon + temas + kuvvet | bant genişliğinde ~70 ms | adveksiyon kuyruğu saf sıvıyla ORTAK GLSL (15 bağlama); kuvvet entegrasyonu alt adım başında, diğerleri sonunda (sıra semantiği) |

Toplam ~%3–5. Karar: şimdi yazılmadı. S1'de (cihazda kalıcı durum) parçacık sonu geçişleri
zaten yeniden düzenlenecek; orada birlikte yapılır.

### 3.3h Ölçülen sertleşmeye göre kar boyutlandırması + tavanla yeniden koşu: CANLI (2026-11-11)

Ölçüm (en küçük / ortalama J_p): dinlenik kar 1.0 / 1.0, su karın üstüne 0.92 / 1.00,
30 cm düşüş 0.54 / 0.92. Yani çoğu kar karesi tavandan (×5) çok uzakta.

**Kod** (`MatterGpuStep.inl`):
- Kar, kare başında ÖLÇÜLEN en sert MPM parselinin sertleşmesi × 1.5 ile boyutlanıyor
  (tavanla sınırlı).
- Kare sonunda en büyük sertleşme bu değeri aşarsa ya da bir parsel dalga kelepçesine
  takılırsa kare atılır ve tavanla yeniden koşulur. Host durumu yayına kadar dokunulmadığı
  için yeniden koşu aynı durumdan başlar.
- Yeniden koşu sarmalayıcıda (`runMatterGpuStep` → `runMatterGpuStepOnce`). ★ İçeriden özyineleme
  denendi: ilk koşunun parçacık kiralaması kapsamda kaldığı için ikinci koşu "canonical particle
  upload failed" ile her karede takıldı.
- DEM ortak saatiyle kapalı: DEM kare içinde kendi durumunu ilerletiyor, yeniden koşulamaz.

**Sonuç:**

| Kol | Alt adım |
|---|---|
| Dinlenik kar | 200 → 110 |
| Su karın üstüne | 200 → 155–162 |
| Düşüş | 110; çarpma karesinden itibaren 200 |

- Düşüşte J_p 0.544 / 0.9195; tavanla koşu 0.543 / 0.9194, yani çarpma birebir korundu.
- Ölçek rig'i: **1M kar 1.76 → 1.22 s/kare** (hedef ≤ 1.5 ✔), 1M kar+su 1.33 → 0.94 s,
  100k kar 143 ms.
- Doğrulama: smoke MPM 30/30, multirate 7/7.
- Statik kontroller: eskimiş beklentiler güncellendi (gpu, pore, solid_weight). Bulut, gaz,
  hacim, DEM uyku aktarımı ve panel alanları kontrolleri bu partiden önce de FAIL'di
  (dosyalar git'te değişmemiş); dokunulmadı.

**Ölçüm notu (TARİHSEL, kökü yukarıda): oturumlar arası oynama ±%15–20.** Aynı kod ve aynı sahnede 1M saçılım 998 ms ve
1171 ms ölçüldü. Bazı uygulama açılışlarında IPC testleri 10–16 kat yavaş koşuyor (smoke
elastic 7 s ↔ 166 s). Aday: arka plan penceresinde kare döngüsü ve GPU güç durumu, ya da
Windows'un arka plan süreç kısması. Bu yüzden A/B'yi aynı oturumda kıyasla ya da farkı bu
aralığın çok üstünde arayın; küme yolunun reddi (100k'da 2×) bu aralığın üstünde.

### 3.3e Aynı turda

- `sim_sparse_mac_mark`: karo kutusu + CAS öncesi okuma ile 81 → ~3 işlem. 1M kar+su'da
  ilk 5'ten çıktı.
- `sim_surface_centres`: şeritler parçacıkları kutu sırasıyla işler (`items[lane]`).
  `rt_test_surface_gpu_ipc.py` paritesi: φ en çok 2.4e-5 h, işaret/bant uyuşmazlığı 0.
  Testteki `0.0 or 1.0` hatası düzeltildi; 8/8 beklenir.
- Yüzey shader'larının 11'i `compile_sim_shaders.bat` listesinde yoktu, eklendi. Dağıtılmış
  `.spv`'ler kaynakla aynıydı.

### 3.4 Bulunan sınır: domain bütçesi 1M'i sessizce bekletiyordu

İlk koşuda 1M kar ve kar+su **hiç adım atmadı**: `gpu_status = "Mixed step held: mixed GPU
working set exceeds domain resource budget"`. Rapor ve kare süresi makul görünüyordu (sessiz
başarısızlık). İki kök:
1. `matterGpuWorkingSetBytes` parçacık başına **2048 B** sayıyordu; ölçülen ~336 B. Tahmin 4×
   fazlaydı: 1M kar için 2.4 GB diyordu, gerçek 0.57 GB. → **384 B** yapıldı (kalibre, ~%15 pay).
   Gözenek alışverişi terimi (2048 B/parçacık) kalibre EDİLMEDİ.
2. Varsayılan bütçe 1024 MiB → **4096 MiB** (Preview ve yeni domain'ler); Interactive
   512 → 2048. Proje yükleyicisinin üst sınırı 32768 iken panel 65536'ya izin veriyordu → ikisi
   de 65536. Kaydedilmiş projeler kendi değerini korur.
   - Açık: Final profili bütçeyi **sistem RAM'inden** türetiyor, oysa bütçe GPU belleği. VRAM
     sorgusu backend'e eklenince düzeltilmeli.

## 4. GPU yüzey (SDF) durumu

- **GPU'da kuruluyor** (Y1+Y2, `GPU_YUZEY_LEVEL_SET.md`; su kolunda `sim_surface_*`
  çekirdekleri koştu).
  - Rota 2 (Vulkan RT canlı) hiçbir şey indirmiyor.
  - Raster/RayFusion ve OptiX ızgarayı indiriyor (~3 MB; ikinci VkDevice, TDR yalıtımı).
- **Ama CPU indirmesini kaldırmıyor:** girdiler adımın cihaz tamponlarından değil **host
  parçacıklarından yükleniyor** (~28 B/parçacık). Gerekçe: adımdan sonra host parçacıkları
  değiştiriyor (açık sınırda silme, gözenek alışverişi, etiketler, sis sürüklenmesi).
- Hâlâ CPU'ya düşen durumlar: anizotropik kernel, volume köpük.
- Yani yüzey, **B adımıyla (§5) birlikte** çözülür: o host düzenlemeleri cihaza taşınınca
  yüzey cihaz tamponunu doğrudan okur.

## 5. Yol (sıralı; her adım N=100k ve N=1M'de ölçülür)

| # | İş | Neden önce | Sahip | Beklenen |
|---|---|---|---|---|
| S0 | **Ölçek ölçüm rig'i:** 100k / 1M kar, su, kum+kar; `step_stats` + host aşamaları + GPU; N'e göre eğim | hangi kalem doğrusal CPU, hangisi GPU | Claude | sayı |
| S1 | **Cihazda kalıcı parçacık durumu (B).** Kare sonu host düzenlemeleri (açık sınır silme, gözenek, etiket, sis, emitter ekleme) GPU'ya; host yalnız istenen alanı ve önbellek yazarken indirir. Yüzey cihaz tamponunu okur. | 1M üstünde CPU kalemi GPU'dan büyük; yüzeyin önkoşulu | Claude | 1M'de −0.5 s, 10M'de −5 s |
| S2 | **Tane↔MPM teması:** her alt adımda hash yeniden kurma yerine aday listesi (Verlet payı, DEM'deki gibi) + sıralı hücre (CSR); MPM tarafı yalnız tane yakınındaki parseller. Alternatif: tane→MPM **ızgara** bağı (parçacık–parçacık değil). | ölçülen en büyük karışım cezası (%70) | Codex ile ortak | kum+kar 158 → ~60 ms GPU |
| S3 | **DEM örtük temas** (Codex Faz 2–3). §3.2'deki 24 temas × aşırı sönüm cezası açık şemaya özgü. | DEM'in alt adım sayısı | Codex | alt adım 5–10× az |
| S4 | **MPM seyrek ızgara** (sıvıdaki seyrek MAC gibi) | 10M'de yoğun ızgara 6 GB | Claude | bellek ~10× az |
| S5 | **Parçacık sıralama + paylaşımlı bellek P2G (C, Gao 2018).** 100k'da gereksizdi (tohum sırası tutarlı), milyonlarda karışma artınca değil. | atomik verim | Claude | P2G ~3× |
| S6 | **Alt adım sayısını kırmak:** madde içi çok-hızlı adım (uyuyan/yavaş bölge) ve sanatçı E kontrolü | N^(4/3)'teki 1/3 | sonra | — |

Notlar:
- S1 ve S2 bağımsız; S2 Codex'in DEM dosyalarına değdiği için `AJAN_KOORDINASYON.md` §4'te
  konuşulur.
- Örtük MPM CFL'yi kaldırır ama adım başına pahalı: şimdilik yok.
- Gerçek zamanlı tavan birkaç yüz bin MPM parçacığı olarak kalacak; hedef çevrimdışı.

### 5.1 S1 haritası: kare başına CPU'daki parçacığa dokunan işler

Host parçacıklarını okuyan/yazan **59 dosya** var. S1 için üç sınıf:

| Sınıf | Örnekler | S1'de ne olur |
|---|---|---|
| **A. Her karede adım içinde CPU döngüsü (fizik)** | `MatterGpuStep` yayınlama (16 MB indirme, tam kopya, sidecar commit, NaN taraması, `countMaterialState`, `advanceMaterialCoordinates`), açık sınır silme, `exchangeMatterPoresGpu` sonrası host düzenleme; `FluidDomainStep`: `refreshMatterConstitutiveModels`, `ensureMatterGrainRestMasses`, termal (`coolThermalLiquid`, `conductThermalLiquid`, `applyThermalConvection`, `exchangeMatterGrainHeat`), `recoverParticlesFromSolidCells`, `updateMatterMpmSoftening`, yoğunluk projeksiyonu, köpük, `splatFluidDensityCPU`; DEM: `buildMatterGrainLiquidField`, `mergeMatterGrainOwners` | GPU çekirdeğine taşınır ya da cihazda kalan tampona yazar; NaN/sayaçlar GPU indirgemesi (birkaç bayt) |
| **B. Kare sonu tüketiciler** | önbellek yakalama (`SimCache`, `SimFrameCacheMemory`), yüzey girdisi (`FluidSurfaceGpu`), splat/render köprüsü (`ParticleRenderBridge`, `SourceSplatRenderPlan`), whitewater | cihaz tamponunu okur; önbellek yazarken **istenen alanları** asenkron indirir |
| **C. İsteğe bağlı okuyucular** | IPC/RtApi (`RtApiFluid`, kabul metrikleri), panel/gizmo, ses telemetrisi | sürüm damgalı tembel indirme: okunduğunda indirilir, kare başına değil |

Sıra önerisi: önce **B + C**, yani indirmeyi tembel yapmak. Ama A'daki her CPU döngüsü host
kopyaya bağlı olduğu için indirme, A sınıfının taşınan ilk alt kümesiyle birlikte kalkar. İlk
dilim yalnız kar/su (termal kapalı) yolunu kapsar; termal ve DEM kendi dilimleri.

## 6. Açık sorular

- 60 tanede 0.34 ms/alt adım: saat durumu mu, çekirdek mi (§3.2-2)?
- Karma sıvı yolu (kum+su) saf sıvıdan neden pahalı: seyrek MAC işaretleme kare başına 5 ms
  (7 çağrı)? S0'da ayrıştırılacak.
