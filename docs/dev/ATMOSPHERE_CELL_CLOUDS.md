# Fırtına hücresi bulutları: tam sütun yapı ve yoğunluğa bağlı optik (Atmosfer Faz 6)

> **Durum:** TASLAK — 2026-10-12. Kullanıcı isteği: "basınç alanlarındaki
> bulutlar tam doğal değil; bu alanlar katman yüksekliklerini ve her katmanı
> kullanmalı, yoğunluğa göre scatter/absorption ayrılmalı". Karar bekliyor
> (§6). Yol haritası: `ATMOSPHERE_STORMS.md` §8.

## 0. Bugün neden doğal değil (kod okuması)

- **Hücre katmanı aşamıyor.** Hücre her katmana yalnızca "örtü tabanı" ve
  "tip tabanı" olarak yazılıyor (`cloud_common.glsl`, `cell.r`/`cell.g`).
  - Kule, bulunduğu katmanın tepesinde kesiliyor. 1.5–2 km'lik bir kümülüs
    katmanında Cb, 2 km'lik bir blok oluyor.
  - Gerçek bir Cb, LCL'den (~1 km) denge seviyesine, yani tropopoza
    (10–14 km) kadar tek bir sütun.
- **Katmanlar birbirinden habersiz.** Üst katmana (Ac/Ci) örs düşmüyor.
  Alt katmanda duvar bulutu (wall cloud), raf bulutu (shelf) ya da yırtık
  taban bulutu (scud) yok.
  - `validateClouds` örtüşen katmanları zaten reddediyor.
  - Bağlı fırtına bölgeleri (`formation_role` base/tower/anvil +
    `formation_group`) var, ama hücreleri değil kendi kafes noktalarını
    kullanıyor.
- **Optik katman başına sabit.** `extinction_per_m`, malzeme ve
  `droplet_diameter_um` (tek g) her yerde aynı.
  - Gerçekte kule tabanı su damlası, tepe ve örs buz kristali.
  - Yağış çekirdeğinde iri damla ve graupel var, yani daha karanlık ve daha
    ileri saçılım.
  - Damla boyu (etkin yarıçap), sıvı su miktarıyla (LWC) büyüyor.

## 1. Model: hücre kendi sütununu taşır ("fırtına kabuğu")

İki seçenek var:

| | A: katmanları hücreye göre bük | B: ayrı fırtına kabuğu (öneri) |
|---|---|---|
| Ne | Her katmanın tepe/taban/rolü hücre haritasından değişir | Hücrelerin olduğu yerde dolu, başka her yerde boş bir sanal katman: taban = LCL, tepe = denge seviyesi + aşma |
| Katman sınırları | Kabuklar sabit kalmalı (march sınırı); kule yine kabukta kesilir | Kabuk troposferi kapsar; boşluk atlama majorantı hücre haritasından (A kanalı) gelir, hücre dışında maliyet sıfır |
| Örtüşme | Kullanıcı katmanlarıyla çakışır | Hücre içinde kullanıcı katmanlarının yoğunluğu `1 − hücre maskesi` ile söner; çift yoğunluk olmaz |
| Örs, duvar bulutu, raf bulutu | Katman rolleriyle sınırlı | Sütunun kendi yükseklik profili, hepsi bir yerde |

**B'nin yükseklik profili** (hücre evresine göre, `h` = taban → tepe):

- **Taban (0–0.1):**
  - Yağışsız taban updraft altında, yağış çekirdeğinin dışında.
  - Duvar bulutu: updraft altında taban 100–300 m iner.
  - Scud: yağış perdesinde yırtık parçalar.
- **Kule (0.1–0.75):** Parti 11'in updraft kuleleri, artık tam yükseklikte.
  Girdap aynı.
- **Örs (0.75–1):**
  - Örtü yükseklikle genişler ve rüzgâr kesmesi yönünde uzanır.
  - Altında mamatus: aşağı sarkan keseler. Bu, örsün alt yüzünde ters
    yönde kabarma olarak yapılır.
- **Aşma tepesi:** En güçlü kule örsün 0.5–1 km üstüne taşar.
- **Raf / arcus bulutu:** Hamle cephesi yayı boyunca alçak, yatay bir
  silindir. Konum ve güç `ATMOSPHERE_LOCAL_WINDS_FOG.md`'deki çıkış
  rüzgârından gelir.

Kabuk, katman dizisinde 4. bir yuva olarak durur (`CloudLayerG` aynı
yapıda). Böylece march, gölge, yağış ayağı ve şimşek kaynağı ek kod yolu
olmadan çalışır. Bedeli: kabuk boyunca uzun boş march'lar. Majorant
segmentleri bunu atlamalı, ölçülecek.

## 2. Yoğunluğa ve faza bağlı optik

Bugün örnek başına sabit olan dört değer, örnek başına türetilir:

| Değer | Neyden | Görünüm |
|---|---|---|
| Faz (su/buz) | sıcaklık: donma seviyesi (`freezing_level_m`) ile −20 °C arası karışık, üstü buz | kule tepesi ve örs lifli, "camsı"; buz faz fonksiyonu (cirrus'unki) |
| Etkin damla boyu | LWC ∝ yoğunluk × yükseklik (adiabatik) | çekirdekte iri damla: kütle başına daha az sönüm, daha yüksek g (gümüş kenar azalır, çekirdek koyulaşır) |
| Saçılım / soğurma | tek saçılım albedosu: bulut ~0.9999, yağış çekirdeği ve graupel biraz düşük | yağmur perdesi ve çekirdeği gri-koyu; "kirli" taban seçeneği (toz, duman) |
| Sönüm ölçeği | buzda kütle başına düşük | örs yarı saydam kenarlı |

Malzeme sistemi değişmiyor: `CloudMaterial`, temel değerler ve sanatçı
çarpanı olarak kalıyor. Fiziksel modülasyon onun üstüne çarpan olarak
geliyor ve bir anahtarla kapatılabiliyor. Maliyet örnek başına birkaç
ALU işi. Ancak çoklu saçılım oktavları g'ye bağlı, bu yüzden g'nin örnek
başına değişmesi gölge march'ını da etkiler. Ölçülecek.

## 3. RayFusion paritesi

Aynı `cloud_common.glsl`, aynı kabuk. RayFusion interleave'i ve geçmişi
değişmez. Gölge haritası kabuğun gölgesini de alır: örs, yere geniş bir
gölge düşürür.

## 4. IPC (kural 1)

- `world.set_clouds {storms: {column: {enabled, overshoot_m, anvil_spread,
  wall_cloud, mammatus}, optics: {physical: bool}}}`
- `world.sample_clouds` her nokta için `phase` (water/ice/mixed),
  `effective_radius_um` ve `albedo` döndürür. Kabul testi bunlar üzerinden.

## 5. Kabul

- Bir olgun hücrenin yoğunluk profili tabandan tropopoza kesintisiz
  (`sample_clouds` dikey tarama). Örs genişliği kule genişliğinin ≥ 2 katı,
  rüzgâr kesmesi yönüne kaymış.
- Hücre içinde kullanıcı katmanının yoğunluğu ≈ 0 (çift sayım yok).
- Donma seviyesinin üstünde `phase = ice`.
- Hücre yokken kare süresi aynı (kabuk boş, march girmiyor).

## 6. Kararlar (kullanıcı)

1. A mı B mi? Öneri: **B** (fırtına kabuğu).
2. Fiziksel optik varsayılan açık mı? Öneri: açık, panelde kapatılabilir.
3. Mevcut `formation_role` ve `formation_group` (elle kurulan fırtına
   bölgeleri) kalsın mı? Kabuk aynı işi hücrelerle yapıyor. Öneri: kabuk
   canlı doğrulanınca sök (kural 5).
