# Yerel rüzgârlar, zemin koşulları ve yerel sis (Atmosfer Faz 5b + 7)

> **Durum:** AKTİF — 2026-10-13. §2–4 parti 17'de yazıldı, derlenmedi
> (§7). §1 (hücre rüzgârları) kullanıcı kararıyla atlandı, sonra yapılacak.
> Kullanıcı isteği: "yağış durumu, hava uygunluğu ve zemin kontrolü ile
> sislenme; lokal sis; bulutları yükselten, merkeze çeken ve çevreye kuvvet
> uygulayan lokal rüzgârlar". Bu not `ATMOSPHERE_STORMS.md` §4'teki (5b)
> taslağın yerini alır.

## 0. İlke: her şey hücrelerin saf fonksiyonu

Hücreler `(seed, t)`'nin saf fonksiyonu. Yerel rüzgâr, "son yağış" ve sis de
**hücrelerden ve sabit girdilerden** (iklim, güneş, arazi yüksekliği)
analitik olarak türetilir. Sonuçları:

- Timeline'da ileri geri gidince aynı rüzgâr ve aynı sis çıkar. Adım adım
  toplama (entegrasyon) yok, ATMOSPHERE_WEATHER §8-4'teki önbellek sorunu
  doğmaz.
- **Geri besleme döngüsü yok.** "Zemin ıslak mı" bilgisi fizik sisteminden
  OKUNMAZ; hücrelerin geçmiş yağışından hesaplanır (§3). İklim → fizik tek
  yön kuralı (ATMOSPHERE_SYSTEM §3) korunur. Arazi yüksekliği bir sınır
  koşuludur, simülasyon durumu değildir; okunabilir.

## 1. Hücre rüzgâr modeli (5b)

Her hücre evresine göre dört bileşen üretir. Hepsi yarıçap ve yükseklik
cinsinden ifade edilir.

| Bileşen | Evre | Nerede | Yön ve güç |
|---|---|---|---|
| **Giriş (inflow)** | büyüme, olgun | taban altı, 0.5–2.5 r | merkeze doğru yatay yakınsama, 5–15 m/s; olgunda giriş kanadında (rüzgâra karşı ve sağ arka) yoğun |
| **Updraft** | büyüme, olgun | kule çekirdekleri (`flow.w`) | yukarı, 10–30 m/s × şiddet; girdap teğet bileşeni (`rotation`) |
| **Downdraft + soğuk havuz** | olgun, dağılma | yağış çekirdeği altı | aşağı; yerde ışınsal dışa yayılma 10–20 m/s; +1–3 hPa mezo-yüksek |
| **Hamle cephesi** | olgun, dağılma | `R(t) = r + c·(yaş − t_olgun)` yayı, c ≈ 8–12 m/s | dar bant, en güçlü dışa rüzgâr; ön kenarda yükselme |

**Tüketiciler (tek tanım, üç yol):**

1. **Bulut şekli:** Parti 11'deki girdap gibi bir SHAPE kayması.
   - Giriş, taban bulutlarını merkeze çeker.
   - Updraft kuleleri yükseltir (zaten var).
   - Hamle cephesinin ön kenarındaki yükselme, raf / arcus bulutunu besler
     (`ATMOSPHERE_CELL_CLOUDS.md` §1).
   - Örtü ve majorant etkilenmez.
2. **Yağış:** Perdenin eğimi yerel rüzgârla değişir. Downdraft, perdeyi
   yere vurduğunda dışa kıvırır ("yağmur ayağı").
3. **Fizik:** `atmosphere::sampleWind(p, t)` = iklim rüzgârı + hücre
   rüzgârları. ATMOSPHERE_SYSTEM §3.2'deki tüketicilerin hepsi bunu okur:
   - gaz, parçacık, foliage, APIC sıvı, okyanus, near-field yağış
     parçacıkları
   - sahnedeki `ForceField`'ler yerel ek olarak kalır
   - IPC: `world.sample_wind {position}`, `components` ayrıntısıyla

   Hücre başına sahneye `ForceField` varlığı EKLENMEZ. Sahne listesini
   kirletir, panelden silinebilir ve iki otorite doğar. Bunun yerine taban
   alanın kendisi büyür.

**GPU:** Hücre haritasına üçüncü bir yarı eklenir (256×768): yerde yatay
rüzgâr (xz), dikey hız, hamle cephesi gücü. Dikey profil analitik:
inflow tabanın altında, outflow yerde, updraft kabukta.

## 2. Hava uygunluğu (nerede ne olabilir)

Bugün otomatik hücreler alan içinde düzgün dağılımla doğuyor. Gerçekte
doğum yeri arazinin ve havanın işi. Doğum olasılığı bir **uygunluk
haritasıyla** ağırlıklandırılır (256², statik + yavaş değişen):

| Etken | Kaynak | Etki |
|---|---|---|
| Orografik kaldırma | arazi eğimi · rüzgâr (rüzgâra bakan yamaç) | doğum ↑, taban ↓ |
| Rüzgâr altı | arazinin gölgesi | doğum ↓, föhn: örtü ↓ |
| Yüzey ısınması | güneş açısı · eğim yönü (gün içi döngü) | öğleden sonra doğum ↑ |
| Hamle cephesi | §1 | yeni hücreler yayın üstünde (çok hücreli hat) |
| Su / kara | arazi su seviyesi | gece denizde, gündüz karada |

Arazi yoksa harita 1 olur, yani bugünkü davranış.

## 3. Son yağış (zemin ıslaklığı, fizikten okumadan)

`recentRain(xz, t) = Σ` hücre yağışı, son ~60 dk boyunca 8 zaman örneğiyle
(hücreler saf fonksiyon olduğu için geçmiş de hesaplanabilir), üstel
sönümle. GPU'da `cloud_cells.comp` her texel için aynı hücre döngüsünü 8
kez koşar; 256² için önemsiz.

Kullanıldığı yerler:
- sis (§4)
- yüzey ıslaklığının ATMOSPHERE TARAFINDAKİ tahmini
- `ATMOSPHERE_SURFACE_COUPLING.md`'deki yer yağış haritası

## 4. Yerel sis

Bugünkü sis tek bir yükseklik sisi (`world.set_aerial`). Yerel sis aynı
froxel geçişine, ikinci bir 2D yoğunluk haritası ve yükseklik sönümü olarak
eklenir. RT ve RayFusion aynı froxel'i okuduğu için parite kendiliğinden
gelir.

| Tür | Koşul (deterministik girdiler) | Nerede |
|---|---|---|
| Radyasyon sisi | gece, güneş ufkun altında · rüzgâr < 3 m/s · RH > 0.85 · açık gök (hücre örtüsü düşük) · son yağış ↑ | arazinin alçak yerleri: vadi tabanında birikir (yerel ortalamaya göre yükseklik) |
| Yağmur sonrası pus | son yağış yüksek, yağış bitti, RH yüksek | yakın zamanda yağmur alan şerit |
| Yağış altı pus ve scud | yağış çekirdeği, downdraft soğuması | perde içinde, tabana kadar |
| Hamle cephesi tozu | outflow > 12 m/s · kuru zemin (son yağış düşük) | yayın önünde toz duvarı (4b tozu) |
| Adveksiyon sisi (sonra) | sıcak nemli hava, soğuk yüzey (su, kar) | kıyı, göl |

Sis rüzgârla sürüklenir (yerel rüzgâr §1) ve gündoğumunda (güneş açısı)
yükselip dağılır.

## 5. IPC (kural 1)

- `world.sample_wind {position, time?}` → toplam rüzgâr ve
  `components {climate, inflow, updraft, outflow, gust_front}`
- `world.sample_weather_local {position}` → uygunluk, son yağış, sis
  yoğunluğu ve türü
- Hepsi `world.cloud_stats.storms` ile aynı zaman tabanında.

## 6. Kararlar (kullanıcı)

1. Rüzgâr fizik tüketicilerine taban alan olarak mı gitsin (öneri), yoksa
   hücre başına `ForceField` varlıkları olarak mı?
2. Arazi yüksekliği uygunluk ve sis için okunabilsin mi? (Sınır koşulu.)
   Öneri: evet, terrain heightfield'den 256² düşük çözünürlüklü bir kopya.
3. Sırayla mı, birlikte mi? Öneri: önce rüzgâr (§1, bulut ve fizik ikisi de
   bekliyor), sonra uygunluk ve son yağış (§2–3), en son sis (§4).

**Verilen kararlar (2026-10-12/13):**
- Sıra: kullanıcı rüzgârı (§1) atladı; önce §2–4.
- Karar 2: evet. Arazi 128² düşük çözünürlüklü kopya olarak okunur; sınır
  koşulu sayılır.
- Karar 1 (rüzgâr → fizik) §1 ile birlikte bekliyor.

## 7. Uygulama (parti 17, 2026-10-13, derlenmedi)

Tek dosya: `Atmosphere/AtmosphereLocalWeather.{h,cpp}`. Ana döngü her kare
`syncLocalWeather()` çağırır (`Main.cpp`, bulut zamanı bloğu).

**Arazi kopyası (`TerrainField`).**
- Bütün arazilerin birleşimini kaplayan bir kare, 128².
- Her texelde iki değer: sahne y'si ve ~1.5 km kutu ortalaması (çukur =
  ortalamanın altı).
- Ucuz parmak iziyle (boyut, dönüşüm, 257 seyrek örnek) yalnız arazi
  değişince yeniden kurulur.
- Bir terrain graph worker'da yükseklik hesaplarken (`defer_mesh_updates`)
  dokunulmaz.

**§2 Uygunluk (`birthSuitability`).**
- `0.3 + 0.7·orografik + 0.25·sırt + 0.45·ısınma − 0.25·rüzgâraltı`,
  0.05..1 arasında.
- Orografik: rüzgârın yamaca bindiği bileşen, 8 m/s'de tam.
- Isınma: güneşe bakan yamacın düz zeminden fazlası, yalnız gündüz.
- `evaluateStormCells` otomatik doğumda en fazla 6 aday dener (ayrı bir
  rastgele akış). Hiçbiri kabul edilmezse en iyisini alır; hücre sayısı
  değişmez.
- Arazisiz sahnede bu kol atlanır, hücreler eskisiyle aynıdır.
- `local_weather.terrain_suitability` ile kapatılır.

**§3 Son yağış (`recentRainAt`).**
- Hücreler 8 geçmiş anda değerlendirilir (450 s aralık, 60 dk), üstel
  ağırlıkla (τ = 30 dk).
- `cellRainAt` = `cloud_cells.comp` B kanalının CPU aynası.
- ★ Yalnız hücreler. GPU hava haritasının stratiform yağışı CPU'da
  bilinmiyor; o yüzden "son yağış" hücresiz stratiform yağmuru görmez.

**§4 Yerel sis — bulut alanının bir katmanı (2026-10-13 yeniden yazıldı).**

İlk sürüm sisi aerial froxel'e koymuştu. Kullanıcı canlıda "yüzeyi yamalayan
tek düze bir örtü, atmosferle etkileşmiyor" dedi. Sebep yapısaldı:
- ekran 32×32 hücreye bölünüyordu;
- sisin şekli ve bulut gölgesi yoktu;
- sis yalnız kameradan yüzeye kadar toplanıyordu.

Froxel yolu SÖKÜLDÜ. Sis artık bulut march'ının 5. katman yuvası
(`kCloudFogLayer = 4`).

- **CPU haritası = ZARF** (256², texel `sis, sis tepesi y, pus, zemin y`).
  Hücre haritasının 4. yaprağına yüklenir (256×1024 RGBA16F,
  `updateCloudRtParams`, revizyon değişince). Yağış ve hücre örtüsü 64²
  ızgarada hesaplanıp ara değerlenir.
- **Radyasyon sisi:** `gece · sakin · nem · açık gök` (formüller
  değişmedi); 0.012 1/m × (0.25 + 0.75·çukurluk). Tepe
  `zemin + (20 + 0.6·dolgu)·(0.6 + 0.4·çukurluk) + 60·sabah açılması`.
- **Pus:** yalnız yağış sonrası. Zeminde 0.0015 1/m, yukarı doğru 250 m
  ölçek yüksekliğiyle söner.
  - Yağış perdesi içindeki pus yok; perde zaten o ortam. Froxel'deki
    kopyası vadiyi güneşli bir pusla yıkamıştı ("kum rengi dağlar").
- **Şekil (`cloudFogDensity`):**
  - taban gürültüsü 700 m hücre, dikeyde 4× basık; sis tabakalar hâlinde
    yatar, kümelere bölünür;
  - tepe derinliğin ±%35'i kadar dalgalanır;
  - ayrıntı gürültüsü (120 m) kenarları tutam tutam aşındırır;
  - her noktada ≤ zarfın tepe değeri. Haritanın en yüksek sönümü majorant.
- **March:**
  - Sis kutusu (harita karesi × yükseklik bandı) ayrı bir aralık.
  - Adım ≥ 20 m, mesafeyle 2 km'de iki katına çıkar.
  - Işık bulutlarla aynı döngüden: güneş march'ı bulutların içinden geçer
    (bulut gölgesi), gök ışığı en alçak bulut tabanına kadar tek tapla
    örtülür, çoklu saçılma oktavları.
  - `cloudDensity` sisi içerir, yani sis kendini ve araziyi gölgeler.
    RayFusion gölge haritası ve IPC `world.sample_clouds` da sisi görür.
- **Render bayrağı:** bütün katmanlar kapalıyken de sis aktifse bulut alanı
  çizilir (açık gece radyasyon sisi).
- **Harita yeri, kenar sönümü, 10 s zaman kuantumu:** değişmedi.
- **IPC:**
  - `world.sample_weather_local` ZARFI verir.
  - Çizilen yoğunluk `world.sample_clouds {mode:"density"}` ile ölçülür.
  - `world.local_weather_stats`: `band_y`, GPU'daki revizyon.
- **Kabul testi:** `scripts/ipc/Probe-LocalWeather.ps1` (5b: march'ın
  yoğunluğu > 0 ve ≤ zarf tepe değeri).

**Bilinen sınırlar.**
- Sis yalnız bulut alanı çizen yollarda var: Vulkan RT ve RayFusion
  (OptiX donduruldu).
- Bulut katmanıyla üst üste binen aralıklar ayrı ayrı birleştirilir.
  Sırayla çarpılır, karışık ortam tam doğru değildir; dağa takılan
  bulutta fark edilebilir.
- Isınma terimi şimdiki güneşi okur. Güneş animasyonu (gün döngüsü, 5c)
  gelince doğmuş hücrelerin doğum yeri de kayar; o zaman doğum anının
  güneşi kullanılmalı.
- Sis rüzgârla sürüklenmiyor (zaten sakin havada oluşuyor).
- Hamle cephesi tozu (§1 gerekli) ve adveksiyon sisi yok.
