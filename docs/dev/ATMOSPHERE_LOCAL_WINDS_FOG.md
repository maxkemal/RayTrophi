# Yerel rüzgârlar, zemin koşulları ve yerel sis (Atmosfer Faz 5b + 7)

> **Durum:** TASLAK — 2026-10-12. Kullanıcı isteği: "yağış durumu, hava
> uygunluğu ve zemin kontrolü ile sislenme; lokal sis; bulutları yükselten,
> merkeze çeken ve çevreye kuvvet uygulayan lokal rüzgârlar". Bu not
> `ATMOSPHERE_STORMS.md` §4'teki (5b) taslağın yerini alır. Karar bekliyor
> (§6).

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
