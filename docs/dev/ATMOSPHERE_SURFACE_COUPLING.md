# Atmosfer → yüzey: yağışın geometri, arazi ve domain'lerle konuşması (ayrı plan)

> **Durum:** TASLAK — 2026-10-12. Kullanıcı: "yüzey etkileşimi, birikme,
> sıçrama, akış, domain/terrain/kar birikimi ayrı iş". Bu not yalnızca
> ARAYÜZÜ şimdi sabitler, uygulama sonra gelecek. Gerekçe: arayüz şimdi
> sabitlenmezse iki taraf "yağış nerede" sorusunu ayrı ayrı hesaplar ve
> sessizce ayrışır. Bu, bu deponun bilinen hata sınıfı.

## 1. Tek arayüz: yer yağış haritası

Atmosfer her güncellemede sahne çevresinde bir 2D harita yayınlar. Hücre
haritasıyla aynı çerçevede, 256² ya da kamera çevresinde daha ince bir
iç halka:

| Alan | Birim | Kaynak |
|---|---|---|
| `rate` | mm/h | `cloudPrecipSample` mantığı, yer seviyesinde (virga dahil) |
| `type` | enum | `ATMOSPHERE_PRECIP_TYPES.md` |
| `particle_size` | mm | aynı |
| `wind` | m/s xz | `atmosphere::sampleWind` (`ATMOSPHERE_LOCAL_WINDS_FOG.md`) |
| `air_temperature` | K | iklim + yükseklik |
| `recent_rain` | mm (son ~60 dk) | `ATMOSPHERE_LOCAL_WINDS_FOG.md` §3 |

- **Yön tek: atmosfer → yüzey.** Yüzeyin durumu (su birikintisi, kar
  kalınlığı) atmosfere geri yazılmaz (ATMOSPHERE_SYSTEM §3).
- CPU'dan da sorgulanabilir: `atmosphere::sampleGroundPrecip(xz, t)`.
- IPC: `world.sample_ground_precip`.

## 2. Tüketiciler (her biri ayrı parti)

| Tüketici | Okuduğu | Ürettiği | Mevcut sistem |
|---|---|---|---|
| Islaklık ve birikinti | rate, type, recent_rain | ıslaklık, birikinti maskesi (eğim ve AO'ya göre) | `surface_wetness_output` (elle giriliyor, bağlanacak) |
| Sıçrama | rate, type, particle_size | yere ve yüzeye çarpma parçacıkları | near-field yağış parçacıkları |
| Akış | rate, birikmiş su | arazi eğiminden kanallar, su domain'ine kaynak | terrain + su domain'i (APIC / sparse) |
| Kar birikimi | rate (kar), air_temperature, wind | kar kalınlığı alanı: eğim, rüzgâr gölgesi, yığılma | terrain + `grain_*` (Snow, sıkışma U0–U5) |
| Erime | air_temperature, güneş | kar → su (gizli ısı) | T2 ısı ve faz katmanı |
| Dolu | type = hail, size | rigid / DEM taneleri, sekme | Jolt / DEM |
| Toz | wind (hamle cephesi), kuru zemin | toz bulutu | 4b toz |

## 3. Zaman ve geri sarma

- Atmosfer tarafı saf fonksiyondur: harita her `t` için aynıdır.
- Birikim (kar kalınlığı, su) zamanla toplanır, yani DURUMLUDUR. Bu iş
  fizik tarafında, onun önbellek ve geri sarma kurallarıyla yaşar.
- Kolay önizleme için atmosfer tarafı analitik bir tahmin de verebilir
  (`recent_rain`). Ama kalınlığın otoritesi fizik tarafıdır.

## 4. Sıra (öneri)

1. Haritayı yayınla + IPC.
2. Islaklığı bağla.
3. Kar birikimi.
4. Sıçrama.
5. Akış → su domain'i.
6. Dolu.
