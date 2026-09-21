# Sıradaki derlemede kontrol edilecekler

> **Durum:** CANLI — 2026-09-21, cihaz yerleşik gaz alt zinciri **doğrulandı**,
> anahtar söküldü, `gpu_host_sync_ms` satırı eklendi.
> Bu partide **yalnız C++** değişti — `compile_shaders.bat` gerekmiyor.

Önceki partinin fizik doğrulaması bitti ve ölçümü
`docs/dev/GAZ_ADIMI_TASIMA_MALIYETI.md`'ye geçti. Bu parti onun **temizliği**:
davranış değişmemeli.

---

## 1. ★★★★★ Fizik yine birebir aynı olmalı

Anahtar artık yok, yani karşılaştıracak ikinci yol da yok. Referans, bir önceki
derlemede ölçülen ve senin görsel olarak da onayladığın tablodur:

| kare | hücre | fill | tepe | merkez | peakT | meanT |
|---|---|---|---|---|---|---|
| 40 | 211 742 | 0.06265 | 16.150 | 11.292 | 8.0034 | 3.41141 |
| 80 | 721 307 | 0.21340 | 31.790 | 22.530 | 5.5467 | 2.27425 |
| 120 | 959 364 | 0.28384 | 34.000 | 28.940 | 3.8441 | 1.29586 |

```
gas.reset  →  timeline.set_frame 1..120  →  gas.measure_plume
```

**Ne görmen gerek:** aynı sayılar. Sökülen dal zaten hiç çalışmıyordu (anahtar
varsayılan olarak açıktı), yani tek satır bile değişmemeli.

**Bozuksa ne demek:** sökerken yanlış dalı aldım — combustion'ın CPU yedeği
devreye girmiştir. Belirtisi hız değil, **farklı sayılar** olur.

## 2. ★★★ Yeni satır: `gpu_host_sync_ms`

```
gas.step_stats  →  gpu_host_sync_ms
```

**Ne görmen gerek:** sıfırdan büyük, kabaca **4 × 12.89 MB'lık bir indirme**
kadar (geçen ölçümdeki muhasebeye göre ~7 ms civarı). Panelde de yeni bir satır
var: *"grid readback (for host solver)"*.

**Bozuksa ne demek:** 0.0 kalıyorsa mark yanlış yere kondu ve satır **yalan
söylüyor** — bu en sinsi hâli, çünkü "çok hızlı" diye okunur. Panelde
`phase_sum` ile `total_ms` arasındaki fark da kapanmış olmalı; kapanmadıysa
ölçülmeyen başka bir boşluk daha var.

★ Bu satır bir sonraki iş için var: `boundaries + solids`'in GPU'ya taşınması
**tam olarak bu sayı kadar** değerli. Satır olmadan o iş yapılır ve ödülü
ölçülemez.

## 3. Panel ve script paritesi

`gas.step_stats` ve `gas.get_settings` çıktısında artık
`device_resident_chain` **olmamalı**; `gpu_host_sync_ms` **olmalı**.

⚠ **Kaydedilmiş projeler:** senin `.rtp` dosyanda `gas_device_resident_chain`
hâlâ yazılı (ben onu IPC'den `false` yapmıştım). Okuyucu söküldüğü için o değer
artık **sessizce yok sayılıyor** ve yerleşiklik her zaman açık. İstenen
davranış bu, ama bilerek olsun.

## 4. Hız — beklenti küçük

| | ölçülen |
|---|---|
| `gpu_combustion_ms` | 14.09 → **6.79** |
| `total_ms` | 218.15 → **212.83** |

Adımın tamamında ~%2.4. Sebebi açık ve kalıcı: `gasSyncGridToHost` combustion'ın
bıraktığı indirmeyi **geri ekliyor**. `boundaries + solids` host'ta ve zincirin
ortasında durdukça bu böyle kalır.

---

## Sıradaki iş (bu parti değil)

1. `boundaries + solids` → GPU. Yapısal geri okumayı kaldırır; değeri artık
   `gpu_host_sync_ms` ile ölçülebilir.
2. Scalar advection'ın handle takası (hâlâ indirip host'ta `swap` ediyor).
3. Sonra pressure projection'ı **yeniden** ölç (47 ms) — transferler kalkmadan
   ölçülen değeri iterasyon maliyeti sanmak hata olur.

## Açık kalanlar

- Kapak f110'dan sonra domain tavanında (34 m). Parametreyle çözülmez.
- `temperature_scale` IPC'den açılmadı (kural 1 boşluğu).
- Canlı domain'de `density_dissipation = 0.63`, doğrulanmış değer **0.18**
  (preset'te 0.18 yazıyor). Kaydedilmiş proje değeri preset'i eziyor.
