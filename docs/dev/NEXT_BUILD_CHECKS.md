# Sıradaki derlemede kontrol edilecekler

> **Durum:** REFERANS — 2026-09-21. Son parti derlendi ve **doğrulandı**;
> bekleyen kontrol maddesi yok. Bir sonraki parti bu dosyanın üzerine yazacak.

Ölçümlerin tamamı `docs/dev/GAZ_ADIMI_TASIMA_MALIYETI.md`'de.

---

## Doğrulanan hâl (2026-09-21 son derleme)

**Fizik referansı** — `gas.reset` + `timeline.set_frame 1..120` +
`gas.measure_plume`, nükleer sahne. Bu tablo üç ardışık derlemede **son haneye
kadar** aynı çıktı; bir değişiklik fiziği bozduysa ilk burada görünür.

| kare | hücre | fill | tepe | merkez | peakT | meanT |
|---|---|---|---|---|---|---|
| 40 | 211 742 | 0.06265 | 16.150 | 11.292 | 8.0034 | 3.41141 |
| 80 | 721 307 | 0.21340 | 31.790 | 22.530 | 5.5467 | 2.27425 |
| 120 | 959 364 | 0.28384 | 34.000 | 28.940 | 3.8441 | 1.29586 |

**Adım maliyeti:** `total_ms` **163.30** (oturum başı ~203–218).
**Sıvı tutuşma:** `burning_cells` ~3037/adım, preset'ler tutuşuyor.
**Blok doluluğu:** 6634 / 7225 = %91.8 → seyrek dispatch elenmiş durumda.

---

## Açık kalanlar (kod değil, karar bekleyenler)

- **`.gitignore:44` satırı `*.ps1`** — `scripts/ipc/*.ps1`'in hiçbiri
  versiyonlanmıyor. CLAUDE.md'nin "bu projenin QA altyapısı" dediği katman bu.
  Muhtemelen kaza; dokunulmadı.
- `max_velocity` `gas.get_settings`/`gas.get` çıktısında **yok** — `max_speed`
  tripwire'ını doğrulamak için gereken tavan script'ten okunamıyor (kural 1).
- `temperature_scale` IPC'den açılmadı (kural 1).
- Adım atarken süreç CPU'su 16 çekirdeğin **%26.4'ü** (boşta %0), oysa zamanlama
  satırlarının topladığı host işi ~11 ms/163 ms. Fark **ölçülmedi**. İlk
  şüpheli OpenMP'nin varsayılan spin-wait'i ve GPU fence beklemesi. Yeniden
  derleme gerektirmeyen test: `OMP_WAIT_POLICY=PASSIVE` ile başlatıp tekrar
  ölçmek.
- Kapak f110'dan sonra domain tavanında (34 m). Parametreyle çözülmez.
- Canlı domain'de `density_dissipation = 0.63`, doğrulanmış değer **0.18**
  (preset'te 0.18 yazıyor). Kaydedilmiş proje değeri preset'i eziyor.

## Sıradaki optimizasyon

Host'ta yalnız surface dust (~3.2 ms) ve skaler dissipation (~2.9 ms) kaldı.
İkisi GPU'ya geçerse `GridFluid::step` gaz yolunda tamamen atlanabilir ve
`gpu_host_sync_ms` (7.06 ms) da kalkar. Ondan sonrası pressure projection
(46.52 ms) ve **transferler kalktığına göre yeniden ölçülmesi gerekir** —
eski değeri iterasyon maliyeti sanmak hata olur.
