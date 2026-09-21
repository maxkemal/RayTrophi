# Sıradaki derlemede kontrol edilecekler

> **Durum:** CANLI — 2026-09-21. Önceki derlemedeki **zayıf patlama regresyonu
> bulundu ve düzeltildi.** Yalnız C++ — `compile_shaders.bat` gerekmiyor.

---

## 0. ★★★★★ Ne oldu: RT yayınlaması deftere sormuyordu

**Belirti:** patlama çok zayıf. f40'ta meanT **0.50** (referans 3.41), aktif
hücre 120 312 (referans 211 741). Değerler kontrol edildi, **hepsi aynıydı** —
yani kod.

**Kök:** adım sonundaki alan yayınlaması dört alanı düz `uploadBuffer` ile
gönderiyordu. Gaz adımındaki deftere bağlanmamış **son** yükleme oydu, ve
doğru çalışmasının tek sebebi zincirin ortasındaki geri okumanın host'u
otoriter bırakmasıydı. Bir önceki partide o geri okumayı adımın sonuna
taşıyınca, yayınlama **bayat host kopyasını** combustion'ın ve surface dust'ın
cihaz-only sonuçlarının üstüne yazmaya başladı — her adımda.

**Düzeltme:** geri okuma yayınlamanın **üstüne** alındı (ilk host tüketicisi
analiz taraması değil, yayınlamanın kendisiymiş), ve dört yükleme
`gasEnsureOnDevice`'a çevrildi.

★ **Bu aynı şeklin üçüncü tekrarı** — `runGpuTurbulence`, MSF/sıvı depozitleri,
ve şimdi yayınlama. Ortak nokta: *bir aşamanın deftere bağlanmamış olması,
zincirin başka bir yeri değişene kadar zararsız görünür.*

---

## 0b. ★★★★ DÖRDÜNCÜSÜ: projeksiyon alev kanalını eziyordu

**Belirti:** görüntü doğru ama `Burning cells` her karede **0**. "Ölçüm hatası
olabilir" diye bakıldı — değildi.

`runGpuPressureProjection` `grid.interaction`'ı düz `uploadBuffer` ile
gönderiyordu. Combustion alevi cihaza yazıp orada bırakıyor; projeksiyon bayat
host kopyasını üstüne yazıyor, **üstelik host'u besleyen geri okumadan önce**.
Alev alanı host'a hiç ulaşmıyordu.

★ **Sadece sayacı bozmuyordu:** binding 5'teki termal genleşme terimi de ölü bir
alan okuyordu. Yani bu kozmetik değil.

★★ **Tuzağı not et:** sıfır okuyan bir alet, bozuk alet gibi görünür. Bu
partide "ölçüm hatası" sanılan şey gerçek bir veri yolu hatasıydı.

**Kontrol:** `gas.step_stats` → `burning_cells` sıfırdan büyük olmalı.
(Önceki çalışan derlemede ~3037/adım.)

---

## 1. ★★★★★ Fizik referansı

```
gas.reset  →  timeline.set_frame 1..120  →  gas.measure_plume
```

| kare | hücre | fill | tepe | merkez | peakT | meanT |
|---|---|---|---|---|---|---|
| 40 | 211 741 | 0.06265 | 16.150 | 11.293 | 8.0049 | 3.41137 |
| 80 | 721 318 | 0.21341 | 31.790 | 22.530 | 5.5467 | 2.27427 |
| 120 | 959 083 | 0.28375 | 34.000 | 28.940 | 3.8441 | 1.29608 |

Bu partide aritmetik değişmedi → **birebir aynı olmalı.**

**Hızlı ön kontrol (30 saniye, tam taramadan önce bunu yap):** `gas.reset`,
30 kare, `gas.step_stats` → **`burning_cells` sıfırdan büyük olmalı.** Bozuk
derlemede **0**'dı. Sıfırsa daha ileri gitme.

## 2. ★★★★ Hız

| satır | iki parti önce | beklenen |
|---|---|---|
| `gpu_velocity_advect_ms` | 25.5 | ~15–19 |
| `gpu_body_forces_ms` | 16.1 | çok küçük (aşağıya bak) |
| `gpu_host_sync_ms` | 6.7 | tekrar dolu, artık **adım sonunda** |
| `total_ms` | 156.9 | **~130–140** |

⚠ **★★★ ÖLÇÜ ALETİ HAKKINDA UYARI — bunu okumadan rakamları yorumlama.**
Aşamalar artık `synchronize()` çağırmadığı için CPU tarafındaki zamanlayıcılar
**GPU işini değil, kuyruğa bırakma süresini** ölçüyor. Bozuk derlemede
`gpu_body_forces_ms` **0.04** çıktı; iş kaybolmadı, bir sonraki senkronizasyona
(çoğunlukla `gpu_pressure_ms`) taşındı.

> Yani `total_ms` hâlâ dürüst, ama **aşama satırları artık birbiriyle
> kıyaslanamaz.** Bunu gerçekten çözmek GPU timestamp query'leri ister; o
> yapılana kadar optimizasyon hedefi **satırlardan seçilmemeli.** Bu notun
> tamamı zaten bu hatanın üç kez yapılmasının kaydı.

## 3. ★★★ Yeni dial: `pressure_iterations`

```
gas.set_settings domain='Nuclear Gas' pressure_iterations=<N>
```
Panelde **"Pressure Sweeps"**, varsayılan 40 (eskiden koda gömülüydü).

**Ölçüm:** N = 10/20/40/80 için `gpu_pressure_ms`. Doğrusal ve kesişim ≈ 0 →
projeksiyon bant-bound, kaldıraç algoritmik. Kesişim büyük → önce o sabit
maliyet alınır. `gpu_pressure_ms` şu an senkronize eden az sayıdaki satırdan
biri olduğu için **bu ölçüm hâlâ güvenilir.**

⚠ Yakınsama dial'i, kalite dial'i değil: düşürmek görüntüyü yumuşatmaz, gazı
duvarlardan sızdırır ve girdabı öldürür. `top` ve `fill`'e bak.

## 4. Çökme yolu

Zincir yarıda kalırsa hız alanları `markHostWrote` ile host'a devrediliyor ve
**advection da CPU'da yeniden yapılıyor**. Bozuksa kuvvetler iki kez uygulanır;
belirtisi çökme değil, ani bir tekme. Log'da `falling back` görürsen bak.

---

## Sıradaki

1. Bu partiyi ölç (0 → 1 → 2).
2. **GPU timestamp query'leri** — artık aşama satırları güvenilir değil ve bu,
   sıradaki her optimizasyon kararının önkoşulu.
3. Analysis scan'i GPU'ya (host'taki son tam ızgara tarama).
4. VRAM muhasebesine `simulation` kategorisi (`perf.get_gpu_memory` 672 MB
   izliyor, **1001 MB izlemiyor**).
5. Projeksiyon: 3. maddedeki taramanın sonucuna göre.

Devir notu: `docs/dev/GAZ_DERSLERI_VE_FLUID_DEVRI.md`.

## Açık kalanlar

- `max_velocity` ve `temperature_scale` IPC'de yok (kural 1).
- Adım atarken süreç CPU'su 16 çekirdeğin %26.4'ü; açıklanmadı.
  Test: `OMP_WAIT_POLICY=PASSIVE`.
- `runGpuVelocityAdvection`'daki `compute->synchronize()` artık geri okuma
  olmadığı için saf duraklama; kaldırılabilir.
- Kapak f110'dan sonra domain tavanında (34 m).
