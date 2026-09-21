# Sıradaki derlemede kontrol edilecekler

> **Durum:** CANLI — 2026-09-21. Host kalıntısının son iki aşaması GPU'ya
> taşındı. **İKİ YENİ SHADER VAR → `compile_shaders.bat` ŞART.**

Yeni kernel'ler: `sim_gas_scalar_dissipate.comp`, `sim_gas_surface_dust.comp`.
Push-constant aralıkları 36 ve 40 bayt olarak `SimulationComputeVulkan.cpp`'ye
kaydedildi; üçü (shader / tablo / host struct) `static_assert` ile bağlı.

---

## 1. ★★★★★ Fizik yine birebir aynı olmalı

```
gas.reset  →  timeline.set_frame 1..120  →  gas.measure_plume
```

| kare | hücre | fill | tepe | merkez | peakT | meanT |
|---|---|---|---|---|---|---|
| 40 | 211 742 | 0.06265 | 16.150 | 11.292 | 8.0034 | 3.41141 |
| 80 | 721 307 | 0.21340 | 31.790 | 22.530 | 5.5467 | 2.27425 |
| 120 | 959 364 | 0.28384 | 34.000 | 28.940 | 3.8441 | 1.29586 |

Bu tablo dört ardışık derlemede aynı çıktı. İki aşama host'tan GPU'ya taşındı,
yani **aritmetik yeniden yazıldı** — burası bu partinin gerçek sınavı.

**★ En sinsi başarısızlık:** `meanT` ve `hücre` doğru ama **tepe (top)** ve
zemin eteği yanlış. Surface dust sadece taban katmanını üretir; kernel'de MAC
indeksi yanlışsa rüzgâr hızı yanlış hücreden okunur ve toz **yanlış yerde**
kalkar. Bulut yine makul görünür. Ayırt edici: f120'de `top = 34.0` ve
`centroid = 28.94`.

**Dissipation için ayırt edici:** `meanT` ve `fill`. Faktör yanlışsa
(`exp(-rate*dt)` yerine başka bir şey) bulut ya hiç incelmez ya da erir.

## 2. ★★★★ Yeni satırlar sıfır olmamalı

```
gas.step_stats → gpu_surface_dust_ms, gpu_scalar_dissipate_ms
cpu_surface_dust_ms, cpu_dissipation_ms  →  ikisi de ~0 olmalı
```

| satır | önce | beklenen |
|---|---|---|
| `cpu_surface_dust_ms` | 3.17 | **~0** |
| `cpu_dissipation_ms` | 2.88 | **~0** |
| `cpu_total_ms` | 6.42 | **~0.3** |
| `gpu_surface_dust_ms` | (yoktu) | küçük, >0 |
| `gpu_scalar_dissipate_ms` | (yoktu) | küçük, >0 |
| `total_ms` | 163.30 | **~157** |

**★ Bozuksa ne demek — ve bunu özellikle kontrol et:** `gpu_surface_dust_ms`
sıfırdan büyük **ama** `cpu_surface_dust_ms` de sıfırdan büyükse, GPU yolu
`false` dönüyor ve host yeniden yapıyordur. İkisi birden çalışırsa toz **iki kez**
eklenir; belirtisi çökme değil, **daha kalın bir zemin eteği**.

★ `runGpuGasSurfaceDust` collider'lı domain'de **kasten `false` döner** (kernel
yalnız taban katmanını yapar, host sürümü katıların üstünü de yürür). Yani
collider'lı bir sahnede `cpu_surface_dust_ms > 0` görmek **doğru davranıştır**.

## 3. Bir şey DEĞİŞMEDİ: yapısal geri okuma

`gpu_host_sync_ms` (~7 ms) **duruyor ve bu partide kalkmıyor**. Host çözücüsü
artık boş, ama host **tüketicileri** hâlâ ızgarayı okuyor: alan analiz taraması
(`max_speed`, `burning_cells`, `active_density_cells`), `gas.measure_plume`,
bake/cache yazımı ve VDB dışa aktarımı. Onlar deftere bağlanmadan geri okumayı
kaldırmak, bake'in **bayat veri yazması** demek olur — sessizce.

> Bu, kendi kuralımızın aynısı: **her TÜKETİCİ deftere bağlanmadan hiçbir
> ÜRETİCİ indirmeyi bırakamaz.**

---

## VRAM: ölçüldü, ve ölç aleti kör

`perf.get_gpu_memory` (bu sahne, açıkken):

| | bayt |
|---|---|
| izlenen device-local | 672 MB |
| **izlenmeyen** | **1001 MB** |
| toplam VRAM kullanımı | 1.67 GB / 12.1 GB |

★ Gaz domain'inin compute tamponları (~25 alan × 13.5 MB ≈ **340 MB**)
kategorilerin **hiçbirinde görünmüyor** — render tarafında `other` 16 KB.
Yani optimize edeceğimiz şey, izlenmeyen 1 GB'ın içinde.

**Bu yüzden VRAM'e dokunmadan önce yapılacak iş bir kernel değil, bir sayaç:**
`ensureComputeBuffer` tahsislerini `simulation` kategorisi altında muhasebeye
sokmak. Aksi hâlde bu oturumda iki kez yaptığımız hatayı üçüncü kez yaparız —
hedefi, o hedefi göstermeyen bir tablodan seçmek.

İzlendikten sonraki bariz adaylar (önce ölç, sonra kes):
- `scratch_scalar` / `scratch_scalar2` / `scratch2_vel_x|y|z` — ping-pong
  tamponları; kaçı aynı anda canlı?
- `divergence` ve `pressure` yalnız projeksiyon içinde yaşıyor.
- `msf_accum_*` dört alan, MSF kapalıyken de tahsis ediliyor mu?

## Sıradaki

1. Bu partiyi ölç (yukarıdaki 1–2).
2. VRAM muhasebesine `simulation` kategorisi.
3. Pressure projection (46.52 ms, adımın %28'i) — **transferler kalktığına göre
   yeniden ölçülmeli**; eski değerini iterasyon maliyeti sanmak hata olur.
4. Aynı dersleri fluid (APIC) yoluna taşımak: orada da aşama başına
   upload→dispatch→download kalıbı ve "adı başka şey olan zamanlama satırları"
   olup olmadığına bakmak.

## Açık kalanlar

- `max_velocity` `gas.get_settings`/`gas.get` çıktısında yok (kural 1).
- `temperature_scale` IPC'den açılmadı (kural 1).
- Adım atarken süreç CPU'su 16 çekirdeğin %26.4'ü, oysa zamanlama satırlarının
  topladığı host işi ~11 ms/163 ms. **Ölçülmedi.** Yeniden derleme gerektirmeyen
  test: `OMP_WAIT_POLICY=PASSIVE` ile başlatıp tekrar ölçmek.
- Kapak f110'dan sonra domain tavanında (34 m).
- Canlı domain'de `density_dissipation = 0.63`, doğrulanmış değer 0.18.
