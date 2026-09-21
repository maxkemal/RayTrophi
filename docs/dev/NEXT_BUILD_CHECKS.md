# Sıradaki derlemede bakılacaklar

> **Durum:** AKTİF — 2026-09-21: nükleer preset elle ayarlanan değerlerle
> güncellendi, `gas.step_stats` açıldı, GPU yükleme defterinin ilk yarısı girdi.
> Arka plan: [GAZ_ADIMI_TASIMA_MALIYETI.md](GAZ_ADIMI_TASIMA_MALIYETI.md).

Sıralama: bağımsız ve hızlı görüleni önce, diğerlerinin sonucunu maskeleyeni sonra.

---

# ★★★★★ 0. ÖNCE BU — GERİ YÜKLENEN İŞ VE CANLI BİR ABI UYUŞMAZLIĞI

2026-09-21'de `git checkout -- src/Physics/ParticleSimulation.cpp` çalıştırdım
(residency denemesini geri almak için) ve o dosyadaki **commit edilmemiş
önceki parti işlerini de sildim**. Kaybolanlar geri yazıldı:

- `params.ambient_stratification` eşlemesi
- `params.surface_dust_*` eşlemesi (altısı birden)
- `gas_dissipation_override` → density/temperature/fuel oranları
- `effectiveTurbulenceOctaves` kırpması
- `GasBuoyancyGpuConstants`'ın `stratification` + `voxel_size` alanları

★★★ **Bir önceki derlemede CANLI bir hata vardı:** `sim_gas_buoyancy.comp` ve
`SimulationComputeVulkan.cpp` push aralığını **44 byte** ilan ediyordu, host
struct ise **36**. Host 36 byte itiyordu, shader `stratification` ve
`voxel_size`'ı **ilklenmemiş bellekten** okuyordu. Validation hatası yok, çökme
yok — sadece çöple sürülen bir kaldırma terimi.

★★ Ve bunun ortaya çıkardığı daha büyük şey: `surface_dust_enabled`
`SolverParams` içinde `false` varsayılanlı ve **hiçbir zaman true yapılmamış**.
Yani yüzey-tozu özelliği **hiçbir sahnede bir kez bile çalışmamış**. Zemin tozu
sandığımız şey, senin elle eklediğin `stem.density = 2.35` idi.

**Ne görmen gerek:** `static_assert(sizeof(GasBuoyancyGpuConstants) == 44)`
derleniyor. Sahnede mantar hâlâ oturuyor ve tepe domain tavanına değmiyor.

**★★★ Bozuksa ne demek:** Bulut ŞİMDİ davranış değiştirdiyse bu bir regresyon
DEĞİL — stratification ve dissipation override ilk kez gerçekten uygulanıyor.
Tepe artık `h* = anomali / stratification`'a göre oturmalı; 0.25 ile hedef
~26 m. Yeni yükseklik bundan çok farklıysa oranı yeniden kalibre et.

**⚠ HEMEN COMMIT AL.** Bu iş üç partidir commit edilmemiş durumda duruyordu ve
tek bir `git checkout` onu sildi.

---

## 1. ★★★ ÖNCE BU: simülasyon sonucu DEĞİŞMEMELİ

Yükleme defteri bir performans değişikliği; görüntüyü değiştirmemeli.

```powershell
Invoke-RtIpc timeline.set_frame @{ frame = 100 }
Invoke-RtIpc gas.measure_plume @{ domain = 'Nuclear Gas' }
```

**Ne görmen gerek:** `top_above_floor`, `active_cells`, `peak_temperature`
2026-09-21 ölçümüyle aynı bantta (f100'de top 34.00, ~1.05M hücre, peakT ~1.99).

**★★★ Bozuksa ne demek — ve bu partinin EN SİNSİ arızası bu:** duman duvarlardan
sızıyorsa ya da bulut şeklini kaybettiyse, `GridFluid::step` sonrasındaki
geçersiz kılma çalışmıyordur. Belirti çökme değil: `boundaries + solids`'in
sonuçları yok sayılır ve domain sessizce sızdırır. Çökmez, **makul görünür**.

---

## 2. Yeni ölçü aleti çalışıyor mu

```powershell
Invoke-RtIpc gas.step_stats @{ domain = 'Nuclear Gas' }
```

**Ne görmen gerek:** `measured = true` ve dolu aşama satırları
(`gpu_velocity_advect_ms`, `gpu_pressure_ms`, `analysis_ms`, `cpu_boundary_ms`, …)
artı `cfl`, `resolution`, `cell_count`.

**Bozuksa ne demek:** `measured = false` ise o domain adım atmamıştır — timeline'ı
oynat. Metot hiç yoksa dört dokunuştan biri eksiktir (`authorize()` fail-closed
çalışır, yani sessizce reddeder).

---

## 3. Yükleme sayısı düşmeli

**Ne görmen gerek:** `gpu_scalar_advect_ms` belirgin düşmüş olmalı — tek başına
adım başına üç yüz-arayüzü yüklemesi kalktı. `gpu_velocity_advect_ms` de bir
miktar düşer (artık indirme sonrası handle takası var, ek kopya yok).

**⚠ Beklentiyi doğru tut:** adım toplamının yarıya inmesini BEKLEME. Yalnızca
hız alanları dönüştürüldü ve host çözücü hâlâ zincirin ortasında. Ölçülü bir
düşüş doğru sonuçtur.

---

## 4. Nükleer preset — sahneden okunan değerlerle

Preset'i yeniden uygula (`particle.add_preset` ya da panel) ve karşılaştır.

**Gaz / domain:** voxel 0.17, `turbulence_octaves` 8 (etkin 3), stratification 0.25.

**Akış kaynakları:** core r=3.02 / fuel=87.6, fireball r=1.80 / **fuel=0**,
stem r=1.95 / density=2.35 / **fuel=0**.

**Shader:** density_multiplier **13.194**, σs **0.84**, σa **1.72**,
blackbody **16.667**, pencere 1340-5000, cutoff 0.007.

**Debris emitter:** point **(0,0,0)**, direction **(0,0.7,0)**, speed **11.9**,
lifetime **5.85 s**, mass **0.4**, burst 320.

**★ İki tavizi bilerek taşıyoruz:**
- `stem.density = 2.35` (0 değil) → sap artık zemin-tozu kuralının kanıtı DEĞİL.
  Kuralı sınamak için density'yi 0 yapıp sap hâlâ oluşuyor mu diye bak.
- `debris.mass = 0.4` → sürükleme 0.8 ve yerçekimi 9.81 ile birlikte yayı
  belirler; 1.0 varsayılanı nötr değildi.

---

## 5. Görsel durum — neyin ÇÖZÜLDÜĞÜ, neyin kaldığı

2026-09-21 ölçümü (aynı sahne, elle ayarlanmış):

| kare | tavan | tepe | centroid | peakT |
|---|---|---|---|---|
| 20 | Hayır | 8,84 | 5,38 | 9,16 |
| 45 | Hayır | 17,17 | 12,30 | 5,27 |
| 90 | Hayır | 30,94 | 23,22 | 2,53 |
| 140 | **Hayır** | **31,28** | 23,43 | 2,42 |

**✔ ÇÖZÜLDÜ — tavan.** `touching_ceiling` artık hiçbir karede true değil ve
tepe 31,3'te OTURUYOR (domain 34). Yükseklik artık kapak değil fizik.

**✔ ÇÖZÜLDÜ — kor sap.** `stem.fuel` 0'a alındı; sütun karardı, yalnızca üst
kısımda artık blackbody parıltısı kaldı ki doğrusu bu.

**✔ BÜYÜK ÖLÇÜDE ÇÖZÜLDÜ — kırpılma.** σa/σs oranı ve daha düşük
density_multiplier ile kapak loblarında gölge ve derinlik var.

**⚠ KALDI — zemin eteği yükselmiyor.** `max_width` hâlâ 22,10 (= domain
genişliği) ve en geniş dilim **y=0,09**'da. Yani zemin katmanı duvardan duvara
yayılıyor ama YÜKSELMİYOR: görüntüde kabaran bir base surge değil, yanan ince
bir hat okunuyor. Adaylar: `surface_dust_max_density` (2,5) tavanı,
`gas_buoyancy_density` (0,02) ile kaldırılan tozun hafifliği, ya da domain
genişliğinin yanal yayılmayı erken duvara dayaması.

**⚠ KALDI — sap çok tekdüze.** Dikey ve düzgün; gerçek sap düzensizdir ve
kapağa doğru genişler. `turbulence_octaves_effective` 3 ve voxel 0,17 ile
sütun ölçeğinde kıvrım üretecek frekans yok.

---

## 6. Sırada (bu partide değil)

- Defteri skaler alanlara genişletmek — ama önce 3. maddenin ölçümü gelsin.
- `boundaries + solids` neden CPU'da (40 ms)? GPU portu var, devrede değil.
  Zincirin ortasındaki yapısal round-trip'i kaldıracak tek şey bu.
- Boş hücreleri dispatch'ten çıkarmak (aktif hücreler ~%31).
- Advection alt adımlama — `cfl` artık `gas.step_stats`'ta, önce onu oku.
