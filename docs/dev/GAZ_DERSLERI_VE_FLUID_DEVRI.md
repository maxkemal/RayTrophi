# Gaz sisteminden çıkan dersler ve fluid'e devir

> **Durum:** AKTİF — 2026-09-21. Gaz adımı 203→157 ms. Sıradaki iş sırası:
> **(1) gaz tarafında bellek muhasebesi ve VRAM, (2) sonra performans,
> (3) sonra aynı yöntemi APIC fluid yoluna taşımak.** Bu not devralan ajan
> içindir.

Ölçümlerin ham hâli `GAZ_ADIMI_TASIMA_MALIYETI.md`'de; burası **nasıl
çalışılacağı** ve **sırada ne olduğu**.

---

## IPC'ye nasıl bağlanılır (önce bu)

Uygulama açıksa yerel pipe zaten kalkmıştır; **token gerekmez** (kimlik
doğrulaması yalnız uzak/TLS istekleri için).

```powershell
# 1. Açık mı?
Test-Path '\\.\pipe\RayTrophiStudio'         # True olmalı

# 2. Değilse başlat (IPC hazır olunca "HAZIR" der)
.\scripts\ipc\Start-RayTrophi.ps1

# 3. Modülü yükle ve çağır
Import-Module .\scripts\ipc\RtIpc.psm1 -Force
Invoke-RtIpc agent.discover @{}
Invoke-RtIpc gas.list_domains @{}
Invoke-RtIpc particle.stats @{}
```

`Invoke-RtIpc <metot> @{ parametre = değer }` — sonuç PowerShell nesnesi olarak
döner, `| ConvertTo-Json -Depth 5` ile bakılır.

**Hangi metot var:**
```powershell
Invoke-RtIpc agent.list_methods @{ domain = 'fluid' }   # alan alan liste
Invoke-RtIpc agent.describe     @{ method = 'fluid.step' }
```
`agent.discover @{ domain = ... }` filtre uygulamaz — alan filtresi için
`agent.list_methods` kullan.

★ **Türkçe locale tuzağı:** çıktıda ondalık **virgül** görünür (`0,063`).
Karşılaştırma veya eşik kontrolü yaparken `[double]` çevir, string karşılaştırma.

★★ **Sayacı yazdığın çağrıyla aynı partide okuma.** Bir ayarı yazıp hemen
sayacı okursan **önceki** durumu ölçersin; araya bir kare koy
(`timeline.set_frame`).

★★★ **Script testi kare döngüsüne kördür.** `physics.step` gibi çağrılar
uygulama kendi karesini işlerken geri alınabilir. Çözücüyü ilerletmek için
`timeline.set_frame` kullan — o gerçekten çözücüleri sürüyor.

⚠ Script'ler **iki yerde** yaşıyor: `scripts/...` ve `x64/Release/scripts/...`.
Uygulama ikincisinden okur; yalnız ilkini güncellemek eski scripti koşturur.

---

## 0. Nerede duruyoruz

3.4M hücreli gaz domain'i (130×200×130), Vulkan compute, RTX-sınıfı GPU.

| | oturum başı | şimdi |
|---|---|---|
| `total_ms` | ~203–218 | **156.9** |
| `cpu_total_ms` (host çözücü) | 31.0 | **0.001** |
| `cpu_boundary_ms` | 23.1 | 0.001 |
| `gpu_publish_ms` | 26.2 | 4.3 |

Host çözücüsü gaz yolunda **boş**. Kalan dağılım:

| aşama | ms | pay |
|---|---|---|
| pressure projection | 45.3 | %29 |
| velocity advection | 25.5 | %16 |
| scalar advection | 19.6 | %13 |
| body forces | 16.1 | %10 |
| **host readback** (`gpu_host_sync_ms`) | 6.7 | %4 |
| surface dust | 5.2 | %3 |
| velocity dissipation | 5.1 | %3 |
| analysis scan | 4.5 | %3 |
| publish | 4.3 | %3 |
| source upload | 4.4 | %3 |
| combustion | 2.5 | %2 |
| scalar dissipation | 0.015 | ~0 |

---

## 1. ★★★★★ Ders: zamanlama tablosundan hedef seçmek, satır adlarının doğru
olduğunu varsaymaktır

Bu oturumda o varsayım **üç kez** yanlış çıktı:

| satır | altında gerçekte ne vardı | ms |
|---|---|---|
| `cpu_boundary_ms` | `clampVelocity` — 10.3M float'lık seri tarama | 23 |
| `gpu_publish_ms` | `sanitizeProjectedVelocity` — aynı büyüklükte ikinci tarama | 20 |
| `cpu_surface_dust_ms` | collider yokken bile 3.4M hücreyi tarayan katı döngüsü | 3 |

Üçü de "boundary", "publish", "surface dust" diye etiketlenmişti ve **hiçbiri o
etiketin tarif ettiği işi yapmıyordu.** İlk ikisi adımın en pahalı iki host
işiydi.

**Nasıl uygulanır:** bir satırı hedef seçmeden önce o satırın **hangi çağrıları
kapsadığını koddan oku.** İki `mark()` arasındaki her şey o satıra yazılır ve
kimse adın doğru olduğunu denetlemez.

★ Aynı hatanın bir başka biçimi: `GasStepStats`'ın kendi başlığı "bu izole
kernel süresi DEĞİLDİR, öyle okumak transfer-bound bir aşamayı compute-bound
gösterir" diye yazıyordu. Ben tabloyu okuyup seyrek dispatch'e yöneldim; kullanıcı
"donanım sınırı mı, metodolojik hata mı" diye sorunca yakalandı. **Ölçtükten
sonra:** seyrek dispatch kazanç tavanı %10, asıl sorun taşımaydı.

## 2. ★★★★★ Ders: bayrağın SET edildiği yer ile OKUNDUĞU yer arasındaki boşluk

Bu kod tabanının en pahalı hata sınıfı. Bu oturumda **dört** örneği çıktı:

1. `gpu_resident_fields_valid` — 15 yerde `false`, 5 yerde okunuyor, **hiçbir
   yerde `true`**. Arkasındaki bütün hızlı yollar ölüydü.
2. `scalar_inputs_resident` — 10614'te tanımlanıp **10730'da okunuyor, 10735'te
   atanıyor**. Okunduğu anda her zaman false, üstelik başka bir aşamanın
   bayrağıyla besleniyor ve tek boolean üç alan için cevap veriyordu.
3. **Ben ürettim:** advection sonucu host'ta `swap` ediliyor — yani host yazıyor
   — ama defter bunu duymuyordu.
4. **Ben ürettim:** `runGpuTurbulence` density/temperature/interaction'ı
   koşulsuz yüklüyordu ve combustion'dan sonra çalışıyor. Cihazdaki tek kopyayı
   bayat host kopyasıyla ezdi. f120'de 959 364 yerine **96 826** aktif hücre.

★★ 4'ün dersi ayrıca genel: **bir aşamanın yalnızca OKUDUĞU alan da o alanın
tüketicisidir.** Turbulence bu üç alanı "maske" olarak alıyordu, o yüzden
"density tüketicisi" diye okunmadı.

## 3. ★★★★★ Ders: her TÜKETİCİ bağlanmadan hiçbir ÜRETİCİ indirmeyi bırakamaz

Cihaz yerleşikliğinin ilk denemesi tam bu yüzden kaybedildi. Doğru sıra:

1. Önce **defteri üç durumlu yap** (host-only / ikisi-eşit / **cihaz-only**).
   Tek bit yalnız gereksiz *yüklemeyi* atlatır; "tek kopya cihazda" durumunu
   ifade edemediği için her aşama sonucunu geri okumak zorunda kalır.
2. Sonra **bütün tüketicileri** `gasEnsureOnDevice`'a çevir.
3. Sonra **güvenlik ağını** kur (`gasSyncGridToHost`) — kurulduğu anda no-op'tur.
4. **En son** bir üretici indirmeyi bıraksın.

★ Ve deftere yazılmayan bir üretici sessizce her şeyi bozar: MSF pyrolysis ile
sıvı yüzey yanması gaz alanlarına doğrudan cihazda yazıp geriye hiçbir şey
vermiyordu. Skaler advection deftere sorup "yükle" cevabını alıyor ve **bayat
host kopyasını depozitin üstüne yazıyordu.** Belirti: *"sıvı tutuşmuyor, auto
ignition açık, eşik 0."* Tutuşma hiç kapı değildi.

## 4. ★★★★ Ders: geçici bir A/B anahtarı, kalıcı bir ikinci kod yolu değil

`device_resident_chain` bake imzasına **kasten dahil edilmedi**, gerekçesi bir
testti: *açıkken ve kapalıyken yapılan bake AYNI olmalı.* Türbülans hatasını
yakalayan şey buydu. Görevi bitince kural 5 gereği söküldü.

**Nasıl uygulanır:** bir veri yolu değiştirirken anahtarı koy, eşitliği
kanıtla, sonra anahtarı **ve** eski yolu sök.

⚠ Ve anahtarı okumayı unutma: kullanıcı "bayrağa dokunmadım" dediğinde ben
varsayılanın açık olduğunu sandım; **kaydedilmiş proje onu kapalı tutuyordu**,
yani "sorunsuz" raporu yerleşiklik KAPALIYKEN alınmıştı. Varsayılan bir ölçüm
değildir.

## 5. ★★★ Ders: bir alanın tamamını tarayıp hiçbir şey bulmamak

Üç ayrı yerde çıktı, hepsi aynı kalıp: bir tampon **her zaman** ayrıldığı için
"var mı" sorusu `size() == cell_count` ile soruluyor, cevap hep evet, ve yoğun
yedek yol çalışıyor.

```cpp
// YANLIŞ: grid.solid her domain'de ayrılır, bu hep true
const bool has_solid = (grid.solid.size() == cells);
// DOĞRU: derlenmiş liste "voxelizer çalıştı ve hiçbir şey bulmadı" der
const bool any_solid =
    has_solid && !(grid.solid_cells_valid && grid.solid_cells.empty());
```

## 6. Kayan nokta: birebir aynılık nereye kadar beklenir

Aritmetik **taşındığında** (host → kernel) birebir aynılık beklenmez; aritmetik
**kaldırıldığında** beklenir.

- Cihaz yerleşikliği, ölü taramaların kaldırılması, clamp'in yer değiştirmesi →
  f40/f80/f120 tablosu **son haneye kadar** aynı çıktı.
- Surface dust'ın kernel'e taşınması → f120'de 959 083 / 959 364 (%0.03 fark).
  Sebep: Vulkan'da `sqrt` 2.5 ULP'ye kadar sapabilir, `std::sqrt` doğru
  yuvarlanır. Sapma **büyüyen** tipte (f40'ta 1 hücre, f80'de 11, f120'de 281)
  — bu kaotik bir sistemde yuvarlama imzasıdır. Mantık hatası olsaydı **ilk
  kareden itibaren sistematik bir yanlılık** görülürdü (ör. toz iki kez
  eklenseydi `fill` baştan yüksek çıkardı).

★ Referans tablosu bu yüzden değerli: **dört ardışık derlemede aynı kaldı** ve
her değişikliğin ne tür bir fark üretmesi gerektiğini önceden söyleyebiliyoruz.

---

## 7. SIRADAKİ İŞ — gaz, bellek (önce bu)

### 7a. ★★★★★ Önce sayaç, sonra optimizasyon

`perf.get_gpu_memory` bu sahnede:

| | |
|---|---|
| izlenen device-local | 672 MB |
| **izlenmeyen** | **1001 MB** |
| toplam VRAM | 1.67 GB / 12.1 GB |

Gaz domain'inin compute tamponları (~25 alan × 13.5 MB ≈ **340 MB**) hiçbir
kategoride görünmüyor — render tarafında `other` yalnız 16 KB device-local.
**Optimize edilecek şey izlenmeyen 1 GB'ın içinde.**

> İlk iş bir kernel değil, bir kategori: `ensureComputeBuffer` tahsislerini
> `simulation` altında muhasebeye sokmak. Aksi hâlde bu oturumda üç kez yapılan
> hata dördüncü kez yapılır — hedefi, o hedefi göstermeyen bir tablodan seçmek.

Ölçüm görünür olunca bakılacaklar (**önce ölç, sonra kes**):

- `scratch_scalar`, `scratch_scalar2`, `scratch2_vel_x|y|z` — ping-pong
  tamponları. Kaçı aynı anda canlı? Aşamalar arasında paylaşılabilir mi?
- `divergence` ve `pressure` yalnız projeksiyonun içinde yaşıyor.
- `msf_accum_*` (dört alan) — MSF kapalıyken de ayrılıyor mu?
- `gas_solid_mask`, `gas_solid_vel_x|y|z` — collider'sız domain'de ayrılıyor mu?
  (Host tarafında `grid.solid`'in her zaman ayrıldığını biliyoruz; cihaz tarafı
  da aynı kalıpta olabilir.)

★ Host RAM'de aynı soru zaten bir kez cevaplandı: `grid_memory_bytes` 125 MB ve
`FluidGrid::resize` sıvı domain'lerde `temperature`/`fuel`/`interaction`'ı
**release ediyor** (sadece `clear` değil). Aynı disiplin cihaz tarafında yok.

### 7b. Sonra performans, bu sırayla

1. **Scalar advection'ın handle takası** (19.6 ms). Hâlâ indirip host'ta `swap`
   ediyor; bu yüzden `markHostWrote` gerekiyor ve bir sonraki aşama 27 MB
   yeniden yüklüyor. Ping-pong'u cihazda yapmak bu zincirdeki son büyük
   gereksiz transferdir.
2. **Pressure projection** (45.3 ms, adımın %29'u) — **transferler kalktığına
   göre YENİDEN ÖLÇÜLMELİ.** Eski değerini iterasyon maliyeti sanmak, bu
   oturumda üç kez yapılan hatanın aynısıdır. Önce: kaç iterasyon, iterasyon
   başına kaç ms, ve ne kadarı hâlâ transfer?
3. **Yapısal geri okuma** (6.7 ms). Host çözücüsü artık boş, ama host
   *tüketicileri* ızgarayı okuyor: alan analiz taraması, `gas.measure_plume`,
   **bake/cache yazımı**, VDB dışa aktarımı. Bunlar deftere bağlanmadan geri
   okumayı kaldırmak, bake'in sessizce bayat veri yazması demektir (bkz. ders 3).
4. **Surface dust'taki senkronizasyon** (5.2 ms satırının bir kısmı). Kernel
   16 900 sütun üzerinde çalışıyor, yani mikrosaniyelik iş; maliyet 68 KB'lık
   reservoir round-trip'inin `compute->synchronize()`'ı. Düzeltme: reservoir'ı
   `gasSyncGridToHost`'un mevcut batch'ine almak. Küçük ama temiz.
5. **CPU kullanımı ölçülmedi.** Adım atarken süreç 16 çekirdeğin %26.4'ünü
   kullanıyor (boşta %0), oysa zamanlama satırlarının topladığı host işi
   ~11 ms/157 ms. Fark açıklanmadı. İlk şüpheli MSVC OpenMP'nin varsayılan
   **spin-wait**'i ve GPU fence beklemesi. Yeniden derleme gerektirmeyen test:
   `OMP_WAIT_POLICY=PASSIVE` ile başlatıp aynı ölçümü tekrarlamak.

---

## 8. FLUID (APIC) TARAFINA DEVİR

Gaz tarafında işe yarayan şey bir kernel değil, bir **yöntem**. Sırası önemli.

### Adım 1 — Ölçü aletini önce denetle

`FluidStepStats` (veya karşılığı) satırlarını **koddan** doğrula: her satırın
hangi `mark()` çağrıları arasında kaldığını, ve o aralıkta gerçekten adının
tarif ettiği işin olup olmadığını. Gaz tarafında üç satırın adı yalan
söylüyordu; fluid yolu aynı dönemde aynı ellerden çıktı.

Özellikle ara: **iki `mark()` arasında duran tam-ızgara host taramaları**
(`clampVelocity`, `sanitize*`, `enforce*`, `clear*`). Bunlar ucuz görünen
adların altında saklanır.

### Adım 2 — Taşıma mı hesap mı, ölçerek ayır

Gaz tarafındaki ayırt edici ölçüm şuydu ve fluid'de aynen tekrarlanabilir:

- **Çözünürlük ölçekleme:** hücre sayısını ~20× düşür, ns/hücre'ye bak. Sabit
  kalıyorsa iş-bound, düşüyorsa latency-bound.
- **Efektif bant genişliği:** her aşama için (taşınan bayt / süre). Cihazdan
  hiç çıkmayan bir kernel ile tur atan aşamaları karşılaştır. Gazda oran
  **125 GB/s'e karşı 4.7–7.3 GB/s** çıktı — 17–27 kat, yani donanım sınırı
  değil.

### Adım 3 — Defteri kur, sırayı bozma

Fluid'de de alan başına üç durumlu bir defter gerekiyor. **Ders 3'teki sırayı
harfiyen uygula.** Fluid'de ek zorluk: alan sayısı gazdan fazla (vel, mass,
affine, mask, weights, level set, foam) ve **partikül ↔ ızgara** iki yönlü.

⚠ Fluid'de bilinen bir tuzak zaten kayıtlı: *"scalar advection bayat host
velocity'sini taze cihaz sonucunun üzerine yazıyor"* — bu, gaz tarafında
düzelttiğimiz hatanın fluid'deki ikizi.

### Adım 4 — Bellek

Fluid domain'leri gazdan daha çok alan tutuyor ve `FluidGrid::resize` sıvı
domain'lerde combustion kanallarını release ediyor — **host tarafında** bu
disiplin var. Cihaz tarafında aynı soru sorulmadı. 7a'daki `simulation`
kategorisi kurulduğunda fluid tamponları da aynı tabloda görünecek; ikisini tek
seferde bakmak doğru olur.

### Adım 5 — Neye dokunulmayacağı

- **MGPCG portu DURAKLATILDI** (`project_fluid_vulkan_mgpcg_port`). Taşıma
  düzelmeden çözücü değiştirmek, hangi kazancın nereden geldiğini ölçülemez
  yapar.
- **Advection alt adımlama** aynı sebeple bekler: CFL 3.05 ölçüldü ama alt
  adımlama taşıma düzelmeden yapılırsa çekirdekleri değil **taşımayı** çoğaltır.

---

## 9. Çalışma protokolü (kısa)

1. Uygulamayı `scripts/ipc/Start-RayTrophi.ps1` ile sür; ölçümü IPC'den al.
2. Her fizik değişikliğinden sonra **referans tablosunu** tekrar çek
   (`gas.reset` → 120 kare → `gas.measure_plume`). Aritmetik taşındıysa ULP
   düzeyinde sapma normal, **sistematik yanlılık değil** (bkz. ders 6).
3. Partiyi `docs/dev/NEXT_BUILD_CHECKS.md`'de **sıralı** kontrol listesiyle
   bitir; her madde için "ne görmen gerek" **ve** "bozuksa ne demek" yaz, ve
   sessizce makul görünen sonucu ayrıca işaretle.
4. Build'i kullanıcı alır. Yeni `.comp` eklediysen `compile_shaders.bat` şart ve
   bunu kontrol listesinin başına yaz.
5. Push-constant ABI'si **üç yerde** yaşıyor (shader / kernel tablosu / host
   struct) ve hiçbir şey uyumu denetlemiyor — `static_assert` ile bağla.
