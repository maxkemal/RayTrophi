# Sıradaki derlemede kontrol edilecekler

> **Durum:** CANLI — 2026-09-22. Bir önceki derlemede **benim ürettiğim
> regresyon düzeltildi** (aşağıdaki 0. madde), ayrıca bir derleme hatası
> giderildi.
>
> ⚠ **Yeni shader var** — `compile_shaders.bat` çalıştırılmalı.
> Yeni `.cpp` yok, `.vcxproj` değişmedi.

---

## 0. ★★★★★ POSTMORTEM: "host okuyucusu yok" ≠ "host tüketicisi yok"

**Belirti:** küre birkaç kare büyüyüp durdu, sonra çok küçük yeniden büyüdü.
Aktif hücre 355 827 → 188 061, toplam yoğunluk 551 384 → **45 230 (12 kat az)**,
yanan hücre 192 996 → 26 055.

**Ne yapmıştım:** projeksiyonun adım başına 141 MB'lık geri okumasını tamamen
kaldırdım. Her alan için host **okuyucularını** aradım ve hepsini karşıladım:

- analiz taraması → yeni `sim_grid_velocity_max_abs` kernel'i
- `gas.measure_plume` → talep üzerine `downloadGasPressureField`
- frame cache → zaten kendi kopyasını indiriyor

**Neyi kaçırdım:** host bu alanlara **yazıyor** da. Flow-source depoziti:

```cpp
float& value = grid.velXAt(x, y, z);
value += (resolved.velocity.x - value) * blend;   // READ-MODIFY-WRITE
```

Bir read-modify-write **okuma değilmiş gibi görünür** — tüketici araması onu
bulmaz. Alan cihazda kalınca depozit bayat bir host kopyasına harmanlandı, sonra
adım başındaki `invalidateDeviceCopies()` "host otoriter" dedi ve o bayat hızı
her adım cihazdaki projeksiyon sonucunun üzerine yükledi.

★★★ **Ders, optimizasyondan değerli:** bir alanı cihazda bırakmadan önce o alana
yapılan **yazmaları** da ara. Host dizisinde `+=` görüyorsan, host o alanın
güncel değerini tutmak ZORUNDADIR.

★★ **Belirtinin şekli de ders:** çökme olmadı. Sim çalıştı, sayılar makul
göründü, yalnızca fizik yanlıştı. Bu deponun en pahalı hata sınıfı bu.

**Ne kaldı:** hız geri döndü. **Basınç ve divergence dönmedi** ve dönmeleri de
gerekmiyor — çünkü `runGpuPressureProjection` her solve öncesi ikisini de
`std::fill` ile **sıfırdan başlatıyor**, yani host'un onlara yazdığı hiçbir şey
zaten yaşamıyor. **56 MB/adım.**

---

## 1. ★★★★★ Fizik referansı — ESKİ TABLO BAYATTI, YENİSİ BU

★★★★★ **ÖNCE BUNU OKU.** Bu notta aylardır taşınan referans tablosu
**bu sahneye ait değildi** ve birkaç partidir yanlış bir şeye bakmışız.

Eski tablo `cells / fill` oranından **~3 379 900 hücrelik** bir ızgara ima
ediyor. Sahnedeki ızgara **7 054 336** hücre — **2,087 kat**. Muhtemelen preset
birleştirmesi sırasında (`quality_profile` / cell budget) çözünürlük değişti ve
tablo hiç yeniden türetilmedi.

★★★ **Nasıl anlaşıldı:** çözünürlüğe BAĞLI ve BAĞIMSIZ nicelikleri ayırarak.
Hücre sayısı ~1,9 kat kaymışken, fiziksel yükseklik ve tepe sıcaklık neredeyse
sabit kaldı:

| | cells oranı | `top` farkı | `peakT` farkı |
|---|---|---|---|
| f40 | 1,87x | −0,50% | −0,38% |
| f80 | 1,89x | −5,58% | −0,36% |
| f120 | 2,21x | 0,00% (kapak) | −0,43% |

Sayımlar ızgarayla ölçekleniyor, fizik ölçeklenmiyor. **Aynı fizik, daha ince
ızgara.**

★★ **Ders:** bir referans tablosunda mutlak hücre sayısı tutmak, tabloyu
sahnenin çözünürlüğüne bağlar. Çözünürlük değiştiği an tablo sessizce yanlış
olur ve **hâlâ makul görünür.** Çözünürlükten bağımsız nicelikler (`top`,
`peakT`, `centroid`) birincil kontrol olmalı; sayımlar yalnız ikincil.

### YENİ TABAN ÇİZGİSİ — 166x256x166 (7 054 336 hücre), `pressure_iterations=40`

```powershell
.\scripts\ipc\Probe-GasPlumeReference.ps1 -Domain 'Nuclear Gas'
```

| kare | cells | fill | top | centroid | peakT | meanT | burning |
|---|---|---|---|---|---|---|---|
| 40 | 395 790 | 0.05611 | 16.070 | 11.562 | 7.9745 | 3.76550 | 161 960 |
| 80 | 1 363 109 | 0.19323 | 30.016 | 21.835 | 5.5267 | 2.42637 | 23 147 |
| 120 | 2 118 214 | 0.30027 | 34.000 | 28.632 | 3.8275 | 1.34937 | 5 480 |

**Tekrarlanabilirlik doğrulandı:** aynı protokol iki kez koşuldu, f40 satırı
**birebir aynı** çıktı (395790 / 0.05611 / 16.070 / 7.9745 / 3.76550 / 161960).
Kaldırılan `synchronize()` çağrıları bir yarış açmadı.

⚠ **`Max speed` bağımsız bir kontrol DEĞİL:** üç karede de tam **12,75** — bu
`max_velocity` kırpma değeri, yani sayı ona *dayanıyor*. Kernel'in sıfır ya da
çöp dönmediğini kanıtlar, fiziği kanıtlamaz.

★ Panel satırı **"Active smoke cells" ile bu tablodaki `cells` AYNI ŞEY DEĞİL**:
panel eşiği 1e-4, `measure_plume` eşiği 0.01. Panelden okunan sayıyı bu tabloyla
kıyaslama.

## 2. ★★★★ `sim_grid_velocity_max_abs` — ve neden zayıf bir kontrol

Panelde `Max speed: 12,75 u/s`, `CFL: 4,00`.

⚠ **Bu sayı bağımsız bir doğrulama DEĞİL.** Üç ölçüm karesinde de tam 12,75
çıkıyor çünkü bu `max_velocity` kırpma değeri — projeksiyon sonrası sanitize
pass'i hızı oraya dayıyor. Kernel'in **sıfır ya da çöp dönmediğini** kanıtlar,
fiziği kanıtlamaz.

Gerçek doğrulaması için kırpmanın altında kalan bir kare gerekir (erken kareler)
ya da `max_velocity` geçici olarak yükseltilmeli.

**Bozuksa:**
- **`0,00`** → indirgeme yazmadı ama okuma "ölçtüm" dedi. Negatif sentinel
  (`gas_velocity_max_abs_host < 0` = ÖLÇÜLMEDİ) çalışmıyor demektir. Sıfır,
  *durmuş bir gaz* gibi okunur ve makul görünür.
- **Bir kare eski** → bayrak adım başında sıfırlanmıyor.

## 3. ★★★ `gas.measure_plume` basıncı

```powershell
Invoke-RtIpc gas.measure_plume @{ domain = 'Nuclear Gas' }
```

**Beklenen:** `pressure_measured = true`, değerler **kareden kareye değişiyor**.

★★★ Eski kapı `grid.pressure.size() == cells` idi — vektör domain boyunca
ayrılmış kalır, yani o test *içerik* değil *depolama* hakkında cevap verir.
Basınç cihazda kaldığı için o test yine geçer ve **bayat alanı
`pressure_measured = true` ile raporlardı.** Artık defter soruluyor, ve **cihaz
kopyası tercih ediliyor** — `hostHasCurrent` önce sorulsaydı, adım ortasında
gelen bir soru `grid.pressure`'daki **cold-start sıfırlarını** okurdu.

**Bozuksa:** `false` → çekme başarısız (dürüst hata). `true` ama hep aynı/sıfır
→ bayat ya da cold-start alanı okunuyor.

## 4. ★★★ Hız — ÖLÇÜLDÜ

| satır | bu iş başlarken | önceki parti | **bu parti** |
|---|---|---|---|
| `pressure projection` | 104,64 | 96,96 | 89,21 |
| `grid readback` | 17,14 | 13,01 | 14,11 |
| `field analysis scan` | 11,68 | 8,76 | **5,73** |
| **`Step total`** | **278,09** | 249,34 | **235,65** |

Toplam **−42,4 ms (−%15)**, fizik tekrarlanabilir.

⚠ **Satır satır kıyaslama tuzağı:** bir aşama senkronize etmeyi bırakınca
faturası sonrakine geçer. Bozuk derlemede `pressure projection` 8,00 ms
görünüyordu — iş kaybolmadı, `grid readback`'e (82,40) taşınmıştı.
**Yalnız `Step total`'a ve GPU kernel toplamına bak.**

```powershell
.\scripts\ipc\Probe-GpuKernelTime.ps1 -Domain 'Nuclear Gas'
```
GPU kernel toplamı **~78 ms** olmalı. Belirgin arttıysa fizik değişmiştir.

## 5. ★★★ Cache scrub'ı sadık mı

İleri oyna, önceki bir kareye scrub et, devam et. Cache anlık görüntüsünün kapısı
`gpu_resident_fields_valid`'den **deftere** çevrildi; yan kazanç olarak host'ta
zaten güncel olan alanları indirmiyor (cache'lenen kare başına ~113 MB tekrar).

**Bozuksa belirtisi çökme değil:** scrub ettiğin kareden devam edince sim *biraz
farklı* akar.

## 6. Çökme yolu

GPU projeksiyonu başarısız olursa `downloadGpuGasVelocity` hızı geri getirir ve
CPU projeksiyonu çalışır. Log'da `falling back` ara.

---

## ★★★★★ ÇÖZÜNÜRLÜKLE ÖLÇEKLEME — sıradakileri okumadan önce bu

Bu notta ölçülen her rakam **7 054 336 hücrelik** bir sahneden geliyor.
Koddaki `MAX_GRID_DOMAIN_CELLS_HARD_CAP` **134 217 728** (512³), yani bugünkü
sahnenin **19 katı**. Bu bölüm var çünkü "şu an küçük görünen kalem" ile
"önemsiz kalem" aynı şey değil ve bu ayrımı yapmayan bir öncelik listesi
çözünürlük büyüdüğünde sessizce yanlış olur — bu notun fizik referans tablosuna
olan tam olarak buydu.

### Transferler hücreyle DOĞRUSAL

| kalem | B/hücre | bugün 7,05M | 256³ = 16,8M | **512³ = 134M** |
|---|---|---|---|---|
| bu partide kaldırılan (pressure+divergence) | 8 | 56 MB / 8,6 ms | 134 MB / 20,3 ms | **1 074 MB / 163 ms** |
| kalan hız geri okuması | 12 | 85 MB / 12,8 ms | 201 MB / 30,5 ms | **1 611 MB / 244 ms** |
| projeksiyonun yüklediği SIFIRLAR | 8 | 56 MB / 8,6 ms | 134 MB / 20,3 ms | **1 074 MB / 163 ms** |

(6,6 GB/s ölçülen round-trip bant genişliğinde.)

★★ **Ve round-trip ile cihazda kalmak arasındaki oran 19 kat:** cihazdan
çıkmayan bir kernel (`sim_gas_majorant`) aynı GPU'da **125 GB/s** ölçüldü,
round-trip yapan aşamalar **4,7–7,3 GB/s**. Yani **round-trip'ten kurtardığın
her bayt, kernel işinden kurtardığın ~19 bayta bedeldir.**

### Host RAM de aynı eğride

`grid.pressure` + `grid.divergence` = 8 B/hücre ve **bu partiden sonra GPU
yolunda hiç okunmuyorlar**: bugün 56 MB, 512³'te **1,07 GB**. Aşağıdaki 1.
madde (sıfırlamayı cihaza taşımak) yapılırsa bu iki vektör **tembel** hale
gelebilir — yalnız CPU fallback'i çalışırsa ayrılır. Fizik değişmeden 1 GB.

Canlı ızgaranın kendisi 35,4 B/hücre (panel: 249,3 MB / 7,05M) → 512³'te tek
kare için **4,75 GB**.

### ★★★ ÇÖZÜCÜ İSE DAHA DİK ÖLÇEKLENİYOR — ve sıralama yer değiştiriyor

"Bugün MGPCG'nin ödül tavanı adımın %21'i" **yalnız bu çözünürlükte** doğru.

SOR'un maliyeti hücreyle doğrusal, ama **gereken süpürme sayısı domainin
DOĞRUSAL boyutuyla** büyüyor: bilgi süpürme başına bir hücre ilerler. Bugün
256 hücre yüksekliğinde 40 süpürme kullanıyoruz, yani zaten yakınsamamış —
basınç taramasında `meanT`'nin N=80'de bile tırmanması bunun kanıtı. Yakınsamayı
sabit tutmak istersen maliyet ~N⁴ gider.

166 → 512 doğrusal boyutta 3,1 kat:

- **transferler ~19 kat** (hücreyle doğrusal)
- **SOR ~90 kat** (hücre × süpürme sayısı)

⚠ **Yani öncelik sırası çözünürlüğün fonksiyonu:** bugün transfer baskın,
cap'te çözücü baskın. Bir sonraki ajan "SOR adımın %21'i, boş ver" diye okumasın
— o cümle 7M hücre için yazıldı.

Sıra yine de transferden başlıyor, iki sebeple: ölçülmüş ve bugün de pahalılar,
ve yukarıdaki 19 kat oran her bir baytı çözücü işinden daha değerli yapıyor.

---

## Sıradaki

1. **★★★ Projeksiyon her adım 8 B/hücre SIFIR yüklüyor.** `std::fill(pressure, 0)`
   + `std::fill(divergence, 0)` host'ta yapılıp ikisi de `uploadBuffer` ile
   gönderiliyor. Cihaz tarafı bir temizleme (ya da `sim_gas_divergence`'ın zaten
   her hücreyi yazdığı doğrulanırsa divergence için hiçbir şey) bunu bedavaya
   indirir. **Bu partide bulundu, bilerek yapılmadı** — bozuk bir derlemenin
   üstüne ikinci bir değişiklik koymamak için.
   ★ Devamı: bu yapılınca host `pressure`/`divergence` vektörleri tembel
   ayrılabilir (yukarıdaki RAM bölümü).
2. **Hızın 12 B/hücre'si** ancak flow-source depoziti cihaza taşınırsa gider.
   Engel bir okuyucu değil, bir **read-modify-write**: depozit host dizisinde
   `value += (target - value) * blend` yapıyor. `sim_gas_injection` bu işi zaten
   cihazda yapıyor; `injectFlowSourcesIntoGridDomains` o yola bağlanmalı.
   **Tek kalemde en büyük kaldıraç.**
3. Kernel-dışı kütleyi böl: `Probe-GpuKernelTime.ps1`'in `Unaccounted` satırı.
   Transfer mi, fence mi, submit mi — ayrılmadı. Mevcut `TransferStats` sondası
   (`synchronize_calls`, `synchronize_ms`, `batch_end_ms`) gaz adımına
   bağlanırsa tek ölçümle çıkar.
4. **MGPCG.** Bugünkü sahnede tavan adımın ~%21'i, ama yukarıdaki ölçekleme
   bölümüne bak: yüksek ızgarada baskın terim bu olur. Karar çözünürlükle
   birlikte verilmeli, tek bir ölçümle değil.
5. VRAM muhasebesine `simulation` kategorisi.

Devir notu: `docs/dev/GAZ_DERSLERI_VE_FLUID_DEVRI.md`.

## Açık kalanlar

- **Frame cache tooltip'i YALAN söylüyor.** Panel "oldest frames are dropped"
  diyor; `captureSimFrame` ise *"Refuse rather than evict"*. Kod doğru,
  **tooltip yanlış.**
- **`kSimFrameCacheBudgetBytes` = 4 GiB sabit, gaza göre boyutlandırılmış.**
  Ölçülen sıvı ~76 MiB/kare → **54 kare**. Yanındaki `resource_budget_mb`
  makineye göre ölçekleniyor, cache bütçesi ölçeklenmiyor ve IPC'de yok.
- **Panelin "~N MB grid est." satırı** 11 float/hücre (44 B) diyor. Gerçek:
  APIC sıvı 25 B, gaz 37 B, gaz+collider gas interaction 57 B — partiküller hiç
  sayılmıyor. Aynı paneli besleyen güvenlik kapısı 224 B/hücre kullanıyor.
- Panelde GPU kernel satırları yok; ölçüm script/IPC yoluyla.
- `max_velocity` ve `temperature_scale` IPC'de yok (kural 1).
- Adım atarken süreç CPU'su 16 çekirdeğin %26.4'ü; açıklanmadı.
- Kapak f110'dan sonra domain tavanında (34 m).
- `pressure_iterations`: `meanT` N ile tekdüze artıyor → **80'de bile
  yakınsamamış.**
