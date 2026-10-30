# İki gaz çakışma maliyeti: nereye gidiyor, ne yapıldı

> **Durum:** AKTİF — A+B ~112 → ~80 ms canlı; `nuke.rtp` tek hacim 414 ms; iki okuma düzeltmesi kaynakta kaldı, kazancı ölçülemedi; süre uyarlanan örneklemeye bağlı, tur farkı %25 açıklanmadı; zemin noktacıkları emisyondan, ertelendi (§8.4); §9 canlı: oran 3.2 → ~1.8; zemin eklenince sekme ışınları A+B ~250–320 ms, §11.1: kök akışa bölünme, tek akış ~131 ms, iki akış 240–280 ms; görsel kabul açık.

★ **DEVİR NOTU (2026-10-30):** Kaynakta ve kurulu: §3 ve §8.1, kazanç ölçülemedi (§8.3). Tau bütçesi (§7) geri
alındı. §6'daki "kalan penaltı neredeyse tamamen gölge" okuması YANLIŞTI, §7.1 düzeltiyor.
Sıradaki adım `NEXT_BUILD_CHECKS.md` başı: `nuke.rtp` ve iki gazlı sahnede ölçüm. Saf
march (~180 ms) hâlâ dokunulmadı. Görsel kabul (madde 4–7) yapılmadı.

## 1. Ölçüm düzeni

Aynı oturum, canlı legacy pipeline, shader SHA `DB9BAB80…`, frame 99 duraklatılmış,
1680x945, sample 8–23, iki timing + ayrı sayaç geçişi. Gaz hash'leri (`a279670b…`,
`743022518d…`, sıcaklık `01a1429f…`) her durumda aynı kaldı; malzeme ayarları
`volume.set_material` ile durum başına değişti ve geri yüklendi (`restored True`).
Kök neden için kalite düşürmek değil, maliyeti tek tek kapatmak amaçlandı.

| Durum (ikisine de) | GPU ms | kamera örneği/yol | gölge örneği/yol |
| --- | ---: | ---: | ---: |
| taban | 110–115 | 13.81 | 6.88 |
| `shadow_steps=0` (yalnız `rwSolidDistance` izi kalır) | 85–91 | 13.67 | 0 |
| `shadow_steps=0`, `shadow_stride=16` (iz ~×0.4) | 63–65 | 13.67 | 0 |
| `shadow_stride=16` | 68–72 | 13.85 | 3.84 |
| saçılma 0, σt korunmuş (`mfDirect` hiç çalışmaz) | 51–53 | 15.91 | 0 |
| aynısı + emisyon 0 (saf march) | 45.5 | 15.37 | 0 |
| emisyon 0 (saçılma açık) | 89–90 | 13.15 | 6.38 |
| `voxel_step_multiplier` 1 / 2 (`base` ×2 / ×4) | 104 / 95 | 13.07 / 12.80 | 6.20 / 4.96 |
| `adaptive_stepping=false` | 35 | 3.80 | 1.70 |
| `max_steps=1024` | 122 | 14.39 | 7.16 |

`adaptive_stepping=false` **adil karşılaştırma değil:** `primaryStep()` o durumda
sabit `step_size=0.15` döndürür (~2.2 voksel), açıkken adım 0.5 voksel. 35 ms sonucu
3.6× daha kaba örneklemedir; "uyarlanan adım maliyetin kaynağı" diye okunmamalı.

## 2. Kök neden

106 ms'lik gaz maliyetinin yaklaşık bölüşümü (eklemeli değil, ablasyon farkları):

- **~30 ms — sayılmayan iz.** Her ışık yenilemesi ve gökyüzü yenilemesi
  `rwSolidDistance` çağırıyordu; kamera ışını başına ~11 `traceRayEXT`, yol başına
  ~1.27. `walk_shadow_traces` yalnız saçılan yolda kaydedildiği için hiçbir sayaçta
  görünmüyordu. Tek gaz yolunda bu iz **hiç yok**; kaynak yorumu da bunu bilinçli
  yaptığını söylüyor. Çakışma penaltısının en büyük parçası buydu.
- **~20 ms — gölge:** kalan 6.88 gölge örneğinin tamamı gökyüzü marchı (ışık önbelleği
  warm hit, miss 0). Tek gaz yolunda `σt·dt > 0.01` kapısı var, karışımda yoktu.
- **~20 ms — sıcaklık/emisyon:** ortak yürüyüşte alan başına her adımda
  `sampleTemperature`: dünya→yerel dönüşümü + yeni accessor + 8 köşe için ağaç inişi.
- **~40 ms — march:** adım sayısı `base`'den bağımsız (×4 `base`, örnek −%7).
  Hücre çıkışı kırpması ince dilim adımlar üretiyor, emissif hücrede `base` tavanı
  ile birleşince ortalama adım tek gazın 0.5 vokselinin altına iniyor. Ayrıca
  `mfCurveSlope` tüm eğrinin en dik parçasını her hücreye uyguluyordu: canlı A eğrisi
  `[0,1,1,…]` ile 7×4×11.83 çarpanı, gerçekte eğimi **sıfır** olan doygun iç hücrelere
  de biniyor; `step_budget_exhausted` kamera ışınlarının %97'sinde tetikleniyordu.
  Kuyruk eşiği 1e-6 tek gazın 1e-3'üne göre iki kat optik derinlik yürütüyordu.

## 3. Yazılan değişiklikler

Hepsi yalnız `shadeFogMixture` yolunda; random walk, `MF_ADAPTIVE_LIMIT`, malzeme
ayarları, ABI, tampon ve otorlama yüzeyi değişmedi. C++ değişmedi.

1. `mfGeometryBlocks`: katı izi yalnız `mfWalking` iken. ★ **Görüntü değişir:**
   çakışma bölgesinde gaz artık katının ışık gölgesini almaz. Bu, aynı gazın tek
   başına davranışıdır, çakışma sınırındaki dikişi de kaldırır. Yine de bu bir
   davranış kararıdır; kabul sırasında bakılmalı.
2. Gökyüzü yenilemesine tek gaz kapısı: `lum(σt)·baseStep > 0.01`; kapanınca tutulan
   görünürlük kalır. Sahne ışığı kapılanmadı (warm cache okuması ucuz, kaçırılan
   gölgelemeye değmez).
3. Temperature için ışın başına hazırlanmış index-space ray + canlı accessor.
   Her alanın ayrı `m_vdbTempBuffers` adresi olduğundan alanlar arası paylaşım
   kanıtlanamaz, iki okuma kalır. `sampleSkyAmbient` ışın başına bir kez.
4. Adım: `mfCellCurveSlope` (hücrenin ham aralığına değen eğri parçaları, cutoff
   kapısı üstünde ×4 yok, cutoff altında sıfır), `mfRefinedSpan` (hücre kenarı yazılı
   adımın altına indiremez; yalnız optik sınır inebilir), kuyruk eşiği 0.001.

Eğri sınırının sıkılaştırılması kanıtlı: 4000 rastgele eğride sayısal oracle geçti
(`check_volume_mixture_adaptive_contracts.py`). `mfRefinedSpan` ise sınırı hücre
ötesine yazılı adım kadar taşır; orada geçerli olan güvence tek gazın sabit örgüsüyle
aynı, kanıtlı köşe sınırı değil.

Beklenen aralık, derleme öncesi: çoğu 45–60 ms; eklemeli referans 34.5 ms.
Bu ölçüm değil. ★ Toplam ablasyon farkları üst üste binebilir.

## 4. Yazılmayan

- `lightMarchAcc`'in `tauHint` adım azaltımı ilk partide taşınmadı; §7'de kullanıcı
  onayıyla eklendi.
- Çok hücreli kaba adım (doygun iç hücrelerde bir hücreden uzun) yok; emissif gazda
  `base` tavanı zaten buna izin vermiyor.
- `nanoEmptyIndexStep` için yaprak bazlı "dolu" önbelleği yazılmadı; ölçüm saf march'ın
  hâlâ baskın olduğunu gösterirse sıradaki aday.

## 5. Doğrulama durumu

Derleme gerektirmeyen hacim/gaz testleri geçti (GLSL çalıştırmazlar). Tek istisna
konu dışı: `check_gas_source_provenance_contracts.py` `SimCache.h` içinde `kVersion = 18u`
arıyor, çalışma ağacında 19; bu partiden önce var. Ham ablasyon verileri scratchpad'deydi,
tablo buradaki tek kalıcı kayıttır.

## 6. Canlı sonuç (kullanıcı derlemesi, 2026-10-30)

Shader SHA `324de5d7…`, kaynak ve kurulu aynı, legacy pipeline hazır. Açılış gaz
hash'lerini değiştirdi (`332cf454…`, `2ad612eb…`), aynı oturum içinde sabit kaldı;
görünürlük ve sahne geri yüklendi. Önceki oturumla kıyas bu yüzden kontrollü değil,
yaklaşık okunmalı.

| Görünen gaz | GPU medyan ms |
| --- | ---: |
| Gizli | 6.05 / 6.08 |
| A | 23.43 / 25.23 |
| B | 16.87 / 17.06 |
| A+B | 76.79 / 77.76 |
| A+B tekrar | 77.04 / 79.85 |

Eklemeli referans ~35.2 ms. A+B/referans oranı 3.24'ten ~2.2'ye indi; A ve B tek
başına önceki bantta kaldı. ★ Derleme öncesi 45–60 ms beklentisi **tutmadı.**

| Sayaç / yol | önce | sonra | beklenen |
| --- | ---: | ---: | --- |
| Kamera örneği | 13.81 | 10.30 | 8–10 |
| Gölge örneği | 6.88 | 4.34 | 2–3 |
| Işık önbelleği isabeti | 1.68 | 1.20 | — |
| `step_budget_exhausted` | 0.113 | 0.022 | düşük |
| Önbellek miss (warm) | 0 | 0 | 0 |

Yeni shader üzerinde ablasyon (A+B, süreler bu turda 69–78 ms oynadı):

| Durum | GPU ms |
| --- | ---: |
| taban / tekrar | 78.0, 69.1 / 71.5, 71.3 |
| `shadow_steps=0` | 44.4, 45.7 |
| `shadow_steps=0`, `shadow_stride=16` | 43.8, 42.5 |
| `shadow_stride=16` | 64.1, 58.3 |

Okuma: `shadow_steps=0` iken stride artık fark etmiyor, yani yenileme başına iz
maliyeti gerçekten gitti. ★ Bu turda yazdığım "gölgesiz A+B referansa yakın, kalan
penaltı gölge" çıkarımı **hatalı:** referans A ve B'nin kendi gölgelerini de içerir,
gölgesiz A+B ile karşılaştırılamaz. Doğru okuma §7'de.

İlk ablasyondaki ~20 ms gölge payı düşük kalmış: o tur izler hâlâ duruyordu ve
kalemler birbirini maskeliyordu. Saçılma kapalı/saf march durumları bu shader'da
ışın sayısını da değiştirdi (0.116 → 0.125), düz çıkarma için kullanılmadı.

## 7. `mfShadow` tau bütçesi (GERİ ALINDI)

Ölçülüp kazanç göstermediği için kaynaktan söküldü; aşağıdaki kayıt tekrar
denenmemesi için duruyor. `mfShadow` yazılı `shadow_steps` sayısıyla yürür.

Kullanıcı onayıyla, yalnız hazırlanmış NanoVDB yolu ve `!mfWalking` için: chord orta
noktası `τ` tahmini verir, örnek sayısı `ceil(N·clamp(√τ, 0.2, 1))`, alt sınır
`min(3, N)`, üst sınır yazılı `N`. Kural ve jitter hash'i `lightMarchAcc` ile aynı;
`mfShadow` 16 örnek tavanını almadı, yazılı değer üst sınır kaldı. Orta nokta okuması
sayaca girer ve integrale katılmaz. Azaltılan chord'da örgü hash ile kayar; yazılı
sayı kullanılıyorsa orta nokta örgüsü ve görüntü aynı kalır.

★ Bilinen sınır: orta noktası boş ama başka yerinde yoğunluk olan chord en az örneği
alır (τ=0 → 3). Tek gaz yolu da böyle. Karışımda ikinci gaz bunu daha sık üretebilir:
ince gaz orta noktada, yoğun gaz chord'un ucunda. Yoğunluk önbelleğe gitmeyen gökyüzü
ve soğuk ışık yolunu etkiler; warm sahne ışığı önbellekten okur.

### 7.1 Canlı sonuç

Shader SHA `9d896cf3…`, kaynak ve kurulu aynı. Yeni açılış gaz alanlarını yine değiştirdi
(B yoğunluk toplamı 184 → 166, yol başına kamera örneği 10.30 → 11.16), yani §6 ile
kıyas kontrollü değil.

| Görünen gaz | GPU medyan ms |
| --- | ---: |
| Gizli | 5.20 / 5.19 |
| A | 23.40 / 23.54 |
| B | 14.46 / 14.63 |
| A+B | 79.53 / 80.77 |
| A+B tekrar | 80.85 / 81.64 |

Eklemeli referans ~32.8 ms, oran ~2.4. Gölge örneği 4.34 → 2.91 (−%33), kamera örneği
+%8, ışık isabeti 1.20 → 1.41. **Süre düşmedi;** tau bütçesinin katkısı oturum gürültüsünden
ayrılamıyor (aynı oturumda aynı A+B 73–82 ms).

Aynı shader'da ayrıştırma (`shadow_strength=0` önbellek okumasını kaldırır, gökyüzü
marchı kalır; `shadow_steps=0` ikisini de kaldırır):

| Durum | GPU ms | kamera örneği | gölge örneği |
| --- | ---: | ---: | ---: |
| taban / tekrar | 72.9, 76.2 / 81.6, 80.0 | 11.16 | 2.91 |
| `shadow_strength=0` / tekrar | 56.6, 56.2 / 63.0, 63.1 | 10.08 | 2.32 |
| `shadow_steps=0` / tekrar | 50.7, 49.8 / 50.5, 48.6 | 10.07 | 0 |

Okuma: gökyüzü marchı artık ~6–13 ms, büyük kalem değil. Işık tarafı ~17–18 ms, ama bu
farkın içinde kamera örneğindeki +%11 de var; neden gölge ayarıyla değiştiği açıklanmadı.
A tek başına zaten 0.87 önbellek okuması yapıyor, karışım 1.41: bu okuma çakışmaya
özgü bir patlama değil. Gölgeler kapalıyken bile A+B net ~45 ms, referansın üstünde.
Yani kalan penaltı tek bir kaleme yığılmıyor: gölgeler ve karışım march'ının alan başına
pahalı örneği birlikte (yaklaşık 3.9 ms/örnek, tek gazda ~3).

Karar: tau bütçesi geri alındı. Örnek sayısını yazılı değerin altına indiren tek
değişiklik bu, §7'deki görsel riski taşıyor ve süre kazancı gösteremedi.

## 8. `nuke.rtp`: tek hacim, asıl büyük maliyet (canlı ölçüm)

Shader SHA `324de5d7…` (tau geri alınmış), frame 94 duraklatılmış, 1680x945, sample
8–23. `render.volume_slots` **tek slot** gösteriyor (`[Gas NanoVDB]`, 42x64x42 ışık
önbelleği), yani karışım yolu çalışmıyor; bu sahne çakışma penaltısını ölçmez. Üç
flow source'un neden ayrı akış olarak yayınlanmadığı incelenmedi. Tekrarlanabilirlik
iyi: taban 412/415, tekrar 415/414 ms.

| Durum | GPU ms | kamera örneği/yol | ışık okuması/yol | yol sayısı (M) |
| --- | ---: | ---: | ---: | ---: |
| taban | 412–415 | 36.6 | 10.3 | 10.2 |
| gaz yok (`density_multiplier=0`) | 4.6 | 0 | 0 | 10.2 |
| `shadow_steps=0` | 308 | 34.3 | 0 | 9.8 |
| emisyon 0 | 288 | 34.2 | 8.1 | 9.6 |
| saçılma 0, σt korunmuş | 223 | 37.8 | 0 | 10.4 |
| saf march (saçılma + emisyon 0) | 184 | 35.0 | 0 | 9.5 |
| adım ×2 (`voxel_step_multiplier=1.4`) | 323 | 24.3 | 6.9 | 10.4 |

Okuma: saf march ~180 ms, ışık gölgesi ~105 ms, emisyon/sıcaklık ~40–125 ms (ablasyon
yol sayısını da değiştiriyor, kesin pay değil). Örnek başına fiyat iki gazlı sahnenin
A alanının yaklaşık iki katı.

Kayıpsız adaylar (ikisi de sonuç değiştirmez, ikisi de yazılmadı):

1. `volumeLightCacheFetch` her okumada başlık, kanal, 5 tur tag hash'i ve 8 köşe okuyor.
   Işık hücresi ~0.52 m, adım ~0.09 m: bir ışın aynı hücrede ~5 adım kalıyor. Köşeleri
   ve tag'i ışın içinde saklamak okumaların çoğunu kaldırır.
2. Sıradan march'ta `sampleTemperature` hâlâ örnek başına ağaç inişi yapıyor; karışımdaki
   hazırlanmış ışın burada yok.

### 8.1 Yazılan iki düzeltme (derlenmedi)

- `volumeLightCacheFetch`: tamamlanmış okumanın 8 köşesi ve hücresi çağrı başına iki
  akışlık özel değişkende tutulur. Aynı akış aynı hücredeyse başlık, kanal, tag ve 8
  yükleme atlanır. Eşleme `(slot + kanal) & 1`: tek ışık + güneş ve bir ışık altındaki
  iki ortam birbirini atmaz. Companion + ışık + güneş gibi dört akışta eşleşen akışlar
  birbirini atar, sonuç doğru kalır, yalnız kazanç düşer. Sayaç tutulan okumayı da hit
  sayar, yani `light_cache_hits` değişmemeli.
- `volume_fog_shading.glsl`: hazırlanmış sıcaklık ışını (`mfPrepareTemperature`) sıradan
  march'a taşındı, yalnız `nanoIndexPath` ve önbelleksiz segmentte. Program veya yoğun
  gaz yolu `sampleTemperature` ile kalır. Companion'lı yolda örnek kutu dışına düşerse
  eski okuma 0, yeni okuma grid arka planı döner; birincil yoğunluk orada 0 olduğu için
  emisyon değişmez.

Sonuç aynıdır, kayan nokta yuvarlaması dışında. Kazanç sayısı verilmiyor.

### 8.2 Canlı sonuç: kazanç gösterilemedi, ölçüm kararsız

Shader SHA `83b6f3a8…`, kaynak ve kurulu aynı, volume slot yine tek ve `volume_type=2`.
Sayaçlar birebir aynı (kamera 36.606, ışık 10.283), yani iş değişmedi, beklendiği gibi.

| Durum (zaman sırasıyla) | önce ms | sonra ms |
| --- | ---: | ---: |
| taban | 412, 415 | 416.6, 429.0 |
| `shadow_steps=0` | 309, 307 | 293.9, 290.3 |
| emisyon 0 | 287, 289 | 278.2, 307.7 |
| saçılma 0 | 223, 223 | 223.8, 245.2 |
| saf march | 184, 184 | 197.9, 203.4 |
| adım ×2 | 325, 322 | 388.9, 401.8 |
| taban tekrar | 415, 414 | 525.8, 523.1 |

Önceki oturumda sapma ±2 ms'ydi. Bu oturumda aynı iş 417'den 525 ms'ye kaydı, ardından
sekiz tekrar 455–520 ms, tek tur içinde kare başına 401–569 ms oynadı. Saatler 1957 MHz
sabit, 63 °C, başka GPU yükü yok (Chrome video decode %0.8), grid'ler cihaz belleğinde.
Kayma ablasyonun ortasında başladı; nedeni bulunamadı.

Bu nedenle sonraki satırlar kanıt değildir. Kaymadan önceki tek temiz çift taban: düşüş
yok (+%1–3). Taban ile `shadow_steps=0` farkı 106'dan ~130 ms'ye çıktı, ışık okuması
ucuzlamış görünmüyor. Düzeltmeler kaynakta duruyor, kazançları KANITLANMADI. Karar için
yeni açılışta, ablasyonsuz, yalnız taban tekrarı gerekir.

### 8.3 Taze açılış: kayma ablasyondan değil, süre örnek sırasına bağlı

Yeni süreç, ablasyon yok, malzemeye dokunulmadı. Sample 8–23 medyanı on turda 419–525 ms,
iki bant: ~420 ms (~120 W) ve ~520 ms (~108 W). Saat 1957 MHz sabit, grid'ler cihazda.

Kare kare (ms, sample 0'dan itibaren, dört tur, kısaltılmış):

| Tur | ilk 4 | 4–13 | sonrası |
| --- | ---: | ---: | --- |
| 0 | ~660 | 512–575 | 420–500 arası gidip geliyor |
| 1 | ~600 | 470–575 | sample 19'da 400'e düşüyor, 373'e iniyor |
| 2 | ~600 | 515–585 | 450–510'da kalıyor |
| 3 | ~660 | 515–565 | 13'te 420, 34'te 500'e dönüyor |

Yani ölçüm penceresi 8–23 bir geçiş bölgesinin ortasında duruyor ve aynı iş aynı shader'da
turdan tura ~%25 fark ediyor. Bu, §8'deki "412–415 çok kararlı" izleniminin neden tekrar
etmediğini de gösterir: tek penceredeki medyan bu sahnede sağlam ölçü değil. Nedeni bulunamadı.
§8.1 için önce/sonra hükmü verilemez, eski turların kare dizisi saklanmadı.

Karar: iki okuma düzeltmesi kaynakta kalır. Sonucu değiştirmeden gereksiz okuma kaldırıyor ve
gerileme kanıtı yok; "ölç ki önceliklendir" kuralına göre A/B ile haklı çıkarmak gerekmiyor.
Sıradaki en büyük ipucu: aynı işin 660'tan 380 ms'ye inmesi ve %25'lik tur farkı.

### 8.4 Örnek sırası ve zemin noktacıkları

**Maliyet düşüşünün ilk kısmı uyarlanan örnekleme.** Proje `use_adaptive_sampling=true`,
eşik 0.01, ısınma 4 örnek. Örnek başına yol: sample 0–3 ~1.59 M, sample 4 sonrası ~0.65 M,
sample 50'de 0.57 M. ~660 ms bu yüzden ilk dört örneğin tam maliyeti. Bölge başına ortalama
örnek (128 üzerinden): gökyüzü 4.0, ön zemin 127.3, tepe ateşi 57.4, noktacıklı bölge 102.7.
Sonraki 520 → 380 sıçramaları bununla açıklanmaz: yol sayısı o aralıkta düzgün iniyor ve
tekrarlarda sayaç birebir aynı. Tur başına ~%25 fark nedeni hâlâ bilinmiyor.
Boşluk: uyarlanan örnekleme IPC'den yalnız okunuyor, kural 1'e aykırı.

**Noktacıklar** (`nuke.rtp`, frame 94, 128 örnek; bölge x .37–.76, y .70–.78; ölçü: 5x5
medyandan +25 gri düzeyden parlak piksel, 8-bit JPEG):

| Durum | parlak aykırı |
| --- | ---: |
| taban / tekrar | 295 / 293 |
| emisyon kapalı | 0 |
| gölge kapalı | 251 |
| saçılma kapalı | 504 |
| cutoff ×4 | 120 |
| SVGF kapalı | 537 |
| `still_noise=0` | 537 |
| `still_noise=0.005` (%100 filtrelenmiş) | 295 |
| `still_noise=0.08` | 306 |

Okuma: noktacıkları SVGF üretmiyor, emisyon üretiyor. SVGF yarısını temizliyor ama %100
filtrede bile ~295 aykırı kalıyor, yani filtre bu pikselleri reddediyor. Olası mekanizma
parlaklık kenar durdurması ve `firefly_clip`'in yalnız yeni geçmişe uygulanması; kodda
doğrulanmadı. Uyarlanan örnekleme bu ölçümde kök değil: bölgede ortalama 103 örnek var
(dağılım ölçülmedi). Emisyon kapalıyken ortalama parlaklık 102.3 → 95.1 değişiyor.
Hangi ışın sınıfının (kamera veya zemin sekmesi) getirdiği ayrılmadı.

## 9. İki gaz: adım başına tekrarlanan iş (derlenmedi)

Kullanıcı kararı: noktacıklar ve SVGF ertelendi, hedef yine çoklu gaz maliyeti. Elimizdeki
sayılar: net süre / kamera örneği A+B'de tek gazın iki katı, gölge kapalıyken ~1.45 katı.
Bu partide yalnız ortak yürüyüş (`mfSharedWalk`) değişti, sonuç değişmez:

- Alan başına birim optik terim ışın başına bir kez hesaplanır, örnekte yoğunlukla çarpılır.
  Malzeme okuması ve soğurma normalizasyonu her adımda tekrarlanmaz. Işık önbelleği okuması da
  aynı birim terimi kullanır.
- Dolu olduğu kanıtlanan yaprak hatırlanır; aynı yaprakta iki boşluk sorgusu atlanır.
  `majorant_queries` belirgin düşmeli.
- İki hacim kaydı ışın başına bir kez kopyalanır, adım başına 672 bayt kopya kalkar. Sürücünün
  bunu zaten yapıp yapmadığı bilinmiyor.

Ölçüm tuzağı: sample 8–23 medyanı `nuke.rtp`de güvenilmez çıktı. İki gazlı sahnede de uyarlanan
örnekleme açık; sabit pencere yerine kare dizisi saklanmalı. Tam karşılaştırma için gaz hash'i
aynı olmalı, yeni açılış bunu bozar.

### 9.1 Canlı sonuç: oran 3.2'den ~1.8'e indi

Shader SHA `2198c2b4…`, kaynak ve kurulu aynı, frame 99, görünürlük geri yüklendi. Yeni açılış
alanları yine değiştirdi (A aktif hücre 42175, B yoğunluk toplamı 169). `majorant_queries`
3.46 → 2.23, kamera örneği 10.98, gölge örneği 4.70, miss 0.

Kullanıcı süreci aynıyken sample 8–39 medyanı, dönüşümlü beş tur:

| Görünen | 1. tur | 2–5. turlar |
| --- | ---: | ---: |
| Gizli | 5.1 | 5.6–5.8 |
| A | 23.9 | 30.9–31.7 |
| B | 15.8 | 18.4–18.9 |
| A+B | 78.2 | 74.1–79.5 |

Matris turunda A+B 75.2 / 83.4, tekrarı 62.0 / 63.7 ms çıktı.

★ Bantlar küresel: gizli, A, B ve A+B aynı anda ~%25 yavaşlıyor. Yani §8.3'teki tur farkı
shader veya sahneye özgü değil ve mutlak süreyi bantlar arasında kıyaslamak yanlış. Doğru ölçü
aynı banttaki oran:

- Hızlı bant: A 23.9, B 15.8, gizli 5.1 → eklemeli 34.6; A+B ~63 → **~1.8×**
- Yavaş bant: A 31.3, B 18.6, gizli 5.7 → eklemeli 44.2; A+B ~79 → **~1.8×**

Başlangıç oranı 3.24×'ti (112 / 34.5). Alanlar açılıştan açılışa değiştiği için bu kontrollü bir
hız yüzdesi değildir. A+B ve A+B'nin bantlarını aynı turda eşleştirmek ölçümde yaklaşıktır:
hızlı banttaki A+B ile A/B aynı tura ait değil. Bantların nedeni bilinmiyor; saat ve ısı sabit.
Görsel kabul yapılmadı.

## 10. Zemin eklenince: sekme ışınları asıl fatura

Kullanıcı domain altına geniş küp koydu; küp domain hacmine girmiyor, A+B ~300–390 ms oldu.

| Durum | Küp yok | Küp var |
| --- | ---: | ---: |
| Gizli | 5 | 25.9 |
| A | 24 | 151.9 |
| B | 16 | 122.2 |
| A+B | 63–80 | 301–376 |

Kamera örneği (5.5) ve gölge örneği (0.8) A'da aynı kaldı. `radiance_cache_marches`
0.019 → 0.114. Bu yürüyüş yoğunluk sayacına hiç yazmıyor, yani yine sayaç körlüğü.

Ablasyon (A+B): `max_bounces=1` 129.6 ms (−172), `shadow_strength=0` ~değişmez,
`shadow_steps=0` −27, `diffuse_bounces=1` fayda yok (ilk sekme zaten kalıyor).

Kök iki maddede: `volumeRadianceCacheMarch` hücre adımıyla yürür ve her adımda 8 köşe × 4
kanal = 32 uzak okuma yapar, yoğunluk örneğinin dört katı, boş alan atlaması da yok. Sekme
ışını gazı çoğunlukla boş tarafından kesiyor. İkincisi: orta köşe `m == 0/1` diye diziliyordu,
slot sırası A, SDF, B olduğundan B her adımda tüm tag hash'lerini yeniden hesaplıyordu.

Yazıldı, derlenmedi: `vrcProvenEmpty` (sekiz yoğunluk köşesi yayınlanmış ve sıfırsa adım
atlanır, sonuç aynı) ve slot anahtarlı ortam. Boş olmayan adımın 32 okuması duruyor.

Eksik enstrüman: yürüyüş adımı sayılmıyor. Sıradaki partide sayaç şart.

### 10.1 Canlı sonuç: küçük kazanç, kök durmaya devam ediyor

Shader SHA `f73f79e9…`, kaynak ve kurulu aynı. Açılış alanları ve sayılar değişti: kamera 12.6 →
13.6, `radiance_cache_marches` A+B'de 0.141 → 0.167, yani iş %8–18 arttı. Kontrollü kıyas değil.

| Durum | Önce | Sonra |
| --- | ---: | ---: |
| Gizli | 25.9 | 31.5, 33.6 |
| A | 151.9 | 133.5, 136.0 |
| B | 122.2 | 88.9, 88.9 |
| A+B | 301–376 | 252–323 |
| A+B `max_bounces=1` | 129.6 | 136.8, 163.1 |

B'de −%27, A'da −%11, sekme yürüyüşü artmış olmasına karşın. A+B aralığı örtüşüyor, küresel
bant da burada. Sekme ışınları hâlâ ~130 ms. `radiance_cache_fallbacks` 0.

Bulgu: `render.set_volume_march_lod` `secondary_step_scale=2` yalnız tam march'a uygulanıyor;
`volumeRadianceCacheMarch` hücre adımıyla yürüyor ve bu ayarı yok sayıyor. Sıradaki aday bu,
ama adımı uzatır: kalite kararı.

### 10.2 Radyans yürüyüşü sekme LOD'una uyuyor (derlenmedi)

Kullanıcı onayıyla `volumeRadianceCacheMarch` adımı hücre × `secondary_step_scale`, adım sayısı
`secondary_max_steps` ile sınırlı (varsayılan 2 ve 128). Görüntüyü değiştirebilir: sekmenin gördüğü
gaz daha kabadır. Kalite kararı, kayıpsız değil. Geri almak için `render.set_volume_march_lod`
ile `secondary_step_scale=1`; derleme gerekmez, aynı süreçte A/B yapılabilir.

### 10.3 Aynı süreçte temiz A/B: yürüyüş adımı faturanın küçük kısmı

Shader SHA `0214b5b7…`, tek süreç, aynı alanlar, A+B, dönüşümlü dört tur (ilk tur ısınma):

| `secondary_step_scale` | GPU ms | 1'e göre görüntü farkı (128 örnek, 8-bit JPEG) |
| --- | ---: | --- |
| 1 | 271.6, 271.4, 271.9 | referans |
| 2 (varsayılan) | 238.7, 239.0, 239.9 | ort. 0.021, p99 1, en çok 14 |
| 4 | 223.9, 224.7, 223.8 | ort. 0.033, p99 1, en çok 27 |

Tekrar ±1 ms. `a + b/k` modeli 1 ve 2'den b≈65, a≈206 verir, k=4 için 222.7 öngörür, ölçülen 224.
Yani adım sayısıyla ölçeklenen yürüyüş tüm kare içinde ~65 ms; ölçeklenmeyen 206 ms kalıyor. Gaz
gizliyken ~32, `max_bounces=1` ~137–163 ms. Dolayısıyla sekme faturasının ~yarısı adımda değil,
ışın başına sabit işte: kapalı kutu girişi, solid probe, SDF arbiter, `vrcMediumSetup` (ışın
başına 2 ortam × 4 tag + malzeme ve ışık hash'i). Bu dağılım ölçülmedi, sıradaki iş o.
Varsayılan 2 kaldı; görüntü farkı JPEG üzerinde, küp yansıması gözle kontrol edilmedi.

### 10.4 Sabit iş: gaz-SDF devri büyük kalem

Aynı süreç, A+B, küplü sahne. Sıvı faz çıkışı `fluid.set_phase_output liquid=hidden` ile
kapatılıp dönüşümlü dört tur ölçüldü (görüntüyü değiştirir, yalnız ablasyon, geri yüklendi):

| | `follow` (sıvı SDF açık) | `hidden` |
| --- | ---: | ---: |
| GPU ms | 281.5, 236.5, 296.6, 282.6 | 233.2, 224.6, 196.2, 214.4 |
| `gas_handoffs`/yol | 0.097 | 0 |
| `radiance_cache_marches` | 0.168 | 0.096 |
| `solid_probe_runs` | 0.394 | 0.236 |
| `sdf_arbiter_samples` | 5.07 | 0 |
| `volume_rays` | 0.309 | 0.236 |

Medyan ~282 → ~219 ms, ~%22. Slot listesi `hidden` iken de üç satır gösterdi; slot kalıyor, yalnız
aktifliği düşüyor olabilir, doğrulanmadı. Sıvı yüzey varsa bu farkın bir kısmı gerçek iştir.

Okunan yol: her gaz girişi önce `nearestSurfaceSDFCrossing` (sorgu başına ~7 örnek), sonra solid
probe izi, sonra önbellekli yürüyüş. Devir ışını yeniden izler, aynı giriş baştan çalışır.
`hasCompanion` iken sekme ışınları solid probe kapısını da atlıyor (`!hasCompanion` koşulu).
Bariz kayıpsız hata bulunmadı. Sıradaki adaylar: companion'lı sekmede probe kapısı (kalite riski
var, kapı yanlışsa yüzey kaybolur), `vrcMediumSetup` hash'lerinin kare başına hesaplanması (C++).
Bu turda kod değişmedi.

## 11. Tek akış testi (geçici sonda, derlenmedi)

Kaynak takibini kapatan ayar yok. `MatterFogSnapshot.cpp` içinde `RT_GAS_MERGE_STREAMS` ortam
değişkeni açıkken gaz tek akış yayınlanır: A malzemesi tüm toplam gaza uygulanır, görüntü değişir,
yalnız maliyet testi içindir. Ölçümden sonra SÖK. Soru: aynı toplam gaz tek hacimde ne kadar tutuyor?

### 11.1 Sonuç: maliyetin kökü akışa bölünme

Küplü sahne, frame 99, aynı yayın kodu, `RT_GAS_MERGE_STREAMS=1` ile tek hacim. Slotlar: gaz + SDF.
Yoğunluk toplamı 783.8 (A+B toplamı), aktif hücre 37379.

| | Tek akış | İki akış (önceki oturumlar) | Yalnız A |
| --- | ---: | ---: | ---: |
| GPU ms | 118.7–152.9, medyan ~131 | 240–280 | 133–136 |
| kamera örneği/yol | 6.18 | 12.6–13.7 | 6.10 |
| gölge örneği/yol | 1.04 | 6.1–6.6 | 0.98 |
| `radiance_cache_marches` | 0.132 | 0.168 | 0.134 |
| `gas_handoffs` | 0.002 | 0.097 | 0.002 |

Aynı toplam gaz tek akışta yalnız A kadar tutuyor, iki akışa bölününce ~2 kat. Sıvı SDF slotu tek
akışta da duruyor ama devirler 0.097 → 0.002 iniyor: devirler sıvıdan değil, karışım yolundan
geliyor. §10.4'teki "gaz-SDF devri" okuması bu yüzden yanıltıcıydı, kullanıcı haklıydı.

Sınır: A malzemesi tüm gaza uygulandı, görüntü aynı değil; iki akış oturumları ayrı açılış.
Sonda kaynaktan söküldü; kurulu exe'de hâlâ var, değişken verilmezse etkisiz.
Sıradaki karar: render'da tek birleşik yoğunluk + akış oranı.
