# Faz 2 — Photon Caustic Pass (Vulkan RT öncelikli)

Hedef: cam/su gibi transmissive yüzeylerden kırılan ışığın **gerçek konumlu, odaklanmış
caustic desenlerini** diffuse alıcı yüzeylere düşürmek; dispersiyon açıkken desenin
zeminde gerçek renk ayrışımı yapması. Aynı altyapı offline (progressive, çok foton)
ve ileride realtime (az foton + temporal birikim + denoise) modda çalışacak —
MNEE yerine photon seçilmesinin nedeni bu ikili kullanım.

Faz 1 (tamamlandı): stokastik hero-channel dispersiyon (yalnız kırılan lob),
renkli parçalı cam gölgesi (Fresnel + bend heuristiği, kanal-başına saçak).
Bu heuristik, photon pass gelene kadar fallback; geldiğinde kısmen devre dışı
kalacak (bkz. Dilim 3'teki çift sayım notu).

---

## Mimari özet

```
Işık kaynağı ──▶ photon.rgen (ışıktan iz)
                   │  mevcut TLAS + material/geometry SSBO'ları (set 0 paylaşılır)
                   ▼
              cam/su yüzeyi → kırıl (scatterGlass'ın foton yönü; kanal-başına IOR)
                   │  Beer-Lambert absorpsiyon → renkli caustic
                   ▼
              DIFFUSE yüzeye ilk temas → world-space HASH GRID'e atomik splat
                                              │
closesthit.rchit (kamera path) ──────────────▶ gather: grid'den yoğunluk oku,
                                              radiance += density * albedo/π
```

- **Foton = sadece caustic taşıyıcısı.** En az bir transmissive arayüz geçmemiş
  fotonlar splat edilmez (LS⁺D yolları). Normal direct/indirect aydınlatma path
  tracer'da kalır → çift sayım yok.
- **Hash grid:** world-anchored uniform grid, sabit boyutlu hash tablosu SSBO.
  Hücre başına RGB enerji, fixed-point uint atomicAdd (radiance × 1024).
  Varsayılan: 2²² hücre × 16 B ≈ 64 MB (parametrik). Hücre boyu: caustic detay
  ölçeği, varsayılan sahne yarıçapı/1024, UI'dan ayarlanabilir.
- **Descriptor:** mevcut set 0 aynen bağlanır + yeni binding: photon grid SSBO.
  Vulkan compute descriptor pool 512 sınırına dikkat (mevcut batching kuralı).
- **Progressive senkron:** foton pass her accumulation frame'inde çalışır;
  `resetAccumulation` tetiklenince grid de temizlenir. Post-process accum reset
  YASAĞI kuralı burada da geçerli (grid temizliği yalnız gerçek reset'te).

---

## Dilimler (her biri tek başına derlenip test edilir)

### Dilim 1 — Photon trace + debug görselleştirme (uçtan uca iskelet)
**İş:**
- `photon.rgen` + `photon.rchit` (+ mevcut shadow/opacity anyhit'leri paylaşılmaz;
  fotonlar opacity cutout için basit stokastik geçiş yapar).
- Yeni küçük RT pipeline + SBT; VulkanBackend'de `renderPhotonPass()` girişi.
- Işık örnekleme (ilk dilimde: tek directional/point/area — sahnedeki ilk ışık).
- Foton payload: `power(rgb), throughput, bounceCount, crossedGlass(flag), channel`.
- Cam kırılması: scatterGlass'ın foton-yönlü kopyası (Fresnel rus ruleti, TIR,
  Beer-Lambert tint). Diffuse yüzeye ilk temasta `crossedGlass` ise grid'e splat, path biter.
- Hash grid SSBO + clear compute'u.
- **Debug görselleştirme:** raygen'de bir debug modu (uniform flag) — primary hit'in
  grid hücresindeki enerjiyi doğrudan renk olarak bas. Gather entegrasyonundan ÖNCE
  fotonların doğru konuma düştüğü gözle doğrulanır.
**Test:** cam küre + zemin + tek ışık → debug modda kürenin altında odak lekesi
(parlak nokta) görünmeli; küre kaldırılınca kaybolmalı.
**Bütçe:** frame başına 64k–256k foton, tek bounce zinciri ≤ 8.

### Dilim 2 — Gather + shading entegrasyonu
**İş:**
- `closesthit.rchit` NEE bloğuna caustic gather: diffuse (metallic<0.5,
  transmission≈0) yüzeylerde grid yoğunluğu oku → `radiance += E * albedo / π`.
- Density estimation: 3×3×3 komşu hücre toplamı / kernel hacmi; yumuşatma
  yarıçapı = hücre boyu × k (k≈1.5).
- Enerji normalizasyonu: foton gücü = ışık gücü / foton sayısı; birim doğrulama
  için "beyaz zemin + cam yok" sahnesinde caustic katkısı ≈ 0 olmalı.
**Test:** debug modu kapalı, gerçek render'da caustic lekesi; foton sayısı
2× artınca desen aynı parlaklıkta kalmalı (normalizasyon doğru), noise azalmalı.

### Dilim 3 — Çift sayım dengesi + kalite
**İş:**
- **Çift sayım:** Faz 1'in renkli geçirgen gölgesi transmisyon ışığını zaten
  yaklaşık ekliyor. Photon gather aktifken (`caustic_enabled` uniform):
  shadow anyhit'teki geçirgen katkı KAPATILIR (cam gölgesi tam koyulaşır),
  aydınlatmayı fotonlar getirir — fiziksel olarak doğru dağılım.
  Kapalıyken Faz 1 heuristiği aynen çalışır (fallback).
- Projection map / importance: foton bütçesini transmissive objelerin
  AABB'lerine doğru koni örnekleme ile yönlendir (boşa giden foton ↓↓).
- Rus ruleti, maks bounce, ışık başına bütçe paylaşımı (çoklu ışık).
**Test:** cam altı gölge: photon açıkken parlak odak + koyu kenar (gerçek dağılım),
kapalıyken Faz 1 görünümü; enerji patlaması/sönmesi yok.

### Dilim 4 — Dispersiyonlu caustic
**İş:** foton izinde kırılma anında hero-channel seçimi (mevcut mantığın kopyası,
`dispersion>0` ise) → foton tek kanal enerji taşır, kanal IOR'uyla bükülür.
Grid RGB olduğundan ekstra yapı gerekmez.
**Test:** Dispersion 0.7 cam küre → zemindeki odak lekesinin kenarında gerçek
konumlu gökkuşağı ayrışımı.

### Dilim 5 — Dinamik sahne + realtime yolu
**İş:**
- Her frame foton yeniden izlenir (TLAS zaten dinamik; su/fluid yüzeyi BLAS
  güncellemeleri mevcut). Grid: her frame clear + yeniden doldur; realtime modda
  temporal decay (grid *= α, α≈0.9) + düşük bütçe (16k–64k).
- Render Settings UI: foton sayısı, hücre boyu, caustic on/off, temporal α.
- OIDN etkileşimi: caustic katkısı denoise ÖNCESİ radiance'a girer (ekstra AOV yok).
**Test:** timeline'da hareket eden cam obje → caustic frame'e oturur, ghosting
temporal α ile kontrol edilir; viewport'ta akıcı kalır.

### Dilim 6 (opsiyonel) — OptiX paritesi
Aynı grid'i CUDA'da doldur (ışık raygen OptiX'te de var olan TLAS'ı kullanır)
veya OptiX render'da Vulkan çıktısı beklenmez — bağımsız port. CPU: düşük öncelik
(offline'da bile pahalı; gerekirse embree ile aynı şema).

---

## Riskler / bilinen ayar noktaları
- **Light leak:** grid hücresi duvarın iki yanını ayıramaz → normal-ağırlıklı splat
  (foton normali ile alıcı normali dot > 0 şartı) Dilim 2'de baştan konur.
- **Enerji birimi:** ışık intensity tanımları backend'ler arasında farklıysa
  normalizasyon sabitini Vulkan NEE ile A/B karşılaştırarak kalibre et.
- **Bellek:** 64 MB grid varsayılan; büyük sahnede hücre boyu otomatik büyür
  (kamera odaklı kaskad ileride — şimdilik tek seviye).
- **Perf:** foton pass ayrı command buffer + timestamp query ile ölçülür;
  bütçe UI'dan kısılabilir. Descriptor pool 512 batching kuralı geçerli.
- **Temizlik:** proje reload'da grid buffer'ları `resetForProjectReload`
  zincirine eklenmeli.

---

# FAZ 2V — VOLUMETRİK CAUSTIC (participating media içinde ışık kolonları)

**Hedef:** Cam/su tarafından odaklanan ışığın sis/toz içinde GÖRÜNÜR huzme ve
kolonlar oluşturması (ör. camdan kırılan güneşin sisli odada gökkuşağı şaftları,
su yüzeyinin havuz içinde dans eden ışık perdeleri). Klasik yöntem: volumetric
photon mapping (Jensen & Christensen) — fotonlar yüzeye değil UÇUŞ YOLU boyunca
ortama da enerji bırakır; kamera ışını ortamda yürürken bu enerjiyi toplar.

**Neden altyapı hazır:** foton yürüyüşü zaten segment segment ilerliyor
(photon.rgen bounce döngüsü); kamera tarafında god-ray/height-fog march'ı zaten
var (raygen computePrimaryGodRays + fog bloğu); HG faz fonksiyonu closesthit'te
mevcut; dispersion fotonda taşındığı için huzmeler bedavaya RENKLİ olur.

**Kritik ayrım — çift sayım yok:** mevcut god-ray/fog yalnız DOĞRUDAN güneş/ışık
katkısını hesaplar. Volume grid'e yalnız `crossedGlass` fotonlar yazılır → eklenen
katkı sadece LS⁺(medium) yolları = path tracer'ın hiç bulamadığı ışık. Yüzey
grid'iyle de çakışmaz (ayrı estimator boyutu: alan vs hacim).

### Dilim V1 — Volume deposit + debug viz
- İkinci grid: binding 20 SSBO (aynı 5-uint hücre düzeni, ayrı header; ~20MB;
  hücre boyu yüzey grid'inden 2-4× KABA — hacim yumuşak, bellek/perf dostu).
- photon.rgen: her segment için, global fog yoğunluğu > 0 ise segment üzerinde
  stokastik k nokta örnekle (transmittance ağırlıklı, k=1-2 yeter — pass birikimi
  varyansı eritir), `power × T(t) × σs` ile volume grid'e splat. Yalnız
  crossedGlass sonrası segmentler.
- Debug mode 3: primary ışın boyunca 16 adım march, volume grid yoğunluğunu
  görselleştir (huzme yerleşimi doğrulama — yüzey Dilim 1'in birebir karşılığı).
- Normalizasyon: hacim kernel'i — ∫(1-d/R)dV = πR³/3 (yüzeydeki πR²/3'ün 3B'si).

### Dilim V2 — Kamera in-scatter entegrasyonu
- raygen: primary segment march'ında (god-ray döngüsü kadansı) her adımda
  `L += T(t) · σs · HG(cosθ) · E_vol(x) · Δt`; passes normalizasyonu yüzeyle aynı.
- Perf kalemi: adım başına tek-hücre okuma (27-komşu gather yerine) — hacim
  zaten yumuşak; gerekirse 8-komşu trilinear V3 cilası.
- Enerji: fog color/density UI'siyle uyum; Energy knob'u ortak.

### Dilim V3 — Lokal hacimler (VDB/gas) + kalite
- Foton segmentinde VDB AABB kesişimi → yoğunluğu volume'dan örnekleyerek
  deposit (fotonlar mask 0x01 kullanıyor; hacimler ışına GÖRÜNMEZ kalır, sadece
  segment-AABB testi + density lookup — TLAS değişikliği gerekmez).
- Normal-ağırlıklı/anizotropik kernel, trilinear okuma, ışık şaftı keskinlik ayarı.

### Dilim V4 — Perf/realtime + parite
- Adım sayısı/foton bütçesi UI; temporal decay (Dilim 5 ile ortak);
  CPU paritesi CPU-photon parkı açıldığında birlikte.

**Riskler:** (a) march başına grid okuması ana maliyet — adım sayısı ve tek-hücre
okumayla sınırlandı; (b) kaba hücre + parlak huzme = bloklu şaft riski → V2'de
jitter'lı march + birikim yumuşatır; (c) fog'suz sahnede sıfır maliyet (deposit
ve march koşulları fog yoğunluğuna kapılı).
