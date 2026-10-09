# Sparse 1. aşama — sıvı Vulkan hattında cihaz MAC hızı yalnız tile sayfalarında

> **Durum:** AKTİF — 2026-10-09. Diğer ajanın devrinden (MATTER_SPARSE_FULL_STORAGE_IMPLEMENTATION.md)
> kullanıcı kararıyla **aşamalı** sürdürülüyor: her aşama kendi build'i ve IPC testiyle kapanır.

## Aşamalar (kullanıcı kararı 2026-10-09)

1. **S1 (bu not):** sıvı Vulkan hattı (saf sıvı + karışık Matter'ın sıvı şeridi),
   periodic değil, `use_sparse_tiles` açık. Cihazdaki kanonik MAC hızı, FLIP
   tabanı ve P2G ağırlıkları yalnız compact tile sayfalarında yaşar; yoğun cihaz
   hız/ağırlık/FLIP bankları bu modda ayrılmaz. Host `FluidGrid` yoğun kalır
   (kare sonunda sayfalar indirilip host'a saçılır) — host/render/cache S2.
2. **S2:** sıvı host/render/cache sözleşmesi (`FluidGrid` yoğun dizileri yerine
   storage servisi, snapshot/export).
3. **S3:** gaz (grid-domain gaz yolu: `ParticleSimulation.cpp` + `sim_gas_*`).
   **`GasSimulator.cpp` ölü alan — kapsam dışı, dokunulmaz** (kullanıcı notu).

## Kapsam kararı: önce karışık Matter'ın sıvı şeridi

Okuma sonucu (2026-10-09): **saf sıvı Vulkan yolu cihazda kalıcı değil.**
`FluidDomainStep.inl` "Call 1" ile CPU'da `Fluid::step` koşturuyor (sınır + FLIP
anlık görüntüsü), viskozite/basınç sonrası host'ta `enforceGridSolidFaceBoundaries`
uyguluyor, basınç aşaması hızı host'tan yükleyip geri indiriyor. Orada cihaz hızını
compact yapmak önce yolun cihazda kalıcı olmasını gerektirir → **S1.5**.

Karışık Matter yolu (`MatterGpuStep.inl`) zaten kalıcı: `matter_model.enabled`
yüklemeleri/indirmeleri atlıyor, yalnız kare sonunda tek yayın var. Mevcut sparse
basınç kabulünün Su/Toprak düzeneği de bu yoldan geçer. **S1 = karışık yolun sıvı
şeridi (lane 0).** Granüler şerit (lane 1) yoğun kalır; temas kernel'i lane 0'ı
compact, lane 1'i yoğun okur.

İki ayrı tile haritası kuralı var ve S1'de birleştirilmiyor: basınç haritası
`slot` (boş = `0xffffffff`, hücre tile'ı 512), MAC haritası `slot + 1` (boş = 0,
yüz tile'ı 576). Hücre alanları (maske, diverjans, basınç yayını, yüz ağırlıkları,
katı hızı) S1'de yoğun kalır — **S1b**.

## S1 envanteri: cihaz hızına dokunan sıvı tüketicileri

| Tüketici | Bugün | S1'de |
|---|---|---|
| P2G (`sim_sparse_mac_p2g/normalize`) | compact, sonra yoğuna `publish` | compact kalır, `publish` yok |
| FLIP tabanı (`sim_sparse_mac_capture`) | yoğundan compact'a kopya | compact hızdan compact'a sayfa kopyası |
| Katı yüz sıfırlama (`sim_fluid_zero_solid_faces`) | yoğun | ortak gövde + sparse giriş |
| Viskozite (`sim_sparse_viscosity_*`) | RHS havuzu sparse, hız yoğun | hız compact |
| Diverjans (`sim_fluid_divergence`, `_var`, `_porous`) | yoğun | ortak gövde + sparse giriş |
| Gradyan çıkarma (`sim_fluid_subtract_gradient`, `_var`) | yoğun | ortak gövde + sparse giriş |
| GFM serbest yüzey basıncı | yoğun | **S1'de yok**: GFM açıkken yol yoğun kalır ve bunu söyler |
| Karışık temas (`sim_matter_contact`) | iki şerit yoğun | sıvı şeridi compact, granüler şerit yoğun |
| G2P (`sim_sparse_mac_g2p`, `_matter_g2p`) | yoğundan compact'a toplama, sonra okuma | doğrudan compact hızdan |
| `advect_tail` | yoğun hız örnekler | ortak gövde + sparse giriş |
| Köpük ölçütleri (`sim_foam_crit`) | yoğun | ortak gövde + sparse giriş |
| Kare sonu host yayını / cache | yoğun indirme | sayfa + harita indirme, host'a saçma |

## Anlam eşitliği

Yoğun yolda parçacık quadratic desteğinin dışındaki MAC yüzleri sıfırdır (P2G
ağırlığı yok, basınç gradyanı yalnız sıvı hücrelerine komşu yüze yazar). Sparse
arka plan değeri de 0 olduğundan okumalar aynı sonucu verir. **Tek fark:** desteğin
dışında kalan hareketli katı yüzlerine katı hızının yazılması — sparse'ta sayfa
yoksa yazılmaz. Bu yüzü yalnız destek dışına çıkan `advect_tail` örneklemesi
okur; etkisi parçacıktan en az bir tile uzaktaki hareketli katılarla sınırlıdır.
Kabulde hareketli duvar sahnesi bunu ölçer.

## Yapılan (2026-10-09, kaynakta; derlenmedi)

- **Ortak sözleşme:** `sim_mac_lane.glsl` — `macAddress(comp, i, j, k)` (yoğun: eski
  indeks formülü, compact: harita + sayfa; yoksa `MAC_ABSENT`, okuma 0 / yazma atlanır),
  `macLaneFace` (yoğun: eski yüz çözümlemesi; compact: etkin tile sayfa değerleri,
  sahiplik `sparseMacOwned`). Yoğun ikizler aynı gövdeden derlenir, aritmetik değişmedi.
- **13 compact giriş** (`sim_sparse_mac_*`): zero_faces (+matter), divergence(+var,
  +porous), subtract_gradient(+var), capture_compact, advect (+matter), matter_contact,
  viscosity_capture/sweep. Gövdesi `.comp`'tan `.glsl`'e taşınanlar: divergence (3),
  subtract_gradient (2), matter_contact; `.comp` artık sarmalayıcı (pencere varyantları
  `-DRT_FLUID_WINDOW` ile aynı dosyadan).
- **Host:** `SparseMacTransferGpuStorage::compact_owner` (kalıcı) ve `canonical`
  (alt adım). Karışık adım sahipliğe ensure'dan önce karar verir; ayırıcı sahip için
  yoğun hız + FLIP bankasını ayırmaz/serbest bırakır (eski handle tespiti artık
  `pressure` üzerinden). Compact P2G başarısızsa sayfalar bırakılır, `compact_blocked`
  kalır (sparse kapatılıp açılana kadar), yoğun banka ayrılıp yoğun P2G koşar.
  Saf sıvı yolu ensure'dan önce sahipliği indirir. Basınç compact hızla yalnız sparse
  basınç çözümünü kabul eder; yoğun CG'ye düşmez, `bail` eder.
- **Kare sonu:** `publishCompactMacToHost` — harita listesi + 3 sayfa indirilir, host
  yoğun MAC'e saçılır (sahip olmayan dolgu atlanır, sayfasız yüz 0).
- **Yüzey:** `transfer_sparse_canonical`, `transfer_sparse_blocked` — core istatistik →
  `fluid.step_stats` IPC, Python, panel ("MAC storage: Compact canonical" / "Dense
  (compact blocked: neden)").
- **Kontrat:** `check_sparse_mac_canonical_contracts.py` (27 girişin bağlama/push ABI'si
  açılmış kaynaktan, host bağlantısı, bağımsız adres/kapsam kâhini) PASS; etkilenen
  eski kontratlar dosya taşımasına göre güncellendi, hepsi PASS.

## Hâlâ yoğun (S1b ve sonrası) — dürüst muhasebe

S1 yalnız **6 yüz boyutlu banka** kaldırır (hız ×3, FLIP ×3). Sıvı şeridinde hâlâ
yoğun: P2G ağırlık takma adları `temperature/fuel/scratch_scalar` (yüz boyutlu ×3; artık
yazılmıyor ama gaz kanalı olarak ayrılı), `var_u/v/w_weight` (yüz ×3), `scratch_scalar2`,
`substance_viscosity` (yüz boyutlu), hücre alanları (maske, diverjans, basınç yayını,
katı hızı ×3), karışık kütle gradyanı (yüz ×3 ×2 şerit). Yani VRAM kazancı tahminen yüz
bankalarının yarısından azdır; `transfer_sparse_resident_bytes` sayfa havuzu ek bellektir.
Ölçüm build sonrası.

## Devir — sıradaki ajan buradan başlar

**Durum (2026-10-09):** S1 kaynakta tamam; kullanıcı build alıyor. Aynı build'de DEM
Parti 8–10 da var. Sıralı test listesi `NEXT_BUILD_CHECKS.md` → "Sparse S1" (DEM partileri
ayrı başlıklarda). **Yeni aşamaya S1 build + IPC testi geçmeden başlama** (kullanıcı kararı).

1. **Önce S1 sonucunu oku.** Kabul: yoğun yol regresyonu yok (sparse kapalı);
   `rt_test_sparse_pressure_ipc.py --transfer` ve `--transfer --viscosity` PASS,
   `transfer_sparse_canonical=true`, `transfer_sparse_blocked=false`. Blocked çıkarsa
   neden `transfer_sparse_status`'ta; ilk bakılacak yer `runSparseMacP2G` hata dönüşleri ve
   `MatterGpuStep.inl`'deki ensure → yeniden deneme bloğu.
2. **S1b (sıradaki iş):** "Hâlâ yoğun" listesindeki yüz/hücre bankaları. En büyük kalem
   P2G ağırlık takma adları (`temperature/fuel/scratch_scalar`) ve `var_*_weight`; compact
   sahipte ayrılmamalı. Aynı ayırıcı kapısı: `releaseDenseMacForCompactOwner`
   (`ParticleSimulation.cpp` ensure). Hücre alanları için basınç haritası (`slot`, 512) kuralı.
3. **S1.5:** saf sıvı yolu (`FluidDomainStep.inl`) önce cihazda kalıcı hale gelmeli (Call 1
   CPU adımı, host katı-yüz kelepçesi, basınçta yükle/indir). O zamana kadar saf yol
   `compact_owner=false` tutar — bu iki satırı kaldırma.
4. **S2 / S3:** yukarıdaki aşama listesi. Diğer ajanın ortak storage çekirdeği
   (`SparseGridStorage`, `SparseGridGpu`) henüz hiçbir tüketiciye bağlı değil; S2'de host
   tarafının adayı o. **`GasSimulator.cpp` ölü — dokunma.**

Kurallar: build/glslc yok; yeni `.comp` → `compile_sim_shaders.bat` + `SimulationComputeVulkan.cpp`
kaydı + `check_sparse_mac_canonical_contracts.py` tablosu; script'ler iki yere kopyalanır.

## S1b ilk parti — P2G ağırlık bankaları (2026-10-09, kaynakta; derlenmedi)

Kullanıcı S1/DEM testleri sürerken devir sırasına göre kod yazılmasına izin verdi.
Bu izin kaynak çalışması içindir; S1'in canlı kabulü henüz bu notta kapanmış değildir.
Yukarıdaki S1 envanteri ve "Hâlâ yoğun" muhasebesi test edilen S1 build'ini anlatır.

- `releaseDenseMacForCompactOwner` artık hız ×3 + FLIP ×3 yanında P2G ağırlık
  takma adlarını (`temperature/fuel/scratch_scalar`) da serbest bırakır.
- Aynı `dense_mac` kapısı bu üç bankanın ayrılmasını ve zorunlu-handle kontrolünü
  yönetir. Yoğun, granüler ve saf sıvı yollarının bankaları korunur. Compact P2G
  başarısızlığı sahipliği düşürür; mevcut `ensure(primary)` → P2G yeniden denemesi
  dokuz yoğun bankayı tekrar ayırır. Page-only descriptor'larda boş yoğun ağırlık
  slotu ilgili compact ağırlık sayfasını bağlar; shader/ABI değişmedi.
- Ek kaldırılan bellek: `3 * max(face_count) * sizeof(float)`; örneğin 40³ hücrede
  787200 B. Compact ağırlık sayfaları zaten S1'de vardı; bu parti yeni havuz eklemez.
- `var_*_weight` katı yüz ağırlıklarıdır, P2G ağırlıkları değildir. Basınç/diverjans/
  gradyan tüketicileri taşınmadan kaldırılamaz. **S1b bütünü tamam değil:** bu üç
  banka ve hücre alanları sıradaki parti; MAC haritası `slot+1/576`, hücre haritası
  `slot/512` ayrımı korunacak. S1.5/S2/S3'e geçilmedi.
- Kaynak doğrulaması: canonical, transfer, dispatch, window, Matter, viscosity ve
  pressure kontratları PASS. Canonical kontratına ağırlık ayırıcı kapısı, descriptor
  alias'ı ve başarısız compact aktarımın yoğun yeniden deneme kontrolleri eklendi.
  Build veya canlı IPC çalıştırılmadı; kullanıcıdaki devam eden testler etkilenmedi.

## S1b ikinci parti — katı yüz ağırlıkları (2026-10-09, kaynakta; derlenmedi)

Kullanıcının "devam edelim, testler sürüyor" izniyle ilk partinin ardından yazıldı.
İlk partideki "var_* hâlâ yoğun" durumu bu kaynak partisiyle değişti; çalışan eski
S1 binary'si ve onun devam eden kabul testleri ayrı kalır.

- Compact sahipte `var_u/v/w_weight` ayrılmaz; aynı ayırıcı artık toplam 12 yoğun
  yüz bankasını bırakır. Yoğun/saf sıvı/gaz/granüler bankaları eski sözleşmesini
  korur. P2G başarısızlığı mevcut yoğun yeniden deneme yolundan bu bankaları da kurar.
- Compact MAC havuzunun 11–13 slotları katı yüz ağırlıklarıdır; yalnız canonical
  + variational kolda ayrılır. Bütçe/trim hesabı 9 yerine bu kolda **12 sayfa alanı**
  kullanır. `transfer_sparse_resident_bytes` bu üç alanı da sayar; UI/script/IPC
  mevcut ortak sparse seçimi ve core istatistiğini kullanır, yeni yazarlık API'si yok.
- `SparseMacSolidWeights.cpp` host uint8 açıklık değerlerini mevcut MAC tile
  listesinin **slot sırasına** paketler. `SparseMacSolidWeightsGpu.cpp` listeyi okur
  ve yalnız etkin 576-yüz sayfalarını yükler; yoğun cihaz geçici bankası yaratmaz.
  Boyut/key/duplicate/kapasite hataları reddedilir; padding ve eksik sayfa açıklığı
  1'dir. Domain duvar açıklığı önce mevcut açık/kapalı kuralıyla belirlenir.
- Her P2G yeniden topolojilemesinde `solid_weights_ready=false`; basınçtan önce
  güncel listeyle doldurulur. Collider verisi kare içinde sabit olsa da slot sırası
  sabit varsayılmaz. Porozite açıklıkları da mevcut host grid'den aynı yoldan alınır.
- Var/porous diverjans ve var gradyan aynı `macSolidWeight` yardımcısını kullanır.
  Basıncın yalnız ağırlık okuyan iki aşamasının compact ikizleri eklendi:
  `sim_sparse_pressure_init_mac_weights`, `sim_sparse_pressure_spmv_mac_weights`
  (**18 buffer / 80 B**). Diğer basınç girişleri 16/80, CG hücre haritası `slot/512`
  kalır; iki ikiz katı ağırlıkları ayrı MAC haritasından `slot+1/576` okur.
- Maliyet sınırı: basınç öncesi **ek tile listesi indirme/senkronizasyonu ve CPU
  paketleme** vardır. Bu parti cihaz belleğini azaltır; hızlanma iddiası yok.
  Kare/alt-adım süresi sonraki build'de ölçülmeli. Host grid S2'ye kadar yoğun kalır.
- 8 derlemesiz kaynak kontratı PASS. Yeni C++ test kaynakta, **derlenmedi/koşulmadı**:
  `sparse_mac_solid_weights_test.cpp` + `SparseMacSolidWeights.cpp` ile paketleme
  testi. `rt_test_sparse_pressure_ipc.py --transfer --solid-weights` kesirli AABB
  çarpıştırıcı paritesini ve en az 12 compact sayfa alanı muhasebesini sınar;
  bu seçenek **canlı koşulmadı**. Eski komutların varsayılan düzeneği değiştirilmedi.

**Sıradaki parti:** hücre alanları (mask/divergence/pressure publication/solid
velocity) ve onların bütün tüketicileri. S1b bütünü hâlâ tamam değil. Ortak storage
çekirdeği bağlanmadı; saf sıvı sahiplik satırları korundu, `GasSimulator.cpp` değişmedi.

### Öncelik değişimi — DEM uyku düzeltmesi

2026-10-09: kullanıcı diğer ajanın S1 PASS ve DEM uyku açık static/wet FAIL
sonuçlarını getirdi. Hücre alanları partisine geçmeden DEM uyku düzeltmesi yazıldı
(revizyon 22, derlenmedi/canlı doğrulanmadı). Ayrıntı `DEM_UYUYAN_TANELER.md`,
kabul `NEXT_BUILD_CHECKS.md` → "DEM uyku düzeltmesi — revizyon 22". S1b iki ağırlık
partisi kaynakta korunuyor; bunların sonraki build kabulü de hâlâ bekliyor.
