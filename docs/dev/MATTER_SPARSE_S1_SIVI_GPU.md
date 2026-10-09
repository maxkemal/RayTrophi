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
