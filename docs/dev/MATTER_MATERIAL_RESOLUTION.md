# Madde mekanik çözümleme raporu

Bu ilk aşama presetleri ve fizik yasalarını değiştirmez. Mevcut viskozite
politikası `resolveSubstanceViscosity` içinde ortaklaştırıldı; çözücü ve rapor
aynı hesabı kullanır. Önceki float işlem sırası ve 1/24s referans eşiği korunur.

## Erişim

- UI: Matter domain → Active Matter → Material resolution.
- Scripting: `import rt` ardından
  `rt.fluid.matter_models("Domain name")["material_resolution"]`.
- Harici IPC: `fluid.matter_models`, `{ "domain": "Domain name" }` isteğinin
  cevabındaki `material_resolution` alanı. Yeni endpoint veya ayar eklenmedi.

Üçü de `inspectMatterMaterialResolution` çekirdeğini kullanır. Mevcut IPC/script
hata taşıma biçimi korunur; bilinmeyen domain mevcut API tarafından, bilinmeyen
varsayılan madde ortak rapor çekirdeği tarafından reddedilir.

## Anlamlar

`schema_version=1`, `scope=matter_domain`. Bu bir yapılandırma ve senkronize
parçacık görüntüsüdür. `execution_measured=false`, `backend_executed=null`;
`backend_requested` gerçek GPU çalışması veya backend desteği kanıtı değildir.
Isıl yüzey nesnelerine parçacık mekaniği atamaz, gaz alanını incelemez.

`default_substance`, `based_on`, `requested_model` ve `requested_transport`
profil tercihini gösterir. Canlı taşıyıcılar `carriers` içinde madde etiketi,
model ve taşıyıcıya göre gruplanır. Model Auto ise mevcut doğum çözümleyicisi
kullanılır; `model_basis` bunu açıkça belirtir. Taşıyıcı kararı mevcut
`substanceTransportOwner`/`isMatterObstacle` mantığındandır. Statik bağlar mevcut
domain bağlarından toplanır. Bilinmeyen etiketin domain maddesine geri düşmesi
`identity_resolution=unknown_tag_domain_fallback` olarak görünür.

`carriers_measured=false` olduğunda liste boştur; faz veya aktif model uydurulmaz.
Sıcaklık0 eski/yazılmamış durumdur, ortalamaya katılmaz. Fusion NaN mevcut
çözülmemiş faz durumudur, ortalamaya katılmaz. Bilinen örnek sayıları ayrıca
verilir; hiç bilinmiyorsa ortalama null olur. Ortalamalar parçacık sayısı
ağırlıklıdır, kütle/enerji ağırlıklı değildir. Fusion oranından tek başına gaz
fazı çıkarımı yapılmaz.

`viscosity` domain tabanını gösterir; parçacık sıcaklığına bağlı viskozite veya
GPU dispatch ölçümü değildir:

- `requested_m2_s`: profil sıvı viskozitesi.
- `policy_effective_m2_s`: mevcut eşik politikasının sonucu.
- `configured_m2_s`: domain üzerinde şu an kayıtlı değer.
- `configuration_matches_policy`: bu iki son değerin eşitliği.
- `reason`: active, zero_requested, granular_skeleton,
  below_resolution_threshold veya thermal_curve_base.
- `reference_dt_s`, `reference_diffusion`, `voxel_size_m`, `skip_threshold`:
  kararın sayısal girdileri; gerçek kare dt'si değildir.

`mpm_parameter_scope=shared_domain_material` mevcut ortak MPM parametrelerini
belirtir. Birden fazla MPM maddesi varsa bağımsız yasalar çalıştığı iddiası yerine
not eklenir. `transport_ready`/`transport_reason` mevcut ortak sahiplik
denetiminden gelir; yeni backend yetenek denetimi eklenmedi.

### Ortak faz/model seçimi (2026-11-08)

Doğum ve fusion sonrası model yenileme artık `resolveMatterConstitutiveState`
çekirdeğini kullanır. Mevcut 0.5 katı oranı eşiği korunur. Bir eriyebilir MPM/DEM
iskeletinde sonlu fusion oranı varsa sıcaklıktan önce o kullanılır. NaN fusion
oranında önceki doğum/sıcaklık kuralı sürer. Bu fonksiyon ısı, entalpi veya kütle
hesaplamaz; mevcut enerji hesabının ürettiği durumu kullanır.

Rapora eklenen `phase_mechanics`, varsayılan ve taşıyıcı maddelerinin mevcut
parçacık davranışını açıklar. `solid=not_defined` ile `liquid=fluid` aynı şey
değildir. `supported_pinned_carrier`, termal zincir ve destek temasıyla sabitlenen
donmuş sıvı taşıyıcısıdır; deformasyon çözen MPM katısı değildir. DEM taneleri de
deformable solid diye gösterilmez. Bu metadata nesnelerin MSF/rijit cisim
mekaniğini değiştirmez ve yeni malzeme yasası tanımlamaz.

Her taşıyıcı grubundaki `phase_counts`, katı/sıvı/karışık/bilinmeyen durumları;
`selection_basis_counts`, model seçiminin profil, doğum sıcaklığı, fusion oranı
veya domain fallback kaynağını sayar. Eriyebilir olmayan Fluid profilden yalnızca
mekanik modeline bakarak termodinamik faz çıkarılmaz; unknown kalır.
`model_mismatch_particles`, kayıtlı model ile mevcut duruma göre çözümlenen model
arasındaki farktır; rapor bu farkı düzeltmez. UI faz/sayı özetini madde adının
üzerine gelince gösterir. Bilinmeyen sıcaklıkla çözümlenen model fallback olabilir;
bu, ölçülmüş sıcaklık anlamına gelmez.

Mevcut eriyebilir Fluid profillerin soğuk doğumda yine Fluid taşıyıcısı oluşturması
`cold_fluid_birth_remains_fluid_carrier` ile açıkça raporlanır. Yeni bir soğuk
metal deformasyon yasası veya bu yola ait yeni ret kuralı bu aşamada eklenmedi.
Duruma özel yazılabilir mekanik şema ve desteklenmeyen üretimin reddi sonraki
kapsamdır; mevcut birth/fusion davranışı burada değiştirilmedi.

## Doğrulama ve sınırlar

Pozitif/sonlu voxel boyutu, sonlu ve negatif olmayan viskoziteler gerekir.
Senkronize görüntüde mevcut yan diziler boş veya parçacık sayısıyla aynı boyutta
olmalı; negatif/sonlu olmayan sıcaklık, bilinmeyen model, [0,1] dışı veya sonsuz
fusion oranı hata verir. Fusion NaN yukarıdaki anlamıyla kabul edilir.
Hata durumunda çıktı değiştirilmez. Sahne, parçacıklar ve domain ayarları hiçbir
rapor çağrısında değiştirilmez. Görüntü anlık CPU senkronizasyonunu temsil eder.

Kullanıcı derlemesinden sonra:

1. Oil ve Honey için aynı h=0.1m, termal yol kapalıyken raporu karşılaştır:
   Oil below_resolution_threshold, Honey active olmalı.
2. Oil termal yolu açıkken thermal_curve_base görülmeli. Bu ayar canlı durumu
   sıfırlayabildiğinden ayrı test sahnesi kullan.
3. Play öncesi canlı görüntü yoksa sıcaklık/aktif taşıyıcı bilinmiyor görünmeli.
4. Kar sahnesinde senkronize MPM taşıyıcıları görünmeli; rapor açmak parçacıkları
   sıfırlamamalı veya simülasyonu başlatmamalı.
5. Dış terminal: `python scripts/test/rt_test_material_resolution_ipc.py "Domain name"`.
6. Kullanıcı test hedefinde `matter_material_resolution_test.cpp` regresyonunu
   ilgili fizik/kütüphane nesneleriyle bağlayıp çalıştır.

Codex uygulama veya shader derlemez; uygulamayı başlatmaz. C++/IPC testleri bu
teslimde çalıştırılmış sayılmaz; statik kaynak kontrolleri ayrıca raporlanır.


`phase_basis_counts`, fazın fusion oranından mı, yalnızca sıcaklıktan çıkarım
olarak mı, yoksa erime tanımı olmayan katı iskeletten mi geldiğini ayırır.
`temperature_inferred`, latent faz oranı ölçümü olarak yorumlanmamalıdır.
