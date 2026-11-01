# T6: tek Matter kimliği ve şerit ön ayarları

**2026-10-29 — şeritler kullanıcı derlemesinde 7/7 PASS.** Tam T6 kabulü
henüz verilmedi. Sonraki RAM/disk frame preflight düzeltmesi bu build'den
sonradır; kullanıcı C++ derlemesi ve son toplu tur gerekir.

| Ön ayar | Gaz şeridi | Sıvı/parçacık şeridi | Depolama |
|---|---|---|---|
| Gas | açık | kapalı | birincil gaz ızgarası |
| Liquid | kapalı | açık | birincil sıvı ızgarası |
| Matter | açık | açık | birincil gaz, ikincil sıvı |

Canlı kimlik `Matter`'dır. `SimulationDomainType::LegacyGas/LegacyFluid` yalnız
eski sayısal sahne değerlerini ve eski creation argümanlarını okumak içindir;
simülasyon koşulları descriptor/state şeritlerini okur. Kapalı şerit adım atmaz,
kaynak kabul etmez ve ayrı ızgara ayırmaz. DEM hâlâ Vulkan gerektirir; CPU MPM
referansı, CPU DEM sahne desteği değildir.

## Ortak işlem

UI Lane preset, Python ve dış IPC aynı `rtapi::setMatterDomainLanes` servisini,
o da `SimulationDomainLanes` çekirdeğini çağırır. UI için ayrı iş mantığı yoktur.

```python
# Embedded Python API örneği; canlı testler ayrı süreçte rt_ipc.py ile koşulur.
rt.fluid.get_lanes("Physics Domain 1")
rt.fluid.set_lanes("Physics Domain 1", gas_lane_enabled=True,
                   liquid_lane_enabled=True)
```

IPC metotları `fluid.get_lanes` (Read), `fluid.set_lanes` (SceneWrite).
Okuma: `type`, `preset`, iki şerit anahtarı, `measured`, `primary_cells`,
`secondary_cells`. `measured=false` iken sıfır depolama sayıları canlı ölçüm
değildir. `fluid.get_phase_grids` faz başına requested/effective yerleşimi verir.

Yazma iki bool'u da ister. İkisi kapatılamaz. Canlı parçacık/gaz/ısı/hız içeriği
olan domain'in şeritleri değiştirilemez; önce reset gerekir. Kapatılacak şeridi
hedefleyen kaynak, kapalı olsa bile, önce retarget edilmeli veya kaldırılmalıdır;
aksi halde daha sonra enable edilince sessizce kaybolan kaynak oluşurdu. DEM
maddesi gerektiren bir domain'in parçacık şeridi kapatılamaz. Red halinde
descriptor, kaynak ve cache değişmez. Aynı anahtarlarla no-op canlıda da kabul
edilir ve reset/cache invalidation yapmaz. Başarılı değişiklik yalnız seçili
domain'in layout'unu senkronlar; diğer domain'lerin state'i resetlenmez.

`fluid.create_domain(type="gas"/"fluid"/"liquid"/"matter")` geriye uyumlu ön
ayar girişidir; dönen kimlik `matter`, gerçek içerik `phases`'tır. Kaynakta
`phase="auto"` varsayılanı gaz açıksa gazı, aksi halde sıvıyı seçer. Açıkça
kapalı şeridi hedefleyen gas/liquid phase isteği değişikliksiz reddedilir.
Script filtreleri `type == "fluid"/"gas"` yerine `phases` üyeliğini kullanır.

## E1–E10 göç denetimi

| Kapı | Kaynaktaki karşılığı |
|---|---|
| E1 çift ısı/faz aktarımı | `resolveMatterGasExchangeOwners`: sıvı için tek gaz sahibi; örtüşüyorsa kendi gazı önce, yoksa ilk etkin/örtüşen gaz. Isı, kaynama, mist, yanma aynı seçimi kullanır. |
| E2 kaynak fazı | Legacy Liquid'in bütün kaynakları Liquid, legacy Gas'ınkiler Gas olarak göçer. Yeni kaynak phase'i ortak resolver doğrular. |
| E3 gaz görüntü maliyeti | Gas-only birincil ızgarayı ve mevcut canlı yoğun GPU görünümünü korur; ikincil sıvı/sis kopyası ayrılmaz. |
| E4 gölgelendirici | Legacy gas `shader` korunur ve Matter gaz slotuna taşınır. Gas-only ↔ dual geçişleri en son gaz materyalini korur. |
| E5 ölü sanal granüler yol | Zaten kayıt/çağrısı kalmamış GranularVirtualScene/Representation/Surface kaynakları söküldü. Aktif DEM virtual grains ve mesh-grain yolu korunur. |
| E6 cache yerleşimi | Disk v19 iki şerit bayrağını kaydeder. Legacy Gas/Fluid frame'leri gerekçeli reddedilir; eski dual-Matter formatları bayrak varsayılanlarıyla okunabilir. RAM restore da domain sayısı/kimlik/şerit uyuşmazlığını reddeder. |
| E7 ortam filmi | Gaz sahibi yoksa ortam filmi, varsa aynı tek gaz sahibine yüzey alışverişi. |
| E8 yağmur | Kendi sıvı şeridi yoksa aynı domain'e yağmur parseli doğurmaz. |
| E9 gaz–tane | Gaz kapalıysa gaz sürüklemesi çalışmaz. |
| E10 GPU/foam buffer | Yalnız çift şerit ikincil sıvı/foam buffer'ını kullanır; single-liquid birincilde kalır. |

E6 frame yayımlama: RAM metadata bütün sistemler için restore öncesi
doğrulanır. Disk frame'leri önce ayrı bir adayda okunup doğrulanır; ikinci
sistemin karesi eksik/bozuk/uyumsuzsa ilk sistem de yayımlanmaz, eski runtime
silinmez ve restore başarı dönmez. Disk cache valid=false olur, log rebake
gerekçesini taşır. `MatterCacheRestore.h` aynı tek-domain lane validator'ını
kullanır; ek grid decompression/copy RAM ön kontrolde yapılmaz.

Kayıtlı domain tipi iki yükleyicide ortak parser'dan geçer: metin biçimi
`Gas/Fluid/Matter`, sayısal biçim `0/1/2`. Alanı olmayan eski dosyanın Gas
varsayılanı korunur; bilinmeyen metin artık sessiz Gas'a çevrilmez, bilinmeyen
sayı enum olarak canlı sisteme taşınmaz. Bu kaynak düzeltmesi son build'den
sonradır. Script audit'i `scripts/test/*.py` AST'sinde eski Gas/Fluid canlı
kimliğine göre filtreyi reddeder; yalnız pre-T6 dosyayı bilerek doğrulayan
legacy fixture testi istisnadır.

Göç sırası iki yükleyicide de domain kayıtları → bütün flow source kayıtları
→ ilk synchronize şeklindedir. Erken synchronize legacy tipi Matter'a
çevirip sonraki kaynakların eski fazını düzeltme fırsatını kaybettiriyordu.
ProjectManager ve SceneSerializer artık kaynaklar hazırken aynı core göçü
çağırır; statik audit bu sıralamayı da zorunlu tutar. Eski yanlış faz taşıyan
kapalı kaynakları içeren synthetic project kabulü son toplu turda koşulur.

Eski granüler sahnelerde transport, kanonik maddenin `granular_transport`
özelliğinden gelir. Eski Fluid tipinin gizli MPM seçimi yeni bir solver tipi
olarak korunmaz. MPM referansı isteyen script maddenin MPM transport'unu açıkça
seçmelidir; Gas/Water T0 eşitliği granüler DEM/MPM fiziksel eşitliği değildir.

## Doğrulama

Kaynak kontrolleri: `check_matter_lane_contracts.py`,
`check_matter_phase_contracts.py`, `check_matter_gpu_contracts.py`,
`audit_ipc_capabilities.py`, `gen_ipc_descriptors.py --check`.

Kullanıcı ortak C++ derlemesinden sonra, boş sahne/Pause, dış terminal:

```text
python scripts/test/rt_test_matter_lanes_ipc.py
python scripts/test/rt_t0_matter_vs_specialized_perf.py --rounds 1
python scripts/test/rt_matter_smoke.py --quick
```

Tamamen boş, harcanabilir bir proje üzerinde ayrıca:

```text
python scripts/test/rt_test_matter_legacy_roundtrip_ipc.py
```

Bu test TEMP altında yeni `.rtp/.bin/.shared` dosyaları kaydeder ve açar;
aktif proje yolu değişir. Bugünkü geçerli proje kabuğuna eski `Gas/Fluid`
tiplerini, phase_grids yokluğunu ve kasıtlı eski source phase değerlerini
yazar. İki domain göçü, disabled source retarget, tek ızgara, gaz shader'ı,
canonical v2 kaydı ve ikinci açılış doğrulanır. Gerçek tarihsel dosyaların
bütün sürümlerini kapsadığı iddia edilmez. `--self-test` yalnız fixture
dönüşümünü IPC olmadan kontrol eder; canlı yükleyici kabulü değildir.

`matter_domain_lanes_test.cpp` C++ çekirdek kabul kaynağı ayrıca iki eski
domain'in kaynak fazı/gaz shader göçünü, phase JSON roundtrip'ini, cache red ve
tek gaz sahibinin kendi-domain önceliğini denetler. Ajan derlemedi/koşmadı.
Bu dar çekirdek testi tam `.rtp` legacy save/open veya görsel kabul yerine
geçmez. Son ortak listede eski Gas+Liquid `.rtp` yükle/save/open, materyaller,
tek/çift grid CPU+Vulkan davranışı ve cache rebake red mesajı ayrıca doğrulanır.
