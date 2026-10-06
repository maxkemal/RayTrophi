# C4a — parçacık kimliği ve ayrı transfer alanları

2026-10-04 kaynak partisi. Derleme ve C++/canlı kabul kullanıcı ortamında
bekleniyor. C4'ün tamamı, GPU karma çözücü veya su-kum etkileşimi tamamlandı
anlamına gelmez.

## Teslim edilen çekirdek

- `FluidParticles` domain-local 64-bit kimlik taşır. Emit yeni kimlik üretir;
  swap-remove ve stable compact kimliği korur. Clear kimlik sayacını geri
  sarmaz. Runtime reset yeni bir kimlik epoch'u başlatır. Kimlik kapsamı
  domain/runtime epoch'udur; farklı domainlerde aynı sayı bulunabilir.
- Memory cache snapshot'ı kimlik ve allocator'ı tam olarak kopyalar; ara
  karelerde kimlikler kuantize edilmez veya atılmaz. RAM hesabı 8 B/parçacık
  kimlik sidecar'ını ve fiziksel rest-mass sidecar'ını içerir.
- Disk sim-cache sürümü **10**: kimlik ve allocator yazılır/okunur; sıfır,
  tekrar eden veya allocator sınırını aşan kimlik reddedilir. Önceki sürümlerin
  disk cache'i yeniden bake edilmelidir. Bu playback cache'i solver-resume
  tensörlerini taşımayan mevcut render cache'idir; H1 resume cache'i değildir.
- Ortak cell-centered quadratic 27-hücre desteği, origin/voxel sözleşmesi ve
  kararlı kimlikli hücre occupant listesi kurulur. Su ve granül için ayrı
  **kg** kütle ve **kg·m/s** momentum alanları vardır. Elastic/unresolved
  parçacıklar yanlışlıkla bu iki alana scatter edilmez.
- Bu CPU referansı fiziksel kütle ve doğrusal momentum scatter'ını kapsar;
  affine APIC momentum/stress scatter ve MAC solver bağlaması C4b'dedir.
- Sınırda kesilen stencil yeniden normalize edilir; domain dışındaki
  parçacıklar ayrı sayılır. Model toplamları, deposit toplamı ve dışarıdaki
  kütle ayrı raporlanır. Fiziksel rest mass önceliklidir; eski sıfır rest mass
  domain chemistry yoğunluk politikasından hesaplanır.
- Ortak hücrede mass-gradient normalinden iki model arasında eşit/zıt
  normal impuls ve Coulomb teğet impuls üreten CPU contact referansı vardır.
  Ayrılan çift çekilmez. Kinetik enerji kaybı ve momentum farkı raporlanır.
  Geçersiz/nümerik aralık dışındaki istek alanları yarım değiştirmez.

## Kullanıcı ve otomasyon yüzeyleri

Matter Domain panelinde **Active Matter** açılır bölümü model başına parçacık
ve kg gösterir. UI ve API aynı `inspectMatterModels` servisini kullanır.

```python
rt.fluid.matter_models("Physics Domain 1")
rt.fluid.matter_models("Physics Domain 1", include_transfer=True)
```

IPC aynı isimle `fluid.matter_models`, `domain` ve `include_transfer` alanlarını
alır; salt okunur yetkiyle çalışır. Varsayılan sorgu yalnız model envanteri
hesaplar. Transfer referansı istek üzerine kurulur ve 250000 parçacıkla
sınırlıdır; frame cache'e ek sparse scratch kopyası koymaz. Kimlik hash'i
SoA sırasına duyarlıdır; compact sonrası değişmesi doğal, aynı snapshot'a
dönüşte aynı olması beklenir. Allocator JSON'da ondalık string olarak döner.

`mixed_transport_ready=false` bilinçlidir: canlı solver C4b'de ayrı transfer
alanlarına ve iki yönlü contact'a bağlanacak. Bu kaynak partisinde mevcut
tek-domain solver physics değiştirilmedi. Domain-geneli granular anahtarı
henüz karma fiziğin otoritesinden tamamen çıkarılmış değildir.

## Kabul ve bir sonraki büyük blok

Codex'in derlemeden çalıştırdığı kontroller:

```powershell
python scripts/test/check_matter_transfer_contracts.py
python scripts/test/check_matter_phase_contracts.py
python scripts/test/check_fluid_window_contracts.py
python scripts/gen_ipc_descriptors.py --check
```

Kullanıcı derlemesinden sonra:

1. `scripts/test/matter_transfer_test.cpp` testini assertions açıkken,
   `MatterTransfer.cpp` ve `source/src/Math/Vec3.cpp` ile çalıştır. Kimlik
   compact/restore/clear, sınırda kg/momentum korunum, lane ayrımı,
   eşit-zıt contact, Coulomb limiti ve separating çiftleri kapsar.
2. Dolu su ve kum sahnelerini duraklatıp dış terminalden
   `python scripts/test/rt_test_matter_models_ipc.py "Physics Domain 1"` çalıştır.
   UI model kg toplamları ve IPC toplamları aynı olmalı. Sorgu solver adımı
   çalıştırmaz veya sahneyi resetlemez.
3. Bake + scrub + aynı kareye dönüşte identity hash ve allocator değerlerini
   karşılaştır. v10 disk save/load kimlik hash'ini korumalı; eski bake'leri
   yeni formatta üret.
4. Tek su/tek kum mevcut fizik ve performans referanslarını korumalı.
   Canlı her adımda sparse reference/contact maliyeti oluşmamalı.

C4b: aynı alt adımda model başına ayrı P2G, granular stress/liquid pressure,
ortak contact ve G2P; Vulkan listeleri ve scratch bütçesi; tek-faz referansları,
su-kum momentum ve performans kabulü. H1 bu çekirdek üstüne gelir. C5/C6 ve
son kalite turundaki RT materyal/taşıma bulguları açık kalır.

## C4a canlı smoke ve C4b CPU aşama ayrımı

Son kullanıcı derlemesi sonrası boş sahnede geçici `C4aSmoke` Matter domaini
kuruldu, CPU su seed edildi ve transfer sorgusu çalıştırıldı. 216 parçacık,
1728 byte identity sidecar, next ID 217 ve 216 kg model/deposit kütlesi eşleşti.
Geçici sistem kaldırıldı; önceki aktif sistem varsa geri seçildi. Bu kısa
test C4a API/kimlik/transfer bağlantısını doğrular; karma fizik, disk cache
roundtrip veya C++ contact kabulü değildir.

C4b'nin ilk CPU solver entegrasyon adımı kaynakta yazıldı. APIC step gövdesi
odaklı `APICFluidStep.inl` modülüne taşındı; klasik varsayılan step aynı yolu
korur. `prepareMatterModelGrid` modelin forces/P2G/boundary/viscosity ve
pressure aşamasını çalıştırıp **G2P'den önce** durur. Granular pressure'a
sokulmaz. Her model kendi FLIP baseline'ını saklar; sonraki modelin global
scratch'i üzerine yazması ilk modeli bozmaz. Contact için grid güncellemesi
bu iki çağrı arasında yapılabilir. `finishMatterModelGrid` kuvvet/P2G/
viscosity/pressure'ı tekrarlamadan G2P, constitutive ve advect/reseed/UVW
aşamalarını bir kez tamamlar. Layout, kimlik sırası, allocator veya count
değişirse gather reddedilir; ikinci finish de reddedilir.

Yeni kaynak statik kontrolü:

```powershell
python scripts/test/check_matter_solver_stages.py
```

`matter_solver_stages_test.cpp` klasik tek su step ile split step'i kıyaslar;
ikinci model hazırlayarak FLIP scratch overwrite, erken UVW/advection ve
topoloji değişimi/çift finish durumlarını kapsar. Bu C++ testi henüz
derlenip çalıştırılmadı. Canlı coordinator, MAC contact alanı ve Vulkan karma
transport hâlâ bağlanmadı; `mixed_transport_ready=false` korunur. Bu kaynak
adımı için ayrı ara derleme istenmiyor; sonraki coordinator bloğuyla toplu
derleme/kabul yapılabilir.


### C4b atomik model batch koordinatörü
`MatterModelBatch` tek bir çözülmüş ortak CPU alt adımını iki model gridine
ayırır. Her model pressure/stress hazırlığını bitirdikten sonra zorunlu temas
callback'i çalışır; ancak sonra G2P/advection yapılır. Hayatta kalan parçacıklar
kanonik kimlik sırasına göre birleştirilir. Tüm SoA yan verileri merkezi
`copyParticleFrom` üzerinden taşınır; allocator ve UVW epoch korunur.
Hatalı kimlik, eksik mekanik yan veriler, geçersiz kütle/zaman veya temas hatası
kanonik parçacık/grid commit'ini engeller.

Bu iç koordinatör henüz canlı step yoluna bağlı değildir. Fiziksel kütleyle
MAC temas servisi, ortak CFL alt adım zamanlayıcısı ve scratch bellek bütçesi
integrasyonu hâlâ gereklidir. `mixed_transport_ready=false` kalır. Derleme
ve C++ çalışma zamanı doğrulaması kullanıcıya bırakılmıştır.


### C4b GPU yönü ve CPU referansı — 2026-10-04
Üretim hedefi GPU compute'tur. CPU batch/MAC temas/CFL kodu yalnız doğrulama
referansıdır; canlı GPU domain için otomatik CPU karma fallback eklenmedi.
Referans temas cell impulse'u MAC yüzlerine eşit/ters impulse olarak kaldırır
ve fiziksel yüz-kütlesi metriğinde enerji artışını global line search ile önler.
Bu metrik APIC particle momentum'unun birebir kabul testi yerine geçmez.
CPU çalışma seti guard'ı konservatif tahmindir; kesin allocator kotası değildir.

`MatterGpuPartition` Vulkan'da kanonik düz parçacık indekslerini fluid/granular
listelerine ayırır. Position, identity veya malzeme state'i yeniden sıralanmaz;
elastic ve unresolved ayrı GPU sayaçlarına gider. Liste ve sayaçlar CPU'ya
indirilmez. Kalıcı capacity buffer'ları, allocation hata temizliği, Vulkan ABI
ve shader derleme listesi eklendi. Guard yalnız partition buffer bütçesidir;
bütün model grid/pressure scratch bütçesi canlı coordinator tarafından
birleştirilmelidir. Metadata şu an dispatch sırasında upload edilir; residency
bağlamasında sadece topology/model version değişiminde güncellenmelidir.

GPU indexed P2G/G2P, fiziksel kütleyle GPU contact ve ortak substep coordinator
henüz canlı yola bağlı değil. `mixed_transport_ready=false` sürüyor. Bu kayıt
bir C4b tamamlanma veya performans/parity kabulü değildir.


2026-10-04 güncel C4b test noktası: `MATTER_MIXED_GPU_TEST_POINT.md`.
Karma GPU canlı bağlantısı, ortak alt adım, fiziksel P2G kütlesi ve GPU temas
kaynakta eklendi; önceki “bağlı değil” kayıtları tarihsel ilerleme notlarıdır.
Derleme/shader/sahne kabulü henüz kullanıcı tarafından yapılmadı. Sonraki adım
tek derleme ile karma GPU kabulü; C4 tamamlandı etiketi henüz verilmedi.
