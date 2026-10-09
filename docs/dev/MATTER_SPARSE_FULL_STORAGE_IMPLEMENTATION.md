# Sıvı ve gaz sparse depolama — tek build öncesi çalışma noktası

> **2026-10-09 devralma notu:** kullanıcı "tek build" yerine **aşamalı, ara
> build'li** ilerlemeyi seçti (ölçüm: ~30 dosyada ~900 yoğun alan erişimi, ~70
> shader; `GasSimulator.cpp` ölü alan, kapsam dışı). Çalışma noktası artık
> [MATTER_SPARSE_S1_SIVI_GPU.md](MATTER_SPARSE_S1_SIVI_GPU.md): S1 karışık sıvı
> şeridi → S1.5 saf sıvının cihazda kalıcılığı → S2 host/render/cache → S3 gaz.

## Kullanıcının belirlediği kapsam

2026-10-09: sıvı ve gaz sparse sisteminin tamamı kaynakta tamamlanacak; ardından
kullanıcı tek shader + C++ build alacak ve dış IPC kabul paketi sıralı koşulacak.
Bu kapsam onaylıdır; sıvı/gaz ayrımı veya devam izni yeniden sorulmaz.
Codex build, shader compilation veya uygulama launch çalıştırmaz.

**Kaynak işi açık. Henüz son build zamanı değildir.** Aşağıdaki internal modüller
tek başına aktif solver'ın tam sparse olduğu anlamına gelmez. `use_sparse_tiles`
şu anda önceki pressure/RHS/transfer havuzlarını seçer. Yeni storage çekirdeği
henüz bu consumer'ların authoritative alanı olarak bağlanmadı.

## Eklenen ortak çekirdek

- `SparseGridStorage.h/.cpp`: bir immutable topology, kanal başına lazy value
  sayfaları, 512 cell/576 MAC ownership, açık background değerleri. Ambient gaz,
  boş yoğunluk ve açık yüz ağırlığı için sadece okumak value page ayırmaz.
- Topology değişimi bütün kanallar için tek transaction'dır. Canlı bir kanalı
  kaybettiren retirement tüm transaction'ı reddeder. Snapshot'lar immutable'dır;
  sonraki yazı state/page seviyesinde copy-on-write yapar.
- Page capacity ledger snapshot'ın tuttuğu eski sayfaları da sayar. Value budget
  yeni page allocation'dan önce reserve edilir. Bu sayaç yalnız float page
  capacity'dir; topology/container overhead veya toplam scene VRAM değildir.
- Cell/MAC page import/export fiziksel tile bazındadır. Clipped/padding yüzlerde
  ikinci otoriteye izin verilmez. Nonfinite import, disabled kanal, eksik destek
  yazısı ve budget ihlali açık hata verir.
- `gasPressureTopology` bütün fiziksel hava domain'ini kapsar. Smoke density
  threshold ile basınç desteği silinmez. Ambient scalar value sayfaları lazy
  kalabilir; pressure/velocity gerçek denklemin gerektirdiği yerlerde yazılır.

## Eklenen GPU transaction katmanı

`SparseGridGpu.h/.cpp` bootstrap/upload, GPU remap/retirement, explicit host
publication ve release işlemlerini ortak storage contract üzerinden sağlar.
Map/list kanallar arasında paylaşılır. Map domain tile sayısı boyutundadır;
hücre sayısı kadar yeni host occupancy bitmap yoktur.

Yeni dört shader `sim_sparse_grid_map_clear/map_seed/retire/remap`: ABI 7 SSBO /
48 push byte. Map GPU'da clear/seed edilir. Remap slot yerine fiziksel tile key
ile yapılır; eski + candidate GPU capacity authored transaction budget'e dahildir.
Retirement validation yalnız dört byte indirilir. Herhangi bir canlı kanal
kaybolacaksa veya nonfinite device value varsa eski DeviceGrid korunur. Explicit
host download da bütün kanalları candidate üzerinde hazırlar ve en son publish
eder; bir kanal transferi başarısızsa mevcut host canonical state değişmez.

Yeni `.h/.cpp` modülleri vcxproj/filters'da, shader'lar simulation batch ve Vulkan
kernel registry'de kayıtlı. Bu katman **henüz runtime consumer migration değildir**;
eski dense banklar kaldırılmadı ve yeni runtime storage mode sunulmadı.

## Kapanacak kaynak kapıları

1. **Canonical alan ve lifetime bağlantısı:** FluidGrid/phase state, model-local
   GPU buffers ve backend/cache invalidation bir storage service üzerinden
   yönetilecek. Snapshot/export renderer ve cache'in gerçekten tükettiği fiziksel
   alanı temsil edecek; unlinked ikinci authoritative state oluşmayacak.
2. **Sıvı consumer dönüşümü:** P2G ağırlıkları, occupancy/mask, solid phi/velocity,
   fractional faces, viscosity, divergence/projection/gradient, GFM, porous
   reaction, intermodel/grain contact ve G2P aynı canonical sparse adreslemesini
   okuyacak/yazacak. Dense publication/MAC/weight/FLIP bankları uygun modda
   kaldırılacak. Destek yalnız quadratic scatter footprint'iyle sınırlanmayacak.
3. **Gaz consumer dönüşümü:** velocity/scalar semi-Lagrange ve MacCormack,
   emitter/collider sources, combustion/heat, buoyancy, force fields, curl/noise,
   vorticity, dissipation/clamp ve divergence/pressure/gradient birlikte taşınacak.
   Open/closed/periodic sınırları ve moving walls korunacak. Görünmeyen havanın
   basınç kapsamı density/heat render activity listesinden türetilmeyecek.
4. **Dış yüzey ve ölçüm:** UI/Python/IPC aynı core storage mode, active/allocated/
   retained tiles, gerçek capacity bytes ve hata nedenlerini gösterecek. Alan
   snapshot/render/cache/native-device sözleşmeleri birlikte taşınacak. CPU veya
   dense fallback sparse GPU diye gösterilmeyecek.
5. **Tek build öncesi source gate:** tüm yeni/ortak shader ABI'leri, model/field
   index width, cache revision, publication ordering ve source regresyonları
   geçecek; phase physics veya sınırlar kabul için kapatılmayacak.

## Mevcut doğrulama

`check_sparse_grid_storage_contracts.py` ve bağımsız CPU GPU-remap oracle
`check_sparse_grid_transaction_math.py` PASS. Önceki MAC transfer, Matter GPU ve
grain source denetimleri de PASS. Bunlar compiler/GPU doğrulaması değildir.

Yeni `sparse_grid_storage_test.cpp` ortak topology, atomic rejection, ambient,
snapshot retention, authored budget ve MAC padding testlerini içerir. Kullanıcı
test hedefinde `SparseGridStorage.cpp` ile birlikte derlenip çalıştırılacak;
henüz derlenmedi/çalıştırılmadı. Beklenen çıktı:
`PASS sparse shared topology, atomic channel remap, snapshots and budget`.

Son build ancak consumer ve dış yüzey kapıları tamamlandıktan sonra gelir. Canlı
kabul: dense/sparse aynı başlangıçla mass/momentum/thermal energy, divergence
residual/dt convergence; ayrı havuz, tile crossing, jet/pool, kapalı gaz kutusu,
moving wall, üç-owner porous ve yeni tile/remap vakaları. Native GPU kernel
timing, transfer bytes, resident/retained capacity ve cinematic maliyet ayrı
raporlanır. Şu an tam sparse veya VRAM/FPS kabulü yapılmış değildir.
