# Matter phase grid — C3 kaynak paketi (2026-10-04)

## C3b: bağımsız faz gridleri

Bu paket C3'ün kalan uygulamasını içerir. Derleme ve canlı kabul kullanıcı
tarafından tek toplu turda yapılacak; C3 henüz canlı kabul edilmiş sayılmaz.

- `MatterPhaseSettings` descriptor'ında gaz ve sıvı ayrı override taşır.
  Varsayılan eski mantıksal domain gridini miras alır; eski dosyalar değişmez.
  Dünya sınırları padded mantıksal origin'e göre offset olarak saklanır ve
  domain taşındığında iki faz da onu izler. Adaptive origin'de padding zaten
  uygulanmış olduğundan ikinci kez eklenmez.
- `MatterPhaseConfig` doğrulama, layout, bütçe, tahsis, kayıt, sorgu ve hash'in
  ortak servisidir. UI ve API mutation aynı `setPhaseGrid` fonksiyonunu çağırır.
  Sonlu, kesin sıralı bounds ve en az 1e-6 m voxel gerekir. Eksik faz ve geçersiz
  phase adı reddedilir. Hatalı ayar descriptor/cache/state'i değiştirmez.
- Her faz kendi hücre sayısıyla 128 B gaz / 224 B sıvı çalışma bütçesine katılır.
  Kombine bütçe önce tahsis edilen boyutları küçültür; 8 hücre/eksen tabanı ve
  512 hücre/eksen tavanı korunur. Kübik voxel bütün istenen kutuyu kapsar;
  clamp sonrası kısa eksenlerde fazla kapsama olabilir. İstenen voxel saklanır,
  etkin voxel ayrıca raporlanır. Olmayan fazın CPU grid'i ve cihaz tahsisi yoktur.
- Tahsis ve kaynak kodları odaklı `MatterDomainSynchronization.inl` ve
  `MatterDomainSources.inl` modüllerine ayrıldı. Sıvı kaynakları kendi faz
  gridinin dışına parçacık eklemez; reddedilen örnekler rastgele dizisini
  ilerletir, başarılı emisyon sayacını artırmaz. Örnek serial'i cache ile taşınır.
- İki serializer aynı version 1 phase schema'sını okur/yazar. Geçersiz kayıt
  reddedilir; yarım config uygulanmaz. Faz hash'i cache imzasına katılır.
  Cache ötelemesi mantıksal bounds üzerinden hesaplanır ve iki fazın layout'u
  uyumluysa uygulanır. UVW ve whitewater taşıma ile birlikte ilerler.
- Primary sıvı SDF/NanoVDB, render bounds, UVW, foam ve rigid-fluid coupling
  sıvı gridini kullanır. Matter'ın secondary fog/gas slot'u gaz gridini kullanır.
  Collider ve GPU pressure/occupancy aktif solver phase scope'unu kullanır.
- Burning Fuel Spill ve Ignited Fuel Jet, eski ayrı gaz/sıvı kutularını tek
  Matter descriptor'ının faz ayarlarına taşır. Flamethrower gaz olarak kalır.

## Authoring ve sorgu

Mevcut domain panelindeki **Gas / Liquid Phase Grids** bölümünde her faz için
inherit, dünya bounds ve voxel girilip **Apply phase grid** seçilir. Ayar
uygulamak grid simülasyonunu yeniden başlatır ve eski frame cache'ini temizler.

Python'da aynı işlemler `rt.fluid` altındadır:

```python
rt.fluid.set_phase_grid("Physics Domain 1", "gas",
                       bounds_min=(-3, 0, -3), bounds_max=(3, 6, 3), voxel=0.2)
rt.fluid.set_phase_grid("Physics Domain 1", "liquid",
                       bounds_min=(-2, 0, -2), bounds_max=(2, 2, 2), voxel=0.1)
info = rt.fluid.get_phase_grids("Physics Domain 1")
# Eski mantıksal grid'e dön:
rt.fluid.set_phase_grid("Physics Domain 1", "liquid", inherit=True)
```

IPC adları `fluid.set_phase_grid` ve `fluid.get_phase_grids`; parametreler aynı.
Get Read, Set SceneWrite yetkisi kullanır ve main-thread enqueue üzerinden gider.
Python hata yükseltir; IPC aynı hata metnini normal error envelope'unda döndürür.
Granüler madde `liquid` gridini kullanır. `measured=false` senkronizasyon öncesi
planı belirtir. Sorgu phase presence, requested/effective bounds ve voxel,
resolution, cells, working bytes, budget ve clamp bilgisini döndürür.

## Tek toplu derleme ve kabul

1. Ana projeyi bir kez derle; splat taşıma düzeltmesi bu pakettedir.
2. Matter sahnesinde yukarıdaki gibi geniş/kaba gaz ve dar/ince sıvı gridlerini
   panelden ayarla. Birkaç sim adımı ilerlet. SDF/splat/Material/Rendered
   konumları ve collider davranışı tutarlı olmalı.
3. Açık uygulamaya harici terminalden readonly probe çalıştır:

   ```powershell
   python scripts/test/rt_test_matter_phase_grids_ipc.py "Physics Domain 1" --expect-distinct
   ```

4. Domain'i duraklatılmışken taşı: yüzey, splat, UVW ve whitewater aynı anda
   gelmeli. Cache scrub/replay iki grid konumunu korumalı. Kaydet/aç sonrasında
   iki override, requested voxel ve faz boyutları korunmalı.
5. Burning Fuel Spill / Ignited Fuel Jet presetlerinde tek Matter kimliği,
   farklı gaz/sıvı bounds, görünür gaz ve sıvı kontrol edilir. Eski tek-faz
   Gas/Fluid sahnesini açıp ek faz/grid oluşmadığını kontrol et.
6. Kombine bütçeyi azalt: `working_bytes <= budget_mb * 1024^2`, istenen voxel
   korunur ve etkin çözünürlük küçülür. Sıralanmamış bounds, NaN/negatif voxel
   ve eksik faz isteği ayarları değiştirmeden hata vermeli.

Kaynak kontrolleri (derleme değildir): `check_matter_phase_contracts.py`,
`check_fluid_window_contracts.py`, `audit_ipc_capabilities.py` ve descriptor
`--check`. Yeni `matter_phase_config_test.cpp`, transactional validation,
serializer round-trip, farklı layout, bütçe, taşıma, absent phase ve cache
layout uyumunu kapsayan kullanıcı ortamında çalıştırılacak test kaynağıdır.
C++ testleri ve canlı probe Codex tarafından çalıştırılmadı.

## C3a tarihsel kayıt

Kullanıcı C3a derleme/testlerinin geçtiğini bildirdi. Taşıma sırasında splat
geometrisinin bir sonraki sim adımına kadar eski yerde kalması tespit edildi:
sync konumları değiştiriyor, fakat render köprüsünün version kapısını açmıyordu.
Taşıma artık parçacık konumlarını ve state version'ını birlikte güncelliyor.
Bu ek düzeltmenin derleme/canlı kabulü henüz yapılmadı.

Kullanıcı C2b derlemesinin başarılı olduğunu doğruladı. Bu parti C3'ün ortak
faz erişimini hazırlar; bağımsız faz sınırı/voxel authoring'i henüz sunulmaz.
Varsayılan iki grid'in mevcut ortak allocation düzeni korunur.

`Fluid/MatterPhaseGrid.h` gaz/sıvı grid seçiminin ortak otoritesidir. Matter'ın
dışarıya sunduğu `grid` gazdır. Sıvı solver sırasında `MatterLiquidScope`
CPU grid, GPU buffer/residency ledger ve etkin bounds/resolution/voxel bilgisini
birlikte değiştirir. Explicit restore gaz aşamasından önce çalışır; destructor
erken çıkış ve exception durumunda aynı işlemi yapar. Nested scope kendi
başlatmadığı geçişi geri almaz. Legacy Gas/Fluid için geçiş yapılmaz.

Seed ve fill-level yerleşimi, sıvı kaynak voxel'i, particle/billboard yarıçapı,
SDF render örnekleme koordinatları, UVW ve fluid step istatistikleri sıvı gridini
okur. Mevcut Python ve IPC sorguları aynı API verisini kullanır; yeni authoring
işlemi eklenmedi. Gaz istatistikleri gaz layout'unu raporlamayı sürdürür.

Mist, yanma, donma kütle/hacim hesabı sıvı voxel'ini kullanır. Gaz örnekleme ve
hareketli sınır seçicileri solver slot'undan bağımsızdır. Faz aktarımı gerçek
grid kesişimini kullanır; mist drag testi gaz gridinin half-open sınırına
uyar. Cache taşıma yordamı 2026-10-04 tarihinde önceki commit davranışına geri alındı.
Bağımsız faz gridleriyle taşıma/cache ve splat kozmetiği kabul bekliyor.

## Kontroller

Codex derleme yapmadı; kullanıcının açık uygulamasında dış IPC problarını çalıştırdı. Kaynak kontrolleri:

```powershell
python scripts/test/check_matter_phase_contracts.py
python scripts/test/check_fluid_window_contracts.py
```

`scripts/test/matter_phase_grid_test.cpp` kullanıcı ortamında çalıştırılacak
C++ test kaynağıdır: farklı origin/voxel/dimensions, gaz/sıvı seçimi, GPU ledger
swap, nested scope, exception restore, çift restore, eski domain türleri,
half-open sınır ve iki-grid öteleme durumlarını kapsar. Derlenip çalıştırıldığı
henüz doğrulanmadı.

Bir sonraki toplu derleme sonrasında aynı Matter sahnesinde seed/fill, sıvı
görünümü, UVW, mist/yanma ve domain taşıma/cache replay regresyonunu kontrol et.
Fluid step istatistikleri ve UVW sorgusu Python/IPC üzerinden aynı sıvı
dimensions/voxel'i göstermeli; gaz istatistikleri gazı göstermeli. C2b canlı
occupancy/pressure matrisi bu toplu kabulde yapılabilir.

## C3a sonundaki kapsam (tarihsel)

Bu aşamada descriptor override, ortak configuration servisi, UI/Python/IPC,
serializer/cache ve bağımsız tahsis açık kaldı. Yukarıdaki C3b bunları uygular;
toplu derleme/canlı kabul beklenmektedir.

## 2026-10-04 canlı kontrol ve taşıma geri alımı

Açık Matter domaininde faz-grid/telemetri probu geçti. Geçici runtime matrisinde
farklı gas/liquid bounds ve voxel, Vulkan P2G/pressure/G2P, aktif pencere, atomik
hatalı istek reddi, ortak 128 MiB bütçe, inheritance dönüşü ve eski Gas/Fluid
eksik faz reddi geçti. Geçici sistem kaldırıldı; eski aktif sistem geri seçildi.
Dünya step çağrıları mevcut sistemleri de ilerletti.

Taşıma +0.25 X ve geri alma probunda parçacık sayısı korundu; geri dönüşte
yaklaşık 6e-7 m centroid farkı görüldü. Görsel splat başarısı doğrulanmadı.
Normal taşımanın UVW/köpük/sürüm ekleri kaldırıldı. Cache rebase yordamı HEAD
sürümüne döndürüldü; bağımsız gridlerle taşıma kabulü kozmetik son tura ertelendi.

CPU karşılaştırması başarısız: tek 1/120 s adımında ortalama hız CPU 0.163418,
Vulkan 0.081668 m/s; centroid Y farkı 0.000681281 m. İncelemede GPU G2P denemesinin
CPU backend seçiminde de çalışabildiği görüldü. Deneme kapısına mevcut
fluid_gpu_requested koşulu eklendi. Bu düzeltme ve taşıma geri alımı açık
binary'de bulunmuyor; kullanıcı derlemesi sonrası aynı matris tekrar gerekli.
Commit/push bu kabul tamamlanana kadar yapılmadı.

## Son derleme kabulü

2026-10-04 tekrarında geçici runtime matrisi tamamen geçti: CPU/Vulkan
centroid farkı 5.960464477539063e-08 m. GPU seçim kapısı düzeltmesi doğrulandı.
Ana açık Matter domaini sıfır liquid parçacığı bildirdi; ana sahnede sıvı
taşıma/splat görsel kabulü yapılmış sayılmaz. Taşıma kozmetiği ertelenmiştir.
Liquid Display panel sırası kullanıcı tarafından bildirildi; bu kontrolde
mevcut bloğun taşındığı bir diff görülmedi, eski panel yapısı yönü korunacak.
