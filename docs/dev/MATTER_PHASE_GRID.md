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

## Checkpoint sonrası manuel taşıma bulgusu — 2026-10-04

Kullanıcı `da93cec` checkpoint sonrasında Matter/su domainini manuel taşıdı.
Taşınan içerik eski domain sınırının dışına çıktığında tamamen çizilmez oluyor;
yeni konumda simülasyon oynatılınca yeniden doğru konumda çiziliyor. Bu bulgu,
önceki yalnız splat materyalinin geride kalması belirtisinden daha geniştir.
Geri alımın render yayınlama/geçersizleştirme tarafını gereğinden fazla geriye
götürmüş olabileceği kullanıcı tarafından bildirildi; kök neden doğrulanmadı.

Son taşıma/kozmetik turunda simülasyon duraklatılmışken eski domain sınırını
aşan öteleme, SDF/splat/fog görünürlüğü, render bounds ve cache scrub birlikte
kontrol edilmeli. Kabul: yeni konumda simülasyon adımı gerektirmeden içeriğin
görünmesi; cache tekrarlarında konumun sürüklenmemesi. Şimdilik yalnız kayıt
eklendi, davranış değiştirilmedi.

## Matter gaz görünümü / RT regresyon adayı — 2026-10-04

Kullanıcı karma yakıt presetinde RayFusion/Solid görünümünün doğru, RT gaz
görünümünün eksik olduğunu bildirdi. Açık sahnede IPC domain adı
`Ignited Fuel Jet Matter #1` idi. Gaz kütlesi 2.48 kg, gaz-faz aktif hücreleri
58036; plume sorgusunda sıcaklık ve dolu gaz hücreleri mevcut. Dolayısıyla
gaz üretiminin hiç çalışmaması bu bulguyu açıklamıyor.

Kaynak incelemesinde Matter gazı ayrı `syncDomainFogVolume` yolunda host
`grid.density/temperature` üzerinden NanoVDB'ye yayınlanıyor ve dense GPU
bağlaması temizleniyor. Eski Gas ana slotu ise GPU field view ve gerektiğinde
host mirror kullanıyor. Bu iki yolun Vulkan GPU residency ve RT hacim
yayınlaması açısından eşdeğerliği doğrulanmalı. Kesin kök neden ve görsel
düzeltme henüz doğrulanmadı; plan tamamlanınca kendiliğinden düzelecek kabul
edilmemeli. RT gas slot/bounds/mask ve density/temperature aktarımı için
ayrı regresyon kabulü gerekir. Sahne sıfırlanmadı, kaynak kod değiştirilmedi.

### Karşı örnekler ve daraltılmış kapsam

Kullanıcı tekil Nuclear Detonation presetinde RT gaz renderinin doğru olduğunu
doğruladı. Ardından tek Matter domainine gaz ve sıvı emitter ekleyerek RT'de
önce alevin göründüğünü, üzerine su dökülünce söndüğünü bildirdi. Bu başarılı
karşı örnek, genel Matter RT gaz yolunun bozuk olduğu yorumunu desteklemiyor.
Önceki yakıt-preset bulgusu preset kurulumu, hacim yaşam döngüsü veya anlık
render güncellemesi açısından araştırılmalı; kök neden hâlâ doğrulanmadı.

Son açık sahnenin salt okunur IPC kaydı: `Physics Domain 1`, Matter,
87552 liquid parçacığı, mevcut anda gas faz kütlesi ve aktif hücreleri sıfır,
SDF/splat canlı, fog slot id 225 mevcut. Bu kayıt sönme sonrası anlık durumdur;
alevin önceki görünümü ve suyla sönme kullanıcı tarafından doğrulanmıştır.
Sahne ilerletilmedi veya sıfırlanmadı; render koduna düzeltme uygulanmadı.

### İlk kare / RT slot probu ve dar düzeltme

Kullanıcının onayıyla timeline 0, 1, 3, 6, 10, 15, 25, 40 kareleri gezildi;
sonunda önceki 15. kareye dönüldü. 1–40 karelerinde viewport tarafında gas
NanoVDB ve liquid SDF slotları mevcutken RT render tarafında yalnız SDF
slotu bulundu. Kayıt: `matter_rt_slots_2026-10-04.json`. Bu, gaz simülasyonu
eksikliğinden ziyade RT hacim slotu yaşam döngüsü sorununu doğruluyor.
Mevcut açık emitter listesi önceki kullanıcı alev/su testini birebir yeniden
tanımlamadığından bu prob sönme fiziğinin tekrar kabulü sayılmaz.

Kaynakta gizli hacim yeniden görünür olduğunda yalnız gas SSBO dirty bayrağı
ayarlanıyordu; RT TLAS oluşturma yolu görünmez hacimleri tamamen atlıyor.
`syncDomainFogVolume` görünürlük dönüşüne geometry, Vulkan/OptiX ve CPU BVH
rebuild istekleri eklendi. Bu dar düzeltme yalnız kaynakta mevcut; derlenmiş
uygulamada sonucu henüz doğrulanmadı. Kabul: gazın boş→dolu geçişi, timeline
0→ilk dolu kare ve RT yenilemesi sonrasında render.volume_slots içinde Matter
Gas slotunun bulunması; RayFusion görünümüyle tutarlı kalması.

### Son derleme: RT SDF+gas kabulü ve 20. kare renk farkı

Kullanıcı son derlemede RT'nin gas ve SDF'yi birlikte çizdiğini doğruladı.
Önceki suyla ateş söndürme gözlemini geri çekti; bu fiziksel davranış kabul
edilmiş sayılmamalıdır.

20. kare üç tekrar ziyarette koyu mavi; 18/19/21/22 açık-beyaz göründü.
Surface materyal adı değişmedi, 87552 parçacık ve aynı 50³ SDF boyutu korundu.
20'de yalnız SDF hacmi var; komşu karelerde ek gas NanoVDB slotu mevcut.
Plume yoğunluk sorgusu 19'da 123, 20'de 0, 21'de 1 aktif hücre bildirdi;
sıcaklık bu örneklerde sıfır. Bu nedenle fark tek seferlik görüntü yenilemesi
değil, kareye bağlı gas/fog verisi ve katman farkıdır. Hangi fiziksel görünümün
doğru olduğu henüz kesinleşmedi; koyu mavi kare SDF'nin fog olmadan görünümünü
gösteriyor. Komşu karelerdeki fog etkisinin yoğunluk/bounds/cache ve gas-SDF
birlikte render açısından incelenmesi gerekir. Eski 20. kareye dönüldü.

### Kullanıcı izolasyonu: gas + SDF Liquid Body parametreleri

Son kullanıcı testleri önceki fog/20. kare yorumunu daralttı: sorun yalnız
20. kareye özgü değil, gas eklenmiş sahnelerde farklı karelerde de oluşuyor.
Fog çıkışı kapatıldığında veya splat görünümüne geçildiğinde de bildirilen
fark sürdü. Liquid Body UI değerleri bazı karelerde SDF görünümünü etkilerken
diğer karelerde etkisiz kalıyor. Dolayısıyla fog slotunun kaybolması tek başına
kök neden veya doğru görünüm ölçütü olarak kabul edilmemeli.

Hipotez: gas ve liquid SDF birlikte render edilirken SDF materyal bağlaması,
yüzey derinliği, scattering veya absorption değerleri başka/default sabitlere
düşüyor. Bu hipotez henüz kaynak veya canlı parametre ölçümüyle doğrulanmadı.
Kullanıcı isteğiyle düzeltme son kalite testlerine ertelendi.

Son kalite kabulü: aynı Liquid Body ayarlarıyla gas yok/var, gas dolu/boş,
fog açık/kapalı, SDF/splat geçişi ve cache scrub karşılaştırılmalı. SDF için
bağlanan materyal ve gerçek RT depth/scattering/absorption parametreleri
izlenmeli; UI değişiklikleri gas bulunmasından ve kareden bağımsız etkili
olmalı. Hangi mevcut karenin fiziksel olarak doğru olduğu kesinleşmedi.


### İki emitter kimya ve splat materyali — 2026-10-04
Kullanıcı aynı Matter alanında bir granular ve bir fluid emitter kurabiliyor;
ancak domain ana davranışındaki kimya ikisine de uygulanıyor ve iki splat kaynak
ayrı materyallerle sürülemiyor. Son kalite turunda iki farklı substance tag ile
kimya çözümlemesi, domain fallback koşulları ve splat materyal bağlaması birlikte
doğrulanacak. Emitter model ayrımı bu iki özelliğin tamamlandığı anlamına gelmez.
