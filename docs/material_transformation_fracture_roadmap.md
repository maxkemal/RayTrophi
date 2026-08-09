# Material Transformation, Mass Transfer and Fracture Roadmap

Vulkan geometry/SDF/gas reset ve rewind guvenlik kurallari:
[`VULKAN_SIMULATION_RESET_SAFETY.md`](VULKAN_SIMULATION_RESET_SAFETY.md).

## 2026-08-09 kapanis ve sonraki sira

Geometry melt, MSF yanma/erime ve molten kutlenin APIC fluid'e aktarildigi dikey
hat cekirdek kapsaminda tamamlandi. Yuksek poligonlu birlesik sahne stres testleri
yapildi; timeline End Frame degisikligi artik reset/rewind uretmiyor.

Bilinen fizik siniri: ana mesh'in geometrik hacim kaybi ile APIC'e eklenen fluid
hacmi henuz gorsel olarak birebir eslesmiyor. En guclu neden, ozellikle SDF kaynak
collider'da molten cikis noktalarinin gercek eriyen yuzey/topoloji yerine cooked
kapali hacimden turetilmesi. Kutle muhasebesi korunuyor; sorun hacim dagilimi,
cikis yolu ve goruntu eslesmesidir. Bu madde mevcut fazi bloke etmez ve sonraki
fizik fazinda ele alinacaktir:

1. Melt alanindan baglantili yuzey akis yolu ve gercek kenar/delik cikisi bulma.
2. Mesh hacim kaybi, density ve APIC particle hacmi arasinda kalibre edilmis esleme.
3. Kaynak collider icin sabit cooked SDF yerine melt-aware proxy/dirty-region
   collision temsili; her-frame tam SDF recook yapilmamasi.
4. Birikme, tasma, taban yayilmasi ve katilasma/deposit geri aktarimi.

Bu fizik eslemesi kapandiktan sonraki ana mimari faz: gas, fluid, collider ve mesh
deformation zincirini node graph'a tasimaktir. Node'lar yeni bir ikinci solver
olusturmayacak; mevcut runtime sistemlerini ortak veri/olay sozlesmeleriyle
yonetecek. Siralama: domain/source/collider node'lari, MSF transform/transfer
node'lari, geometry deformation cikisi, cache/reset lifecycle node'lari ve son
olarak telemetry/debug node'lari.

## Amaç

Yanma, erime, kütle kaybı ve parçalanmayı ayrı efektler olarak değil, aynı kalıcı
malzeme durumunun sonuçları olarak ele almak. Veri katmanı yangına özel
olmayacak; mesh paint, sculpt, terrain, fizik, fluid/gas ve render aynı kimlik,
alan, dirty-region ve cache sözleşmelerini kullanabilecek.

Mevcut sistemler korunur:

- `SurfaceMeshCache`: ortak yüzey/topoloji ve dünya konumu kaynağı,
- `MaterialStateField` (MSF): sıcaklık, nem, yakıt, char ve melt otoritesi,
- mesh paint/layer stack: kullanıcı tarafından yazılan yüzey alanları,
- APIC/fluid ve gas domain'leri: taşınan sıvı/gaz kütlesi,
- Jolt: ayrılmış makro parçaların hareketi ve çarpışması,
- terrain: height, hardness, moisture, flow ve malzeme alanları.

## 2026-08-09 geliştirme checkpoint'i

Tamamlanan ana fazlar:

- Ortak `SurfaceField` semantik/veri sözleşmesi ve MSF integrity/mass özetleri.
- Yanmış malzemede integrity tabanlı fracture eşiği zayıflatma ve Jolt darbe köprüsü.
- Gas basınç darbesi -> yapısal impulse -> fracture zinciri.
- Kamera mesafesi ve global bütçe kontrollü ash/debris LOD sistemi.
- Plastik için katı, piroliz ve molten-reservoir kütle bütçesi.
- MSF molten reservoir -> APIC parçacık aktarımı; kütle yalnız başarılı spawn sonrası
  kaynaktan düşülür ve timeline cache bu durumu korur.
- Plastic, Wax, Water/Ice ve inert metal fluid kimya eşlemesi; yanıcılık ve viskozite
  hedef domaine otomatik aktarılır. Surface SDF render modu transferde hazırlanır.
- Rest mesh'ten türetilen, geri alınabilir bağlı yüzey erime deformasyonu; yerel havuz
  katı + henüz aktarılmamış molten kütleyi korur.
- Obje bazlı molten transfer, maksimum yükseklik kaybı/havuz yayılması ve dinamik SDF
  yenileme kontrolleri; Python/API ve proje save/load eşliği.
- Mesh SDF collider eriyen geometriyi eşik ve MSF revision aralığıyla asenkron yeniler.
  Eski bake yeni sonucu ezemez; worker sonucu canlı collider'a yazmak yerine ana
  simülasyon tick'inde yayınlanır.
- Vulkan timeline rewind güvenliği: paused frame değişiminde render fence alınır;
  sıfır parçacıklı cached karede fluid RT havuzu silinmez, görünmez tutulur. Böylece
  eriyen mesh + fluid + BLAS/TLAS geçişindeki yapısal rebuild/TDR baskısı azaltılır.
- Kapalı kaynak Mesh SDF'nin kendi molten parçacıklarını hapsetmesini önleyen
  yerçekimi çıkışı; alt taraf domain dışında ise en yakın uygun yan çıkış kullanılır.

Python kabul sahnelerinde Faz 0–6 sözleşme, integrity, fracture, pressure, ash LOD,
plastik kütle bütçesi ve APIC mass-transfer testleri PASS verdi. Derleme ve Vulkan
stres doğrulaması kullanıcı tarafında faz sonlarında yapılır; kısa aralıklı test
zorunluluğu yoktur.

Bir sonraki çalışma oturumunun açık maddeleri:

- Frame 0 her zaman pristine MSF/melt geometrisi olmalı; ortam koşulunun ilk fizik
  adımından önce görünür erime üretip üretmediği doğrulanacak ve cache sözleşmesi
  gerekirse düzeltilecek.
- Mevcut erime, gerçek Joule/entalpi bütçesi değil latent-heat sıralamasını koruyan
  rate yaklaşımıdır. Kısa patlama ile sürekli alevi toplam aktarılan enerjiye göre
  karşılaştıran entalpi birikimi tasarlanacak.
- Gravity outlet güvenli ilk çözümdür. Daha ileri sürümde molten yüzey üzerinde
  bağlantılı akış yolu, kenar/delik bulma ve birikim/taşma eşiği kullanılacak.
- Vulkan RT'de erimiş mesh + Surface SDF fluid ile hızlı rewind/Play/Solid->RT
  geçişleri sürücü stres testinden geçirilecek; gerekirse geçiş işleri karelere
  bölünecek, genel her-frame `waitIdle` eklenmeyecek.

## Temel kararlar

1. **Kimya MSF'de yaşar.** Jolt yanma veya erime hesaplamaz.
2. **Kütle tek kez harcanır.** Yanma ve erime aynı kütleyi iki kez tüketemez.
3. **Fluid devredilmiş sıvı kütlenin sahibidir.** Mesh üzerindeki `melt` henüz
   fluid kütlesi değildir.
4. **Jolt yalnız kopmuş makro parçaları taşır.** Kül, kıvılcım ve küçük kırıntılar
   particle sisteminde kalır.
5. **Runtime mesh kesme ilk sürüme girmez.** Görsel detay maskeden, fiziksel
   parçalanma önceden hazırlanmış shard/cluster'lardan gelir.
6. **Tam alan readback yapılmaz.** Köprüler seyrek özet ve olay kuyruklarıdır.
7. **Türetilmiş geometri geri alınabilir olur.** Rest mesh değişmez; sculpt,
   erime ve hasar katmanlı deformasyondur.
8. **Canavar dosya büyütülmez.** Mevcut bir kaynak dosya yaklaşık 2000 satırı
   geçmişse yeni alt sistem kodu ayrı `.h/.cpp` modülüne yazılır; büyük dosyada
   yalnız kısa deklarasyon, kayıt veya yönlendirme çağrısı kalır.

## Kullanıcı kontrolü ve UI sözleşmesi

Yeni bir malzeme dönüşümü özelliği yalnız solver/API tarafında tamamlanmış
sayılmaz. Kullanıcının hem tek nesneyi hem de bütün sahneyi yönetebilmesi gerekir.

### Obje bazlı kontroller

- MSF/substance seçimi, ignition, burn rate ve fuel capacity override,
- char/burn mask açma-kapama, çözünürlük ve obje hasarını temizleme,
- termal geometrik deformasyon açma-kapama, deformasyon şiddeti/limiti ve rest
  mesh'e geri dönme,
- integrity ile zayıflama, temel kırılma eşiği, exponent ve minimum eşik,
- melt, mass transfer, ash/debris üretimi için obje bazlı opt-in ve bütçe,
- salt-okunur canlı telemetry: sıcaklık, yakıt, char, melt, mass loss, integrity,
  etkin kırılma eşiği ve üretilen parça sayısı.

### Sahne bazlı kontroller

- ambient Kelvin, Kelvin/unit kalibrasyonu, konveksiyon ve oksijen,
- char/mask, termal deformasyon, fracture, ash/debris ve mass-transfer için global
  master switch'ler,
- sahne kalite/bütçe ayarları: toplam mask texel'i, deform edilen vertex bütçesi,
  aktif Jolt shard sayısı, ash particle bütçesi ve readback sıklığı,
- `Clear All Damage`, simülasyon/cache sıfırlama ve güvenli varsayılanlara dönme,
- toplam maliyet ve aktif/atlanmış obje sayılarını gösteren sahne telemetry'si.

Global anahtarlar obje ayarlarını silmez; yalnız runtime değerlendirmesini durdurur.
Obje ayarları proje save/load içinde kalıcıdır. Görsel char kapatıldığında kimya ve
yakıt tüketimi çalışmaya devam edebilir; geometrik deformasyon kapatıldığında da MSF
integrity/kütle verisi korunur. Böylece performans seçenekleri fiziksel durumu sessizce
yok etmez. Her ana fazın kabul testi Python API ile birlikte karşılık gelen obje ve
sahne UI kontrollerinin save/load davranışını da doğrular.

### Fracture geometri sınırı

Mevcut `FractureGenerator` kaynak mesh'in gerçek üçgen kabuğunu kesmez; convex hull
üzerinde Voronoi hücreleri üretir. Bu yalnız kutu, taş veya yaklaşık dışbükey test
varlıklarında kabul edilebilir. Su kulesi, ev, çatı, boru, içi boş ya da içbükey bir
asset üzerinde `Generate Shards` kullanıldığında boşluklar dolar ve sonuç kaba blok
geometrisine dönüşür. Flat/facade adaptörü yalnız kaynak üçgenleri görünür kılar;
bu convex-hull davranışını değiştirmez.

Üretim çözümü iki yoldan biri olacaktır:

- DCC/import aşamasında hazırlanmış gerçek shard asset ve cluster metadata'sı,
- kaynak yüzey üçgenlerini Voronoi hücreleriyle kırpan, açık kesitleri cap eden,
  dış yüzey material/UV'lerini koruyan exact mesh fracture (veya OpenVDB level-set
  fracture) modülü.

Exact fracture tamamlanana kadar karmaşık sahne varlıklarında otomatik Voronoi
kullanılmayacak; fracture UI bu durumu açıkça `Convex Hull Preview` olarak
etiketlemeli ve içbükey/hollow geometri için uyarı vermelidir.

## Ortak veri sözleşmesi

### Kalıcı kimlik

Geçici VDB id, GPU adresi veya vector index kimlik değildir.

```text
ObjectKey     = sahne objesinin kalıcı kimliği
MaterialSlot = objenin materyal bölümü
FieldId      = ObjectKey + MaterialSlot + semantic + version
TopologyGen  = SurfaceMeshCache topoloji nesli
```

Mevcut `source_name/object_key` geçiş anahtarı olarak korunabilir; uzun vadede
yeniden adlandırmadan etkilenmeyen scene UUID kullanılmalıdır.

### Genel alan başlığı

MSF kanallarını dev bir genel struct'a çevirmeyeceğiz. Ortak başlık ve semantik
kanal kataloğu kullanılacak:

```cpp
struct FieldHeader {
    ObjectKey owner;
    FieldSemantic semantic;
    FieldDomain domain;       // SurfaceUV, Vertex, VolumeGrid, Terrain2D
    FieldFormat format;       // UNorm8, Float16, Float32, Vec2, Vec3
    uint64_t topology_generation;
    uint64_t content_generation;
    DirtyRegion dirty;
};
```

| Grup | Kanallar |
|---|---|
| Termal/kimya | temperature, moisture, fuel_remaining, char, melt, mass_loss |
| Yapısal | integrity, fracture_damage, support, thickness |
| Paint | selection, material_mask, user_mask0..3 |
| Sculpt | displacement, stiffness, pin, erosion_resistance |
| Terrain | height, hardness, wetness, sediment, flow, fuel_load |
| Taşıma | released_fuel, released_smoke, molten_reservoir, ash_reservoir |

MSF termal/kimyasal kanalların uzman sistemi olarak kalır. Ortak katman yalnızca
kimlik, örnekleme, dirty takibi, cache ve erişim sözleşmesini sağlar.

### Alan uzayları

- `SurfaceUV`: paint, char, sıcaklık ve ince yüzey hasarı.
- `Vertex`: sculpt/deformasyon çıktısı.
- `VolumeGrid`: fluid/gas ve kapalı hacim süreçleri.
- `Terrain2D`: heightfield tabanlı terrain kanalları.

Uzay dönüşümleri açık operatörlerdir:

```text
SurfaceUV → Vertex       sample/max/average
SurfaceUV → VolumeGrid   conservative scatter
VolumeGrid → SurfaceUV   gather/contact sampling
Terrain2D → VolumeGrid   boundary/source projection
```

Hiçbir sistem diğerinin buffer düzenini doğrudan varsaymaz.

## Malzeme yaşam döngüsü

Her MSF elemanı için mantıksal kütle bütçesi:

```text
initial_mass = solid_remaining
             + char_mass
             + ash_mass
             + burned_released_mass
             + molten_transferred_mass
```

Her terimin ayrı texture olması gerekmez; mevcut kanallardan ve frame-local
reservoir değerlerinden türetilebilir. Telemetri toplam kütle hatasını raporlar.

Bir madde aynı anda birden fazla süreçte olabilir:

- Plastik: erir, pirolizle yanıcı gaz salar; fluid'e geçen damla yanmayı sürdürür.
- Ahşap: kurur, piroliz olur, char üretir, bütünlüğü azalır ve kül bırakır.
- Metal: yanmaz; ısınır, kızarır, erir ve molten fluid'e aktarılır.
- Kauçuk/mum: erime ve yanma eşzamanlıdır; viskoz sıvı yakıt üretir.

Deterministik çözüm sırası:

```text
ısı/nem → faz değişimi → piroliz → reservoir → integrity → çıktı olayları
```

## Düşük maliyetli kâğıt ve ağaç yanması

### Görsel katman

- Char maskesi renk, roughness ve normal ayrıntısını sürer.
- `mass_loss` coverage erosion üretir; kâğıtta delikler maskede büyür.
- Ahşapta damar yönlü çatlak, kül ve emissive kenar shader katmanıdır.
- BLAS/TLAS yalnız gerçek parça ayrıldığında değiştirilir.

### Yapısal özet

MSF tam alanı Jolt'a verilmez. Obje/cluster başına seyrek aralıkla şu özet çıkar:

```text
mean_integrity
minimum_integrity
remaining_support_ratio
weakest_world_position
released_mass_since_last_query
```

Statik ve soğuk objeler aktif listeden çıkar. Özet yalnız dirty cluster'larda ve
örneğin 6–12 sim frame'de bir hesaplanır.

### Parça LOD'u

- Büyük parça: hazır shard + Jolt rigid body.
- Orta parça: debris particle/collider.
- Küçük parça: GPU ash/ember particle; Jolt body yok.

Sınırlar kütle, ekran boyutu ve kamera mesafesine göre belirlenir.

## Basınç, patlama ve Jolt köprüsü

Mevcut darbe tabanlı fracture tüketicisine gas ve force-field kaynaklı olay da
gönderilir:

```cpp
struct StructuralImpulseEvent {
    ObjectKey target;
    Vec3 world_point;
    Vec3 impulse;
    float peak_pressure;
    DamageSource source; // Contact, GasPressure, Explosion, ForceField
};
```

Gas domain tüm voxel'leri okumaz. Collider SDF/OBB yüzeyindeki 6–24 sabit
örnekten basınç farkı entegre edilir:

```text
impulse += normal * max(local_pressure - ambient_pressure, 0)
           * sample_area * dt
```

İki eşik ayrılır:

- `launch_impulse`: obje kırılmadan savrulur,
- `fracture_impulse`: shard grubu dinamikleşir.

Termal hasar kırılma eşiğini düşürür:

```text
effective_fracture_impulse = base_impulse * integrity^exponent
```

Kırılma olayı tekilleştirilir; aynı frame içinde tekrar rebuild yapılamaz.
Patlama sonucu mevcut `pending_launch_velocity` yoluna çevrilir.

## MSF → Fluid kütle aktarımı

Her obje küçük bir `MoltenReservoir` taşır:

```text
mass, temperature, substance, mean_velocity,
weighted_spawn_position, source_object
```

MSF elemanları her frame parçacık doğurmaz. Reservoir eşik kütleyi geçince toplu
bir `MassTransferEvent` çıkarır. Hedef sırası:

1. spawn noktasını içeren uygun fluid domain,
2. kullanıcı tarafından bağlı domain,
3. domain yoksa reservoir'da bekleme veya render-only damla.

Fluid parçacığı substance/material id, sıcaklık/entalpi, viskozite, yoğunluk,
kalan yanıcı fraksiyon ve render material id taşır. Mesh kütlesi yalnız başarılı
spawn sonrasında azaltılır; domain yokluğu kütleyi yok etmez.

## Diğer sistemlerle yeniden kullanım

### Paint

- Substance, fuel load, moisture, fracture ve melt resistance boyanabilir.
- Paint layer doğrudan fizik buffer'ına yazmaz; flatten edilmiş sonuç semantik
  field kanalına publish edilir.
- Fizik cache imzasına paint `content_generation` değeri girer.

### Sculpt

- Sculpt rest mesh'i değiştiren authoring işlemidir ve `topology_generation`
  artırır.
- Melt slump ve yanma çökmesi runtime deformation layer'larıdır.

```text
rest mesh → authored sculpt → animation/soft body → thermal deformation → render
```

- Topoloji değişince MSF world-position/UV ile yeniden örneklenir; eski vertex
  index'i sessizce kullanılmaz.

### Terrain

- Terrain `hardness`, `wetness`, `fuel_load` ve `char` kanallarını aynı katalogla
  yayınlar; `Terrain2D` depolamasını korur.
- Orman yangını terrain fuel yükünü ve foliage instance maddesini tüketir.
- Isı sertliği düşürebilir; nem yangını bastırır; kül terrain layer'a birikir.
- Mevcut dirty-sector yapısı ortak `DirtyRegion` için referans olur.

### Fizik

- Jolt yalnız `integrity` özetini ve impulse olaylarını tüketir.
- Fluid/gas solver Jolt body yaşam döngüsünü doğrudan yönetmez.
- Collider proxy, kopmada intact gruptan shard grubuna atomik geçer.

## Scheduler ve performans

```text
1. Authoring/paint dirty alanlarını yayınla
2. World/domain sıcaklık ve nem gather
3. MSF kimya + faz değişimi
4. Gas/fluid conservative scatter/resolve
5. Alan özetleri ve olay kuyrukları
6. Jolt impulse/fracture tüketimi
7. MassTransferEvent → fluid particle batch
8. Derived geometry/render mask güncellemesi
9. Cache snapshot
```

Kurallar:

- sıfır dirty field = sıfır dispatch,
- readback yalnız küçük özet/event counter tamponlarında,
- particle spawn toplu ve eşik tabanlı,
- fracture yalnız olay anında topology/TLAS dirty eder,
- mask çözünürlüğü fizik doğruluğından bağımsız görsel kalite ayarıdır,
- uzak objelerde çözünürlük ve update frekansı düşürülebilir.

## Cache ve determinizm

Cache imzasına substance/override, paint generation, topology generation,
fracture asset, hedef domain bağlantısı ve eşikler girer.

Snapshot şunları taşır:

- MSF kalıcı kanalları,
- molten/ash reservoir durumları,
- kırılmış cluster bitset'i,
- Jolt shard pose/velocity,
- devredilmiş toplam kütle sayaçları.

Olaylar `(frame, source, monotonic_event_id)` ile tekilleştirilir; timeline scrub
aynı particle batch veya fracture olayını iki kez üretmez.

## Test stratejisi: ana faz kapıları ve script ile sahne üretimi

Kısa alt adımların sonunda tekrar tekrar manuel görsel test yapılmayacak. Bir ana
fazın veri modeli, yürütme yolu, UI/API bağlantısı ve telemetrisi tamamlandıktan
sonra tek bir **faz kabul testi** çalıştırılacak.

Faz içindeki doğrulamalar yalnız hızlı ve yerel kontrollerdir:

- struct/shader ABI boyut kontrolü,
- serialization round-trip,
- kütle dengesi ve NaN/limit assert'leri,
- event tekilleştirme sayaçları,
- dirty-region ve dispatch sayısı telemetrisi.

Bunlar kullanıcıdan sahne kurmasını veya uzun simülasyon bake'i istemez. Görsel,
fiziksel ve entegrasyon testi faz kapısında birlikte yapılır.

### Script-first kabul sahneleri

Her ana faz doğrudan Scripts panelinden yüklenip çalıştırılan bağımsız bir Python
test dosyasına sahip olacak. Addon kurulumu veya enable adımı gerekmeyecek. Script:

1. boş veya bilinen bir sahneden başlar,
2. geometri, materyal ve substance profillerini oluşturur,
3. collider, MSF, gas/fluid domain, Jolt ve fracture bağlantılarını kurar,
4. kamera/ışık ve deterministik seed/frame aralığını ayarlar,
5. simülasyonu çalıştırır veya bake komutunu başlatır,
6. telemetri/API sonuçlarını kontrol eder,
7. sahneyi tekrar açılabilir test varlığı olarak kaydeder.

Bu yaklaşım iki ürünü aynı anda test eder:

- fizik/render özelliği doğru çalışıyor mu,
- aynı özellik Python API ve addon sistemiyle eksiksiz kurulabiliyor mu?

Bir ayar yalnız UI'da bulunuyor ve script tarafından yazılamıyorsa faz tamamlanmış
sayılmaz. Eksik binding önce API → IPC/capability → addon yüzeyine eklenir; test
sahnesinde özel C++ kestirmesi kullanılmaz.

### Test varlığı düzeni

```text
scripts/material_phase_tests/
    common.py
    phase_00_contract_scene.py
    phase_01_integrity_scene.py
    phase_02_fracture_scene.py
    phase_03_pressure_scene.py
    phase_04_ash_lod_scene.py
    phase_05_plastic_scene.py
    phase_06_mass_transfer_scene.py
    phase_07_field_integration_scene.py
```

`common.py` yalnız sahne temizleme, kamera/ışık, seed, frame aralığı ve raporlama
gibi test altyapısını paylaşır. Fizik özelliğini gizleyen özel davranış içermez.

Her script sonunda makinece okunabilir kısa bir sonuç üretir:

```text
PASS/FAIL
mass_initial, mass_remaining, mass_transferred, mass_error
active_msf_fields, dirty_dispatches
fracture_events, rigid_body_peak
fluid_particles_spawned, gas_fuel_released
cache_replay_match
```

Görüntü karşılaştırması tek otorite değildir; fizik sayaçları ve kütle dengesi
zorunludur. Referans ekran görüntüsü yalnız görünür regresyonları yakalamak için
faz kapısına eklenebilir.

### Ana faz kapısı çalışma düzeni

Bir faz şu sırayla kapatılır:

```text
implementasyon tamamla
→ binding/capability kapsamını denetle
→ doğrudan test scriptiyle sahneyi sıfırdan üret
→ kısa deterministik sim/bake çalıştır
→ sayaç + kütle + cache replay doğrula
→ görsel sonucu kontrol et
→ fazı commit et
```

Derleme ve uzun bake kullanıcı tarafından çalıştırılabilir; script sonuç ve log
formatı sorunun hangi katmanda olduğunu uzaktan incelemeye yeterli olmalıdır.

## Fazlar ve kabul ölçütleri

### Faz 0 — Veri sözleşmesi ve telemetri

- `ObjectKey/FieldHeader/FieldSemantic/DirtyRegion` tasarımı.
- MSF ABI'sini bozmayan adapter katmanı.
- Kütle dengesi ve dirty-field sayaçları.

**Faz kapısı:** script ile aynı objeye paint/MSF alanı bağlanır, save/load yapılır;
kimlik ve generation değerlerinin korunduğu raporlanır. Mevcut yanma/cache sonucu
değişmeden alanlar inspector'da izlenebilir.

### Faz 1 — Integrity ve ucuz görsel erozyon

- `integrity` türetimi.
- Kâğıt coverage deliği; ahşap char/çatlak yalnız render maskesinde.

**Kabul durumu (2026-08-09): geçti.** `phase_01_integrity_scene.py` ile kâğıt ve
ahşap alanlarında GPU `mass_loss` readback'i, azalan integrity ve render maskesi
doğrulandı. İlk coverage görünümü bilinçli olarak düşük maliyetli/prototip kaldı;
kâğıt hızlı ve bloklu şeffaflaşıyor.

Kabul testinde mevcut MSF yayılımının zaten alevin temas ettiği bölgede ilerlediği
ve `msf_mask_resolution` yükseldikçe uzamsal ayrıntının arttığı doğrulandı. Bu
nedenle ayrı bir ağır fraktal/temas sistemi eklenmeyecek. Görsel iyileştirme,
mevcut alanın yanma cephesinde önce sıcak kırmızı/emissive bir piroliz hattı,
arkasında siyah char ve sonrasında coverage kaybı üretmesiyle sınırlı tutulacak.

Öncelikli düzeltme kâğıdın kısa yakıt ömrüdür: yerel `fuel_remaining` hızla
tükenmeli ve sıfıra geldiğinde yüzey yeni yakıt/alev üretmemelidir. Test sahnesinin
sürekli çalışan pilot `flow_source` alevi, kâğıdın kendi yanması olarak
değerlendirilmeyecek; kabul testi pilotu kısa bir ateşleme süresinden sonra kapatıp
alevin yalnız kâğıdın kalan yakıtıyla sürmesini ve ardından sönmesini doğrulayacak.
Topoloji ve BLAS yalnız gerçek parça ayrılmasında değişecek.

Performans kararı: integrity/mass-loss değişimi vertex deformasyonu üretmeyecek.
İlk ahşap çökme denemesi her MSF jenerasyonunda BLAS refit/rebuild tetiklediği
için kaldırıldı. Makro şekil değişimi Faz 2'de yalnız gerçek fracture/cluster
olayında yapılacak; Phase 1 boyunca yanma yalnız texture/instance verisi günceller.

**Faz kapısı:** `phase_01_integrity_scene.py` kâğıt ve ahşap örneklerini üretir.
Topology rebuild ve Jolt body olmadan ilerleyen yanma; integrity ve kütle sayaçları
ile birlikte doğrulanır.

### Faz 2 — Termal fracture

- Cluster özetleri ve integrity ile eşik düşürme.
- Hazır shard'lı ahşap test varlığı.

**Uygulama durumu:** MSF tam alanı Jolt'a verilmeden sekiz fizik çağrısında bir
`mean_integrity`, `minimum_integrity`, `remaining_support_ratio`, en zayıf dünya
konumu ve toplam kütle kaybı özeti üretilir. Mevcut contact impulse tüketicisi ve
script `apply_fracture_impulse` aynı merkezî eşik fonksiyonunu kullanır. Fracture
ayarları proje dosyasında saklanır; runtime `broken` durumu cache/re-sim tarafından
yeniden türetilir. Uygulama modüler `ThermalFracture*` ve `Rt*Fracture` dosyalarında
tutulur.

**Kabul durumu (2026-08-09): geçti.** `phase_02_fracture_scene.py` 134 frame
sonunda aynı `3.4361` impulse'u iki eş ahşap gruba uyguladı. Yanmış örneğin
`mean_integrity` değeri `0.6952`, etkin kırılma eşiği `2.8722` oldu ve kırıldı;
domain dışında tutulan sağlam kontrolün integrity/eşiği `1.0 / 4.0` kaldı ve
kırılmadı. Böylece MSF integrity → termal fracture eşiği → ortak impulse tüketici
zinciri doğrulandı. Bu kapı eşik sözleşmesini tek shard ile sınar; üretim çoklu
Voronoi shard ayrılması sonraki geometrik kabul testinde ayrıca doğrulanacaktır.

**Faz kapısı:** `phase_02_fracture_scene.py` sağlam ve önceden yakılmış iki eş
ahşap obje üretir. Aynı darbede yalnız zayıflamış obje kırılır; rewind/replay aynı
frame ve shard hızlarını üretir.

### Faz 3 — Gas pressure → Jolt

- `StructuralImpulseEvent` kuyruğu.
- Seyrek yüzey örnekli basınç integrasyonu.
- Launch/fracture eşiklerinin ayrılması.

**Uygulama durumu:** `gas.pressure_pulse` domain içinde lokal radial gas hız/basınç
darbesi üretir ve aynı kaynağı küçük bir `StructuralImpulseEvent` olarak kuyruğa
yazar. Jolt köprüsü tam pressure alanını CPU'ya indirmez; fracture gruplarını olay
yarıçapında bir kez örnekler, mesafe falloff'u uygular ve mevcut merkezi fracture
impulse tüketicisini çağırır. Normal combustion/alev bu kuyruğa olay yazmaz.
Python ve IPC telemetry'si queued/consumed/affected/fractured sayaçları ile son
tepe basıncı ve maksimum yapısal impulse'u verir. Faz kapısı kullanıcı derlemesi
sonrası bekleniyor.

**Kabul durumu (2026-08-09): geçti.** `phase_03_pressure_scene.py` normal alev
boyunca `0` structural event üretti. Ardından tek `500 kPa` pressure pulse kuyruğa
girdi ve bir kez tüketildi (`queued=1`, `consumed=1`). Olay bir fracture grubunu
etkiledi; mesafe falloff'u sonrası `6.25` impulse, sağlam hedefin `4.0` eşiğini
aştı ve yalnız o grubu kırdı (`affected_groups=1`, `fractured_groups=1`,
`broken_count=1`). Böylece normal combustion ile yüksek basınçlı yapısal olayın
ayrımı ve gas → event queue → merkezi fracture impulse zinciri doğrulandı.

**Faz kapısı:** `phase_03_pressure_scene.py` normal alev ve yüksek basınçlı patlama
senaryolarını yan yana kurar. Yalnız patlama launch/fracture olayı üretir ve impulse
büyüklüğü script raporunda görünür.

### Faz 4 — Kül ve debris LOD

- `AshReservoir`, toplu spawn ve otomatik LOD downgrade.

**Uygulama durumu:** Küçük yanma/kırılma kütlesi `AshDebrisSystem` üzerinden mevcut
discrete particle SoA'sına toplu aktarılır. Yakın olaylar authored yoğunlukta,
uzak olaylar `far_lod_scale` ile daha az fakat toplam kütleyi temsil eden aggregate
particle üretir. `max_particles` sert sahne bütçesidir; bütçe üstü görsel ayrıntı
reddedilir ve requested/spawned/LOD-reduced/budget-rejected sayaçlarında raporlanır.
Makro fracture shard'ları mevcut Jolt yolunda kalır; bu katman onlar için yeni rigid
body üretmez. Faz kapısı kullanıcı derlemesi sonrası bekleniyor.

**Kabul durumu (2026-08-09): geçti.** `phase_04_ash_lod_scene.py` üç debris olayı
için toplam `120` ayrıntı parçacığı talep etti. Yakın `0.1 kg` olay `10`, uzak eş
kütleli olay LOD ile `3`, bütçeyi aşan `1.0 kg` olay kalan `7` slotu kullandı.
Toplam canlı/spawned particle sert `20` bütçesinde kaldı; `7` örnek uzaklık LOD'u
ile azaltıldı, `93` bütçe üstü ayrıntı reddedildi ve toplam `1.2 kg` kütle aggregate
particle'larla temsil edildi. Sahne ağacına yeni mesh/Jolt nesnesi eklenmemesi ve
makro shard yolunun ayrık kalması doğrulandı.

**Faz kapısı:** `phase_04_ash_lod_scene.py` yakın/uzak ve farklı kütleli parçalar
üretir. Jolt body bütçesi aşılmaz; downgrade ve kül particle sayıları raporlanır.

### Faz 5 — Plastik: eşzamanlı erime ve yanma

- Tek bütçede melt + pyrolysis paylaşımı.
- Molten reservoir ve render-only damlama prototipi.

**Uygulama durumu:** MSF elemanının başlangıç combustible kapasitesi artık tek
kütle otoritesidir. `mass_loss` yalnız pirolizle harcanan kütleyi, `melt` aynı
başlangıç kütlesinden molten reservoir'a ayrılan oranı taşır. Gather shader her
adımda melt payını kalan bütçeyle sınırlar ve `solid + pyrolyzed + molten <= initial`
invariantını korur. Obje bazlı Python/IPC telemetry başlangıç, solid, pyrolyzed,
molten reservoir ve conservation error değerlerini raporlar. APIC'e gerçek kütle
devri yapılmaz; reservoir Faz 6'nın girişidir. Faz kapısı kullanıcı shader/C++
derlemesi sonrası bekleniyor.

**Kabul durumu (2026-08-09): geçti.** `phase_05_plastic_scene.py` plastik yüzeyi
36 frame içinde aynı anda eritti ve piroliz etti. Başlangıç kütlesi `7.6050000`,
kalan solid `6.5458536`, pyrolyzed `0.0275604` ve molten reservoir `1.0315862`
olarak ölçüldü. Hesaplanan toplam `7.6050002`, raporlanan conservation error `0.0`
oldu. Böylece melt ve pyrolysis'in aynı başlangıç bütçesini iki kez harcamadığı ve
molten reservoir'ın APIC devrinden önce izlenebilir kütle taşıdığı doğrulandı.

**Faz kapısı:** `phase_05_plastic_scene.py` plastik objeyi tek ısı kaynağıyla hem
eritir hem piroliz eder. Fuel, smoke, molten reservoir ve toplam kütle hatası aynı
raporda doğrulanır.

### Faz 6 — MSF → APIC mass transfer

- `MassTransferEvent`, domain seçimi ve kimya taşıyan particle batch.
- Başarılı spawn sonrası mesh kütle kaybı.

**Uygulama durumu:** İşlemsel aktarım çekirdeği yazıldı. İstek, objeyi içeren
uygun Fluid domain'i seçer; APIC sert kapasitesi kadar particle üretir ve yalnız
gerçekten kabul edilen batch için molten reservoir'ı debit eder. Domain veya
kapasite yoksa istek ertelenir ve rezervuar değişmez; debit/upload başarısızlığında
eklenen particle'lar geri alınır. Particle sidecar'ı Kelvin sıcaklığı,
combustible fraction ve substance tag taşır. `transferred_mass` MSF/cache kütle
sözleşmesine eklendi. Combustible yüzeyler authored fuel kapasitesini, meltable
non-combustible yüzeyler ise yoğunluk × 1 mm yüzey kabuğunu kütle tabanı olarak
kullanır; böylece metal transferi yakıt uydurmadan ölçülebilir. Python kapısı
`rt.mass_transfer.queue/stats` ile hazır; kullanıcı derleme testi bekleniyor.

**Faz kapısı:** `phase_06_mass_transfer_scene.py` plastik ve metal örneklerini
uygun APIC domain üzerinde kurar. Başarılı spawn, mesh kütle azalması, taşınan
sıcaklık/kimya ve plastik damlasının fluid yangınıyla yanması doğrulanır.

### Faz 7 — Katılaşma ve paint/terrain entegrasyonu

- Molten deposit/katılaşma yaklaşımı.
- Boyanabilir substance/fuel/moisture/integrity.
- Terrain fuel, wetness, ash ve hardness bağlantıları.

**Faz kapısı:** `phase_07_field_integration_scene.py` boyanmış fuel/moisture
maskesini, sculpt edilmiş mesh'i ve terrain fuel/wetness alanını aynı sahnede
üretir. Mesh paint, sculpt, terrain ve fizik aynı field sözleşmesini kopya veri
modeli oluşturmadan tüketir; save/load ve cache replay sonucu eşleşir.

## İlk dikey testler

1. Kâğıt levha: ilerleyen delik ve kül; sıfır/çok az Jolt body.
2. Ahşap sandık: yanma → integrity kaybı → darbe/patlamada shard ayrılması.
3. Plastik şişe: erime + piroliz → yanan viskoz damla.
4. Demir çubuk: kızarma → gizli ısı platosu → molten APIC aktarımı.
5. Orman parçası: terrain fuel/wetness + foliage maddesi + düşük maliyetli kül.
6. Basınç testi: sağlam, yanmış ve ıslak ahşabın farklı kırılma davranışı.

## İlk sürümde kapsam dışı

- Runtime Voronoi/fraktal mesh kesme,
- her kül tanesi için rigid body,
- tam kimyasal reaksiyon ağı ve oksijen CFD'si,
- katı-sıvı için her frame yeniden meshleme,
- solver'lar arasında tam alan CPU readback'i.

İnce detay alanlarda, taşınan kütle fluid/particle'da, makro hareket Jolt'ta kalır.
