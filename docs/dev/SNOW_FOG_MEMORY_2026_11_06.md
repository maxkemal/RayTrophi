# Snow Fog yayını ve VRAM — 2026-11-06

Etiketleme VRAM'inin ilk kaynak optimizasyonu ve yeni kabul listesi:
[SIMULATION_MEMORY_FIRST_SLICE.md](SIMULATION_MEMORY_FIRST_SLICE.md).

## Açık sahne ölçümü

Çalışan executable revision 37; aktif liste (revision 38) henüz kullanıcı
tarafından derlenmemişti. Aşağıdakiler salt okunur dış IPC ile alındı;
reset, seed, step, cache clear veya authoring değişikliği yapılmadı.

- `Physics Domain 1`: Snow/MPM snow, Vulkan, 100 000 carrier, 222³ grid,
  voxel 0,0225225 m. Grid gas+granular lane'leri açık, kaynak bütçesi
  18 688 MiB, guard açık. Başlangıçta fog rotası body/spray dahil bütün
  100 000 taneyi seçiyordu. Daha sonraki sorguda kullanıcı splat'a geçmişti.
- `fluid.inspect_fog_fields`: `Fog inspection snapshot exceeds diagnostic memory budget`.
  Render ve viewport volume slot sayısı 0; yayımlanmış fog/surface VDB yok.
- Ortak `prepareMatterFogSnapshot`, publisher tarafından da inspection'ın
  128 MiB sınırıyla çağrılıyordu. 222³ * 9 float = 393 877 728 byte
  (375,63 MiB), selection hariç; bu yüzden builder/VDB kaydından önce reddediliyor.
  Splat/SDF'nin çalışması bu bağımsız Fog yayın hatasını çözmüyor.
- Sürücü device-local kullanım: **9 816 973 312 byte** (9,82 GB).
  Render tracked 894 527 168, viewport tracked 357 258 784 byte;
  toplam tracked 1 251 785 952. **8 565 187 360 byte untracked**.
  Untracked, doğrudan sim compute tahsisleri yanında CUDA/driver/staging gibi
  başka kullanımları da içerebilir; tamamı simülasyon veya leak diye sunulmadı.
- Cache: 160 RAM frame, 5 518 214 145 byte RAM, baking false, valid false.
  Bu RAM sayısı driver VRAM sayısından bağımsızdır. Cache öncesi kullanımın
  tarihsel ölçümü elimizde yok; kullanıcı o aşamada da yaklaşık 9 GB bildirdi.

## Kaynak düzeltmeleri — build/canlı doğrulama bekliyor

1. **Fog publication/inspection bütçeleri ayrıldı.** Inspection varsayılan
   128 MiB kalır. Publication authored domain bütçesinin en fazla 1024 MiB'ını
   geçici Fog snapshot/field işi için kullanabilir. Guard kapalı veya budget=0
   expert durumunda da bu render işinin 1 GiB tavanı kalır. Sabit VRAM rezervasyonu
   değildir; yalnız gereken dense host alanları ayrılır. Simülasyon çözünürlüğü
   veya domain ayarları değiştirilmez.
2. Snapshot conversion scratch'ı çıktıktan sonra builder'a yalnız retained
   density/temperature/source alanlarından **kalan** bütçe verilir; snapshot ve
   builder aynı tavanı iki kez tam kullanamaz. Hidden gas için gereksiz gas
   snapshot ön-hesabı yapılmaz. Çok büyük grid/çok stream yine açık budget
   hatasıyla reddedilebilir. Bu tavan, tüm simülasyon/renderer prosesinin toplam
   bellek sınırı değildir; yalnız Fog hazırlama çalışma alanını sınırlar.
3. **Compute buffer census eklendi.** `perf.get_gpu_memory` ve
   `rt.perf.get_gpu_memory()` aynı core raporunda `simulation` döndürür:
   supported, scope, requested_bytes, allocation_bytes, device_local_bytes ve
   buffer adı/boyutu/property listesi. Vulkan `VkMemoryRequirements.size` gerçek
   tahsis boyutu olarak kaydedilir; logical/requested boyut ayrıca korunur.
   CPU/CUDA için destek yoksa supported=false. Query senkronize/download yapmaz.
4. Bu census yalnız **live compute buffers** kapsamındadır; pipeline/query pool,
   driver rezervleri ve ayrı staging pool'larının tamamını kapsadığı iddia edilmez.
   Host-visible + device-local aynı tahsistir; iki kez toplanmamalı. Eski
   `untracked_bytes` semantiği korunur ve simulation'ı içerir; yeni sim bytes'ını
   onun üstüne ekleyerek toplam bellek hesaplama.

VRAM azaltımı henüz yapılmadı: hangi buffer'ların baskın olduğu yeni census
ile ölçüldükten sonra gereksiz lane tahsisi/kapasite tutulması/leak ayrılacak.
Bu turn'de 9 GB'ın kesin kaynağı veya leak bulunduğu iddia edilmiyor.

## Doğrulama

- Ajan build/shader compilation veya uygulama açılışı çalıştırmadı.
- Fog/census değişiklikleri C++ tarafındadır. Önceki bekleyen aktif liste işi
  için ayrıca revision 38'in beş DEM shader'ı gerekir; bu bugfix yeni bir shader
  revision eklemez.
- `scripts/test/check_matter_fog_snapshot_contracts.py` ve mevcut DEM kaynak
  denetimi geçti. Header-only `matter_fog_snapshot_budget_test.cpp`, 222³ vaka,
  küçük domain bütçesi ve zero/unlimited/uint32-max sınırlarını kapsar; derlenmedi.
- C++ build sonrası aynı sahnede snow carrier rotasını Fog'a getir. Simulation
  reset gerekmez; normal render sync yeni VDB yayınını oluşturmalı. Kontrol:
  viewport ve render backend'lerinin `render.volume_slots` listelerinde gerçek
  slotlar, VDB panelinde bağımsız Matter Fog stream'i. Yüksek grid'de inspection
  API'sinin 128 MiB refusal'u hâlâ beklenebilir; bu artık publication refusal'u
  anlamına gelmez. Görüntü ve optik malzeme kabulü ayrıca gereklidir.
- Dış terminalden `python scripts/test/probe_simulation_memory_ipc.py` çalıştır.
  Read-only rapor buffer'ları ada göre toplar; yeni census yoksa açık rebuild
  hatası verir. Çıktıdan büyük buffer grupları, duplicate isimler ve toplam
  actual device-local allocation yorumlanacak. Otomatik cache/domain clear yok.

## Derleme sonrası canlı ölçüm ve Fog yayılım kontrolü

Kullanıcı Fog'un artık üretildiğini doğruladı. Salt okunur yeni census:

| Alan | Byte | Ondalık GB |
| --- | ---: | ---: |
| Driver device-local usage | 8974462976 | 8,974 |
| Renderer/viewport tracked allocation | 1330119536 | 1,330 |
| Live compute device-local allocation | 6501072848 | 6,501 |
| İki tahsis grubundan sonra kalan driver farkı | 1143270592 | 1,143 |
| Cache RAM, 18 frame | 590627959 | 0,591 |

Driver usage ve allocation toplamları farklı ölçümlerdir; driver farkı doğrudan
leak kanıtı değildir. Önceki 160 frame cache artık 18 frame'e düşmüş olsa da
baskın VRAM hâlâ canlı compute tamponlarında. Render ve viewport'ta birer
volume instance var; yayın yolu düzeltmesi canlı olarak doğrulandı.

Büyük compute grupları:

- `GridDomain*`: 114 buffer, 4213178496 byte. Pek çok kanal üç kez aynı
  boyutta mevcut. Örneğin `GridDomainDensity`: 3 x 43764224 byte. Bunların
  hangi owner/scratch yaşam süreleri olduğu henüz ayrıştırılmadı; üç kopya
  görülmesi tek başına gereksiz tahsis veya leak kanıtı değildir.
- `FluidLabel*`: 1343357040 byte. Bunun `FluidLabelBinItems` kısmı tek başına
  **1322093184 byte**. `FluidParticleLabelsGpu.cpp`, bütün domain'i 1,5 voxel
  yarıçaplı bin'lere bölüp HER bin için 64 indeks ayırıyor; ensureBuffer da
  %50 büyüme payı bırakıyor. Bellek dolu bin/parçacık sayısına göre değil,
  domain grid hacmine göre büyüyor. 100 000 tane için başlıca optimizasyon adayı.
- `matter_face_mass_gradient`: 6 buffer, 791303904 byte. MPM/face alanları
  yaşam süresi ve tekrar tahsis incelemesine dahil edilmeli.
- Diğer compute buffer'lar yaklaşık 153 MB; ilk üç grup toplam compute'ın
  büyük çoğunluğu. Bu turn'de tahsis azaltımı yapılmadı.

Fog boyutu sorusu için kaynak bulgusu:

- Şu anki Matter carrier yolu `splatFogDensityWeighted` ile çevredeki 8 hücreye
  trilinear deposition yapıyor; footprint voxel'e bağlı, fiziksel tane yarıçapı
  veya splat görsel yarıçapı bu çağrının girdisi değil.
- `fluid_fog_spread_voxels` / `fluid.set_fog(spread_voxels=...)` mevcut;
  legacy fog yolunda sigma 0..6 voxel olarak uygulanıyor. Ancak yeni
  `MatterFogSnapshot`/`MatterFogFields`/`MatterFogPublisher` zinciri spread'i
  okumuyor; sıcaklık splat'ına da açıkça 0 sigma veriliyor.
- `Fog Spread` paneli `!simulationDomainHasBothPhases(domain)` koşuluyla
  Matter'ın gas+carrier durumunda gizleniyor. Dolayısıyla mevcut çalışan
  Snow/Matter yolunda bu ayarı etkiliymiş gibi kullanıcıya tarif etmemek gerekir.
  Mevcut 1,5 değerine rağmen gerçek field ham footprint kullanıyor.
- `fog_resolution_multiplier` da bu yeni builder zincirine aktarılmıyor.
  Spread/shaping/resolution bağlanacaksa retained/gather/blur peak memory
  bütçeleri ve publisher'ın cache/signature invalidation'ı birlikte ele alınmalı.
  Bu turn soru/ölçüm kapsamında bu davranış değiştirilmedi.
