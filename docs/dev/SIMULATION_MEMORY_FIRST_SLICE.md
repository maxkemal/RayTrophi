# Ortak simülasyon belleği — ilk uygulama: parçacık etiketleme

2026-11-06. Kaynak hazır; kullanıcı C++/shader build'i ve canlı kabul bekleniyor.
MPM/fluid/DEM'in ortak particle-label core'u değiştirilir; solver konum/hız,
deformasyon, DEM temas geçmişi, fizik eşikleri veya canonical geometri değiştirilmez.

## Sahne ve tahsis envanteri

Kullanıcının açık Snow sahnesi, 100 000 carrier, 222³ grid. İlk sorguda kare 0,
compute buffer henüz yoktu (driver 2 467 262 464 byte). Bir dış IPC
`fluid.step(dt=1/24)` çağrısı yapıldı; sonra sahne kare 1'de duraklatıldı.
Reset/seed/cache clear yapılmadı. Ajan uygulama veya build başlatmadı.

| Grup | Sahip / yaşam süresi | Ölçülen tahsis | Durum |
| --- | --- | ---: | --- |
| `GridDomain*` | domain/phase GPU çalışma kümeleri | 4 213 178 496 byte | üçlü kümeler; owner/lifetime incelemesi sonraki iş |
| `FluidLabel*` | domain GPU classifier scratch, her çağrıda yeniden kurulur | 1 343 357 040 byte | bu değişiklik |
| `matter_face_mass_gradient` | mixed MPM lane scratch | 791 303 904 byte | sonraki lazy-allocation adayı |
| Diğer compute | canonical particle/granular sidecar ve küçük scratch | yaklaşık 153 MB | bu değişiklikte korunur |

Yeni gerçek adım sonrası toplam driver kullanım 8 956 833 792,
compute device-local allocation 6 501 072 848 byte. Etiketler GPU'da:
100 000 body, unknown/spray/mist/frozen 0; occupied bins 5947,
center_resolved 98657. Label wall time 0,6959 ms. Bunlar yeni yapının
sonucu değil, mevcut executable ile alınan değişiklik öncesi tabandır.

Bir sonraki inceleme için kaynak bulgusu: `MatterGpuStep.inl` içinde
`view.mass_gradient_wanted = has_fluid` iken gradient tahsisi iki lane/üç
eksende yine yapılıyor. Snow granular-only durumda bu 6 büyük buffer'ın
tam boyutunun gerekli olup olmadığı binding/kullanım yollarıyla doğrulanmalı;
bu turn'de serbest bırakılmadı. Üçlü `GridDomain*` tahsisleri de henüz leak veya
gereksiz kopya diye sınıflandırılmadı.

## Sorun ve yeni düzen

Eski GPU classifier, her bin için 64 indeks ayırıyordu. Boş binler de aynı
kapasiteyi alıyor ve allocator %50 büyüme payı bırakıyordu. Bu sahnede
`FluidLabelBinItems` tek başına **1 322 093 184 byte**.

Yeni düzen:

- Bin başına yalnız **bir head indeksi**, empty=`0xffffffff`.
- Parçacık başına **bir next indeksi**, boyut `N * sizeof(uint32_t)`.
- Scatter, atomik head exchange ile her uygun taneyi kendi bin'ine bir kez
  bağlar. Scatter/classify arasındaki mevcut compute barrier bütün next
  yazılarını görünür yapar; publication kısmi zincir okumaz.
- Classify aynı 27 komşu bin'i ve geometrik yarıçapı kullanır. 64 eleman tavanı
  yoktur. Altı komşu bulunduğunda erken dönüş, 2/6 eşikleri ve 3..5 komşuda
  body/spray hysteresis korunur. Frozen/mist önceliği ve unrelated flag bitleri
  eski GPU yoluyla aynıdır. Links canonical veri değil, yeniden kurulan scratch'tır.
- Hatalı indeks veya bitmeyen zincir GPU sonucunu reddeder; host flags ancak
  geçerli readback/istatistikten sonra yayımlanır. Mevcut CPU fallback sürer.
- Buffer capacity gerekenden çok büyük kalmışsa (>2x, minimum 4 KiB bazında)
  yeniden ayrılır. Küçük artışlarda mevcut %50 büyüme payı korunur.
- Büyük bin sayılarında dispatch 2D'ye katlanır; 65535 x-group sınırı korunur.

Bu sahnede beklenen:

| Buffer | Eski | Yeni, hesaplanan |
| --- | ---: | ---: |
| Counts/Heads | 20 657 712 byte | 20 657 712 byte |
| Items/Next | 1 322 093 184 byte | 600 000 byte |
| Flags | 600 000 byte | 600 000 byte |
| Stats | 6144 byte | 6144 byte |
| Toplam | **1 343 357 040 byte** | **21 863 856 byte** |

Beklenen buffer tahsis azalması **1 321 493 184 byte** (~1,32 GB). Bu henüz
canlı VRAM kazancı değildir; driver residency/diğer sahne tahsisleri ayrı ölçülür.
Bin head tablosu hâlâ grid hacmiyle büyür; bu ilk sürüm tam sparse hash değildir.
Önce 64x rezervasyonu kaldırarak büyük ve ölçülebilir hedefe odaklanır.

## Güvenli shader geçişi ve ortak yüzeyler

Yeni entry point'ler:

- `sim_fluid_label_link_clear` — 2 binding / 80-byte PC
- `sim_fluid_label_link_scatter` — 5 binding / 80-byte PC
- `sim_fluid_label_link_classify` — 6 binding / 80-byte PC

Legacy adlar bu küçük next-buffer'a hiçbir zaman dispatch edilmez. Yeni SPIR-V
eksikse dispatch başarısız olur ve CPU fallback kullanılır; eski fixed-slot
shader'ının N boyutlu buffer'a yazmasına izin verilmez. PC padding[0] linked ABI=1;
clear/scatter/classify başarı bitleri toplamı 7 olmadan host flags yayımlamaz.
DEM shader revision 38 bu değişiklikten etkilenmez.

Yeni UI-only mantık veya solver'a özel kopya yok. Mevcut `fluid.step`, label
readout'ları ve Python/IPC aynı güncellenmiş core'u kullanır. Yeni authoring
ayarı eklenmedi. Sonuç `fluid.get/list_domains particle_labels.last_step` ve
`perf.get_gpu_memory.simulation.buffers` ile iki API'den de okunur. Yeni isimler
`FluidLabelBinHeads` ve `FluidLabelNext`.

## Kontroller ve kullanıcı build/kabul listesi

Ajan C++ veya shader compilation çalıştırmadı.

1. C++ ve üç **yeni linked-label shader** birlikte derlenmeli. Genel
   `RayTrophiStudio/compile_shaders.bat` dosyaları otomatik toplar/deploy eder;
   `source/shaders/compile_sim_shaders.bat` da yeni adları içerir. Uygulama yeni
   pipeline'ları yüklemesi için yeniden başlatılmalı. Project/filter girişleri eklendi.
2. `check_fluid_label_linked_contracts.py` geçti: 128 üyeli tek bin, negatif ve
   komşu hücreler, rastgele atomik insert/ertelenmiş next-write sırası, bağımsız
   O(N²) geometri oracle'ı, hysteresis, storage/ABI/publication kontrolleri.
   Bu SPIR-V çalıştıran bir GPU testi değildir. Mevcut DEM source kontratı da geçti.
3. Build sonrası ayrı boş test sahnesinde mevcut
   `rt_test_fluid_label_gpu_fallback_ipc.py` çalıştırılmalı. Legacy isme rağmen
   artık >64/bin vakasının **GPU'da** geçmesini doğrular; unknown=0, doluluk
   eski sınırı aşıyor ve on_gpu=true bekler. Mevcut labels/cache/views/mist
   testleri ve `fluid_particle_labels_test.cpp` CPU oracle regresyonu da ilgili.
4. Kullanıcının Snow sahnesinde bir **gerçek yeni adım** sonrası on_gpu=true,
   complete primary labels ve hata yokluğu doğrulanmalı. Sonra
   `python scripts/test/probe_simulation_memory_ipc.py` ile yaklaşık 22 MB
   label tahsisi ve eski 1,32 GB `FluidLabelBinItems` tahsisinin yokluğu ölçülmeli.
   Cache oynatımı veya hiç classifier çalışmamış kare 0 kıyas olarak kullanılmaz.
5. Bu ilk işin hedefi VRAM'dir. Eski classifier zaten ~0,7 ms olduğundan MPM'in
   bütün adım süresinde büyük hızlanma iddiası yok. >64/bin durumda CPU fallback'in
   önlenmesi ayrı performans kazancı olabilir; ölçülmeli.

Fog Spread bağlantısı, ortak transient pool ve üçlü grid sahipliği bu değişikliğin
kapsamında değildir; sonraki işler olarak açık kalır.

## Build sonrası canlı tahsis doğrulaması — 2026-11-06

Kullanıcının açık Snow sahnesinde dış IPC ile salt okunur ölçüldü. Adım, seed,
reset, cache silme veya ayar değişikliği yapılmadı.

| Ölçüm | Önceki ölçüm | Güncel ölçüm |
| --- | ---: | ---: |
| Compute tahsisleri | 6 501 072 848 | 5 179 579 664 byte |
| Label tamponları toplamı | 1 343 357 040 | 21 863 856 byte |
| Driver GPU kullanımı | 8 956 833 792 | 7 275 245 568 byte |

Compute farkı tam **1 321 493 184 byte**; beklenen label tasarrufu canlı tahsis
listesinde gerçekleşti. `FluidLabelBinHeads` 20 657 712, `FluidLabelNext` 600 000,
flags 600 000, stats 6144 byte; eski büyük `FluidLabelBinItems` bulunmuyor.
Driver farkı diğer renderer/sahne durumları da değiştiği için yalnız bu düzeltmeye
atfedilemez. Cache 26 kare / 873 114 031 byte RAM; bu VRAM değildir.

Kalan compute yükü: GridDomain/DenseMac tamponları önceki gibi yaklaşık 4,21 GB;
`matter_face_mass_gradient` 6 tampon / 791 303 904 byte. Üç grid kopyasının sahiplik
ve yaşam süresi hâlâ incelenmeli; yalnız isim/adet üzerinden sızıntı denemez.

`perf.list` son kayıtlarında timeline step 2200,13 ms, cache capture 420,33 ms,
fog render sync 176,88 ms. Bunlar aynı yeni adımdan alınmış kontrollü profil değil;
oturumdan kalan son kayıtlar. Şu an domain label last_step sıfır, primary unknown
100000 ve settle_pending=true bildiriyor. Bu nedenle mevcut okuma shader'ın yeni
bir adımda doğru sınıflandırdığını veya bir performans regresyonunu kanıtlamaz.
Tahsis kazancı doğrulandı; yeni gerçek adım/kernel profili ayrı doğrulanmalı.

## Play / cache sınırında bekleme araştırması

Kullanıcı bu beklemenin label optimizasyonundan önce de olduğunu bildirdi:
ilk Play ve pause sonrası cache dışına devam ederken uzun ana-thread beklemesi.
Kaynakta önemli mekanizma bulundu (`scene_data.h`):

- `kSimCacheVelocityKeyframeStride=25`; ara cache kareleri render-only olabilir.
- Restore sırasında `has_velocity=false` görülürse `sim_live_velocity_valid_`
  false olur. Devam ederken önceki hız keyframe'i restore edilir ve hedefe kadar
  yeniden simüle edilir. Salt pause/resume, geçerli canlı durum korunuyorsa bu
  rewind yolunu zorunlu olarak tetiklemez; scrub/restore durumuyla ayrılmalı.
- `kMaxStepsPerTick=8`: tek UI turunda sekiz pahalı sim adımı ve her biri için
  cache capture yapılabilir. 2,2 s/adım yalnız kaba örneklemesiyle 8 adım ~17,6 s
  eder; bu kullanıcı olayında ölçülmüş toplam veya kesin kök neden değildir.
- Capture host state sync, grid inventory snapshot ve compression yapar.
- Yeni seed'de `settleGridDomains` aynı çağrı içinde tekrar `stepGridDomains`
  çalıştırır; limit domain ayarından gelir (açık sahnede 240). Mevcut salt okunur
  durumda settle_pending=true, settle_ran=false; gerçek settle süresi yok.

Bu incelemede `fluid.step_stats(domain='Physics Domain 1')` measured=false
döndürdü. Bu nedenle mevcut durumda tahsis/upload/BVH/solver ayrımı sayısal olarak
tamamlanmış değildir. Önceki oturum kayıtlarında solid RT rebuild son 8,72 ms,
raster geometry 26,24 ms iken sim step 2200,13 ms idi; bunlar aynı olaydan alınmış
profil değildir ve BVH'nin her koşulda ucuz olduğunu kanıtlamaz.

Sonraki kontrollü ölçüm ilk Play, canlı pause/resume ve render-only cache'den
resume'u ayrı örneklemeli; rewind başlangıç/hedef/gerçek adım sayısı, settle,
buffer create/reuse, host upload/download, cache capture, render sync ve AS
update sürelerini birlikte kaydetmeli. Cache hız stride'ını hemen 1'e indirmek
RAM maliyetini artırır; sırf UI beklemesi için fizik dt'si değiştirilmemeli.
