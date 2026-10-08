# Güncel kabul: T2 söküm partisi (Phase türet, Initial Model kalkar)

> **Durum:** AKTİF — 2026-10-08. Kaynak yazıldı; kullanıcı build alır. Descriptor ve denetim script'i geçti.

## Bu partinin kaynak değişikliği

- **Initial Model kalktı:** akış kaynağının `initial_constitutive_model` alanı API, IPC,
  Python, serileştirici, proje kaydı, UI ve kayıt karmasından çıktı. Doğum modeli yalnızca
  maddeden gelir (`MatterDomainSources.inl`). Eski sahnelerdeki anahtar okunmaz, sessizce atılır.
- **Bağlama modeli kalktı:** madde bağlamasının `constitutive_model` alanı ve
  `set_substance_material`'ın model parametresi kalktı (IPC, Python, UI, proje).
- **Bağlama `phase` yeniden etiketlendi:** artık yalnızca "statik katı blok" işareti.
  Sıvı/katı ayrımı sıcaklıktan türetilir. Alanın enum'u ve katı tag yolu değişmedi.
- **Hata B düzeltildi:** eritilemeyen madde için donma bayrağı artık hiç yazılmaz ve temizlenir
  (`FluidThermalLiquid.cpp`, `updateThermalFreeze`).
- **Betikler:** 9 test betiği (iki kopya) eski anahtarı göndermiyor. Descriptor'lar
  `gen_ipc_descriptors.py` ile yeniden üretildi; `audit_ipc_capabilities.py` OK.

## Sıra

1. **Build (kullanıcı):** Release x64. Değişen dosyalar: `FluidThermalLiquid.cpp`, `MatterDomainSources.inl`,
   `ParticleSimulation.h`, `RtApi.h`, `RtApiFluid.cpp`, `RtIpc.cpp`, `RtPython.cpp`, `RtIpcMethodDescriptors.cpp`,
   `SceneSerializer.cpp`, `ProjectManager.cpp`, `scene_data.h`, `scene_ui_simulation_domains.cpp`, `MatterModelControls.cpp`.
   ★ Derleme hatası için ilk bakılacak yer: `setFluidSubstanceMaterial` çağrı sayısı (7 argüman) ve
   kaldırılan alanlara kalan referanslar.
2. **Eski sahne açılışı:** `initial_constitutive_model` içeren bir proje açılmalı, hata vermemeli.
3. **Katı blok regresyonu:** Stone (statik katı bağlama) hâlâ akışı engellemeli. Donmuş bayrağı
   Stone parçacıklarında hiç görünmemeli (Hata B).
4. **Su ve buz:** su parçacıkları sıcaklığa göre donup erimeli; Water/Wax karışık domain davranışı T2a ile aynı.
5. **IPC beklenen ret:** `flow_source.create` içinde `initial_constitutive_model` artık bilinmeyen anahtar olarak
   reddedilmeli. Bu beklenen davranış, hata değil.

★ En sinsi: sahne hata vermeden açılır ama emitter'ın eski "fluid" zorlaması sessizce kalkmış olur.
Eski bir sıvı kaynağı artık maddenin modelini alır; maddenin varsayılanı granül ise davranış değişir.

## Açık

- Ice: granül mü rijit gövde mi (ölçümle karar).
- Katı bağlamaların donmuş-başlatılması kararı için: statik katı blok tag yolunda kalır, donmuş bayrağı
  değil (bkz. devir notu).
- Ertelenen ölçüm bulguları aşağıda.

---

## Önceki parti: T2a donma eşiği maddeden okunur

Güncel kabul: T2a donma eşiği maddeden okunur

> **Durum:** AKTİF — 2026-10-08. T2a kaynakta (C++ değişikliği, yeni dosya yok). Kullanıcı build alır; önce aşağıdaki sıra.

## Önce: ölçüm partisi (T2b-ölçüm), yüksek çözünürlük sweep'i

Kaynak: `RTPERF_FRAME_SCOPE` kapsamları eklendi — `sim.matter.gpu_partition_upload`
(`MatterGpuPartition.cpp`, model dizisi yüklemesi) ve `sim.matter.emit`
(`MatterDomainSources.inl`, kaynak başına doğum). Yeni dosya yok, vcxproj değişmedi.

1. **Build (kullanıcı):** Release x64. Değişen dosyalar: `MatterGpuPartition.cpp`, `MatterDomainSources.inl`.
2. **Sweep:** `python scripts/test/rt_matter_perf_sweep.py 30 10` (tek matter domain + su kaynağı açık).
   voxel 0.08 / 0.05 / 0.035 × ppc 4 / 8. Sonuç: bölüm başına ortalama ve en kötü kare.
   Hangi bölüm parçacık sayısıyla hızlanıyorsa (`sim.matter.gpu_partition_upload`,
   `sim.fluid.*`, `upd`/`rsync` dışı kalan süre) o darboğazdır.
   ★ En sinsi: sahne küçükken her şey makul; yalnız en ince voxel'de bir bölüm
   doğrusal değil, karesel büyürse fark edilir. Sweep'i küçük sahnede değil bu
   aralıkta çalıştırın.
3. **Kayıt:** çıktıyı `docs/dev/` altına ölçüm kanıtı olarak kaydedin ve devir notunu güncelleyin.

## Bu partinin kaynak değişikliği

- `FluidThermalLiquid.cpp` `updateThermalFreeze`: donma ve erime eşiği artık parçacığın kendi maddesinden okunur.
  - Eritilebilir madde → `melt_kelvin` (erime bandı aynı: `max(1, 0.1·viscosity_range)`).
  - Eritilemeyen madde → hiç donmaz.
  - Etiketsiz parçacık veya profili olmayan etiket → domain `thermal_freeze_kelvin` (eski davranış).
- Yeni dosya yok, vcxproj değişmedi, IPC/API yüzeyi değişmedi. Script değişmedi.
- Bilinçli olarak yapılmadı: `buildThermalViscosityField` (ν(T) eğrisi) hâlâ domain eşiğini kullanıyor.

## Sıra

1. **Build (kullanıcı):** Release x64 derlemesi. Hata beklenen yer `FluidThermalLiquid.cpp` içindeki `tryFindSubstanceByTag` / `SubstanceProfile` erişimi; include'lar `FluidThermalPhaseExchange.cpp` ile aynı. Derleme hatası çıkarsa bildirin.
2. **Tek madde regresyonu (önce bu):** Yalnız Water ve yalnız Wax domain'leri. Donma/erime zamanı ve yığın şekli eski sürümle aynı olmalı. Domain Default Substance'ın `melt_kelvin` değeri domain eşiğiyle zaten eşit (`FluidDomainSubstance.cpp:67`), o yüzden değişiklik görünmemeli. ★ En sinsi başarısızlık: sonuç makul görünürse ama eşik sessizce farklıysa kimse fark etmez; eşiği `fluid.get`/parçacık sıcaklığıyla doğrulayın.
3. **Karışık domain (asıl kazanım):** Aynı domain'de Water ve Wax parçacıkları, sıcaklık ~300 K. Wax parçacıkları donmalı, Water parçacıkları (273.15 K eşiğinin üstünde) sıvı kalmalı. Eski sürümde ikisi de domain eşiğine göre davranıyordu.
4. **Eritilemeyen madde:** Sand/Gravel gibi bir maddenin etiketli parçacığı liquid domain'de soğutulunca donmamalı.
5. **Dayanıklılık:** Wax donmuş bir katmanın üstüne sıcak Wax dökülünce erimeli (histerezis bandı korunur).

## Ertelenen ölçüm bulguları (ana plan bitince)

- Sweep (voxel 0.035, ppc 8): en kötü kare ~1.8 s. Hipotez: bellek tahsisi ve/veya rebuild
  (buffer büyümesi, tek seferlik yeniden kurulum). Kanıt yok; ana plan bitmeden kovalanmayacak.
- `sim.matter.gpu_partition_upload` saf su sahnesinde hiç çalışmıyor (yalnız mixed/granül).
  Model yükleme darboğazı hipotezi karışık sahnede test edilecek.
- `sim.timeline.capture_frame` parçacık başına 0.73 → 0.96 µs büyüyor: host kopyası şüphesi.

## Açık T2 (sonraki partiler, §6a)

- Ice → Water (263 K'de katı doğma) kuralı henüz yok. Şu an soğuk ama desteksiz parçacık sıvı kalıyor (`cold_unsupported`). "Doğduğu anda katı" bir tasarım kararı gerektiriyor: desteksiz donmuş parçacık pinlenir ve havada asılı kalır. Karar bekleniyor.
- `initial_constitutive_model` ve bağlamadaki Phase/Constitutive sökümü.
- Frozen bayrağı ile solid mask'in tek yola indirilmesi.
- `buildThermalViscosityField` ν(T) eğrisinin parçacık başına eşiğe geçmesi.

## Devralınan T1/T3 açık kabulleri

Devir notunun listesi aynen geçerli: [MADDE_T2_T3_HANDOFF.md](MADDE_T2_T3_HANDOFF.md) "Tam kapanıştan önce kalan kabul" bölümü (save/open, legacy göçü, panel görseli, Wax/MSF görseli, önceki H1 regresyonları).
