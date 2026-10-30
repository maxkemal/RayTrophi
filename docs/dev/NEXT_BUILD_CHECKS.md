# NEXT BUILD CHECKS

## Gaz akışı birleştirme aşama 1 — 2026-10-30 (önce bu)

★ **Canlı sonuç:** 1–3 ve 5 tamam: 262 → ~135 ms. **Madde 4 (görüntü) gözle yapılmadı**; JPEG farkı p99 19.
Açılışta birleşme ilk yeniden yayına kadar gecikiyor.

C++ derlemesi gerekir, shader değişmedi. Not: `RT_GAS_STREAM_MERGE_20261030.md`.

1. **Sahne bozulmamalı.** İki gazlı sahneyi aç: A ve B materyalleri farklıysa `render.volume_slots`
   hâlâ iki gaz göstermeli. Biri kayıpsa birleştirme yanlış tetiklenmiştir.
2. **Birleşme.** B'ye A'nın emisyon ayarlarını ver (`volume.set_material`, `ramp_enabled`,
   `color_ramp`). Slotlarda tek gaz kalmalı, `volume.get_material` iki hedefi de açmalı.
3. **Süre.** Küplü sahnede yaklaşık tek akış bandı (~131 ms) beklenir; iki akış 240–280 ms.
4. **★ Sessiz hata:** görüntü A+B ile aynı kalmalı. Gaz belirgin koyulaşır veya açılırsa eğri iki kez
   işleniyor ya da hiç işlenmiyordur; hata vermez.
5. **Görünürlük.** B'yi gizle: birleşme bozulup tek A kalmalı, gazın B payı kaybolmalı. Geri aç.

## Sekme ışını: radyans önbelleği yürüyüşü — 2026-10-30 (önce bu)

★ **Son ölçüm:** A+B 271 → 239 ms (adım 2), 224 ms (adım 4), notun §10.3'ü. Madde 3 hâlâ yapılmadı.

★ **Bu derlemede yeni:** yürüyüş `secondary_step_scale`'e uyuyor (notun §10.2'si). Aynı süreçte
`secondary_step_scale` 1 ve 2 ile A+B ölç: fark sekme yürüyüşünün payıdır. ★ Görsel: küpteki gaz
yansıması koyulaşırsa veya bantlanırsa kaba adım fazla gelmiştir, kimse bug diye raporlamaz.

★ **Canlı sonuç:** 1–2 tamam: A −%11, B −%27, A+B 252–323 ms; sekme hâlâ ~130 ms (notun §10.1'i).
Madde 3 (görüntü) yapılmadı.

Yalnız shader. `.\RayTrophiStudio\compile_shaders.bat -VolumeOnly -VolumeOptimization Off`,
aynı küplü sahne, frame 99, duraklat. Notun §10'u.

1. **Sayaçlar.** `radiance_cache_marches` ~0.14, kamera ~12.5: değişmemeli. `radiance_cache_fallbacks` 0.
2. **Süre.** A, B, A+B ve `max_bounces=1`. Aynı oturumda oranla oku. A+B 300'ün belirgin
   altına inmeli. İnmediyse kök boş adımlar değil, dolu adımların 32 okumasıdır.
3. **★ Sessiz hata:** küp üstündeki gaz yansıması koyulaşır veya parlarsa `vrcProvenEmpty`
   yayınlanmamış köşeyi boş saymıştır. Görüntü önceki gibi kalmalı.

## İki gaz adım maliyeti — 2026-10-30 (önce bu)

★ **Canlı sonuç:** 1–3 tamam: oran ~1.8× (başlangıç 3.2×), notun §9.1'i. Süreleri bantlar
arasında kıyaslama, oran kullan. **Madde 4 (renk) ve önceki partinin görsel maddeleri yapılmadı.**

Yalnız shader. `.\RayTrophiStudio\compile_shaders.bat -VolumeOnly -VolumeOptimization Off`,
sonra iki gazlı sahneyi aç (legacy, frame 99, duraklat, Rendered). Notun §9'u.

1. **Pipeline hazır, SHA aynı.** Değilse sonraki sayılar eski shader'ındır.
2. **Sayaçlar.** `camera_density_samples` ve `shadow_density_samples` önceki düzeyde kalmalı
   (~10–11 ve ~4.3), `majorant_queries` 3.5–3.9'dan belirgin düşmeli. Düşmediyse yaprak
   önbelleği çalışmıyor; ortak yürüyüş açılmıyor olabilir.
3. **A, B, A+B matrisi, kare dizileriyle.** A ve B değişmemeli, A+B ~80 ms'nin altına inmeli.
   Yalnız medyana bakma, iki bant varsa tur sayısını artır.
4. **★ Sessiz hata:** görüntü aynı kalmalı. Birim terim yanlış alana bağlanırsa iki gazın rengi
   yer değiştirir, hata vermez. A ve B farklı malzemeyle bir kez bak.

## Tek gaz okuma maliyeti — 2026-10-30 (önce bu)

★ **Canlı sonuç:** madde 1 tamam (sayaçlar aynı). Madde 2 hükümsüz: süre örnek sırasına bağlı,
sample 8–23 medyanı bu sahnede sağlam ölçü değil (notun §8.3'ü). Madde 3 ve 4 yapılmadı.

Yalnız shader değişti. `.\RayTrophiStudio\compile_shaders.bat -VolumeOnly -VolumeOptimization Off`
ardından `nuke.rtp` aç, frame 94'te duraklat, Rendered. Hash'ler açılışla değişebilir
ama bu sahne kendini iyi tekrarlıyordu (412–415 ms).

1. **Sayaçlar kıpırdamamalı.** `camera_density_samples` ~36.6, `light_cache_hits` ~10.3,
   `light_cache_misses` ~0.07. Hit düştüyse tutulan okuma kaydı kırılmış; miss arttıysa
   anahtar yanlış. Bu değişiklik iş sayısını değil, iş fiyatını düşürür.
2. **GPU süresi.** Taban 412–415 ms. Düştüyse okuma payı doğrulanır. Düşmediyse okuma
   fiyatı sanıldığı gibi bellek değildir; `shadow_steps=0` ile yeniden ayır.
3. **Görüntü aynı kalmalı.** ★ Sessiz hata: sıcaklık ışını kayarsa ateş topu biraz başka
   renkte ama makul görünür. Eski ekran görüntüsü yok; ramp sınırlarını ve gaz kenarını
   önceki izleniminle karşılaştır, şüphede `sampleTemperature` ile A/B yap.
4. **İki gazlı sahnede tekrar.** A, B, A+B. A+B ~75–80 ms bandında veya altında olmalı.

## nuke.rtp + Play çökmesi — 2026-10-30 (önce bu)

Kök bulundu: `Gas source provenance exceeds 8 emitters / 256 MiB budget`. 166x256x166
= 7.05 M hücre, tek emitter için bile sabit bütçenin üstünde; istisnayı yakalayan kimse
yoktu. Tavan söküldü, reddedilen kaynak artık domain'i hold eder. `Main.cpp` log
koruması da kaldı. C++ derlemesi gerekir (shader'lar daha önce derlendiyse yeterli).

1. `nuke.rtp` aç, Play. Görmen gereken: gaz kaynağı çalışır, uygulama kapanmaz.
   Bozuksa: `StartupCrash.log` son `[FATAL]` satırı artık gerçek hatayı söyler.
2. `render.volume_stats` ile gerçekten gaz yürüdüğünü doğrula: yoğunluk artıyor.
3. ★ Sessiz başarısızlık: uygulama açık kalır ama gaz hiç büyümez. Bu hold'dur;
   domain için `gas_stats.inventory_error` dolu demektir (reset ister).
4. Düşük VRAM'de `GasSourceCurrent` ayırma hatası hâlâ istisna atar ve uygulamayı
   kapatır: yalnız kaynak bırakma noktası korundu, adım içi `require` korunmadı.

## İki gaz çakışma maliyeti — 2026-10-30

★ **Canlı sonuç (tau bütçesi öncesi):** 1–3 tamam. A+B ~77 ms (önce ~112), A ve B
değişmedi. **4–7 görsel maddeleri henüz yapılmadı.**

★ **Tau bütçesi geri alındı:** A+B ~80 ms, gölge örneği −%33, süre kazancı gürültüden
ayrılamadı (notun §7.1'i). Kaynaktan söküldü; kurulu shader hâlâ onu içeriyor.
Aşağıdaki komutla yeniden derle, sonra 1–3'ü koş. Beklenen: gölge örneği yeniden
~4.3/yol, A+B ~75–80 ms bandı. Gölge örneği 2.9 civarında kaldıysa eski `.spv`
ölçülüyordur. ★ Görsel maddeler (4–7) bu derlemeyle yapılmalı.

Ayrıntı: `RT_GAS_OVERLAP_COST_20261030.md`. Yalnız shader değişti, C++ derlemesi
gerekmez. Studio kapalıyken:

```powershell
.\RayTrophiStudio\compile_shaders.bat -VolumeOnly -VolumeOptimization Off
```

Aynı sahneyi legacy modda aç. Gaz hash'leri açılıştan açılışa değişebilir; bu
yüzden ölçümler kendi içinde karşılaştırılır, önceki 112 ms ile değil. Sıra:

1. **Pipeline kurulu mu.** `render.rt_pipeline_status` `ready`, `volume_closesthit`
   mevcut, kaynak ve `x64/Release/shaders/volume_closesthit.spv` SHA'sı aynı.
   Değilse: eski shader ölçülüyor, aşağıdaki hiçbir sayı geçerli değil.
2. **Sayaçlar çalışıyor mu (GPU süresinden bağımsız, hızlı).** Tek geçiş A+B:
   `camera_density_samples` 13.8'den belirgin düşük (beklenti ~8–10/yol),
   `shadow_density_samples` 6.9'dan düşük (~2–3/yol), `step_budget_exhausted`
   0.113'ten düşük, `light_cache_misses` warm 0. Kamera örneği değişmediyse
   `mfRefinedSpan`/`mfCellCurveSlope` etkin değil; gaz karışım yoluna girmiyor olabilir.
3. **A+B/A/B/gizli matrisi.** `rt_probe_volume_split_cost.py` ile A+B alıp onu
   `rt_probe_two_gas_cost_matrix.py --initial-both` ile ver. Görmen gereken: A ve B
   tek başına yaklaşık aynı (shader tek gaz yolunu değiştirmedi; değiştiyse bir
   include sızmıştır), A+B çok daha düşük. Hedef çoğu 45–60 ms, referans 34.5 ms.
   80 ms üstü: izler veya kapılar beklenen yolda değil, 2. adıma dön.
4. **Dumanın ışığı (görsel).** Yarı saydam ince saçak, özellikle yoğun çekirdeğin
   uzak tarafındakiler. Gökyüzü kapısı burada görünürlüğü tutar; saçak parlıyorsa
   kapı fazla erken kapıyor.
5. **★ En sinsi: katı gölgesi kaybolmuş ama gaz makul görünüyor.** Çakışma bölgesinde
   gaz artık katının gölgesini almaz; bu bilinçli (aynı gazın tek başına davranışı),
   ama bug gibi rapor edilmez. Gaz içine/önüne katı koy, A+B ile A tek başına
   karşılaştır: ikisi aynı davranmalı ve çakışma sınırında dikiş olmamalı. Kabul
   edilmezse tek satır: `mfGeometryBlocks` içindeki `!mfWalking` erken dönüşü.
6. **Yoğun gaz arkası.** Opak gaz gövdesinin arkasından parlak emisyon/arka plan
   sızmamalı. 1e-3 eşiği 0.001 geçirgenlik bırakır, tek gazla aynı.
7. **Random walk, üç katkıcı, gazın hizalı olmadığı durum.** Görüntü önceki gibi
   olmalı: iz ve `sampleTemperature` geri düşüşü bu yollarda duruyor. Fark varsa
   `mfWalking` kapısı veya ortak yürüyüş bayrağı yanlış yere bağlanmıştır.

## Hacimsiz pipeline A/B canlı sonucu — 2026-10-30

Kullanıcı build'i ardından iki gerçek stage listesi IPC'den doğrulandı. Aynı
kamera/hedef, frame, render ayarları, 1680x945 ve sample 8–39 ile medyan:
volume dahil 99.145424 ms, volume hit grubu hariç 98.818768 ms (%0.329 fark).
Bu sahnede büyük genel RT maliyetinin salt volume grubunun bağlanmasından
kaynaklandığı hipotezi desteklenmedi. `-O0` karşılaştırması veya iki-gaz sorunu
çözülmüş değildir. Geçici kamera hedefi ve sayaçlar geri yüklendi. Normal
kullanım için test environment değişkeni kaldırılarak uygulama yeniden açılmalı.

## Hacimsiz sahnede volume pipeline A/B — 2026-10-30

C++ kaynakları hazır: `RT_VK_AB_NO_VOLUME=1` process-start test modu volume hit
grubunu pipeline hash'inden önce çıkarır. Diğer shaderlar aynı kalır; published
volume veya TLAS volume slotu varsa RT çizimi reddedilir. Normalde değişken yok/0
olmalı. Shader derlemesi gerekmez, yeni C++ build gerekir. Altı non-build test
geçti; canlı GPU kabulü yapılmadı. Aynı kaydedilmiş hacimsiz sahne ve aynı binary
ile iki ayrı açılışta ölçüm alınmalı; mevcut gaz sahnesinde uygulanmamalı.
Komutlar: [RT_VOLUME_PIPELINE_AB.md](RT_VOLUME_PIPELINE_AB.md).

## Son compiler denemesi ve scalar validation düzeltmesi — 2026-10-29

Kullanıcı `-O` için uzun tek çekirdek çalışması ardından -1073741819 /
0xC0000005 (access violation) bildirdi. `-O0` derlemesi ardından gelen
`VkWorldDataExtended` offset 120 hatası, doğrulayıcıya scalar layout özelliğinin
iletilmemesiydi. Shader GL_EXT_scalar_block_layout gerektiriyor; backend de
scalarBlockLayout özelliğini açıyor. Validator artık `--scalar-block-layout`
kullanır; buffer ABI değiştirilmedi. İncelemede runtime'da eski teşhis anahtarının
scalarBlockLayout özelliğini false tuttuğu da görüldü; artık cihazın desteklediği
özellik isteniyor ve mevcut pNext zincirine ekleniyor. Bu düzeltme C++ derlemesi
gerektirir; pipeline oluşturma süresi ve görüntü ayrıca kontrol edilmeli.
Mevcut SPV scalar seçeneğiyle gerçek spirv-val salt okunur kontrolünden geçti.
Normal varsayılan `Off` / `-O0` oldu.
`-VolumeOnly` artık uzun `-O` denemesine girmez; Auto/Full yalnız açık seçimdir.
Optimizasyon denemesi performans sorununu çözmedi; GPU kabulü hâlâ açık.

## Hacim shader derleme optimizasyonu — 2026-10-29

`compile_shaders.bat -VolumeOnly` artık yalnız `volume_closesthit` için `-O`
dener, SPIR-V'yi `spirv-val` ile doğrular ve başarılıysa yayımlar. `Auto`
modunda compiler/validation hatası bir kez `-O0` fallback yapar; yayınlama
hatası fallback ile gizlenmez. Her iki derleme başarısızsa eski SPV korunur.
`-VolumeOptimization Full` fallback yapmaz; `Off` eski `-O0` referansıdır.
Log'daki son `OK ... [-O ...]` veya `[-O0 ...]` gerçek yayımlanan seçimi gösterir.
VolumeOnly diğer shaderları yeniden derlemez, dağıtmaz veya silmez. Agent gerçek
derleme çalıştırmadı; mock hata yolu ve salt kaynak planı kontrolleri yapıldı.
Yeni shader uygulamada yüklenmeden GPU karşılaştırması geçerli değildir.

## Gas RT ortak hücre yürüyüşü — son kaynak partisi, 2026-10-29

Tam hizalı iki Matter gas alanı için ortak kamera hücre yürüyüşü eklendi.
Kaynakların yoğunluk eğrileri, sıcaklığı ve optik katkıları ayrı korunuyor.
Genel ve hızlı yol aynı optik yanıt modülünü kullanıyor. Üçüncü alan, hizasız
alanlar, grafik/noise, random walk veya adaptif kapalı durumda genel yol sürer.
Bu parti RT shader derlemesi gerektirir; GPU hız/görüntü kabulü henüz yapılmadı.
43 birim testi ve iki ek kaynak denetimi geçti. Ayrıntılar ve kabul listesi:
[GAS_RT_SHARED_CELL_WALK.md](GAS_RT_SHARED_CELL_WALK.md).

## Gas RT maliyeti — 2026-10-29

İki görünür gas alanının adaptif mixture yolunda aynı voxel için tekrarlanan
NanoVDB köşe/sınır hesapları ışın başına önbelleğe alındı. Kalite ayarları
azaltılmadı. Kullanıcı RT shader derlemesinden sonra mevcut frame 89 / Pause
sahnesinde GPU süre ve görüntü eşliği kabulü gerekli; henüz hızlanma ölçülmedi.
Ayrıntı: [GAS_RT_MIXTURE_CELL_CACHE.md](GAS_RT_MIXTURE_CELL_CACHE.md).

## En son ana kaynak değişiklikleri — D2 / T6 E2/E6, 2026-10-29

Son başarılı kullanıcı build'inden sonra RAM/disk cache frame preflight
eklendi. Yeni C++ derlemesi gerekir; shader revision hâlâ 33. Kullanıcının
son yönlendirmesi: ana plan kaynakları önce, hassas fizik ölçüm/tuning en son.
Toplu kabulde iki sistemli cache'de ikinci domain'in lane/count veya frame
dosyası bozuk olduğunda ilk domain/MSF yarım restore edilmemeli, disk cache
valid=false ve rebake log'u görülmeli. Eşleşen RAM/disk frame restore ve
legacy project roundtrip de aynı turda yeniden kontrol edilmeli.
İki loader'ın ortak saved-type parser'ı da bu partide: Gas/Fluid/Matter ve
0/1/2 korunmalı; bilinmeyen tip artık Gas'a sessiz dönüşmemeli. Eksik eski
type alanının Gas varsayılanı değişmedi. Karşılaştırmalı C++ test kaynakları
eklendi; canlı dosya kabulü son toplu tura bırakıldı.
Loader ilk synchronize çağrısı artık bütün flow source kayıtlarından sonra;
aksi halde legacy type erken kayboluyor, kaynak phase göçü atlanıyordu.
Legacy roundtrip'in iki kapalı/yanlış-faz kaynağı yeni build'de doğru faza
dönmeli; ikinci save/open'da da değişmemeli.
D2 kuvvet maliyeti düzeltmesi de C++ partisinde: yüzey Wind/atmosfer rüzgârı
yoksa kolon clear/build ve kolon okuması atlanır. Son kabulde body Gravity/
Vortex alanı, surface-drag Wind ve atmosfer rüzgârı davranışları ayrı kontrol
edilmeli; süre karşılaştırması aynı sahne/sayılarda yapılmalı. Shader ABI ve
revision bu değişiklikte değişmedi. `fluid_force_columns_test.cpp` ilgili
bağımsız C++ kontrat kaynağıdır (ajan derlemedi).
Nested-clock yuvarlama artık authored DEM limit'i aşamaz: örnek 512 istek,
40 continuum, limit512 →64 continuum x8 DEM. Normal sınır içi zamanlama aynı.
Son mixed_clock/4-emitter probunda DEM=continuum*per yanında DEM<=limit de
doğrulanır. `matter_common_clock_test.cpp` bağımsız exhaustive oracle içerir;
Python aritmetik oracle uygulamasız geçti, C++ derlenmedi. Yeni shader yok.

## Yeni kullanıcı derlemesi canlı kabul — 2026-10-29

Kullanıcı "derledim açık" dedi. Başlangıç: boş domain/source, timeline mode,
Pause/frame0. Dokuz seçili smoke kolu sırayla dış IPC ile koşuldu: **7/9 PASS**.

- matter_lanes 7/7, simulation_transport 8/8, substance, P9 melt 4/4,
  P9 floor 2/2, MPM+DEM clock 5/5 ve run_readback geçti.
- Snow-only artık 300 parsel, COM y=0.18331 m; mixed 600 parsel ve inferred
  Snow y~0.185 m. Ölçülen schedule 40 continuum x2 DEM=80, grid=40.
- Floor min-y=0.209966 m, 960 buz yüzü, step_held=false. Status temiz:
  önceki `density position projection failed` artık yok.
- Açık: seed_settle liquid_at_rest (4/5; Water p95=4.311 mm/kare,
  Sand=0.1056 mm/kare). Ritter 0.3 hız=1.1093 m/s, hata=%54.28;
  0.9 hız=1.9494 m/s, hata=%19.65; ideal referans=2.4261 m/s.
  İki doğruluk kapısı ve 0.3 nonregression geçmedi. **0.3'e geçiş reddedilir**;
  0.9 da bu deneyde kabul doğruluğunu sağlamadı. Built-in değiştirilmedi.
- Son durum: Pause/frame0, timeline mode; domain/source/forcefield boş.
  Loglar `C:/Users/maxkemal/AppData/Local/Temp/rt_matter_smoke/`.
  Bu tur full suite, legacy save/open veya 4-emitter performans kabulü değildir.

## Son testten sonraki kaynak partisi — 2026-10-29

**Bu parti C++ yanında simülasyon shader derlemesi de ister (kullanıcıda).**
DEM host/shader revision 33: history_blocks kapasite koruması 8 kelimelik
yerleşime düzeltildi. `sim_matter_grain_list_clear/hash/list_build/step`
çıktıları aynı güncel GLSL'den gelmeli; eski revision 32 SPIR-V kabul edilmez.

Kullanıcı derlemesi sonrası boş sahne/Pause üzerinde dış IPC ile sırayla:

Önce `python scripts/test/rt_test_simulation_transport_ipc.py`: gerçek kare
döngüsünde Play ilerlemeli, Pause kareyi tutmalı, Live'da playhead sabitken
kaynak parçacık üretmeli. Test fluid.step/run kullanmaz; gelen mode'u geri
yükler. Bu test full smoke'da `simulation_transport` adıyla kayıtlıdır.

1. `python scripts/test/rt_test_run_readback_ipc.py` — son sample/dış okuma
   eşitliği, pozitif kütle, manual claim ve ilerleyen timeline frame.
2. `python scripts/test/rt_test_mpm_dem_shared_clock_ipc.py` — Snow/Sand/mixed
   sayılar ve düşüş; run sample ile dış digest uyuşmazsa fizik yorumu yapılmaz.
3. `python scripts/test/rt_test_substance_profiles_ipc.py` — Sand kapasitesi 0.1.
4. `python scripts/test/rt_test_p9_matter_ipc.py --arm floor` — buz teması ve
   status'ta density projection failure olmaması ayrıca kontrol edilmeli.
5. `python scripts/test/rt_test_p9_matter_ipc.py --arm melt` — ilk kütle sıfır
   olmamalı ve final conservation kapısı geçmeli.
6. `python scripts/test/rt_test_ritter_flip_ipc.py` — türetilmiş Water 0.3/0.9
   karşılaştırması, 0.4 s kuru yatak salımı. Her kolun son cephe hızının
   `2*sqrt(9.81*0.15)` referansına bağıl hatası <=%15; 0.3 hata payı 0.9'dan
   en çok 5 yüzde puan kötü olabilir. Çıktı sistem TEMP/rt_ritter_flip.json.
   Bu sonlu derinlik/çözünürlük deneyi tek başına genel solver doğrulaması
   değildir; rest/column kapılarıyla birlikte preset kararı verilir.
7. `python scripts/test/rt_test_matter_legacy_roundtrip_ipc.py` — yalnız boş,
   harcanabilir projede: TEMP dosyalarına save/open yapar, aktif proje yolu
   değişir. Synthetic legacy Gas+Fluid göçü, kapalı kaynak fazı, tek grid, gaz
   shader korunumu, canonical save/open. Gerçek tarihsel dosya kabulü ayrıdır.

Toplu son tur (boş proje, Pause): `python scripts/test/rt_matter_smoke.py`.
Yeni matter_lanes, grain_sleep ve ritter_flip kolları full turda kayıtlıdır;
quick tur kapsamı aynı kalır. Legacy proje turu yalnız açıkça
`--with-legacy-project` eklenirse çalışır; harcanabilir proje gerekir. Tekil
örnek: `python scripts/test/rt_matter_smoke.py --with-legacy-project --only legacy_project`.
Phase-grid read-only probe ise authored domain adı ister, boş-scene otomatik
suite'e eklenmedi. 4-emitter maliyet ölçümü kendi sahnesinde ayrı koşulur:
`python scripts/test/rt_probe_grain_clock_ipc.py --steps 204 --every 12 --require-clock`.
Clock kabulü: DEM=continuum*per, 0<grid_steps<=continuum ve hiçbir held kare
yok. Prob JSON'u tam host_ms/clock/contactless/transfer verisini, ayrı
`TEMP/rt_probe_grain_clock_summary.json` ise örneklenmiş medyan/p95 dökümünü
verir. İç içe süreleri toplama; IPC wall-clock ile solver süresini karşılaştırma.

Bu parti fluid.step/run için physics.step ile ortak manual timeline claim,
sıvısız density no-op GPU restore düzeltmesi ve Sand test beklentisini içerir.
Henüz derlenmedi/canlı kabul edilmedi. Son teşhis pipe error 2: bağlantı yok;
ajan uygulama açmadı. Liquid-at-rest, kalan D2 ve geniş T6 kabulü hâlâ açık.

## Kullanıcı derlemesi hızlı kabul — 2026-10-29

Boş sahne/Pause, dış named-pipe istemcisi. Build ajan tarafından yapılmadı.

- `rt_test_matter_lanes_ipc.py`: **7/7 PASS** (canonical identity, presets,
  absent grid, invalid/no-mutation, source guard, lane switch, auto phase).
- `rt_matter_smoke.py --quick --timeout 360`: **7/11 test PASS**.
  Thermal films 5/5, phase3 freeze, fire/steam 8/8, grain readiness 1/1,
  P9 floor 2/2, run readback 1/1 ve fast fall 2/2 geçti.
- Açık: substance Sand `grain_water_capacity_fraction == 0` beklentisi;
  seed_settle liquid_at_rest (su p95 4.31 mm/kare; tane kapıları geçti);
  p9_melt mass kapısı (`mass_after_pour_kg=0`, son 140.325079 kg; bu okuma
  fiziksel kütle kaybını kanıtlamaz); mpm_dem_clock 1/4 (snow-only digest 0,
  sand 300, mixed 600; boş Snow artık yanlış PASS sayılmıyor).
- Düz buz levhası: 4608 frozen parcel, 960 yüz, kum min-y 0.2099565 m,
  COM y 0.2120414 m; step_held=false. **Ek açık:** status'ta
  `density position projection failed` var. Floor gate geçti diye density
  pass kabulü verilmez; bu hata sonraki kaynak düzeltmesine girer.
- Son salt-okuma: Pause, frame 0, driver none; domain/source/forcefield listeleri
  boş. Loglar `%LOCALAPPDATA%/Temp/rt_matter_smoke/`.

Bu hızlı tur T0, tam legacy save/open ve uzun hacim/Ritter kabulü değildir.
Tur sırasında kaynak düzeltmesi veya uzun ayıklama koşusu yapılmadı.

## Rig: Human Walk v2 (W1–W4) + Rig Studio kalıntı sökümü — 2026-10-28 (yalnız C++, derlenmedi; madde partisiyle birlikte derlenecek)

### R1 otomatik işaret + R3 geodezik ağırlık (2026-10-29, derlenmedi; derleme planın sonunda)

**R1 otomatik işaret (2026-10-29, derlenmedi):** `Animation/RigAutoLandmarks`,
`rig.auto_landmarks` (salt okunur) + Python + Alignment penceresinde **"Auto place"**.
`get_fit_setup` ile aynı uzay ve biçim → doğrudan `preview_fit` / `commit_fit`'e girer.
Şablonun sınır kutusuna ölçeklenmiş halinden başlar. Humanoid: baş tepesi en yüksek
noktalardan, ayak ucu/top/bilek her taraftaki en alçak banttan, el ucu kalçanın
üstündeki en yanal noktadan (T ve A poz), dirsek/bilek omuz→el çizgisinde, diz kalça→
bilek çizgisinde; her eklem kendi kesitinin ortasına çekilir ve kesit yoğunluğu + ne
kadar kaydığıyla puanlanır. Parmaklar tespit edilen kol yönüne ve uzunluğuna taşınır,
bilerek .45 (eşiğin altı) — eller hep bir bakış ister. Diğer aileler: her eklem ebeveyn
kemiğine dik kesitte ortalanır. Mesh −Z'ye bakıyorsa (`forward_check`) tüm güvenler
.3'e kırpılır. Panelde düşük güvenliler **amber**; elle taşınan eklem listeden düşer.

**R3 geodezik voksel ağırlık (2026-10-29, derlenmedi):** `Animation/RigGeodesicWeights`
(Dionne & de Lasa 2013). Bağlı mesh'ler bind pozisyonlarında birlikte vokselleşir
(çözünürlük 32..256, varsayılan 96): üçgenler yüzey, dış kenardan taşma doldurma = dışarı,
kalan = iç. Her deforme eden kemik (envelope ile aynı segmentler, root hariç) yalnız
**katı** vokseller içinde 26 komşulu Dijkstra ile mesafe büyütür (boyun yarısına kadar);
vertex en yakın 4 kemiği tutar, ağırlık = max(d, ½ voksel)^-falloff, sonra ortak en
güçlü-4 sözleşmesi. Açık mesh taşar → yalnız yüzey kabuğu mesafe taşır
(`closed_volume` false), yine kullanılır. `rig.preview_geodesic_weights` (salt okunur) /
`rig.apply_geodesic_weights` (envelope ile **aynı** ağır geri alma komutu; algoritma
`geodesic_voxel_v1` projeye yazılır, yükleyici kabul eder; envelope yarıçapları ve
profiller korunur). Panel: Skin binding → envelope bölümünün altında "Geodesic voxel
weights". Ölçüm aracı: `scripts/test/rt_probe_rig_auto_ipc.py [--apply]`.

Sırayla:
1. `python scripts/test/rt_probe_rig_auto_ipc.py Rig` (sahneyi değiştirmez) →
   auto_landmarks satırı + geodezik önizleme. Bağlı rig'de fit hedefi zaten skinli olduğu
   için auto_landmarks **reddedilir** (tasarım: `target_already_skinned`) — R1'i test
   etmek için yeni bir humanoid mesh içe al, şablon rig oluştur, bağlamadan önce koş.
2. Yeni mesh'te: Alignment → Prepare → **Auto place** → amber eklemler az olmalı (T poz
   insan: çoğunlukla yalnız parmaklar). Önden ve yandan bak: bilek/dirsek/diz kol ve
   bacak içinde. Preview → `outside_bounds` 0. Bozuksa: el ucu kalçaya düşüyor → kol
   gövdeye yapışık (uyarı satırı söyler); her şey .3 → mesh −Z'ye bakıyor.
3. Geodezik: önizlemede `fallback_vertices` 0, `closed_volume` true (kapalı mesh),
   `bones_outside_mesh` küçük (parmak ucu kemikleri olabilir).
4. `--apply` ile koş → ROM önce/sonra satırları; `folded` ve `pinched` azalmalı ya da
   eşit. Uygulamada Ctrl+Z → eski ağırlıklar ve `rig.get_binding` eski algoritma.
   Kaydet → yeni .rtp → aç → `geodesic_voxel_v1` yükleniyor (yükleyici reddetmemeli).
★ Sinsi: çözünürlük düşükse (≤48) parmaklar aynı voksellere düşer ve birbirinin
ağırlığını alır — ROM'da parmak hareketleri "katlanma" gösterir, gövde temiz görünür.
Sayı kötüyse önce çözünürlüğü artır, falloff'a sonra dokun.
★ Süre: 96'da birkaç saniye UI donar (UI iş parçacığında koşar) — hata değil.

### Rig gizmosu + R2 ROM testi (2026-10-29, derlenmedi; derleme planın sonunda)

**R2 ROM testi (2026-10-29, derlenmedi):** yeni çekirdek `Animation/RigRomTest`
(vcxproj'a eklendi). Her uzuv zinciri için kök swing ±60 / spread ±45 / twist ±45, orta
eklem 110° bükülme (bükülme ipucundan **uzağa** katlanır: diz öne gider, baldır geriye),
uç eklem twist ±60 / flex ±45; her spine/neck/head rolü flex ±30, yan ±25, twist ±35.
Eksenler kemikten + bükülme ipucundan + ayaklardan ölçülen karakter çerçevesinden
kurulur, sabit eksen yok; yazılmış eklem limitleri her pozu kırpar (`limited`). Deri
CPU'da `pose · rest⁻¹` ile bind pozisyonları ve kayıtlı flat ağırlık satırlarından
hesaplanır; her geçerli üçgen için alan oranı: < .5 **sıkışma** (çökme, şeker ambalajı),
> 2 **gerilme**; normali baskın kemiğinin taşıyacağı yöne ters dönen üçgen **katlanma**.
`rig.preview_rom_test` (salt okunur, her modda) satır başına hareket, kemik, açı, en
kötü mesh/vertex/kemik ve klipteki tepe karesini verir; `rig.create_rom_clip` aynı
hareketleri dinlenim/tepe/dinlenim anahtarlarıyla klibe yazar (tek geri alma adımı).
Panel: Pose → Quick Motion → "Range of motion test" (kontrol / klip üret / en kötü 10
tablosu). Yürüyüşün klip yayınlama gövdesi ortak `publishAuthoredClip`'e taşındı.
Ölçüm aracı: `scripts/test/rt_probe_rig_rom_ipc.py`.

**Gizmo düzeltmesi (2026-10-29, derlenmedi):** kemik/rig gizmosu fare bir panelin
üstüne geçince **kayboluyordu** — `RigViewportUI` panel hover'ında çizmeden dönüyordu.
Artık IK gizmosu gibi: her zaman çizilir, panel fareyi tuttuğunda yalnız sürükleme
başlatılamaz.

Sırayla (bağımsız ve hızlı önce):
1. **Gizmo (gözle, 10 sn):** Edit veya Pose'da bir kemik seç, fareyi viewport'tan sağ
   dock'a ve geri gezdir → gizmo hiç kaybolmamalı. Panelin üstündeyken gizmonun altına
   denk gelen bir yere tıkla → panel tepki verir, kemik **hareket etmez**. Bozuksa:
   kaybolma sürüyor → derleme eski; panel tıklaması kemiği oynatıyor → sürükleme
   engeli çalışmıyor.
2. `python scripts/test/rt_probe_rig_rom_ipc.py Rig` → 4/4 PASS; tablo en kötü 10
   hareketi listeler. Bozuksa: `unknown_influences` > 0 → deri satırları bu rig'in
   indekslerini göstermiyor (binding/indeks remap sorunu, ROM değil); motions 0 →
   anatomide zincir/rol yok.
3. Panelde "Check deformation" → aynı sayılar; "Generate ROM clip" → `ROM_Test` seçili
   klip, oynatınca her eklem sırayla bükülüp dinlenime döner. En kötü satırın
   `frame_peak` karesine git → sıkışma/katlanma tabloda adı geçen kemikte görünmeli.
4. Kendi kontrolü: Envelope Editor'da kol kapsülünü daralt/genişlet → `min_area_ratio`
   değişmeli (ROM ağırlığa duyarlı).
★ Sinsi: diz `bend` hareketi ipucu olmayan eski rig'de karakter ileri yönünden
türetilir; dirsek de öne katlanır ve "kol yanlış bükülüyor" görünür. Önce
`rig.derive_limb_hints` (yukarıdaki v8 bölümü) — ROM raporu bunu kendisi söylemez.


**Şablon v4 / anatomi v8 (2026-10-29, derlenmedi):** iki kalıcı anatomi alanı —
`bend_hints` (zincir başına rig uzayında bükülme yönü) ve `foot_pivots` (bacak
zincirinin üçüncü kemiğine, bileğe göre topuk/top/uç, bileğin dinlenim çerçevesinde).
`IKControl.bendHint` → `matchIKPose` polü dinlenim bükülmesi yerine ipucundan kurar
(K4 artık **Pose IK'da da** kapalı, kontroller yeniden oluşturulunca). Yürüyüş kayıtlı
pivotları okur; kayıt yoksa aynı ortak `deriveFootPivots` ile türetir; kayıt parmaklarla
.02h'den fazla ayrışırsa `foot_pivots_differ_from_toes` raporlanır (refit sonrası bayat
kayıt sessiz kalmasın). Şablonlar: humanoid/detaylı → diz +Z, dirsek −Z, ayak pivotları
(boyla ölçekli); kuş → bacak +Z. Eski rig'ler için `rig.derive_limb_hints`
(UI düğmesi + Python + IPC): ileri yönü **ayaklardan** ölçer (bilek→parmak), sabit
eksen yok. Panel: Rig anatomy → "Bend hints and foot pivots" (doldur / yeniden türet /
Enter ile düzenle / sil); zincir silinince ona bağlı ipucu ve pivotlar da silinir.
Eklem yönelim çerçeveleri bu partide **yok** — parmak curl eksenleri, limit eksenleri
ve bind matrisleri yerel eksene bağlı; ayrı parti.

Prob artık önce `rig.derive_limb_hints` çağırır ve 9 eşik koşar (+ pivotlar anatomiden
ve parmaklarla uyumlu, + diz polü anatomi ipucundan). Ek elle kontroller, sırayla:
1. Rapor: `forward` ≈ (0,0,1), `forward_assumed` false, 4 ipucu, 2 pivot seti,
   `pivots_estimated` boş. Bozuksa: forward ters → ayak bilek/parmak sırası ters
   okunuyor; ipucu 0 → aile `humanoid` değil.
2. `rig.get_anatomy` → `version` 8, `bend_hints` + `foot_pivots` var. Kaydet → yeni
   .rtp → aç → aynı.
3. Pose'da IK kontrollerini yeniden oluştur → `rig.get_controls` satırlarında
   `bend_hint`; bacak hedefini yukarı çek → diz **öne** bükülür, dirsek geriye.
4. Yeni humanoid şablon rig (`rig.create`) → anatomi doğrudan v8, pivotlar boyla ölçekli
   (1,8 m'de topuk z ≈ −0,051, top z ≈ 0,171).
5. Panelde bir pivotu değiştir (Enter) → `foot_pivots_differ_from_toes` true olur;
   "Re-derive all" → false. Zincir sil → kayıt reddedilmeden ipucu/pivot da gider.
★ Sinsi: IK kontrolleri v8'den ÖNCE oluşturulduysa ipucu taşımaz ve diz eski gibi
dinlenim bükülmesinden davranır — hata yok, sadece eski davranış. 3. madde bu yüzden
"yeniden oluştur" diyor.


**W1b canlı (2026-10-29, ikinci derleme):** prob 6/6 PASS verdi ama **yanlış**: diz
f0–f13 boyunca 11,5°'de sabitti (= `leg_reach` .995 tavanı). Testin pencere hatası
itiş öncesi kareleri de sayıyordu. Kök: diz profili bacak boyunu **milimetre**
değiştiriyor (18° ile 11,5° arası ~6 mm), `body_bounce` .01h ise **1,8 cm** ekliyordu —
zıplama pelvisi tavana itip profili eziyordu. Derlemesiz doğrulandı (IPC'den
`body_bounce=0`, `leg_reach=.998`): diz temasta 7,3° → yükte 17,8° → itiş öncesi 7,3° →
salınım tepesi 70°, penetration 1e-7, hata 1e-7. Aynı tabloda yeni bulgu: pelvis
temas anında **tek karede 3–3,4 cm düşüyor** (her adımda "küt" oturma) — öndeki
bacak tavana son anda yetişip pelvisi çekiyor.

**W1c (2026-10-29, derlenmedi):** pelvis düşüşü iki geçişte: önce kare başına ham
düşüş, sonra döngüsel **genişlet (max, r) + kutu bulanıklığı (r)**, r = döngünün
%8'i. Sonuç hiçbir karede ham değerin altında değil → her erişim tavanı hâlâ tutar.
Varsayılanlar: `body_bounce` **0**, `leg_reach` **.998**. Prob düzeltildi: diz değişimi
yalnız temas→topuk kalkışı penceresinde ölçülür, yeni eşik **pelvis adımı ≤ .006h/kare**.

Beklenen (W1c sonrası): 7/7 PASS; tabloda pelvisY kareden kareye ≤ ~1 cm değişmeli,
temas karesinde (f0/f29) sıçrama yok. Diz eğrisi yukarıdaki derlemesiz koşuyla aynı
biçimde (temas ~7°, yük ~18°, salınım ~70°), yalnızca birkaç derece daha bükük olabilir
(yumuşatma düşüşü yalnız artırır). Bozuksa: pelvis adımı hâlâ büyük → yumuşatma
uygulanmıyor; diz yine sabit → `body_bounce` 0 değil ya da tavan her karede bağlıyor.


`python scripts/test/rt_probe_human_walk_ipc.py Rig` — karakter Pose'da. `Walk_Loop`'u
varsayılanlarla yeniden üretir, sol bacağın bir döngüsünü kare kare basar, 6 eşik:
ayak hedef hatası ≤ .005h, zemine girme ≤ .002h, erişilemeyen kare 0, **basma
fazında diz en az 8° değişmeli**, temasta diz ≤ 12°, salınım dizi tepesi 50–75°.
Bozuksa: diz yine sabit → pelvis profili uygulanmıyor (ya da `leg_reach` tavanı
her karede bağlıyor, `max_reach_ratio` .995'e yapışık); penetration sürüyor → parmak
bükülmesi yanlış işaretli (parmak aşağı kıvrılıyor) ya da rigde toe rolü yok
(`toe_bend` false). Sonra gözle: yandan bak, basan dizin temasta neredeyse düz,
yüklenirken hafif bükük, itişte yine düz olması gerekir.
★ Sinsi: dikey zıplama fazla gelirse `body_bounce`'u 0 yap — artık asıl dikey hareketi
bacak geometrisi üretiyor, parametre yalnızca üstüne ekler.

**İlk canlı ölçüm (2026-10-29, W1–W4 derlemesi, `humanoid_detailed`, 1,8 m, varsayılanlar):**
`max_reach_ratio` .970, `foot_target_error` 1,9e-7 m, `unreachable_frames` 0,
pelvis düşüşü 2,8–7,2 cm → W1 sayısal olarak PASS. **İki FAIL, ikisi de gözle görülen
"robotik" his:** (1) basan bacakta diz bütün basma fazı boyunca **28,1°'de sabit** —
pelvis çözümü bacağı her karede tam `leg_reach`=.97 sınırına çiviliyordu (.97 = 28°
diz, kosinüs kuralı), bacak bükük bir değnek gibi pergel yürüyüşü yapıyordu; kalça hiç
geriye açılmıyordu (min +5°). (2) `ground_penetration_max` 3,99 cm: topuk top
etrafında kalkarken parmaklar bükülmediği için uç zemine giriyordu. Salınım dizi tepesi
74°, dirsek 25–57°, ayak yuvarlanması (topuk 15°, kalkış 30°) beklendiği gibi.

**W1b düzeltmesi (2026-10-29, derlenmedi):** pelvis yüksekliğini artık basan bacağın
**diz bükülme profili** belirler: temasta ~5°, `stance_knee_deg` (18°) yükleme
bükülmesi, topuk kalkmadan önce ~6°; bacaklar destek ağırlığıyla harmanlanır.
`leg_reach` yalnız tavan (varsayılan .97 → **.995**), `body_bounce` bunun üstüne eklenir
(varsayılan .02 → **.01**, çünkü dikey hareketi artık geometri üretiyor). Parmak kemiği
topuk kalkışında ayak pitch'ine karşı bükülür, salınımın ilk üçte birinde düzelir
(`toe_bend`); uç ölçümü parmak kemiğinden yapılır. Göğüs ve kollar pelvisi döngünün
%5'i kadar geriden izler (`upper_body_lag`). Yeni alan `stance_knee_deg` 0..35 her
katmanda. Ölçüm aracı: `scripts/test/rt_probe_human_walk_ipc.py` (6 eşik).

Kaynak: `RigWalkRecipe.cpp` yeniden yazıldı (W1 pelvis yüksekliği, W2 ayak
yuvarlanması, W3 gövde dönüşleri karakter çerçevesinde, W4 diz polü karakterin
baktığı yönden); yeni tarif alanları `leg_reach`,
`heel_strike_deg`, `toe_off_deg` UI/Python/IPC'de. Plan ve teşhis:
`RIG_AUTHORING_ROADMAP.md` §11. 20 ölü Rig Studio dosyası silindi (vcxproj'da
zaten yoktu; `RigFingerFK*` canlı, dokunulmadı). Önden arkaya, bağımsız ve hızlı
olan önce:

1. **Derleme.** Hatasız bitmeli. Bozuksa: silinen 20 dosyadan birini include eden
   bir yer kalmış demektir (hata satırı dosyayı söyler) — ya da yeni `RigWalkRecipe.cpp`
   tip hatası.
2. **Check recipe, varsayılanlarla** (Pose → Quick Motion; ya da IPC
   `rig.preview_human_walk`). Paneldeki yeni iki satır ve `metrics`:
   - `max_reach_ratio` ≈ .97 (≤ .98), `unreachable_frames` = 0
   - `foot_target_error_fraction` < .005 (panelde "%0.5" altı)
   - `pelvis_drop_min` ≈ 0, `pelvis_drop_max` > 0 (beklenen ~.02–.05 × boy)
   - `ground_penetration_max` ≈ 0 (≤ .002 × boy)
   Bozuksa: reach > .98 → pelvis çözümü uygulanmıyor; reach doğru ama foot error
   büyük → ayağı hedeften koparan şey IK sonrası **eklem limitleri** ya da diz polü
   (teşhis K4/K5); penetration büyük → pivot türetimi yanlış (`foot_pivots`
   alanındaki heel/ball/tip noktalarına bak, `foot_pivots_estimated`).
3. **Karşılaştırma:** aynı rig'de `leg_reach=1` ile tekrar → `max_reach_ratio` ≈ 1 ve
   `pelvis_drop_max` 2'dekinden küçük olmalı. Bu, sayının pelvis çözümünden
   geldiğinin kanıtı.
4. **Generate `Walk_Loop`, YANDAN izle.** Ayak artık yerden havada kalmamalı, diz hiç
   kilitlenmemeli; topuk vuruşunda parmak yukarıda, kalkıştan önce topuk ball
   etrafında yükselmeli; duruş sonu → salınım başı geçişinde ve döngü sınırında
   ayak sıçramamalı. Bozuksa: salınım başında sıçrama → `moveFoot` uç noktaları
   duruş eğrisiyle eşleşmiyor.
5. **`heel_strike_deg=0`, `toe_off_deg=0`** → yuvarlanma kaybolur, düz taban; pelvis
   çözümü sürer (metrikler 2'deki gibi kalmalı).
6. **Python paritesi:** `rt.rig.preview_human_walk(char, leg_reach=.95, toe_off_deg=40)`
   ile aynı değerleri IPC'ye ver → `metrics` birebir aynı. Geçersiz değer
   (`leg_reach=.5`) iki yoldan da `rig_walk_invalid_recipe`.
7. **W3/W4 — önizleme alanları:** `body_frame` = `character`, `knee_pole` =
   `character_forward`. Şablon rig'de yürüyüş öncekiyle **gözle aynı** görünmeli
   (şablonun eksenleri zaten dünyaya hizalı). Fark görünürse W3'ün çevrimi yanlış.
8. **W3/W4 — asıl test:** içe alınmış, yönlendirilmiş kemikli bir humanoid
   (ör. Mixamo FBX, humanoid anatomisi eşlenmiş) üzerinde Generate. Gövde öne
   eğilmeli, kalça yürüyüş yönünde dönmeli, dizler HER karede öne bükülmeli.
   Bozuksa: gövde yana yatıyor/ters dönüyor → `context.forward/left` yanlış
   türetilmiş (ayak→parmak yönü ve omuz farkı, önizlemede görünmüyor); diz yana
   bükülüyor → pol hesabı.

★ **Sinsi, makul görünen sonuç:** büyük stride'da karakter belirgin şekilde
**çömelerek** yürür. Bu hata değil, bacak boyunun dürüst sonucu (eski hali aynı
stride'da ayağı 5 cm havada bırakıyordu). Çömelme fazlaysa stride'ı düşür.
★ İkinci sinsi durum: `ground_penetration_max` ≈ 0 ama `foot_target_error`
büyük → ayak hedefin **üstünde** kalıyor; önden bakınca görünmez.

Yapılmadı: pivotlar ve bükülme yönü kalıcı/düzenlenebilir değil (şablon v4);
Pose modundaki IK kontrolleri hâlâ dinlenim bükülmesinden pol alıyor (W4 yalnız
yürüyüş tarifinde). Önizleme artık kare başına ~9 poz örneklemesi yapar; çok
kemikli rig'de "Check recipe" bir an takılabilir — hata değil.

## Birleşik madde canlı kabul — 2026-10-28 (kullanıcı derlemesi)

**4-emitter sahnesi devam kabulü (aynı gün):** ayarlar değiştirilmeden
`rt_probe_emission_ceiling_ipc.py` ile 204 adım, dt 1/24 (8.5 s simülasyon,
199 s duvar süresi). Domain tavanı 100000; son okuma 99146 parsel,
`step_held=false`. Tavan kısılması kaynaklarda ve domain'de birebir aynı:
89.571863067 kg (Water 51.040814146, Snow 2.053671730, Sand 36.477377191).
Son adımda da kısılan kütle pozitif. Son parçacık sayısının tavanın altında
olması, o adımda emisyon kapısının bağlanmadığı anlamına gelmez.
Durum çubuğundaki sarı **EMITTERS THROTTLED: Physics Domain 1** uyarısı
gözle doğrulandı. Tek seferlik HUD mesajı gözlenmedi; pill tıklama testi
yapılmadı. Sahne resetlenmeden ölçülen durumda bırakıldı; ayarlar kaydedilmedi.
Rapor: `%LOCALAPPDATA%/Temp/rt_probe_emission_ceiling.json`.

Boş sahne, Pause; dış `scripts/test/rt_ipc.py` istemcisi kullanıldı.
Sandbox named-pipe hata 5 sonrası testler gerekli escalation ile koşuldu.

- `rt_test_emission_budget_ipc.py`: **6/6 PASS**. 2 kg sınırında
  2.009578 kg / 129 parsel (parsel 0.015578 kg; bir parsel toleransında).
  400 parçacık tavanında kaynak/domain kısılan kütlesi aynı: 87.237503 kg;
  `emission_withheld` defter kaydı mevcut, eski sayı anahtarı reddediliyor.
- `rt_test_matter_exchange_ledger_ipc.py`: **PASS**, kütle hatası 0 kg,
  enerji hatası 0 J.
- `rt_matter_smoke.py --quick`: **7/11 test PASS**. Geçenler: thermal_films,
  phase3_freeze, fire_steam (8/8), grain_readiness, p9_melt (4/4),
  run_readback, fast_fall.
- Açık: substance testi Snow `granular_compaction_hardening > 0` assertion;
  seed_settle 4/5 (`liquid_at_rest`, su p95 4.53 mm/kare);
  p9_floor 1/2 (`sand_rests_on_ice`, 864 buz yüzüne rağmen kum min y 0.00999 m);
  mpm_dem_clock 3/4 (`equal_counts`: snow-only okuması 0, sand-only 300,
  mixed 600; kök bu turda ayıklanmadı).

Loglar: `%LOCALAPPDATA%/Temp/rt_matter_smoke/`; emisyon raporu:
`%LOCALAPPDATA%/Temp/rt_test_emission_budget.json`. Son salt-okuma kontrolü:
Pause, frame 0; domain, kaynak ve forcefield listeleri boş. UI pill/HUD ve
4-emitter sahnesinde uzun koşu kabulü henüz yapılmadı. Bu kayıt tüm D2/D3
işlerini veya önceki tarihli build bekleyen maddeleri kapatmaz.

## Gaz §24: doğal GPU ölçümü (build gerekmiyor)

İki ana prob düzeltildi: viewport.render_frames artık kullanılmaz. GPU süreleri
sayaçlar kapalıyken doğal display akışından, counters ayrı pass'ten alınır.
Aynı voxel/kamera kimliğiyle son256/16/256=611/572/606ms; örnekler32.88/10.31/32.88.
MaxSteps çalışıyor ama kalan per-entry maliyet yüksek. Yeni shader/C++ değişmedi.

- `python scripts/test/rt_probe_gas_max_steps_ipc.py` yeni sampling_method etiketini
  ve aynı scene_signature'ı göstermeli; orijinal ayarlar restore edilmeli.
- `python scripts/test/rt_probe_gas_overlap_cost_ipc.py --frames 4` aynı doğal
  helper'ı kullanır. Eski forced-pass ms sonuçlarını yeni sayılarla karıştırma.
- Sabit/per-entry maliyet için GPU profiling/ayrıntılı maliyet ayrımı gerekiyor;
  yeni bir spekülatif march algoritması eklenmedi.

Canlı ölçüm ve Python sampling/snapshot testleri geçti; yeni build gerekmiyor.


## Gaz §23: snapshot kimliği / tekrar yükleme tutarlılığı (shader değişmedi)

Yeni GPU optimizasyonu yok. Yeni Python probu şimdi kullanılabilir; eski
build'de summary_only kimlik verir. Tüm voxel hash'leri için C++ build gerekir
(FogFieldDigest.h + RtApiMatterFogInspection), shader build gerekmez.

1. Yeni C++ build sonrası `fluid.inspect_fog_fields` her field'da density_hash /
   temperature_hash döndürmeli. UI/IPC/Python aynı read-only API'yi kullanır.
2. Aynı oturumda `python scripts/test/rt_probe_gas_max_steps_ipc.py`:
   başta scene_signature/fingerprint_strength, sonra her koşunun kimliği ve
   çözünürlüğü yazılır. Kimlik değişirse sonuç INVALID; eski raporla oran kurma.
   Orijinal cap ve instrumentation finally ile geri alınmalı.
3. Reset/step yapmadan aynı karede iki read-only snapshot aynı olmalı. Kaydet/
   yeniden yükle veya timeline cache restore sonrası aynı alan bekleniyorsa
   field hash ve optical_bounds karşılaştır. Float hash değişirse sim alanı
   aynı değildir; yalnız AABB farkı üzerinden renderer regresyonu ilan etme.
4. Son kullanıcı256/16/256-repeat:625.6/488.7/603.8 ms; iki256'nın sayaçları
   birebir aynı. Son reload buhar kutusu%6.06, önceki yavaş snapshot%45.08 ve
   kamera hedefi farklı. Girdi farkının kaynağı hâlâ açık.

Statik snapshot kimliği testleri geçti. Ajan uygulamayı açmadı/derlemedi.


## Gaz §22: genel karışım Max Steps cap'i (yalnız shader, DERLENMEDİ)

Kaynakta bug doğrulandı: genel karışım max_steps okumuyordu. Yeni cap,
segmentteki ilgili üyelerin en düşük değeridir; kaba adım kalan yolu kaplar,
gaz kuyruğu kesilmez. Boş atlama örnek bütçesi yemez. Random-walk/cache değişmez.

1. Kullanıcı `compile_shaders.bat` çalıştırır. Yeni C++ build GEREKMEZ.
2. Aynı kare/kamera/çözünürlükte dış terminal:
   `python scripts/test/rt_probe_gas_max_steps_ipc.py`.
   256/16/256-repeat ölçülür, orijinal ayarlar sonunda geri alınır. 16'da
   uzun kamera segmentleri daha az örneklenmeli; step_budget_exhausted yerel
   adaptif hedefin work cap yüzünden kabalaştığı ışınlarda artabilir.
3. Çok düşük bütçede görüntü kabalaşabilir; gaz sonu kaybolmamalı. İnce buhar,
   emissif pikler ve SDF/solid önündeki sınır görsel kontrol edilsin.
4. UI, Python ve IPC max_steps aynı sonucu vermeli. İki çakışan üyeden biri
   16 diğeri256 iken ortak cap16; off-ray bir16-lı slot etkilememeli.
5. Pipeline süre/boyut kaydedilsin. §21 son GPU tabanı185/84/280 ms,
   kamera density9.482/yol; residency dört NanoVDB grid için1. Sekme density
   ve radiance cache fallback0 korunmalı. Sonuç GPU'da henüz ölçülmedi.

Statik/nümerik örnek tavanı ve Beer–Lambert tam segment coverage kontrolleri geçti.


## Gaz §21: NanoVDB yoğunluk/sıcaklık device-local (yalnız C++, DERLENMEDİ)

Somut açık: CPU_TO_GPU buffer isteği host-visible/coherent ister, VRAM istemez.
RTX 3060 memory type sıralamasında ilk eşleşen type 3 sistem heap'indedir;
host-visible VRAM type 5 sonra gelir. Yeni allocation GPU_ONLY ile
DEVICE_LOCAL zorunlu. Shader ve örnek politikası değişmez; kazanç ölçülmedi.

1. C++ build al, uygulamayı yeni executable ile yeniden aç. Shader build GEREKMEZ.
   Yeni module VulkanVolumeResidency.cpp vcxproj/filter listelerine eklendi.
2. Aynı scene/kare 50/kamera/1680×945, Vulkan Rendered:
   `python scripts/test/rt_probe_fog_bounds_ipc.py`.
   Her iki Matter NanoVDB slotunda density_device_local=1 ve
   temperature_device_local=1 olmalı. 0=host heap (FAIL), -1=bilinmiyor;
   -1'i VRAM'de sanma. Dar kutular korunmalı.
3. `python scripts/test/rt_probe_gas_overlap_cost_ipc.py --frames 4`.
   Son aynı oturum authored birlikte GPU 537 ms; önceki 50. kare koşusu 573 ms.
   GPU süreleri, kamera density, majorant/probe sayıları ve görüntüyü karşılaştır.
   Örnek sayısı değişmeden süre azalması bu partinin hedefidir. İkincil density
   ve radiance_cache_fallbacks sıfır korunmalı. Süre düşmediyse RAM hipotezini
   tek neden sayma; yeni residency kanıtını ve GPU profiling'i kullan.
4. Kamera sabitken görüntü aynı olmalı. Sim playback'te first upload/host
   beklemeleri ayrıca ölçülsün: GPU_ONLY staging/copy ekleyebilir. Content
   update, timeline seek, backend switch ve kaynak görünürlüğü çalışmalı.
5. OptiX ~110 ms kullanıcı referansı kaydedildi ama optik formüller farklı;
   eşit kaliteli/eşit GPU-dispatch kapsamlı benchmark varsayılmamalı.

Statik residency/Fog bounds/material/march kontrolleri geçti. Gerçek GPU
bellek yerleşimi ve performans kullanıcı build'i sonrası doğrulanacak.


## Gaz §20.1: blok adımı geri alındı (CANLI §20.2, yeni build gerekmiyor)

Geri alma sonrası güncel canlı ölçüm: kare 50, buhar kutusu %51.5, GPU
446/78/573 ms. Önceki 329 ms snapshot kutusu %11.27 idi; sabit kare/kamera
olmadan doğrudan geri dönüş oranı çıkarılamaz. Mevcut shader yeniden
derlenmek zorunda değil. Aşağıdaki liste tarihsel geri alma kontrolüdür.

§20 kullanıcı ölçümünde FAIL: birlikte 766 ms (önce 329), kamera örneği
24.794/yol (önce 10.745), majorant sorgusu 11.542/yol (önce 4.463).
İkincil density/fallback hâlâ sıfır; sekme cache'i sorun değil.

1. Kullanıcı `compile_shaders.bat` çalıştırır. Bu geri almanın performans
   düzeltmesi için C++ gerekmez; eski tooltip'e dönmek için C++ build isteğe bağlı.
   Yeni optical-block include silindi; tek hacim yönlendirmesi de kaldırıldı.
2. Aynı kare/kamera/çözünürlükte dış terminal:
   `python scripts/test/rt_probe_gas_overlap_cost_ipc.py --frames 4`.
   §19 tabanı buhar/duman/birlikte 261/84/329 ms. Geri dönüş henüz ölçülmedi;
   kamera density ve majorant sorguları da tabana yaklaşmalı.
3. `python scripts/test/rt_probe_fog_bounds_ipc.py`: dar kutu hâlâ korunmalı
   (son kullanıcı ölçümü buhar %11.27, duman %2.66). Görünür buhar değişmemeli.
4. İkincil density/cache fallback sıfır kalmalı. Pipeline süresini kaydet.

Statik geri alma, adaptive, optical, cache ve Fog bounds kontrolleri geçti.
Yeni deney eklenmedi. Sonraki değişiklik öncesi kalan GPU işinin profillenmesi
veya daha ayrıntılı sayaçlarla ayrılması gerekiyor.

## Gaz §19: materyale göre daralan Fog kutusu (CANLI hız/kutu geçti)

Canlı sonuç: buhar kutusu %11.27, duman %2.66; GPU buhar/duman/birlikte
261/84/329 ms. İkincil density ve cache fallback sıfır. Yeni build gerekmiyor;
aşağıdaki listede görsel ve materyal değişimi kontrolleri açık. Ayrıntı §19.1.

Bu parti shader değiştirmez. Önceki §18.1 kullanıcı tarafından derlendi,
anlamlı iyileşme gözlenmedi. Yeni kutu kodu görünür katkısı olmayan ham
yoğunluk kalıntılarını güvenli biçimde dışarıda bırakır. Kazanç ölçülmedi.

1. C++ build al. Header proje listelerine eklendi; yeni .cpp modülü yok.
   Gerçek çekirdek regresyonu istenirse VS Developer terminalinde:
   `cl /nologo /std:c++17 /EHsc /I RayTrophiStudio/source/include scripts/test/fog_optical_bounds_test.cpp RayTrophiStudio/source/src/Math/Vec3.cpp "/Fe$env:TEMP\fog_optical_bounds_test.exe" "/Fo$env:TEMP\"`
   ardından `& "$env:TEMP\fog_optical_bounds_test.exe"` çalıştır. Ajan derlemedi.
2. Aynı sahne/kare/kamera (matter4emiter, kare 38–39). Dış terminal:
   `python scripts/test/rt_probe_fog_bounds_ipc.py`.
   Buharın eski AABB oranı %93.02 idi. `render_box_fraction` ve gerçek
   `render.volume_slots` kutusunu kontrol et. fallback=false olmalı;
   kutu gerçekten daralmadıysa performans kazancı varsayma.
3. `python scripts/test/rt_probe_gas_overlap_cost_ipc.py --frames 4`.
   Önceki aynı koşu 771 / 185 / 1052 ms (buhar/duman/birlikte), sonraki
   baseline 1153 ms; çözünürlük 1680×945. İkincil density/fallback=0 korunmalı.
4. Cutoff'u azalt, eğriyi değiştir, yoğunluk multiplier'ını 0 yapıp geri al:
   aynı timeline karesinde kutu büyümeli/daralmalı, gizlenen akış geri gelmeli.
   UI, `volume.set_material` IPC ve Python `rt.volume.set_material` aynı
   davranmalı. Değerleri geri yükle. Graph/noise'da fallback tam grid olmalı.
5. İnce buhar kenarları, zemine düşen gölge ve sıcak emisyon önceki görüntüyle
   karşılaştırılsın; görünür veri kesilmemeli. Hareketli plume kutusu büyürken
   yeni yoğunluk görünür olmalı. Fog → SDF/Splat kaydı temizliği korunmalı.

Statik kontroller geçti; C++ build, C++ nümerik test ve canlı hız/görüntü
kabulü kullanıcı build'inden sonra yapılacak.


## Gaz §18.1: boş örnek erken dönüşü / kamera dönüşümü reuse (yalnız shader)

§17 CANLI: ikincil density=0 ve fallback=0, fakat birlikte 1052 ms,
üstüne yalnız buhar 771 ms. Cache bypass artık kök değil. Sonraki küçük
parti `volume_mixture.glsl` ve `volume_mixture_adaptive.glsl` içindeki
tekrarlı işleri azaltır; örnek çözünürlüğünü değiştirmez. Kazanç bilinmiyor.

1. Kullanıcı `compile_shaders.bat` çalıştırır; bu yeni parti için C++ gerekmez.
2. Aynı kamera, 1680×945, kare 38–39. Pipeline compile_seconds taban 219.734 sn;
   %10 artış sınırını ve SPV boyutunu kaydet.
3. Dış terminal: `python scripts/test/rt_probe_gas_overlap_cost_ipc.py --frames 4`.
   Kamera örnekleri, sekme örnekleri ve görüntü aynı kalmalı; GPU sürelerini
   771 / 185 / 1052 ms ile karşılaştır. Cache fallback=0 korunmalı.
4. Graph'ın boş kaynakta yoğunluk üretmesi ve random-walk regresyonu.
   Değişim hızlandırmazsa yoğunluk sayısını tekrar azaltmaya çalışmak yerine
   GPU profiler ile NanoVDB erişimi / register spill / ray tracing işini ayır.

Canlı A/B ayarları geri yüklendi. Başka build veya uygulama başlatma yapılmadı.


## Genel gaz karışımı cache + Fog kaydı temizliği — 2026-10-28 (C++ ve shader)

Bu parti §16b'deki genel karışım bypass'ını düzeltir. Aşağıdaki eski
"yalnız shader" partisi önceki denemedir; bu parti onun devamıdır.

1. C++ build ve `compile_shaders.bat` çalıştır. Yeni shader include:
   `volume_cached_segment.glsl`; değişenler: `volume_radiance_cache.glsl`,
   `volume_mixture.glsl`, closest-hit bağlantısı. C++: MatterFogPublisher ve VDB UI.
2. `E:\rt_k\matter4emiter.rtp`, kare 38–39, kamera sabit, Vulkan Rendered.
   `render.rt_pipeline_status` derleme süresini ve SPV boyutunu kaydet.
   Taban 206 sn / 982,912 B; %10'dan fazla artış kabul dışı.
3. Dış terminalden `python scripts/test/rt_probe_gas_overlap_cost_ipc.py`.
   Isınma sonrası both: `secondary_density_samples` yaklaşık 0,
   `radiance_cache_marches` artmalı; GPU ms ayrı hacimlerin toplamına yaklaşmalı.
   Son ölçüm 598 + 112 → 1157 ms idi. Yeni kazanç henüz ölçülmedi.
4. Cache kapalı/açık karşılaştır: zemin/sıvı dolaylı ışığı ve sıcak emissif
   çakışmalar; kamera görüntüsü değişmemeli. Tek hacim, programlı hacim,
   random-walk ve SDF önündeki fog sınırı regresyonlarını kontrol et.
5. Bir emitter'in katı/sıvı çıktısını Fog yap, VDB materyalini düzenle;
   SDF ve Splat'a geçir. Eski carrier Fog kaydı listeden ve seçili özelliklerden
   kalkmalı; gerçek gaz/vapor akışları kalmalı. Fog'a dönünce materyal geri gelmeli.
   Aynı değişimi Python/IPC çıktılarıyla ve undo/redo ile tekrarla.

Statik kontroller geçti; ajan build çalıştırmadı ve uygulamayı açmadı.


## Gaz RT: çakışan iki gaz hacminde radyans önbelleği — 2026-10-28 (yalnız shader)

Emitter bütçesi partisi CANLI geçti (kullanıcı: hepsi PASS). Bu parti yalnız
`volume_radiance_cache.glsl` ve `volume_closesthit.rchit`'e dokunuyor:
`compile_shaders.bat` yeter, C++ derlemesi gerekmez. Sahne:
`E:
t_k\matter4emiter.rtp`, kare 39, kamera değişmeden.

1. **Derleme süresi (ilk iş, kural):** shader'ı derle, uygulamayı aç, Rendered'e
   geç ve ilk RT görüntüsüne kadar bekle. Sonra `render.rt_pipeline_status` →
   `stage_seconds` toplamını ve `x64/Release/shaders/volume_closesthit.spv`
   boyutunu yaz. Taban: 977,104 B; geçmiş ölçümler 314 sn (normal) ve 495 sn
   (geri alınan deneme). Belirgin artış (> %10) varsa tek çağrı yeri kuralı
   tutmamış demektir.
2. **Hız (asıl kabul):** iki gaz görünürken bir RT örneği önceden 964 ms
   (yalnız buhar 366, yalnız duman 86). Aynı ölçüm, kamera sabit:
   `python scripts/test/rt_probe_gas_overlap_cost_ipc.py`.
   Görmen gereken: ikisi birlikteyken ikincil yoğunluk örneği ~0 (önce 3.3/yol),
   GPU ms ~ ayrı ayrı toplamına (~450) yakın. Hâlâ 900+ ise önbellek çakışmada
   kaçırılıyor (`radiance_cache_fallbacks` sayacına bak).
3. **Görüntü:** iki gazın çakıştığı yerde sekme ışığı (zemine ve sıvıya düşen
   dolaylı ışık) öncekinden belirgin parlak ya da karanlık olmamalı. Hafif fark
   beklenir: her ortamın J'si yalnız kendi ortamıyla gölgeleniyor.
   ★ Sinsi olan: buhar ve dumanın İKİSİ de blackbody (emission_mode 2). Kamera
   march'ı companion'ın emisyonunu atlıyor, önbellek atlamıyor: düşük sıcaklıkta
   fark görünmez, ama sıcak bir duman companion olursa sekme ışınlarında
   kameradan fazla parlar ve hiçbir şey hata vermez.
4. **Regresyon:** tek gazlı sahnede (yalnız buhar ya da yalnız duman) süre ve
   görüntü değişmemeli; tek ortam yolu aynı formüle iner (J = J₁).

Statik: `check_volume_march_lod_contracts` OK, gaz optik / boş eleme / ışık
önbelleği / malzeme / karışım kontrolleri OK. `check_rt_gpu_timing_contracts`
FAIL, bu partiden bağımsız: kontrol `beginRayGpuTiming(cmd, w, h)` arıyor, kod
FSR partisinde `traceW, traceH`'ye geçmiş (diğer ajanın alanı).

## Emitter bütçesi kg + Max Particles kısma uyarısı — 2026-10-28 (yalnız C++)

Ortak saat partisinin 2. derlemesi CANLI geçti (128'de GRAINS SOFTENED ile
devam; 512'de bölünme 24 × 8). Bu parti shader'a dokunmuyor.
Kaynak sınırı artık kg: `use_mass_limit` + `max_emitted_mass_kg` (eski
`use_particle_limit`/`max_emitted_particles` IPC ve Python'da gerekçeli
reddediliyor, proje yüklemesinde uyarıyla düşüyor). Sıvı parseli doğumda
kütlesini alıyor. Tavan bağlayınca kısılan kütle sayılıyor, defterde
`emission_withheld`, durum çubuğunda sarı EMITTERS THROTTLED.

1. **Kabul testi (en hızlı, her şeyi kapsar):**
   `python scripts/test/rt_test_emission_budget_ipc.py` (boş sahne, Pause).
   Beklenen 6/6: eski anahtar reddi, 2 kg'da durma (bir parsel içinde),
   parsel sayısı = kütle / parcel_mass_kg, tavan kısıyor, domain = kaynak
   toplamı, defterde emission_withheld.
   Bozuksa: `budget_stops_at_mass` FAIL ve kütle 2 kg'ı çok aşıyorsa döngü
   içindeki kg kesmesi çalışmıyor; `ceiling_withholds` FAIL ise tavan kapısı
   artık kısılanı saymıyor.
2. **Panel:** bir sıvı kaynağında "Limit Emitted Mass" + "Max Emitted Mass
   (kg)" ve ilerleme çubuğu kg cinsinden. Kapalıyken "Emitted: x kg (n
   particles)". Domain Max Particles'ı küçük yap (ör. 400): kaynağın altında
   "Max Particles withheld ... kg" satırı, durum çubuğunda sarı **EMITTERS
   THROTTLED** pill'i ve bir kez HUD mesajı. Tıklayınca domain açılmalı.
3. **Senin sahnende (E:
t_k\matter4emiter.rtp) uzun koşu:** su 10k/s,
   tavan 100k. Tavana ulaşınca sarı pill çıkmalı; `fluid.list_domains` →
   `emission_withheld_kg_step` > 0. Buharlaşma sürerken pill kalır, bu doğru:
   kaynak artık kaybı dolduruyor.
4. **Test istemcisi:** test betikleri eski sayı sınırı yerine `parcels=N`
   (ve `limit_parcels=`) geçiyor; `rt_ipc.py` bunu kaynağın
   `parcel_mass_kg`'sıyla kg'a çeviriyor ve tane yarıçapı gibi ayarlar sonradan
   değişirse bir sonraki reset/adımda yeniden boyutluyor. Kontrol:
   `rt_test_snow_heat_ipc.py` (kaynaktan SONRA tane yarıçapını değiştiriyor)
   önceki gibi 11/11 ve aynı tane sayıları (sand 1500, snow 3000).
   Sayılar farklıysa yeniden boyutlama tetiklenmiyor.
5. **Regresyon:** `rt_test_matter_exchange_ledger_ipc.py` (yeni türler izinli),
   `rt_matter_smoke.py --quick`.
6. ★ **Sinsi olan:** sıvı parseli artık doğumda kütlesini alıyor (önceden 0
   doğup bir sonraki adımda dolduruluyordu). Değer aynı formül
   (voksel³/PPC × yoğunluk) ama hacim `state.voxel_size`'tan; bir domain'in
   sıvı ızgarası farklı bir voksel kullanıyorsa ilk adımdaki kütle önceki
   davranıştan sapar ve hiçbir şey çökmez. Kontrol: smoke'taki kütle
   korunumu kapıları (`p9_melt`, `snow_heat` ledger) değişmemeli.
7. `matter.exchanges` özetine (source/target/mass_error) emisyon kayıtları
   girmiyor, yalnız `exchanges` listesinde görünüyor; özet sayıları önceki
   gibi kalmalı.

## Ortak saat: sıvı yol minimumu söküldü — 2026-10-27 (2. derleme, yalnız C++)

1. derlemenin canlı sonucu (kullanıcı sahnesi, probe, limit 512): 120 adımda
**hiç durmadı**; köprü hacmi tabanda (6.54e-9 m³), sınırlayan yay `tangential`
(köprü değil), varsayılan temas 12–14 (eskiden 24). Ama bölünme `154 x 2`:
sürekli ortamı sıvının yol minimumu (her yenilemede < 2.5 mm) 154 adıma
zorluyordu, ve limit 128'de "Matter common clock needs 132 substeps" ile
domain'i durduruyordu (DEM ise 0.5 yumuşatmayla 126'ya sığmıştı).
Bu derleme o minimumu söküyor; shader değişmedi.

1. Limit **128**'de (sahnede şu an 128) Play: durmamalı. Sarı GRAINS
   SOFTENED pill'i görünebilir (DEM 128'e sığmak için yumuşar), kırmızı
   SIM STOPPED görünmemeli. Kırmızı çıkarsa ve mesaj "liquid/MPM continuum
   needs N" diyorsa sürekli ortamın KENDİ CFL'i (sıvı hızı / kar sertliği)
   limiti aşıyor demektir; o gerçek bir sınır.
2. Probe (limit 512): `python scripts/test/rt_probe_grain_clock_ipc.py --every 12`.
   Bölünme `C x K`'da C artık ~24 civarı (kar MPM dalga sınırı 23), K ~13;
   toplam yine ~300. `grain_step_ms` 1. derlemede 142 → 361 ms (parçacık
   arttıkça) idi; aynı adımlarda belirgin düşmeli. Düşmüyorsa maliyet sürekli
   ortamda değil DEM'in kendisinde demektir (o zaman D2-2'nin ikinci yarısı:
   yalnız temastaki taneler).
3. ★ Sinsi olan: tanelerin gördüğü sıvı alanı artık daha seyrek tazeleniyor
   (sürekli adım başına bir kez). Hızlı akan suda taneler sürüklenmeyi bir
   adım geç görür; çökme yok, yalnız taneler akıntıyı biraz "geç" takip eder.
   Kontrol: aşağıdaki 6. madde (`momentum_residual_n_s` ~0) ve gözle kum
   suya düşerken önceki derlemeye göre farklı savruluyor mu.
Aşağıdaki 1. derleme listesi hâlâ geçerli (madde 1–2 CANLI geçti).

## Tane alt adımı: köprü tabanı, sertlik uyarlama, ayrık ortak saat — 2026-10-27

C++ **ve shader** (`sim_matter_grain.glsl` → `compile_shaders.bat`; revizyon
31 → 32). Sahne: `E:
t_k\matter4emiter.rtp` (su, kar, 600 K kum, gaz).
Önceki ölçüm: ortak saat 322 alt adım, ıslanmayla 32. adımda 527 > 512 → durma.
Durma uyarısı (önceki parti) kullanıcı tarafından CANLI doğrulandı.

1. **Derleme + shader.** Exe ve shader'lar birlikte. Shader eskiyse ilk karede
   "grain step shader revision mismatch; rebuild simulation shaders" görürsün:
   bu, derlemenin değil shader'ın eksik olduğunu söyler.
2. **Probe (en hızlı, her şeyi gösterir):**
   `python scripts/test/rt_probe_grain_clock_ipc.py` (sahne açık, Pause).
   Görmen gereken:
   - `never held in 120 steps`.
   - Bölünme `sub N = C x K`: C sürekli ortamın kendi sayısı (kar MPM dalga
     sınırı ~23-24 civarı), K DEM'in her birindeki iç adımı. Eskiden C = 322.
   - Islanmadan sonra (adım ~22+) kaynak `contact`, `bridge` DEĞİL;
     `Vb` ≥ ~6.5e-9 m³ (2.5 cm tanede 1e-4 × tane hacmi).
   - `contacts a/b`: b (varsayılan) 12–10 civarı, 24 değil.
   - `grain_step_ms` öncekinden düşük (önceki tabanı ölçmedik; sayıyı yaz).
   Bozuksa: kaynak hâlâ `bridge` → shader eski (revizyon) ya da taban hesabı
   yanlış; C hâlâ yüzlerce → `.inl`'deki `clock.minimum_substeps` değişikliği
   derlenmemiş.
3. **Sertlik uyarlama:** Physics → Advanced solver → Contact substep limit'i
   64'e indir, Play. Sim durmamalı; durum çubuğunda sarı
   "GRAINS SOFTENED x0.xx" pill'i, DEM panelinde "Contact springs softened"
   satırı. Taneler patlamamalı, daha fazla iç içe geçebilir. Limit'i geri al:
   pill kaybolmalı. Bozuksa (yine kırmızı SIM STOPPED): mesaj "even with the
   contact springs softened 1000x" diyorsa sınırı köprü ya da çok küçük tane
   koyuyor, ayrıntı `cfl_terms`'te.
4. **Islak kum hâlâ yapışkan mı:** `rt_g2_dry_wet_compare_ipc.py` ve H1
   `--wet-only`. Köprü tabanı yalnız çok az sulu taneyi etkilemeli; ıslak kum
   davranışı değişmemeli. Islak taraf kuruya yaklaştıysa taban çok yüksek.
5. **Ortak saat regresyonu:** `rt_test_mpm_dem_shared_clock_ipc.py` (4/4
   bekleniyor), `rt_test_snow_heat_ipc.py`, `rt_matter_smoke.py --quick`.
6. ★ **Sinsi olan — sıvı sürüklemesi/kaldırma ölçeği:** DEM artık bir sürekli
   adımda K kez dönüyor; sürükleme dürtüsü K adım boyunca birikip sıvıya bir
   kez dönüyor, kaldırma kuvveti sürekli adımın dt'siyle. Yanlışsa hiçbir şey
   çökmez: taneler suda biraz fazla yüzer ya da batar ve "fizik böyle" gibi
   görünür. Kontrol: `fluid.matter_models` → `grain_diagnostics.liquid`
   `momentum_residual_n_s` önceki gibi ~0 kalmalı ve `buoyancy_impulse_n_s`
   ≈ Σ(ρ_su·g·V_batık)·t. Artık büyüdüyse tepki K ile ölçeklenmiş demektir.
7. **Temas bütçesi aşımı:** yoğun bir yığında (`rt_h1_dense_grain_scene_ipc.py`
   ortak saatte) `runtime.cfl_contacts_exceeded` arada bir true olabilir; bu
   bir sonraki kareyi büyütür. Sürekli true ve yığın titriyorsa margin (+4)
   yetmiyor.

Statik kontroller: 11 `check_matter_*`'tan 7'si PASS. Dördü **bu partiden
önce de** eski pinlerle düşüyor: `check_matter_grain_contracts`
(`const uint SLOTS = 24u` arıyor; revizyon 30'dan beri kapasite çalışma
zamanında), iki sleep kontrolü (`5 * capacity * sizeof(uint32_t)`),
`check_matter_gpu_contracts` (`stress_update` push 100 ≠ 68). Sonra listesine.

## SVGF ilk à-trous geçmişi — 2026-10-27

Kaynak yazıldı; shader/C++ derlenmedi, canlı kalite ve maliyet kabulü bekliyor.

1. C++ ve iki shader varyantını (`realtime_svgf.spv`, `realtime_svgf32.spv`)
   kullanıcı derlemeli. Yeni historyFlags biti 32 iki tarafta eşleşir; eski shader
   ile yeni C++ bu iyileştirmeyi uygulamaz.
2. Önceki derlemeyle aynı iç mekan, kamera hareketi, çözünürlük, sample ve SVGF
   ayarlarında karşılaştır. İlk à-trous rengi yalnız hareket karelerinde geçmişe
   gider; son geniş filtre sonucu, momentler ve history length değişmez.
3. FSR kapalıyken duvar/tavan lekesi ve kareler arası oynama azalmış mı ölç.
   İnce pervaz, tel, doku, yazı ve parlak yansımada zamanla yumuşama olmamalı.
   Aynı kontrolleri FSR Quality/Balanced ve timeline/simülasyon hareketinde yap.
4. Hızlı kamera hareketi, yeni açılan yüzey, hareketli nesne ve ışık değişiminde
   hayalet izi olmamalı. Reactive > 0.5 pikseller uzaysal filtreden önceki rengi
   saklamayı sürdürür; geçmiş kabul/reddetme testleri değiştirilmedi.
5. Kamera durunca donmuş hareket geçmişi tekrar filtrelenmemeli. OIDN açık/kapalı
   duruş geçişini, resize, sahne değişimi ve SVGF kapat/aç işlemlerini kontrol et.
   Ham birikim/final render sonucu aynı kalmalı.
6. `render.realtime_denoiser_stats` ile aynı render çözünürlüğünde temporal,
   variance, atrous ve total GPU ms karşılaştır. Yeni dispatch veya tampon yok;
   ilk geçişte piksel başına bir history okuması/yazması var. Kazanç ve maliyet
   canlı ölçülmeden PASS sayılmaz.

## FSR proje kaydı — 2026-10-27

Kullanıcı derleyip kayıt/yükleme işleminin başarılı olduğunu doğruladı.
Shader değişmedi. Aşağıdaki ayrıntılı regresyon maddelerinin tamamı için ayrı
kabul kaydı henüz yok.

1. C++ derlemesinden sonra FSR modunu Auto, kaliteyi Balanced, sharpness'ı 0.35,
   rejection_reactive'i false yapıp projeyi kaydet. Uygulamayı yeniden açıp projeyi
   yükle; panel ve IPC `render.upscale_stats` / Python `render.upscale_stats()`
   üzerinden dört tercihin korunduğunu kontrol et.
2. Off ve Fsr modlarını, diğer kalite seçeneklerini, sharpness 0/1 ve reactive true
   değerlerini de kaydet/yükle ile kontrol et. UI ve `render.set_upscale` ile yapılan
   değişiklikler aynı `RenderSettings::upscale` kaydını kullanır.
3. `render_settings.upscale` içermeyen eski proje, önceki açık projenin FSR tercihini
   taşımadan Off / Quality / 0.2 / true varsayılanlarına dönmeli.
4. Geçersiz mode/quality, sayısal olmayan veya aralık dışı sharpness ve yanlış tipte
   reactive değeri yükleme hatası vermeli. Kayıtta GPU status/history bulunmamalı.
5. FSR kullanılamayan backend/donanımda kayıtlı tercih korunmalı; mevcut uygunluk
   kontrolü ve tam çözünürlük fallback'i çalışmalı. Kaydet/yükle işlemini IPC ve
   Python proje API'leri üzerinden de doğrula.

> **B4 Auto geçişi çökme — 2026-10-27:** Kullanıcı ffxCreateContext içinde
> DLL → null function çağrısı bildirdi. SDK 1.1.4 kaynak denetimi, backend'in
> vkGetBufferMemoryRequirements2KHR adresini yüklediğini doğruladı; uygulama bu
> extension'ı açmıyor (Vulkan core sürümünü kullanıyor). FSR'ye verilen resolver,
> KHR adresi yoksa ABI eşdeğeri core vkGetBufferMemoryRequirements2'ye dönüyor.
> DLL çağrısından önce SDK'nin gerekli device komutları kontrol ediliyor;
> eksik komut durumunda adıyla tam çözünürlük fallback yapılıyor.
> Kök neden canlı debugger ile henüz doğrulanmadı. C++ Release yeniden derlenip
> uygulama yeniden açılarak Auto/Pro hareket testi yapılmalı; shader değişmedi.


> **B4 ilk canlı kontrol — 2026-10-27:** Kullanıcı derledi. 1920×1080,
> Auto / Ultra Performance ile durağan durumda giriş=çıkış (beklenen).
> Harici IPC kamera hareket turunda da active=false, dll_loaded=false,
> dispatches=0; genel uygunluk engeline takıldı. Kamera tur sonunda geri yüklendi.
> Kaynak denetimi: camera_mode==0 koşulu varsayılan Pro (1) modunu da dışlıyordu.
> Auto+Pro desteği ve engelleyen koşulu ayrı ayrı bildiren inactive_reason eklendi.
> Bu düzeltme için kullanıcı C++ Release derlemesi gerekir; FSR GPU kabulü henüz PASS değil.


> **Durum:** DERLEME BEKLİYOR — 2026-10-27 B4d (lens distortion için hız vektörleri, FSR distortion kapısı kaldırıldı). B4c ve öncesi CANLI, `e44153d9`. Ayrıntı: `RT_REALTIME_DENOISE_UPSCALE_PLAN.md` en üstteki devir notu.

## B4d — distortion'lı kamerada doğru hız vektörü + FSR (2026-10-27, derlenmedi; RAYGEN + C++)

**Kullanıcı:** `compile_shaders.bat` (`raygen.rgen` değişti) + C++ x64 Release. Push/binding
değişmedi. **Statik:** glslc `--target-env=vulkan1.3 --target-spv=spv1.4` çıktısız temiz.

1. **Kapı kalktı** — distortion'lı eski bir projeyi aç, FSR balanced, kamerayı hareket ettir.
   `render.upscale_stats`. Görmen gereken: `active=true`, render ~988×555, `inactive_reason`
   artık "camera lens distortion is active" DEĞİL. Bozuksa: eski exe.

2. **Vektör doğru mu (distortion'sız regresyon önce)** — distortion 0'lı sahnede
   `render.set_settings debug_view=16` (motion vectors) + küçük yörünge: B4'teki gibi.
   Sonra distortion'lı sahnede aynısı. Görmen gereken: düz bir duvarda vektör renkleri
   merkezden kenara **pürüzsüz** değişir, kenar bandında ani sıçrama yok.
   Bozuksa: kenarda dairesel bir halka/sıçrama → Newton tersi yanlış işaretli veya aspect
   ölçeği hatalı (`motionLensPx`).

3. **SVGF geçmişi kenarlarda** — distortion'lı sahne, `debug_history=true`, yavaş yörünge.
   Görmen gereken: kenarlar da yeşil (uzun geçmiş). Önceden kenar bandı sarı/kırmızıya
   kayıyordu (yanlış piksele bakıyordu). Bozuksa: kenarlar hâlâ kırmızı → vektör hâlâ pinhole.

4. **★ Sessiz arıza: FSR ile kenarda kayma** — distortion'lı sahnede FSR açık, ince
   çizgili/desenli bir yüzeyi ekran kenarına getirip yatay yörünge. Görmen gereken:
   desen kenarda da keskin ve yerinde. Bozuksa: kenarlarda "biraz yüzen/bulanık" desen —
   kimse hata diye raporlamaz; vektör ölçeği kenarda yanlış demektir.

5. **Güçlü barrel** — distortion'ı panelde en negatif değere çek. Görmen gereken: çökme yok,
   aşırı kenar bölgesi en kötü ihtimalle reddedilir (debug_history kırmızı, iz yok).

**B4d+ — gizli distortion durumu (aynı derleme, yalnız C++):** araştırma: model CPU/OptiX/
Vulkan RT'de aynı ve doğru; RayFusion uygulamıyor. Sorun durumdu: canlı panel YOKTU
(`scene_ui_camera.cpp` `drawCameraContent` yorum satırında), değer projede kayıtlı/yüklü,
iki çelişen otomatik formül (`Camera.cpp` her modda FOV'dan, `Camera.h` Cinema'da).
Düzeltme: proje anahtarı `lens_distortion` (eski `distortion` OKUNMAZ → eski projeler 0 açılır),
`Camera.cpp` formülü söküldü (tek sahip `calculateLensCharacteristics`, yalnız Cinema),
FOV altına her modda "Lens Distortion" slider'ı, IPC `camera.set_lens_distortion` +
`camera.get` `lens_distortion`/`auto_lens`, Python `rt.camera.set(lens_distortion=..)`.

6. **Eski proje sıfır açılır** — önceden FSR'ı "distortion" yüzünden kapatan eski projeyi aç.
   `camera.get`. Görmen gereken: `lens_distortion=0`, panelde slider 0. Kaydet, yeniden aç:
   hâlâ 0. Bozuksa: eski anahtar hâlâ okunuyor.
7. **Yeni değer kalıcı** — slider'ı −0.1 yap, kaydet, yeniden aç → −0.1. IPC ile 0.6 dene →
   ret (kırpma yok). Görüntü kenarları bükülmeli, RayFusion'da bükülmemeli (beklenen).
8. **Auto lens sahipliği** — Cinema + Auto Lens Characteristics aç: slider kilitli ve değeri
   gösterir, `camera.set_lens_distortion` reddedilir. Pro'ya geç: slider açılır ve değer artık
   FOV değiştikçe KENDİLİĞİNDEN DEĞİŞMEZ. **★ Sessiz arıza:** Pro'da FOV oynatınca kenar bükümü
   değişiyorsa eski `Camera.cpp` formülü hâlâ çalışıyor (eski exe).

## B4c+ ek (derlendi, CANLI) — son kare, parçacık hayaleti, merkez okuma

Kullanıcı derleyip doğruladı: son kare artık ham gürültüye dönmüyor (duruş SVGF'si
yakınsamış karede de gürültüsü hedefin üstündeki piksellerde filtreli), FSR'da akışkan
parçacık hayaleti yok (`rejection_reactive`: kümelenmiş raygen reddi → reactive).
Ölçülen: skin sahnesi SVGF 10.5 ms (à-trous merkez okuma düzenlemesi yalnız %4).
Bilinen uç durum: yakınsamadan sonra yalnız tonemap'i yenileyen işlem (pozlama
değişikliği) SVGF'siz ham görüntüyü gösterebilir — gözlenirse raporla.

## B4c — sahne animasyonu SVGF ve FSR için hareket sayılır (2026-10-27, derlenmedi; YALNIZ C++)

**Kullanıcı:** yalnız C++ x64 Release; shader değişmedi. Yeni dosya yok.

**Kök (canlı, skin animasyonlu sahne, kamera duruyor):** `moving=false`, `static_active=true`,
`sample=1`, saniyede 38.7 SVGF çalışması ve **38.7 geçmiş sıfırlaması**, ortalama geçmiş **1.0**,
SVGF **11.8 ms/kare** (à-trous 7.6) tam çözünürlükte, FSR kapalı. Nesne hareketi
algılaması (TLAS seri ↔ "görülen" kopya) aynı karenin render'ında ikisi birden
ilerlediği için policy anında hep eşitti: yapısal olarak hiç tetiklenmiyordu.
Yeni: önceki trace'e bakılır; kamera dururken art arda ≥2 trace birikimi sıfırladıysa
`motion_source="scene"` ve SVGF ile FSR hareket modunda çalışır. Eski seri karşılaştırması söküldü.

**Statik kontrol:** IPC audit PASS, descriptor üretildi, probe AST temiz ve Release aynasında.

1. **API probe'u** — `python scripts/test/rt_probe_realtime_denoiser_ipc.py`.
   Görmen gereken: PASS, `scene_motion` bool değilse reddediliyor, stats'ta `motion_source`
   ve `scene_reset_streak` var. Bozuksa: alan IPC listesinde yok (eski exe?).

2. **Skin sahnesi, animasyon oynuyor, kamera duruyor** — `render.realtime_denoiser_stats`
   (synchronize=true) 2 sn arayla iki kez.
   Görmen gereken: `motion_source="scene"`, `moving=true`, `static_active=false`,
   `history_resets` artışı ≈ 0, `history_length_mean` 1.0'dan belirgin yüksek (arka plan 24'e
   yaklaşır). FSR açıksa `upscale_stats.active=true` ve render ~988×555, SVGF `total_ms`
   11.8'den belirgin düşük.
   Bozuksa: `scene_reset_streak` 0–1'de kalıyorsa animasyon her trace'te birikimi sıfırlamıyor
   (ya da kamera hareketli sanılıyor); streak ≥2 ama source "none" → `scene_motion` kapalı.

3. **Animasyon durunca normale dönüş** — oynatmayı durdur.
   Görmen gereken: birkaç kare içinde `motion_source="none"`, FSR tam çözünürlüğe döner (bir kez
   reset), birikim yükselir, duruş SVGF'si (B4b) devreye girer.
   **★ Sessiz arıza:** streak kendini besler ve source "scene"de takılı kalırsa görüntü hep
   düşük çözünürlüklü ve 1 spp kalır ama "hareket ediyor" gibi makul görünür. Kontrol:
   durduktan sonra `sample` artıyor mu?

4. **Tek değişiklik animasyon değildir** — kamera dururken tek bir `scene.set_transform` ya da
   bir slider değişikliği yap. Görmen gereken: `scene_reset_streak` 1'e çıkar ve sıfırlanır,
   `motion_source` "none" kalır, FSR devreye girmez. Bozuksa: eşik 1'e düşmüş ya da her
   düzenlemede gereksiz çözünürlük geçişi oluyor (titreme).

5. **Görüntü: skin karakteri hareket ederken** — `viewport.get_screenshot`, `debug_history=true`.
   Görmen gereken: duvar ve zemin yeşil (uzun geçmiş), deforme olan karakter kırmızı/sarı
   (tutarlılık testi reddediyor, beklenen). Karakterin arkasında iz varsa → rijit hız
   yaklaşımı deformasyonu yanlış reprojekte ediyor (ayrı iş, hacim ajanının hız sözleşmesine).

6. **Kapatma A/B'si** — `render.set_realtime_denoiser scene_motion=false`.
   Görmen gereken: eski ölçümler geri gelir (geçmiş 1.0, ~11.8 ms). Kanıt, bu partinin farkın
   kaynağı olduğunu gösterir.

## B4b — duruş sönmesi ölçülen gürültüyle + FSR'ı SVGF'den ayırma (2026-10-27, derlenmedi; SHADER + C++)

**Kullanıcı:** `compile_shaders.bat` (yalnız `realtime_svgf.comp` değişti → `realtime_svgf.spv`
ve `realtime_svgf32.spv`), ardından C++ x64 Release. Yeni C++: `Backend/VulkanRawGuard.cpp`
(vcxproj/filters'ta). SVGF push hâlâ **64 bayt**; yeni bindingler 15 (r32f Welford M2)
ve 16 (slot başına sayaç SSBO). **Eski .spv + yeni exe = descriptor uyumsuzluğu**: shader
derlenmeden C++ çalıştırılırsa SVGF pipeline'ı hata verir (reason'da görünür).

**Statik kontrol:** glslangValidator iki varyant (HDR16 / RAW32) temiz; IPC audit PASS
(709 metot), descriptor üretildi; iki probe `py_compile` temiz ve Release aynasında.
Derleme ve canlı GPU doğrulaması DEĞİLDİR.

Sıra: bağımsız ve hızlı olanlar önce; görüntü sonuçları en sonda.

1. **API/IPC probe'ları** — `python scripts/test/rt_probe_realtime_denoiser_ipc.py`,
   ardından `python scripts/test/rt_probe_upscale_ipc.py`.
   Görmen gereken: ikisi de `"api_validation": "PASS"`. Upscale probe'u artık SVGF kapalıyken
   FSR isteğinin **korunduğunu** ve `mode="auto"`'nun SVGF'siz **kabul edildiğini** doğrular;
   denoiser probe'u `still_noise` aralığını ve eski `static_samples` adının **reddini** doğrular.
   Bozuksa: "FSR refused without SVGF" → `RtApiUpscale.cpp` kapısı hâlâ duruyor (eski exe?);
   `static_samples` kabul edildiyse IPC alan listesi eski.

2. **Hareket AOV'u FSR'ı korur** — FSR açıkken, SVGF kapalıyken
   `render.set_motion enabled=false`. Görmen gereken: "disable upscale before disabling the
   motion AOV" hatası. Bozuksa: FSR hız vektörsüz çalışmaya çalışır (eskiden bunu yalnız SVGF
   kuralı dolaylı engelliyordu).

3. **Temiz sahnede duruş SVGF'si kendiliğinden durur** — az gürültülü bir dış sahne, OIDN
   kapalı, kamera dur. `render.realtime_denoiser_stats` birkaç saniye arayla.
   Görmen gereken: `static_active=true` ile başlar, `still_filtered_fraction` 1.0'dan düşer,
   kısa sürede `still_complete=true` olur ve `dispatches` artmayı keser.
   Bozuksa: fraction hep -1 → sayaç okunmuyor (slot etiketi / eşleme); hep 1.0 → shader M2'yi
   okumuyor (binding 15 eski .spv) ya da `still_noise` 0.

4. **★ Gürültülü iç mekanda duruş: örnek arttıkça DAHA TEMİZ** — yatak odası, OIDN kapalı,
   hedef 128. 8, 32, 64, 100 ve 127. örneklerde `viewport.get_screenshot` + aynı bölgede
   `render.probe` (duvar/tavan).
   Görmen gereken: gürültü monoton azalır; eskiden 32→64 arasında artıyordu (σ√N/64).
   `still_filtered_fraction` 64'te hâlâ büyük olabilir (doğru: oda 2% RSE'ye inmiyor);
   son çeyrekte (96→127) ham görüntüye yumuşak geçiş, 127'de **sıçrama yok**.
   Bozuksa: 8'den önce lekeli ham noktalar → güven eşiği çalışmıyor; 127'de sert geçiş →
   ev sahibi rampası (rawBlend) gelmiyor.
   **★ Sessiz arıza:** her şey "makul" görünür ama yakınsamış kare filtreli kalır. Kontrol:
   hedefe ulaşınca SVGF kapalıyken alınan 128 spp karesi ile **bit-bit aynı** olmalı
   (B3c ölçütü: meanabs 0).

5. **SVGF kapalı + FSR açık** — `render.set_realtime_denoiser mode=off`, FSR balanced,
   kamera hareket. `render.upscale_stats`. Görmen gereken: `active=true`,
   `input_provider="none"`, `svgf_filter_ms=0`; görüntü gürültülü ama doğru konumda, durunca
   tam çözünürlüğe döner ve birikim temiz. Sonra SVGF'yi aç: `input_provider="svgf"`.
   Bozuksa: hareket karesi siyah/eski → guard.save çalışmıyor (raw yedeği yok); durunca
   kalıcı bulanıklık veya FSR izi → guard.restore atlanıyor (birikim kirlendi — ciddi).

6. **SVGF + FSR regresyonu** — varsayılanlar (SVGF açık, FSR balanced), aynı oda, 14°
   yörünge. Görmen gereken: B4'teki gibi active 988×555, FSR ~1.1 ms, parlaklık durağanla
   aynı (bu oturumda 0.470 / 0.462). Bozuksa: guard sırası SVGF'nin yedek okumasını bozdu
   (SVGF artık backup'ı guard'dan alıyor).

## B4 — FSR 3.1 Vulkan, tek derleme ve parti sonunda tek IPC kabul turu

**Kullanıcı:** tüm shader'lar için `RayTrophiStudio/compile_shaders.bat`, ardından
C++ x64 Release. Bu tek partidir; ajan SDK/shader/C++ derlemesi çalıştırmaz.
Yeni shader: `upscale_inputs.comp` → `upscale_inputs.spv` (HDR16),
`upscale_inputs32.spv` (`RAW32`, HDR32). Push **64 bayt**, 8 storage image.
RT camera push **212**, B2 motion record **112**, SVGF push **64 bayt** değişmedi.
Yeni kaynaklar proje/filter'da. `FidelityFX.targets` DLL+MIT lisansı `$(OutDir)`'e
kopyalar; **x64/Release/amd_fidelityfx_vk.dll** şu anda hazır. SDK derlenmez,
import lib bağlanmaz. Uygulama yeniden açıldığında yeni Vulkan mantıksal cihaz
özellikleri de kullanılacak. Varsayılan upscale off; SDK varlığı hız PASS değildir.

**Statik kontrol:** IPC audit PASS (709 metot), descriptor `--check` PASS;
proje/filter/targets XML, 8 yeni C++ modül kaydı, probe AST ve Release aynaları
PASS. Bunlar C++/shader derlemesi veya canlı GPU doğrulaması değildir.

Ön koşul: Vulkan Rendered, standart perspektif/lens, SVGf ve motion açık;
OIDN viewport seçeneği ayrıca korunur. İç oda 1680x945, aynı bounce/pozlama,
motion profile kapalı; ardından bulut/gaz sahnesi. Dış named pipe
`scripts/test/rt_ipc.py`; gömülü script içinde bekleme döngüsü YOK.
Windows error 5 olursa aynı probe gerekli sandbox yükseltmesiyle yeniden koşulur.

1. **İLK: soğuk DLL-yok kapısı.** Yeni açılan uygulama henüz upscale off iken,
   FSR DLL'ini geçici olarak Release dışında tut; ilk mode=auto ve kamera
   hareketinde uygulama çalışmalı, available/active false + açıklayıcı reason,
   tam çözünürlük. Sonra DLL'i geri koy, mode off/on ve tekrar hareket: dll_loaded,
   available, active true; ffx_version provider'dan dolu. Zaten yüklenmiş DLL'i
   sonradan kaldırmak bu testi yapmaz. Başka DLL indirme/taşıma yok; orijinali koru.
2. **API atomikliği/bağımlılıklar.** Dış terminal:
   `python scripts/test/rt_probe_upscale_ipc.py`. 11 geçersiz güncelleme reddi,
   kısmi yazım yok, aktif upscale'nin SVGF/motion bağımlılığı korunmalı; başlangıç
   ayarları geri yüklenir. Python eş yüzeyleri `rt.render.set_upscale(...)`,
   `rt.render.upscale_stats(synchronize=False)`; yanlış bool/enum/range reddi.
3. **Boyut ve jitter.** Rendered'da
   `python scripts/test/rt_probe_upscale_ipc.py --motion-seconds 4`.
   balanced: render_w/h display'den küçük, active/dispatches artar; jitter phase
   SDK'dan (>8), authored Halton tercihi değişmez. Durunca active false ve
   render=display. ★ Sessiz arıza: mode fsr iken active false / dispatches sabit
   ise FPS karşılaştırmasını kazanım sayma; reason/inactive_reason oku.
4. **İşaret/ölçek.** Aynı 1° sağ dönüş: native RT mean_dx yaklaşık +28 px;
   balanced RT vektörü yaklaşık +28 * render_w/display_w. FSR ekranında kayma
   yine yaklaşık28 display px. Ters işaret, ölçek katlanması, yüzme/bulanıklık
   → motionVectorScale/jitter/depth sözleşmesi bozuk. Duran kamera ~0 px.
   Motion sample'ı hareket çağrısının hemen ardından al, sonraki dispatch üstüne yazar.
5. **Hız.** Mode off ve balanced için ayrı `set_gpu_timing enabled=true`
   pencereleri; RT ms/örnek medyanı ve ölçülen piksel boyutu. Balanced RT
   **<=0.5x** native hedefi. `upscale_stats synchronize=true`: GPU timing
   available true, gpu_ms + svgf_filter_ms OIDN fast ~44 ms'nin altında hedef.
   İlk context/pipeline kurulumu hariç. `realtime_denoiser_stats.total_ms` FSR'ı
   da içerdiğinden iki kez toplama; gerçek hareket FPS'ini ayrıca sayıyla yaz.
6. **Duruş/tek örnek.** OIDN açık/kapalı iki durumda ani duruşta eski poz,
   çift kontur veya kararma/gürültü sıçraması olmamalı. Bridge yalnız doğru
   poz/epoch/yüzeyde; slider düzenlemesinden sonra eski FSR görüntüsü karışmamalı.
   Target=1'de durunca eski düşük çözünürlüklü karede kalmamalı; son tonemap
   pompalanır. Çalışmazsa epoch/slot/reset/bridge kimliği hatalıdır.
7. **Referans.** Sabit pozlama/seed/zamanla native off ve hareketten sonra full
   128 örnek: ham radyans ve son ekran bit bit aynı hedef. Birikim sayısının
   ÖNCE sıfırlanmasını bekle; eski completed bayrağıyla erken çıkma. Filtreli veya
   upscaled radyansın Welford/OIDN girdisine sızması en ağır sessiz arızadır.
8. **Kalite.** Pervaz/avize/hasır/saç yörüngede titrememeli; hareketli nesne
   arkasında iz <4 kare; gaz/nükleer/bulut reactive bölgede hayaletsiz.
   Gürültü, parlaklık ve hayalet sayısını görüntü ölçümüyle raporla; yalnız FPS
   artışı kalite kabulü değildir. B3 §7 açık işlerini yeni B4 başarısı sayma.
9. **Yaşam döngüsü.** Hareket sırasında quality değiştirme, tek/çift viewport
   boyutlarıyla resize, Material/Solid/Rendered geçişi; siyah kenar/eski slot/
   hatalı boyut yok. Raster/durağan/final/debug ve desteklenmeyen lenslerde
   active false, tam çözünürlük; raster boşta gereksiz sürekli redraw olmamalı.
   Backend değişimi/device recovery eski DLL context/descriptors taşımamalı.
10. **Hata ve temizlik.** SDK/feature/context/shader hatasında full-res reason;
    başarısız FSR slot'u yayımlanmaz. Kamera ve tüm seçenekleri başlangıca al,
    kullanıcı projesini kaydetme. Geçici resimleri yalnız `docs/dev/.captures/`
    altında tut ve sonunda sil. Sonuçları buranın başına tek rapor olarak yaz;
    yeni bulgular plan §7'ye. Kabulden sonra kullanıcı auto'yu açar.

## B4 — SDK hazırlığı (2026-10-26) — kaynak/dağıtım doğrulandı

1. **PASS:** resmî SDK v1.1.4 arşivi (470,858,939 bayt), SHA-256 manifestte.
   `amd_fidelityfx_vk.dll`: x64, AMD Authenticode Valid, 5 FFX API export'u mevcut.
2. **PASS:** `x64/Release/amd_fidelityfx_vk.dll` + `FidelityFX-LICENSE.txt` hazır;
   mevcut `x64/Debug` kopyaları da aynı hash/lisans. Beş API başlığı third_party'de.
3. **PASS:** `scripts/setup_fidelityfx.ps1` yerel pinli arşivle çalıştı; PowerShell
   parser, dosya hash'leri, proje/filter/targets XML ve import/kopya bağlantıları
   doğrulandı. `FidelityFX.targets` sonraki normal x64 derlemelerinde `$(OutDir)`
   DLL+lisans dağıtımını yapar; SDK kaynak derlemesi veya import lib bağımlılığı yok.
4. **HENÜZ YOK:** FSR loader/context, düşük çözünürlük dispatch, UI/API ve canlı
   GPU kabulü. DLL indirildi/kopyalandı diye `available=true`/FPS kazanımı sayılmaz.
   Uygulama ve SDK DLL çalıştırılmadı. SDK hazırlığı için ayrı uygulama derlemesi
   gerekmiyor; §6 somut planı onaylandıktan sonra tüm B4 tek derleme partisi,
   canlı kabul listesi o parti sonunda güncellenecek.

## B3f SONUÇLARI (2026-10-26, canlı, iç oda, OIDN kapalı) — PASS; B3 ailesi KAPANDI

- (1) PASS durma anı: yakınsamış ham 0.470; hareket 0.475–0.481 (+%1–2); durunca 4. örnek
  0.469, 16. örnek 0.472 → düşüş ~%1, hareket karelerinin kendi oynamasından küçük
  (B3e: %4.5). Gürültü 0.026 → 0.033 (B3e: 0.023 → 0.039).
- (4) PASS 128 örnekte ekran bit bit aynı (static 128/256 vs 0).
- (2) eski poz / çift kontur ve (3) slider sıfırlaması: koşulmadı (kullanıcı gözü yeterli görüldü).

B3 ailesinden Sonra'ya kalanlar (plan §7): sağ dikey pervazda hafif noktalanma; hareket
karesi yakınsamıştan +%1–2 parlak; à-trous 6.8 ms (hedef ≤ 6); hayalet (<4 kare), hacim
reactive, bulutlu sahne siluetleri, timeline oynatımında SVGF tetiklenmemesi ölçülmedi.

## B3f — durma geçişi: dondurulmuş hareket geçmişi (2026-10-26, derlenmedi; YALNIZ realtime_svgf shader'ı)

Durağan karede SVGF artık hareket sırasında biriken geçmişi atmıyor: geçmiş DONDURULUR
(değişmeden ileri taşınır, birikime girmez) ve birikimle örnek sayısına göre harmanlanır,
`(geçmiş·n_h + birikim·N) / (n_h + N)`; geçmişin payı örnek geldikçe kendiliğinden söner.
Geçmiş yalnız SVGF geçmişi ardışık geçerliyse kullanılır (kameranın durmasıyla gelen
geçiş; slider düzenlemesi gibi hareketsiz sıfırlamada kullanılmaz).

1. **★ Durma anı.** İç oda, OIDN kapalı, static_samples 64: hareketten sonra her yeni
   örnekte ekran ortalaması ve gürültüsü. B3e: son hareket 0.487 / 0.023 → 4. örnek
   **0.465** / 0.039 → 16. örnek 0.479. Hedef: 1–8. örnek parlaklığı hareket karesinin
   %1 içinde, gürültü sıçraması yok.
2. **★ Sessiz arıza — eski poz:** dondurulan geçmiş bir önceki pozdan kalırsa durunca
   bir an çift görüntü / kayma olur. Hızlı dönüşten ani durmada kenarlarda çift kontur
   olmamalı.
3. Slider düzenlemesi (kamera sabit, örn. bounce değiştir): durağan kare geçmiş
   kullanmamalı (eski ayarın görüntüsü karışmamalı) — ilk örneklerde yeni ayar görünür.
4. 128 örnekte ekran bit bit aynı (B3d PASS korunmalı).

## B3e SONUÇLARI (2026-10-26, canlı) — ince hatlar düzeldi; durma anı FAIL

- ★ İnce hatlar: hareketli karede pervaz/avize kolu çizgileri belirgin şekilde sürekli;
  sağ dikey pervazda hafif noktalanma kaldı (B3d'den çok az).
- Hareket gürültüsü 0.031 → **0.023** (ön-filtre filtreyi güçlendirdi), parlaklık 0.487.
- ★ FAIL (kullanıcı gördü: "durunca kararma ve gürültü, hızla temizleniyor"): son hareket
  0.487/0.023 → durağan 4. örnek 0.465/0.039, 16. örnekte 0.479. Kök: durağan mod ~20
  karelik hareket geçmişini atıp 1–4 örnekli birikimin yalnız uzaysal filtresine geçiyordu
  (çarpık dağılım + kenar durdurucu → koyu, az örnek → gürültülü). → B3f.

## B3e — à-trous varyans ön-filtresi (2026-10-26, derlenmedi; YALNIZ realtime_svgf shader'ı)

C++ değişmedi; yalnız `realtime_svgf.comp` (iki varyant). Luminans kenar durdurucusunun
eşiği (`4·√varyans`) artık 3×3 Gauss ile ön-filtrelenmiş varyanstan (Schied 2017).

1. **★ İnce hatlar.** İç oda, durağan 8 örnek (static_samples 64, OIDN kapalı) ve hareket:
   pervaz olukları / avize kolu kırpıntısı yakınsamış kareyle — B3d'deki noktalı koyu
   çizgiler sürekli ve yakınsamış kadar soluk olmalı.
2. **Gürültü / parlaklık.** Hareket parlaklığı 0.482 civarı kalmalı; gürültü B3b'nin 0.031'inden
   kötüleşmemeli (ön-filtre filtreyi biraz güçlendirir, gürültü düşebilir).
3. **★ Sessiz arıza — fazla yumuşama:** doku/kontak gölgesi detayı yakınsamış kareden belirgin
   silik → ön-filtre eşiği fazla genişletti; à-trous ms aynı kalmalı (+9 okuma, ~0.3 ms).
4. Durağan eğri (B3c tablosu): 8 örnekte fark ≤ 12.9.

## B3d SONUÇLARI (2026-10-26, canlı, iç oda) — 3/3 PASS, kök yeni yerde

- (1) PASS sızıntı: static_samples 128 ve 256, max 128 → yakınsamış ekran static 0 ile bit bit
  aynı, static_active false.
- (2) PASS varsayılan 64 (yeni başlatmada).
- (3) ★ Geçmiş debug: hareket sırasında piksellerin %98'i yeşil (uzun geçmiş), sarı %1,
  kırmızı ~0; kırmızı/sarı yalnız kenar şeritlerinde serpiştirilmiş. Koyu noktaların çoğu
  YEŞİL piksellerde → kök geçmiş değil FİLTRE. Kullanıcı gözlemi: "ince hatları kırıyor".
  Durağan 8 örnekte de pervaz oluklarının 1 px koyu çizgileri yakınsamıştan koyu ve
  NOKTALI. Kök: à-trous luminans eşiği piksel başına ham varyanstan → komşu hat pikselleri
  rastgele farklı eşik alıyor. Makaledeki 3×3 varyans ön-filtresi eksikti → B3e.
- Not: B3c'deki kamera piksel izi geçmiş bulmayı iyileştirdi (kırmızı neredeyse yok) ama
  görünen koyu noktalar filtrenin işiydi; iki adım birlikte gerekliydi.

## B3d — durağan limit düzeltmesi + varsayılan 64 + SVGF geçmiş debug (2026-10-26, derlenmedi)

**Derleme:** `realtime_svgf.comp` (iki varyant) + C++; RT pipeline derlenmez.
- Durağan limit artık `hedef − 1` ile kırpılıyor (B3c'de hedefe eşit limit son ekrandaki
  kareye sızdı, aşağıda). `static_active` son durağan kareden sonra false'a döner.
- `static_samples` varsayılanı **64**.
- Ölçü aleti: `render.set_realtime_denoiser {debug_history:true}` (panel: "SVGF History
  Debug") → SVGF karelerinde piksel başına geçmiş uzunluğu: kırmızı = yok, sarı = < 4,
  yeşil rampa = uzun. IPC/Python/overlay/panel hazır, audit OK.

1. **Sızıntı yok.** static_samples 128 ve 256, max samples 128: yakınsamış ekran static 0 ile
   bit bit aynı (B3c: meanabs 0.84, max 27). `static_active` 128'de false.
2. **Varsayılan.** Yeni başlatmada static_samples 64; OIDN kapalı, kamera durunca
   `static_active` true, 63'ten sonra false.
3. **★ Koyu noktaların kökü (önce bu ölçülür, düzeltme sonra).** iç oda, hareket sırasında
   `debug_history:true` + ekran görüntüsü: hasır delikleri / pervaz kenarı / avize kolu
   kırmızı mı (geçmiş bulunamıyor) yoksa yeşil mi (geçmiş var, koyuluk başka yerden: albedo
   demodülasyonu / alfa / à-trous)? Sonuca göre düzeltme seçilir; tahminle yazılmaz.

## B3c SONUÇLARI (2026-10-26, canlı, iç oda) — durağan mod PASS (sızıntı düzeltmesi B3d), koyu noktalar SÜRÜYOR

- Panel: kullanıcı doğruladı (bounce + denoise örnek sayılarının altında).
- Durağan mod varsayılan 0 olduğu için kullanıcıda devreye girmedi (tasarım gereği, ama
  yanlış varsayılan → B3d 64). 64 ile: örnek 2–63 her dispatch static_active, 64'te durur.
- ★ Kalite eğrisi (yakınsamış 128 spp ham kareye ortalama mutlak fark, 8-bit; iki koşu ort.):

  | örnek | 1 | 2 | 4 | 8 | 16 | 32 | 48 | 64 |
  |---|---|---|---|---|---|---|---|---|
  | static 0 | 26.7 | 33.6 | 30.2 | 30.6 | 23.1 | 16.6 | 13.3 | 11.1 |
  | static 64 | 26.6 | 24.4 | **14.1** | **12.9** | 11.9 | 11.4 | 11.1 | 11.1 |

  8 filtreli örnek ≈ 48 ham örnek. Maliyet: limit altında ~12.5 örnek/sn, üstünde ~14.4 (~%13).
- ★ FAIL sızıntı: limit 128 = hedef 128 → yakınsamış ekran meanabs 0.84 / max 27 farklı
  (256 de aynı: 128'e kırpılıyor); limit 100 → bit bit aynı. Kök: asenkron ping-pong ekranda
  bir önceki submit'i gösterir ve birikim bitince son submit tüketilmez → ekrandaki kare
  127. örnek, o da hâlâ filtreli. Düzeltme B3d (limit ≤ hedef − 1).
- Koyu tekil noktalar (hasır delikleri, pervaz, avize kolu) kamera piksel izinden sonra da
  SÜRÜYOR → tahmin yanlış/eksik; B3d'de geçmiş debug görünümüyle ölçülecek.

## B3c — izole piksel geçmişi (2026-10-26, derlenmedi; yalnız realtime_svgf + C++)

B3b canlı testinde parlak kenar hattı gitti ama ince/izole özelliklerde (hasır
delikleri, pervaz kenarı, avize kolu) **koyu tekil noktalar** kaldı. Kök: piksel izi
yalnız aynı malzemeli komşudan ölçülüyordu; komşusu olmayan pikselde 0 → tutarlılık
testi reddediyor, à-trous malzeme sınırını geçmiyor → piksel 1 spp kalıyor (parlakları
ateş böceği kırpması alıyor, koyular kalıyor). Düzeltme: aynı malzemeli komşu yoksa iz
= kamera uzaklığı × piksel açısı / max(|cos|, 0.2). `VulkanMotionHistory::cameraFootprint`
(prepareDispatch'e verilen vektörlerden), SVGF push 32 → **48 bayt** (`camera` vec4).

1. Rendered açılıyor, `realtime_denoiser_stats` available true (push ABI tuttu).
2. **★ Koyu noktalar:** iç oda, hareket sırasında avize/pervaz/hasır kapak kırpıntısı
   aynı pozdaki yakınsamış kareyle: tekil koyu noktalar büyük ölçüde gitmeli.
   history_length_mean B3b'deki 20.3'ten yükselmeli.
3. **★ Sessiz arıza — hayalet:** izole pikseller artık geniş toleransla geçmiş kabul
   ediyor; hızlı 3° dönüşte ince özelliklerin arkasında iz/çift görüntü olmamalı
   (iz ≤ 4 kare). Varsa fallback çarpanı (0.2 alt sınır / 3 iz) fazla gevşek.
4. Parlaklık B3b ile aynı kalmalı (hareket 0.482 / yakınsamış 0.484).
5. Bulutlu sahne siluet noktaları (B3b madde 5, henüz koşulmadı).

**B3c'ye eklenenler (aynı derleme):**
- **Durağan mod** `static_samples` (0 = kapalı, varsayılan 0; 0..1024): kamera duruyor ve
  viewport OIDN KAPALI iken SVGF yalnız uzaysal kısmını birikmiş görüntü üzerinde
  çalıştırır (temporal yok: birikim zaten yansız temporal filtre), varyans = birikimin
  uzaysal yayılımı (yakınsadıkça küçülür), görüntü `örnek / limit` ile ham birikime söner.
  Limit hedef örnek sayısıyla kırpılır → yakınsamış kare hiçbir zaman filtrelenmez.
  OIDN açıkken devreye girmez (OIDN durağan kareye sahip). Push 48 → **64 bayt**
  (`sampleCount`). IPC/Python `static_samples`, istatistikte `static_active`. Panel:
  "SVGF On Still Frames" (0..256).
- **Panel düzeni:** iki render panelinde de Light Paths (bounce + RR + hareket profili +
  Motion Vectors) ve Denoise (SVGF, sonra OIDN + takvim) bölümleri örnek sayılarının
  hemen altında (`DrawBounceControls`, `DrawDenoiseControls`, ortak). Eski "Light
  Transport & Denoise" / "Light Paths" bölümleri "Light Transport" oldu (kostik, ışık
  şaftları, debug görünümü); ayrı "Denoising" bölümü kaldırıldı. OIDN kutusu
  "Enable Denoiser (OIDN)".

6. **Panel.** İki panelde de Sampling bölümünde Max Samples'ın altında Light Paths ve
   Denoise; Light Transport'ta bounce/denoiser tekrarı YOK; OIDN kapalıyken SVGF
   ayarları görünür. (Panel IPC'den sürülemez — kullanıcı gözü.)
7. **API.** Probe script `static_samples` geçersizlerini (−1, 2000, 1.5) reddetmeli,
   rejections 15.
8. **★ Durağan mod kalite eğrisi (iç oda, OIDN KAPALI).** `static_samples` 0 ve 64 ile
   kamerayı 0.0001 kıpırdatıp örnek 1, 2, 4, 8, 16, 32, 64'te ekran görüntüsü; her
   birinin 128 örnekli ham yakınsamış kareye ortalama mutlak farkı. Beklenen: SVGF'li
   eğri düşük örnekte belirgin altta, 64'te ikisi eşit (fade = 1). `static_active`
   1..63'te true, 64+ false. Kesişme noktası (ham birikimin SVGF'yi yakaladığı örnek)
   varsayılan limit adayıdır — sayıyı yaz.
9. **★ Sessiz arıza — durağan mod yakınsamış kareye sızıyor:** `static_samples` 256,
   max samples 128: 128'de görüntü ham kareyle bit bit aynı olmalı (limit hedefe
   kırpılıyor). Değilse son kare filtreli kalıyor ve kimse fark etmez.
10. **Maliyet.** Durağan karede örnek/sn `static_samples` 0 vs 64 (SVGF ~10 ms/örnek,
    RT ~95 ms → ~%10 beklenir), yalnız limit altında.

## B3b SONUÇLARI (2026-10-26, canlı, iç oda) — 6 PASS, 1 kısmi, 2 koşulmadı

- (1) PASS varsayılan açık: yeni başlatmada IPC yazmadan mode svgf, firefly_clip 3.
- (2) PASS probe: api_validation PASS, 12 ret; `variance_clip` "unknown field" ile reddedildi.
- (3) PASS (API yolu): SVGF açıkken `set_motion enabled=false` reddedildi, motion açık
  kaldı. Panel tıklaması IPC'den sürülemez — kullanıcı gözüyle doğrulanacak.
- (4) ★ PASS parlaklık (profil kapalı): hareket SVGF **0.482** vs yakınsamış OIDN 0.484
  (**−%0.5**; B3: −%11). firefly_clip 0.5/3/10 → 0.478/0.482/0.484 (etkisiz),
  à-trous 1 iterasyon 0.479. Gürültü 0.031 (B3 0.030), OIDN her kare 0.012.
- (5) KISMİ siluet: iç odada parlak kenar hattı yok; ince/izole özelliklerde koyu
  tekil noktalar kaldı (kullanıcı da gördü) → B3c. Bulutlu sahne koşulmadı.
- (6) Hayalet: koşulmadı (B3c ile birlikte).
- (7) Maliyet: total 10.3 ms (temporal 2.37, à-trous 6.82) — temporal'ın 9 tap +
  kayıt okuması ölçülebilir maliyet eklemedi; history_length_mean 20.3.
- (8) PASS referans: 128 örnekte on/off/on bit bit aynı.

## B3b — SVGF geçmiş düzeltmesi + panel + varsayılan açık (2026-10-26, derlenmedi)

**Derleme:** yalnız `realtime_svgf.comp` (iki varyant: `realtime_svgf.spv` +
`realtime_svgf32.spv`, compile_shaders.bat ikisini üretir) + C++. **Raygen
DEĞİŞMEDİ → RT pipeline yeniden derlenmez.** ABI: SVGF descriptor 13 → 15 binding
(13/14 = raygen motionHistory0/1, rgba32ui), push yine 32 bayt (alan adları
`historyFlags`, `fireflyClip`). IPC/Python alanı `variance_clip` → **`firefly_clip`**
(anlamı değişti: artık geçmişi değil yalnız geçmişsiz pikseli kırpar); eski ad
"unknown field" ile reddedilir. Descriptor üretildi, audit OK.

Ne değişti:
- Temporal geçiş geçmişi artık KIRPMIYOR (B3: −%11 karartmanın kökü).
- Geçmiş araması: önce 2×2 bilineer ayak izi, her tap için tutarlılık (aynı
  malzeme, normal > 0.8, teğet düzlem mesafesi < 0.5 piksel izi, uzaklık < 3 iz;
  piksel izi = aynı malzemeli komşu kayıtların en kısa adımı); hiçbiri tutmazsa
  3×3 arama. Raygen'in kendi testinin kabul ettiği tap her zaman geçerli
  (hareketli rijit nesneler bugünkü davranışı korur).
- Geçmişi < 4 olan pikselin O ANKİ örneği ateş böceği kırpmasından geçer
  (aynı malzemeli 3×3 komşu, merkez hariç, ortalama + k·σ, k = firefly_clip 3).
- Varsayılan AÇIK (`RealtimeDenoiserSettings::enabled = true`; proje dosyasına yazılmaz).
- Panel: "Denoise While Moving (SVGF)" + ayarları Light Paths maliyet bloğunda,
  Motion Vectors altında; OIDN anahtarına bağlı değil. "Motion Vectors" kutusu
  artık `rtapi::setMotion` ile yazıyor — SVGF açıkken kapatma hata mesajıyla reddedilir.

Kontrol (sıra: bağımsız ve hızlı önce):
1. **Varsayılan açık.** Uygulamayı yeni başlat, sahne aç, hiçbir IPC yazmadan
   `render.realtime_denoiser_stats {synchronize:false}`: mode `svgf`, firefly_clip 3,
   available true (ilk hareketten sonra). Panelde kontrol viewport OIDN KAPALIYKEN
   de görünüyor. mode off → globals varsayılanı tutmadı.
2. **API.** `python scripts/test/rt_probe_realtime_denoiser_ipc.py` → api_validation
   PASS, 12 ret. Ek: `render.set_realtime_denoiser {variance_clip:3}` reddedilmeli.
3. **Panel bağımlılığı.** SVGF açıkken panelde Motion Vectors'ı kapat → kutunun
   altında "disable SVGF before disabling the motion AOV", `motion_stats.enabled`
   true kalır. Sessizce kapanıyorsa panel API'yi atlıyor.
4. **★ Parlaklık (iç oda, hareket profili KAPALI).** B3 yöntemi: hareket
   sırasında ekran ortalama luma vs yakınsamış OIDN (0.491). B3: SVGF 0.435
   (−%11). Hedef: **yakınsamışın %3 içinde**, gürültü ≤ 0.035 (B3 0.030).
   Hâlâ −%5'ten koyu → kalan kayıp tonemap Jensen'i ya da à-trous; iterasyon
   1 vs 4 ile ayır (B3'te à-trous etkisizdi).
5. **★ Siluet noktaları (bulutlu dış sahne).** Hareketli SVGF karesinde dağ sırtı
   ve uzak arazi kırpıntısı: B3'teki beyaz nokta hattı kaybolmalı. Debug 17'deki
   kırmızı piksellerle aynı yerlerde kalan nokta varsa 3×3 arama tutmuyor
   (piksel izi 0 mı: ince/izole özellik).
6. **★ Sessiz arıza — yanlış kabul edilen geçmiş:** kenarlarda iz/hayalet (hareket
   bulanıklığı gibi görünür, kimse hata demez). İç odada nesneyi +0.05 y taşıyıp
   ve hızlı 3° dönüşte arkada kalan kenarda iz 4 kareden kısa olmalı. Uzun iz →
   tutarlılık eşiği gevşek (düzlem 0.5 iz / uzaklık 3 iz).
7. **Maliyet.** `realtime_denoiser_stats {synchronize:true}` hareket medyanı:
   B3 total 10.1 (temporal 2.4, à-trous 6.7). Temporal 9 tap + kayıt okuması
   ekledi; ~3–4 ms beklenir. history_length_mean B3 20.6'dan yüksek olmalı
   (siluetler artık geçmiş biriktiriyor).
8. **Referans.** 128 örnekte SVGF on/off ham kare bit bit aynı (B3 PASS) — değişmemeli.
9. **Temizlik.** Kamera/nesne/ayarlar başlangıca; görüntüler silinir.

## B3 SONUÇLARI (2026-10-26, canlı, iç oda 1680×945, RTX 3060) — 8 PASS, 1 FAIL (parlaklık), 3 ölçülmedi

- (1) PASS raster kamera hızı: 1° dönüş dx **+28.47** (RT 28.46), dy −0.01, max 34.6,
  ret %2.2 (≈ giren şerit %1.7 + kenar).
- (2) PASS TAA boşta birikiyor: 2 → 4 → 6 → 8 (0.7 sn arayla), artık 1'de kalmıyor.
  Yavaş (~3 kare/sn boşta) — not, hata değil.
- (3) PASS raster nesne hızı: +0.05 y → dy +0.049, max 6.26 (RT 6.3), durunca 0.001;
  raster_objects_available true; Solid'de raster_available false + doğru açıklama.
  Çoklu instance / culling açık tekrarı YAPILMADI.
- (4) PASS `rt_probe_realtime_denoiser_ipc.py`: api_validation PASS, 12 ret,
  motion bağımlılığı reddi doğru.
- (5) PASS politika: SVGF açıkken 4 sn harekette SVGF +70 dispatch, viewport OIDN
  2 çağrı (ilk kare); kapalıyken OIDN 51 çağrı. moving/presenting true.
  ★ SVGF viewport OIDN anahtarından BAĞIMSIZ çalışıyor (OIDN kapalı: +57 dispatch).
- (6) ÜSTÜNDE: total medyan **10.1 ms** (hedef ≤6): à-trous 6.7 (4 iter), temporal
  2.4, variance 0.6. OIDN fast 44 ms'ye göre 4.3× ucuz. history_length_mean 20.6.
- (7) Gürültü (ekran JPEG, komşu piksel farkı — P0 metriğiyle AYNI DEĞİL, yalnız
  göreli): hareket ham 0.19–0.25, SVGF 0.030, OIDN her kare 0.012, yakınsamış 0.014.
- (8) Hayalet/ayrılma: ÖLÇÜLMEDİ.
- (9) PASS referans: 128 örnekte SVGF on/off/on ham kareleri **bit bit aynı**
  (meanabs 0, max 0) — filtre birikime sızmıyor.
- (10) Hacim sahnesi + final: ÖLÇÜLMEDİ (sahne açılmadı).
- (11) B2 hız maliyeti tekrar: +4.0 / +0.6 ms → ortalama ~2 ms, gürültü bandında.
- ★★ **FAIL — hareket halinde karanlık (kullanıcı fark etti: "durunca OIDN gelince
  sahne birden aydınlanıyor").** Ekran ortalama parlaklığı (yakınsamış OIDN 0.491):
  SVGF + hareket profili 0.379 (**−23%**), SVGF profil kapalı 0.435 (**−11%**),
  OIDN her kare 0.489 (0). İki ayrı neden:
  a. Hareket profili 2 bounce → iç mekânda dolaylı ışık kesiliyor (~12 puan).
     Fiziksel, SVGF'nin hatası değil; profil bir hız/kalite takası.
  b. SVGF temporal geçmiş KIRPMASI enerji kaybettiriyor: variance_clip 0.5 →
     0.412, 3 → 0.437, 10 → 0.461; à-trous 1..5 iterasyon parlaklığı DEĞİŞTİRMİYOR
     (0.433–0.437). 1 spp yol izleme örnekleri çarpık dağılır; 3×3 komşuluğun
     ortalama ± kσ penceresi çoğu pikselde gerçek ortalamanın ALTINDA kalır ve
     doğru geçmişi aşağı kırpar. Kırpma TAA'dan gelme; Schied 2017 SVGF geçmişi
     kırpmaz (yalnız geometri testiyle reddeder). Kalan ~−6% (clip 10) muhtemelen
     tonemap Jensen'i (artık gürültü konkav eğride karartır) — düzeltmeden sonra ölç.
- ★ UI: SVGF kontrolü `DrawDenoiserScheduleControl` içinde, yani yalnız viewport
  OIDN açıkken görünüyor; ama SVGF ondan bağımsız çalışıyor → OIDN kapalıyken SVGF
  açık ve düğmesi GİZLİ. Panel çekirdeği göstermiyor (CLAUDE.md "panel yalan").

- ★★ **İKİNCİ BULGU (bulutlu dış sahne, kullanıcı fark etti: "obje sınırlarında ışıklı
  hat"):** hareketli SVGF karesinde siluetler (dağ sırtı, yatık uzak arazi) boyunca
  beyaz noktalar; köprü ve yakınsamış OIDN karesinde yok. Debug view 17 ile birebir
  çakışıyor: o pikseller HER KARE geçmiş reddi alıyor. Kök: B2 geçmiş testi tek
  pikselli ve birincil ışın rastgele alt-piksel jitter'lı → siluette vuruş her kare
  nesne/arka plan arasında değişir, malzeme testi reddeder, piksel hiç geçmiş
  biriktirmez (1 spp kalır). À-trous luminans kenar durdurucusu parlak aykırı merkezi
  komşulardan ayırdığı için bu 1 spp pikseller PARLAK nokta olarak kalır (çarpık
  dağılım: karanlıklar ortalanır, parlaklar korunur). Düzeltme yönü (B3b): SVGF
  makalesindeki gibi 2×2 bilineer ayak izinde tap başına tutarlılık + 3×3 arama
  yedeği (biri tutarsa kabul), reddedilen/kısa geçmişli piksellere yalnız ORADA
  ateş böceği kırpması. B2 RT ret oranı bu sahnede durgun %0.7.
- **Bulutlu sahne hız:** SVGF devrede (12 sürükleme karesinde SVGF 11, OIDN 2–3),
  ama kare süresi aynı (~600 ms/iter her iki modda): hareketin her ilk örneği RT
  ~470 ms (durgunda 160–330 ms'ye iner), OIDN eşzamanlı çalışıp RT'nin gölgesinde
  kalıyor (duvar 560–760 ms, GPU paylaşımı). Bu sahnede denoiser seçimi FPS'i
  değiştirmez; kaldıraç RT (ilk örnek maliyeti, B4 ölçek). Bulutlar gökyüzü yolunda
  (reactive 0); timeline oynatımı SVGF'yi tetiklemiyor (yalnız kamera + TLAS seri).

**Sonraki küçük parti (B3b) önerisi:** (1) geçmiş kırpmasını kaldır ya da
yalnız reactive/ayrılma bölgesinde uygula; kabul: hareket parlaklığı profil kapalı
yakınsamışın %3 içinde, gürültü ≤ 0.035. (1b) siluet geçmişi: 2×2 ayak izi + 3×3 arama, ret piksellerinde ateş böceği kırpması.
(2) SVGF kontrolünü OIDN bloğundan
çıkar, "Fewer Bounces While Moving" yanına koy. (3) Ardından varsayılan AÇIK
kararı (aşağıda). (4) à-trous 6.7 ms: fp16 ara görüntü / paylaşılan bellek — ölç.

**Varsayılan açık kararı (kullanıcı sordu):** sayılar destekliyor — hareket
karesi OIDN'in ~¼ maliyeti, referans bit bit korunuyor, durgun karede hiçbir şey
değişmiyor. ŞART: (1) parlaklık düzeltmesi, (2) UI yerleşimi. Hareket profili
varsayılan kapalı kalmalı (açılırsa karartma beklenen takas, yardım metninde yazmalı).

## B3 — tek derleme, ardından tek kabul turu (2026-10-26)

Önce kullanıcı `RayTrophiStudio/compile_shaders.bat` ile **tüm shader'ları**,
sonra aynı partide C++ derler. Ajan derleme çalıştırmaz / uygulamayı başlatmaz.
Yeni shader: `realtime_svgf.comp` → `realtime_svgf.spv` (HDR16),
`realtime_svgf32.spv` (`RAW32`, HDR32); sürücü ikisini de üretir.
Değişenler: `material_preview.vert`, `material_preview_frag.frag` ve covered
varyantı, `raster_cull.comp`, `raster_taa.comp`.

ABI: raster HDR 4 → 5 attachment (4 renk + derinlik), material vertex 6 binding /
12 attribute; cull 8 → 10 storage binding; TAA 5 → 6 binding, push yine **80 bayt**.
SVGF push **32 bayt**, 12 storage image + 1 storage buffer. Yeni `.cpp` dosyaları
vcxproj ve filters'a kayıtlı. OptiX değişmedi.

**Kaynak son kontrolü (2026-10-26):** IPC capability audit PASS (707 metot),
üretici `--check` PASS; proje/filter XML ve 7 yeni modül kaydı, probe AST +
Release kopyası, shader binding/push kaynak incelemesi ve `git diff --check`
PASS. Bunlar derleme veya canlı GPU kabul testi değildir.

**Başlangıç:** iç oda, 1680×945, sabit pozlama, aynı kamera ve bounce ayarları.
SVGF varsayılan **off**; 4 iterasyon / 24 geçmiş / variance_clip 3. Ayarlar
B2 hız denetimleri gibi oturumluk, proje dosyasına yazılmaz.
IPC dış terminalde `scripts/test/rt_ipc.py` üzerinden; gömülü script alanından
çağrılmaz. Named pipe error 5 olursa probe aynı transportla yükseltilmiş çalışır.

1. **İLK: B2 raster kamera düzeltmesi.** Material/RayFusion, TAA açık,
   bir kare geçmiş hazırken 1° sağa dönüş ve hemen
   `render.motion_stats {sample:'raster'}`: mean_dx ≈ **+28 px**, dy ≈ 0.
   Ret: giren şerit + kenarlar/disocclusion. Hâlâ %100 ret / sıfır hız:
   eski SPV, `historyFlags` ayrımı veya önceki kamera hattı bozuk.
   B3 derinlik reddi yeni: eski beklenti olan yalnız giren şeride ek ret
   olabilir; nesne kenarlarıyla ilişkili olduğunu sayısal olarak not et.
2. **Boşta TAA 16'ya gidiyor.** Kamerayı bırak, `viewport.taa` raporunu
   birkaç saniye aralıkla oku: accumulated_samples 1 → target_samples (16),
   converged true; ardından CPU/GPU gereksiz çizimle meşgul kalmamalı.
   1'de kalırsa dış `needsViewportRender()` kapısı; her kare 1 olursa
   yakınsama isteği yanlışlıkla sahne dirty'sine çevrilmiş demektir.
3. **Raster nesne hızı + culling eşleşmesi.** Material modunda sabit kamerada
   bir nesneyi +0.05 y taşı ve hemen raster motion oku: nesnede dy pozitif,
   arka plan ~0; durunca hız ~0. `raster_objects_available` true olmalı.
   GPU culling açıkken birden fazla instance/proxy ile tekrarla: başka
   nesnede vektör oluşması önceki matrislerin compaction sırasının bozulmasıdır.
   Görünürlük/topoloji değişiminde ilk yeni nesne kareleri geçmişi reddetmeli.
   Material per-mesh fallback kamera hızı taşır; `raster_objects_available=false`
   bunu bildirir. Solid/Matcap mevcut tasarımda TAA geçişini çalıştırmaz;
   raster_available=false ve açıklama beklenir, eski Material AOV okunmamalı.
   Material → Solid geçişinde TAA yüzünden sonsuz yeniden çizim olmamalı.
4. **Yüzey ve atomik doğrulama.** Dış terminalde
   `python scripts/test/rt_probe_realtime_denoiser_ipc.py` çalıştır.
   12 geçersiz ayar reddi, kısmi yazım yok, açık SVGF'nin motion bağımlılığı
   korunuyor ve başlangıç ayarları geri yükleniyor → API validation PASS.
   Python eş yüzeyleri: `rt.render.set_realtime_denoiser(...)`,
   `rt.render.realtime_denoiser_stats(synchronize=False)`.
   Embedded Python yalnız ayar/okuma için; kare bekleyen kabul testi IPC'den.
5. **SVGF hazır ve politika doğru.** `render.set_motion {enabled:true}`;
   `render.set_realtime_denoiser {mode:'svgf'}`. Rendered'da kamera hareketi:
   available/moving/presenting true, dispatches artıyor; viewport OIDN calls
   hareket süresince artmamalı (ilk kaynak kurulumu/slot gecikmesi sınırı hariç).
   Mode='off' anında ham/OIDN yoluna dönmeli. available=false ise reason'ı oku;
   off ve debug/final yolda SVGF dispatches sabit kalmalı. **Sessiz arıza:**
   mode='svgf' görünürken dispatches artmıyor; bunu performans kazanımı sayma.
6. **GPU maliyeti.** Hareketten sonra `render.realtime_denoiser_stats
   {synchronize:true}`: gpu_timing_available true, temporal_ms, variance_ms,
   atrous_ms ve total_ms son tamamlanan GPU sorgularından gelir. CPU kayıt
   süresi değildir. history_length_mean 1..history_max aralığında olmalı.
   total_ms **≤6 ms** hedefi; toplam backup/compose/tonemap/restore'u da
   içerir (geçiş sürelerinin toplamından büyük olması beklenir). İlk shader
   kurulumu hariç, birkaç hareket karesinin medyanını yaz. Aşılıyorsa hangi
   geçişin baskın olduğunu raporla; partinin içine yeni optimizasyon sokma.
7. **Gürültü.** Aynı pozlama/kamera hareketi dizisinde ham 1 spp, OIDN fast,
   SVGF için komşu piksel farkını ölç. P0: ham **0.065**, OIDN **0.012**;
   SVGF OIDN'e yakın olmalı. Sadece FPS artışını kalite PASS sayma.
   Doğrulama görüntüleri `docs/dev/.captures/` altında geçici; sayıları yazıp sil.
8. **Hayalet ve ayrılma.** Hareketli nesne arkası, yeni açılan duvar,
   ekran kenarından giren yüzey: iz **4 kareden kısa** olmalı. Nesne durunca
   vektörün sıfırlanması ve yeni bölgede history length düşmesi birlikte
   doğrulansın. Kamera ileri/geri ve saf dönüş ikisini de kullan.
9. **Hareketten durgunluğa geçiş + referans.** Durunca ilk 8 örnekte SVGF
   görüntü köprüsü ham birikime söner, OIDN B1 takvimine döner. SVGF geçmişi
   Welford tahminine yazılmaz. 128 örnekte mode off/on ham HDR ve son OIDN
   görüntüsü aynı olmalı (sabit pozlama, aynı seed). Görünmez ama ağır hata:
   filtrelenmiş görüntünün progressive/OIDN girdisine sızması; 128 örnek
   karşılaştırması bunu yakalar. Ayar değiştirme, viewport resize ve backend
   değişiminde eski çözünürlük/epoch görüntüsü görünmemeli.
10. **Hacim reactive + final.** Nükleer/gaz sahnesinde hareket: reactive
    bölgede geçmiş ağırlığı azalır, yüksek reactive'de à-trous atlanır;
    hacim detayları yüzey normaliyle yıkanmamalı. Sonra final render:
    SVGF dispatches değişmez, OIDN mevcut yüksek kalite yolunda çalışır.
11. **B2 maliyet takibi.** SVGF off iken `render.set_motion` açık/kapalı,
    aynı oda/ayar ve `render.set_gpu_timing {enabled:true}` ile ayrı pencere:
    RT ms/örnek karşılaştır. B2 farkı ~3.5 ms (%3.5). SVGF on için RT ve
    SVGF maliyetlerini ayrı yaz. Geçmiş yazımı / önceki konum / register
    paylaştırması henüz GPU profiler ile ölçülmedi; tahmini bulgu diye yazma.
12. **Temizlik ve tek rapor.** Kamera/nesne dönüşümü/denoiser/motion/debug
    ayarlarını başlangıca al; kullanıcı projesini kaydetme. Sayısal B3
    sonuçlarını buranın en üstüne tek seferde yaz; yeni bulgular plan §7'ye.
    Geçici doğrulama ekran görüntülerini sil. B1'in RR gürültü ve geçici
    proje roundtrip işleri ayrı Sonra maddeleri olarak kalır.

## B2 — denoiser kopya etiketi + hız AOV + RayFusion hızı + jitter (2026-10-26) — CANLI 11/12, raster düzeltmesi derleme bekliyor

**Sonuçlar (iç oda, 1680×945, RTX 3060):**
- (1) pipeline ready (ilk derleme 213 sn), görüntü normal.
- (2) ★ converged_only eski-kare hatası DÜZELDİ: 0.25 birim kaydırıp yakınsatınca
  denoise karesi yeni ham kareyle faz korelasyonu 0 px (fark 9.9), eski pozla
  fark 23.7 (B1: 6 px kayma).
- (3) every_sample 8.61 örnek/sn, 128 çağrı, 44.7 ms/çağrı — B1 ile aynı;
  senkron kopya yolu her çağrıda tetiklenmiyor.
- (4) render cihazı: rt_available/enabled/history_valid true, dispatch artıyor.
- (5) Duran kamera mean 0.0001 px. **rejected %4.1** (beklenti <%1 yanlıştı):
  debug view 17'de yalnız geometri kenarları (hasır kapaklar, avize, kitaplar) —
  rastgele alt-piksel jitter kenarda birincil vuruşu başka örneğe düşürüyor,
  doğru davranış.
- (6) Saf 1° sağa dönüş: mean_dx **+28.5 px** (işaret doğru), dy −0.01, max 34.6
  (merkez 25.9 × sec² kenarı ✓).
- (7) Dönüşte rejected %6.4 (≈ %4.1 kenar + %1.7 giren şerit), durunca %4.2.
- (8) Nesne +0.05 y: dy +0.044 ortalama, max 6.3 px, object_motion_dispatches her
  taşımada +1.
- (9) Debug 16 düz gri, 17 koyu mavi + kırmızı kenarlar.
- (10) ★ **FAIL — raster:** material modunda `sample='raster'` hep %100 rejected,
  hareket 0. Kök: TAA kamera hareketinde renk geçmişini sıfırlıyor
  (`m_taaFrameIndex = 0`) ve hız yazımı aynı `historyValid` bayrağına
  bağlıydı → hareketli her kare "geçmiş yok". Düzeltme yazıldı: push alanı
  `historyFlags` (bit 0 renk geçmişi, bit 1 önceki kamera), hız bit 1'e bağlı.
  Ayrıca: raster viewport boştayken yeniden çizmiyor gibi
  (`viewport.taa accumulated_samples` 1'de kaldı) — B3'te bak.
- (11) Halton: jitter 8 fazlı (0.125,0.278) (0.375,0.056) (0,−0.167) (0.25,−0.389) …,
  duran kamerada hız yine 0.0001.
- (12) Maliyet: hız açık 94.5 / 99.5 ms, kapalı 91.4 / 95.3 ms → **+~3.5 ms (%3.5)**,
  eşik %2'nin üstünde. Kapalıyken de kod shader'da (yalnız bayrakla atlanıyor),
  B1 tabanının (96–100) üstüne çıkmadı. B3'te ölç: hangi kısım (geçmiş
  yazımı / önceki konum tersi / canlı register).

**Raster düzeltmesinin kontrolü (B3 derlemesinde):** material modunda 1° dönüş →
`sample='raster'` mean_dx ≈ +28 px, rejected ≈ giren şerit; hareketten sonra
ilk karede w=0.

Ayrıntılı madde listesi (B2 derlemesi için yazılan):


Plan: RT_REALTIME_DENOISE_UPSCALE_PLAN.md §4. **Önce `compile_shaders` — TÜMÜ:**
`rt_payload.glsl` değişti (payload'a `primaryInst`), yani onu içeren her RT
shader'ı (raygen, closesthit, hair/volume closesthit, miss, photon …) ve
`raster_taa.comp`. RT pipeline önbelleği geçersizleşir: ilk Rendered açılış
~3 dk derler (`render.rt_pipeline_status` "compiling" → "ready"), bu hata değil.
Push constant 200 → 212 bayt; RT descriptor seti 38 → 41 binding; raster TAA
seti 4 → 5. Yeni dosyalar: Backend/VulkanMotionHistory.*, Backend/VulkanMotionAov.cpp,
Api/RtApiMotion.* (vcxproj + filters'a eklendi). Yeni IPC: `render.motion_stats`,
`render.set_motion`; `debug_view` 16 (Motion Vectors), 17 (Motion History).
Panel: Light Paths altında "Motion Vectors" + "Halton Jitter". Audit OK.

Sözleşme (RT ve raster aynı): xy = noktanın önceki karedeki izdüşümü − şimdiki
izdüşümü (piksel, y aşağı), z = reactive, w = 1 geçmiş reddedildi.
Raster: yalnız KAMERA hareketi, yalnız viewport TAA açıkken.

1. **Shader ABI / görüntü aynı.** Rendered açılıyor, pipeline "ready", güzel
   görüntü B1 ile aynı (hız AOV ışıltıyı değiştirmez). Siyah ekran / pipeline
   hatası → eski .spv (payload veya push constant ayrışık; static_assert 212).
2. **★ Denoiser eski kare düzeltmesi.** B1'deki test aynen: denoiser açık,
   `schedule=converged_only`, kamerayı 0.25 birim kaydır, yakınsat, ekran
   görüntüsü al; denoiser'ı kapat, ham kareyi al → faz korelasyonu 0 px
   (B1'de 6 px idi). Hâlâ kaymışsa etiket karşılaştırması tutmuyor.
3. **every_sample hızı düşmedi.** every_sample'da 128 örneğe örnek/sn ≈ 8.6
   (B1 ölçümü). Belirgin düşüş → her çağrı senkron kopya yoluna giriyor
   (etiket yanlış: `m_currentSamples - m_denoiserCopySamples` hep > 1).
4. **motion_stats temel.** `render.motion_stats`: render cihazı satırında
   `rt_available` true, `rt_enabled` true, `dispatches` artıyor,
   `rt_history_valid` true. rt_available false → görüntüler ayrılmadı (log'da
   "Motion AOV allocation failed").
5. **Duran kamera = 0.** DURAN kamerada `render.motion_stats @{ sample='rt' }`:
   `mean_motion_px` < 0.05, `rejected_fraction` küçük (< %1; ince kenarlar).
   ★ Sessiz arıza: mean_motion ~0.3–0.7 px → jitter sadeleşmiyor; "hafif
   bulanık" diye görülür, kimse bug demez.
6. **★ İşaret ve ölçek: saf dönüş.** Kamerayı yalnız HEDEFİ kaydırarak sağa
   küçük bir açıyla (θ ≈ 1°) döndür ve HEMEN ardından `sample='rt'`.
   Beklenen: `mean_dx` POZİTİF (içerik sola kayar, nokta önceden sağdaydı) ve
   ≈ W·θ/fov_x (1680 px, fov_y 35.3°, en-boy 1.778 → fov_x ≈ 58.3° → 1° ≈
   28.8 px), `mean_dy` ≈ 0. Gökyüzü de aynı kaymayı verir (dönüş derinlikten
   bağımsız). İşaret ters → sözleşme ters; ölçek %10'dan fazla sapıyor →
   izdüşüm ya da genişlik yanlış. Örnek bir sonraki dispatch'e kaldıysa
   mean_motion 0 çıkar — tekrarla.
7. **Ayrılma.** Aynı hareket sırasında `rejected_fraction` ≈ ekrana yeni giren
   kenar şeridi (≈ |mean_dx|/W) + kenarlar; kamera durunca bir dispatch içinde
   ~0'a döner. Hep ~1 → geçmiş slotları ters (okuma = yazma).
8. **Nesne hareketi.** Bir nesneyi IPC transform ile kaydır: render cihazında
   `object_motion_dispatches` artmalı; Debug View 16'da yalnız nesne renkli,
   arka plan gri. Artmıyor → TLAS önceki dizi yakalanmıyor (CPU yolu: aynı
   sayı + aynı yol şartı; GPU yolu: kopya).
9. **Debug görünümleri.** `render.set_settings @{ debug_view=16 }` duran kamerada
   düz gri; 17 duran kamerada koyu mavi (geçmiş kullanıldı), hacim pikselleri
   yeşilimsi. Sonra `debug_view=0`.
10. **RayFusion / raster.** Material/RayFusion moduna geç, `viewport.taa` açık;
    madde 6'daki dönüşü yap, `sample='raster'`: mean_dx RT ile aynı işaret ve
    ~aynı büyüklük. `raster_available` false ise `raster_reason`.
11. **Halton.** `render.set_motion @{ halton_jitter=$true }` → birikim baştan,
    `motion_stats` jitter 8 fazda değişen (|x|,|y| ≤ 0.5) değerler; madde 5
    yine ~0. Sonra kapat (varsayılan kapalı).
12. **Maliyet.** RT dispatch ms, `render.set_motion @{ enabled=$false }` ile
    karşılaştır (iç oda ~96–100 ms). Fark %2'den büyükse ölç ve not et.

## B1 — OIDN takvimi + hareket bounce profili + RR düğmesi (2026-10-26) — CANLI 8/9, 1 HATA

**Sonuçlar (iç oda, 1680×945, denoiser fast):**
- (1) shader ABI OK, pipeline ready. (2) RR=3 → 100 ms (taban 96, gürültü içi).
- (3) RR 2 → 82.9 ms (−%17), RR 1 → 75.3 ms (−%25); 0 reddedildi.
  Gürültü karşılaştırması YAPILMADI — varsayılan 3 kalıyor.
- (4)(6) Takvim, 128 örneğe kadar: every_sample 8.6 örnek/sn (124 çağrı),
  progressive 10.3/sn (5 çağrı, 119 önbellek), converged_only 10.5/sn
  (1 çağrı). Denoiser kapalıyla aynı hız bandı (+%20 every_sample'a göre).
- (5) 8 kamera hareketi → 18 çağrı: her harekette yeni çağrı var.
- (7) Hareket profili: hareket sırasında active=true, effective 2, authored 10
  (değişmedi), RT 34.7 ms; durunca active=false, effective 10, RT 98 ms.
- (8) Kapalıyken effective = authored. Geçersiz settle_ms reddedildi.
- (9) Proje kaydı denenmedi (kullanıcının projesine yazmamak için).

**★ HATA (kullanıcı fark etti: "converged'de son denoise karesinde kamera
hafif sağa dönmüş gibi"):** ölçüldü — kamera 0.25 birim kaydırılıp
converged_only ile yakınsatıldığında denoise karesi YENİ ham kareye göre
6 px kaymış ve ESKİ konumun ham karesine daha yakın (ortalama fark 10.2 /
16.9); every_sample doğru (fark 10.2, kayma 0). Kök neden:
`VulkanBackendAdapter::getDenoiserFrameGPU` "fence-deferred ping-pong" —
her çağrı BİR ÖNCEKİ çağrıda kopyalanan kareyi döndürür, yenisini arka
planda başlatır. Her örnekte çağrılınca gecikme 1 örnek (görünmez); takvim
çağrıları seyrekleştirince "önceki çağrı" önceki kamera konumu olabilir.
progressive'de etkisi: sıfırlamadan sonraki ilk denoise eski pozu gösterir,
bir sonraki çağrıda düzelir. converged_only'de bir sonraki birikime kadar
kalır. Düzeltme B2 partisinde (plan §4): kopyayı (epoch, örnek) ile etiketle,
eşleşmiyorsa senkron kopya + bekleme. O zamana kadar güvenli seçim
`every_sample`.

Ayrıntılı madde listesi (B1 derlemesi için yazılan):

Plan: RT_REALTIME_DENOISE_UPSCALE_PLAN.md §3. **Önce `compile_shaders` (raygen
değişti: push constant'a `rrStartBounce` eklendi, 196 → 200 bayt), sonra C++.**
Yeni/değişen IPC: `render.set_denoiser schedule`, `render.denoiser_stats`
(`skipped_by_schedule`, `cache_presents`), `render.set_settings rr_start_bounce`,
`render.get_settings` (`effective_*`, `motion_profile_active`),
`render.get_motion_profile` / `render.set_motion_profile`. Panel: Light Paths
altında "Roulette From Bounce" + "Fewer Bounces While Moving"; Denoiser altında
"Viewport Schedule". Üç ayar da projeye kaydediliyor. Audit OK.
Varsayılanlar: schedule = progressive (YENİ davranış), RR = 3 (bugünkü),
hareket profili KAPALI.

Ölçüm tabanı (iç oda, 1680×945): RT 95.7 ms/örnek, denoiser fast açıkken
örnek/sn ~9.8 (her örnek denoise), kapalıyken ~11.2.

1. **Shader ABI.** Rendered mod açılıyor, görüntü bugünküyle aynı, log'da
   push constant / pipeline hatası yok. Siyah ekran ya da pipeline hatası →
   raygen.spv eski (compile_shaders koşmadı) ya da C++ ile ABI ayrışık
   (static_assert 200).
2. **RR varsayılanı birebir.** `render.get_settings` → `rr_start_bounce` 3;
   RT dispatch bugünkü gibi ~96 ms. Farklıysa RR yolu varsayılanı değiştirdi.
3. **RR düğmesi çalışıyor.** `rr_start_bounce=2` → RT ms belirgin düşmeli,
   `samples` sıfırlanmalı (birikim baştan). Düşmüyorsa shader eski alanı okuyor.
   Gürültüyü ölç (komşu piksel farkı, 128 örnekte): 3'e göre ne kadar arttı —
   varsayılan bu sayıya bakılarak değişir, ölçmeden değil. Sonra 3'e geri al.
4. **Takvim: progressive.** Denoiser açık, kamera kıpırdat, birikim bitene kadar
   bekle: `calls` ≈ log2(128)+1 = 8 civarı (birden çok örnek aynı karede
   düşerse daha az), `skipped_by_schedule` ≈ 120, `cache_presents` =
   `skipped_by_schedule`. Örnek/sn ≥ 10.8 (bugün 9.8). Son kare
   `every_sample` ile aynı görünmeli (ikisi de 128'de denoise eder).
   ★ Sessiz arıza: ekran birikim boyunca ESKİ bir denoise karesinde donar ve
   "denoiser çok yumuşak" diye okunur — `cache_presents` artarken `calls` hiç
   artmıyorsa takvim hiç tetiklenmiyor.
5. **★ Hareket halinde eski kare YOK.** Denoiser açık, progressive, kamerayı
   birkaç kez peş peşe kıpırdat (`camera.set_position` adımları): her
   kıpırdatmada `calls` artmalı (yeni epoch → önbellek geçersiz). Artmıyor ve
   `cache_presents` artıyorsa ekran ÖNCEKİ kamera konumunun görüntüsünü
   gösteriyor — en kötü hata, kimse fark etmeden "gecikme" sanır.
6. **converged_only / every_sample.** `converged_only`: birikim boyunca
   `calls` 0, `cache_presents` 0 (ham görüntü), tamamlanınca 1 çağrı.
   `every_sample`: eski davranış, `skipped_by_schedule` 0.
7. **Hareket profili.** `render.set_motion_profile @{ enabled=$true }`
   (2/2/2, 150 ms). Kamerayı ~1 sn boyunca sık kıpırdat ve bu sırada
   `render.get_settings`: `motion_profile_active` true, `effective_max_bounces` 2,
   `max_bounces` 10 (yazılan değer DEĞİŞMEMELİ). RT dispatch ~31 ms.
   Durduktan >150 ms sonra: active false, effective 10, birikim baştan,
   RT ~96 ms. Active hiç true olmuyorsa poz tespiti `scene.camera`'yı değil
   başka kamerayı izliyor.
8. **Profil kapalıyken etkisiz.** enabled=false: kamera hareketinde
   effective = authored, RT ~96 ms. Sonra profili kapalı bırak.
9. **Proje kaydı.** Ayarları değiştir, projeyi kaydet/aç: schedule, RR ve
   profil geri gelmeli; `motion_profile_active` açılışta false.

## Birikim bitmiş kareye bounce ve denoiser düzenlemesi ulaşsın (2026-10-26 parti 2) — CANLI PASS

Sonuç: (1) bounce 9 → 1 sn'de samples 10, aktif; (3) denoiser aç → 1 çağrı, converged 1, samples 128'de kaldı; (5) boşta 5 sn 0 çağrı; (4) komşu piksel farkı kapalı/açık/kapalı 0.065 / 0.012 / 0.065 — kapatınca ham kareye dönüyor. (2) panel sürükleme elle denenmedi.

İkisi de canlı ölçüldü (aşağıda), ikisi de "çekirdek değişti, ekran değişmedi"
sınıfı. Render bloğu `start_render` kapısının arkasında; birikim bitince kimse
kapıyı açmıyordu, açılsa da backend yeni kare basmıyordu.
Değişenler: `rtapi::setRenderBudget` (her bounce yazımı `start_render`),
scene_ui iki "Light Paths" bloğu (DragInt değişince `start_render`),
`IBackend::requestPresentRefresh()` (Vulkan: `m_tonemapRefreshPending`),
Main.cpp'de denoiser imzası tespiti (değişince `start_render` + yenileme).

1. **Bounce, IPC.** Birikim bitmiş (`viewport.status` complete=true), sonra
   `render.set_settings @{ max_bounces=9 }`: 1 sn içinde `samples` küçülmeli,
   `rendering_active` true. Öncesi: 3 sn boyunca 128'de kaldı. Geri 10 yap.
2. **Bounce, panel.** Aynı şey Total Bounces sürükleyerek. Render yeniden
   başlamıyorsa iki panel bloğundan biri atlanmış.
3. **Denoiser AÇ, bitmiş kare.** Denoiser kapalı, birikim bitmiş,
   `render.reset_denoiser_stats`, `render.set_denoiser @{ viewport_enabled=$true }`,
   2 sn: `calls >= 1`, `converged_calls >= 1`, ekran temiz. Öncesi: 0 çağrı.
   Birikim YENİDEN BAŞLAMAMALI (`samples` 128'de kalır) — başlıyorsa yenileme
   yolu yanlışlıkla reset tetikliyor.
4. **★ Denoiser KAPAT, bitmiş kare.** `viewport_enabled=$false`: ekran
   gürültülü ham kareye dönmeli. Sessiz arıza hâli: ekran temiz KALIR ve
   "kapattım, hiçbir şey değişmedi" diye okunur — `requestPresentRefresh`
   g_backend'e ulaşmıyor demektir (g_backend Vulkan adapter değil mi bak).
   `viewport.get_screenshot` ile iki kare karşılaştırılabilir (scratchpad'e).
5. **Boşta maliyet yok.** Birikim bittikten sonra hiçbir şeye dokunmadan 5 sn:
   `denoiser_stats.calls` 0 kalmalı. Artıyorsa imza her karede değişiyor.

## Denoiser IPC'ye açıldı + maliyet sayacı (2026-10-26 parti 1) — CANLI PASS

Yeni: `render.get_denoiser`, `render.set_denoiser`, `render.denoiser_stats`,
`render.reset_denoiser_stats` (+ `rt.render.*` Python), perf `loop.denoise`,
dosyalar `Api/RtApiDenoiser.h/.cpp`. Canlı sonuç (iç oda, 1680×945, RTX 2503):

| Durum | RT dispatch ms/örnek | denoise ms/çağrı | örnek/sn |
|---|---|---|---|
| Denoiser kapalı | 95.7 | — | ~11.2 |
| Açık, fast | 110.7 | 43.0 | ~9.8 |
| Açık, balanced | 119.9 | 63.3 | ~9.0 |

- Yol `gpu` (CUDA sıfır kopya). Denoise RT ile ÖRTÜŞÜYOR: 43 ms'lik çağrı
  örnek hızını yalnız ~%13 düşürüyor, ama aynı GPU'yu paylaştığı için RT
  dispatch'ini +15 ms (fast) / +24 ms (balanced) şişiriyor.
- `repeat_sample_calls` = 0, birikim bitince `calls` = 0: kod okumasından
  çıkan "OIDN boşta her turda koşuyor" şüphesi YANLIŞ çıktı.
- Geçersiz `quality` reddedildi, hiçbir alan değişmedi.

## Gaz march'ı: büyüyen adım söküldü (2026-10-25, derlenmedi; SHADER + C++)

RT_GAS_SECONDARY_MARCH.md §14. Radyans önbelleği canlı doğrulandı (−%53,
geri düşme 0); büyüyen adım örnek azaltmadı ve başlığı kararttı, söküldü.
Değişen shader: volume_closesthit. C++: VulkanBackend(_Volumes), RtApi, IPC,
Python, panel, descriptor. Kaynak: `check_volume_march_lod_contracts.py` → OK.

1. **Açılış değeri.** `render.volume_march_lod` beş alan döner:
   1/4/2/128 ve `secondary_radiance_cache` true. `*_transmittance_growth` yok.
   Panelde büyüme kaydırıcıları yok. Yedi alan dönüyorsa eski exe.
2. **Referans birebir.** 1/1/1/0, önbellek kapalı: önceki referansla aynı
   (nükleer sahne, düzlem açık ~500 ms). Farklıysa geri alma eksik
   (`grep growStride shaders` boş olmalı).
3. **Önbellek hâlâ çalışıyor.** Varsayılanda `secondary_density_samples`
   ≈ 0, `radiance_cache_fallbacks` ≈ 0; başlık referansla aynı tonda (büyüme
   onu −1.8/255 kararttıyordu).

## Gaz: sekme ışınları için radyans önbelleği + geçirgenliğe göre büyüyen adım (2026-10-25, derlenmedi; SHADER + C++)

RT_GAS_SECONDARY_MARCH.md §13. Yeni shader include:
`volume_radiance_cache.glsl` (vcxproj'a eklendi). Değişen shader'lar:
volume_closesthit, volume_light_cache.glsl (kanal 6 → 10),
volume_instrumentation.glsl (+2 sayaç, struct 248 → 256 B). C++: ışık önbelleği
kanal sayısı, VulkanBackend(_Volumes), RtApi, IPC, Python, panel, descriptor.
RT pipeline baştan derlenir. Kaynak kontrolleri:
`check_volume_march_lod_contracts.py` ve `check_volume_light_cache_contracts.py`
→ OK.

1. **Ayar ve ABI.** `render.volume_march_lod` yedi alan döner:
   1/4/2/128, `secondary_radiance_cache` true, büyüme 3/3. `render.volume_stats`
   içinde `radiance_cache_marches` / `radiance_cache_fallbacks` var. Sayaçlar
   saçmaysa shader ile C++ struct'ı hizasız. `render.volume_light_cache`
   `channels: 10` döner.
2. **Referans birebir.** 1/1/1/0, önbellek kapalı, büyüme 1/1: maliyet ve
   görüntü önceki referansla aynı. Farklıysa büyüme ya da önbellek kapalıyken
   de çalışıyor.
3. **Önbellek doluyor.** Varsayılanda, statik karede birkaç örnekten sonra
   `radiance_cache_fallbacks`, `radiance_cache_marches`'ın yanında küçük
   kalmalı; `secondary_density_samples` neredeyse sıfıra inmeli. Geri düşmeler
   yüksek kalıyorsa radyans etiketi her karede değişiyor (ışık durumu hash'i
   kararsız) ya da sahiplik bırakılmıyor.
4. **Işık önbelleği bozulmadı.** `light_cache_hits` / `light_cache_misses`
   eskisi gibi. Sahnede 5'ten fazla ışık varken 6.+ ışığın gölgesi march'tan
   gelmeli. ★ Kanal eşlemesi toplam kanal sayısını izlerse ışık 5–8 radyans
   kanallarına yazar: yanlış gölge, hata yok.
5. **Görüntü.** Nükleer sahne, düzlem açık; referans ile varsayılan aynı
   oturumda, aynı karede:
   - zemindeki dolaylı ateş ışığı;
   - dumanın gölgeli tarafı;
   - yoğun başlığın arkasındaki kenarlar.
   Hafif yumuşama beklenir; zeminin belirgin kararması ya da açılması
   beklenmez.
6. ★ **Sinsi olan: donmuş önbellek.** Işık ya da güneş hareket ettirildiğinde
   sekme ışınlarının gördüğü gaz eski aydınlatmada kalırsa, ışık durumu
   hash'i değişmeyi kaçırıyor demektir. Görüntü inandırıcı görünür ama yanlış.
   Kontrol: güneşi döndür; zemindeki dolaylı ışık birkaç örnek içinde yeni
   yöne uymalı.

## Gaz march'ı: uyarlanan adım söküldü, varsayılan 1/4/2/128 (2026-10-25, derlenmedi; SHADER + C++)

RT_GAS_SECONDARY_MARCH.md §12. Önceki derlemedeki uyarlanan adım canlı
ölçüldü: daha çok örnek aldı ve görüntüyü açtı, söküldü. Değişen shader:
volume_closesthit. C++: VulkanBackend(_Volumes), RtApi, IPC, Python, panel,
descriptor. Kaynak: `python scripts/test/check_volume_march_lod_contracts.py` → OK.

1. **Açılış değeri.** `render.volume_march_lod` dört alan döner:
   1/4/2/128, `adaptive_*` alanı yok. Panelde uyarlanan adım kaydırıcıları
   yok. Yedi alan dönüyorsa eski exe.
2. **Referans birebir.** 1/1/1/0'da maliyet ve görüntü uyarlanan adımdan
   önceki referansla aynı. Fark varsa geri alma eksik
   (`grep segStride shaders` boş olmalı).
3. **Varsayılan.** Nükleer sahne, düzlem açık: referansa göre ~−%30, zemin
   referansla aynı ton (ortalama fark ≈ +1/255). Zemin belirgin açıksa sınır
   hâlâ 32.

## Obje görünürlüğü çekirdekte: `scene.set_visible` / `scene.get_visible` (2026-10-25, derlenmedi)

Outliner gözü artık `rtapi::setObjectVisible` çağırıyor (yeni dosya
`RtApiObjectVisibility.cpp`, vcxproj'a eklendi). IPC `scene.set_visible`
(SceneWrite) + `scene.get_visible` (Read, backend maske okuması dahil),
`scene.object_info.visible`, Python `rt.scene.set_visible/get_visible`.
Test: `python scripts/test/rt_object_visibility_ipc.py --object Plane_1`.

1. **Derleme + `agent.discover`**: `scene.set_visible` ve `scene.get_visible`
   listede, ikisi de documented. Yoksa eski exe ya da üretilmiş tablo bayat
   (`python scripts/audit_ipc_capabilities.py` yeşil olmalı).
2. **Bilinmeyen ad reddedilir**: `scene.set_visible {name:"yok", visible:false}`
   hata döner. Başarı dönüyorsa `forEachEntry` boş eşleşmeyi saymıyor —
   script'ler yanlış adla "gizledim" sanır.
3. **Panel gözü hâlâ çalışıyor** (tek obje, grup, Characters/Models): göz
   kapanınca obje viewport'tan kalkar, açınca geri gelir. Çalışmıyorsa UI →
   rtapi yönlendirmesi kırık (`setHierarchyObjectVisibility` artık yalnızca
   `rtapi::setObjectVisible` çağırıyor; `rtapi::g_ctx` bağlı mı?).
   Characters/Models gözü eskiden yalnız `member->visible` çeviriyordu ve
   backend maskelerine hiç ulaşmıyordu — artık ulaşmalı.
4. **Test script'i PASS** (Plane_1, Vulkan RT açık): `hidden:` satırlarında
   `viewport_raster`, `viewport_rt`, `render_rt` maskeleri 0/N, `pick_gpu`
   objeyi döndürmez; `restored:` satırlarında N/N. `SKIP` = o backend'de
   read-back yok (-1) ya da obje hiç kurulmamış (0 instance) — PASS değil.
5. **★ EN SİNSİ: viewport'ta gizli, RT render'da görünür (ya da tersi).**
   Sessizce makul görünür çünkü baktığın pencere doğru davranır. Ölç:
   `scene.get_visible` → `viewport_raster_visible` ile `render_rt_visible`
   gizlenmiş objede İKİSİ DE 0 olmalı. Gözle: objeyi gizle, final render al
   (`render.start`), objenin gölgesi/yansıması da kalkmış olmalı. Biri 0
   diğeri değilse iki rol (`g_viewport_backend` / `g_backend`) tek backend
   sanılıp ikincisine `setVisibilityByNodeName` gitmemiştir
   (`render_is_viewport` alanına bak).
6. **★★ Alan raporu (2026-10-25, Nuclear Detonation / Plane_1): "göster"
   ilk seferde RT'ye ulaşmıyordu.** Kök: Vulkan `updateGeometry`, raster
   rebuild ve OptiX snapshot GİZLİ objeyi tamamen ATLAR — gizliyken herhangi
   bir rebuild olduysa objenin instance'ı yoktur ve `setVisibilityByNodeName
   (true)` maskeleyecek bir şey bulamayıp sessizce hiçbir şey yapmaz; obje
   alakasız bir sonraki rebuild'e kadar yoktur. Düzeltme: `setObjectVisible`
   gösterirken backend read-back'te ad için 0 instance (backend'de başka
   instance varken) görürse rebuild planlar; OptiX'te read-back yok, her
   göstermede OptiX rebuild planlanır. Ölç: test script'inin `rebuilt:` ve
   `restored:` fazları (`--rt-counters` ile RT sonucu da: yüzey sekmesi/yol
   gizlide değişmeli, restore'da baseline'a dönmeli). `restored: ... 0
   instances, baseline had N` FAIL'i tam bu hatadır. Log'da
   `[Visibility] ... geometry rebuild scheduled` görülmeli.
   Aynı testte gizliyken outliner gözü kapalı kalmalı: `rebuildMeshCache`
   rep facade'a `tmesh->visible` kopyalıyor; göz açık görünüyorsa o satır
   çalışmıyor. **H tuşu (hepsini göster) aynı hataya açık** — `showAllInstances`
   da düşmüş instance'ı geri getiremez; bu partide dokunulmadı.
7. **Pending-delete**: silinmiş (undo bekleyen) objeye `visible:true` → hata,
   obje gizli kalır.
8. **Bilinen sınır**: `scene.raycast` görünürlüğe bakmaz (snap yüzeyi
   sorgusu, eskisi gibi); gizli objeye hâlâ isabet etmesi bug değil. Proje
   kaydında görünürlük yazılıyor mu bu partide değişmedi.

## Kamera ışık adımı varsayılanı 1 (2026-10-25, derlenmedi; yalnız C++ varsayılanları)

1. **Açılışta `render.volume_march_lod` 1/4/4/32 döner**, panelde Camera Light
   Stride 1. 2 dönüyorsa eski exe. Kutunun gölgeli alt yüzünde yatay bant
   yok. Bant varsa kamera adımı hâlâ 2, ya da başka bir şey bant çiziyor
   (ayar 1/1/1/0 ile kontrol et).

## Gaz march LOD'u: ışık adımı + ikincil adım (2026-10-25, CANLI: splat+gaz −%47, tam sahne −%39; kamera varsayılanı 1 derleme bekliyor)

RT_GAS_SECONDARY_MARCH.md §8. Değişen shader'lar: volume_closesthit (march),
closesthit, hair_closesthit, raygen, rt_sky.glsl (yalnız dünya aynası
yeniden adlandırıldı). C++: vulkan_world_data.h, VulkanBackend(_Volumes),
RtApi, IPC, Python, yetki, panel, descriptor. RT pipeline baştan derlenir.
Kaynak: `python scripts/test/check_volume_march_lod_contracts.py` → OK.

1. **Ayar okunuyor.** `render.volume_march_lod` 2/4/4/32 döner; panelde dört
   kaydırıcı aynı değerleri gösterir. `render.set_volume_march_lod
   camera_light_stride=17` reddedilir. Okunmuyorsa ya da 0'lar dönüyorsa
   IPC/yetki zinciri eksik.
2. **Referans birebir.** 1/1/1/0'da görüntü ve maliyet bu partiden önceki
   gibi: splat_gas ~286, gas_only ~93 ms. Farklıysa kelime yanlış çözülüyor
   (0 = eski march olmalı) ya da `volumeShadowDue` k = 1'de eski koşul değil.
3. **Kazanç.** `python scripts/test/rt_volume_march_lod_ab_ipc.py`: splat_gas
   belirgin düşer (hedef ~150–180 ms); gas_only bir miktar düşer; splat_only
   aynı kalır. splat_gas'ta `secondary_density_samples` ~1/3'e iner,
   `camera_density_samples` değişmez.
4. **Görüntü.** Referans ile varsayılanı aynı kamerada karşılaştır:
   - doğrudan görünen gazın yumuşak gölgeleri ve ışık konisi kenarı;
   - gaz içindeki splat'ların dolaylı aydınlanması.
   Kamera tarafında hafif yumuşama beklenir; bant ya da basamak beklenmez.
5. ★ **Sinsi olan: donmuş gölge.** Gazın içinde, ışığa göre gölgede kalan
   bölge 2/4/4/32'de referanstan belirgin AYDINLIK ya da bütün ışın boyunca
   tek tonda görünüyorsa, önbelleklenen gölge ilk değerinde kalmış demektir
   (`volumeShadowDue` yenilemiyor). Bu "LOD biraz yumuşattı" gibi okunur ama
   yanlış ışıktır. Kontrol: `camera_light_stride` 1 ile 4 arasında değiştir;
   gölgeli bölge yalnız yumuşamalı, kaybolmamalı.
6. **Panel.** Kaydırıcı bırakılınca uygulanır ve birikim sıfırlanır; sürükleme
   sırasında her karede yeniden yükleme olmamalı.

## Gaz march'ı: kamera / ikincil sayaç ayrımı (2026-10-25, CANLI: toplam tam, ikincil pay %45.5 → ölçüt kaldı)

RT_GAS_SECONDARY_MARCH.md §4 adım 0. Yalnız ölçü aleti; görüntü ve maliyet
değişmemeli. Değişen shader'lar: volume_instrumentation.glsl,
volume_closesthit, closesthit (küre varyantı dahil).
C++: VulkanVolumeInstrumentation.h (224 → 248 B), RtApi, IPC, Python, panel,
descriptor.

1. **ABI tutuyor.** `render.volume_counters enabled=true`, sonra
   `render.volume_stats`. Eski alanlar önceki ölçümle aynı mertebede:
   splat + gazda yol başına `density_samples` ~33. Hepsi sıfır ya da saçma
   büyük sayılarsa shader ile C++ struct'ı hizasız; ikisi birlikte derlenmedi.
2. **Sınıflar toplamı kapatıyor.** `camera_density_samples +
   secondary_density_samples == density_samples`; aynısı rays ve shadow için.
   Eksik kalıyorsa sınıflanmamış örnek var: kayıt çağıran bir shader main()'de
   `volumeInstrumentationSetSegment` çağırmıyor.
3. **Yalnız gazda kamera ağır basıyor.** gas_only'de kamera payı büyük olmalı.
   Kamera bitinin anlamı doğru mu, buna bakılır.
4. ★ **Sinsi olan: ikincil sıfır.** splat_gas'ta `secondary_density_samples`
   0 ise toplam yine tutar (madde 2 geçer), ama kamera biti sekmelerde
   temizlenmiyor demektir. Bu durumda "ikincil pay yok, LOD işe yaramaz" diye
   yanlış karar çıkar. Splat'lardan seken ışınlar gaza giriyor: ikincil sayı
   sıfır olamaz.
5. **Karar ölçütü.** splat_gas'ta ikincil örnek payı > %60 ise LOD (adım 1)
   doğru hedefte.

## Gaz içindeki splat'lar: iç içe katı izi GERİ ALINDI (2026-10-25, CANLI DOĞRULANDI: 27 / 97 / 295 ms)

RT_GAS_OPTICAL_PERFORMANCE.md §"Seçenek A ölçüldü: gerileme, geri alındı".
Geri dönen shader'lar: volume_closesthit, closesthit (küre varyantı dahil),
hair_closesthit, miss, rt_payload.glsl. C++: özyineleme derinliği yeniden 2.
RT pipeline baştan derlenir.

1. **Maliyet eski yerine döndü.** Aynı sahne, kare 28, `fluid.set_phase_output`:
   gas_only ~94 ms, splat_gas ~311 ms, splat_only ~25 ms. A'lı derlemede
   265 / 406 / 25 ms ölçüldü. gas_only hâlâ ~265 ise sebep A değil; aynı ağaçtaki
   başka bir değişiklik (paralel ajan) ya da eski .spv. Önce .spv zaman damgasına bak.
2. **Görüntü.** Gaz içindeki splat'lar görünür, sonraki sekmenin gökyüzü yerinde.
   Bu yol A'dan önce canlı doğrulanmıştı; bozuksa geri alma eksik kalmış demektir
   (`grep PL_NESTED_SOLID shaders` boş olmalı).

## Hacim kutuları dolu hücrelere daralır: sıvı SurfaceSDF + host gaz (2026-10-25, derlenmedi; yalnız C++)

Matter Fog akışı daraltması önceki build'de canlı doğrulandı (kutular dar geldi).
Bu parti aynısını sıvı SurfaceSDF / host gaz yayınına ekler.
RT_GAS_OPTICAL_PERFORMANCE.md §"Devam: ilk sahne". Ölçüm: aynı sahne, kare 28.

1. **SDF kutusu daraldı.** Katı splat, gaz hidden, sıvı sdf iken
   `render.volume_slots`: "[Fluid NanoVDB]" slotu artık [0,5]³ değil, su ve
   köpüğün etrafında küçük bir kutu. Daralmadıysa: yükleme bu karede olmadı ve
   kutu hiç hesaplanmadı (bir sim adımı ilerlet), ya da canlı GPU yolu.
2. **Maliyet.** Aynı rota (splat / sdf / hidden) 135 ms'den ~26 ms'ye (boş
   sahne) yaklaşır; tam inmez, çünkü suyun kendisi gerçek bir yüzey.
3. **Su görünümü aynı.** Kenarda kesilmiş su yüzeyi ya da köpük = kutu fazla dar.
4. ★ **Sinsi olan:** sahne kaydedilip yeniden açılınca, ya da sim geri sarılınca
   su kutusu yeniden hesaplanmadan eski dar kutu kalırsa, yeni su o kutunun
   dışında GÖRÜNMEZ ve hiçbir hata vermez. Geri sar → oynat → suyun yeni
   konumunda göründüğünü kontrol et.
5. **Gaz (host yolu).** Host'tan yüklenen gaz domain'lerinde de kutu daralır;
   canlı GPU gazında değişmez.

## Hacim ışık önbelleği — A/B anahtarı söküldü (2026-10-25, derlenmedi; yalnız C++)

V1 canlı kabul edildi (docs/dev/VOLUME_LIGHT_CACHE.md §11–12). Bu parti yalnız
`enabled`'ı söker; shader değişmedi, RT pipeline yeniden derlenmez.
Kaynak kontrolü: `python scripts/test/check_volume_light_cache_contracts.py` → OK.

1. **Derleme temiz.** `VolumeLightCacheConfig` hiçbir yerde kalmamış olmalı;
   kalmışsa bağlayıcı/derleyici hatası verir.
2. **Panel.** Performance'ta onay kutusu yok, "Light Cache Cells" var; değiştirince
   `render.volume_light_cache.max_cells` aynı değeri gösterir.
3. **IPC.** `render.set_volume_light_cache {max_cells: 48}` çalışır,
   `{enabled: false}` "max_cells" eksik diye reddedilir. `render.volume_light_cache`
   çıktısında `enabled` alanı yok.
4. ★ **Sinsi olan:** önbellek gerçekten açık mı? Counters açıkken duran karede
   `light_cache_hits` > 0 ve misses ≈ 0. Anahtar sökülürken `wanted` koşulu
   yanlış kalsaydı hiçbir hata olmaz, yalnızca eski hıza dönülürdü.

## Islak kum 4. parti — tane hash kovası 16 → 32 (2026-10-23, derlenmedi; C++ + shader)

Canlı (teğet seed build'i): 50 000 ıslak Sand tanesi (ıslaklık 1.0) düşerken
"grain neighbour bucket overflow (more than 16 grains hashed to one bucket)"
ile tutuldu — havada, küp collider'ın 35 cm üstünde (çarpma değil; sertlik ×4
de aynı). Sebep: ıslak hash hücresi 3r, yoğun seed kafes adımı 1.474r → kötü
hizada hücrede ~14 merkez (meşru); 16'da tek hash çakışması taşırıyor.
Değişiklik: `kMatterGrainBucketCapacity` 32, shader `BUCKET` 32, revizyon 31
(32-bit indeks 2^31'e sığar; 50k tanede kova belleği ~18 → ~35 MB).

1. **Shader derlendi:** `sim_matter_grain` revizyon 31; eski .spv ile adım
   "shader revision mismatch" der.
2. **Aynı sahne:** 50k ıslak Sand bloğu düşüp yere çarpar, tutulma yok.
3. ★ **Sinsi olan:** "neighbour list full" (bit 8) — 32'lik Verlet listesi
   yoğun ıslak paketlemede dolabilir (3r içinde FCC ~18 komşu, iç içe
   geçmede daha fazla); retry var ama taşarsa yine tutar.

## Islak kum 3. parti — teğet seed (2026-10-23, derlenmedi; yalnız C++)

FCC aralığı 2.2002R → 2.0851R (jitter 0.07R → 0.03R): boşluk 0.2R → 0.085R,
doluluk 0.556 → 0.653. Not: ISLAK_KUM_DOGRULAMA.md 3. parti + DEVİR bölümü.

1. **C++ testi** `matter_grain_birth_test.cpp`: 172 tane, ≥ 2R ayrılık geçmeli.
2. **Seed sayısı:** 0.3×0.5×0.3 m Sand, r 12.5 mm → ~3300 (önce 2813).
3. **İlk kare sakin:** seed sonrası 5 adımda ortalama hız 0.52 m/s değil, küçük
   (blok bulut gibi düşmüyor); `fluid.get settle_converged` true olmalı.
   ★ **En sinsi sonuç:** kuru blok ayakta kalıyor — FCC kilitlenmesi; gerçek
   kuru kum çöker. Öyleyse jitter'ı ölçerek büyüt (d formülüyle birlikte).
4. **Islak kısa kolon** (0.2 m, ıslaklık 0.3, Sand r 0.25 mm): önceki 0.029'dan
   yüksek olmalı; hâlâ çöküyorsa yapışma ölçeği (DEVİR maddesi) sırada.

## Islak kum 2. parti — FCC seed + köşeli kum (2026-10-23, derlenmedi; yalnız C++)

Canlı (18:09): seed ıslaklığı çalışıyor (wet_grains, 8834 köprü), ama seed'li
blok kuru da ıslak da tek tane kalınlığına yayıldı. Kökler (ISLAK_KUM notu):
Sand μ_r 0.02 (kuru döküm ~8°; 0.2 ile 30°, ıslak 47°) ve basit kübik seed
(0.39, kayma direnci yok). Değişiklik: seed FCC (0.556), Sand μ_r 0.2 + su
kapasitesi 0.1, Gravel μ_r 0.25; `matter_grain_birth_test.cpp` güncellendi.

Sıra:

1. **C++ testi:** `scripts/test/matter_grain_birth_test.cpp` geçmeli (172 tane,
   ≥2R ayrılık, voxel bağımsızlığı, bütçe öneki).
2. **Seed yoğunluğu:** Sand seed 0.3×0.5×0.3 m, r 12.5 mm → ~2600–2900 tane
   (önce 1800). *Bozuksa:* hâlâ 1800 = eski exe.
3. **Kolon:** `.\scripts\ipc\Probe-WetSandBlock.ps1 -Case column -Wetness 0,0.3`
   — artık türetme gerekmez ama probe WetSandProbe kullanır (Sand + kapasite
   0.1; μ_r Sand'den 0.2 gelir). Kuru: yığına çöker (merkez ~0.06–0.08);
   ıslak 0.3: belirgin yüksek (ayakta ya da dik). ★ **En sinsi sonuç:** kuru
   kolon da ayakta kalır → FCC kristal kilitlenmesi (gerçek kumda yok); o zaman
   jitter'ı ölçerek büyütmek gerekir.
4. **Kaya:** `-Case rock` — ıslak blok topak halinde çatlamalı.
5. **Regresyon:** kuru Sand dökümü artık ~30° (önce düz yayılıyordu); Sand +
   Water sahnelerinde taneler ıslanır (köprü) — tasarım gereği.

## Islak kum 1. parti — seed'e başlangıç ıslaklığı (2026-10-23, derlenmedi; yalnız C++)

★ Önce: kullanıcı "derledim" dedi ama `x64/Release/RayTrophiStudio.exe` ve
`RayTrophiStudio.log` 16:10 — kar 6. parti (tek Snow, substep paneli) bu
exe'de YOK (madde listesinde Packed Snow duruyor). Bu build ikisini birden getirir.

Bulgular (canlı, 16:10 exe):
- Sand = DEM (`granular_transport dem`), `grain_water_capacity_fraction = 0` →
  Sand hiç ıslanamaz; ıslak kum için türetip kapasite vermek gerekiyor.
- Islak tane yalnız akış kaynağından (`grain_birth_saturation`) doğuyordu;
  `fluid.seed` hep KURU → ıslak, sıkıştırılmış blok yazılamıyordu.
- Kalıp testi: 3600 ıslak tane döküldü (wet_grains true, merkez 0.277 m), ama
  kalıp collider'larını kapatmak simülasyonu 0. kareye sardı → 0 parçacık.
  Sim sürerken collider/kaynak düzenlemek = geri sarma; test tek kurulum olmalı.
- Köprü kuvveti ölçeklemesi doğru: F = 2πγcosθ·R·(R/r) (2026-10-21).

Değişiklik: domain seed tarifine `fluid_seed_grain_saturation` (0..1 kapasite);
seed DEM tanelerine `initMatterGrainBirthWater` ile su verir; ownership bunu
ıslak doğum sayar. Kayıt (proje + sahne), imza hash'i, `fluid.seed
grain_saturation`, `rt.fluid.seed(grain_saturation=)`, `fluid.get
seed_grain_saturation`, panelde "Seed Grain Wetness" (yalnız tane domain'inde).

Sıra:

1. **Exe yeni mi?** `RayTrophiStudio.exe` tarihi > 16:10, madde listesinde
   Packed Snow YOK. Değilse aşağıdakilerin hiçbiri koşamaz.
2. **Kolon testi:** `.\scripts\ipc\Probe-WetSandBlock.ps1 -Case column`
   (kuru 0 ve ıslak 0.3). Kuru: merkez 0.25'ten belirgin düşer (yayılır);
   ıslak: ~0.25'te kalır (ayakta). `wet_grains` ıslakta True, kuruda False.
   ★ **En sinsi sonuç:** ikisi de aynı düşüyor ve wet_grains True — seed suyu
   veriliyor ama köprü kuvveti tutmuyor (yapışma bu ölçekte zayıf); ikisi de
   ayakta → kuru kum da yapışıyor (sürtünme/uyku tutuyor), o da yanlış.
3. **Kaya testi:** `-Case rock`: ıslak blok kayanın üstünde topak halinde
   kalmalı/çatlamalı (merkez ~0.45), kuru blok kayanın etrafına akmalı.
4. **Panel:** tane domain'inde seed bölümünde "Seed Grain Wetness" kaydırıcısı;
   değiştirip bırakınca yeniden seed eder; proje kaydedip açınca değer kalır.

## Kar 6. parti — tek Snow, substep yalanı (2026-10-23, derlenmedi; yalnız C++)

Kullanıcı: "iki kar ön ayarı kafa karıştırır" ve "UI alt adımı 12 gösterse de
200 çıkıyor". Ölçüm: panelde görünen tek sayı **Max Advection Substeps = 12**
(`APICSolverParams::max_substeps`) — Matter granular yolunda hiçbir şeyi
sınırlamıyor (her ortak alt adımın içindeki taşımayı sınırlar, orada zaten 1).
Gerçek sayıyı elastik dalga CFL'si veriyor: 1/24 s ÷ (0.35 × 0.025 m ÷
√(1.4e5·5 / 400) = 41.8 m/s) = 200. Panel yalnız "adaptive" diyordu.

Değişiklik: `Packed Snow` kaldırıldı (kimse türetmemişti; ad hash etiketi ⇒
kayıtlı projeye etkisi yok). `Granular::planGranularSubsteps` (planlayıcıyla
aynı aritmetik) → panel "Granular Skeleton" başlığında: dinlenmede substep,
son kare, formül (kare dt / (0.35·voxel / dalga hızı)), dalga hızının
bileşenleri. Advection kaydırıcısının altında granular için açıklama. IPC/Python:
`granular_predicted_substeps`, `granular_wave_speed_m_s`,
`granular_sizing_young_modulus`, `granular_frame_dt`.

Sıra:

1. **Madde listesi:** yalnız `Snow`; `Packed Snow` yok.
2. **Panel = koşu.** Snow domain'i (voxel 0.025, 24 fps): "Solver substeps per
   frame: 200 at rest | last frame: 200". `fluid.get granular_predicted_substeps`
   200. *Bozuksa:* tahmin ≠ son kare (dinlenmede) ⇒ planlayıcı ile UI farklı
   formül kullanıyor (ikisi de `planGranularSubsteps`/`elasticStepInfo` olmalı).
3. **Advection kaydırıcısı** altında "Granular material: the solver substep
   count is set by its stiffness…" satırı (yalnız granular domain'de).
4. ★ **Sinsi olan:** çarpma anında "last frame" tahminden BÜYÜK olabilir
   (gerinme hızı terimi) — bu doğru; dinlenmede eşit olmalı.

## Kar 5. parti — sertleşme tabanı (2026-10-23, derlenmedi)

Build: **C++ VE `compile_shaders.bat`** (stress_update; push constant boyutu aynı,
son pad → `snow_min_hardening`). Yeni madde alanı `granular_snow_min_hardening`
(Snow 0.1). `detached` artık "tabana inmiş" demek (eskiden %99 kayıp).

4. parti canlı (15:43 exe): topak yapısı atandı (J_P 0.952..1.105), kaya sonucu
DEĞİŞMEDİ. Hız taraması 2.3 / 3.3 / 4.6 m/s → hepsinde ~%70 yumuşama, %26 toz
(enerjiden bağımsız). Kayaya ÇARPMADAN konan blok 1.5 s'de dağıldı (varsayılan
E); E 7e5 ile aynı blok sağlam durdu. Görsel (ekran görüntüsü, silindi): kaya
bloğu DELİP içine girmiş, blok kayayı sarıp zemine inmiş, dışarıdan sağlam küp
gibi; sayaçlardaki "toz" kayanın çevresindeki kesme bölgesiydi. Kütle merkezi
y 0.305 (kaya üstünde dururken beklenen ~0.39).

Kök: exp(ξ(1−J_P)) bütün modülleri (sıkışma dahil) çarpıyor ve tabanı yoktu;
gevşeyen kar basınçsız toz oldu, engelin etrafından aktı.

Sıra:

1. **Shader derlendi** (`.spv` > `.glsl` tarihi).
2. **Delip geçme bitti.** `Probe-SnowImpact.ps1 -Target rock`: `centroid y`
   0.305'ten belirgin yüksek (0.35+). `powder` sütunu artık tabandaki
   parçacıklar; çatlak yüzeyleri kadar olmalı, bloğun yarısı değil.
   ★ **En sinsi sonuç:** merkez yükselir ama blok tek parça kayanın üstünde
   asılı kalır (çatlamaz). Görsele bak: kenarlar aşağı kopmalı.
3. **Durgun kaya testi** (varsayılan E): kayanın üstüne konan blok dağılmamalı
   ya da yalnız sarkan kenarları kopmalı.
4. **Zemin regresyonu:** `-Target floor` önceki gibi (ortalama J_P ~0.68).
5. **Packed Snow (karar verildi: Snow toz kar kalır).** Yeni yerleşik madde
   `Packed Snow`: aynı yasa, ρ 550, E 1e6, iletkenlik 0.45; katalogda Snow'un
   yanında, görünümü Snow ile aynı. Test: `fluid.set_param SnowTest
   default_substance="Packed Snow"` → `Probe-SnowImpact.ps1 -Target rock`:
   kaya üstünde konan blok sağlam durmalı (E 7e5 ölçümü gibi), 4.6 m/s'de
   kenarlar kopmalı, `centroid y` ~0.39'a yakın. Substep Snow'un ~2.3 katı
   (voxel 0.025: ~180).
   *Bozuksa:* madde listede yoksa eski exe; Snow ile aynı davranıyorsa E
   okunmuyor (`fluid.get granular_young_modulus` 1e6 olmalı).

## Kar 4. parti — Voronoi topak yapısı (2026-10-23, derlenmedi; yalnız C++)

Build: yalnız C++ (shader değişmedi). Madde alanı yeni: `granular_snow_clump_size`
(m, varsayılan 0.12); Snow dağılımı 0.05 → 0.1.

3. parti canlı (15:14 exe, 15:11 .spv; `Probe-SnowImpact.ps1`, voxel 0.025):
buz tabanı PASS, `hcap` 0. Zemin: 5.6 → 0.5 m/s iki adımda (FLIP'siz 0.17),
ortalama J_P 0.686. Kaya: %65 parçacık %50+ yumuşamış, %20–45 toz — FLIP öncesiyle
aynı. Mandal yalnız ~12 basınçsız yüzey parçacığında (sönüm basınç kapılı).
Homojenlik testi: bağımsız (beyaz) dağılım 0.05 → 0.3 hiçbir şeyi değiştirmedi →
sorun genlik değil İLİŞKİ UZUNLUĞU (P2G ~50 parçacığı ortalıyor).

Sıra:

1. **Yapı atandı.** `Probe-SnowImpact.ps1 -Target floor` ilk satırında J_P aralığı
   ≈ e^±0.1 (0.905 .. 1.105) ve ortalaması 1'den biraz büyük (sınırlar gevşek).
   *Bozuksa:* 0.951..1.051 = eski exe (beyaz gürültü, 0.05).
2. **Kaya: topaklar kalıyor (asıl hedef).** `-Target rock`: `powder` ve `d90`
   3. partiden (yaklaşık 8 000 / 10 000) belirgin düşük, `d50` ise yüksek kalabilir
   (çatlak yüzeyleri). ★ **En sinsi sonuç:** sayılar düşer ama kar hâlâ tek
   kütle — çatlak açılmamış, yalnız kırılganlık azalmış. Ayırt etmek için
   görsel bak: sınırlar boyunca ayrılan 10–15 cm'lik parçalar olmalı.
3. **Zemin:** blok yine sıkışır (beklenen); kenarlardan topak kopması varsa artı.
4. **Topak boyutu duyarlılığı:** SnowProbe gibi türetilmiş bir maddede
   `granular_snow_clump_size` 0.06 / 0.2 dene; parça boyu onunla değişmeli.
   0.1 m voxelde 0.12 → 0.3 m'ye (3 voxel) yükseltilir; bu bilinçli.

## Kar 3. parti — FLIP + yavaş parçacık sönümü (2026-10-23, derlenmedi)

Build: **C++ VE `compile_shaders.bat`**. ★ Önce şunu bil: 15:03 exe'si kar
2. partinin C++'ını içeriyor ama `.spv`'ler 13:15 tarihli — tavan ve buz tabanı
kernel'de YOK. Canlıda J_P 0.303'e indi (taban 0.436) ve durgun blokta %76
parçacık `hcap` bayrağı taşıdı. Bu partinin değişen shader'ları:
`sim_fluid_granular_stress_update.glsl` (2. parti), `sim_fluid_granular_settle.glsl`
(push constant boyutu aynı, pad0 → `damping_max_speed`).

Canlı ölçüm (`scripts/ipc/Probe-SnowImpact.ps1`, voxel 0.025, 17 576 parçacık,
0.3 m blok ~2 m'den, ~6 m/s; adım 49 ms, 80 substep):
düz zemin → ortalama J_P 0.68, 13 parçacık %50+ sertlik kaybı, dinlenmede J_P max
1.06 → 1.17 tırmanıyor (titreşim mandalı); küre kaya → ortalama J_P 1.73, %67
parçacık %50+ kayıp, %49 toz. Model kırılıyor (geometri ayrımı PASS), ama düz
zeminde ölü durma ve kayada topaksız toza dönme.

Değişiklik: Matter yolunda kar için FLIP (Snow `solver_flip_blend` 0.95; P2G
gerilme impulsundan ÖNCE taban kopyası, aynı ağırlıklarla normalize;
`granularFlipSnapshotWanted`), settle sönümü kar için yalnız < 0.5 m/s parçacıklara.

Sıra:

1. **Shader'lar derlendi.** `.spv` tarihleri `.glsl`'lerden yeni. Bozuksa her
   şeyi maskeler.
2. **Kar 2. parti kernel'i.** `Probe-SnowImpact.ps1 -Target floor`: J_P floor
   PASS (≥ 0.436), `hcap` ~0.
3. **Titreşim mandalı yok.** Aynı koşuda "rest ratchet" PASS: dinlenmede J_P max
   sabit. *Bozuksa:* settle `damping_max_speed` okunmuyor (eski settle .spv).
4. **FLIP etkin.** Kaya vakasında çarpmadan sonra `v_max` 3. partiden yüksek
   kalmalı (parçalar uçuyor); düz zeminde blok "ölü durmamalı" (çarpma sonrası
   ilk örnekte v_mean önceki 0.17'den belirgin büyük).
   ★ **En sinsi sonuç:** FLIP açık görünür ama taban alanı boş/sıfırsa G2P
   v_p + v_new − 0 yapar → kar ivmelenip patlar ya da `invalid` > 0. Patlama
   görürsen önce scratch alanın dolup dolmadığına bak.
5. **Kayada topak kalıyor mu.** `d90`/`powder` 3. partiden (8 662 / 17 576)
   belirgin düşük olmalı; hâlâ yarısı tozsa θs (0.0075) bu çözünürlük için
   fazla kırılgan — ayarı ölçerek değiştir.
6. **Kum regresyonu.** Sand/Gravel (DP) domain'inde FLIP KAPALI kalmalı
   (`flip_blend` yalnız snow'da okunur): yığın davranışı değişmemeli.

## Ayrı parti — ölçeklenmiş küp + collider takibi (2026-10-23, derlenmedi)

Build: yalnız C++ (shader yok, yeni .cpp yok). Dosyalar: `scene_data.h`,
`SceneSelection.h/.cpp`, `scene_ui_hierarchy.cpp`, `RtApi.h`, `RtApiFluid.cpp`,
`RtIpc.cpp`, `RtPython.cpp`.

Kökler:
- **Küp eski boyuta dönüyor** = inspector'ın Pos/Rot/Scl alanları. Herhangi bir
  alan değişince `fromTRS(pos, rot, scale)` TÜMÜYLE handle'a yazılıyor; bu üçlü
  seçim anında alınmış bir KOPYA (`sel.selected`). IPC `scene.set_transform`,
  undo, script onu hiç tazelemiyor; gizmo da bir kare geriden tazeliyor
  (`getPivotMatrix()` yeni matris uygulanmadan ÖNCE okunuyor). Sonra Pos'a
  dokunmak eski ölçeği geri yazıyordu. IPC yolu kendisi temiz: canlıda
  ölçek 2 → taşı → ölçek 2 kaldı. Düzeltme: alanlar çizilmeden önce
  `syncObjectTransformFromHandle()` canlı handle'dan tazeler (önbellek hâlâ
  aynı matrisi veriyorsa Euler korunur).
- **Collider eski pozda kalıyor** = `getSurfaceMeshCacheForObject` epoch memo'su.
  Epoch yalnız geometri jenerasyonu ya da keyframe re-pose ile bitiyordu; düz
  transform düzenlemesi (inspector, gizmo, `TransformCommand`=IPC/undo, script)
  bitirmiyordu. Tek istisna seçili obje + duraklatılmış timeline (gizmo
  çiziminde). Düzeltme: memo, handle'ın build anındaki matrisiyle eşleştikçe
  geçerli (tüketici tazeliği kontrol eder; tüm yollar kapsandı).
- **IPC körlüğü**: `collider.list/get` yalnız yazılmış `bounds_min/max`'ı
  (obb modunda KULLANILMAYAN -1..1) gösteriyordu. Yeni salt-okunur alanlar:
  `resolved_valid`, `resolved_world_min/max`, `resolved_size` — çözücülerin
  kullandığı resolver'dan.

Sıralı kontrol (bağımsız/hızlı önce):

1. **IPC readback var mı** — `Invoke-RtIpc collider.list @{}` →
   `Default_Cube Collider` satırında `resolved_valid: true`, `resolved_size`
   küpün ölçeğine eşit (ölçek 1 küp için ~2,2,2).
   Yoksa: eski exe (zaman damgasına bak).
2. **Collider IPC ile takip ediyor** — `scene.set_transform @{name='Default_Cube';
   scale=@(2,1,1)}` → hemen `collider.list`: `resolved_size` ≈ (4,2,2);
   ardından `translation=@(0.5,…)` → `resolved_world_min/max` x'te +0.5 kaydı,
   boyut aynı. Timeline OYNARKEN de aynı sonuç gelmeli.
   Eski boyut/konum dönüyorsa: memo kontrolü devreye girmiyor (matris
   karşılaştırması); `resolveObjectTransformForSimulation(…, false)` bulamıyor olabilir.
3. **Panel: ölçek korunuyor** — küpü seç, IPC ile ölçekle (2,1,1), sonra
   inspector'da Pos'u sürükle. Ölçek 2 KALMALI, Scl alanı 2 göstermeli.
   Aynısını gizmo ile ölçekleyip inspector'dan taşı. Bozuksa: sync çağrısı
   alanlardan önce çalışmıyor.
4. **Panel: döndürülmüş obje Rot sürüklemesi** — Rot.y'yi 0→120 sürükle;
   değer sıçramamalı/titrememeli. Titriyorsa: sync her kare tetikleniyor
   (karşılaştırma toleransı / konvansiyon uyumsuzluğu).
5. **Undo** — ölçek + taşıma sonrası Ctrl+Z: obje ve collider birlikte döner.
6. ★ **Sinsi olan** — kar MPM sim'i (reset + oynat) ölçeklenmiş/taşınmış küpe
   çarpmalı. Collider overlay doğru görünüp sim eski yere çarpıyorsa: sim kare
   önbelleği eski kareleri oynatıyor (bu parti önbelleği temizlemiyor —
   reset gerekir). Bu sessizce makul görünür; kar havada eski küpe oturur.
7. **Perf** — `rt.perf`'te `sim.collider.surface_cache_rebuild` statik sahnede
   kare başına 0 kalmalı (memo hâlâ tutuyor). Artıyorsa: handle matrisi her
   kare yeniden yazılıyor (bit düzeyinde değişen bir yazıcı var).

Bilinen sınır: inspector Pos/Rot/Scl düzenlemesi hâlâ undo kaydı ATMIYOR ve
`fromTRS`'in dönüş inşası `getPivotMatrix`'ten (Rz·Ry·Rx) farklı görünüyor —
doğrulanmadı; döndürülmüş objede ayrı iş, bu partide dokunulmadı.

## Kar 2. parti — sertleşme tavanı, buz tabanı, yoğunluk dağılımı (C++ derlendi, SHADER DERLENMEDİ)

Build: **C++ VE shader** (`compile_shaders.bat`; push constant boyutu değişmedi, iki pad alan oldu).
Not: `docs/dev/KAR_STOMAKHIN_MPM.md` §6.

Canlı ölçüm (1. parti build'i, 87 552 parçacık, 0.1 m voxel, ~3 m düşüş, 7 m/s çarpma):
serbest düşüşte J_P = 1.000 sabit (kayma yok), 9 substep / 92 ms. Çarpma karesinde
7646 parçacığın modülü dalga sınırına kırpıldı (substep bir kare geç kaldı), blok
eşit sıkıştı (ortalama J_P 0.74, min 0.395 = buzdan yoğun), hiçbir parçacık
gevşemedi (max J_P 0.88), sonra 184 substep / 430 ms'de KALICI kaldı.

Ek ölçüm (1. parti exe, 13:26): çarpmada adım 131 → 528/420 ms (dt 1/60);
GPU kernel toplamı substep başına ~2.3 ms, bunun ~1.7 ms'si `sim_matter_p2g`.
Substep başına maliyet makul; patlayan şey substep SAYISI (9 → 184). 24 fps
zaman çizelgesinde dt 2.5× → "kare başına saniyeler". Domain DEM yolundan
geçiyor: Sand → Surface madde bağı Sand'in `dem` taşımasıyla tane çözücüsünü
açıyor (0 tane varken) ve backend'i Vulkan'a kilitliyor.

Sıra: bağımsız ve hızlı görülen önce.

1. **Shader yüklendi.** Log'da stress_update push-constant hatası yok.
   *Bozuksa:* .spv eski; aşağıdakilerin hepsini maskeler.
2. **Maliyet sabit.** `Probe-SnowStomakhin.ps1` çıktısında `sub` serbest düşüşte
   VE çarpmadan sonra aynı (~9·√5 ≈ 20). Adım süresi ~200 ms civarı, 430 değil.
   *Bozuksa:* çarpmadan sonra artıyorsa boyutlandırma hâlâ ölçülen J_P'ye bakıyor
   (eski exe). Daha ucuz istersen Snow'da `granular_snow_max_hardening` = 1 →
   ~9 substep (sıkışınca sertleşme yok, gevşeme/kırılma yine çalışır).
3. **Kırpma sıfıra yakın.** `hcap` (hardening_capped) çarpma karesinde ~0.
   *Bozuksa:* yüzlerce = substep tavana göre boyutlanmıyor.
4. **J_P buz tabanı.** Probe "J_P floor respected" PASS (≥ 0.436).
5. **Başlangıç dağılımı.** Seed sonrası ilk satırda J_P ≈ 0.951 .. 1.051.
   *Bozuksa:* 1.000/1.000 = `assignSnowHeterogeneity` çağrılmıyor.
6. **Gevşeme (asıl hedef).** Probe "loosening" PASS: J_P max, başlangıç
   dağılımının üstüne çıkıyor. ★ **En sinsi sonuç:** blok yine eşit sıkışıp
   esniyor, J_P max başlangıçtaki 1.05'te kalıyor. Bu, düz levhanın düz zemine
   çarpmasının geometrisi olabilir (yanal kuşatma, çekme yok) — 7. madde ayırır.
7. **Geometri ayrımı (kullanıcı/ajan):** aynı karı küre olarak ya da eğik bir
   collider'a düşür. Kürede/eğimde kırılıp levhada kırılmıyorsa model doğru,
   levha testi çekme üretmiyor. İkisinde de kırılmıyorsa granular hatta FLIP
   eksikliği (KAR notu §6) sıradaki iş.
8. **Görsel:** kenarlardan topak kopması, açık kalan çatlaklar.
9. **Backend kilidinin sebebi görünüyor.** Domain sekmesinde kilitli backend
   altında "DEM is on because Sand (substance binding) uses granular transport
   'dem'…" yazmalı; `fluid.matter_models` → `grain_ownership.origin` aynı şeyi
   döner. Sand bağını kaldırınca (ya da reset sonrası) CPU seçilebilmeli.
   *Bozuksa:* yalnız "Grains need the Vulkan backend." = eski exe.

---

## 1. parti (2026-10-23) — Stomakhin dalı + sayaç körlüğü + gerilme yoğunluğu

Canlı sonuç: sayaçlar ölçülüyor (measured true, counted = n), snow yasası seçili,
serbest düşüşte J_P sabit — 1–3. maddeler PASS sayılabilir. 4–7 yukarıdaki partiye
devredildi.

## Önceki parti (2026-10-22, 3. parti) — henüz doğrulanmadı

Önceki partinin açık deneyi (RT gaz phase-cache var/yok iki build) bu partiyle
çakışmıyor; ayrıntısı `RT_GAS_OPTICAL_PERFORMANCE.md` sonunda.

Build: C++ yeterli (shader değişmedi). Değişen: `MaterialNodesV2.h` (derleyicide
`inPin`: değeri pinde yaşayan 25 giriş artık pin değerini derler),
`ParticleRenderBridge.cpp` (yakın mesh havuzu kapasite kuralı
`nearPoolCapacityFor`). Gerçek shard + mineral ölçümü bu build'den önce canlı
koşuldu (KUM_TANE…md §8b).

Sıra: bağımsız ve hızlı görülen önce.

1. **Mix Color rengi render'a ulaşıyor.** Yeni principled materyal → node
   grafiği: Mix Color (Fac 0, Color1 kırmızı, bağlantısız) → Base Color;
   bir küpe ata, Rendered (Vulkan RT). Küp **kırmızı** olmalı.
   Ajan yolu: `rt_grain_mineral_material.py` yerine aynı grafiği
   `nodes.set_param` ile kurup `render.probe`/ekran görüntüsü.
   *Bozuksa:* gri (0.5) = derleyici hâlâ literal basıyor (eski exe — zaman damgası).
2. **Eski sahneler.** Mix Color / Hue-Sat / Combine Color kullanan kayıtlı bir
   materyali aç. ★ **En sinsi sonuç:** render'ı DEĞİŞEBİLİR — bu düzeltmenin
   kendisi: artık panelde görünen pin değeri çiziliyor, eskiden 0.5/1.0 literal.
   Değişen sahne bug değil; değişmeyen ama panelde farklı renk gösteren sahne bug.
   *Bozuksa:* bağlı girişli bir node'un rengi değiştiyse `inPin` bağlantıyı
   atlıyor (olmamalı — ilk satırı `inOr` ile aynı).
3. **Küre yolu değişmedi.** Near-mesh kapalı bir kum sahnesinde
   `render.instance_stats` ve ms/örnek önceki build'le aynı.
   *Bozuksa:* değişiklik ana havuza sızmış (yalnız yakın havuz kuralı değişti).
4. **Boşluk azaldı.** Ajan fixture'ı kurar ve tam K0'ı koşar:
   `python scripts	est
t_k0_grain_mesh_near_ipc.py --domain K0_Sand --apply --distance 1.5 --library fracture`.
   Beklenen hepsi-mesh vakası (4591): `tlas_instances` ≈ 6.1 bin (önce 12321);
   1.5 m (780): ≈ 1.0–1.6 bin (önce 3105). Script artık kullanıcının near
   `sources` listesini de geri yükler (`near_restored` true).
   *Bozuksa:* hâlâ ~12 bin = eski exe ya da kapasite yolu eski kuralı kullanıyor.
5. **Kamera hareketi.** Fixture'da kamerayı birkaç kez az oynat (aynı mesafe
   bandında): her harekette yapısal yeniden kurma olmamalı (`perf.list`
   `accel.vulkan_rt.rebuild` sayısı hareket başına artmamalı, yalnız kapasite
   aşılınca artar).
   ★ ms/örnek boşluk azaldıktan sonra hiç değişmezse "boşluk bedava" diye
   okunur — opak shard ölçümü (§8b) zaten bunu söylüyor; ama yalnızca
   `tlas_instances` gerçekten düştüyse geçerli. Önce slot sayısına bak.

6. **Grafik silme + Color Ramp durakları + pin değeri (tek script).**
   `python scripts	est
t_material_node_parity_ipc.py` → `PASS`
   (pin_value, remove, ramp_stops). Kendi küpünü 60 m yukarıda kurar, `render.probe`
   ile ölçer, kamerayı geri yükler.
   ★ **ESKİ exe'de bu script uygulamayı ÇÖKERTTİ** (2026-10-22): `nodes.set_property`
   iç içe dizi değeri (`[[...],...]`) alınca pipe koptu, süreç kapandı. Eski
   `nodeParamFromJson` 3 elemanlı diziyi Vector3 sanıp `get<float>()`'u bir dizi
   üzerinde çağırıyordu; atılan `json::type_error` yakalanmıyor gibi. Yeni build'de
   bu yol `Rows` olarak ayrıştırılıyor. Yine de kontrol et: IPC dispatch'i
   `std::exception` (yalnız `runtime_error` değil) yakalıyor mu? Yakalamıyorsa
   herhangi bir metoda bozuk tipli değer göndermek uygulamayı düşürür.

Doğrulama (ajan): değişiklikler derlenmedi. Derleyici hatası canlıda bulundu
(kırmızı Mix Color gri çizildi), düzeltme sonrası koşulmadı.

---

## Ayrı parti (aynı build): fracture shard'ları flat SoA

Kaynakta, derlenmedi. Değişen: `scene_ui_menu.hpp` (shard üretimi + silme),
`scene_data.h` (`nodeWorldCenter`, `getRigidBodySourcePivot`, kayıt öncesi
rest snapshot'ı, yeni `forgetBodyMeshCaches`), `ThermalFractureBridge.cpp`
(`nodeMeshVolume` artık `indices` üzerinden). Yeni .cpp yok, IPC yüzeyi değişmedi.
Her shard artık **tek** `TriangleMesh` (ardışık üçlü, `indices[i]==i`); facade üretilmiyor.

Sıra: bağımsız ve hızlı görülen önce.

1. **Shard tek flat mesh.** `scene.add_primitive cube` → `physics.fracture_object`
   (site_count 24) → bir `shard_objects` adına `scene.object_info`.
   Görmen gereken: `meshes: 1`, `triangles` > 0, `has_bounds: true`.
   *Bozuksa:* `meshes: 0` = hâlâ facade üretiliyor (eski exe — zaman damgasına bak).
2. **Görünüm/malzeme aynı.** Viewport'ta preview gap'li shard'lar; exact modda dış
   yüz kaynak dokusunu taşır, kesik yüzler `<obje>_Interior` malzemesinde.
   *Bozuksa:* hepsi tek renk/doku (0,0) = materialID/uv özniteliği yazılmamış.
3. **Seçim/picking.** Bir shard'a viewport'ta tıkla → hiyerarşide o shard seçilir;
   Rigid Body listesinde tıklamak da viewport'ta seçer.
   *Bozuksa:* `rebuildMeshCache` flat shard'ı `direct_mesh_nodes`'a almıyor.
4. **Re-fracture + unfracture.** Aynı objeyi farklı seed'le ikinci kez kır, sonra
   `physics.unfracture_object`. Beklenen: ikinci kesimde eski shard'lardan **hiçbiri**
   kalmaz (obje sayısı = yeni shard sayısı + diğerleri); unfracture orijinali geri
   koyar, `__shard_` adlı obje kalmaz.
   *Bozuksa:* çift shard = `fractureEraseNodes_` flat'i silmiyor.
5. **Make Breakable → kır → simüle et.** `physics.make_fracture_group` (veya panel),
   `physics.break_fracture_group`, birkaç kare oynat. Shard'lar ayrılıp düşmeli,
   grup kütlesi hacme göre (24 shard'lık 2 m küp ≈ 8 m³ × yoğunluk), **24 kg değil**.
   Patlama merkezden dışa doğru olmalı.
   *Bozuksa:* hepsi aynı yöne/orijinden uçuyorsa `nodeWorldCenter` flat'i okumuyor;
   kütle = shard sayısı kg ise `nodeMeshVolume` 0 dönüyor.
6. **Re-fracture SONRASI simülasyon.** 5'i yap, sonra objeyi re-fracture et ve tekrar
   kır/oynat. Çökme veya eski kesimin pozunda donmuş shard olmamalı.
   *Bozuksa:* rigid bake cache'i ölü `TriangleMesh*` tutuyor (`forgetBodyMeshCaches`
   çağrılmamış) — bu UB, bazen "çalışır" görünür.
7. **Kayıt/açma.** `scripts/material_phase_tests/phase_10_fracture_roundtrip.py`
   (script değişmedi, Release kopyası aynı). PASS + yeniden açılışta shard adları ve
   cluster'lar aynı.
8. **★ Simülasyon ortasında kayıt (en sinsi).** 5'teki sahnede shard'lar havadayken
   (kare ~30) `project.save`, sonra `project.open`. Açılışta shard'lar **dinlenik
   (kesilmiş) pozda** olmalı, havadaki pozda değil; kayıttan sonra da canlı sahne
   havadaki pozda kalmalı.
   *Bozuksa:* açılışta shard'lar dağınık duruyorsa kayıt öncesi rest snapshot'ı flat
   mesh'i atlıyor — hata vermez, "kaydettiğim yer" gibi makul görünür. Bu
   add_primitive ile eklenen her flat rigid body için de geçerliydi (eski hata).
9. **Termal kırılma.** Yanmış bir objede `pattern=2` ile kır; panel mesajında
   "no recorded burn damage" yalnız gerçekten hasar yoksa çıkmalı; kırılma bölgesi
   yanık bölgeye yoğunlaşmalı.
10. **K0 tane örneği olarak shard.** (Diğer oturumun scripti; o oturum facade reddini
    kaldırınca) `rt_k0_grain_mesh_near_ipc.py --library fracture`: `exemplars` = shard
    sayısı, `exemplar_triangles` > 0, `status: active`.
    *Bozuksa:* `missing source` = `gatherNormalizedSceneFaces` shard'ı bulamıyor.


## 2026-10-22: prosedürel tane kütüphanesi ve DEM/MPM/fluid görünümü

Derlemeyi kullanıcı yapar; ajan build veya uygulama başlatmadı. Yeni
`UI/GrainMeshLibraryUI.cpp` proje/filters listelerine kaydedildi.

1. Üstteki materyal/K0 maddeleri 1–6'yı yeni executable ile tamamla.
2. Domain'i Splat/Particles görünümünde, geometry=icosphere olarak hazırla;
   birkaç carrier doğduktan sonra timeline'ı duraklat, kamerayı Near Distance
   içine getir. Ayrı terminalde:
   `python scripts/test/rt_grain_library_options_ipc.py --domain K0_Sand --render`
   Beklenen PASS: rock ve voronoi için near_instances>0; yanlış string/sayı/
   iç içe dizi sonrası IPC hâlâ cevap verir. Probe ayarları geri yükler.
3. 2'yi küçük **DEM, MPM ve fluid** sahnelerinde ayrı çalıştır. Virtual Grains
   8 iken mesh açık → primary_visual_children=1; kapalı → eski ayar geri
   gelmeli (MPM Physical Carriers açıksa primary=1 kalması doğrudur).
   Whitewater sayısı etkilenmemeli. Rendered RT ve Solid/Matcap görünümünde
   aynı carrier hem küre hem mesh olarak çift çizilmemeli.
4. Near Grain Meshes → Grain Shape: Procedural Rock / Rounded ve Procedural
   Voronoi / Angular kaynak obje seçmeden çalışmalı. Scene obje sayısı sabit
   kalmalı. Near Distance/bütçe dışında kalan carrier'lar küre kalmalı.
5. Scene objects seç; iki farklı materyalli kaynak (mümkünse bir kaynakta iki
   yüz materyali) ekle. Use Exemplar Face Materials açıkken yüz materyalleri
   ve UV'ler korunmalı; kapalıyken domain/substance/ıslak bantlar kullanılmalı.
   Kaynakta materyal atamasını değiştir: paused sahnede yakın mesh güncellenmeli.
   Uzak küreler bu seçenekten etkilenmez; yakın/uzak renk farkı bu modda beklenir.
6. Kaydet/aç: shape ve source_materials aynı kalmalı. Eski projede scene/false.
   K0 A/B ölç: near_restored=true; benchmark kendi shape/materyal seçimlerini
   geri yüklemeli. K0 fixture önce yeniden kurulmalı, önceki sahne kaydedilmemişti.
7. `scripts/test/grain_mesh_library_test.cpp` kaynak testi kullanıcı test
   derlemesinde çalıştırılmalı; ajan proje kuralı gereği derlemedi.
8. Fiziksel boyut dağılımı sonraki iştir. Önceki emitter durması kullanıcı
   tarafından sahne kurulumuna bağlandı; çözücü bug'ı olarak açık tutulmuyor.

Ajanın build gerektirmeyen kontrolleri: Python AST, JSON/XML okuma,
IPC descriptor üretimi/--check, ilgili dosyalarda diff whitespace kontrolü.
Canlı render, kayıt round-trip ve C++ kabul testleri çalıştırılmış sayılmaz.


## 2026-10-22: Add Mesh / Grains–Crystals üreticisi

Ayrıntılı sözleşme: `docs/dev/GRAIN_CRYSTAL_GENERATOR.md`.

1. Kullanıcı derlemesinde yeni `Scene/GrainGenerator.cpp`, `UI/GrainGeneratorUI.cpp`
   ve `Api/Rt{Api,Ipc,Python}GrainGenerator.cpp` modülleri alınmalı (vcxproj/filters
   kaydedildi; CMake source glob'ları kapsıyor). Ajan derlemedi/uygulamayı başlatmadı.
2. Add Mesh → Procedural Generator → Grains / Crystals: Rock/Sand, Voronoi,
   Snow Crystal ve Mineral Crystal. Seed, adet, biçim, çap aralığı ve ağırlıklı
   materyaller kullanıcı kontrolünde olmalı. Yeni kütüphane gerçek flat objeler olmalı.
3. Ayrı terminal: `python scripts/test/rt_grain_generator_ipc.py` → PASS.
   `--keep` geometri/materyal/kayıt incelemesi için örnekleri bırakır.
4. Mesh Grains → Create Grain / Crystal Library… aynı pencereyi doğru hedefle
   açmalı; Generate seçili domain'e kaynakları ve kendi materyallerini bağlamalı.
   Tek Undo/Redo objeleri ve önceki domain kaynak ayarlarını birlikte geri getirmeli.
5. Karı alt/üst/yandan incele: kapalı, kalınlıklı dallı plaka; uç/kenar delikleri
   ve üst üste iç dal yüzleri olmamalı. Salt: 4 kenar/0 uç; Quartz/Ice: 6 kenarlı
   prizma; Sugar: seçilebilir uzama/uç. Aspect ve seed sonuçta görülmeli.
6. Üretilen örnekleri içeren proje kopyasını kaydet/aç. İsimler, geometri,
   yerleşim, UV ve materyaller korunmalı. Küçük çaplı kar örnekleri de Mesh Grains
   kaynağı olmalı (SoA yüz okumasında mutlak minik-alan elemesi kaldırıldı).
7. Eski near_mesh_shape=rock/voronoi sahneleri açılmalı; yeni UI'de ayrı sabit
   üretici dropdown'ı görünmemeli. Scene object listesi tasarımı ertelendi.
8. Kullanıcı kaynak testlerinde `grain_generator_test.cpp` + mevcut
   `grain_mesh_library_test.cpp` çalıştırılmalı. Fiziksel boyut dağılımı hâlâ ayrı iş.

## Voronoi edge rounding acceptance (2026-10-22)

- User build: GrainGenerator.cpp and GrainGeneratorUI.cpp integration.
- Generate Voronoi with Examples=1: an irregular cut cell, not an uncut cube.
- Change seed: different geometry; repeat seed: deterministic geometry.
- Compare Roundness=0/0.6 with Edge Detail=6 under lighting and wireframe.
- Compare Edge Detail=2/8: real triangle density must change when rounded.
- Aspect=(0.7,1.6,1): verify smooth normals and closed surface.
- Repeat via Python and scene.generate_grains IPC with identical parameters.
- User compile/run scripts/test/grain_generator_test.cpp: new single-cell,
  seed, closed-surface, smooth-normal and detail regressions.
- Recheck Snow Crystal, materials, Undo/Redo and save/load.

Codex did not build or launch the application; visual acceptance remains pending.

## DEM physical size distribution (2026-10-22, user build/live checks pending)

- Build C++ and DEM / grain-fluid / grain-MPM shaders together (DEM revision 28).
- Build/run scripts/test/matter_grain_size_test.cpp and existing birth/stages tests.
- In a paused empty scene, external terminal: python scripts/test/rt_grain_size_ipc.py.
- UI: min radius .01 m, max .02 m, Sand + large stones, large fraction .10.
  Verify Sphere and Near Mesh agree, larger stone resting heights, impacts,
  sleep/wake, source spacing, reset replay, project save/load and cache scrub.
- Recheck uniform/log-uniform, max=0 legacy fixed size, invalid patches unchanged,
  and all five size fields locked while particles are live.
- Recheck wet/liquid/gas/thermal/MPM momentum/heat regressions for mixed sizes.
- Dense 1:2 and 1:4 piles: inspect explicit bucket/list/contact capacity holds;
  existing 16 bucket / 32 list / 24 history limits remain, no silent truncation.
- See docs/dev/DEM_SIZE_DISTRIBUTION.md for settings and physical semantics.

No project/shader compilation or application/live IPC verification performed by Codex.

## DEM size controls and visible refusal feedback follow-up

- Pending user build: size edits are staged together, applied atomically with
  Apply size settings; invalid ranges are shown beside controls; reset required
  blocks applying sizes while preserving the draft. Discard restores saved sizes.
- Large stone share now uses percent with small/large count labels and estimated
  volume/mass dominance; optional 95% sand / 5% stones preset. Physics RNG unchanged.
- Rejected settings and held solver steps display prominent orange/red wrapped text.
- Verify two radius edits do not apply individually; Apply/Discard; external IPC
  settings refresh the draft; pending size edits do not alter other solver changes.
- Live IPC diagnosis in Physics Domain 1: damping bound requested 529 substeps,
  cap 512. Raised cap to 1024; steps dt=1/60 and dt=1/24 succeeded, held=false.
  Same 1143 grains retained; center Y .828615 -> .808913 -> .747721 m.
  Timeline remained paused at frame 22; sizes and source unchanged, no reset/save.
- This verifies the current executable; pending UI source changes need a user build.

## DEM mixed-size contact pile / runtime neighbour lists (revision 29)

Live reproduction at paused frame 17, Physics Domain 1: minimum radius .026 m,
maximum .083 m, bimodal large fraction .07, seed 1360, max_substeps 1024.
One external dt=1/24 step held with "grain neighbour list full (>32 ...)" at
4289 grains. This is a separate capacity failure from the previous CFL limit.
No reset, size/source edits or save were performed during this reproduction.

Pending user build:
- Rebuild C++ and all sim_matter_grain shader variants together (revision 29).
- Fixed-size domains retain 32-list baseline; mixed sizes start with 128 slots.
- Standalone DEM retries list-only overflow with a doubled budgeted buffer from
  unchanged host state; shared MPM/liquid clocks refuse replay to avoid double steps.
- Allocation includes old+new peak memory and checks shader index width.
- IPC diagnostics: runtime.neighbour_list_capacity / neighbour_list_retries.
- With the failing pile open and paused, external terminal:
  python scripts/test/rt_grain_neighbour_capacity_ipc.py --steps 8
- Check actual floor contact, dense mixed pile, fixed-size baseline, storage/working
  budget failures, and shared-clock mixed cases. Actual contact history remains 24;
  this change removes the 32-candidate ceiling, not all other solver budgets.

Codex performed source checks only for revision 29; no compilation or live run of
new code. The open executable still has revision 28 and cannot use this fix yet.

## DEM contact history overflow at floor impact (revision 30)

Live read-only diagnosis: paused frame 29, Physics Domain 1, 94,915 grains,
minimum radius .011 m, maximum .036 m, large count 995 (1% authored), seed 1360,
max_substeps 2048. Status: contact count/history-in-flight exceeds legacy 24.
No reset, setting mutation, extra tick or save performed during this diagnosis.

Pending user build and acceptance:
- Rebuild C++ plus all sim_matter_grain shader variants together, revision 30.
- Mixed-size history starts at 64 records, fixed-size remains 24. Four 32-bit
  mask words support records across boundaries 31/32, 63/64 and 95/96 up to 127.
- Block metadata is now owner + four masks + rest + compensation (7 words),
  following the capacity-sized identity->block map. Sleep/audit offsets changed.
- Standalone DEM history overflow retries from unchanged canonical host state
  with capacity doubled to at most 128, charged to actual/peak memory budget.
- Shared MPM/liquid clocks cannot replay and continue to refuse overflow.
- Adaptive CFL retries use observed maximum contacts + 4, bounded by allocated
  history; shared clocks use the full allocated history bound.
- runtime.contact_history_capacity / contact_history_retries expose allocation.
- External terminal with the failed pile paused:
  python scripts/test/rt_grain_neighbour_capacity_ipc.py --steps 8
- Verify floor impact and sustained dense packing beyond 24 contacts, sleep/wake,
  changing neighbour identities, population reorder, GPU-resident fixed-size baseline,
  friction/rolling persistence, conservative memory refusal, and mixed-owner safety.
- This raises mixed-size memory use (64 records for every grain initially);
  it is not sparse per-grain contact storage and does not establish million-grain
  cinematic performance. At 128 records a million-grain history alone is ~3.34 GiB,
  potentially above a device's single-buffer limit. The checked budget must refuse it.

Revision-30 work received source checks only. No build/shader compilation or live
physics acceptance of the new code was performed by Codex.

## Size-aware contact CFL bound (host-only follow-up, shader ABI stays revision 30)

Live diagnosis at frame 26: .011/.036 m bimodal 1%, 86,462 grains, cap 2048,
held at damping bound 2081. Existing bound combined globally smallest mass with
largest radius stiffness. This pair does not describe a physical grain.

Source change: for each inertial mass mi and radius ri, bound incident stiffness
by alpha*max(ri,2*ri*Rmax/(ri+Rmax)); reduce maximum ki/mi across particles.
Then omega^2 <= 2*contact_budget*max(ki/mi), retaining conservative damping and
accuracy algebra. Wet bridge bound remains conservative. Mixed-size rolling
uses the conservative unequal-radius rotational factor. Equal dry sizes retain
the old normal bound. No stiffness, restitution, size or substep cap is modified.

Validation:
- Analytical wall/pair inequalities checked for .011/.016/.026/.036 m; new
  normal-rate bound is .468085 of the old fictitious min-mass/max-k bound.
- Added C++ regression for unequal pairs, walls and equal-size parity in
  scripts/test/matter_grain_size_test.cpp (not compiled/run by Codex).
- Live revision-30 executable, unchanged authored settings: 12 external ticks
  at dt=1/48 succeeded, 36..42 peak contacts, history 64, neighbour list 128,
  no hold. This validates shorter dt under old bound, NOT the new host code.
- User rebuild C++, no shader ABI change. Repeat original dt=1/24 dense impact
  and sustained pile, inspect cfl_stiffness_mass_rate_s2 and substeps.
- Repeat fixed-size convergence, high rolling friction and wet/shared-clock
  cases. Lower substep count is accepted only with stable/converged physics.
- No reset, material/size edits, substep cap changes or scene save by Codex;
  12 manual half ticks advanced the live simulation, timeline left paused.

## Explicit reset clears stale DEM CFL hint

Live check confirmed contact_resolution=8 was applied; a NEW dt=1/24 step still
held on damping (2129 > 2048), not accuracy. Lowering accuracy cannot remove a
damping limit. The new size-aware bound remains pending a user C++ build.

Source audit additionally found resetGridDomainStates cleared host particles but
only invalidated generic field copies; retained matter_runtime->grain kept the
previous dense pile's cfl_contact_hint, publication identities and sleep context.
Added invalidateMatterGrainAfterReset in the focused DEM module and minimal calls
in both explicit reset buffer loops. Reusable allocations/growth requests remain;
CFL hint resets to baseline, history/residency/sleep/motion publication is cold.
Do NOT call this on per-step invalidateDeviceCopies (would erase valid history).

User build and regression:
- C++ rebuild only, shader ABI stays revision 30.
- After a dense pile/error, reset to sparse births and verify first-step
  cfl_contacts_per_grain starts at baseline instead of the old peak.
- Test fixed/mixed sizes, empty reset, repeated reset, and births restarting IDs.
- C++ reset-state regression added to matter_grain_size_test.cpp, not executed.
- Source checks passed. No reset or authored setting changes made in the live
  scene by Codex in this follow-up; one failed probe tick was attempted.

## Live close-out 2026-10-22: DEM continuation and MPM/DEM fixture

User-built executable: existing 100,000-grain DEM scene completed six dt=1/24
steps without holds, 1162..1260 substeps, max contacts 33, mean step_total 1358 ms.
This closes the reported short continuation failure, not long-run/scale acceptance.

Created MPM Sand + DEM Stones in the same scene. Previous domain/source retained
but disabled; editing setup reset previous live particles. New fixture completed
86 dt=1/120 ticks with 12,288 MPM carriers and 10 DEM stones, nonzero bidirectional
contact events and sampled momentum residual ~1e-8 N*s. It remains paused at frame 0.
Actual penetration/support/convergence acceptance remains next work.

Details: docs/dev/DEM_MPM_ACCEPTANCE_20261022.md
Raw measurements and previous configuration: docs/dev/dem_mpm_acceptance_20261022.json
No compilation or application launch by Codex; live tests used external named pipe.

## Seed material UX follow-up

- User C++ build: existing MatterModelControls.cpp now provides the seed panel's
  shared domain-default substance picker and granular MPM/DEM/fluid model label.
- Minimal integration call in scene_ui_simulation_domains.cpp; no feature body
  added to the over-2000-line panel file. No shader changes.
- Verify Seed/Matter pickers remain synchronized; liquid/MPM/DEM labels match
  substance.get; errors are visible; reset+seed and save/load preserve material.
- UI, Python and IPC all use existing updateFluidDomain; no UI-only material state.
- Current live bed is granular MPM, NOT fluid (0 fluid / 12288 MPM / 10 DEM).
  Fluid-like appearance is not dismissed; pile/slope/penetration acceptance remains.
- Source checks passed, no compilation and no live acceptance of the new UI yet.


### DEM seed ownership and physical packing (2026-10-22)

- First Sand seed regression: resolve substance ownership and material before
  birth, after replacement clears old particles. Previously a first seed could
  become MPM and lock that owner until reset.
- UI, Python `rt.fluid.seed`, IPC `fluid.seed` share this core. DEM seeds use
  sphere-volume mass and a maximum-radius lattice independent of voxel/PPC.
  Variable radii keep deterministic IDs; smaller grains begin with gaps and settle.
  This is conservative initial packing, not dense equilibrium packing.
- DEM requested count is a lower bound when truncated (emitted + 1). Candidate
  work is bounded; capacity/scan cutoff has a separate DEM warning.
- User: C++ build only, no shader changes. Codex did not compile or run C++ tests.
- Run updated `scripts/test/matter_grain_birth_test.cpp`: physical spacing/mass,
  granular model, voxel independence, append collision rejection, budget, small box.
- Live: empty domain, builtin Sand, seed replace=true BEFORE any step.
  `fluid.matter_models`: DEM > 0, MPM = 0, reset_pending=false. Repeat after
  reset/reseed. Replace existing incorrectly born MPM particles.
- Identical box/radius with different voxel/PPC must give identical DEM count.
  Custom granular_transport=mpm and Water retain voxel/PPC sampling and owners.
  MPM default + DEM source must seed MPM, even with domain grains enabled.
- Drop Sand above floor to test impact scattering. A zero-velocity resting seed
  is not a bounce test. Restitution and friction were not changed.


### DEM seed column symmetry regression (2026-10-22)

Live read-only evidence after the ownership fix: Physics Domain 1 has 34,512
DEM grains, zero MPM/fluid owners, reset_pending=false, dry grains, zero liquid
bridges, restitution=0.5. Horizontal bounds remain the seed footprint; horizontal
momentum and spin energy are zero, maximum contacts per grain=2. This matches
perfectly aligned columns compressing/rebounding without lateral excitation,
not an adhesive material or MPM ownership fallback.

- Seed now uses deterministic bounded position jitter (10% of maximum radius
  per coordinate) with spacing 2.2002 R. The worst adjacent jitter extremes
  retain at least 2.0002 R separation along their separating axis. Bounds include
  the jitter margin. No velocity kicks or changes to material forces.
- Existing size_seed controls reproducibility through the same UI/script/IPC
  core. Seed positions remain independent of voxel/PPC and particle budget.
- Initial count is lower than the old perfectly touching lattice; this remains
  conservative packing, not a dense packing generator. The box needs room for
  the maximum diameter plus its jitter margin.
- Updated C++ regression assertions cover pair separation, containment, layer
  misalignment, seed repeatability/change, append rejection, budget-prefix stability.
  Codex did not compile/run C++ tests or launch the application.
- User build: ordinary C++, no shader change. Clear/replace and seed again: old
  live positions cannot acquire the new birth jitter. Run with settle-on-seed
  disabled for an observable drop. Verify horizontal spreading and nonzero spin
  after impact, no coherent column bounce, no held step or capacity overflow.
  Repeat with settle enabled and with radius variation; check a final stable pile.
