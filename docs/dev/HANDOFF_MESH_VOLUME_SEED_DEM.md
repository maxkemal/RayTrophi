# Devir: Mesh Volume seed + DEM kum "step held" + açık kalanlar

> **Durum:** AKTİF — 2026-11-05 devir notu. Mesh Volume seed CANLI. DEM komşuluk kovası taşması kökü
> BULUNMADI. 2026-11-07 (§9): fışkırmanın kökü keyframe'li kürenin tabana ezdiği kum + DEM mesh
> temasının iki yüzlü olması (tane kapalı cismin içine geçiyor); IPC'de bozuk float parametresi
> uygulamayı abort ediyor (`ExceptionHandling=false`).

## 1. Bitenler (canlı doğrulandı)

- **Mesh Volume seed** (kum kale: gizli bir mesh'in içini madde ile doldur).
  - `include/Fluid/MeshVolumeTest.h`: `ColliderMeshBVH` ve üç ışınlı parite. Collider SDF
    pişirmesiyle aynı içeride kuralı.
  - `MeshVolumeCellMask`: hücreler bir kez sınıflanır; nokta testi yalnız sınır hücrelerinde.
  - `FluidSeedMode::MeshVolume`, `fluid_seed_mesh_object`.
  - `seedBox` ve `seedMatterGrainBox` maske alıyor.
  - IPC `fluid.seed_mesh`, Python `rt.fluid.seed_mesh`; panelde Seed Object seçicisi.
  - Test `scripts/test/rt_test_seed_mesh_volume_ipc.py`: 4/4 PASS. Küre 33 086 parçacık, beklenen
    33 510. Gizli nesneyle de çalışıyor.
  - Taşınan nesne düzeltmesi (önbellek anahtarına poz matrisi ve sınırlar girdi): kullanıcı canlıda
    doğruladı.
- **İlk Play'de tek seferlik seed'in silinmesi:** artık olmuyor (kullanıcı doğruladı).
  - `adoptSeedAsSimulationStart`, seed'i 0. kare olarak cache'e alıyor.
  - Tek seferlik "seed-start drift guard" (`scene_data.h`, `updateSimulationTimeline`), ilk
    adımdan sonraki imza kaymasını yutuyor.

## 2. Açık iş A — DEM kum adımı bekletiliyor (ÖNCELİK)

**Ölçülen** (kullanıcı sahnesi; kum + collider, 74 439 DEM tanesi, `mpm_law: none`, voxel 0,1,
50³). `fluid.step_stats.gpu_status` her adımda şunu söylüyor:

> "Mixed step held: grain neighbour bucket overflow (more than 32 grains hashed to one bucket);
> reduce packing or the physical size ratio"

- Alt adım ya da bütçe değil: `granular_required_substeps = 1`. Kullanıcı alt adımı ve bütçeyi
  2048'e çıkardı, değişmedi.
- Kullanıcı DEM min yarıçapı 1'e çıkardı: yine bekletildi. Bekletmeye kadar olan adımlar da çok
  pahalıydı (ölçümde `total_ms` ~1490, `dispatch_call_ms` ~847, `synchronize_ms` ~532).
- MPM sıvı aynı sahnede çok hızlı.
- Seed taneleri temas halinde doğuruyor (FCC, doluluk ~0,653, `MatterGrainBirth.cpp`). Dökülen
  kumda görünmeyen yoğunluk, mesh/kutu seed'inde ilk kez tüm hacimde oluşuyor.

**Yapılacak:**
1. Mesajı üreten kodu bul (`grep "bucket overflow"`). Kova hücre boyutunun neyden geldiğini oku
   (voxel mı, sabit mi, max yarıçap mı). Min yarıçap değişikliğinin etkisiz olması kovanın min
   yarıçaptan gelmediğini gösteriyor.
2. Kova hücresini tane çapından (2·R_max) türet, ya da kapasiteyi yoğunluğa göre büyüt. 32 sabit
   kapasite.
3. Pahalılık ayrı ölç: `dispatch_calls` 837/adım. Zincir mi, kova taraması mı?
4. **Panel kör noktası:** "step held" nedeni panelde görünmüyor. Kullanıcı bunu alt adım sorunu
   sandı. `gpu_status`'u panelde göster; kullanıcı ayrıca daha adaptif alt adım/bütçe istiyor.

### 2026-11-05 — Açık sahnede IPC profil ölçümü

- Kullanıcının açık sahnesi, dış `scripts/test/rt_ipc.py` istemcisi. Sahne yeniden
  yüklenmedi/resetlenmedi; ayarlar değiştirilmedi. İki `fluid.step(dt=1/24)` denemesi
  yapıldı: ilki başarılı, ikincisi taşma nedeniyle reddedildi. Playhead 13 -> 15;
  uygulama duraklatılmış ve manuel script sürücüsünde bırakıldı. GPU profil ölçümü kapatıldı.
- `wet_grains=false`, seed saturation 0; 74 439 tanenin tamamı kuru. Etkin fiziksel
  yarıçap 0,025 m, maksimum da 0,025 m. Önceki min yarıçap denemesinin aksine şu an
  çalışan sahnenin değeri 1 değil. `max_substeps=2048`, `contact_resolution=24`,
  `stiffness_scale=1`, temas sertliği 20 000 N/m.
- Hücre boyutu `MatterGrainGpu.cpp` içinde `2*r_min + rupture_cap + 0.5*r_min`;
  kuru sabit boyutta 0,0625 m. Voxel 0,1 m bu hesabın girdisi değil.
- `sim_matter_grain.glsl::bucket` fiziksel hücre koordinatlarını hash'liyor.
  `insert` aynı hash yuvasına düşen **bütün hücreleri** ortak 32 girişe yazıyor.
  Dolayısıyla taşma tek bir fiziksel hücrenin aşırı dolduğunu kanıtlamıyor.
  Fiziksel yoğunluk ile hash çakışmasının payı henüz ölçülmedi. Hücreyi doğrudan
  `2*R_max` ölçeğinde büyütmek bu sabit boyutlu sahnede kök çözüm değil.
- Panel/API'deki `granular_required_substeps=1`, DEM temas alt adımı değil.
  Gerçek sayaç `fluid.matter_models.grain_diagnostics.cfl_terms` içinde:
  başarılı adımda 200, reddedilen sonraki adımda 236. Sınırlayıcı kararlılık adımı
  sırasıyla 0,00021015 ve 0,00017702 s; varsayılan temas sayısı 14 -> 17.
  Tavanı artırmak bunu hızlandırmaz; tavan zaten devrede değil.
- Başarılı adım: toplam 565,68 ms; GPU timestamp toplamı 443,00 ms.
  Temas/integrasyon 338,49 ms; liste kurma 98,09 ms; hash 4,29 ms;
  liste temizleme 2,03 ms. Her aşama 200 kez çağrıldı.
- Reddedilen adım: toplam 2017,36 ms; GPU timestamp toplamı 1893,34 ms.
  Temas/integrasyon 1817,31 ms; liste kurma 70,38 ms; hash 3,79 ms;
  liste temizleme 1,76 ms. Her aşama 236 kez çağrıldı.
  Bunlar iki farklı ardışık durumdur; kontrollü bir optimizasyon A/B ölçümü değildir.
- `MatterGrainStages.h`: her DEM alt adımı dört dispatch üretir. Liste yeniden
  kurulmasa bile ilk üç kernel gönderilir, shader içinde erken döner.
  `SimulationComputeVulkan.cpp`: 512 dispatch'te `synchronize()` çağrılır.
  Bu yüzden `dispatch_call_ms` salt CPU kayıt süresi değildir; ara GPU beklemesini
  de içerir. Başlangıçtaki 805 çağrı = 200*4 DEM + 5 yardımcı çağrı.
- Taşma bayrağı GPU'da yükseldikten sonra kalan temas alt adımları yine çalışıyor;
  host yalnız bütün zincirin readback'inden sonra sonucu reddediyor. Reddedilen
  adımın temas kernel'i maliyeti baskın; taşmayı hash kernel'inin doğrudan süresi
  olarak yorumlamamak gerekir. Taşmanın ilk oluştuğu alt adım henüz ölçülmedi.
- Başarısız adım host konum/hızını yayımlamadı: başarılı adım sonrası ve hata sonrası
  hash'ler aynı (`bfb1468a953300a2`, `64f8c6f03d45607f`).
- Sonraki uygulama işi: hücre doluluğu/hash çakışması/ilk taşan alt adım ölçümleri;
  komşu kaybetmeyen taşma depolaması veya güvenli yeniden deneme; fatal bayrak sonrası
  gereksiz solve işini durdurma. Ardından aynı durumdan kernel profiliyle kıyasla.
  Collider katkısı bu ölçümde temas kernel'inden ayrıca ayrıştırılmadı.

### Collider teması için yeni ölçüm — kaynak hazır, canlı doğrulama bekliyor

Kullanıcı collider teması yokken simülasyonun devam ettiğini bildirdi. Bu, önce
temas kaynaklı sıkışmayı/kararsızlığı hash çakışmasından ayırmayı gerektiriyor.
Kapasite veya kuvvet yasası değiştirilmedi; taşma hatası henüz çözülmüş değil.

- `fluid.set_grain_settings(domain=..., contact_probe_enabled=True)` ile açılır;
  varsayılan kapalıdır. Yalnız bu anahtar değişince timeline cache temizlenmez.
  JSON boolean gerekir; sayı/string reddedilir, diğer ayarlar değiştirilmez.
- Python `rt.fluid.set_grain_settings(...)` ve IPC aynı ayar servisini kullanır.
  Sonuç her iki kanalda `fluid.matter_models(...).grain_diagnostics.contact_probe`.
  Başarısız adımın raporu da korunur; `runtime` null olsa bile probe okunabilir.
- `measured=false`: bu denemeden geçerli ölçüm yok (kapalı, kernel çalışmadı,
  uyumsuz shader veya başarısız readback dahil). Sonuç nonfinite ise readback
  başarılı olduğu sürece probe korunur, `output_finite=false` olur; nonfinite
  maksimumlar JSON'da null olabilir. Hata mesajı readback ve nonfinite'i ayırır.
- `first_bucket_overflow_substep`: sıfır tabanlı; -1 taşma yok.
  `overflow_grain_index`: GPU DEM alt kümesindeki indeks; kalıcı parçacık ID'si
  veya domain'in host indeksi değildir. Aynı alt adımda birden çok taşma olursa
  atomik yarışı kazanan bir örnek seçilir; en yoğun hücre olduğu iddia edilmez.
- `overflow_bucket_population`: ilk taşan hash yuvasının hash aşaması sonundaki
  tüm nüfusu. `overflow_cell_population`: seçilen tanenin fiziksel hücresindeki
  **tam** nüfus; atılan taneler de taranır. İlki >32, ikincisi <=32 ise seçilen
  fiziksel hücre tek başına kapasiteyi aşmamıştır; farklı hücreler kovayı paylaşır.
- `peak_bucket_population`: denemenin tamamı; taşma sonrası durumları da içerir.
- `max_grain_overlap_fraction`, `max_collider_overlap_fraction`, `max_speed_m_s`,
  `max_acceleration_m_s2`, `max_collider_contacts`: ilk taşan hash'ten **önceki**
  aktif temas çözümlerinin maksimumları. Tane örtüşmesi merkezler arası mesafenin
  yarıçap toplamından eksik kısmı / yarıçap toplamı; collider örtüşmesi / tane
  yarıçapı. Collider sayısı tutulan mesh manifold destekleri (en çok 4); domain
  duvarları dahil değil. İlk hash'te taşarsa `contact_metrics_measured=false`;
  sıfır metrikler temas olmadığının kanıtı değildir. Sleeping erken dönüşleri
  bu maksimumlara katkı vermez. Taşmadan sonra eksik listelerle hesaplanan
  kuvvet/hızlar kök neden kanıtı olarak kaydedilmez.
- Ölçüm ek atomik maksimumlar ve ilk taşmada bir tam tane taraması kullanır;
  açık ölçümdeki süreyi normal performans karşılaştırması olarak kullanma.
  Aynı sahnenin collider açık/kapalı koşullarını başlangıçtan ayrı ayrı ölç;
  ardışık farklı fiziksel durumlar kontrollü A/B sayılmaz.

**Kullanıcının build/doğrulama adımları (ajan çalıştırmadı):**

1. C++ uygulamasını ve dört DEM shader'ını birlikte yeniden derle:
   `sim_matter_grain_list_clear`, `sim_matter_grain_hash`,
   `sim_matter_grain_list_build`, `sim_matter_grain_step`.
   Güncel shader revision **38**, ortak diagnostics/audio header **47 word**.
   Yeni `sim_matter_grain_active_classify` dahil artık beş DEM shader'ı var.
   Mevcut `RayTrophiStudio/source/shaders/compile_sim_shaders.bat` bunları kapsar;
   executable'ın kullandığı shader dizinine güncel SPIR-V çıktıları gitmeli.
2. Mevcut `matter_grain_params_test.cpp` testini çalıştır: boolean doğrulama,
   ayar round-trip'i ve runtime null iken başarısız adım raporu için kontroller eklendi.
3. Yeni executable ile sahneyi aç, collider temasına yakın durumda duraklat.
   Ayrı terminalden, repo kökünde:
   `python scripts/test/probe_dem_collider_ipc.py --domain "Physics Domain 1" --steps 4`
   Script reset/seed/collider değişikliği yapmaz. Playhead'i adımlatır ve ilk held
   denemede durur; sonunda önceki probe anahtarını geri yükler. Manuel script
   sürücüsü Play/scrub yapılana kadar kalır. Çıktı JSON satırlarıdır.
4. Özellikle ilk taşmadaki iki nüfus değerini ve önceki temas örtüşmesi/hızını
   karşılaştır. Kök neden kararı bu ölçümler alındıktan sonra verilecek.

### Yeni sahne, 8. kare ölçümü

Kullanıcı revision 34'ü derledi. Dış IPC probe ile kare 1..8 ölçüldü:

- Kare 1..5: serbest düşüş, ivme 9,81 m/s², mesh collider teması 0,
  tane örtüşmesi 0, en büyük kova nüfusu 10.
- Kare 6: tane örtüşmesi 0,06890, en büyük ivme 966,34 m/s²; mesh teması 0,
  kova nüfusu 10. Kare 7: örtüşme 0,10022, ivme 1032,91 m/s²; mesh teması 0,
  kova nüfusu 14. Domain duvarları mesh temas sayacına dahil değildir.
- Kare 8: `grain publication rejected nonfinite/readback state`, 1635,04 ms,
  186 DEM alt adımı. Host konum/hız hash'leri değişmedi. Bu denemenin son hata
  metni bucket overflow değildi. Eski hata metni readback ile nonfinite'i tek
  başlıkta verdiğinden hangisi olduğu bu executable'ın API verisinden kesin değil.
- Probe yanlışlıkla nonfinite durumda da `measured=false` oluyordu. C++ readback
  başarısı ile sonuç finiteness kontrolü ayrıldı; başarısız fizik sonucu artık
  probe'u gizlemiyor. IPC script de held durumda raporu yazıp duruyor; ölçüm
  yok diye ikincil assertion ile asıl hatayı gölgelemiyor.
- Bu son düzeltme C++ tarafında; shader revision 34/header 28 değişmedi.
  Kullanıcının C++ rebuild'inden sonra aynı sahnede probe yeniden okunmalı.
  Domain sınırları y=[-2,26196; 2,73804], 74 439 tane, r=0,025 m,
  max_substeps=512. Reset/seed veya collider ayarı değişikliği yapılmadı;
  ölçüm adımları playhead'i 8'e taşıdı. Probe kapatıldı, sahne duraklatılmış
  ve manuel script sürücüsünde kaldı.

### 2026-11-06 — Readback değil, GPU fizik sonucu nonfinite

Kullanıcı C++ rebuild yaptı ve Seed Now ile aynı sahneyi doldurdu. Probe kare
1..8 çalıştırıldı; 8. karede durdu, anahtar tekrar kapatıldı:

- Kare 6/7: mesh temas sayısı 0; tane örtüşmesi 0,06879 / 0,09999,
  ivme 965,92 / 1032,31 m/s². Önceki denemeyle aynı gelişim.
- Kare 8: `grain publication rejected nonfinite position/velocity/affine state`.
  Readback başarılı, `output_finite=false`, `overflow_bits=0`, ilk kova taşması
  -1, tepe kova nüfusu 10. Bu denemenin durma nedeni bucket overflow değil.
  186 DEM alt adımı, toplam 1302,16 ms; açık probe maliyeti dahil.
- Hatalı adımda mesh temas sayısı 4, collider örtüşme maksimumu 0; hız/ivme
  maksimumları null (nonfinite), tane örtüşmesi 0,13561. Bu tutarsızlık NaN'in
  collider manifolduna girdiği ihtimalini gösteriyor; ilk kaynağı kanıtlamıyor.
- Sahnenin tek collider'ı `Sphere_1 Collider`, kaynak `Sphere_1`, mesh_sdf;
  etkin, fluid collision açık. CPU BVH kurucusu nonfinite/dejenere yüzleri
  reddediyor. Bu yüzden dejenere mesh kök neden ilan edilmedi.
- Host parçacık hash'leri held adımda değişmedi, sahne kare 8'de duraklatılmış.

**Revision 35 kaynak değişikliği (henüz build/canlı doğrulama yapılmadı):**

- Sonlu olmayan collider closest-point/normal, alt adım girdisi, temas
  kuvveti/torku ve integrasyon sonucu ayrı ayrı denetleniyor.
- `contact_probe.failure_stages` IPC/Python ortak çıktısı: `collider_geometry`
  (bit 32), `substep_input` (64), `contact_force` (128), `integration` (256).
  `overflow_bits` artık bu geçersiz durum bitlerini de taşır.
- Geçersiz collider sonucu manifold/kuvvet hesabına alınmıyor. Geçersiz giriş
  veya kuvvet halinde tane o alt adımın girişini taşır; tüm attempt yine host
  tarafından reddedilir. Hata sonrası hash/liste/temas işinin kalan kısmı
  atlanır, böylece NaN'in tüm yığına yayılması ve boşuna solve maliyeti önlenir.
  Geçerli temas kuvvet yasası değiştirilmedi.
- Bu aşama kök fizik hatasının çözümü olarak sunulmamalı: yeni açık hata
  aşaması sonraki ölçümde hangi hesap yolunun inceleneceğini belirleyecek.
- Kullanıcı C++ ve dört DEM shader'ını birlikte yeniden derlemeli (revision 35,
  header 28). Sonra açık/seed edilmiş sahnede aynı probe çalıştırılacak.

### 2026-11-06 — Revision 35 sonucu ve closest-point düzeltme adayı

- Kullanıcı revision 35'i derledi, kum hazırken dış probe 1..8 çalıştırıldı.
  Kare 8: yalnız `failure_stages=["collider_geometry"]`, bit 32. Kova taşması,
  geçersiz alt adım girdisi, temas kuvveti veya integrasyon biti yok.
  `output_finite=true`, host durum değişmedi. Böylece ilk yakalanan hata
  DEM mesh collider closest-point/normal yoluna daraltıldı.
- Hatalı adım 125,22 ms; önceki revision 34 denemesi 1302,16 ms idi.
  Bu bir başarılı fizik hızlandırması değil: revision 35 ilk geçersiz değerden
  sonra geri kalan gereksiz işi atlayıp adımı reddediyor.
- Kullanıcı MPM/fluid'de aynı sorunu görmediğini bildirdi. İlgili fonksiyon
  DEM'e ait `sim_matter_grain.glsl` içinde; MPM/fluid collider hesabı değiştirilmedi.
- Revision 36 kaynak adayı: barycentrik dot-product determinantlarına bölünen
  closest-point yerine normal düzleme izdüşüm + yönlü kenar testleri kullanılır.
  Dış noktalar en yakın kapalı kenara gider; sıfır uzunluklu kenar uç noktasına
  indirgenir. İnce yüzlerde birbirine çok yakın determinantların çıkarılması
  ve denetimsiz bölünmesi kaldırıldı. Bu aday henüz GPU'da doğrulanmadı;
  sahnenin hatalı yüzü/koordinatları yakalanmadığından sayısal mekanizma kesin
  kanıt olarak sunulmamalı.
- Revision 36, hata ayrımını da netleştirir: bit 32 `collider_closest_point`,
  bit 512 `collider_normal`; diğer aşama bitleri aynı. Header 28 değişmedi.
- Kullanıcı C++ + dört DEM shader'ını birlikte derlemeli. Kum hazır ve sahne
  duraklatılmışken repo kökünde dış terminal komutu:
  `python scripts/test/probe_dem_collider_ipc.py --steps 12 --require-success`
  Başarı ölçütü 8. kareyi geçip 12 denemenin tamamını held/nonfinite olmadan
  tamamlaması. Daha uzun yığılma ve eski bucket overflow sahnesi ayrıca denenmeli.

## 3. Açık iş B — Islak kum çabuk dağılıyor

> Revision 36 kullanıcı tarafından çalışır doğrulandı. Sonraki DEM performans
> ölçümü: [DEM_PERFORMANCE_PROFILE_2026_11_06.md](DEM_PERFORMANCE_PROFILE_2026_11_06.md).
> Aktif iş listesi kaynak uygulaması ve build/kabul listesi:
> [DEM_ACTIVE_WORK_QUEUE.md](DEM_ACTIVE_WORK_QUEUE.md), revision 38; canlı kabul bekliyor.
> Sonraki Snow/high-grid Fog ve yaklaşık 9 GB VRAM incelemesi:
> [SNOW_FOG_MEMORY_2026_11_06.md](SNOW_FOG_MEMORY_2026_11_06.md).

- Kullanıcı ıslaklığı tam yaptı ama kutu/küre seed'i hemen yığıldı.
- "Seed Grain Wetness" (`fluid_seed_grain_saturation`) **yalnız DEM** için geçerli; MPM parçacıkları
  yok sayar (alan yorumu).
- Ölçülen domain'de `granular_cohesion = 0.0`.
- Geçici öneri (kullanıcıya verildi): kum kale için MPM granular kullan ve `granular_cohesion` > 0
  yap.
- DEM ıslak kum A çözülünce yeniden denenmeli. Hâlâ dağılıyorsa sıvı köprüsü kuvvetine bakılır.

## 4. Açık iş C — İmza kaymasının kökü (korumayı sökmek için)

- Projeyi aç, Seed, Play; sonra `sim.control_state.rewind_history`'yi oku (son 8 kayıt).
  - `absorbed: true` satırının `first_changed_part`'ı, kaymanın bölümünü verir:
    `fluid 'X' thermal` ya da `fluid 'X' solver/seed`.
  - O bölümdeki türetilmiş (runtime'ın geri yazdığı) alanı imzadan çıkar, sonra
    `seed_start_drift_guard_until_frame_` korumasını sök.
- Satır yoksa koruma doğrudan sökülebilir.
- Ölçüm: yüklemeden sonra imza `10057163858437137944` idi, ilk adımdan sonra
  `17420775324187008440` oldu.

## 5. Küçük işler

- Seed'den sonra yanıltıcı "Sim changed - cache reset" uyarısı çıkıyor.
  `adoptSeedAsSimulationStart`, `sim_rewind_request_`'i ödünç alıyor; ayrı bir istek ya da mesaj
  gerekli.
- **Emitter `MeshVolume` + inflow/outflow + PPC** (kullanıcı istedi, yazılmadı).
  - Aynı `MeshVolumeTest` kullanılacak.
  - Inflow: hacimde PPC'nin altındaki hücreleri tamamla. Tanelerde `MatterGrainBirthFilter`
    zaten tamamlar.
  - Outflow: içeridekileri sil.
  - Emitter'larda saniyede parçacık yerine PPC; `MatterDomainSources.inl`'deki aşırı paketleme
    koruması bunun yaması.

## 6. Aynı oturumun diğer çıktıları (bağlam)

- GPU yüzey level set (Y1+Y2) canlı. Ayrıntı: `GPU_YUZEY_LEVEL_SET.md`.
- RayFusion madde materyali düzeltildi: etiketsiz parçacık artık Default Substance bağlamasını
  alıyor (`collectSubstanceMaterialEntries`).
- Splat materyali ve sim reset düzeltmeleri.
- Kontrol listesi: `NEXT_BUILD_CHECKS.md`.

## 7. Kullanıcı gözlemi (2026-11-05, devirden sonra) — A'yı daraltır

Kullanıcı: **collider teması yokken sim devam ediyor; kova taşması collider'a değince başlıyor.**
Yani taşma seed yoğunluğundan değil, tanelerin collider yüzeyine **yığılmasından** geliyor
(temasta sıkışma, yüzeye itilen taneler aynı kovaya toplanıyor). A için ilk bakılacak yerler:
- collider temas/kurtarma itmesinin DEM tanelerini yüzeyde tek bir kabuğa (aynı hash hücresine)
  sıkıştırıp sıkıştırmadığı (`mpm_surface_pushed_particles` / `recovered_solid_particles`
  benzeri DEM sayaçları; `fluid.step_stats`);
- collider SDF bandındaki taneler için komşuluk hash'inin hücre boyutu (yüzey yakınında yoğunluk
  doğal olarak yüksek; 32 kapasite orada aşılıyor);
- doğrulama: aynı seed, collider kapalı → bekletme yok; collider açık → temasla birlikte
  `gpu_status` "bucket overflow". Taşma anındaki kare ve kova konumu (collider yüzeyinde mi)
  ölçülmeli.

## 8. Canlı ölçüm (2026-11-07, kullanıcı sahnesi, kod değişmedi)

Sahne: 74 439 DEM tanesi (r 0,025 m, k 20 000 N/m), gizli `Default_Cube` (y −1,95…0,05) içine
Mesh Volume seed, domain tabanı −2,26 (küp tabanının **0,31 m altında**), collider `Sphere_1`
y 0,25…2,25. `fluid.run` dt 1/24, 37 adım.

- **Bekletme yok** (`gpu_status` "Dry grain Vulkan DEM candidate"), 236 alt adım, sınır `damping`,
  medyan adım 215 ms (devirdeki ~1490 ms yerine). ★ **Ama kum collider'a hiç değmedi** (kumun
  tepesi y −0,44, küre 0,25'te başlıyor): §7'deki "taşma collider temasıyla başlıyor" koşulu bu
  koşuda oluşmadı. Bu sonuç A'nın kapandığını KANITLAMAZ.
- Köşelerdeki 5–6 m/s taneler hata değil: seed'in son kimlikleri (üst katman), küp köşelerinden
  çapraz devrilip havada düşüyorlar; bir adım sonra yere inip listeden çıkıyorlar.
- ★ **Sinsi olan:** ~1,1 s'de yığının tepesinde ortadan bir fışkırma: en hızlı 64 tanenin tamamı
  x,z ≈ 0'da, +11…12 m/s **yukarı** (12 m/s için 7,3 m düşüş gerekir, sahnedeki en büyük düşüş
  2,3 m). Aynı anda **bütün yığının ağırlık merkezi 5 adımda +4,9 cm yükseliyor**
  (−2,0296 → −1,9807), sonra geri düşüyor: yığın trambolin gibi zıplıyor.
  Enerji: `E/kg ≥ g·y_c + ½·mean|v|²` alt sınırı bu sırada +0,16 J/kg artıyor, ama mean|v|² gerçek
  ⟨v²⟩'yi eksik sayar (hızlı kuyruk); enerji üretimi **kanıtlanmadı**.
  Aday (ölçülmedi): "yığılmış bilye" etkisi. 2 m'lik sütun 0,31 m düşüp tabana 2,5 m/s çarpıyor;
  basınç dalgası serbest üst yüzeyde en üst taneleri fırlatıyor. Doğum neredeyse kristal
  (`MatterGrainBirth.cpp` jitter 0,03 R): kuvvet zincirleri hizalı, dalga dağılmadan taşınıyor;
  yumuşak yaylar (gerçek kumdan ~100× yumuşak) sıkışmayı büyütüyor.
  Ayırma deneyi (derlemesiz): (1) seed kutusu tabana oturtulmuş, düşüş yok → fışkırma kalıyor mu;
  (2) `stiffness_scale` 10 → tepe hızı ve ağırlık merkezi sıçraması küçülüyor mu.
  Düzeltme adayı ancak (1) sonrası: doğum jitter'ı / düzensizliği.
- IPC boşluğu: DEM toplam kinetik enerjisi (öteleme) okunamıyor; `grain_diagnostics` yalnız
  `spin_energy_j` veriyor. Enerji kapısı için ⟨½mv²⟩ ve potansiyel enerji okuması eklenmeli.

## 9. Ayırma deneyleri ve KÖK (2026-11-07, aynı gün, kod değişmedi) — §8'in adayı ÇÜRÜDÜ

★ **Fışkırmanın sebebi animasyonlu collider.** `Sphere_1`'de 0. ve 20. karede dönüşüm
keyframe'i var (`anim.list_keys`). Küre ease-in/out ile y 1,252 → −1,417 iniyor (tepe hız
10–11. karede 4,8 m/s) ve **alt yüzü domain tabanının 15,5 cm altında** duruyor: kumu tabana
ezen, kaçış yolu olmayan kinematik (sonsuz enerjili) bir sürücü. `scene.object_info` 0. kare
temel pozunu okuduğu için küre "yerinde" görünüyordu; çözümlenen collider sınırları
(`collider.list` `resolved_world_*`) animasyonu izliyor. Bu yüzden §8'deki "2,3 m düşüş en
fazla 6,7 m/s verir" akıl yürütmesi yanlış zemindeydi: enerji kaynağı animasyon.

Deneyler (her biri `project.open` ile taze; 74 439 tane; 1/24 s adım):

| Koşu | Tepe yukarı hız | Kare |
|---|---|---|
| temel | 13,2 m/s | 19 |
| küp tabana oturtulmuş (düşüş yok) | 12,8 | 19 |
| `stiffness_scale` 10 | 24,4 | 21 |
| `contact_resolution` 96 (≈2,6× alt adım) | 12,6 | 19 |

Temas sondası (16–22. kare): `max_grain_overlap_fraction` 0,57–0,71 (taneler birbirine %70
giriyor), `max_collider_overlap_fraction` 0,99999 (tane merkezi collider yüzünün üstünde).

★★ **Çözücü açığı (gerçek, sahne tetikliyor):** DEM mesh teması İKİ YÜZLÜ
(`sim_matter_grain.glsl` BVH yaprağı: `n = (p-q)/d`, derinlik `r-d`). Kapalı bir cisme ezilen
tanenin merkezi yüzeyi geçince itme yönü TERSİNE döner, tane cismin içine çekilir ve içeriden
fırlar. Fışkıran taneler (0, −0,9, 0) civarında, yani 19. karede kürenin İÇİNDE. MPM yolu
işaretli SDF kullanıyor, DEM kullanmıyor. Düzeltme adayı: kapalı collider için yüz normalinin
tarafı (ya da SDF işareti) içerideki taneyi DIŞA, derinlik `r+d` ile iter; açık levha iki
yüzlü kalır (collider'ın kapalılık bilgisi shader'a taşınmalı). Sertlik arttıkça fışkırmanın
büyümesi burada çözücü hatasının kanıtı DEĞİL: kinematik sürücü sınırsız enerji veriyor.

★ **Yarım seed hata değil:** simüle edilmiş bir karede seed atılınca, o karenin collider
damgası (küre tabanda) sıvı ızgarasında katı sayılıyor ve taneler küreyle çakışan hücrelere
doğmuyor (küp 74 439 yerine 37 379). Doğru davranış. Tuzak: `fluid.sample_solid_phi`
`state->grid`'i okuyor, seed ise `liquidGrid(state)`'i. İki fazlı Matter domain'de bunlar ayrı
ızgaralar, ölçü aleti boş gösterdi.

★★★ **Ayrı bulgu: bozuk tek bir IPC parametresi uygulamayı ÖLDÜRÜYOR.** `fluid.seed`'e
`seed_max` dizisinde bir dize eleman gitti (PowerShell komut kipi `-1.25`'i dize geçirdi).
`optionalVec3` → `json::get<float>` → `abort()` → SIGABRT
(`x64/Release/StartupCrash*.txt`, 14:22:26). Kök: projede `ExceptionHandling=false`
varsayılan (Debug/Release/ReleaseDev); istisnasız derlenen birimlerde nlohmann `JSON_THROW`'u
`std::abort()` açıyor, linker o COMDAT kopyasını seçince `processJsonMessage`'in
`catch (json::exception)` blokları ölü kod. Aynı tuzak her `get<T>` tür hatasında (proje
dosyası okuma dahil) geçerli. Düzeltme kararı kullanıcıda: (a) proje çapında `/EHsc`
(tam rebuild) ya da (b) IPC yardımcılarında eleman başına `is_number()` kontrolü (yerel).

**2026-11-07 YAZILDI, derlenmedi:** kapalı kabuk tek yüzlü temas (host `closedShellFlags` → patch bit 30/31, shader rev 39, `closed_collider_faces` okuması) ve IPC Vec3/matris eleman tür kontrolü (`numberElement`). Kontrol listesi NEXT_BUILD_CHECKS en üstte. Proje çapında `/EHsc` kararı hâlâ açık.

**2026-11-07 canlı (exe 15:31):** IPC düzeltmesi PASS (bozuk `seed_max` ve `matrix` hata
döndürdü, uygulama ayakta). Toleranslı kaynakla küre kapalı tanındı (`closed_collider_faces`
1770). Ama fışkırma BÜYÜDÜ: 22. karede yukarı 38 m/s (önce 13), `max_collider_overlap_fraction`
2,0, tane örtüşmesi 0,83; 21. kareden sonra alt adım 512 tavanında (`travel (capped)`), 30. karede
`stiffness adapted` 0,84 (kullanıcının gördüğü "yumuşatma"). Kök sahne geometrisi: küre 20–29.
karelerde y −2,472'ye iniyor, yani **tabanın 21 cm altına**. Tabanla kürenin alt yüzü arasındaki
kum için geçerli bir konum yok. İki yüzlüyken taneler kürenin içine kaçıyordu; tek yüzlüde kaçamıyor,
sıkışma daha büyük. Kinematik sürücü sınırsız enerji veriyor, bu bir çözücü kararlılık hatası değil.
Seçenekler: (a) animasyon kürenin altını tabanın üstünde bitirir; (b) çözücü güvenliği: temas
itmesinin ürettiği ayrılma hızına tavan ("max depenetration velocity", fizik motorlarındaki gibi).
