# Sıradaki derleme kontrolleri — Player Android A1 (2026-11-22)

> **Durum:** CANLI — Windows Studio ve Player artımlı derleme ve çalışma zamanı kabulü bekliyor. İki hedefte değişen C++ birimleri MSVC x64 `/Zs` PASS.

## Player Android A5 — dokunmatik giriş (2026-11-22)

1. **APK.** Repo kökünde varlıkları hazırla ve x86_64 APK'yı derle. Kurulumdan sonra `Medium_Phone_API_37.0` (`-gpu host`) üzerinde Gizli Renkler açılmalı. Sahne açılmazsa önce varlık hazırlama çıktısını ve Player başlangıç logunu kontrol et. Android HUD, sahnesiz durumda da proje bırakma metnini göstermemeli.

   ```powershell
   python scripts/stage_player_android_assets.py
   $env:JAVA_HOME = 'C:\Program Files\Java\jdk-17'
   Push-Location android
   .\gradlew.bat assembleDebug --no-daemon -PrtAbis=x86_64 -PrtSdl2Dir=E:/SDL2-2.30.4/SDL2-2.30.11 -PrtSdl2ImageDir=E:/SDL2-2.30.4/SDL2_image-2.8.2_src
   Pop-Location
   adb -e install -r android/app/build/outputs/apk/debug/app-debug.apk
   adb -e forward tcp:47391 localabstract:RayTrophiPlayer
   Import-Module ./scripts/ipc/RtIpc.psm1 -Force
   Connect-RtIpc -Port 47391
   ```

2. **Tut, sürükle, bırak.** Görünür bir balonun ortasından başlayan `adb -e shell input swipe X1 Y1 X2 Y2 1500` komutunu ayrı terminalde çalıştır. Swipe öncesinde, sürerken ve sonrasında `(Invoke-RtIpc paint.wet_demo @{operation='status'}).interaction` oku. Beklenen sıra `held=-1 → balon ID'si → -1`, sürüklerken `owner=finger:<id>` ve görünür konum değişimi. Yatay ekranda koordinatı o anki kareden al. Eski APK'da 1725,540 noktası bir balonu tuttu.
   **En sinsi hata:** Fare sentezi açık kalırsa tek dokunuş iki olay dizisi üretir. Balon tutulup hemen bırakılmış gibi görünebilir veya sahip `mouse` olur. Swipe sürerken ölç. Tap sonrasındaki `held=-1` tek başına arızayı göstermez.
3. **Sıkıştırma ve boya.** Emülatörün çoklu dokunma girişinde ilk parmakla tut, ikinciyle aralığı daralt. Aynı balon ve sahip korunurken `interaction.depth` artmalı. İkinci parmak kalkınca tutuş sürmeli. Ardından balonu duvara bastırıp patlat. Patlama sayacının artması ve duvarda görünür boya beklenir. Patlama olup boya görünmezse A4 boya hattındaki `active_tiles` ve `device_event_dispatches` değerlerine bak.
4. **Windows fare regresyonu.** Üç değişen birim Player x64 `/Zs` PASS. Önceki tam derleme gereksinimleri tamamlandıysa A5 için artımlı Player derlemesi yeterli. Sol tuşla tut ve sürükle, tekerlekle derinliği değiştir, sol tuşu bırak. Tutarken IPC sahibi `mouse` olmalı. Shift ile bırakma ve odak kaybı mevcut tutuşu iptal etmeli. Sahnesiz ve hatasız açılışta proje bırakma metni görünmeli.

## ★ ÖNCE OKU — bu Windows derlemesi TAM derleme olmalı (Android A1–A3 + NDK düzeltmeleri, 2026-11-22)

Yaygın başlıklar değişti: `GpuInteropTypes.h` (yalnız clang dalı), `globals.h` (`gamma` silindi), `JoltWorld.h`, `PaintLayerStack.h`,
`PlayerVideoCard.h`, `Platform/Platform.h`, `GasSimulator.h`, `VDBVolumeManager.h`. Artımlı derleme eski `.obj` bırakırsa erişim
ihlali kod hatası sanılır → **Rebuild/temiz derleme**, Studio + Player. Visual Studio "proje dışarıda değişti" derse yeniden yükle.

W1. **Link (önce bu).** ✔ Studio temiz derleme linklendi (2026-11-22, kullanıcı); Player bekliyor. İki hedef hatasız linklenir. Bozuksa: `g_*` tanımsız → `using ::g_x;` değişikliği (VulkanBackend*,
    VulkanViewportBackend, TerrainNodesV2, AnimationNodes) global bildirimi göremiyor; `gamma` tanımsız → silinen global'i
    kullanan bir birim kaldı (grep'te bulunmadı); `BCryptGenRandom` → Platform.cpp pragma.
W2. **Terrain node gamma / animasyon klipleri / viewport raster** her biri bir kez açılsın — `using ::g_x` değişen yollar bunlar.
    Sessizce makul görünen hata: yanlış global'e bağlanma (Windows'ta beklenmez, MSVC zaten global'e bağlıyordu).
W3. **Boya katmanı tohumu.** Paint Edit'te doku setinden katman tohumla (`seedFromTextureSet` varsayılan argümanı aşırı yüklemeye
    döndü). Katman boş gelirse aşırı yükleme varsayılan `PaintUVParams`'ı geçmiyor.
W5. **(Sonradan eklendi, artımlı derleme yeter)** `Platform.cpp` → `PlatformOs.cpp` (LNK4042 kaybolmalı), `RtIpc.cpp` Windows-only
    korumasız (önişlemci çıktısı Windows'ta aynı), `RayFusionViewportPasses.h` +3 fonksiyon (7 birim yeniden derlenir), `PlayerMain`
    başlangıç hata bildirimi, `AppWindow` Android dalları. Windows'ta görmen gereken: Player normal açılır, HUD aynı. Sahnesiz açılışta
    "Drop an .rtp project here" görünür. Hatalı bir .rtp sürüklenirse artık ORTADA "could not open the scene" kutusu çıkar (yeni davranış).
W4. Not (önceden var olan): `VulkanBackend.h` `<sstream>` include etmiyor; vcxproj'da başka bir başlık maskeliyor olabilir.

## Player Android A2c — OpenVDB (2026-11-22)

1. **En sinsi hata — Windows'ta sessizce boş VDB hacmi.** Studio'da önceki sürümde görünür olduğu bilinen bir `.vdb` aç. Hacmin görünmesini ve aynı sahnede yoğunluğun önceki sürümle eşleşmesini bekle. İçe alma tamamlanmış görünüyorsa ama hacim boşsa önce Windows Studio/Player derleme tanımlarında `RT_NO_OPENVDB` bulunmadığını ve VDB'nin gerçek yöneticiye ulaştığını doğrula.
2. **Android NDK linki.** SDL2 ve SDL2_image kaynaklarını vererek arm64-v8a ve x86_64 hedeflerini derle. OpenVDB, TBB, Blosc veya Imath ön eki gerekmemeli. VDB yönetici ya da gaz sembolü çözümlenmezse önce üretilen kaynak listesindeki iki Android stub kaydını kontrol et.
3. **Android davranışı.** A3 yerel IPC üzerinden `scene.import_vdb` ve sparse VDB backend seçimini dene. İkisi de `OpenVDB not available on Android` hatası vermeli ve sahneyi değiştirmemeli. Paketli balon sahnesi normal açılmalı. Dışa aktarma denemesi dosya yazarsa, zaman çizelgesini değiştirirse veya boş hacim oluşturursa koruma yanlış noktadadır.

## Player Android A3 — IPC güvenliği OpenSSL'siz (2026-11-22, Windows'u da etkiler)

1. **Mevcut token hâlâ geçiyor mu (önce bu, hızlı).** `RtIpcSecurity` SHA-256'yı artık OpenSSL yerine ağaç içi `Utils/Sha256.h` ile, rastgele baytları `BCryptGenRandom` ile üretiyor. Daha önce verilmiş bir uzak token ile (`Start-RayTrophi.ps1 -Remote`, kayıtlı `ipc_tokens.json`) bağlan. Görmen gereken: kimlik doğrulama geçer. Bozuksa: özet uyuşmazlığı → SHA-256 uygulaması (test vektörleri 5/5 PASS ölçüldü, yani beklenmez) ya da `hex` biçimi.
2. **Yerel pipe davranışı aynı.** `Invoke-RtIpc agent.discover` Studio ve Player'da cevap verir; `connection_id` 32 hex karakter (boş değil). Boşsa: `secureRandomBytes` başarısız → bcrypt link/pragma.
3. **Linkleme.** Studio/Player hatasız linklenir (`bcrypt.lib` `Platform.cpp` içinde pragma ile). `BCryptGenRandom` çözümlenmemiş sembol → pragma derleyiciye ulaşmamış.
4. **Android (NDK sonrası).** `adb forward tcp:47391 localabstract:RayTrophiPlayer`, `Connect-RtIpc -Port 47391`, `Invoke-RtIpc scene.list_objects`. ★ En sinsi: IPC'nin Android'de HİÇ açılmaması — TLS stub'ı önceden her durumda `false` dönüyordu ve `RtIpc.cpp` buna yerel taşımayı da kapatarak cevap veriyor; düzeltildi (uzak IPC istenmediyse `true`). Log'da `IPC server started` yoksa önce buna bak. Bağlantı reddediliyorsa `SO_PEERCRED` uid'i (adb shell = 2000) logla.

## Player Android A2b — SIMD (2026-11-22)

1. **En sinsi hata — x86 yolunun sessizce yavaşlaması.** Windows Studio'da aynı exe yapılandırması, sahne, kamera, örnek sayısı ve ısınma süresiyle önceki sürüm ile CPU render ve AABB/bounding-box sürelerini karşılaştır. Aynı sahne sonucunu ve benzer süreleri bekle. Çıktı aynıyken süre artıyorsa `/arch:AVX2`, include sırası ve seçilen SIMD yolunu incele. Android x86_64 emülatöründe `-mavx2 -mfma` ile ayrı bir açılış kontrolü yap.
2. **AArch64 sözdizimi ve kabul.** NDK kurulduğunda önce arm64-v8a hedefinde `Vec3SIMD.cpp` ile `Matrix4x4.h` kullanan `AABB.cpp` birimlerini Android API 29 için kontrol et. Intrinsic, hedef başlığı veya istisna modu hatası beklenmez. Android araç zinciriyle kullanıcı derlemesinden sonra gerçek arm64 cihazda paketli sahneyi aç ve önceki kabuldeki görünür varlıkları karşılaştır.

0. **Player linkleniyor mu (önce bu).** Player vcxproj yeniden üretildi; `UI/PlayerOperator*UI.cpp` (6 dosya) A1 üretiminde düşmüştü, manifest düzeltmesiyle geri geldi. Visual Studio "proje dışarıda değişti" derse yeniden yükle. Görmen gereken: Player hatasız linklenir, F2 operatör menüsü açılır. Bozuksa: `drawPlayerOperator`/`releasePlayerOperatorUI` çözümlenmemiş sembol → vcxproj yeniden yüklenmemiş ya da manifest satırı kaybolmuş (`python scripts/gen_player_vcxproj.py --check`).
1. **Player açılış logu.** Paketli Player'ı paketin dışındaki bir çalışma dizininden başlat. `[Player Startup]` satırındaki çalışma dizini exe'nin klasörü, paket shader dizini eski Windows paketindeki aynı `shaders/` dizini olmalı. İlk sahne karesi shader eksikliği hatası olmadan gelmeli. Farklı kök çalışma dizini yönlendirmesine, doğru kökte eksik shader dosyası paket içeriğine işaret eder.
2. **En sinsi hata — AssetRegistry sessizce boş sahne.** Aynı paket ve `.rtp` ile önceki kabulde varlık kullandığı bilinen bir sahneyi aç. Beklenen arşiv kökünü, görünen model/doku/önizlemeleri ve sahne nesne listesini önceki sürümle karşılaştır. Açılış başarılı görünürken varlıkların kaybolması, `assets/` kökü veya kayıt çözümlemesi hatasıdır. Workshop balon sahnesinde duvarda ıslak boyayı da gör.
3. **UTF-8 ve yan dosyalar.** Studio ve Player'da boşluk ve Türkçe karakter bulunan bir yoldan aynı `.rtp` dosyasını aç. Varsa `.rtp.bin` ve harici kaynakları okuyarak ilk kareyi üretmeli. Uzun ya da sistem ANSI kod sayfası dışındaki bir exe yolunu ayrıca dene. Yalnız bu durumda açılış farklıysa geniş karakterli exe veya dosya yolu tüketicisini incele.
4. **Windows yükleyicileri ve tanılama.** İki hedefte Vulkan başlat. Paketli isteğe bağlı FidelityFX ile FSR'yi aç, CUDA bulunan Studio'da Terrain FFT'yi çalıştır ve performans profilindeki çalışma kümesinin ölçüldüğünü doğrula. Vulkan başlatılamazsa sürücü DLL yolu ve logu, FSR/FFT düşerse isteğe bağlı DLL varlığı ve sembol sonuçları kontrol edilmeli.
5. **Yerel IPC ve ajan başlatma.** Açık uygulamanın pipe'ına ayrı terminalden `scripts/test/rt_ipc.py` ile ilgili Studio/Player pipe adını kullanarak `agent.discover` çağır. Önceki Windows yerel erişim davranışı korunmalı. Kurulu RayTrophiAgent yapılandırmasıyla Windows'ta ajanı başlat ve durdur. Android'de IPC ve cihaz hazır olduğunda alt süreç başlatma açık bir `not supported on this platform` hatası vermeli.

Android cihaz kabulü A2/A4 ve SDK/NDK sonrasında. Paket varlıkları SDL iç depolama köküne çıkarıldığında Player'ın aynı `shaders/`, `scenes/` ve `assets/` yerleşiminden okuduğunu doğrula.

---

# Sıradaki derleme kontrolleri — Player eski GPU çökmesi (2026-11-20)

> **Durum:** CANLI — derleme bekliyor (Player, GTX 850M / i7-4700 laptop)

Pipeline trace artik varsayilan KAPALI: normal acilista dosya acma/yazma yok.
Yalniz process baslatilmadan once `RT_SIM_PIPELINE_TRACE=1` verilirse etkin.
Env bir kez okunur. Bu degisiklik icin kullanici Player artimli build almali.
Normal acilista mevcut SimulationPipelineTrace.txt boyutu/tarihi degismemeli;
env=1 ile module/pipeline satirlari append edilmeli. Codex build calistirmadi.

SON DURUM: kullanici shader deneylerinin geri alinmasini istedi. Iki derleme
scripti eski ayarlarinda; forces SPV kaynak/runtime/dist orijinal SPV1.6/27620 byte,
SHA256 2225C782CD67556B8828F1E36452BC25BD3B58426254558CE2F4009925A98158.
Asagidaki Vulkan1.2/-O0 dagitim talimatlari gecersiz; yalniz tani gecmisi.
Hem SPV1.5/-O hem SPV1.5/-O0 ayni pipeline.begin'de kesildi. Sebep kesinlesmedi.

## Shader adini kesinlestirme (Codex, 2026-11-20)

GUNCEL ikinci test: laptop SPV1.5/27676 byte optimizasyonlu forces shader'ini
yukledi ama ayni pipeline.begin'de tekrar coktu. Hedef surumunu dusurmek yetmedi.
Simdi yalniz forces Vulkan1.2/-O0 ile derlendi: SPV1.5, 37944 byte, validation PASS.
Source/runtime/dist kopyalari yenilendi; iki bulk script bu ayari korur.
Laptopta shaders/sim_fluid_particle_forces.spv degistir, Player'i yeniden ac;
trace'te 37944 byte ve pipeline.end kontrol et. Kok sebep henuz kesin degil.

Trace sonucu: iki deneme de `sim_fluid_particle_integrate_forces` /
`shaders/sim_fluid_particle_forces.spv` pipeline.begin sonrasi kesildi.
Yalniz bu shader Vulkan1.2/-O ile SPIR-V1.5 olarak yeniden derlendi ve dogrulandi;
source/runtime/dist kopyalari hash esit. C++ rebuild bu shader testi icin gerekmez.
Laptopta ayni isimli shaders dosyasini degistirip Player'i tamamen yeniden ac.
Yeni trace'te ilgili pipeline.end result=0 beklenir; sonraki hata varsa tum trace'i
topla. Bu henuz cihazda test edilmedi ve kok sebep kaniti degildir. ZIP eski kaldi.

- Player artimli C++ build kullanici tarafindan alinmali; shader degismedi.
- `SimulationPipelineTrace.h` yalniz startup'ta module/pipeline cagrilarinin
  hemen oncesini ve sonrasini `SimulationPipelineTrace.txt` dosyasina append eder.
  Her satir driver'a girmeden once dosya kapatilarak yazilir. Dosya process'in
  calisma dizinindedir; yazma izni yoksa trace olusmaz, acilis devam eder.
- Laptopta sahneyi bir kez ac; cokerse trace + SceneLog + crash stack'i topla.
  Son `pipeline.begin` icin karsilik gelen `pipeline.end` yoksa driver o shader'i
  olustururken donmedi. Kayit kernel adi, gercek SPV yolu, SPIR-V surumu ve
  byte boyutunu icerir. `module.begin` / `module.end` ayni ayrimi modul icin yapar.
- Son shader kesinlesince dagitilan SPV'nin capabilities/extensions listesini
  GPU'da acik ozelliklerle karsilastir; sonra yalniz ilgili shader/fallback'i duzelt.
  Kernel adi tek basina donanim uyumsuzlugunun kesin sebebini kanitlamaz.
- Derleme veya uygulama Codex tarafindan calistirilmadi; kaynak diff kontrolu yapildi.

Kök bulgu: `StartupCrashStack.txt` (PlayerMain'e eklenen yakalayıcı) + PDB ile çözüldü.
Çökme `vkCreateComputePipelines` içinde, NVIDIA `nvoglv64.dll`; zincir
`deserializeParticleSimulation → createParticleRuntime → syncSimulationWorld →
VulkanSimulationComputeBackend::loadPipelines → createPipeline`
(`SimulationComputeVulkan.cpp`). Hipotez: float atomic / float64 özelliği açık
olmayan cihazda, o yeteneği bildiren SPIR-V derleniyordu (UB → sürücü çöker).

Değişenler: PlayerMain.cpp (pencere erken gösterilir + çökme yakalayıcı),
ProjectManager.cpp (`load step:` logları), SimulationComputeVulkan.cpp
(`kernelNeedsMissingFeature` hem dispatch'te hem pipeline oluşturmadan önce).

Sırayla:
1. **Studio/dev makinede açılış** (hızlı, bağımsız). Float atomic ve float64
   destekleyen GPU'da hiçbir kernel atlanmamalı: sıvı/kar sahnesi eskisi gibi
   çalışır. Bozuksa: kapı yanlışlıkla atlıyor demektir (`m_has_float_atomics`
   / `m_has_shader_float64` ctor'a yanlış geçiyor).
2. **Laptopta Player.** Pencere açılıp sahne gelmeli (render siyah olabilir).
   ★ Sinsi durum: açılıyor ama parçacık/sıvı hiç çalışmıyor — sessizce makul
   görünür. SceneLog'da atlanan kernel sayısı yok; gerekirse log eklenmeli.
3. Hâlâ çökerse yeni `StartupCrashStack.txt` + `SceneLog.txt`'deki son
   `load step:` satırı. Aynı `createPipeline` çıkarsa hipotez yanlış: kernel adı
   için pipeline döngüsüne geçici log koy ve sürücüyü güncelle (850M için 4xx
   değil, mevcut son Maxwell sürücüsü).
4. Fullscreen'den çıkılamıyor raporu: F11 ile `SDL_SetWindowFullscreen`
   (PlayerMain.cpp:432). Pencere çökmeden önce takılı kaldıysa ayrı değerlendir.
