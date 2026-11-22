# Player balon oyunu — Android portu (dokunmatik)

> **Durum:** AKTİF — 2026-11-22: A1–A5 emülatörde PASS (sahne, ıslak boya, ses, IPC, dokunma); arm64 gerçek cihazda açılıyor ve desteklenmeyen GPU'yu ekranda söylüyor; Windows regresyon PASS. Sıradaki A6: referans cihaz + performans.

## A5 devir notu (2026-11-22, Codex)

- Eski APK, `Medium_Phone_API_37.0`: görünür balonda `adb shell input swipe 1725 540 1725 540 1100` sırasında IPC `held=3, owner=mouse`, bırakınca `held=-1` döndürdü. Ölçülen noktada koordinat ve SDL fare sentezi çalışıyor. Bir tap bittikten sonra sorgulanan `held=-1` kısa süreli tutmayı kaçırır.
- Android SDL parmak olayları normalize ekran koordinatları ve `finger:<id>` sahibiyle ortak `paint.wet_demo` pointer yoluna gidiyor. İkinci parmak mevcut `wheel` ile derinliği ayarlıyor. İlk parmak kalkınca balon bırakılıyor. Android pencere kurulmadan önce dokunmadan fare sentezi kapanıyor ve HUD'daki proje bırakma metni gizleniyor.
- Mevcut APK ile kullanıcı emülatörde fareyle balon taşıma ve patlatmayı doğruladı. Tekerlekle ileri/geri taşıma doğrulanmadı. `-gpu host` ve 1600×720 ekran ayarında boşta yaklaşık 56 FPS, çoklu patlamada yaklaşık 21 FPS görüldü. Patlamalar bitince yaklaşık 56 FPS'ye döndü. İç render boyutu bu ölçümde doğrulanmadı. Sayılar emülatöre ait.
- Windows Player MSVC x64 `/Zs`, Android NDK x86_64 ve arm64-v8a `-fsyntax-only` üç değişen birimde geçti. Gizli Renkler için 387 varlık (49.1 MiB) hazırlandı. Kullanıcı Windows Player Release x64 linkini ve demoda fareyle balon yakalamayı doğruladı. Studio derlemesi de kullanıcı onayıyla geçti. Yeni APK ile dokunarak duvarda boya kabulü bekliyor. [A5 kabul listesi](NEXT_BUILD_CHECKS.md#player-android-a5--dokunmatik-giriş-2026-11-22)


## Kullanıcı doğrulaması (2026-11-22 akşam)

- Windows Studio + Player derlemeleri GEÇTİ (e1cd0506 sonrası), Player'da fareyle balon yakalama doğrulandı.
- Mevcut x86_64 APK, emülatör `-gpu host` 1600×720: fareyle taşıma ve patlatma çalışıyor. Boşta ~56 FPS, çoklu patlamada 21 FPS,
  sonra yine 56. İç render çözünürlüğü doğrulanmadı. Patlama düşüşü emülatör ölçümü değil; A6b geçiş bazlı GPU zamanıyla gerçek cihazda bakılacak.
- Bekleyen A5 kabulü: yeni x86_64 APK ile dokunarak sürükleme, iki parmakla derinlik, duvarda boya.
- **A5 emülatör kabulü (Sonnet, aynı akşam):** yeni x86_64 APK (72.8 MiB). Dokunarak sürükleme PASS: `input swipe` sırasında
  `held=1, owner="finger:0"` (mouse DEĞİL → fare sentezi kapalı, çift olay yok), bırakınca `held=-1`. Duvar boyası PASS (IPC pop:
  `active_tiles` 0→113). İç render 1600×720 (`render_scale` 1.0), sunum 1600×594; boşta 46 FPS. **Elle kalan:** iki parmakla derinlik
  ve dokunarak patlatma (adb çoklu dokunma üretemiyor).
- **Kullanıcı elle (emülatör):** dokunarak patlatma PASS (emülatör fareyi dokunma olarak iletir; balon duvara sıkıştırılınca patlıyor).
  Ctrl ile iki dokunma noktası: yakalanıp ortaya çekilebiliyor, çökme yok → iki parmak yolu çalışıyor. **A5 emülatörde PASS.**
  Açık kalan tek şey: "bir parmak tutarken ikinciyle derinlik" — emülatör Ctrl ile yalnız simetrik iki parmak üretir; ilk uygun gerçek
  dokunmatik cihazda (A6 referans cihazı) denenecek.
- ★ QA tuzağı: emülatör/telefon ekranı uykudayken SDL ana döngüyü duraklatır → IPC `main thread dispatch timeout`. Test öncesi
  `adb shell input keyevent KEYCODE_WAKEUP` + `adb shell svc power stayon true` (emülatörde açık bırakıldı).

## A2c devir notu (2026-11-22, Codex)

- Android manifesti `RT_NO_OPENVDB=1` tanımlar. Gradle ve üretilen CMake artık OpenVDB, TBB, Blosc veya Imath ön eki istemiyor. Windows projelerinde OpenVDB derleme ayarları, çağrıları ve veri düzenleri korunuyor.
- Android manifesti iki VDB stub'ını seçer. Legacy gas ve VDB manager için Windows Player host birimleri de Android listesinden çıkarıldı. Legacy gas emiter/ayar JSON kodu ortak dosyada aynı biçimde paylaşılır.
- VDB içe alma, dışa aktarma, legacy gas başlatma ve sparse VDB backend seçimi `OpenVDB not available on Android` bildirir. Desteklenmeyen çağrılar için ortak uyarı süreç başına bir kez yazılır. VDB dizi dışa aktarımı zaman çizelgesine dokunmadan sıfır kare döner.
- Studio ve Player için MSVC x64 `/Zs` normal tanımlarla ve `/DRT_NO_OPENVDB=1` ile PASS. Makrolu denetimlerde OpenVDB bağımlılık başlıklarına erişim görülmedi. İki üretici ve Player include denetimi PASS. Android NDK link ve cihaz kabulü bekliyor. [Kontrol listesi](NEXT_BUILD_CHECKS.md)

## arm64 + A6a ret ekranı — gerçek cihaz (2026-11-22, Claude)

- **arm64-v8a ilk derlemede hatasız** (SIMDe yolu dahil), APK 76 MB. Lenovo TB330FU (Mali-G52 MC2, Android 15) üzerinde açılıyor,
  387 dosya çıkarılıyor, IPC açık, çökme yok. Telefon (OPPO CPH2687) kurulum sırasında bağlantıdan düştü — denenmedi.
- Player'ın kendi kontrolü vkjson okumasıyla AYNI: `Skipping Vulkan adapter Mali-G52 MC2: required descriptor indexing features unavailable`.
- ★ **A6a (ret ekranı) yazıldı ve tablette doğrulandı:** önce yalnız debug HUD'da "Player Vulkan viewport initialization failed" + yanıltıcı
  "Drop an .rtp project here" vardı. Şimdi: `Backend::recordRayFusionDeviceRejection/rayFusionDeviceRejections` (RayFusionViewportPasses,
  seçim başında temizlenir) → `RuntimeSceneViewer` hatası "This device's GPU is not supported - <GPU>: <sebep>" → PlayerMain
  `drawStartupFailureNotice`: sahne yok + hata varsa EKRAN ORTASINDA, HUD kapalıyken de. Hata `last_open_error`'dan gelir (backend
  açılış içinde kuruluyor) — ilk deneme yalnız `error`'a bakıp kaçırdı. "Drop..." satırı yalnız hata yokken.
- ★ **Ekran klavyesi:** `AppWindow.cpp` herkese `SDL_StartTextInput()` çağırıyordu → Android'de oyun boyunca klavye açık. Android'de
  çağrılmıyor + `SDL_StopTextInput()`. Oyunda metin alanı yok; gerekirse o alan kendisi başlatır.
- Kalan A6a işi: metinler İngilizce sabit (i18n'e bağlanmadı); `paint_wet` SSBO şartı açılışta kontrol ediliyor mu (bu cihazda 35 ≥ 17,
  sınanamadı).

## A4 KABUL — Gizli Renkler emülatörde, DUVARDA BOYA (2026-11-22, Claude)

- **Akış:** `python scripts/stage_player_android_assets.py` → `android/app/build/rt_assets/` (git dışı). Kaynak TEK paket
  `dist/RayTrophiPlayer` (kural 7): shader LİSTESİ paketten, İÇERİĞİ `x64/Release/shaders`'tan (eski .spv + yeni libmain = sessiz ABI
  uyuşmazlığı; fark varsa betik söyler), `assets/`, seçilen tek sahne klasörü, TEK sahneli `player_scenes.json` (Windows'taki kütüphane
  UI'si operatör katmanında, Android'de yok → tek sahne doğrudan açılır). Token/operatör/ajan yapılandırması, exe/DLL, perception ASLA girmez.
  `rt_asset_manifest.txt` 1. satır `stamp <sha256>`. Gradle `assets.srcDirs += build/rt_assets`, büyük ikililer sıkıştırılmaz.
- **Çıkarma:** `Platform::extractBundledAssets` (PlatformOs.cpp; Windows'ta no-op). Stamp aynıysa atlar; önce stamp'i siler, `.partial` + rename
  ile kopyalar, eski manifestte olup yenide olmayanı siler, stamp'i EN SON yazar. Uygulamanın kendi dosyalarına (ayarlar, log) dokunmaz.
  Log: `extracted 387 bundled files (stamp c0dc5da8d1b4)`. APK 76 MB (49 MiB varlık).
- **Sonuç (emülatör, RTX 3060 üzerinden gfxstream, Vulkan 1.3):** `[Player GPU] SSBO per_stage=1048576 ... collision_15=yes wet_paint_17=yes`,
  RayFusion kabul (descriptorIndexing=1), 27 mesh / 51 malzeme / 35 doku / EXR HDRI yüklendi, `paint_wet` pipeline kuruldu.
  IPC `paint.wet_demo pop {balloon:0}` → `active_tiles` 0→34, `device_event_dispatches` 175; ekranda duvar/tavan/zeminde boya, gizli
  objeler açığa çıktı. **A4 PASS** (kabul "duvarda boya", sahne açıldı değil).
- İlk ekran görüntüsü SİYAHTI ama log `First scene frame uploaded ... nonzero_rgb_samples=256` diyordu → ilk karelerin gecikmesi, hata değil.
  İlk saniyeler ~13 FPS (pipeline ilk derlemesi), sonra HUD 34 FPS / backend 4.96 ms — emülatör ölçümü DEĞİL performans.
- **Ses PASS:** miniaudio Android'de açılıyor (`audio.status device=ready`, 48 kHz), `paint.wet_demo status pop_sounds_played=15`,
  düşen/hata yok; kullanıcı emülatörde patlama seslerini DUYDU (ilk denemede bilgisayar sesi kısıktı).
- Emülatörde patlama anında belirgin yavaşlama (kullanıcı gözlemi). Emülatör ölçümü değil (gfxstream her Vulkan çağrısını taşır);
  gerçek cihazda A6b geçiş bazlı GPU zamanıyla bakılacak — "emülatör yavaş" diye kapatılmadı.
- Cihaz taraması (`adb shell cmd gpu vkjson`): Lenovo Tab M11 TB330FU = Mali-G52 MC2, Vulkan 1.3, SSBO 35, **descriptor indexing YOK**;
  OPPO CPH2687 = Snapdragon 680 / Adreno 610, **Vulkan 1.1.128**, push 128 B. İkisi de yetenek tabanının altında (Note 8 gibi) →
  yalnız arm64 açılışı + açık ret mesajı (A6a) testi. Satıcı adı yongayı söylemez: önce vkjson oku.
- `adb shell input tap` ile balon tutma bu turda doğrulanmadı (`interaction.held=-1`) → A5'te bakılacak (SDL fare sentezi).
- Bilinen uyarılar (zararsız görünüyor, izlenecek): `SimDiskCache manifest.json` yok (workshop.simcache paketlenmemiş), `MatUpload` transmission uyarıları (Windows'ta da var mı bak).

## İLK AÇILIŞ + A3 KABUL — emülatör (2026-11-22, Claude)

- Emülatör: BIOS'ta SVM kapalıydı (`Virtualization Enabled In Firmware: No`) → kullanıcı açtı; WHPX kullanılabilir (AEHD gerekmedi,
  Hyper-V ile çakışır). AVD `Medium_Phone_API_37.0`, `-gpu host` (RTX 3060 üzerinden gfxstream).
- **Player açılıyor:** SDL main, pencere, ImGui HUD ("Drop an .rtp project here", IPC durumu). Log: `/data/data/com.raytrophi.player/files/SceneLog.txt`
  (`adb shell run-as com.raytrophi.player cat files/SceneLog.txt`). Sahne yok → Vulkan viewport henüz kurulmadı, `[Player GPU]` satırı yok (A4 ile gelecek).
- ★ Bulunan: `rtipc::start()` başında `#ifndef _WIN32 → "IPC server is only supported on Windows"` vardı → A3 soketi hiç çağrılmıyordu.
  Söküldü. Sonra: `[Player] IPC server started`, `/proc/net/unix`'te `@RayTrophiPlayer`; PC'den
  `adb forward tcp:47391 localabstract:RayTrophiPlayer` + `Connect-RtIpc -Port 47391` → `agent.discover` 102 metot, `scene.list_objects` = []. **A3 PASS.**
- ★ Bulunan: yeniden boyutlanabilir SDL penceresi manifestteki `sensorLandscape`'i eziyordu (dikey açıldı) → `AppWindow.cpp` Android'de
  `SDL_HINT_ORIENTATIONS = "LandscapeLeft LandscapeRight"`. Doğrulandı: yatay.
- Gradle: `android/gradlew.bat assembleDebug --no-daemon -PrtAbis=x86_64 -PrtSdl2Dir=... -PrtSdl2ImageDir=...` (JAVA_HOME = jdk-17).

## A2 KABUL — ilk NDK derlemesi (2026-11-22, Claude + Sonnet döngüsü)

- **x86_64 `libmain.so` derleniyor ve LİNKLENİYOR** (NDK 27.3, android-29, Release; ninja FAILED=0). Derleme dizini repo dışı:
  `E:/rt_android_build/x86_64` (CMake doğrudan; Gradle/APK henüz yok). 432 → 194 → 4 → 1 → 0 birim, 5 tur.
- Kök düzeltmeler: `GpuInteropTypes.h` clang dalı, ölü `gamma` global'i silindi (POSIX `gamma()` çakışması), manifest include
  (imgui, FidelityFX başlıkları) + x86_64 `-mf16c -mlzcnt -mbmi` (Jolt `__AVX2__`'den türetiyor) + `-fdeclspec`,
  `Platform::systemTotal/AvailableMemoryBytes`, float `from_chars` yerine Android'de `strtod` (glTF, OBJ), `JoltWorld::init` ve
  `PaintLayerStack::seedFromTextureSet` varsayılan argümanı → aşırı yükleme, `PlayerVideoCard.h` Windows'a koruma,
  `vkGetBufferDeviceAddress` Android'de `vkGetDeviceProcAddr` ile.
- ★ **Sessiz fark:** namespace içindeki blok kapsamlı `extern g_x;` MSVC'de global'e, clang'de namespace'e bağlanıyor → 6 tanımsız
  sembol. `using ::g_x;` ile düzeltildi (VulkanBackend, VulkanBackend_Raster, VulkanViewportBackend, TerrainNodesV2, AnimationNodes).
  Yeni kodda namespace içinde blok `extern` YAZMA.
- ★ **Bilinen performans borcu:** `Platform/AndroidCompat.h` (yalnız Android, force-include) `std::execution::par/par_unseq`'i
  SIRALI çalıştırıyor (NDK libc++'ta paralel politika yok). Etkilenen: Hair fırça, PostSurface, Renderer derinlik — balon oyununun
  sıcak yolunda değiller, ama A6 profillemesinde bakılacak; gerekirse OpenMP'ye çevrilir (sahte paralellik değil, açık borç).
- `libmain.so` ~590 MB (Release + `-g`); APK paketlemesinde strip edilir.
- Sıradaki: Gradle wrapper + APK (x86_64) → emülatörde açılış logu (`[Player GPU]`, SSBO per_stage) → A4 varlık çıkarma. arm64 derlemesi ayrı tur.
- **Windows temiz derleme (kullanıcı, 2026-11-22):** Studio + Player linklendi; skin anim ve terrain doğru. LNK4042 uyarısı çıktı:
  `Platform.cpp` ile ozz `src/base/platform.cc` aynı `platform.obj`'ye yazıyordu (Windows büyük/küçük harf ayırmaz) → biri sessizce
  düşüyordu. Dosya **`src/Platform/PlatformOs.cpp`** oldu (Studio vcxproj + filters, manifest, üretilmiş Player + Android listesi).
  Ders: yeni .cpp adını `external/` altındaki adlarla da çakıştırma.
- **APK (Sonnet, 2026-11-22):** `android/app/build/outputs/apk/debug/app-debug.apk` (x86_64, 23.5 MB; libmain.so stripped 70.9 MB,
  SDL2, SDL2_image, c++_shared). Gradle 8.9 wrapper, AGP 8.5.2, JDK 17 (`C:\Program Files\Java\jdk-17`), compileSdk 34; `-PrtAbis=x86_64`
  tek ABI. **Emülatör açılmadı:** "x86_64 emulation requires hardware acceleration" — hipervizör sürücüsü yok (kullanıcı: Windows
  Hypervisor Platform ya da SDK'dan AEHD). AVD adı `Medium_Phone_API_37.0`. Açılış logu henüz yok.

## A3 devir notu (2026-11-22, Claude) — Windows /Zs PASS, Android dalı derlenmedi

- **Taşıma:** `RtIpcTransportLocal.cpp` `#else` dalı = **soyut unix soketi** `@RayTrophiPlayer` (TCP değil). Güvenlik sınırı
  pipe ACL'inin karşılığı `SO_PEERCRED`: yalnız uygulamanın uid'i, root ve adb shell (2000). Loopback TCP her uygulamaya açık
  olurdu. Çerçeve 4 bayt BE uzunluk + JSON (TLS ile aynı). 4 istemci yuvası. PC: `adb forward tcp:47391 localabstract:RayTrophiPlayer`
  + `Connect-RtIpc -Port 47391` (`RtIpc.psm1`, iki kopya).
- **OpenSSL söküldü (IPC güvenlik/oturum, iki platform):** `Utils/Sha256.h` (başlık-only, FIPS test vektörleri 5/5 PASS) +
  `Platform::secureRandomBytes` (BCryptGenRandom / getrandom). Kayıtlı token özetleri aynı kalır. TLS taşıması Windows'ta OpenSSL'i
  kullanmaya devam eder. Android manifestinden `RT_OPENSSL_PREFIX` çıktı.
- ★ **Bulunan hata:** A2'nin TLS stub'ı her zaman `false` dönüyordu → `RtIpc.cpp` yerel taşımayı da kapatıyordu → Android'de QA kanalı
  hiç açılmazdı. Stub artık gerçek taşımanın sözleşmesini izliyor (uzak IPC istenmediyse başlatacak bir şey yok).
- Sıradaki: NDK gelince ilk derleme (A2 kabulü) → A4 varlık çıkarma → A5 dokunma.

## A2b devir notu (2026-11-22, Codex)

- SIMDe v0.8.2 başlıkları ve MIT lisansı `RayTrophiStudio/external/simde/` altında. AArch64, ortak `Math/SimdIntrin.h` üzerinden SSE, AVX2 ve FMA takma adlarını alır. x86 aynı derleyici başlığını ve intrinsic çağrılarını kullanır. 43 farklı intrinsic için SIMDe karşılığı bulundu. Skaler dal gerekmedi.
- Gerçek kullanım `Vec3SIMD.cpp/.h`, `RayPacket.h`, `water_shaders_cpu.h`, `Matrix4x4.h` ve `AABB.cpp` içinde. Yanlış pozitifler `World.cpp`, `AtmosphereClimate.cpp/.h` ve `AtmosphereClouds.cpp/.h` içindeki `precipitation_mm_h` ve `rate_mm_h` yağış alanları. simdjson'un AArch64 seçimi, stb_image'ın x86 kapısı ve isteğe bağlı NEON yolu, tinyexr'ın ARM FP16 koruması kaynakta doğrulandı.
- AArch64 CPU bilgisi `SIMD: NEON` bildirir. x86 özellik sorguları false döner. Android manifesti, `Vec3SIMD::get` sınır dışı indis hatası için yalnız ilgili birimde istisnaları açar.
- Studio ve Player ile yedi kaynak birimi ve ortak başlık probu MSVC x64 `/Zs` PASS. Windows Player ve Android CMake üreticilerinin `--check` denetimleri ile Player include denetimi PASS.
- Android AArch64 **doğrulanmadı**. NDK ve bağımsız clang sürücüsü bulunamadı. VS LLVM 19.1.5 clang-tidy ile `aarch64-linux-android29` hedefinde denenen `Vec3SIMD.cpp` ve `AABB.cpp` standart başlıklar eksik olduğu için durdu. [Derleme ve performans kontrolü](NEXT_BUILD_CHECKS.md)

## A1 devir notu (2026-11-22, Codex)

- Platform sözleşmesi uygulanıp Studio ve üretilen Player projelerine kaydedildi. Studio ve Player tanımlarıyla 12 değişen C++ biriminin her biri MSVC x64 `/Zs` PASS. Player proje üreticisi ve include denetimi PASS.
- A2/A4 için Android çalışma dizini ve exe-göreli aramalar SDL iç depolama köküne bağlanır. AssetRegistry burada `assets/` bekler. A4 bu dizin yapısını dolduracak.
- Windows Vulkan/CUDA yükleme politikası, FSR DLL arama bayrakları ve IPC pipe ACL kodu korundu. Eski ANSI exe sorguları sözleşmedeki geniş karakterli sorguya geçti. Uzun ve ANSI kod sayfası dışında kalan yollar çalışma zamanı kontrol listesine eklendi.
- Windows link ve canlı kabul, kullanıcı artımlı derlemesinden sonra [NEXT_BUILD_CHECKS.md](NEXT_BUILD_CHECKS.md) ile yapılacak. Android NDK derlemesi ve cihaz testi bekliyor.

## A2 devir notu (2026-11-22, Claude/Sonnet) — derlenmedi, NDK yok

- `scripts/gen_player_android_cmake.py` (gen_player_vcxproj fonksiyonlarını import eder) → `android/app/src/main/cpp/player_sources.cmake`,
  `--check` PASS. Android verisi ayrı: `scripts/player_android_manifest.json` (`android_exclude`, `android_stubs`, define, bağımlılık).
  851 kaynak, 11 hariç, 3 stub (`src/Platform/Stubs/`: operatör katmanı, TLS taşıması, DirectXTex doku önbelleği).
- `android/` Gradle iskeleti + `PlayerActivity` (SDLActivity). `gradlew` yok; SDL kütüphane sırası doğrulanmadı.
- ★ **Windows Player link kırığı düzeltildi (Claude):** A1'de vcxproj yeniden üretilince `UI/PlayerOperator*UI.cpp` (6 dosya) düştü —
  commit'li vcxproj'da elle duruyorlardı, manifestte yoklardı; `PlayerMain` onları koşulsuz çağırıyor. Manifeste `src/UI/PlayerOperator*`
  eklendi, iki üretici + include denetimi PASS. Ders: üretilmiş projeye elle eklenen satır ilk yeniden üretimde sessizce kaybolur.

### Açık kararlar (A2 sonrası)

| Konu | Durum | Öneri |
|---|---|---|
| **arm64 SIMD** | A2b uygulandı. AVX/AVX2/FMA ve SSE için tek ortak başlık var. Windows x64 sözdizimi kabulü geçti. | AArch64 NDK sözdizimi ve cihaz kabulü bekliyor. Windows x86 SIMD performansını önceki sürümle karşılaştır. |
| **OpenVDB** | A2c uygulandı. Android, `RT_NO_OPENVDB=1` ile VDB stub'larını seçiyor ve OpenVDB ön eki istemiyor. | NDK link ve cihazda açık hata kabulü bekliyor. Windows Studio'da bilinen bir .vdb hacmi görünür olmalı. |
| **OpenSSL** | `RtIpcSecurity`/`RtIpcSession` SHA-256 + `RAND_bytes` | A3'te (Claude): Android dalında ağaç içi SHA-256 + `getrandom`; libcrypto derlenmez |
| **İstisna modu** | Global `-fno-exceptions`, EHsc dosyalara `-fexceptions` (vcxproj aynası); yedek `RT_ANDROID_FORCE_EXCEPTIONS` | NDK ilk derlemesinde bak |
| **OpenMP** | `-fopenmp -static-openmp` (APK'ya libomp.so gerekmez) | Kabul |

## İş bölümü (2026-11-22, kullanıcı kararı: Claude yönetici)

| Kim | Parti | Dosyalar | Not |
|---|---|---|---|
| **Claude (Opus)** | Sözleşme + A3 IPC soketi + A5 dokunma + iki partinin gözden geçirmesi | `Platform/Platform.h` (YAZILDI, sözleşme), `RtIpcTransportLocal.cpp` `#else`, `PlayerBalloonInput.cpp` | Ağır tasarım, az token |
| **Codex** (kullanıcı çalıştırır) | **A1** Platform katmanı uygulaması + Win32 dokunuş noktaları | `src/Platform/Platform.cpp` (yeni, iki vcxproj), tablodaki ~19 dosyanın yalnız `#ifdef`/çağrı değişimi | Windows davranışı bit-bit aynı; Studio + Player /Zs PASS |
| **Sonnet alt ajanı** | **A2** CMake üreticisi + `android/` Gradle iskeleti + stub host'lar | `scripts/gen_player_android_cmake.py`, `android/**`, `src/Platform/Stubs/*.cpp` — **yalnız yeni dosyalar** | Ortak dosyaya dokunmaz → A1 ile çakışmaz |

- Sıra: A1 ∥ A2 → (Android SDK/NDK kurulumu, kullanıcı) → A3 → A4 → A5 → A6.
- ★ Engel: makinede Android SDK/NDK **yok** (2026-11-22 kontrol). A2'nin "NDK linkleniyor" kabulü ve emülatör buna bağlı;
  A2 kurulumdan önce yalnız üretici + `--check` seviyesine kadar gider.

## Devir notu (2026-11-21, Claude → Codex)

- **Bitti:** kaynak taraması, plan A0–A6, A0 cihaz kararı (Note8 ✘, Mali-G57 sınıfı ✔, geliştirme emülatörde).
- **Sıradaki:** A1 (`Platform/` ince katman + iki korumasız Win32 dosyası) ya da emülatör kurulumu — kullanıcı seçecek.
  A1'den önce `AJAN_KOORDINASYON.md` §1'e satır; Windows Player + Studio /Zs ile aynen derlenmeli.
- **Yarın başlangıç (kullanıcı onayı 2026-11-21):** A6 profillemesi ucuz G57 ile değil, kaliteyi taşıyabilecek
  bir **referans cihazla** başlar: önce "hedef görüntü bunda akıcı mı", sonra tabanın ne kadar inebileceği.
  Referans cihaz seçimi ilk konuşulacak konu.
- **Kod yazılmadı, derleme yok, commit yok.** Bu not ve `README.md` indeks satırı tek değişiklik.
- Tuzaklar: tek parmak SDL fare sentezinden zaten gelir — A5'te parmak olaylarına geçerken sentezi kapat
  (çift olay); `paint_wet` kurulamazsa oyun boyasız-hatasız açılır, kabul "duvarda boya".

## Karar ve kapsam

- ★ **Konumlandırma (kullanıcı kararı 2026-11-21):** bu, motorun ilk ürünü — pazar genişliği değil **yenilik ve
  kalite** ile öne çıkacak. Sonuç: donanım tabanı kalitenin tam görüldüğü cihaza göre seçilir; geniş cihaz
  uyumluluğu için görsel ödün ve ikinci render yolu önerilmez. A6d tablosu tabanı "kaç cihaz" ile değil
  "bu kalite hangi cihazda akıcı" ile belirler.

- Hedef: Gizli Renkler / balon boyama sahnesinin Android telefonda/tablette **tek oyunculu, dokunmatik** oynanması.
- Studio ve Windows Player **değişmeden** derlenmeye devam eder. Android üçüncü bir derleme hedefidir,
  aynı kaynak ağacını ve aynı `player_include_manifest.json`'u kullanır.
- Kapsam DIŞI (bu aşama): algı köprüsü (Python/MediaPipe/OpenCV), kiosk/operatör katmanı (F1–F11, PIN),
  OpenVDB hacimleri, RT/RayFusion RT yolları, ses dışındaki ağır varlıklar.

## Mevcut durum (2026-11-21 taraması)

### Taşınabilir olan

| Parça | Neden sorun değil |
|---|---|
| CUDA/OptiX/Embree/OIDN/Python | Player zaten bunlarsız linkleniyor (bkz. `RAYTROPHI_PLAYER_PLAN.md`) |
| Pencere/olay döngüsü | SDL2; resmi Android desteği (`SDLActivity`, Vulkan yüzeyi, dokunma olayları) |
| Jolt, ozz-animation, imgui, miniz, xatlas | Çok platformlu |
| OpenMP (51 dosya) | NDK clang `-fopenmp` destekler (`libomp.so` APK'ya girer) |
| Balon/ıslak boya | Saf Vulkan compute (`paint_wet.comp`, `sim_particle_*.comp`); yalnız `GL_GOOGLE_include_directive` |
| Vulkan RT | İsteğe bağlı: `VulkanBackend.cpp` AS/RT pipeline/ray query, atomic float, BDA'yı yoksa düşürüyor |

### Cihaz alt sınırı (bugünkü kod)

`Backend::rayFusionDeviceRejection` (`RayFusionViewportPasses.cpp:115`) şunları ister:
Vulkan **1.2**, descriptor indexing (`runtimeDescriptorArray`, `shaderSampledImageArrayNonUniformIndexing`,
`descriptorBindingPartiallyBound`), `independentBlend`, `fragmentStoresAndAtomics`, birleşik graphics+compute kuyruğu.
Bunlar Adreno 6xx+/Mali Valhall+ sınıfında güncel sürücüyle karşılanır; eski Mali Bifrost sürücüleri
non-uniform indexing'de takılabilir.

★ Ek risk: `paint_wet.comp` tek sette **≥16 storage buffer** bağlıyor (binding 0–15). Hedef cihazda
`maxPerStageDescriptorStorageBuffers` bundan küçükse boya pipeline'ı kurulamaz. Değer Player açılışında
zaten loglanıyor (`[Player GPU] SSBO per_stage=`) — ilk cihaz testinde ilk okunacak satır budur.

### A0 sonucu — 2026-11-21 (vulkan.gpuinfo.org raporları)

| Cihaz / GPU | Vulkan | Descriptor indexing | SSBO per_stage | Karar |
|---|---|---|---|---|
| Galaxy Note8 (Exynos 8895, Mali-G71), rapor 20980 / 45487 | **1.0.97** (sürücü 16; özel ROM'larda da aynı) | **yok** | 35 | ✘ hedef değil |
| Mali-G57 (ör. ZTE Z2469N, rapor 53461; Galaxy A serisi sınıfı) | 1.3.278 | var (üç özellik de true) | 500000 | ✔ alt sınır adayı |

Note8'in tek eksiği "biraz eski sürücü" değil: Vulkan 1.0 + descriptor indexing yokluğu bütün raster
backend'in (bindless doku dizileri) varsayımını kırar. Ona inmek ayrı bir 1.0 render yolu demek — CLAUDE.md §5
(iki yol yaşatma) ve "Studio/Player bozulmasın" ile çelişir. **Karar önerisi: Note8 hedef dışı.**
(SSBO 35 ve `fragmentStoresAndAtomics`/`independentBlend` Note8'de bile yeterli; engel yalnız sürüm + indexing.)

Mali-G57 bütün şartları geçiyor → "çalışır" alt sınırı 2021+ giriş/orta segment. "Akıcı" ise ölçülmeden
söylenemez. G57 yalnız **yetenek** alt sınırı; **performans** tabanı A6 profillemesiyle belirlenir ve gerekirse
yukarı çekilir (kalite düşürülmez).

**Emülatör:** Android Studio emülatörü (gfxstream) Vulkan'ı ana makinenin GPU'sundan geçirir; işlevsel test için
uygun, **performans ölçümü için değil**. Emülatör imajı x86_64 olduğundan Android derlemesi `arm64-v8a` +
`x86_64` iki ABI üretmeli. Emülatörün raporladığı Vulkan sürümü/indexing doğrulanmadı — Player açılış logu
(`[Player GPU]` + `rayFusionDeviceRejection` nedeni) bunu ilk açılışta söyler.

### Win32 dokunuş noktaları (Player'ın 680 kendi dosyasından 19'u)

Çoğu **zaten `#ifdef _WIN32` arkasında**; Android'de yalnızca POSIX/SDL karşılığı gerekir.

| Dosya | Win32 kullanımı | Android'de |
|---|---|---|
| `Api/RtIpcTransportLocal.cpp` | named pipe | Yerel TCP soket (adb forward) — QA kanalı açık kalmalı (CLAUDE.md §1) |
| `Api/RtIpcSecurity.cpp` | pipe ACL, `GetModuleFileNameW` | Soket yalnız `127.0.0.1`'e bağlanır; yol → SDL |
| `Api/RtAgentLifecycleManager.cpp` | `CreateProcessA` | Android'de alt süreç yok → "desteklenmiyor" hatası (sahte başarı YOK) |
| `Core/PlayerCamera.cpp` | `CreateProcessW` + `MapViewOfFile` (algı köprüsü) | Bu aşamada kapalı; sonra uygulama içi MediaPipe Tasks → JNI |
| `Core/PlayerMain.cpp`, `Scene/AssetRegistry.cpp`, `Backend/VulkanFsrLoader.cpp` | `GetModuleFileName` (exe klasörü) | Tek bir `Platform::appDataRoot()` (SDL_AndroidGetInternalStoragePath) |
| `Core/ProjectManager.cpp`, `Utils/SceneSerializer.cpp` | geniş karakter yollar, `CreateFileW` | `#else` dalı UTF-8 `std::filesystem`/`fopen` |
| `Backend/VulkanBackend.cpp` | `LoadLibrary`/`GetProcAddress`, `external_memory_win32` | `dlopen("libvulkan.so")` ya da doğrudan link; win32 eklentisi zaten koşullu |
| `Physics/TerrainFFT.cpp` | `GetProcAddress` (isteğe bağlı FFT) | `dlopen` ya da yok |
| `UI/PlayerOperatorLibraryUI.cpp`, `Core/PlayerOperator.cpp`, `Core/PlayerGameSettings.cpp` | GDI kapak resmi, operatör | Operatör katmanı Android'de derlenmez |
| `Utils/PerfProfile.cpp` | **korumasız** `GetCurrentProcess`/`K32GetProcessMemoryInfo` | `#ifdef` + `/proc/self/statm` |
| `Backend/VulkanBufferAddressLedger.cpp` | **korumasız** `windows.h` + `GetCurrentProcess` | `#ifdef` + `getpid` |
| `Core/RuntimeState.cpp`, `Backend/VulkanViewportBackend.cpp`, `Utils/ImageCodecs.cpp` | yalnız include | koruma zaten var |

### Windows'a özgü link kütüphaneleri

`DirectXTex`, `openvdb`+`blosc`+`tbb`, `OpenEXR`/`Imath`, `libssl`/`libcrypto` (uzak TLS IPC), `Crypt32`, `Ws2_32`.
Balon sahnesi bunların hiçbirine ihtiyaç duymaz → Player'ın **host stub** deseniyle Android hedefinde dışarıda kalır
(stub'lar başarı taklidi yapmaz: açık hata/tek seferlik log). PNG/JPG için `SDL2_image` (Android derlemesi var) yeterli.

## Girdi: dokunmatik

`AppRuntime::PlayerBalloonInput` (`Core/PlayerBalloonInput.cpp`) SDL fare olaylarını ekran-normalize
koordinatla `Paint::balloonPointerCommand`'a (`down/move/up/wheel/cancel`, `owner="mouse"`) veriyor.

- **İlk gün:** SDL Android'de tek parmağı varsayılan olarak fare olayına çevirir (`SDL_HINT_TOUCH_MOUSE_EVENTS`).
  Yani tek parmakla tut-sürükle-patlat **kod değişmeden** çalışmalı.
- **Doğru hâli:** `SDL_FINGERDOWN/MOTION/UP` → aynı komut, `owner="finger:<id>"`, normalize koordinat zaten
  `tfinger.x/y`'de. İki parmak sıkıştırma → mevcut `wheel` (sıkıştırma) eylemi. Fare sentezi kapatılır
  (`SDL_HINT_TOUCH_MOUSE_EVENTS=0`) ki her dokunuş iki kez gelmesin.
- ★ Sınır: etkileşim çekirdeği şu an **tek tutulan balon** taşıyor (`Paint::heldBalloon()` tek değer).
  Çoklu parmakla aynı anda iki balon tutmak çekirdek değişikliği ister; ilk sürümde ikinci parmak sıkıştırmadır.
- Kural §1: dokunma yolu yeni bir yetenek değil, mevcut `paint.wet_demo` pointer komutunun yeni bir kaynağıdır;
  IPC'den aynı komutla sürülebilir — ayrı API gerekmez.

## Derleme sistemi

- Yeni klasör `android/` (Gradle + `externalNativeBuild` CMake, NDK r26+, `arm64-v8a`, minSdk 29).
- `CMakeLists.txt` kaynak listesi **üretilir**: `scripts/gen_player_android_cmake.py` —
  `gen_player_vcxproj.py` ile aynı girdiler (Studio vcxproj + `player_include_manifest.json`) ve aynı
  `--check` bayatlık denetimi. Elle tutulan liste YOK (bkz. elle tutulan katalog dersi, CLAUDE.md §1).
- Mevcut kök `RayTrophiStudio/CMakeLists.txt` eski (CUDA'lı Studio, 2026-11-04) — Android için kullanılmaz.
- Shader'lar: mevcut `.spv`'ler (SPIR-V 1.5, Vulkan 1.2) APK `assets/shaders/` altına aynen girer.
- Varlıklar: sahne `.rtp` + `.rtp.bin` + yan dosyalar + ses APK `assets/`'te; ilk açılışta iç depolamaya
  çıkarılır, kodun geri kalanı `std::filesystem` ile aynı kalır.

## Sıralı partiler

| Parti | İş | Dokunduğu yer | Kabul |
|---|---|---|---|
| **A0** ✔ | Cihaz yetenekleri (yukarıdaki A0 sonucu): Note8 ✘, Mali-G57 sınıfı ✔, geliştirme emülatörde | — | Gerçek cihaz alınınca Player logu ile teyit |
| **A1** | `Platform/` ince katman: `appDataRoot`, `processMemory`, `loadVulkanLoader`; 2 korumasız dosyayı koru | ~19 dosya, yalnız `#ifdef` dalları | Windows Player + Studio aynen derlenir (/Zs) |
| **A2** | `gen_player_android_cmake.py` + `android/` iskeleti; stub host'lar (VDB, EXR, DirectXTex, TLS, alt süreç) | Yeni dosyalar + manifest `android_link_alternatives` | NDK derlemesi linkleniyor |
| **A3** | IPC yerel soket taşıması | `RtIpcTransportLocal.cpp` `#else` | `adb forward` + `Invoke-RtIpc` ile `scene.list_objects` |
| **A4** | APK varlık çıkarma + sahne açma | `PlayerMain` platform dalı | Gizli Renkler cihazda görüntüde |
| **A5** | Dokunma olayları (`finger:<id>`, pinch→wheel) | `PlayerBalloonInput.cpp` | Tut-sürükle-patlat, boya yüzeyde |
| **A6** | **GPU yetenek + performans profilleme → donanım tabanı kararı** (kullanıcı kararı 2026-11-21: kaliteden ödün yok; yetmeyen GPU için taban yükseltilir). A6a yetenek kapısı: `paint_wet` şartları da açılışta denetlenir, yetmeyen cihaz açık mesajla reddedilir (boyasız açılış YOK). A6b geçiş bazında GPU zamanı (timestamp: partikül, `paint_wet`, raster) IPC'den okunur. A6c **sürekli** yük: 10–15 dk ısıl kısma sonrası kare süresi, tepe değil. A6d cihaz sınıfı × 30/60 fps tablosu → taban. | Ölçüm yüzeyi (`rt.perf`), yetenek kapısı | Tek kalite, ölçülmüş taban; düşük kalite dalı yok |

★ A1 ortak dosyalara dokunur (Backend, ProjectManager, IPC). Codex bugün Player/balon dosyalarında aktif —
A1'e başlamadan `AJAN_KOORDINASYON.md` §1'e satır yazılır, dokunulan dosyalar taze okunur.

★ En sinsi başarısızlık adayı: `paint_wet` pipeline'ı SSBO sınırı yüzünden kurulamaz ve oyun **boyasız ama
hatasız** açılır — balonlar patlar, damlalar uçar, duvarda iz kalmaz. A4 kabulü "duvarda boya var" olmalı,
"sahne açıldı" değil; GPU boya hazırlığının `GPURequired` hatası cihazda log'da aranır.
