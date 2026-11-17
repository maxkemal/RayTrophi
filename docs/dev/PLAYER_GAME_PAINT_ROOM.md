# Player ilk oyun — Boyama Odası (AVM etkileşimli oyun alanı)

> **Durum:** AKTİF — 2026-11-16: adım 0–4 Studio/Player IPC kabulü CANLI; ilk poz/jest + kural/oturum çekirdeği Player'da 4/10 çocukla tam oturum IPC PASS. Rig/IK, fiziksel vekiller, Studio kural paneli, özel balon/görsel final, kamera ve kiosk açık. İlk ürün hedefi: AVM’lerde 6–10 yaş etkileşimli oyun alanı; mevcut viewport/runtime ve GPU partikül + çok yüzeyli ıslak boya kullanılır.

Üst plan: [RAYTROPHI_PLAYER_PLAN.md](RAYTROPHI_PLAYER_PLAN.md). Partikül olay tasarımı:
[PARTICLE_SYSTEM_GPU_ROADMAP.md](PARTICLE_SYSTEM_GPU_ROADMAP.md) §9. İlk kare kabulü:
[PLAYER_STATIC_VIEWER_ACCEPTANCE.md](PLAYER_STATIC_VIEWER_ACCEPTANCE.md) (2026-11-14 PASS).

## Devir notu

- **2026-11-17 — ortak oda hedefi şovu KAYNAK HAZIR (derlenmedi, /Zs PASS):** `room_goal`
  (vars. 0,5), %25/50/75 konfeti, %100 yağmur + balon kaskadı + `finale_reset_seconds` sonra
  temiz oda; `celebrate_sound`. Test `rt_room_goal_show_ipc.py`. Açık: "Make Unique For
  Paint" IPC'de yok; damla başına leke boyu global.

- **2026-11-17 — canlı palet + sürpriz balon KAYNAK HAZIR (derlenmedi, /Zs PASS):**
  Varsayılan 8 canlı renk, `colors` ile 2..8 düzenlenebilir. Sürpriz: `surprise_chance`
  (vars. 0,1), altın metal balon, gökkuşağı boya (1,5× damla), boya bırakmayan
  `BalloonConfetti` CPU konfetisi, `surprise_sound`. Oyun kurgusu kararı (kullanıcıyla):
  ceza/can kaybı YOK, çocuk başına puan YOK; ortak oda hedefi + kutlama şovu sıradaki aday.
  Test `rt_surprise_balloon_ipc.py`.

- **2026-11-17 — oda sıfırlama + balon boyu KAYNAK HAZIR (derlenmedi, /Zs PASS):**
  `paint.wet_demo reset` (boya + balonlar), boşta sıfırlama (`idle_reset_seconds`),
  oturum sonu `room.reset`, Player F9 görevli tuşu; `size_min/size_max` ile boy, damla
  sayısı boy², ses perdesi boyla. Damla başına leke boyu açık iş (global splat yarıçapı).
  Test `rt_balloon_reset_size_ipc.py`. Sıradaki öneri: çocuklar için efekt turu (ölçümle).

- **2026-11-16 gece 3 — damla collider'ı (SDF dahil) + balon başına ses KAYNAK HAZIR
  (derlenmedi, /Zs PASS):** Collider yeni sistem değil, mevcut parçacık collider ataması;
  Game Design dört damla sistemine aynı collider'ı yazar (`set_collider`). SDF pişerken demo
  bekler. Ses: `audio.play_at` (bağımsız konumlu tek-atış, 16 eşzamanlı) + demo `pop_sound`.
  Ayrıca bu turda: Game Design ayrı sekme, kaldırılan yüzey dokusunu bırakır (beyaz top
  kökü), listeler kaydırmalı. Test `rt_balloon_collider_sound_ipc.py`.

- **2026-11-16 gece — oyun tasarımı PROJEYE kaydediliyor + Studio "Game Design" KAYNAK HAZIR
  (derlenmedi, /Zs 13/13 PASS):** Kanonik çalışma sahnesi `pastel-paint-workshop/workshop.rtp`
  (kullanıcı). Proje `game_design` taşır: ıslak boya yüzeyleri+ayarları, splat kuralları,
  balon demosu (yüzey listesiz). Tek kopya canlı durum; yükleme ortak sim saatinin ilk
  tikinde uygulanır (Studio=Player). Studio Paint panelinin altında "Game Design" bölümü
  yalnız rtapi çağırır. IPC `paint.wet_design`. Test `rt_game_design_roundtrip_ipc.py`.
  Açık karar: `workshop.rtp` kaydedilip yan dosyalar (`.balloons.json`, paint-profile)
  emekliye ayrılsın mı. Sıra `NEXT_BUILD_CHECKS.md` başında.

- **2026-11-16 akşam — tıkla-patlat + kanal ayarı KAYNAK HAZIR (derlenmedi):** Player'da sol
  tık imlecin altındaki balonu patlatır (`paint.wet_demo pop`, IPC ile aynı çekirdek).
  Roughness/kabartma lekenin gövdesine daraltıldı, normal varsayılanı 0.5, kuruma varsayılanı
  12 sn. Player kanal kabulü 5/5 ve maliyet ölçümü (kare maliyeti gürültü içinde) aynı gün
  CANLI; ilk kanal açılışında bir kez görülen 7,2 sn takılma ve merkezdeki beyaz kürenin kökü
  AÇIK. Sıra `NEXT_BUILD_CHECKS.md` başında.

- **2026-11-16 — GPU boya PBR kanalları KAYNAK HAZIR (Codex başladı, Claude bitirdi):**
  ıslak boya artık isteğe bağlı roughness ve height→normal üretir (mevcut Paint kanal
  dokuları ve height→normal kuralı; yeni boya modeli yok). Roughness yalnız boyalı alanda
  değişir. Varsayılan kapalı; `paint.wet_set_settings`. Shader derlendi, C++ derlenmedi,
  canlı kabul yok. Kontroller `NEXT_BUILD_CHECKS.md` başında; test
  `rt_wet_paint_channels_ipc.py`. Atölye profili kanalları açmıyor; maliyet ölçülmedi.

- **2026-11-16 — Kullanıcı pastel oyun atölyesini seçti / kaynak paket HAZIR:**
  Yoğun oyuncak.rtp stres testidir; hedef oyun odası değildir. Yeni kanonik ilk
  sanat/sahne paketi `RayTrophiStudio/assets/scenes/pastel-paint-workshop/`:
  8×8m açık/boş merkez, üç geniş boya duvarı ve zemin, birkaç köşe oyuncağı;
  32 flat mesh / 17.200 üçgen, dört ayrı tam 0–1 UV, 12 parlak gerçek küre.
  Ortak wet_demo opsiyonel visual_nodes ile authored mesh poz/renk/gizleme/respawn
  kullanır; damlalar mevcut GPU yolunda, eski fixture particle modu korunur.
  Kullanıcı build + canlı mesh pop/respawn/renk/poz ve dört yüzey boya PASS:
  tekli/üçlü 1800/5400 damla; taşma/host byte/device-loss0. Asset roughness .18
  encoding'i düzeltildi; yeniden build gerekmedi. Wet12s profili canlı uygulandı,
  8.saniyede üç duvar da ıslak. GPU height/normal/roughness kanal bağlantısı henüz
  yok; mevcut Paint yolu yeniden yazılmadan bağlanacak, A'ya pano sorusu yazıldı.
  [Kabul kanıtı](PLAYER_WORKSHOP_ACCEPTANCE_2026_11_16.json).
  Yeni shader/renderer/fizik yok.
  Ayrık ince duvar collider'ları nedeniyle bu ilk odada SDF zorunlu değil.
  64 kaynak/tile taşması ve genel boya beklemeleri hâlâ açık.
  [Paket ve kabul](../../RayTrophiStudio/assets/scenes/pastel-paint-workshop/README.md).


- **2026-11-16 — Sentetik input → gerçek balon/GPU boya CANLI 3/3 PASS:**
  Otomatik demo kapalı; tek arms-up kenarı → tek pop/1800 damla → dört yüzeyde
  boya. Nötr/held-high ek pop yok; kayıp/host indirme/device loss 0. Bu kabul
  jestle rastgele balon seçimini ölçer; uzaysal el teması/gerçek kamera/rig/visual
  veya controller kabulü değildir. [Rapor](PLAYER_BALLOON_INPUT_ACCEPTANCE_2026_11_16.md).
  Kullanıcı kamera adımını şimdilik bekletti, donanım bağımsız §5.1 yönünü teyit etti.
  Yeni motor yeteneği veya C++/shader/build yok; mevcut ortak hizmetler sınandı.


- **2026-11-16 — Güncel öncelik / kullanıcı kabulü:** Player'da kayıtlı graph/state
  machine iki klip arasında otomatik geçiyor ve mesh düzgün (kullanıcı görsel PASS).
  Pencere resize/F2 çözünürlük ayrımı kullanıcı PASS. Bunlar doğrudan anim.play
  controller skin/crossfade veya Space testinin kabulü değildir. Yoğun sahne GPU
  ~60ms, shadow atlas~33ms; Player/Studio GPU maliyeti aynı düzeyde. Uncapped CPU
  ölçümü pacing ile değişiyor; saf hesap farkı diye raporlanmayacak.
  Sonraki balon kabulü sentetik el/beden girdisi -> balon -> damla/boya zinciri,
  sonra tek kişi gerçek kamera ve çoklu kişi/oturum. Adım5–7 kısmi; kamera/IK-domain/
  kiosk/kalibrasyon/uzun koşu açık. [Güncel devir](PLAYER_HANDOFF_2026_11_16.md).


- **2026-11-15 — C2 dış rig/domain KAYNAK HAZIR, build/kabul bekliyor:**
  Kullanıcı yönüyle mevcut rig IK → authored kinematik proxy → domain yolu yeniden
  kullanıldı, ayrı Jolt iskeleti kurulmadı. Script/IPC, kayıpta/güven düşüşünde çözüm
  kesme, hareket geçmişi reseti ve host maliyet sayaçları birlikte teslim edildi.
  [PLAYER_EXTERNAL_BODY_RIG.md](PLAYER_EXTERNAL_BODY_RIG.md). Bu kaynak teslimi önceki
  C/B canlı kabulünü C2 için genişletmez; GPU partikül proxy tüketimi hâlâ açık.

- **2026-11-15 — C/B ilk paket kullanıcı Studio/Player build + CANLI PASS:**
  36 s / 4 ve 10 çocuk: tam oturum, 29/71 başarılı eylem, sıfır eylem hatası/olay kaybı,
  boya %16,43/%32,48. Player 28 / Studio 21 hedefli IPC + 9 Python binding kontrolü PASS.
  Gerçek flat zemin temasından grup eylemi, timer ve kapsamayla erken final doğrulandı.
  Native bağımsız assert testi çalıştırılmadı. RPC/takvim sıçramaları sürüyor; performans
  kapısı açık. Kanıt [PLAYER_BODY_GAME_ACCEPTANCE_2026_11_15.json](PLAYER_BODY_GAME_ACCEPTANCE_2026_11_15.json).
  Bu kabul turunda ürün C++/shader değişmedi, yeni build gerekmez.

- **2026-11-15 — Poz/jest + kural/oturum KAYNAK HAZIR (Codex, C/B devralma):**
  Ortak Input/Game çekirdeği, script/IPC ve 36 s sahte oturum planı eklendi.
  Derleme/canlı kabul henüz yok; rig/IK, fiziksel vekil, Studio kural paneli ve özel
  balon/görsel final açık. Sözleşme ve kontrol: [PLAYER_BODY_GAME_RUNTIME.md](PLAYER_BODY_GAME_RUNTIME.md).

- **2026-11-15 — Balon demosu / kullanıcı Player build'i sonrası IPC CANLI PASS (Codex):**
  Sürekli spreyin yanına yeni test fixture'ı eklendi: `docs/dev/fixtures/balloon-paint-room/balloons.rtp`
  + `.balloons.json` yan dosyası. 12 rastgele konumlu/tek renkli balon; 2.5–5 s aralıkla
  1/2/3 balon, balon başına 1800 fiziksel damla; çarpışmada boya/ölüm, 2.8 s ömür,
  3.5 s sonra yeni balon. Dört 1024² yüzeyin atlası yüklemede tek GPU layout olarak
  hazırlanır; olaylar atlası yeniden kurmaz. UV bake flat TriangleMesh/DNA okur.
  Ortak `BalloonPaintDemo` çekirdeği; script `rt.paint.wet_demo` / IPC `paint.wet_demo`
  configure/status/start/stop/burst. Player yükleme 959.71 ms (demo 352.04, bunun
  atlası 317.21 ms); tekli/üçlü doğum 0.51/3.47 ms, kontrollü olay dt maks 10–11 ms.
  Aktif ek pencerede ~168 fps / dt maks 9.18 ms. Uzun idle aralıklarını kapsayan
  kayıtta 64–72 ms dt / sim maks 48.15 ms sıçrama da var: koşulsuz sıçramasızlık veya
  ≤2 ms hedefi PASS sayılmadı. 170.82 oyun saniyesi / 46 olayda kapasite sabit,
  hata/device loss/düşen kaynak/host boya baytı 0; ilk kabul kapsaması %41.19.
  Tekli/üçlü tetik, damla bitişi ve 12 balon kalması, gerçek çözünürlük, validation,
  rastgele olay/kapsama/GPU yayın kabulü PASS. Bu tur görsel screenshot kabulü yapılmadı.
  Rapor: [PLAYER_BALLOON_PAINT_DEMO.md](PLAYER_BALLOON_PAINT_DEMO.md),
  [ham kabul](PLAYER_BALLOON_PAINT_ACCEPTANCE_2026_11_15.json).
  Tekrar: `python scripts/test/rt_player_balloon_paint_ipc.py --seconds 30`.
  **Devir / sıradaki:** kare başı istatistik synchronize/readback ve yüzey başı GPU
  kopya beklemeleri; idle/aktif sıçramaları ayrı profille. 64-kaynak/tile sınırı
  bu koşuda kayıp vermedi, genel düzeltmesi hâlâ açık. OBB ve spread bulguları da
  düzeltilmedi; demo AABB/PlaneY ve doğrudan küresel hız dağılımı kullanır.
  Genel kural motoru/oturum, çocuk algısı, final ve ses bu sınırlı demo ile kapanmaz.


- **2026-11-15 — Player boyama odası ölçümü (Claude):** `rt_player_wet_paint_room_ipc.py` (fixture Studio'da,
  koşu Player'da). 8×8 m oda, zemin + 3 duvar (düzlem, ayrı materyal), 4 renkli sprey (~6900 canlı partikül),
  die + olay kaydı. Player 1680×945, sürekli mod:
  | | taban (boyasız) | boya 512² ×4 | boya 1024² ×4 |
  |---|---|---|---|
  | rendered fps | 263 | 201 | 165 |
  | advance ms (host adımı) | 1.57 | 2.77 | 3.77 |
  | frame_cost ms (render) | 2.23 | 2.19 | 2.26 |
  | tick ms ort./maks. | — | 0.79 / 1.08 | 1.59 / 3.38 |
  | aktif / işlenen döşeme maks. | — | 337 / 527 (1024'ten) | 939 / 1650 (4096'dan) |
  Render maliyeti değişmiyor; boya maliyeti host adımında: 512'de +1.2 ms, 1024'te +2.2 ms (≤2 ms hedefi
  sınırda). Yayın `device_copy` (host 0 B), olaylar cihazdan (host olayı 0), ekranda 4 renk. 15 s'de kapsama
  %23.5 (zemin %63). **Kaldıraçlar (sırayla):** (1) her tick'te `synchronize` + istatistik indirmesi = CPU↔GPU
  durması → bir önceki karenin istatistiğini asenkron oku; (2) yayın, yüzey başına kare başına `waitIdle` +
  kopya (Player'da ~4×fps kopya) → tek komut tamponunda toplu, waitIdle'sız bariyer; (3) `sources_dropped`
  7–9 bin / 15 s: döşeme başına 64 kaynak tavanı yoğun spreyde boya kaybettiriyor → tavanı büyüt ya da
  kaynakları döşeme başına önceden kovala. Akma yönü görsel olarak doğrulandı (sol duvarda leke altına iz).
  **Partikül sisteminde iki ayrı bulgu (ıslak boyadan bağımsız):** ölçeklenmiş/taşınmış kutunun `obb`
  çarpıştırıcısı odanın ortasını kapsıyor (partiküller nozulda ölüyor; `aabb` doğru: sınırlar −4.21…−4.01),
  ve `spread=14` neredeyse her yöne saçıyor (−Z nişanlı spreyin olaylarının %0.3'ü arka duvarda).
- **2026-11-15 — adım 3–4 CANLI PASS Studio 8/8 (Claude):** `rt_wet_paint_gpu_ipc.py` — ekranda boya
  (material modda kırmızı piksel sayımı), RT render cihazına GPU içi kopya, ayrı-cihaz önizlemeye host
  yolu + fırçanın materyal başı senkronu. Canlıda bulunan 6 kök (sırayla): sim compute bağlamı GPU
  isteyen yoksa CPU'ya dönüyor (→ kendi Vulkan bağlamı); `downloadBuffer` kayıtlı dispatch'leri
  göndermiyor (→ önce synchronize); damga CPU tohumunu kullanıp fırça vuruşunu atlıyordu (→ birikim +
  `fresh_bias`/pickup, `pickup_rate` alanı); gather araması kuru texel'in kendi iziyle başlıyordu (→ sabit
  adım); test nesneleri tek materyali paylaşıyordu (→ `wet_add_surface` reddeder); host yolu materyal
  senkronunu tetiklemiyordu (→ `updateBackendMaterial` materyal başı). Açık: Player fixture kabulü;
  "kırmızı+mavi=mor" hedefi (CPU modelinde pickup ~%1) ürün kararı.
- **2026-11-15 — adım 3 YAZILDI, derlenmedi (Claude):** GPU ıslak boya, ayrıntı §5.3. Yazılanlar:
  `shaders/paint_wet.comp` (derlendi; .spv iki yerde), `Paint/WetPaintGpu.*` (atlas, bake, tick,
  yayın), ortak sim saatinden tick (`RuntimeSimulationClock`), Studio (`Main.cpp`) ve Player
  (`RuntimeSceneViewer`) yayını, Vulkan `copyDeviceBufferToTexture`, partikül
  `deviceCollisionEventView` / `takeHostCollisionEvents`. Adım 4'ün IPC yüzü de hazır: 11
  `paint.wet_*` metodu + Python + yetki + tarif + Player izin listesi. /Zs 13/13, IPC denetimi
  PASS. Kabul: `rt_wet_paint_gpu_ipc.py` (Studio, boş sahne). Kontrol sırası: NEXT_BUILD_CHECKS.
  **Sıradaki:** derleme → kabul. Sonra Player fixture'ında aynı test, ardından yayının
  `waitIdle`'sız bariyere taşınması (ölçüm pahalı derse), sonra kiosk/erişim katmanı.
- **2026-11-15 — adım 2 CANLI PASS (Claude):** GPU/CPU olay sayısı birebir (4214), konum hatası
  < 1e-7 m, normaller ≥ 0.9999999; bounce'ta zeminin altında partikül yok. Yazılanlar: GPU partikül çarpışması
  (`sim_particle_collide`, `ParticleDeviceCollision.cpp`) + sistem düzeyi `collision_response`
  (bounce|die) + cihazda olay tamponu (`ParticleCollisionEvent`, 48 B; tüketilene dek birikir).
  CPU referansı aynı davranışı aldı. IPC `particle.get_collision/set_collision/collision_events`,
  Python, panel, kayıt. Adım 3 (GPU ıslak boya) bu olayları tüketecek: `readCollisionEvents`
  yerine cihazda doğrudan `resident_buffers_.collision_events` + `collision_counter` okunmalı
  (CPU'ya inmeden), tüketim = sayaç sıfırlama. Kabul: `rt_particle_gpu_collision_ipc.py`.
  Önceki adımlar (ortak sim saati, materyal/partikül/scatter/VDB/bulut, ekran+FPS) CANLI.

## 0. Ürün bağlamı — bu bir PC oyunu değil

İlk oyunlar AVM'deki kapalı oyun alanlarında koşacak: bir oda, duvara/zemine yansıtma
ya da büyük ekran, aynı anda birkaç çocuk, gün boyu gözetimsiz çalışma. Bu bağlam
demodan çok **motorun** gereksinimlerini belirler:

| Bağlam | Motor gereksinimi |
|---|---|
| Aynı anda 4–10 çocuk | N kişilik poz girişi, kalıcı kişi kimliği, örtüşmede (occlusion) kişiyi kaybetmeme |
| 6–10 yaş, okuma yok | Talimatsız oynanış: ilk hareket anında görsel + ses cevabı; başarısızlık durumu yok |
| Oturum (3–5 dk) ve sıra | Oturum durum makinesi: bekleme (attract) → oyun → final → sıfırlama |
| Gün boyu gözetimsiz | Cihaz kaybında kendini kurtarma (Player'da OptiX cankurtaranı yok), watchdog, otomatik başlatma, kayıt |
| Birden çok mekân | Uzaktan izleme: mevcut TLS IPC (`RtIpcSecurity`) zaten var; sağlık/oturum sayaçları oradan okunur |
| Yansıtma / ekran | Çoklu çıkış, projeksiyon bükme (warp) + kenar karıştırma; kamera ↔ oda ↔ projektör kalibrasyonu |
| Çocuk görüntüsü | Varsayılan: görüntü yalnız bellekte, kayıt/ekran görüntüsü yok; ailelere verilen çıktı yalnız **eser** |
| Operatör (oyun alanı görevlisi) | Başlat/durdur/ses/zorluk/içerik seçimi — IPC'den, üstüne basit operatör ekranı |

### 0.1 İki gösterim modu (aynı çekirdek)

- **Ayna/pencere modu:** büyük ekran ya da tek duvar yansıtması. Çocuk kendini maskeli
  video kartı olarak sanal odada görür. Kullanıcının ilk tarifindeki mod.
- **Oda modu (projeksiyon eşleme):** gerçek odanın duvarları ve zemini tuvaldir. Sanal
  oda fiziksel odanın kalibre kopyası; video kartı gerekmez, çocuğun gerçek bedeni oradadır.
  Boya gerçek duvarda akar. Mevcut oyunlardan ayrışmanın en güçlü yolu budur.

Ortak olan her şey (poz, kurallar, partikül, boya, ses) iki modda aynı; fark yalnız
kamera/çıkış eşlemesi ve video kartının açık olması.

### 0.2 Erişim seviyeleri ve kurulum profili (karar 2026-11-15)

Genel bir motor katmanı, demoya özel değil:

| Seviye | Kim | Erişim | Yol |
|---|---|---|---|
| 0 Oyuncu | çocuklar | hiçbir şey (tuş, menü, HUD yok) | kiosk modu varsayılan |
| 1 Görevli | oyun alanı personeli | oturum başlat/durdur, ses, içerik seçimi, sıfırla | **uzaktan**: telefon/tablet, yalnız oturum+ses yetkili rol token'ı (TLS IPC) |
| 2 Teknisyen | kurulum/test | kalibrasyon, ekran, tanılama HUD, log, test modları | **ekranda** gizli tuş kombinasyonu + PIN ile kurulum katmanı (ImGui) + uzaktan |
| 3 Geliştirici | biz | her şey | yerel IPC |

- **Kurulum profili** `kiosk.json` (projeden ayrı, kuruluma ait): ekran, ses, kalibrasyon,
  içerik listesi, kilit modu, hash'li PIN'ler. `--profile` ile; alanlar IPC'den rol yetkisiyle.
- **Girdi kapısı:** ★ 2026-11-14'te eklenen F1–F11 operatör tuşları şu an HERKESE açık; kiosk
  modunda kapalı, yalnız teknisyen kilidi açıkken çalışmalı. Saha demosundan önce şart.
- **Rol yetkileri:** mevcut `RtIpcSecurity` yetki bitleri + uzak token; Görevli / Teknisyen
  hazır yetki setleri.
- Sıra kararı (kullanıcı): önce oynanış (ses tick'i → adım 3 ıslak boya), kiosk katmanı ilk
  saha demosundan önce.

### 0.3 Ses

Ses sistemi (sahne sesi, mikser, transport, fizik kaynaklı ses, hava tepkileri, gök gürültüsü,
akustik) Player projesinde derleniyor ama **Player'da ilerletilmiyor**: tick'i
(`audio::updateListener`: transport + dinleyici + akustik + gök gürültüsü + hava tepkileri)
panel dosyası `UI/AudioUI.cpp`'de; sim saatiyle aynı kök. Çözüm: gövde çekirdeğe, panel ve
Player aynı fonksiyonu çağırır; Player transport'u oyun saatiyle sürer. **Kaynak hazır (2026-11-15):**
`audio::tickRuntime` (Scene/AudioRuntime.cpp); **CANLI PASS 01:23**: transport oyun saatiyle,
pozlar çözülüyor, Player IPC `audio.play` ile cihaz `ready` + kaynak `playing`. Manuel kaynaklar
çalınana dek sessiz (Studio'da da aynı). Oyun sesleri (balon
patlaması, boya sıçraması) mevcut ses API'siyle, yeni ses sistemi yazılmadan.

## 1. Rakiplerden ayrışma — ne "daha etkileşimli" demek

Mevcut etkileşimli zemin/duvar sistemleri çoğunlukla 2B: dokunma noktası → önceden
çizilmiş animasyon. Bu motorun farkı **fiziğin gerçekten çalışması**:

- Boya bir sprite değil: akar, damlar, renkler ıslakken karışır, emilir, kurur
  (mevcut ıslak boya modeli). Aynı hareket iki kez aynı sonucu vermez.
- Patlama bir animasyon değil: partiküller sahneye çarpar, nereye çarptıysa iz bırakır.
- Çocuğun bütün bedeni etkileşir (iskelet + vekiller), yalnız temas noktası değil.
- Oda birikir: oturum boyunca çocukların ortak eseri büyür ve finalde canlanır.

## 2. Demo v1 — oturum akışı

1. **Bekleme (attract):** oda sakin, birkaç balon süzülür. Çocuk alana girer girmez
   bedeninin çevresinde kendi renginde hafif bir iz/parıltı belirir — talimatsız ilk cevap.
   Her çocuğa bir renk atanır.
2. **Oyun:**
   - Balon patlat → balonun renginde partikül patlaması → çarptığı yerde ıslak leke.
   - **El boyası:** duvara yakın el (oda modunda gerçek duvar) kendi renginde damla bırakır.
   - **Ayak boyası:** zemine sert basış (aşağı hız eşiği) zeminde sıçrama bırakır.
   - **Islakken karışım:** kırmızı lekeye mavi gelirse mor — ıslak modelin pickup rezervuarı.
   - **Özel balonlar:** dev balon yalnız 2–3 çocuk birlikte dokununca patlar (iş birliği);
     gökkuşağı balonu çok renkli; **su balonu** boyayı yıkar/akıtır (ıslak modelin runoff'u).
3. **Ortak hedef:** odanın boyanmış oranı (atlas üzerinde GPU indirgeme) bir çubukla
   gösterilir; eşiklerde oda tepki verir (müzik katmanı eklenir, ışık değişir).
4. **Final:** eşik ya da süre dolunca boyalı bölgeler **canlanır** — mavi lekelerden
   balıklar, yeşilden çiçekler (mevcut scatter/partikül/animasyon; boyanmış maskeyi
   yoğunluk kaynağı olarak kullanır). Konfeti. Odanın eseri (çocuk görüntüsü olmadan)
   çıkış ekranında / QR ile gösterilir.
5. **Sıfırlama:** boya kurur ve solar, oda bekleme durumuna döner.

## 3. Platform ilkelleri — demo değil, sonraki oyunların da kullanacağı katman

Demoya özel kod yazmak yerine her oyunun kullanacağı ilkeller kurulur. Python Player'da
yok; oyun mantığı **veri olarak kural**dır ve CLAUDE.md §1 gereği tamamı IPC'den
okunur/yazılır, Studio'da panelden düzenlenir.

| İlkel | Örnek |
|---|---|
| **Tetik** | temas (vekil × nesne), bölge giriş/çıkış, hareket (basış, kollar yukarı), grup (N kişi aynı nesnede), eşik (kapsama %), zamanlayıcı, partikül olayı |
| **Eylem** | emitter tetikle (renk/yön parametreli), boya damlası, ses çal, animasyon/state tetikle, nesne göster/gizle/doğur, durum değiştir |
| **Durum** | oturum durum makinesi (bekleme/oyun/final/sıfırlama), sayaçlar, kişi başı renk |
| **Ölçüm** | kapsama oranı, kişi sayısı, olay sayaçları, kare süresi — hepsi IPC'de |

İlk oyun ve sonrakiler (balık tutma, top havuzu, ritim zemini…) bu tabloyla kurulur.

## 4. Bugünün gerçeği (2026-11-14 ölçümü, kodu okuyarak)

| Konu | Durum | Kanıt |
|---|---|---|
| Player sahne + görüntü | **PASS** (EchoRoomsLab, 26 mesh, device-loss yok) | kabul belgesi |
| GPU partikül yolu | Yalnız **çarpıştırıcısız**, self-collision'sız, domain'siz sistem device-resident. Çarpıştırıcılı sistem CPU referansında. | ROADMAP §2; `sim_particle_ballistic.comp` 47 satır, çarpışma bağlaması yok |
| GPU collider | Var ama **ızgara sorgusu**: `Fluid::ColliderGpuQuery` mesh BVH'ini cihazda tutar, kafes noktalarına uzaklık/SDF döndürür. | `include/Fluid/ColliderGpuQuery.h` |
| Partikül olayları | `OnCollision` / sub-emitter / surface splat **tasarımda var, kodda yok**. | ROADMAP §9 |
| Jolt temasları | Var: `RigidBodySystem::contactEvents()` | `include/RigidBodySystem.h:397` |
| Islak boya modeli | Var, CPU: `noteWetDab` + `tickWetPaint` (difüzyon, akma, emilme, damla başı, uç birikimi, pickup rezervuarı). | `src/Paint/MeshPaintAdapter.cpp:1909` |
| Islak boya tick'i | **Panelde** (`scene_ui_modifiers.cpp:6202`, ImGui DeltaTime). Player'da ilerlemez. | — |
| Islak boya ölçeği | Adaptör başına **tek** `active_region`; Studio tek nesne boyar; adaptör `Triangle` facade'dan kuruluyor (`RtApiPaint.cpp:85`). | `MeshPaintAdapter.h:122` |
| Boya API'si | Katman işlemleri var; **dünya noktasına damla** yok. | `RtApiPaint.cpp` |
| Rig sürüşü | Klip/state machine/graf parametresi var; **dış poz girişi yok**. `RigChainIK` var. | `RtApi.h:3151–3247` |
| Görüntü kanalı | IPC JSON; paylaşımlı bellek yok. | — |
| Çoklu çıkış / warp / kalibrasyon | Yok (tek SDL penceresi). | — |
| Cihaz kaybı kurtarma | Player'da yok (plan §4 riski). | — |

## 5. Mimari

```text
 [Algı köprüsü — harici süreç; herhangi bir kamera (1..N) + herhangi bir kütüphane]
     │ IPC (değer): input.set_body_poses {kişi id, eklem[], güven, t}  (N kişi)
     │ Paylaşımlı bellek (görüntü, yalnız ayna modu): RGBA, alpha = kişi maskesi
     ▼
 [Player — ortak kare döngüsü, sabit adım, oturum durum makinesi]
   girdi → anim.set_external_pose → rig/IK → el/ayak/gövde kinematik vekilleri
   kural motoru: tetikler (Jolt temas, bölge, hareket, grup, kapsama, partikül olayı)
                 → eylemler (emitter, boya, ses, anim, göster/gizle, durum)
   GPU partikül: balistik + statik sahne SDF + analitik kapsüller → olay buffer'ı (cihazda)
   GPU ıslak boya: olay + el/ayak damlaları → çok yüzeyli atlas → ıslak tick → materyal
   render: RayFusion (sabit bütçe); çıkış eşleme (ekran | projektör warp/blend)
```

### 5.1 Genel kamera girişi (karar 2026-11-14)

**2026-11-16 kullanıcı teyidi:** Sistem belirli kamera/marka/SDK'ya bağımlı
olmayacak. Fiziksel kamera seçimi test kurulumunun ayrıntısıdır; motor API'sinin
ön koşulu değildir. Sentetik/replay, webcam, telefon/IP ve derinlik kaynakları
aynı kanonik poz sözleşmesine dış adaptörlerden gelir. Çoklu kaynak füzyonu,
ID/zaman birleştirme ve kalibrasyon dış köprüde; bu tur yeni IPC alanı eklenmedi.
Gerçek kamera denemesi kullanıcı isteğiyle şimdilik beklemede.


- Özel donanım istenmez. Kamera ve kütüphane (ör. MediaPipe Pose + segmentasyon; varsa
  RGB-D SDK'sı) **motorun dışında**. Motor kamerayı tanımaz. Birden çok kamera varsa
  füzyon ve kişi kimliği köprünün işidir; motor N iskelet alır.
- **D'siz konum:** kalibre zemin düzlemi + ayak piksel konumu → zemindeki X/Z.
  Derinlik kamerası kaliteyi artırır, şart değildir.
- **İskelet → mevcut rig:** `anim.set_external_pose` dış eklemleri mevcut rig'e/IK'ya
  verir. Rig görünmez olabilir; görevi vekilleri taşımak.
- **Görüntü → video kartı (ayna modu):** kişinin derinliğinde kameraya bakan kart, doku
  mevcut yükleme yoluyla (`IBackend::uploadTexture2D`) her kare güncellenir.
- **Neden iki kanal:** 640×480 RGBA @30 fps ≈ 37 MB/s; JSON/base64 ile pipe'tan geçmez.
- **Gizlilik:** çocuk görüntüsü yalnız bellekte; `viewport.get_screenshot` /
  `render.start` varsayılan olarak video kartını **gizler**; eser çıktısında hiç yok.
- **Test kazancı:** ajan sahte N kişi pozu + sahte maskeli kare ile bütün oturumu
  kamerasız sürer (sahte çocuk koşucuları). Bu oyunun QA altyapısıdır.

### 5.2 GPU partikül çarpışması

Device-resident balistik yol genişletilir; CPU referansı Studio'da doğrulama için kalır.

- **Statik sahne:** oda sahne yüklenince **bir kez** SDF'e pişirilir (`ColliderGpuQuery`
  kafes sorgusu, signed_distance). Kernel SDF'i üç doğrusal örnekler, normal = gradyan.
- **Dinamik:** vekiller ve balonlar **analitik** kapsül/küre (kare başına küçük buffer).
- **Olay:** çarpan partikül ölür ve sınırlı append buffer'a `{konum, normal, renk,
  enerji, yüzey kimliği}` yazar. Taşma sayılır, IPC'den okunur. Olaylar CPU'ya inmez.

### 5.3 GPU ıslak boya (çok yüzey, maliyet ıslak alanla)

- **Aynı model GPU'ya:** `tickWetPaint` denklemleri ve `BrushSettings` ıslak parametreleri
  compute shader'a taşınır; yeni boya modeli yok.
- **Çok yüzey atlası** + statik yüzeyler için bir kez **UV→dünya konumu/normal atlası**.
- **Döşeme aktif listesi** (32×32): damga yalnız olayın dünya yarıçapındaki döşemeleri
  açar; tick yalnız aktif döşemelerde; kuruyan döşeme düşer.
- **Damga:** aktif texel, karenin olaylarını kendi dünya konumuna göre tartar — üçgen
  arama yok, dikişler kendiliğinden doğru.
- **Kapsama ölçümü:** atlas üzerinde GPU indirgeme → oran + renk başına oran (IPC).
- **Kuruma/solma** sıfırlama için zamanla parametrelenir (oturum sonu temiz oda).
- **Studio:** fırça da aynı GPU yoluna geçer (CLAUDE.md §5); adaptör o partide flat SoA'ya.

**Uygulanan sözleşme (2026-11-15, YAZILDI derlenmedi):**

| Parça | Ne | Nerede |
|---|---|---|
| Atlas | Yüzey = (nesne, materyal), W×H sayfa; texel ↔ uv = (px/(W−1), 1−py/(H−1)), CPU haritası | `WetPaintGpu.cpp` |
| Bake (bir kez) | Üçgenler texel'e rasterize: dünya konumu, oct normal, piksel akışı (`computeTrianglePixelFlow`, yerçekimi); 2 texel gutter | `addSurface` |
| Döşeme | 32×32, dünya AABB'si; bir workgroup = bir döşeme | `GpuTile` 48 B |
| Kaynaklar | host damlaları (`paint.wet_dab`) + partikül olayları **cihazda** (aynı compute bağlamı); CPU referansındaki sistemin olayları hosttan damla olur (sayılır) | `tick` |
| Kare | damga → tüket (sayaç sıfır) → tick (aktif + 8 komşu) → çözümle (composite + "hâlâ ıslak") → döndür → küçük istatistik indirmesi | `paint_wet.comp` modları |
| Boşta | ıslak yok + kaynak yok → dispatch yok, indirme yok (`status: idle`) | `tick` |
| Yayın | kirli döşeme dikdörtgeni → materyalin BaseColor VkImage'ına cihaz içi kopya; başka VkDevice → host indirmesi (sayılır) | `publish`, `VulkanBackendAdapter::copyDeviceBufferToTexture` |
| Kapsama | GPU indirgeme: alfa ≥ 0.5 olan kaplı texel oranı, ıslak texel, ortalama renk | `paint.wet_coverage` |
| Materyal | mevcut boya hedefi (`assignTextureToChannel(BaseColor)`), paint.ensure ile aynı doku | `addSurface` |

**CPU modelinden bilinçli farklar:**
- Aşağı akış taşıması CPU'da tarama sırasında hedef piksele **yazılıyor** (scatter); bu GPU'da
  yarış demek. Cihazda her texel **topluyor** (gather): hedefi kendisi olacak yukarı akış texel'ini
  kendi akış yönünde iki sabit nokta adımıyla buluyor, ve taşımayı yalnız o texel'in CPU hedefi
  yuvarlanınca bu texel'e düşüyorsa kabul ediyor. Bu, CPU kuralının tersten okunmuş hali.
- CPU tarama sırasında güncellenmiş komşuyu okuyor; cihaz her yerde önceki adımı okuyor.
  Sonuç sıradan bağımsız.
- Damga, boyanın kendisi. CPU'da `noteWetDab` bir `paintLayerAtUV` vuruşunun eşi, ve renk
  tohumunu yalnız alfa < 0.2 olan yere atıyor. Cihazda tohum alfa-over olarak geçitsiz
  uygulanıyor. Kırmızının üstüne mavi böylece karışıyor.
- Renk katmanı bir boya KATMANI: alfa kapsamadır, özgün taban rengin üstüne composite edilir.
  Sıfırlama alfayı sıfırlar.
- Dikiş aynalaması yok: damga dünya uzayında, gutter bilineer okumayı kapatıyor.
- Uç birikimi (bead) ve yükseklik maskesi yok. CPU'da da yalnız `write_height_mask` açıkken
  çalışıyorlar. Kalınlık bir durum kanalı olarak taşınıyor.
- Simülasyon frekansı kalite ayarı yok: her kare tikliyor, dt ≤ 2.5/60.
- Bake'te kuvvet alanı yok (statik oda için yerçekimi yeterli).
- Boya oturum durumu, projeyle kaydedilmiyor. Aynı hedefte bir CPU boya işlemi kendi
  katmanlarını composite edip üstüne yazar.

### 5.4 Çıkış eşleme ve kalibrasyon

- Sanal kamera(lar) fiziksel projektör(ler)in iç/dış parametreleriyle kurulur (oda modu).
- Çıkış başına warp ızgarası + kenar karıştırma maskesi (son işlem geçişi).
- Kalibrasyon: projektöre desen yansıt, kameradan oku — köprü süreci yapar, sonucu
  IPC'den yazar. İlk sürüm: elle köşe eşleme (operatör ekranı + IPC).

### 5.5 Kiosk sağlamlığı

- Vulkan cihaz kaybı → cihazı ve sahne GPU kaynaklarını yeniden kur, oturum durumunu koru.
- Watchdog (harici): IPC nabzı kesilirse Player'ı yeniden başlat; çökme logu korunur
  (atexit dersi: FATAL logu silinmemeli).
- RT pipeline önbelleği kurulumda ısıtılır (plan §4: ilk derleme 168 sn).
- Sağlık sayaçları (kare süresi, kişi sayısı, oturum sayısı, hata) IPC + TLS uzaktan.

## 6. Sıra

| # | İş | Kabul |
|---|---|---|
| 0 | ✔ Player ilk kare | 2026-11-14 PASS |
| 1 | Ortak kare döngüsü: partikül + Jolt + anim tick tek yerde, sabit adım | Studio ve Player aynı tick; IPC `runtime` adım sayacı |
| 2 | GPU partikül: statik SDF + analitik kapsül + olay buffer'ı | 50k partikül ≤1.5 ms GPU (hedef); taşma sayacı IPC'de |
| 3 | CANLI IPC PASS 2026-11-15 (Studio + Player balon fixture) — GPU ıslak boya: atlas, döşeme listesi, damga, tick, kapsama | 4+ yüzeyde eşzamanlı leke; maliyet ıslak döşemeyle orantılı; ≤2 ms (hedef) — `rt_wet_paint_gpu_ipc.py` |
| 4 | CANLI IPC PASS 2026-11-15 (Studio + Player) — `paint.wet_dab` + kapsama okuması (3'ün IPC yüzü) | IPC ile damla → atlas/kapsama değişimi okunur |
| 5 | KISMİ CANLI PASS: veri tabanlı kurallar + oturum, boya/balon eylemleri; Studio paneli açık | 4/10 çocuk tam oturum, grup/cooldown/atomiklik IPC PASS |
| 6 | KISMİ CANLI PASS: N kişi poz/jest, analitik küre × flat AABB temasları; rig/IK ve fiziksel vekiller açık | 4/10 çocuk, gerçek zemin grup teması IPC PASS; `anim.set_external_pose` henüz yok |
| 7 | KISMİ CANLI PASS: ayak boyası, arms-up balon, süre/kapsama → final durumu; özel balon/görsel final/ses kabulü açık | 29/71 eylem hatasız; kapsamayla erken final PASS |
| 8 | Ayna modu: paylaşımlı bellek görüntü + video kartı + gizlilik | sahte maskeli kare kartta; ekran görüntüsünde yok |
| 9 | Oda modu: çıkış eşleme, warp/blend, kalibrasyon | tek projektörle köşe eşleme |
| 10 | Kiosk: cihaz kaybı kurtarma, watchdog, pipeline ısıtma, sağlık sayaçları | zorlanmış device-lost sonrası oturum sürer |
| 11 | Algı köprüsü (webcam + MediaPipe; RGB-D isteğe bağlı; çoklu kişi) | gerçek kamera ile 7'nin testi |

1–7 kamerasız ilerler; gerçek kamera ve donanım en sonda.

## 7. Açık kararlar (kullanıcı)

- ✔ **Gösterim modu (karar 2026-11-14):** ilk demo **ayna modu, büyük ekran**
  (75–86" TV ya da LED duvar + ekranın üstünde tek webcam). Gerekçe: müşterinin
  projeksiyon kalitesi bilinmiyor; AVM ortam ışığı projeksiyon kontrastını düşürür;
  kalibrasyon gerektirmez, demo her yere taşınır; çocuk kendini gördüğü için "vay"
  anı en güçlü. Oda modu mimaride kalır (çıkış eşleme bir ayar, adım 9); 8 ve 9 yer
  değiştirmez ama 9 demo için şart değildir. Ayna modunda sanal duvarlar ekran
  düzleminin arkasında durur; el boyası "ekrana uzanma" (derinlik eşiği) ile tetiklenir.
- **Hedef donanım:** AVM kurulumunda GPU sınıfı (kare bütçesi buna göre).
- **Aynı anda azami çocuk sayısı** ve oturum süresi.

## 8. Riskler

- **Paralel ajan:** Codex `ParticleSimulation.*` ve partikül render köprülerinde
  (AJAN_KOORDINASYON §1). Adım 2 aynı dosyalara dokunur → başlamadan iş bölümü.
- Webcam tabanlı çoklu kişi takibinde örtüşme ve kimlik değişimi — köprü sorunu, ama
  oyun kuralları kişi kaybını tolere etmeli (renk yeniden atanabilir).
- SDF çözünürlüğü ↔ ince eşyalar; atlas boyutu × yüzey sayısı bellek bütçesi.
- Oda modunda projektör gölgesi (çocuk ışığı keser) — tasarım bunu oyun öğesi yapabilir.
- Video kartının sahne ışığıyla ilişkisi (unlit mi, ortamla çarpılmış mı) — görsel karar.
- `material_count=1` şüphesi (kabul belgesi) — renkli materyalli sahnede doğrulanmalı.
