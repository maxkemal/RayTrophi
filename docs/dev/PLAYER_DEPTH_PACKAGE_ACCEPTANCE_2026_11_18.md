# Player derinlik ve algı paketi kabulü — 2026-11-18

**Paketleme güncellemesi:** Tek güncel dağıtım `dist/RayTrophiPlayer/`.
Sesler dahil yeniden paketlendi; önceki depth klasörü `dist/.archive/` altındadır.
Aşağıdaki NoSceneAudio/377.5 MB kaydı ilk test paketinin tarihsel bilgisidir.
Güncel ortak süreç: [PLAYER_PACKAGING.md](PLAYER_PACKAGING.md).

Kullanıcının derlediği, açık workshop balon sahnesinde **sentetik Vulkan derinliği,
balon/damla görsel örtmesi ve paketli MP4 süreç yaşam döngüsü PASS**.
Codex C++ build çalıştırmadı, uygulama başlatmadı. Yeni C++ build beklenmiyor.

## Kabul

- 96×128 bellek içi ovalin yarılarına kamera-ileri -2 / +2 metre ofset verildi;
  ikinci evrede işaretler ters çevrildi. Yakın yarı balonları örttü, uzak yarının
  önündeki balonlar görünür kaldı. Balon 5 patlamasında mor damlalar yakın sol
  yarıda gizlendi; uzak sağ yarıda maskenin önünde görüldü. Gözlem pencere üzerinden
  yapıldı; screenshot dosyası oluşturulmadı.
- İlk koşu 385'er yayın, 58.16 / 57.77 presented FPS; son görsel koşu 535 / 534
  yayın, 57.81 / 56.87 FPS. Device loss yok. Bu küçük sentetik maskenin işlev
  kabulüdür; 640×640 kamera veya 10 kişi performans kabulü değildir.
- Paketli MP4: doğal bitiş 437 kare / 12 geçerli poz ve kesit; oyun sahibinin erken
  durdurması 129 kare / 8 geçerli poz ve kesit. Kart yayın kaybı 0, hata listesi boş.
  Kapanışta aktif kişi 0, device_lost=false, runtime error boş.
- Capture.read→RPC toplamı doğal koşuda p50 20.42 / p95 34.53 ms; erken koşuda
  20.28 / 34.09 ms. Sensör kuyruğu ve ekranda görünme süresi dahil değildir.
- 35 Python testi PASS; Player include audit 668/668 temiz; iki shader için
  spirv-val PASS; source/runtime/paket shader hashleri eşit.

MP4 yalnız kısa bir bölümde güvenilir ayak pozu sağlıyor. Bu sonuç gerçek oda
kalibrasyonu veya gerçek kamera gecikmesi kabulü değildir. Ham toplu raporlar
ignored `tools/perception_bridge/out/packaged-live-mp4-2026-11-18.json` ve
`depth-drops-visual-2026-11-18.json` dosyalarındadır; kişisel görüntü içermezler.

## Paket

`dist/Player-depth-2026-11-18/`: 377.5 MB, 11 DLL, iki sahne; kişisel MP4 ve
kalibrasyon dosyası içermez. `-NoSceneAudio` kullanıldı, sahne sesleri eklenmedi.
Bu test paketli köprü/runtime'ın açık Release Player'a bağlanmasıdır; paketli
exe başka makinede ayrıca başlatılmadı.

`perception/python/python.exe`: CPython 3.10.6; MediaPipe 0.10.35,
OpenCV-contrib 4.14.0.94, NumPy 2.2.6. Model `perception/models/pose_landmarker.task`,
SHA-256 `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a`.
CPython lisansı ve paketlerin lisans/.dist-info dosyaları korunur.

Player'a Python linklenmez. Dış oyun betiği `BridgeProcess` ile köprüyü başlatır,
hazır olma/ilk geçerli poz olayını bekler. Stdin kapanışı veya EOF normal finally
yolundan görüntüyü boşaltır, pozları temizler, video/modeli kapatır. Zorunlu kill
başarılı temizlik sayılmaz. Native ML kurulumu stdin okuyucusundan önce tamamlanır;
hazır olma takılması bu sıralamayla giderildi. Native stderr tutulmadan atılır.
Taşınabilir Python `_pth` dosyası kullanıcı site-packages/ortam yollarını almaz.

## Kullanım ve sınırlar

Mevcut `player.set_display(video_card=...)` IPC ve `rt.player.set_video_card` aynı
hizmeti kullanır. Köprüde `--video-card --synthetic-depth 0` Vulkan kanalını,
yalnız `--video-card` eski SDL kanalını kullanır. Yeni eşleme
`Local\RayTrophiDepthCard.v1`: 40 B başlık, piksel başına RGBA8 + float metre ofseti.
Ofset gerçek sensörün eksenel Z değeri değildir; gerçek RGB-D adaptörü henüz yok.

Maske kenarı Bayer mekânsal kapsama kullanır; Python referansındaki tam alpha/
additive birleşimiyle piksel eşitliği kabul edilmedi. Viewport capture açıldığında
Vulkan kartı kapatılır; o sırada canlı aynanın sunumda sürmesi açık iyileştirmedir.
Görüntü sahne dosyası/doku varlığı olmaz. Kamera bağlantısı, oda kalibrasyonu ve
gerçek uçtan uca gecikme saha işi olarak kaldı. Personel paneli eklenmedi.

Tekrar testleri: `depth_card_probe.py` ve paketli `replay_owner_probe.py`.
Replay koşucusu `--python`, `--worker`, `--model`, `--video`, `--calibration`,
`--synthetic-depth 0`, `--output` alır; `--offline` IPC'siz kabulü seçer.
Player önceden açık olmalı; hiçbir test uygulama başlatmaz. Test sonunda
video_card/continuous eski false değerine döndü, maske kayboldu, Player sağlıklı kaldı.
