# Sentetik beden → balon → GPU boya kabulü

**2026-11-16: Player canlı dış IPC PASS, 3/3 deneme.** Ürün C++/shader kodu
veya IPC şeması değiştirilmedi; uygulama başlatılmadı, proje kaydedilmedi.
Kullanıcı açık Player'ı bu kabul için ayırdı. Başlangıçtan hemen önce Player boş
projeydi; balon fixture'ı ortak project.open ile yüklendi ve demo takvimi durduruldu.

Kalıcı sayısal kanıt: [JSON özeti](PLAYER_BALLOON_INPUT_ACCEPTANCE_2026_11_16.json).
Ham yanıtlar/istek izi ignored `tools/perception_bridge/out/balloon-input-live-2026-11-16.json`
altında; özette SHA256 var. Öncesi/sonrası ayrı ignored JSON snapshot'larıdır.

## Sınanan mevcut çekirdek zinciri

`input.set_body_poses` → mevcut `BodyPoseRuntime` arms-up yükselen kenarı → tek
Game rule → mevcut `balloon.burst` eylemi → `BalloonPaintDemo` gerçek pop/1800
fiziksel damla → mevcut GPU collision event yolu → `WetPaintGpu` → dört yüzey atlası.
Yeni balon, fizik, boya veya input motoru yazılmadı.

`balloon_input_probe.py` doğrudan `paint.wet_demo burst/start`, `game.post_event`
veya `paint.wet_dab` göndermez. Islak boya yalnız kontrollü trial resetlerinde
sıfırlanır. Başta 5.2 saniye nötr poz, sonraki denemelerde yeniden nötr poz;
her yükselen kenardan sonra 4.5 saniye aynı yüksek poz gönderilir. Otomatik demo
kapalı kalmalı, nötr/held-high dönemleri fazladan pop üretmemelidir.

| Deneme | Jest / pop | Damla doğumu | Boyanan texel | Boyanan yüzey | GPU stamp dispatch artışı | GPU yayın kopyası artışı |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1 / 1 | 1800 | 154008 | 4 | 1028 | 356 |
| 2 | 1 / 1 | 1800 | 208696 | 4 | 1012 | 402 |
| 3 | 1 / 1 | 1800 | 221366 | 4 | 836 | 405 |

Her denemede rule fires/actions_ok/pop olay sayısı tam bir arttı. Action hatası,
oyun olayı kaybı, boya kaynağı düşmesi ve host boya indirmesi/baytı artışı 0.
Damlalar bitince toplam partikül 12, bütün balonlar yeniden şişmiş; device-loss yok.
GPU stamp dispatch sayısı çarpışma adedi değildir. `stamped_texels` son-tick
ölçüsüdür ve damlalar bittikten sonra 0 olması normaldir; toplam sayaç diye
kullanılmaz. Atlas kapsaması + kümülatif device event/copy sayaçları birlikte aranır.

675 RPC: p50 15.01 ms, p95 31.84 ms, maks 57.12 ms. Bu yalnız IPC gidiş-dönüşüdür;
kamera gecikmesi, sensör→ekran gecikmesi veya genel performans kapısının PASS'ı değildir.

## Kabul sınırları

Bu kabul **elleri kaldırma jestiyle rastgele mevcut bir balonun seçilmesidir**.
Elin belirli balona uzaysal temasını, gerçek kişi/kamera algısını, görsel piksel
kanıtını, rig/IK'yi veya anim.play controller/crossfade'i doğrulamaz. Bunları
ayrı açık maddeler olarak koru; kayıtlı graph görsel PASS'ını da genişletme.

Probe bitişinde pozlar temizlendi, demo durduruldu, atlas resetlendi; oturum attract.
Başta config olmadığı için izole test config'i configured=true olarak attract'ta
kaldı; eski sayaç/çizim/oturum geri yüklendi iddiası yok. Test öncesi display
continuous=false / pacing_hz=60 geri uygulandı. Sonraki salt okunur kontrolde
Player tekrar `E:/rt_k/oyuncak.rtp` projesindeydi; kişi yok, session unconfigured
attract, demo prepared=false, device-loss yok. Test sonrası bu proje değişikliği
probe tarafından yapılmadı; mevcut proje yeniden değiştirilmedi.

## Tekrar ve uygulamasız kontroller

Balon fixture'ı hazır/açık ve başka pose kaynağı yokken dış terminal:

```powershell
python tools/perception_bridge/balloon_input_probe.py --run --reset-paint
```

Betik proje yüklemez; yanlış projede veya aktif kişi/oturum varsa mutasyon
başlamadan NOT_READY döndürür. Atlastaki sanat/test boyası sıfırlanır; yalnız
ayrılmış test fixture'ında kullanılır. Yeni build gerekmez. `--plan` bağlantısız
planı gösterir. Sandbox error 5 için aynı dış prob escalation ile çalıştırılır.

6 çevrimdışı negatif/preflight kontrolü PASS: otomatik takvim, yinelenen jest/pop, sadece
başarı sayacı, eksik gerçek damla/GPU boya, host fallback/kayıp ve bitmeyen damla
koşulları kabul edilmez. Yanlış projede tek project.path okumasından sonra durduğu,
hiç mutasyon göndermediği de kontrol edildi. Bu testler motor canlı kabulünün yerine geçmez.
Mevcut algı köprüsü 23 birim testi ve body/game sözleşme denetimi de PASS.

## Kamera ve donanım yönü — kullanıcı kararı

Kullanıcı gerçek kamera denemesini şimdilik bekletti. Kamera açılmadı, görüntü
alınmadı, MediaPipe/model kurulmadı. Envanterde yalnız e2eSoft iVCam görüldü;
bu bir ürün donanım tercihi veya çalışan kamera kabulü değildir.

Ürün **donanım bağımsızdır**; bu, ana plan §5.1'deki mevcut karardır. Motor sadece
zaman/kişi ID/güven/oda uzayı eklem sözleşmesini tüketir. Webcam, telefon/IP kamera,
RGB-D sensör, dosya/replay ve sentetik kaynaklar motor dışında değiştirilebilir
adaptörler olmalıdır. Algılama SDK'sı opsiyonel adaptör bağımlılığıdır; motorun
veya sentetik testlerin zorunlu bağımlılığı değildir. Kalibrasyon/koordinat dönüşü
kaynağa ait veridir; marka/cihaz koşulları motorun içine taşınmaz. Çoklu kamerada
füzyon/ID eşleme dış köprüde birleşip mevcut tek paket akışına gelir. Bu tur mevcut
IPC şeması değiştirilmedi; tüm kaynakların donanım kabulü yapılmış sayılmaz.
