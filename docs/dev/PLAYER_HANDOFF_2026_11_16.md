# Player / balon oyunu — güncel devir

Hedef: çocuk hareketleriyle balon patlatılan, fiziksel boya damlalarıyla oda
yüzeylerinin boyandığı oyun. Kanonik ürün planı `PLAYER_GAME_PAINT_ROOM.md`;
genel runtime ayrımı `RAYTROPHI_PLAYER_PLAN.md`.

## Devralan sohbetin son durumu

- 2026-11-16 sentetik beden/arms-up → gerçek balon/damla → dört yüzey GPU boya
  3/3 canlı IPC PASS. Her denemede tek kenar/pop, 1800 damla; kayıp/host indirme/
  device loss 0. Takvim kapalı, doğrudan burst/post_event/dab RPC'si yok.
  [Kalıcı rapor](PLAYER_BALLOON_INPUT_ACCEPTANCE_2026_11_16.md).
- Bu, rastgele balon seçen jest kuralıdır; belirli balona uzaysal el teması,
  gerçek kamera, rig/IK, görsel piksel veya controller kabulü değildir.
- **Kullanıcı kamera adımını şimdilik bekletti.** Kamera/model/ML kurulumu yok.
  Kullanıcı donanım bağımsız yönü teyit etti: kaynak/SDK/kalibrasyon dış adaptörde;
  motor yalnız mevcut kanonik poz sözleşmesini tüketir. Bir cihazı zorunlu tutma.
- Probe cleanup: attract, 0 kişi, demo kapalı, atlas reset; test config'i geçici
  configured attract'ta kaldı, display geri uygulandı. Sonraki okumada Player
  yeniden oyuncak.rtp'de, session unconfigured attract/kişi 0/demo prepared=false;
  bu sonraki geçiş probe'a ait değildir. Mevcut projeye yeni müdahale yapılmadı.
- Bu tur yalnız Python kabul aracı/test/rapor yazıldı; C++/shader/build/commit yok.

## Kabul edilenler

- Önceki canlı kabul: bağımsız Player, ortak Vulkan render/sim/ses, materyal,
  partikül, scatter ve volume yolları; balon patlama -> GPU damla -> çok yüzeyli
  boya demosu. Bunlar gerçek kamerayla tamamlanmış oyun anlamına gelmez.
- Poz/jest/kural/oturum çekirdeği sentetik 4/10 kişiyle IPC kabulünden geçti.
- Bugün kullanıcı Studio Fire'ı, iki yönlü None/Exit Time döngüsünü; doğru kayıtlı
  projeyi Player'da açıp iki klip arasındaki otomatik geçişi ve düzgün mesh'i
  görsel doğruladı. Bu **graph/state machine** kabulüdür.
- Sabit render çözünürlüğü kullanıcı manuel PASS: pencere resize/fullscreen
  sunumu ölçekler, F2 veya mevcut IPC render_scale sabit render_base'i ölçekler.
  Otomatik display betiği yeni binary'de çalıştırılmış sayılmaz.

## Performans — ölçüm var, optimizasyon iddiası yok

Yoğun `E:/rt_k/oyuncak.rtp`: 1627 draw, ~3.72M görünür üçgen, 11 ışık/8 gölgeli.
Player uncapped 1680×945: 11.38 FPS, 85.71ms render+upload; GPU59.98ms,
shadow atlas33.10ms, upload~1ms. 1260×709:12.71FPS; 840×473:13.90FPS.
Sabit gölge maliyeti nedeniyle yalnız çözünürlük düşürmek yeterli değil.

Studio aynı kamera/çözünürlük/uygulanan ayarlar: 41 kontrollü redraw frame,
CPU27.20ms, GPU60.49ms, shadowCPU24.32ms. Bu sürdürülebilir Studio FPS değildir.
Player10Hz tempo kontrolü CPU26.91ms/GPU60.54ms/shadowCPU24.38ms verdi.
Uncapped shadowCPU77.73ms'yi Player'ın saf gölge hesabı diye yorumlamayın;
GPU/driver backpressure veya pipeline örtüşmesi adaydır, kesin kök değildir.
Önce eşit tempolu, ısınmış, capture-kapalı sustained A/B; sonra gölge atlası ve
kayıt/gönderim/okuma örtüşmesi incelenmeli. Kaliteyi sessizce düşürmeyin.

Kalıcı raporlar: `PLAYER_HEAVY_SCENE_PROFILE_2026_11_16.md/.json`.
Ham raporlar ignored `tools/perception_bridge/out/` altında. Kamera, display,
capture ayarları geri alındı; ağır sahne kaydedilmedi.

## Kaynak değişiklikleri ve açık sınırlar

- Fire UI, mevcut rtapi/Python/IPC trigger hizmetini çağırıyor; editör asset'ine
  ikinci trigger bırakmıyor. Push yeni runtime klonunun transient trigger'larını
  temizliyor; trigger hizmeti çizimi uyandırıyor.
- Graph/controller palet yayını `Animation/CharacterSkinPalette.h` ile ortak;
  canonical global kemik slotları/inverse/bind offset. Yeni skin/retarget motoru
  veya ağırlık düzenlemesi yok. Controller cache aktarımındaki eski farkın kesin
  kökü ölçülmedi; yeni sürümde direct controller kabulü hâlâ açık.
- Sıfır ağırlıkta başlayamayan ortak controller crossfade predicate düzeltildi.
  Önceki eski Player binary'sinde geçiş takılması görüldü; yeni binary'de ayrıca
  doğrulanmadan crossfade PASS denmeyecek.
- Harici Space IPC köprüsü test aracıdır, ürünün yerel klavye action sistemi değil.
  İlk kullanımında kullanıcı controller mesh bozulması bildirdi; süreç durduruldu,
  0 Space olayı vardı. Player düzgün graph oynatmasına geri alındı.
- Kullanıcı ağırlık/retarget verisinin Studio'da doğru olduğunu doğruluyor.
  Graph görünür PASS'ı controller veya IK/domain PASS'ına genişletmeyin.

## Balon oyunu için devam sırası

1. Sentetik arms-up → gerçek pop → damla/boya zinciri 2026-11-16 3/3 PASS;
   uzaysal el/balon hedeflemesi ayrı açık. Otomatik demo ile karıştırmadan ölçme
   yöntemi balloon_input_probe.py ve kalıcı kabul raporunda.
2. **Kullanıcı isteğiyle beklemede:** aynı zincirin tek kişi gerçek kamera kabulü;
   donanım bağımsız adaptör sözleşmesini koru. Kalibrasyon, güven/kayıp,
   gecikme ve kişi kimliğini ölç. Algı köprüsü `tools/perception_bridge/` hazır
   kaynak içerir; donanım üzerinde doğrulanmış varsayma.
3. Çoklu kişi, oturum start/finale/reset, ses/görsel final; ardından kiosk ve
   uzun koşu. Kamera/oda/projektör kalibrasyonu ve maskeleme/video kartı açık.
4. Avatar gerekiyorsa imported rig/IK ownership ve domain/proxy canlı kabulü;
   karakter delete/undo/reopen ve kapanış süresi ayrı borçlar olarak açık.

Sahne ve uygulamaların son halini yeniden salt okunur sorgula; kullanıcı bu
oturumda birkaç proje değiştirdi. Eski wiezzorek-balloon tek klipli dosya ile
sonradan kaydedilmiş iki state'li projeyi karıştırma. Yeni devir için otomatik
uygulama başlatma, C++ build veya commit yapılmadı. Kullanıcı son display
davranışını test edip geçtiğini bildirdi; aynı paket için tekrar build isteme.

## Çalışma kuralları

Önce `AJAN_KOORDINASYON.md` oku, dosya sahipliğini kaydet; çalışan başka ajanın
alanını biçimlendirme. C++ build kullanıcıda; shader değişirse yalnız etkilenen
varyantları derleyip iki shader klasörüne dağıt. Kullanıcı o turda istemedikçe
Studio/Player başlatma. Canlı IPC ayrı süreçten `scripts/test/rt_ipc.py` ile;
Windows error5'te gereken sandbox escalation. Flat TriangleMesh/DNA esas;
Studio'nun ortak hizmetlerini kullan. Per-face otorite veya ikinci runtime yazma.
Yeni authoring işlemi ortak servis + script + IPC ile teslim edilir. 2000 satırı
aşan dosyalara yalnız minimal integration wiring; yeni mantık odaklı modülde.
