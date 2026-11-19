# Player — algı köprüsü ve sahte çocuk koşucusu

> **Durum: KAYNAK HAZIR / UYGULAMASIZ PASS — 2026-11-15.** 23 Python testi,
> N=1..10 deterministik senaryo doğrulaması, N=1 ve N=10 CLI dry-run PASS.
> Güncelleme: sahte 4/10 çocuk Player IPC tam oturum kabulü **PASS**;
> gerçek webcam, MediaPipe kurulumu ve kamera gecikmesi **ÇALIŞTIRILMADI**.
> C++/shader değişikliği, derleme, uygulama başlatma ve commit yok.

Kod: `tools/perception_bridge/`. Kaynak sözleşme:
[iş bölümü ve olay adları](AJAN_KOORDINASYON.md),
[Boyama Odası §5.1 ve §6](PLAYER_GAME_PAINT_ROOM.md).

## Kurulum ve uygulamasız doğrulama

**2026-11-18 güncellemesi:** Kullanıcı build sonrası sentetik Vulkan derinliği,
balon/damla örtmesi ve paketli MP4 başlatma/doğal bitiş/erken kapatma canlı PASS.
35 Python testi geçiyor. Üstteki 2026-11-15 özeti tarihseldir.
[Güncel kabul ve sınırlar](PLAYER_DEPTH_PACKAGE_ACCEPTANCE_2026_11_18.md).
Gerçek kamera bağlantısı, oda kalibrasyonu ve uçtan uca gecikme henüz ölçülmedi.

Depo kökünden, Python 3.10+ ile:

```powershell
python -m unittest discover -s tools/perception_bridge/tests -v
python tools/perception_bridge/fake_children.py --dry-run --children 1 --seconds 24 --output tools/perception_bridge/out/verification-n1
python tools/perception_bridge/fake_children.py --dry-run --children 10 --seconds 36 --output tools/perception_bridge/out/verification-n10
```

Bunlar yalnız standart kütüphaneyi kullanır; kamera ve uygulama gerekmez. Çıktılar:
`messages.jsonl` (gerçek IPC zarfı, artan istek ID'si) ve `acceptance.json`.
`out/`, `models/`, `.venv/`, `__pycache__/` yerel `.gitignore` içindedir.
Dry-run sanal zamanda hızlı çalışır; poz zamanları tam 1/30 saniye aralıklıdır.
Okuma/temizleme istekleri de dosyadadır; motor yanıtı üretilmez.
`SCHEMA_PASS_ENGINE_NOT_RUN` ve `engine_acceptance: NOT_RUN` canlı PASS anlamına gelmez.

Webcam için ayrı sanal ortam:

```powershell
python -m venv tools/perception_bridge/.venv
tools/perception_bridge/.venv/Scripts/python -m pip install -r tools/perception_bridge/requirements.txt
```

MediaPipe Tasks **Pose Landmarker** `.task` modelini resmi
[Python kılavuzundan](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python)
edinip `tools/perception_bridge/models/pose_landmarker.task` yoluna koyun.
Kod model indirmez. Klasik tek kişilik `solutions.pose` kullanılmaz; Tasks `num_poses=1..10`,
VIDEO modu ve artan milisaniye damgaları kullanılır.
[Resmi seçenek sözleşmesi](https://developers.google.com/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmarkerOptions).
Bağımlılık aralıkları requirements dosyasındadır; bu teslimde ML paketleri kurulup
donanım üzerinde denenmediği için sabitlenmiş, doğrulanmış bir kurulum kilidi değildir.

## Kalibrasyon

1. Kamerayı sabitleyin; görüntüyü aynalamayın. Sabit çözünürlük ve aynı lens/zoom kullanın.
   Oda koordinatında Y yukarı, X/Z zeminde, birim metre. Dört zemindeki işareti ölçün;
   işaretler aynı doğru üzerinde olmamalı, oyun alanını çevrelemeli.
2. Aşağıdaki komutta sırasıyla `(0,0), (4,0), (4,3), (0,3)` metre işaretlerini seçin.
   Önizleme yalnız RAM/ekrandadır. Dört tıklamadan sonra Enter kaydeder, R temizler,
   Esc iptal eder. Kaydedilen dosyada yalnız çözünürlük, piksel/metre eşleri ve dönüş vardır.

```powershell
tools/perception_bridge/.venv/Scripts/python tools/perception_bridge/calibrate.py --camera 0 --size 1280 720 --room-xz 0 0 4 0 4 3 0 3 --yaw 0 --pitch 0 --roll 0 --output tools/perception_bridge/out/floor.json
```

3. Köprü, dört eşten homografi çözer; iki görünür ayak bileğinin ortalama pikselini
   zemindeki X/Z'ye taşır. Tek ayak güvenilir ise onu kullanır. Ayaklar görünmüyorsa kişi
   o karede gönderilmez; ufuk/sayısal hatalı noktalar reddedilir. Yanlış çözünürlükte çalışmaz.
4. MediaPipe'ın metre cinsinden yerel iskeleti, ayak ankrajına göre taşınır; Y ters çevrilir.
   Dönüş sırası kamera Y-yukarı eksenlerinde X/pitch, Y/yaw, Z/roll'dür (sağ el kuralı).
   Sıfır açılar kamera eksenlerinin oda eksenleriyle eşleştiğini varsayar. Açılar dört
   zemin noktasından otomatik çözülmez; kurulumda ölçülmelidir. Baş, burun landmark'ıdır;
   boyun omuzların, pelvis kalçaların ortasıdır.
5. Ayrı bir beşinci işareti ölçüp konum hatasını kontrol edin; tüm zeminde ve farklı
   boylarda çocuklarla tekrar edin. Kamera taşınır, zoom/çözünürlük değişirse yeniden kalibre edin.

**Derinliksiz sınır:** homografi yalnız zeminde doğrudur. Havada iki ayağın görüntüsü
yanlış X/Z üretebilir; ayak ankrajı mutlak sıçrama yüksekliğini güvenilir ölçmez.
El uzanma derinliği MediaPipe tahminidir. Lens distorsiyonu ayrıca giderilmez; geniş açıda
kenar hataları büyür. Bu yöntem güvenilir RGB-D/hacimsel takip iddiası taşımaz.

## Kamera köprüsü

Kullanıcı güncel Player'ı ve uygun projeyi açtıktan sonra **ayrı terminalde**:

```powershell
tools/perception_bridge/.venv/Scripts/python tools/perception_bridge/bridge.py --camera 0 --people 10 --model tools/perception_bridge/models/pose_landmarker.task --calibration tools/perception_bridge/out/floor.json --seconds 60 --metrics tools/perception_bridge/out/camera-metrics.json
```

`--camera` OpenCV'nin cihaz indeksidir; kamera SDK'sı motora bağlanmaz. `--seconds 0`
Ctrl+C'ye kadar sürer. Kaynak kapanınca veya hata olunca pozları temizlemeyi dener,
kamera/model kaynaklarını kapatır. Kamera kaybında otomatik yeniden bağlanma bu dilimde yok.

Hedef gönderim 30 Hz'dir. İnferans/IPC 33,3 ms'yi aşarsa gerçek hız düşer; sahte başarı
yazılmaz, `over_budget_frames` artar. Kamera buffer boyutu 1 talebi backend'e bağlıdır.
Donanımın 10 kişide 30 Hz vereceği doğrulanmadı. Ayrı terminal pipe istemcisi mevcut
`scripts/test/rt_ipc.py` yolunu kullanır; varsayılan `\\.\pipe\RayTrophiPlayer`.
Sandbox Windows hata 5 verirse gerçek IPC probu gerekli sandbox yükseltmesiyle tekrarlanmalı;
başka taşıma uydurulmamalı. Gömülü script workspace içinde çalıştırmayın.

Konuma dayalı global eşleme, sabit hız tahmini ve metre cinsinden kapı kullanılır.
En çok 0,5 saniyelik kayıp aynı ID ile dönebilir; daha uzun kayıpta yeni ID atanır ve eskisi
yeniden kullanılmaz. ID yalnız süreç/oturum içindir; biyometrik kimlik değildir. Tam örtüşme,
ani yön değişimi ve birbirine çok yakın çocuklarda ID değişimi mümkündür. Köprü kayıp
iskeletleri uydurup göndermez; motorun 0,5 saniyelik düşürme davranışını kullanır.

`bridge.py` görüntü veya landmark kaydı yazmaz. Metrics yalnız toplu zamanlar,
sayaçlar ve hata sınıfları içerir. Kalibrasyon önizlemesi bellekte kalır.
İsteğe bağlı ayna prototipi, RGBA kareleri Windows paylaşımlı belleğinde taşır.

## Kamera olmadan video replay

Sabit açıdan, sabit kare hızlı kısa bir MP4 kullanın. İlk karede dört zemin
referansı, jest boyunca eller ve ayaklar görünür olsun. Kollar aşağıdayken
başlayıp iki kolu kaldırın. Aşağıdaki oda koordinatları örnektir. Tıklama sırasını
sahnenizin X/Z koordinatlarına göre ayarlayın. Video modunda çözünürlük dosyadan okunur.

```powershell
$bridgePython = 'tools/perception_bridge/.venv/Scripts/python'
$clip = 'tools/perception_bridge/out/gesture.mp4'
$floor = 'tools/perception_bridge/out/video-floor.json'
& $bridgePython tools/perception_bridge/calibrate.py --video $clip --room-xz 0 0 4 0 4 3 0 3 --output $floor
& $bridgePython tools/perception_bridge/bridge.py --video $clip --calibration $floor --model tools/perception_bridge/models/pose_landmarker.task --people 1 --offline --metrics tools/perception_bridge/out/video-offline.json
```

`frames_with_pose` ve `accepted_poses` geçerli oda pozuna dönüşen kareleri ve pozları
sayar. Player'da jest kuralı etkin test sahnesi açıkken, aynı komutu dış terminalde
`--offline` olmadan çalıştırın. Video dosyanın kare hızında gönderilir ve bittiğinde
pozlar temizlenir. Kabul için kollar aşağıdayken pop olmamasını, kaldırınca bir pop
oluşmasını ve GPU boyanın yüzeye ulaşmasını kontrol edin.

### Player iskelet çizimi

Güncel Player derlemesinde `player.set_display` isteğine `body_overlay: true` verin.
Aşağıdaki komutlar yerel IPC'yi kullanan harici Python örnekleridir.
İlk satırla çizimi açın, üstteki video komutunu `--offline` olmadan çalıştırın
ve Player penceresini önde tutun. İskeletin kollarla birlikte hareket ettiğini
ve görüntü bitince kaybolduğunu gözleyin. Son satır çizimi kapatır.

```powershell
python -c "from tools.perception_bridge.transport import connect; c=connect(); print(c.call('player.set_display', body_overlay=True)); c.close()"
# Run bridge.py --video ... while Player is in the foreground.
python -c "from tools.perception_bridge.transport import connect; c=connect(); c.call('player.set_display', body_overlay=False); c.close()"
```

`player.display` geçerli `body_overlay` durumunu bildirir. Yerleşik Python
bağlaması `rt.player.set_body_overlay(enabled)` aynı hizmeti çağırır ve bir
Player görüntü hostu ister. Paketli Player'ın betik istemcisi dış Python'dur.
Çizim varsayılan olarak kapalıdır ve Player pencere sunumunda görünür.
Oda uzayı pozları sahne kamerasından çizilir. Yaklaşık video kalibrasyonu
ekrandaki yerleşimi de yaklaşık yapar. Kamera arkasındaki ve güven eşiğinin
altındaki eklemler çizilmez.

### Maskeli video kartı

Windows Player'da `player.set_display` ile `video_card: true` verin. İskeleti
aynı anda görmek için `body_overlay: true` ekleyebilirsiniz. Scripting bağlaması
`rt.player.set_video_card(enabled)` aynı görüntü hizmetini kullanır. Paketli
Player için dış Python istemcisini kullanın.

```powershell
python -c "from tools.perception_bridge.transport import connect; c=connect(); c.call('player.set_display', video_card=True, body_overlay=True); c.close()"
& $bridgePython tools/perception_bridge/bridge.py --video $clip --calibration $floor --model tools/perception_bridge/models/pose_landmarker.task --people 1 --video-card --metrics tools/perception_bridge/out/video-card-live.json
python -c "from tools.perception_bridge.transport import connect; c=connect(); c.call('player.set_display', video_card=False, body_overlay=False); c.close()"
```

Player penceresini önde tutun. `--offline` eklenirse yalnız kesit üretimi ölçülür
ve Player bağlantısı veya paylaşımlı bellek açılmaz. `card_frames` geçerli RGBA
kesit üretilen kare sayısıdır. Kart için güvenilir bir ayak bileği ve baş gerekir.
Poz ve kesit aynı kaynak karesinden çıkar. Bir sonraki sunum turunda en güncel
geçerli kare kullanılır. Bitimde kart ve iskelet kaybolur.

İlk sürüm tek kişiyi, ayak hizasına yerleştirilen kamera yönelimli bir kart
olarak Player pencere kompozisyonuna çizer. Poz kalibrasyonu kartın yerleşimini de
belirler. Sahne nesneleri ve partiküller kartı derinlikte örtemez. Yalnız Player
penceresine eklenen kart, sahne render çıktısı ve viewport yakalamasına girmez.

Windows oturumu başına bir yayıncı, `Local\RayTrophiVideoCard.v1` eşlemesine
en güncel kareyi yazar. 40 bayt little-endian başlık ve en fazla 640×640
straight-alpha RGBA8 ayrılır. Kare verisi IPC JSON'a girmez. Okuyucu kilidi
beklemeden dener ve 300 ms'yi geçen kareyi gizler. Normal kapanışta yayıncı
eşlemeyi boş kareye çevirip piksel alanını sıfırlar.

### RGB-D derinlik referansı

`rgbd_reference.py` hizalanmış RGBA8 ve metre cinsinden derinlik alır. Sensör
intrinsics ile noktaları açıp kalibre edilmiş rijit dönüşümle Player kamera
uzayına taşır. Sanal kamera intrinsics ile tekrar piksellere yerleştirir.
Aynı hedef piksele gelen örnekler arasından en yakını seçer. Geçersiz derinlik
ve maske dışı pikseller boş kalır. Giriş derinliği kamera Z ekseni boyunca
ölçülür. Bir sensör adaptörü radyal mesafe sağlarsa önce eksenel Z'ye çevirmelidir.

```powershell
tools/perception_bridge/.venv/Scripts/python -m unittest discover -s tools/perception_bridge/tests -p test_rgbd_reference.py
```

Sentetik testler en yakın derinliği, sahne nesnesi örtmesini, öndeki ve arkadaki
partikülü ve yarı saydam maske kenarında alpha/additive birleşimini sınar.
Bu referans bellek içi Python hesabıdır. Canlı Player gerçek sensör RGB-D verisi
henüz tüketmez; yeni sentetik kanal piksel başına kamera-ileri metre ofseti alır.

Player entegrasyonunda RGB-D kare en güncel örnek olarak tek bir slotta
aktarılmalı. 640×480 RGBA8 + 16 bit derinlik 30 Hz'de yaklaşık 55 MB/s
ham piksel verisidir. Kamera verisini GPU'ya bir kere yükleyip yeniden
izdüşüm ve örtme testlerini GPU'da tutun. Sahne derinliğini CPU'ya indirmeyin.
Düz kart için tek sahne derinliği kullanılabilir. Piksel başına kipte
Vulkan raster derinliğini ve karakter derinliğini aynı uzaya getirin.

Opak sahne karakteri ve partikülleri örter. Yarı saydam kenarlarda arkadaki
partikül karakterden önce, öndeki partikül karakterden sonra birleştirilir.
Tam opak karakter piksellerinde arkadaki etki elenir. Kompozisyon Player
penceresine ayrılmalı ki viewport yakalaması ve render çıktısı temiz kalsın.
Bu katmanın performansı GPU kompozisyon süresi, kare yaşı, atlanan kaynak
kareler ve Player render/present p95 ile ölçülmeli. İlk hedef 30 Hz kaynak,
Player sunum hızı ayrı ölçülür.

## Sahte çocuklar ve canlı oturum planı

`--seed 417` varsayılanı tekrarlanabilir yerleşim verir. Her 24 saniyede her çocuk:
yürür, uzanır, sıçrar, ayağını kaldırıp basar, kollarını kaldırır, X yönünde alandan
çıkar/girer, kısa örtüşmede kaybolup aynı ID ile döner, uzun örtüşmede yeni ID alır.
Her bölüm 3 saniyedir; ilk tam çevrim için `--seconds >=24` gerekir. Bölge örneği yaklaşık
X=±3 m oyun alanını varsayar; sahnenin bölge/balon konumları buna uyarlanmalıdır.
Koşucu gesture olaylarını doğrudan post etmez; C'nin pozdan bunları üretmesi gerekir.

Şema: `{t, persons:[{id, confidence, joints:{ad:[x,y,z,güven]}}]}`.
15 eklem: head/neck/pelvis ve iki tarafta shoulder/elbow/wrist/hip/knee/ankle.
Bu köprü tam eklem kümesi gönderir; doğrulayıcı sonlu sayıları, [0,1] güveni, benzersiz
int32 ID'yi, en çok 10 kişiyi ve artan zamanı kontrol eder. Boş persons geçerlidir.
Adlar `gesture.stomp`, `gesture.arms_up`, `gesture.reach`, `contact`, `zone.enter/exit`,
`balloon.pop`, `coverage.threshold`, `timer`, `session.state` sözleşmesiyle uyumludur.

**2026-11-15 devir güncellemesi:** Poz ve kural/oturum çekirdeği artık kaynakta var;
[motor sözleşmesi ve build kontrolü](PLAYER_BODY_GAME_RUNTIME.md).
`tools/perception_bridge/session_plan.json` gerçek IPC çağrılarıyla hazırdır.
Kullanıcı build'i sonrası 36 saniyelik 4/10 çocuk kabulü PASS; tekrar koşusunda kullanılabilir.
Canlı koşucu `--session-plan` ister; farklı kurulumlar için plan biçimi şöyledir
(aşağıdaki yer tutucular çalışır API değildir):

```json
{
  "setup": [{"method": "<gerçek yapılandırma/sıfırlama metodu>", "params": {}}],
  "start": [{"method": "<gerçek oturum başlatma metodu>", "params": {}}],
  "cleanup": [{"method": "<gerçek durdurma/sıfırlama metodu>", "params": {}}],
  "state_path": "<game.session yanıtında durum alanı>",
  "display_fps_path": "frame.presented_fps",
  "expected_states": ["<bekleme>", "<oyun>", "<final>", "<reset>", "<bekleme>"]
}
```

Kurulum planı hazır sahne/atlas, jest kuralları, boya eylemleri ve zamanları sağlamalı;
oturum + final + reset toplamı `--seconds` içine sığmalı. Her durum 0,2 s sorgu aralığında
görülebilmeli. Koşucu mevcut pozları başlangıçta temizler; setup → ilk ölçüm → start →
30 Hz pozlar + 5 Hz ölçüm → boş poz → son ölçüm → poz temizleme + cleanup sırasını yürütür.
Setup/start/cleanup açıkça verilen IPC isteklerini gönderir; örnek yer tutucuları kullanmayın.

```powershell
python tools/perception_bridge/fake_children.py --children 4 --seconds 36 --session-plan tools/perception_bridge/session_plan.json --output tools/perception_bridge/out/live
```

Kabul: `game.session` gözlenen durum geçişleri beklenen sırayla tam eşleşmeli,
`paint.wet_coverage.valid_texels > 0` olmalı ve kapsama ilk örneğe göre artmalı,
`player.display.frame.presented_fps` en az bir örnekte pozitif olmalı.
Ekran FPS'i gerçek görsel içerik doğrulaması değildir. RPC/alan/kapanış hatası veya
başarısız koşul FAIL + çıkış kodu 1 verir. Normal bitiş/dry-run 0 verir.
Yanıtlar ve denetimler JSON raporundadır. `--dry-run --session-plan ...` plan isteklerini
de kaydeder, fakat onların motor şemasını/etkisini doğrulayamaz.

## Ölçülen gecikme ve kabul sınırı

2026-11-15, bu Windows çalışma ortamında Python 3.10.6; CLI dry-run, hızlandırılmış
sanal zaman. Değerler poz üretimi + iki Python şema kontrolü içindir; dosyaya yazım,
MediaPipe, kamera, IPC, motor ve ekran **dahil değildir**:

| Senaryo | Poz sayısı | p50 | p95 | en yüksek |
|---|---:|---:|---:|---:|
| N=1, 24 s | 720 | 0,0313 ms | 0,0344 ms | 0,0548 ms |
| N=10, 36 s | 1080 | 0,2876 ms | 0,3705 ms | 0,6193 ms |

Yerel kanıt: `tools/perception_bridge/out/verification-n1/acceptance.json` ve
`out/verification-n10/acceptance.json` (ignored). Testler tüm N=1..10 için 720'şer
deterministik kareyi ve taklit IPC ile tam durum sırası/kapanış/hata raporunu doğruladı.

Gerçek kamera→IPC gecikmesi **ölçülmedi**. `bridge.py --metrics` capture/inference/
mapping_tracking/rpc/total için son 10.000 karede p50/p95/max verir; total,
`capture.read` başlangıcından IPC yanıtına kadardır, sensör kuyruğu ve render hariçtir.
Sahte koşucu canlı modda RPC süresini, takvim gecikmesini ve gerçekleşen hızı ayrıca yazar.

Tamamlanan kabul: kullanıcı build'i → hazır plan → ayrı terminalde 4 ve 10 çocuk
canlı koşu PASS. Kanıt: [motor kabulü](PLAYER_BODY_GAME_RUNTIME.md).
Sonraki kamera kabulü: gerçek kamerada kalibrasyon hatası, örtüşme ve gecikme ölçümü.
Bu teslim kendi başına C++ derlemesi gerektirmez; yeni script/IPC motor yüzeyi eklemez,
mevcut ortak sözleşmeyi harici istemci olarak kullanır.
