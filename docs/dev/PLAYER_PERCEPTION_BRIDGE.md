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

Görüntü kodlama, görüntü dosyası, kayıt, görüntü logu ve video IPC kanalı yoktur.
`bridge.py` landmark/poz günlüklemesi de yapmaz; metrics yalnız toplu zamanlar, sayaçlar
ve hata sınıfları içerir. Kalibrasyon aracı RAM'de canlı önizleme gösterir. İki araç da
kameranın görüntüsünü diske yazmaz. Paylaşımlı bellek/ayna görüntüsü sonraki dalgadır.

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
