# Yoğun sahnede balon / boya — 2026-11-16

Canlı dış IPC ile dört koşul ölçüldü. **OBB temaslarıyla boya oluşmadı;
mevcut ObjectMeshSDF yoluyla gerçek boya oluştu. Genel üretim kabulü değil.**
Motor C++/shader kodu değişmedi, uygulama veya build başlatılmadı.

Sayısal kanıt: [JSON özeti](PLAYER_HEAVY_BALLOON_PAINT_2026_11_16.json).
Ham raporların yolları ve SHA256 değerleri özettedir; ham dosyalar ignored
`tools/perception_bridge/out/heavy-balloon/` altında tutulur.

## Düzenek

- Kaynak `E:/rt_k/oyuncak.rtp`: 1.627 flat mesh grubu, 3.729.294 üçgen.
  Dört koşul aynı kamera, 1680×945 render, material shading, aynı ışık/kalite
  ayarları ve 417 tohumu ile çalıştı. Geometri binary kopyaları birebir aynı.
- İki materyal koşulu: özgün 165 materyal; her materyal kimliği korunarak
  nötr gri, pürüzlülük 0,65 ve dokusuz default özellikler. Tek ortak materyale
  birleştirilmedi; draw sayısı/batching değişmedi. Gömülü doku dosyaları kopyada
  durur; bu bir doku belleği tasarrufu ölçümü değildir.
- Gerçek sahnedeki `1_Walls`, `1_Ground`, `1_Mixer`, `1_Computer` yüzeyleri,
  her biri 512² atlas ve dört renkteki sistem başına dört collider kullanıldı.
  Seçilen materyaller 1/3/105/123 başka objelerle paylaşılmıyor. Bilgisayarın
  yalnız ilk materyal bölümü boyanabilir hedefe dahil; bütün bölümler değil.
- Mevcut BalloonPaintDemo ve ortak GPU partikül/çarpışma/boya hizmetleri çağrıldı.
  Yeni simülasyon yazılmadı. Her koşulda 1/3/1/3 balon patlatıldı: tekli 1.800,
  üçlü 5.400 gerçek damla; koşul başına toplam 14.400 damla.
- Her koşul ayrı özel proje kopyasında başladı. Capture ölçüm boyunca kapalı,
  continuous açık, pacing 0. Sonunda özgün proje yeniden açıldı; continuous=false,
  pacing=60, capture=false geri alındı. Kaynak .rtp/.bin/.shared/.aux.json
  SHA256 değerleri değişmedi. Kaydedilmemiş kamera hareketi yerine projenin
  kayıtlı kamerasına dönüldü; iki uygulamanın eski canlı durumunu aynen geri
  yükleme iddiası yok. Studio pipe bulunmadı, Studio açılmadı.

## Eş kameradaki sonuç

GPU sütunu patlama pencerelerindeki **raster çizim** ortalamasıdır; bütün
oyun tick'inin GPU süresi veya ekran FPS'i değildir.

| Collider | Materyal | Raster GPU | Son boyalı texel | Kaynak taşması sayacı |
|---|---|---:|---:|---:|
| OBB | Mevcut | ~26,7 ms | 0 | 248 |
| OBB | Default | ~17,8 ms | 0 | 122 |
| SDF | Mevcut | ~26,7 ms | 14.469 | 224 |
| SDF | Default | ~17,8 ms | 9.400 | 190 |

SDF/mevcut: duvar 2.052, zemin 862, mikser 7.629, bilgisayar 3.926 texel.
SDF/default: duvar 1.534, zemin 885, mikser 0, bilgisayar 6.981 texel.
Renkler gerçek partikül görünüm profillerinden geldi. Kapsama farkı bir
materyal emiciliği karşılaştırması değildir: dt, zamanlama, temas örnekleri ve
64 kaynak sınırının etkileri ayrıştırılmadı. Her materyalde her objeye temas
sağlandığı iddia edilmiyor.

Ölçümden SONRA yapılan ayrı kontrol: zemine küçük unfaced dab iki collider
koşulunda da boyadı; tüm sahneyi kapsayan unfaced dab dört atlasın toplam
578.524 geçerli texelini boyadı. Bu kontrol texelleri yukarıdaki patlama
sonuçlarına dahil değildir. Dolayısıyla OBB'deki sıfır kapsama, atlasın hiç
çalışmaması veya yalnız materyal türüyle açıklanamaz.

GPURequired havuzlar hazırlandı, device-loss görülmedi. Patlama boyunca host boya
indirmesi ve indirilen bayt 0; SDF'de yayın device_copy yolunda ilerledi.
`device_event_dispatches` çarpışma sayısı değildir. `sources_dropped` tile başına
taşan adayları toplar; benzersiz kayıp damla sayısı veya kayıp yüzdesi değildir.

Mevcut materyal koşullarında sim tick duvar süresi maksimumu ~27–28 ms,
defaultta ~19 ms. Bu süreler GPU beklemelerini içerebilir. Boyasız, boş partikül
sistemli baseline'da da advance ortalaması mevcutta ~25 ms, defaultta ~16 ms;
dolayısıyla 28 ms'nin tamamı ek boya hesap maliyeti diye yorumlanamaz.
Atlas hazırlığı ~105–107 ms, toplam demo hazırlığı ~139–148 ms; bunlar yükleme
maliyetidir. SDF cook süresi bu deneyde ayrıca ayrıştırılmadı.

Player arka planda continuous çiziyordu: `rendered_fps` ilerlerken
`presented_fps=0` kaydedildi. Bu deney ekran sunumu, görünür patlama animasyonu
veya input→ekran gecikmesi kabulü değildir. Yeni readback düzeltmesiyle submit/
consume ilerledi; kullanıcı hareket akıcılığını önceki derlemede ayrıca onayladı.

## SDF ve kalan işler

Ana duvar mesh'i odanın çevresini tek objede tutar. Tek OBB, içteki boş hacmi
de katı kabul edebilecek bir yaklaşımdır; karmaşık mobilyalarda da kutu yüzeyi
gerçek yüzeyden uzaklaşabilir. Mevcut SDF'ye geçişle aynı sahne/atlaslarda boya
oluşması **statik oda için pişmiş çarpışma geometrisini destekliyor**. Bu deney,
OBB uygulamasındaki olası dönüşüm sorunlarını tek başına tamamen dışlamaz.

Önerilen yön mevcut yolun kullanımıdır: statik oda için yüklemede pişirilen,
paylaşılan SDF; basit/hareketli cisimler için uygun analitik collider veya OBB.
Bütün objelere körlemesine ayrı yüksek çözünürlüklü SDF üretmek bu ölçümle
gerekçelendirilmedi. Bu test dört gerçek collider/yüzeyi kapsar; ağır sahnenin
tamamı çizildi fakat bütün sahne çarpışmaya ve boyaya açılmadı.

Üretimden önce ayrıca:

1. `paint_wet.comp` içindeki 64 kaynak/tile sınırı: üçlü patlamada iki collider
   yolunda da taşma var. SDF bunu çözmüyor. Çekirdek dosyalar Claude A'nın sahipliğinde;
   bu tur değiştirilmedi, koordinasyon panosuna ölçümler yazıldı.
2. 1.627 mesh grubunun 131'inde UV aralığı sabit. Bazı ana duvar bölümleri de
   sıfır UV taşıyor. Default materyal bu eksikliği gidermez; boya UV hazırlığı ve
   paylaşılan/üst üste UV'ler ayrıca ele alınmalı.
3. Demo hâlâ 8×8 m fixture için sabit balon yerleşimi kullanıyor. Bu sahne daha
   küçük: pek çok balon odanın dışında doğuyor. Oyun sahnesine uygun doğum hacmi
   ve tüm gerekli yüzey/collider hazırlığı bu testle tamamlanmış sayılmaz.

## Tekrar

Özgün oyuncak.rtp Player'da açık, aktif boya/demo yokken dış terminal:

```powershell
python scripts/test/rt_heavy_balloon_paint_ipc.py --collider-mode ObjectOBB
python scripts/test/rt_heavy_balloon_paint_ipc.py --collider-mode ObjectMeshSDF --camera-from-report tools/perception_bridge/out/heavy-balloon/ObjectOBB-report.json
```

Betik kaynak projeyi kaydetmez; ignored kopyalara particle sistemleri ve
materyal varyantı yazar. Her koşuldan sonra geçici boya atılır; sonunda özgün
proje açılır. `--help`, Python AST ve odaklı whitespace kontrolleri geçti.
Kaynak hash, eş kamera, doğan damla sayıları, atlas kontrolleri, GPU sağlık ve
geri dönüş kanıtları ayrıca rapordan doğrulandı. Genel performans/oyun kabulü
olarak PASS verilmedi.
