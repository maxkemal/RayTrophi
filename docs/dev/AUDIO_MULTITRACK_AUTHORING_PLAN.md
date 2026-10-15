# Çok katmanlı ses kurgu planı

Durum: M0–M3 ana uygulama kaynakta; M4 kayıt/yükleme ve export bağlantıları mevcut.
Son M2/M3 revizyonu için derleme ve kısa canlı kabul bekleniyor.
Ana yön: [Sound Designer planı](AUDIO_SOUND_DESIGN_PLAN_2026_10_14.md).

## Amaç ve yerleşim

Simülasyon ve eğitim sahnelerinde ses üretimi, kurgu, ön dinleme ve sahneye bağlama
aynı çalışma alanında yapılabilmeli. Alt editörün büyük merkezi ses tuvali olacak.
Sol ana Audio paneli asset, kaynak, giriş/çıkış, kayıt hedefi ve sahne bağlantılarını
taşıyacak. Alt editördeki Tools/Properties panelleri kapatılabilir ve boyutlandırılabilir
kalacak; ayrıntılar seçilen track/klip/efekt için açılacak. Üstte kompakt transport ve
Fit bulunacak; yeni kalıcı slider/buton sıraları eklenmeyecek.

Referanslar: RX'in bağlamsal araç yaklaşımı, Photosounder'ın geniş ses tuvali ve
Vital'ın doğrudan grafik düzenlemesi. Bunların etkileşim ilkeleri mevcut RayTrophi
panel mimarisine uyarlanacak; ürün arayüzü veya DSP kodu kopyalanmayacak.

## Mevcut temel ve gerçek eksikler

Mevcut clip_create/update/remove/duplicate/trim/split/resize işlemleri, waveform,
fade/zarf verileri, mixer bus'ları, capture ve offline export yeniden kullanılacak.
Bugünkü timeline satırları mixer bus'larıdır; bağımsız kullanıcı track'leri değildir.
Designer üretim katmanları da kurgu track'i değildir. Bu üç kavram birbirine
karıştırılmayacak ve yeni bir UI-only klip modeli oluşturulmayacak.

| Aşama | Yapılacak iş | Tamamlanma ölçütü |
|---|---|---|
| M0: ortak kurgu tuvali | Mevcut klipleri bus satırlarında göster; taşıma, kırpma, split, fade ve envelope etkileşimlerini ortak tuval gezinmesine bağla | UI hareketi mevcut canonical clip işlemini çağırır; eski timeline sonucu korunur |
| M1: gerçek track modeli | Kalıcı track ID/ad/sıra/bus, mute/solo/gain/pan; kliplerde track ilişkisi; track CRUD/sıralama | UI, Python, IPC ve proje yükleme aynı doğrulanmış modeli kullanır |
| M2: seçime dayalı kurgu | Çoklu seçim, toplu taşıma/silme/çoğaltma, hizalama, snap; bağlamsal araçlar | İşlem atomik ve tek geri-alma hareketidir; çakışma/negatif zaman hataları açıkça tanımlanır |
| M3: otomasyon ve dinleme | Gain/pan eğrilerini doğrudan çiz; track/klip audition, mute/solo ve loop aralığı | Ön dinleme ile offline export aynı gain/pan/fade/sıra hesabını kullanır |
| M4: proje ve kabul | Track/klip/otomasyon serileştirme, eski proje geçişi ve eksik asset davranışı | Kaydet/aç sonrası kimlikler, sıralama ve duyulan sonuç korunur; regresyon testleri geçer |

İlk uygulama M0 olacak. Gerçek track modeli, kabul edilmiş mevcut clip sözleşmelerinin
üzerine eklenecek; mevcut bus satırları track tamamlanmış gibi sunulmayacak.

## Tuval etkileşimleri

- Tekerlek zoom, orta tuş pan; Fit ve orta çift tık ile görünümü toparla.
- Klip seç/taşı; kenarlardan kırp veya mevcut loop-extension davranışıyla uzat.
- Fade tutamakları ve zarf noktaları klibin üzerinde bulunur; hassas sayılar Properties'te.
- Sağ tık: böl, çoğalt, sil, taşı/hizala ve audition. Çoklu seçimin kapsamı görünür olur.
- Track başlığı: ad, küçük mute/solo göstergeleri ve meter. Efekt/otomasyon ayrıntıları
  yalnız seçimle açılır; bütün satırlarda sürekli kontroller gösterilmez.
- Audition timeline duraklamasından bağımsızdır; Scene Play sahne saatiyle çalışır.
  Hangi yolun dinlendiği ve Master/preview mute nedeniyle sessizlik açıkça gösterilir.

## Canonical servis, proje ve geçmiş

Track işlemleri için mevcut AudioService içinde uygun sözleşmeler tanımlanacak:
listeleme, oluşturma, güncelleme, sıralama ve silme. Bunlar henüz mevcut API diye
belgelenmeyecek. UI, rt.audio ve audio.* IPC aynı servis işlemlerini çağıracak.

Track silme ve içindeki kliplerin davranışı açık parametreyle belirlenecek; sessizce
asset dosyaları silinmeyecek. Eski projelerde track'siz kliplerin bus/routing davranışı
korunacak. Geçiş deterministik olacak ve kalıcı ID'ler yeniden yüklemede değişmeyecek.

Sürükleme sırasında canlı taslak ve commit/iptal sınırı mevcut authoring gesture
mekanizmasıyla uyumlu olacak. Bir sürükleme tek undo adımı; hatalı toplu işlemler
tamamen reddedilecek. Designer tarif geçmişi ve sahne kurgu geçmişi ayrı anlamlarını
koruyacak. DSP, UI bindings veya IPC'de kopyalanmayacak.

## Kabul ve test

Codex derleme yapmaz veya uygulamayı başlatmaz. Kullanıcı derlemesi sonrasında açık
Studio'ya dış terminalden scripts/test/rt_ipc.py ile probe çalıştırılacak.

- M0: taşı/trim/split/resize/fade sonrası source offset, faz, loop ve zarf korunumu.
- M1: track CRUD/sıralama, mute/solo/routing; eski proje uyumu ve round-trip ID'ler.
- M2: hatalı toplu işlemin hiçbir klibi değiştirmemesi; gesture undo/redo ve iptal.
- M3: audition'da callback meter/ilerleyen cursor; çoklu solo ve mute; export PCM ile
  otomasyon sonucu uyumu. Playing=true tek başına duyulan sesin kanıtı sayılmayacak.
- UI: dar/geniş alan, kapalı yan paneller, seçimde zoom/pan/fit, yüksek DPI ve
  seçim/sürükleme sırasında kararlı hit-test. Üst satır sayısı artırılmayacak.
- Performans: waveform decode/cache bütçesi, çok sayıda klip, uzun asset ve export;
  kabul edilen kapasite ölçülecek, ölçülmemiş sınırsızlık vaat edilmeyecek.

## Sonraki bağımsız aşamalar

Arka plan render/analiz, gerçek spektral bölge işleme/yeniden sentez ve sahne-event
modülasyon grafikleri ana plandaki sonraki aşamalardır. Bu kurgu planı onların
tamamlandığı anlamına gelmez. Spektrogram şimdilik analizdir; ses boyama için gerçek
işleme çekirdeği ve ayrı UI/script/IPC sözleşmeleri gerekir.


## M0 uygulama kaydı

İlk M0 entegrasyonu kaynakta yazıldı; yeni derleme/canlı UI kabulü bekliyor. Edit
alanında Source ve Arrange ayrımı var. Seçim Tools veya üstteki View menüsünden
yapılıyor; View dar panelde de erişilebilir. Arrange mevcut AudioTimelineCanvas'ı
yeniden kullanır; ek navigation/toolbar satırı üretmez ve Designer'a geri çağrı
yapmaz. Klip taşıma, kırpma, loop uzatma, gain/fade tutamakları, A/B/Delete kısayolları
aynı canonical clip işlemlerini ve mevcut gesture geçmişini kullanır. Sağ tık
bölme/çoğaltma/silme menüsü eklendi; bölme konumu menüyü açan tıklamadan alınır.

Tools assetten klip ekler; Properties seçilen klibin mevcut inspector mantığını
aynı panelde kullanır. Wheel zoom ve orta tuş pan Arrange tuvaline bağlıdır; üst Fit
ortak zoom/pan durumunu sıfırlar. Satırlar gerçek bağımsız track'ler değil mevcut
mixer bus'larıdır. Designer Play burada Play input olarak etiketlenir; kliplerin
sahne saatine bağlı dinlenmesi Scene transport ile mevcut Timeline'a geçerek yapılır.
Ayrı transport/DSP iş mantığı eklenmedi.

Yeni canonical workspace alanı: designer_edit_view = source|arrange, varsayılan
source. UI, rt.audio.workspace_update ve IPC aynı doğrulayıcıyı kullanır; geçersiz
değer/başarılı round-trip dış araç probuna eklendi. Bu revizyon derlenmedi; Codex
kaynak delimiter/boyut, Python syntax ve diff whitespace kontrollerini yaptı.

Yeni kullanıcı derlemesi sonrasında: Edit > Arrange seç, mevcut assetten klip ekle,
sürükle/kırp/uzat/fade/gain düzenle, B ile böl, sağ tık çoğalt/sil, Properties'te
değerleri kontrol et ve undo/redo dene. Wheel/pan/Fit ve Source'a dönüşü doğrula.
Mevcut rt_test_audio_clips_ipc.py clip/export regresyonunu ve güncel
rt_test_audio_design_tools_ipc.py workspace doğrulamasını dış terminalde çalıştır.
İleri çoklu seçim, bağımsız track modeli ve doğrudan zarf noktası çizimi tamamlanmış
sayılmıyor; M1/M2/M3 kapsamlarında duruyor.


## M1 track modeli: kaynak teslimi

Bağımsız track çekirdeği ve Arrange UI bağlantısı kaynakta yazıldı. Yeni derleme,
proje round-trip, undo/redo ve canlı ses kabulü henüz yapılmadı.

| İşlem | Parametreler / sonuç |
|---|---|
| track_list | Parametresiz; sıralı track dizisi, Read |
| track_create | name, bus, gain, pan, muted, solo; ID servis tarafından ayrılır |
| track_update | id + aynı alanların patch'i |
| track_reorder | ids: bütün track ID'lerinin tekrarsız tam permütasyonu |
| track_remove | id, clip_policy=reject; klipli track için açık detach veya delete gerekir |

Yazma işlemleri SceneWrite + FilesRead olarak sınıflandırıldı ve scene authoring
geçmişine bağlı. UI, rt.audio ve IPC aynı AudioTracks çekirdeğini kullanır. Toplam
dinamik audio işlemi 72 oldu. Kalıcı `tracks` dizisi ve klip `track_id` alanı save/restore
yolunda korunur. Track olmayan eski projeler boş track dizisiyle aynı bus davranışını
sürdürür; track'siz klipler legacy bus satırlarında görünür. Eski uygulama sürümünün
yeni track alanlarını yükleyebileceği garanti edilmiyor.

En fazla 128 track; ad 1..256 UTF-8 byte, linear gain 0..16, pan -1..1; mute/solo
sadece boolean. Track ID'leri mevcut ortak next_id sayacını ve restore ID kontrolünü
kullanır. Eksik track referansı reddedilir; attached clip routing'de track bus'ı
authoritative olur. Track bus güncellemesi bağlı klip bus'larını atomik günceller.
Klip gain/zarfı track gain ile çarpılır; track pan otomasyon uygulanmış klip pan'ına
eklenip -1..1'e sınırlandırılır. Track solo aktifken solo olmayan ve track'siz klipler
susturulur; birden çok solo birlikte çalar, muted+solo track kendi sesini susturur.
Sonrasında mevcut bus/Master mix uygulanır. Aynı AudioTrackMix işlemi canlı clip
runtime ve export tarafından kullanılır; export mix bilgisi job başlangıcında cache'lenir.

Detach dosyaları ve klipleri korur, track_id'yi temizler, son bus bağlantısını tutar;
track gain/pan/solo etkisini klibe bake etmez. Delete yalnız track ve bağlı klip
authoring kayıtlarını siler; asset veya kaynak dosyası silmez. Hatalı CRUD/reorder
işlemleri candidate scene üzerinde reddedilir; next_id dahil mevcut durum korunur.
`selected_track` mevcut workspace doğrulamasına eklendi; silme/restore eksik seçimi
temizler. Track/clip seçimi ve panel durumu transient workspace'tir.

Tools: Add track, kompakt M/S ve seçim; sağ tık move up/down ve iki açık silme
politikası. Properties: seçilen track adı, bus, gain/pan (Enter ile tek işlem). Klip
Properties'te Track seçimi var; bağlı klibin Bus alanı track tarafından yönetilir.
Seçilen track'e asset ekleme ve klibi satırlar arasında sürükleme canonical
clip_create/update(track_id) yolunu kullanır. Track'lerin aynı satırdaki çakışan
klipleri birlikte mix edilir; üst üste hit-test/görsel ayrıştırma iyileştirmesi M2'dedir.
Per-track canlı meter henüz yok; mevcut bus meter'ları kullanılmaya devam eder.

Static delimiter/boyut, VS kayıt/XML, Python syntax, 72 benzersiz audio işlemi ve
IPC/security mirror denetimleri geçti. Codex derleme yapmadı veya Studio başlatmadı.
Yeni derleme sonrasında dış terminal:

```powershell
python scripts/test/rt_test_audio_tracks_ipc.py --output-dir docs/dev
```

Probe geçici kendi track/klip/assetlerini oluşturur; export PCM'de gain oranı, pan,
mute, tek/çoklu solo, routing, atomik hata, reorder ve detach/delete davranışını
kontrol eder. Mixer/workspace ve önceki track sırasını geri alır; asset/dosyalarını
temizler. 85000 saniye aralığında mevcut klip varsa çalışmayı reddeder.

Manuel: track oluştur/adlandır/sırala, klipler arasında cross-track drag, M/S,
gain/pan, undo/redo; proje kaydet/aç sonrası ID/sıra/routing/track_id aynı mı;
track'siz eski proje ve mevcut clip/export regresyonu. M2 toplu seçim ve M3 zarf
noktası çizimi, ayrı kurgu audition transport'u henüz tamamlanmış sayılmıyor.


## M2/M3 son kaynak teslimi ve M4 kapanış kontrolü

Kullanıcı M1 track modelini derleyip çalıştığını bildirdi. İsteği üzerine bu aşamada
yeni test scripti yazılmadı. Kalan ana kurgu işlemleri uygulanarak mevcut plan ilerletildi:

- Ctrl+click çoklu seçim; Tools clip listesi çakışan klipleri de ayrı seçtirir. Primary
  selection `selected_clip`, tam seçim `selected_clips` üzerinden canonical workspace'tedir.
- `clip_batch(ids, action, delta_seconds?, time_seconds?, patch?)` atomik move, duplicate,
  remove, align_start, align_end ve ortak gain/pan/muted/bus/track_id update uygular.
  Hatalı işlem next_id dahil bütün durumu korur. Duplicate varsayılan olarak grubun
  toplam genişliği kadar sağa kopyalar. Hizalama aynı başlangıç/bitiş hedefini kullanır.
- Çoklu taşıma track ilişkilerini koruyan yatay görsel draft'tır; release tek authoring
  undo işlemi, Escape iptal. Track değiştirme Tools'tan açık bulk update'tir. Script/IPC
  seçimi veya seçili klipleri değiştirdiğinde draft bırakılır. Tek klip drag mevcut
  canlı gesture yolunu korur. Sağ tık seçime yönelik duplicate/delete/alignment sunar.
- Arrange altında gain/pan envelope: çift tık ekle, drag değiştir, sağ tık sil. Gain
  dB gösterilir ama canonical gain_curve linear multiplier saklar; pan -1..1. Draft
  release'te tek clip_update/undo olur, Escape iptal; concurrent clip edit draft'ı iptal
  eder. Trim'den gelen envelope_offset/horizon korunur; görünüm dışında kalan knot'lar
  kaybolmaz. `designer_envelope=none|gain|pan` UI/script/IPC üzerinden ortak ayardır.
- Arrange Play/Stop/Loop artık bağımsız klip miksini dinler; scene timeline/frame
  başlatılmaz. `arrange_preview(ids=[], start_seconds=0, end_seconds=5, loop=false)`,
  `arrange_stop`, `arrange_status` aynı çekirdeği kullanır. Boş ID seçimi tüm kliplerdir;
  seçim varsa yalnız onlar dinlenir, track/bus mute/solo yine geçerlidir.
- Seçim audition aralığını ilk başlangıçtan en fazla 30 saniyeye ayarlar; Tools'ta
  gerçek aralık açıkça gösterilir ve değiştirilebilir. Çok uzak seçimlerin tamamı tek
  30 saniyelik aralığa sığmayabilir. `audition_start_seconds/end_seconds` workspace
  alanları 0.05..30 saniyelik aralığı atomik doğrular. Preview en fazla 32 intersecting
  aktif klip ve 256 MiB decoded asset bütçesiyle sınırlıdır; dosya/asset oluşturmaz.
- Audition hareketli cursor ve peak/RMS gösterir. Clip/track/mixer/selection/range
  değişimi eski miks PCM'ini temizler; tekrar Play yeni miks üretir. Bu henüz kesintisiz
  realtime DSP değildir. Sahne sesleri audition sırasında mevcut Stop All izolasyonuyla
  durur; otomatik sesleri yeniden açmak için mevcut Resume automatic/scene Play kullanılır.
- Ön dinleme ve offline export artık sample interpolation, clip envelope, track ve
  bus/Master gain/pan için ortak AudioTimelineMix hesabını kullanır. Preview mevcut
  global output mute'u izler; export preview mute'tan bağımsızdır. Üretim/analiz
  arka plan işleri genel ses planındaki ayrı aşamadır.

M4: track/clip/curve verileri mevcut AudioService save/restore ve portable asset
packaging yolunda korunur. Batch authoring scene history'ye tek mutation olarak
girer. View/selection/audition state transient workspace'tir. Yeni test yazmadan
son kontrol için kullanıcı build'i sonrasında şu kısa manuel tur yeterli ilk kapıdır:

1. İki track'e birkaç klip ekle; Ctrl+click seç, grup taşı, Escape/undo/redo dene;
   hizala/çoğalt/sil ve track'e toplu taşı.
2. Gain/pan noktaları ekle/drag/sil; trim/split sonrasında eğri konumu korunuyor mu.
3. Arrange Play/Stop/Loop, preview cursor ve aralığı; track M/S ve değişiklik sonrası
   eski miksin durması. Preview ile aynı aralığın export'unu dinle.
4. Projeyi yeni adla kaydet/aç: ID, track sırası, track_id, source offset, loop ve
   envelope knot'ları aynı mı. Track'siz eski proje ve mevcut kayıt/capture yolu.

Kaynak delimiter/boyut/whitespace, VS kayıt/XML, 76 benzersiz audio işlemi ve IPC
capability/security mirror kontrolleri geçti. Son revizyon derlenmedi ve canlı
koşturulmadı; kullanıcı build/kabul sonucu gelmeden M4 doğrulama tamamlandı denmeyecek.
Per-track meter, keyframe easing, ripple editing ve spektral boyama bu ana kurgu
tesliminin tamamlanmış özellikleri olarak sunulmuyor.


## 2026-10-15 dış IPC kontrol noktası

Açık boş sahnede mevcut track kabul probu geçti. Ek dar canlı tur; batch move,
duplicate, align_start, ortak patch, geçersiz negatif hareketin atomik reddi,
gain/pan curve yazımı, Arrange loop/ilerleyen cursor/PCM enerjisi ve klip değişimiyle
eski audition'ın durmasını doğruladı. Geçici varlıklar temizlendi; audio asset/clip/track
sayıları sıfıra döndü. Bu, manuel UI undo/redo veya proje save/reopen kanıtı değildir;
M4 bu kapıları bekler. Yeni worker kodu bu binary'de yoktur; ayrı derleme/probu
[AUDIO_BACKGROUND_JOBS.md](AUDIO_BACKGROUND_JOBS.md) içinde verilmiştir.
