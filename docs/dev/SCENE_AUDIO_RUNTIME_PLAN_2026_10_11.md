# Sahne sesi, timeline miks ve çalıştırılabilir çıktı planı

> **Durum:** AKTIF — temel sahne sesi ve timeline transport kullanıcı tarafından doğrulandı; bus mikserinin kısa canlı IPC/PCM kabulü kullanıcı derlemesinde geçti. Ayrıntılı kalan kontroller: [SCENE_AUDIO_MIXER.md](SCENE_AUDIO_MIXER.md).

Tarih: 2026-10-11; uygulama durumu 2026-10-12 güncellendi. Tam plan henüz tamamlanmadı.

2026-10-13: [Matter ses kanalları](SCENE_AUDIO_MATTER_CHANNELS.md) ile Vulkan DEM
normal temas sönüm işi, süre, güç ve temas merkezi mevcut ses/RAM cache hattına
bağlandı. Aynı domain içinde DEM, karma temas, sıvı/gaz akışı ve MPM deformasyonu
ayrı seçilir; sıvı/gaz ve MPM üreticileri henüz yoktur ve sessiz/ölçümsüz kalır.
Shader revizyonu 25 kullanıcı tarafından derlendi. Canlı DEM/ses bağlantısı ve
24/24 RAM kare karşılaştırması geçti; kullanıcı Play ile doğru çalıştığını doğruladı.
DEM sürtünme/yuvarlanma, sıvı/gaz akışı ve MPM deformasyon üreticileri hâlâ açık.

2026-10-13 devam durumu: güçlü yakın çatlama + uzun doğal thunder kuyruğu ve derin
uzak thunder kayıtları sahneye bağlandı; kullanıcı ses sonucunu beğendi. Kaydedilmemiş
sahne yeniden açılınca bağlantılar dış IPC ile tekrar kuruldu; kalıcılık için sahne
kaydı gerekir. [Yıldırım kanal büyümesi](LIGHTNING_CHANNEL_GROWTH.md) kaynakta hazır;
bu konuşmada yeni shader/Studio derlemesi ve görsel kabul henüz doğrulanmadı.

Kullanıcının istediği sonraki UI işi: ses düzenini mevcut editör omurgasında ele al.
Asset seçimi/dinleme, seçili kaynak ve fizik kanal bağlantısı sağ contextual dock'ta;
bus mikseri ve ileride waveform/klip kurgu alt editörde yer alır. Atmosfer ve thunder
bağlantıları anlaşılır bir sahne ses bölümünde toplanır; teknik tanılama katlanır.
Gain, transport, otomasyon durumu ve hata mesajları tutarlı gösterilir. Kalıcı büyük
shelf ekleme; UI, script ve IPC aynı servis işlemlerini kullanmaya devam eder.
Bu madde plan kaydıdır; UI yeniden düzenlemesi henüz uygulanmadı.

Temel ses/asset sahne bağlantıları ve timeline Play/Pause/Stop kullanıcıyla doğrulandı.
Taşınabilir proje, parent ve diğer ayrıntılı kabul maddeleri ayrı izlenir.
Kapsam, mevcut sınırlar, script/IPC sözleşmesi ve kullanıcı doğrulama listesi:
[SCENE_AUDIO_STAGE1_API.md](SCENE_AUDIO_STAGE1_API.md). Ayrıntılı kontroller ilgili belgededir.

2026-10-12: sürekli rüzgâr/yağmur tepkisi ve dış domain temas sinyali girişi için ilk
kaynak uygulaması eklendi; derleme/canlı kabul bekliyor. Sözleşme ve sınırlar:
[SCENE_AUDIO_REACTIONS.md](SCENE_AUDIO_REACTIONS.md). Sonraki teslimlerde otomatik
domain temas adaptörü ve thunder bağlandı; yerel yağış ve malzeme çifti hâlâ açık.

Devam: [SCENE_AUDIO_AUTOMATION.md](SCENE_AUDIO_AUTOMATION.md) ile otomatik kaynak
başlatma/sönme, hazır atmosfer bağlantısı ve mevcut karma temas enerji raporunu okuyan
ilk domain adaptörü eklendi; derleme/canlı kabul bekliyor. Malzeme çifti ve kesin temas
konumu telemetrisi, kuru DEM dökülme sesi ve yerel yağış hâlâ açık. Thunder bağlantısı
ve mesafe gecikmesi kullanıcıyla doğrulandı.

2026-10-14 yeni teslim (kaynakta; sonraki kullanıcı derlemesi ve kabul bekliyor):
alt barda tek Audio girişi; editör içinde Timeline/Mixer; canlı klip zamanı/bus
sürükleme, gain çizgisi, fade tutamaçları ve cursor konumunda Razor kesme.
Mixer kanal şeritleri ve metreleri yenilendi. Gain/fade/zaman değişimleri tüm klip
voice'larını yeniden kurmaz. [Audio workspace](AUDIO_WORKSPACE_UI.md).
İlk çevrimdışı prosedürel tone/wind/metal/wood asset üreticisi UI/script/IPC ile
kaynakta eklendi: [Sentetik ses](SCENE_AUDIO_SYNTHESIS.md). Bu teslim fizik
telemetrisinden sürekli malzeme sesi sentezini tamamlamaz; sonraki aşama açık.

2026-10-14 ölçekleme güvenliği: fizik ses telemetrisi varsayılan kapalı, açık kullanıcı
seçimiyle etkinleşir. UI/script/IPC ortak sahne policy'si; kapalıyken DEM ek audio
tail/readback, sıvı/MPM taraması ve gaz audio dispatch'i atlanır. Revision 27
kullanıcı Studio/shader derlemesi ve canlı kabul bekler. Ayrıntılar ve sonraki
GPU reduction/geometry acoustics mimarisi: [Telemetri policy](SCENE_AUDIO_TELEMETRY_POLICY.md).

## Ürün hedefi

2026-10-12 sonraki adım: CPU karma domain temaslarının darbe ağırlıklı hücre merkezi
ve adım süresine göre güç (W) ses eşlemesi kaynakta eklendi; derleme/kabul bekliyor.
Sözleşme, GPU sınırı ve kontrol listesi:
[SCENE_AUDIO_DOMAIN_CONTACTS.md](SCENE_AUDIO_DOMAIN_CONTACTS.md).

RayTrophi açık kaynak kalır. Kullanıcı sesleri assets üzerinden ekler, 3D sahnede
gizmo ile yerleştirir, timeline üzerinde düzenler ve miksler. Aynı sahne RT,
RayFusion, material ve solid önizlemede; çevrimdışı render çıktısında ve gelecekteki
çalıştırılabilir simülasyon paketinde tutarlı ses davranışı sunar. İlk hedef küçük
ama tamamlanmış bir oyun/simülasyon ses altyapısıdır; tam bir DAW veya akustik
simülatör ilk teslim kapsamı değildir.

Kullanıcı doğrulaması: EXE simülasyon çıktısı şu an yok, ileride geliştirilecek.
Miniaudio tabanlı ses motoru eklendi ve kullanıcı derlemesinde çalışıyor.
EXE paketleme ayrı gelecek iş paketi ve bağımlılıktır; yalnız ses
motorunun eklenmesi EXE export özelliğini tamamlamaz.

## Format ve bağımlılık kararı

WAV eski bir kapsayıcıdır fakat PCM WAV kayıpsız çalışma/miks teslimi için uygundur;
yaşı kaliteyi düşürmez. Uzun seslerde depolama ve dağıtım boyutu büyüktür. Tek format
dayatılmayacak:

| Amaç | Format/politika |
|---|---|
| İçe aktarma, kısa efekt, kurgu teslimi | WAV PCM; FLAC kayıpsız alternatif |
| EXE paketinde kısa ve sık tetiklenen efekt | Önceden çözülmüş PCM; kaynak dosya WAV/FLAC/Opus olabilir |
| EXE paketinde uzun ortam/müzik | Ogg kapsayıcısında Opus, akıştan çözme |
| Kayıpsız miks dışa aktarma | FLAC varsayılan; WAV seçilebilir |
| Video paylaşımı | MP4 + uyumlu ses kodlayıcısı; WebM + Opus alternatifi |

Motor miks saatinin başlangıç standardı 48 kHz, float32; kaynak örnekleme hızları
dönüştürülür. Konumlu nokta efektleri mono tercih edilir. Stereo ortam/müzik
yönsüz veya ayrı genişlik politikasına sahip olur. Kullanıcı kaynak dosyası
değiştirilmez; paketleme sırasında üretilen sürümler önbelleğe alınır.

Miniaudio temel cihaz, miks ve konumsal oynatma motorudur. WAV/FLAC/MP3 yerleşik;
Opus için libopus + libopusfile decoder entegrasyonu gerekir. Opus, ilk küçük
oynatma prototipinin değil, runtime paketleme kabulünün bağımlılığıdır.
Gelişmiş HRTF/occlusion yerleşik değildir. İleride ayrı bir uzamsal işleme
adaptörü eklenebilir; temel servis bu adaptöre bağımlı kurulmaz.

Miniaudio Public Domain veya MIT No Attribution seçeneği sunar. Opus referans
uygulamasının BSD lisansı ve kullanılan decoder bağımlılıklarının lisansları
dağıtım manifestine eklenir. FFmpeg lisansı seçilen derleme seçeneklerine bağlıdır;
binary, kaynak erişimi ve bildirimler birlikte değerlendirilir. Açık kaynak olmak
asset veya codec yükümlülüklerini kendiliğinden kaldırmaz. Lisans/patent konularında
evrensel risksizlik iddiası yapılmaz.

Kaynaklar:
- https://github.com/mackron/miniaudio/blob/master/LICENSE
- https://miniaud.io/docs/manual/index.html
- https://miniaud.io/docs/examples/custom_decoder_engine.html
- https://opus-codec.org/license/
- https://ffmpeg.org/legal.html

## Ortak veri ve servis

Odaklı Audio modülleri oluşturulur; 2000 satır üzerindeki dosyalara yalnız küçük
include/kayıt/çağrı bağlantıları eklenir. Renderer başına ayrı ses motoru kurulmaz.

- AudioAsset: kalıcı kimlik, dosya, süre, kanal/örnekleme bilgisi, dalga biçimi
  önbelleği, döngü noktaları, lisans/author/source, paketlenmiş sürüm bilgisi.
- AudioEmitter: sahne nesnesi kimliği, dönüşüm/parent, asset, bus, gain, pitch,
  mesafe eğrisi, yön konisi, döngü, tetikleme ve öncelik. Ses nesnesi geometri değildir.
- AudioClip: asset/source bağlantısı, timeline başlangıcı, kaynak giriş/çıkışı,
  süre, döngü, fade, gain eğrisi; düzenleme kaynak dosyasını bozmaz.
- AudioBus: Master, Weather, Effects, Ambience, Music başlangıç grupları; gain,
  mute/solo ve seviye ölçümü. Kaynak/klip/bus sessizliği önizleme sessizliğinden ayrıdır.
- AudioEvent: kalıcı olay kimliği, onset, konum, seed, kategori/yoğunluk;
  görsel olay ile sesin ortak kaynağıdır.
- AudioService: ekle/sil/çoğalt/bağla, klip düzenleme, miks, transport, event,
  offline çıktı. UI, script ve IPC aynı doğrulanan servisi çağırır.

Proje serialization, undo/redo, asset taşıma ve eksik dosya çözümü aynı teslimde
gelir. Eski sahneler sessiz ses bölümüyle açılır. Geçersiz ID, aralık, bozuk dosya,
NaN ve eksik cihaz açık hata/durum üretir; cihaz yokluğu görüntü renderını engellemez.

## Saatler, renderer bağımsızlığı ve yıldırım

Timeline zamanı ve canlı runtime zamanı açıkça ayrılır. Realtime hava olayları
çalışırken timeline duruyor olabilir; kullanıcı timeline sesi ve canlı ortam sesini
ayrı açabilir. Aynı hava olayı iki saatten iki kez tetiklenmez.

Görsel yıldırımın mevcut C++ üretimindeki kimlik/onset/konum hesabı ortak olay
üreticisine çıkarılır. GPU geri okuması veya her render örneğinde tetikleme yapılmaz.
Ses, görsel etkinin ömrü bittikten sonra da gecikmeli başlayabilir; olay kaydı
yalnız o an görünür dört GPU slotuyla sınırlanmaz. Uzaklık gecikmesi ve boğuklaşma
uygulanır. Geniş yatay olaylar sınırlı sayıda kaynakla uzun gürültü üretir.

Play/Pause/Stop, seek, loop ve playback hız değişimi tanımlanır. Seek sırasında
eski olay kuyruğu temizlenir; uzun klipler doğru kaynak örneğine konumlanır.
Varsayılan scrub sessizdir; kısa dinleme ayrı seçenek olabilir. Önizleme mute,
transport ilerlemesini ve export miksini değiştirmez. Audio callback içinde sahne
kilidi, dosya erişimi, shader/geometry işi ve bellek tahsisi yapılmaz; komut kuyruğu
ve hazır ses verisi kullanılır. Render yavaşlarken ses donmaz; görüntü zamanına
senkronizasyon ve aşırı gecikmede davranış açıkça tanımlanır.

## Editör ve assets

Mevcut viewport-first omurga korunur: Add > Audio Source, seçme/silme/çoğaltma,
transform gizmo, ikon, attenuation halkaları ve yön konisi. Parent dönüşümü ve
sahne birimleri dinleyici/kaynak koordinatlarına tutarlı çevrilir. Kameranın
yönü ve aktif dinleyici tercihi uygulanır.

Alt editör timeline ses kanalları ve waveform gösterir: sürükleme, trim, split,
duplicate, loop, fade, snapping, klip gain ve automation. Sağ dock seçili kaynak
ve klip özelliklerini sunar. Mikser alt editör sekmesi/dock içinde açılır; kalıcı
büyük shelf eklenmez. Önizleme hoparlör düğmesi tüm shading modlarında çalışır.

AssetRegistry'nin mevcut license/source/author/tags alanları kullanılır. Audio
türü, süre, kanal, örnekleme, preview play/stop, waveform ve kategori filtreleri
eklenir. Viewport'a drop kaynak, timeline'a drop klip üretir. Preview yalnız bir
asset'i çalar; sahne transportuyla çakışma politikası tanımlıdır.

Hazır başlangıç paketi: thunder yakın/uzak varyantları, rain hafif/şiddetli,
wind sakin/gust. Yalnız dosya bazında CC0 kanıtı bulunan içerikler paketlenir;
lisans metni ve kaynak manifesti birlikte saklanır. Genel bir sitenin ücretsiz
olması yeterli değildir. Otomatik izinsiz indirme veya belgesiz içerik eklenmez.

## Render ve video çıktısı

Seçili kare aralığı ve rasyonel FPS tek export manifestinde tutulur. Başlangıç
karesi, bitişin dahil olup olmadığı ve audio örnek sayısı açık tanımlanır;
float zaman biriktirmeyle drift üretilmez. Görüntü render süresi audio süresi değildir.

PNG + FLAC/WAV, yalnız miks ve video seçenekleri aynı offline AudioService
işleme yolunu kullanır. Kamera hareketi, emitter animasyonu, gain automation ve
deterministik olaylar seçili zamanda değerlendirilir. Ses offline bloklar halinde
üretilir; canlı hoparlör çıktısı kaydedilmez. Başlangıçtan önceki yıldırımın seçili
aralığa ulaşan gecikmeli sesi hesaba katılır. Renk/render ayarı audio saatini bozmaz.

Video encoder ayrı adaptör/işlemdir. MP4/WebM seçenekleri codec bulunabilirliğini
ve lisans manifestini kontrol eder. Kareler ve miks korunur; iptal/hata durumunda
tamamlanmamış dosya final çıktı gibi gösterilmez. Yeniden paketleme görüntüyü
tekrar render etmeyi gerektirmez. Önizleme mute ve export include-audio ayrıdır.

## EXE simülasyon/runtime hedefi

İlk teslim tek dev EXE içine her şeyi gömmek değildir: çalıştırılabilir player +
sahne/asset paketi + lisans manifesti. Tek dosyalı taşıma sonradan değerlendirilebilir.
Player editör panellerine bağlı olmadan aynı AudioService ve olay üreticisini
kullanır; geliştiricinin mutlak dosya yolları pakete sızmaz. Başka bilgisayarda
Studio kurulumu olmadan çalışması ayrı kabul koşuludur.

Oyun motoru düzeyinde başlangıç davranışı: hareketli dinleyici ve parent'lı kaynak,
tek atımlık/döngülü olaylar, bus miksleri, pause/resume, güvenilir cihaz yaşam döngüsü,
asenkron asset yükleme, uzun ses streaming ve öncelikli voice yönetimi. İlk hedef
32 aktif fiziksel voice, ayarlanabilir tavan; uzaktaki/sessiz kaynaklar sanallaştırılır,
zamanları ilerler ve tekrar duyulur olunca doğru noktadan döner. Olay kuyruğu,
decode belleği ve efekt maliyeti sınırlandırılır; GPU renderer hızına bağlı değildir.

Runtime olay bindings: weather/lightning ve açık play/stop/gain/position işlemleri
ilk kapsam. Çarpışma/ateş/su sesleri ilgili sistemin olaylarını dinleyen sonraki
adaptörlerdir; diğer ajanın simülasyon çekirdeğine bu iş kapsamında değişiklik
eklenmez. Geometrik occlusion için gelecekte düz TriangleMesh/DNA SoA/BVH yolu
kullanılır; per-face facade sahne kaynağı olmaz. İlk occlusion aşaması düşük
sıklıklı bütçeli görünürlük + filtre; tam akustik ray tracing değildir.

## Ulaşılabilir teslim sırası ve kabul

1. **Çekirdek + sahne + assets:** miniaudio, mono konumlu kaynak/stereo ortam,
   gizmo CRUD, paketlenebilir asset kimliği, serialization/undo, script/IPC.
   Kabul: taşı/sil/çoğalt ve yeniden aç; eksik cihaz/dosya güvenli; tüm modlarda dinle.
2. **Timeline + mikser:** waveform, trim/split/fade, gain eğrisi, bus/mute/solo,
   transport ve seek/loop. Kabul: kaynak değişmeden düzenleme; seek sonrası çift
   ses yok; önizleme sessizken timeline ilerler.
3. **Hava olayları:** ortak lightning event kaydı, gecikmeli thunder, wind/rain
   geçişleri, belgeli CC0 başlangıç paketi. Kabul: RT örneklerinden bağımsız tek
   olay; uzun mesafe/seek/döngü; viewport ve offline olay kimliği eşleşir.
4. **Offline çıktı:** aynı evaluator ile FLAC/WAV, PNG manifesti ve sonrasında
   MP4/WebM adaptörü. Kabul: 24, 30 ve 30000/1001 FPS aralıkları; kamera hareketi;
   drift/başlangıç kayması yok; iptal ve yeniden paketleme düzgün.
5. **Runtime paket:** Opus decoder/streaming, voice bütçesi, taşınabilir player
   entegrasyonu. EXE export altyapısının teslimine bağlıdır. Kabul: Studio olmayan
   bilgisayarda assets/lisanslarla çalışma; mute, pause ve cihaz değişimi.
6. **Gelişmiş uzamsal ses:** bütçeli occlusion, ortam bölgeleri/yankı ve gerekirse
   HRTF adaptörü. Temel sistem bu aşamayı beklemez; maliyet ölçülmeden varsayılan açılmaz.

Gecikmeli gök gürültüsü için etkileşimli adaptör ve CC0 varyasyon paketi eklendi;
ortak görsel yıldırım üreticisi, olay tekilleştirmesi, konum/şiddet, uzaklığa bağlı
gecikme ve UI/script/IPC sözleşmesi:
[SCENE_AUDIO_THUNDER.md](SCENE_AUDIO_THUNDER.md). Derleme ve canlı dinleme kabulü
bekliyor. Bu teslim offline ortak olay evaluator'ünü veya 3. aşamanın bütün kabul
koşullarını tamamlandı olarak işaretlemez.

Her aşama UI/script/IPC, hata semantiği ve dokümantasyonuyla birlikte teslim edilir.
Sürekli önizleme / timeline kontrolü seçimi, duraklayan ses cursor'ları, sahne
zamanına bağlı thunder/signal süreleri ve ortak kayıt/UI/script/IPC sözleşmesi
[SCENE_AUDIO_TRANSPORT.md](SCENE_AUDIO_TRANSPORT.md) içinde belgelenmiştir.
Yeni transport davranışının kullanıcı derlemesi ve canlı kabulü beklemektedir.
Görsel ve işitsel kabul kullanıcıyla yapılır. Derlemeleri kullanıcı çalıştırır;
gereksiz test scripti üretilmez. Zaman/örnek hesabı, serialization ve event tekilleştirme
gibi kritik mantık için gerekli dar doğrulamalar planlanır; canlı IPC dış süreçten
yürütülür. Hiçbir aşama yalnız kaynak yazıldığı için kullanıcı kabulü tamamlandı
olarak işaretlenmez.

2026-10-12: gerçek Master/Weather/Effects/Ambience/Music bus grafiği, gain/mute/solo,
PCM peak/RMS göstergeleri ve kayıt/undo/UI/script/IPC sözleşmesi kaynakta eklendi:
[SCENE_AUDIO_MIXER.md](SCENE_AUDIO_MIXER.md). Yeni mixer derlemesi ve canlı kabul
bekliyor; waveform/trim/split/fade ve klip seek çalışmaları henüz başlamadı.


## 2026-10-14 kullanıcı kabulü ve devam

Kullanıcı sentetik ses sonucunu, klip fade ve canlı taşımayı doğruladı. Yeni kaynak
değişikliği: tam genişlikte inspector alanları; sağ kenardan kaynak sınırını aşınca
açık auto_loop politikasıyla tekrar eden clip_resize UI/script/IPC işlemi.
Son UI/resize değişiklikleri yeni Studio derlemesini bekler; shader değişikliği yok.
Ayrıntılar: [Audio workspace](AUDIO_WORKSPACE_UI.md).

## 2026-10-14 sound design continuation

Approved direction: [Sound design plan](AUDIO_SOUND_DESIGN_PLAN_2026_10_14.md).
Source delivery: [Sound designer v2](AUDIO_SOUND_DESIGN.md): in-memory preview,
DaisySP synthesis/filtering, Signalsmith independent pitch/time, waveform/envelope/
filter/spectrum and name-based save to the existing project/library folder.
Studio build, external IPC probe and listening/visual acceptance remain pending.

## 2026-10-14 sound design continuation

Approved direction: [Sound design plan](AUDIO_SOUND_DESIGN_PLAN_2026_10_14.md).
Source delivery: [Sound designer v2](AUDIO_SOUND_DESIGN.md): in-memory preview,
DaisySP synthesis/filtering, Signalsmith independent pitch/time, waveform/envelope/
filter/spectrum and name-based save to the existing project/library folder.
Studio build, external IPC probe and listening/visual acceptance remain pending.

2026-10-14 latest build: user confirmed Studio compilation after Sound Designer
MSVC fixes. New Designer listening/UI/external IPC results are pending. This does
not close shader telemetry, full capture/replay, video/player or geometry-acoustics gates.
