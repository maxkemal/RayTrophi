# Sahne sesi, timeline miks ve çalıştırılabilir çıktı planı

> **Durum:** AKTIF — sahne sesi / timeline miks / EXE çıktı planı; 1. aşama kaynakta, derlenmedi

Tarih: 2026-10-11. Durum: PLANLANDI; uygulama teslimi değildir.

1. aşamanın ilk kaynak uygulaması eklendi; derleme ve görsel/işitsel kabul bekliyor.
Kapsam, mevcut sınırlar, script/IPC sözleşmesi ve kullanıcı doğrulama listesi:
[SCENE_AUDIO_STAGE1_API.md](SCENE_AUDIO_STAGE1_API.md). Aşama kabulü tamamlanmış değildir.

2026-10-12: sürekli rüzgâr/yağmur tepkisi ve dış domain temas sinyali girişi için ilk
kaynak uygulaması eklendi; derleme/canlı kabul bekliyor. Sözleşme ve sınırlar:
[SCENE_AUDIO_REACTIONS.md](SCENE_AUDIO_REACTIONS.md). Otomatik domain telemetrisi,
yerel bulut yağışı ve lightning/thunder olay adaptörü henüz bağlanmış değildir.

Devam: [SCENE_AUDIO_AUTOMATION.md](SCENE_AUDIO_AUTOMATION.md) ile otomatik kaynak
başlatma/sönme, hazır atmosfer bağlantısı ve mevcut karma temas enerji raporunu okuyan
ilk domain adaptörü eklendi; derleme/canlı kabul bekliyor. Malzeme çifti ve kesin temas
konumu telemetrisi, kuru DEM dökülme sesi, yerel yağış ve thunder olayları hâlâ açık.

## Ürün hedefi

RayTrophi açık kaynak kalır. Kullanıcı sesleri assets üzerinden ekler, 3D sahnede
gizmo ile yerleştirir, timeline üzerinde düzenler ve miksler. Aynı sahne RT,
RayFusion, material ve solid önizlemede; çevrimdışı render çıktısında ve gelecekteki
çalıştırılabilir simülasyon paketinde tutarlı ses davranışı sunar. İlk hedef küçük
ama tamamlanmış bir oyun/simülasyon ses altyapısıdır; tam bir DAW veya akustik
simülatör ilk teslim kapsamı değildir.

Kullanıcı doğrulaması: EXE simülasyon çıktısı şu an yok, ileride geliştirilecek.
Mevcut kaynak incelemesinde miniaudio tabanlı ses motoru da doğrulanmadı.
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
Görsel ve işitsel kabul kullanıcıyla yapılır. Derlemeleri kullanıcı çalıştırır;
gereksiz test scripti üretilmez. Zaman/örnek hesabı, serialization ve event tekilleştirme
gibi kritik mantık için gerekli dar doğrulamalar planlanır; canlı IPC dış süreçten
yürütülür. Hiçbir aşama yalnız kaynak yazıldığı için kullanıcı kabulü tamamlandı
olarak işaretlenmez.
