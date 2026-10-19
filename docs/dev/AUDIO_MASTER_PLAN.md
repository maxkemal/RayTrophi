# Audio ana planı: mevcut sistemi kapatma ve doğal sentetik ses

> **Durum:** AKTİF — 2026-10-19. Kullanıcı önce ana planın, ardından bütün olay
> aileleri için sentetik üretici planının yazılmasını istedi. Bu bir plan teslimidir;
> yeni DSP uygulaması, derleme veya tüm audio sisteminin kabulü değildir.

2026-10-19 kod partisi: A1'in odaklı menü/kısayol yönlendirmesi ve mevcut sekiz
üreticide açık v2 motor seçimi kaynakta. Ayrıntılar [üretici kaydında](PHYSICAL_AUDIO_GENERATION.md).
A1'in remap bölümü, A2 ve ileri S/A partileri bu kod teslimiyle tamamlanmış sayılmaz.

## Hedef ve belge önceliği

Ses seçme/üretme, düzenleme, sahnede fiziksel olaya bağlama, dinleme, kaydetme ve
yeniden üretme tek anlaşılır sistemde çalışmalı. Asıl kalite hedefi gerçek doğa
olaylarına uygun sentetik sestir. Kaliteyi yalnız EQ, ses yüksekliği veya analiz
grafiğinin hareket etmesiyle değerlendirmeyeceğiz.

Bu belge kalan işlerin sırasını ve kapanış koşullarını birleştirir. Üreticinin
teknik uygulama sırası [sentetik üretici planındadır](PHYSICAL_AUDIO_SYNTHESIS_PLAN.md).
[Kalite kapsamı](PHYSICAL_AUDIO_QUALITY_CLOSURE.md) bütün olay ailelerini tanımlar.
Eski tarihli planlar uygulama geçmişi ve ayrıntılı sözleşme olarak korunur; eski
“henüz yok” notları daha yeni kaynak teslimlerini geçersiz kılmaz.

## Mevcut durum: yeniden yazılmayacak temel

| Alan | Mevcut dayanak | Kapanmamış sınır |
|---|---|---|
| Asset/kaynak, Timeline/Mixer | Ortak audio servisi, kalıcı kimlikler, track/clip ve otomasyon | Son kayıt/açma/taşınabilirlik ve kapsamlı UI kabulü |
| Designer/EQ | Ortak DSP, ad/kaynak bağlamı, kayıt ve hedef track/zaman, kompakt başlık | Klipten işleme/remap, odaklı Undo/Redo, ayrıntılı uçtan uca kabul |
| İşleme kapasitesi | 600 s input/output; tahmini 512 MiB PCM bütçesi ve ayrı import sınırları | Streaming tam dosya işleme; sınırsız süre desteklenmiyor |
| Canlı Analyze | Gerçek mixer PCM, spektrum/bantlar, alternatif grafikler, mouse gezinme | Kullanıcı canlı analiz ve Ctrl+wheel'i doğruladı; bütün grafik/API kabulü aynı şey değil |
| Analyze durum satırı | Sabit yükseklik düzeltmesi kaynakta | Son kullanıcı build/görsel kontrolü |
| Fiziksel üretici | Sekiz model, ortak PhysicalAudioDsp | İlk modeller doğal ses kalitesini tamamlamıyor |
| Geçmiş/replay/bake | PhysicalAudioHistory/Schedule/Timeline/Bake; kayıtlı sınırlı kabul | Exact solver olay geçmişi, otomatik cache bağlantısı, akustik çıktı eşliği |
| Sahne akustiği | Doğrudan yol, spektral işleme, erken yansımalar ve sınırlı diffuse yol | Geniş fizik kapsamı, çoklu/late reverb, export ve ölçek kabulü |

2026-10-18 fiziksel kabul kayıtları temel/prosedürel PCM ve deterministik kuru bake
kontrolleri içerir: `audio_physical_acceptance_20261018.json` ve
`audio_physical_continuation_20261018.json`. Bunlar bütün doğa olaylarında gerçekçilik,
tam fiziksel enerji veya tüm export zinciri onayı değildir. Kontrol sayıları ve eski
başarılar yeni değişiklikleri otomatik kabul edilmiş yapmaz.

## İş sırası ve kapanış kapıları

| Parti | İş | Tamamlanma koşulu |
|---|---|---|
| A0 — kapsam/altyapı | Mevcut sözleşmeler, modül sahipliği, kabul sahneleri ve domain veri arayüzü | Aynı işi iki yerde yazmadan uygulanabilir dosya/API listesi |
| A1 — mevcut authoring kapanışı | Odaklı Undo/Redo, kaynak/klip bağlamı, güvenli yeni asset/remap | Doğru geçmiş, korunmuş klip zamanlaması ve atomik hata davranışı |
| A2 — kayıt ve proje güveni | Kaydet/aç, taşınabilir assetler, tarif/track/clip kimlikleri ve eksik dosya davranışı | Round-trip ve hatadan kurtarma kabulü |
| S0–S6 — sentetik üretici | Sürümlü ortak çekirdek, bütün aileler, olay/geçmiş ve doğal karakter | Her aile için fizik/veri, deterministik çıktı ve dinleme kabulü |
| A3 — akustik çıktı eşliği | Canlı/replay/export aynı belirtilmiş yayılım kapsamını kullanır | Dry ve işlenmiş çıktı karşılaştırması; eksik kapsam açık hata/durum |
| A4 — ölçülebilir analiz | Kaynakta direct/reflected/transmitted bant ayrımı, toplam miksle ayrı anlam | Ortak birim/normalizasyon, eksik veri görünürlüğü, toplamların doğruluğu |
| A5 — ileri kapasite ve yayılım | Streaming; çoklu/late reverb ve diğer fizik katmanları | Her alt özellik kendi performans/doğruluk kabulünü geçer |
| A6 — bütünleşik kapanış | Aile sahneleri, uzun kullanım, 1/8/16/32 kaynak, kayıt/replay/export | Sayısal, işitsel ve performans sonuçları kayıtlı; açık madde gizlenmez |

Önce A0–A2 ile mevcut kullanım ve kayıt temeli kapanır; ardından üretici partileri
uygulanır. A5'teki geniş araştırma işleri doğal ses çekirdeğinin başlamasını
engellemez. Üreticinin akustik olarak tamamlanması A3/A6'ya bağlıdır. Bütün aileler
kapsamdadır; bir ailede başarılı olmak diğerlerini tamamlanmış yapmaz.

## A1: odak, kimlik ve güvenli düzenleme

- Ana menü ve Ctrl+Z/Ctrl+Y/Ctrl+Shift+Z odaktaki editörün geçmişini kullanır.
  Designer tarifi ile sahne kurgusu kendi geçmişini korur. Menü hedefi belirtir;
  text editor/InputText odağında önce yazı geçmişi çalışır. Aynı tuş iki geçmişi
  birden çalıştırmaz; Designer odağında geçmiş boşsa sahneye sessiz fallback olmaz.
- Klipten Designer açılışı asset, kullanılan source aralığı, track ve zaman
  bağlamını taşır. “İşlenmiş dosyayı düzenle” ve “tarifi tekrar aç” ayrı anlamlarını korur.
- Yeni asset kaydetme ile seçili klibi yeni assete bağlama ayrı açık eylemlerdir.
  Remap sadece seçili klibi etkiler; trim/offset/loop/fade/otomasyon için açık eşleme
  yapar. Desteklenmeyen time-stretch/remap durumu sessizce yanlış zaman üretmez.
- Asset kaydı başarılı/klip ekleme başarısız durumu korunur. Yalnız klip eklemeyi
  yeniden denemek yeni WAV üretmez. Dosya yayınlama ile scene history geri alma
  kapsamı görünürdür; Undo kullanıcı dosyasını örtük silmez.

## A2/A3: aynı işin kayıtta ve export'ta karşılığı

Tarif sürümü, seed, asset/clip/track ID'leri, olay/kontrol geçmişi, source aralığı ve
gain/pan/fade ilişkileri kaydet/aç sonrası korunmalı. Ortak servisin doğrulaması
load, UI, Python ve IPC için geçerli olur. Hata/iptal mevcut geçerli kaynağı korur.

Prosedürel kuru bake zaten vardır; yeni bir ikinci exporter yazılmaz. Eksik olan
canlı akustik/konum geçmişi ve export kapsamı aynı sözleşmeye bağlanır. “Dry”,
“akustik işlenmiş” ve “cihaz çıkışı kaydı” aynı sonuç diye sunulmaz. Geçmiş veya
gerekli fizik verisi yoksa mevcut karedeki aktiviteyle geçmiş uydurulmaz.

Streaming aşaması bloklar arası filtre durumunu, uzun dosya iptal/ilerleme bilgisini,
geçici dosyanın başarıda yayınlanmasını ve bellek/disk bütçesini tanımlar. Önceki
600 s / bellek sınırı yalnız bu aşama kabul edildiğinde ilgili işlemler için değiştirilir.

## A4/A5: analiz ve ileri akustik sınırı

Canlı spektrum/bantlar mixer sinyalini ölçer. Direct/reflected/transmitted görünümü
ayrı bir kaynak/yol tanılamasıdır; mixer toplamı gibi sunulmaz. Yol enerjileri ortak
birim ve alıcı/bant normalizasyonu olmadan toplanmaz; bilinmeyen veri sıfırla gizlenmez.

Malzeme geçirgenliği, sınırdaki analitik ortam hesabı ve sahnede gerçekten izlenen
kırılan yol farklı kabiliyetlerdir. Gaz/sıvı/katı geçişi, çok katmanlı iletim,
diffraction, Doppler ve geç yansımalar ayrı kapsam/kabul taşır. Mevcut erken
yansıma desteği bu katmanları tamamlanmış yapmaz.

## Birleşik domain ajanıyla çalışma sınırı

Bu audio çalışma hattı; solver, ortak Substance çekirdeği, GPU buffer ABI, domain
shader/cache düzeni veya diğer ajanın testlerini değiştirmez. Var olan telemetry
yayınını salt okunur tüketir. Yeni ihtiyaçlar aşağıdaki üretici planındaki sürümlü
veri sözleşmesiyle kaydedilir; domain sahibi yayın desteği hazır olduğunda adapter
entegrasyonu yapılır. Bu plan diğer ajana mesaj gönderme veya dosyalarını değiştirme
talimatı değildir.

Domain verisi hazır değilken audio tarafında typed giriş, doğrulama, deterministik
fixture ve açık `unavailable` durumu geliştirilebilir. Test verisi gerçek solver
ölçümü diye gösterilmez. Geometri gerektiğinde flat TriangleMesh/DNA yolu kullanılır.

## Uygulama ve kabul kuralları

- Yeni iş odaklı modüllerde; 2000 satır üzeri dosyalarda yalnız en küçük wiring.
- Ortak core/service, UI/Python/IPC, şema/discovery, hata anlamları ve ilgili test
  aynı partide teslim edilir. İş akışı veya düzenlenebilir durum UI'ya özel tutulmaz.
- Eski tariflerin ses sonucu sürüm belirtilmeden değiştirilmez. Yeni sürüme geçiş
  açık ve geri alınabilir olur; desteklenmeyen eski sürüm sessizce yeniden yorumlanmaz.
- Teknik kabul: finite PCM, block-size tutarlılığı, olay dedup/sıralama/overflow,
  pause/seek/reset, kaynak ömrü, hata atomikliği ve kayıt/export kapsamı.
- Kalite kabulü: aynı ailede düşük/orta/yüksek olay, yakın/uzak, farklı malzeme ve
  üst üste olaylar. Kullanıcı dinlemesi ile referans sahneler birlikte değerlendirilir.
- Performans: hedef donanım ve cihaz buffer'ı kaydedilerek callback bütçesi,
  underrun, ana thread/GPU maliyeti ve bellek. FPS tek başına ses kabulü değildir.

Codex build/shader compilation veya Studio başlatma yapmaz. Kaynak kontrolleri
sonrasında kullanıcı derler; canlı kontroller açık Studio'ya dış süreçten
`scripts/test/rt_ipc.py` ile yapılır. Çalışan domain testleri için scene/playback
değiştiren probe bu plan yazımı sırasında çalıştırılmaz.

## Başvuru

- [Designer bağlamı ve kayıt](AUDIO_DESIGN_CONTEXT_WORKFLOW_PLAN.md)
- [Çok track kurgu](AUDIO_MULTITRACK_AUTHORING_PLAN.md)
- [Canlı analiz](AUDIO_LIVE_ANALYSIS.md)
- [Sahne akustiği yol haritası](SCENE_ACOUSTICS_ROADMAP.md)
- [Mevcut üretici](PHYSICAL_AUDIO_GENERATION.md), [bake](PHYSICAL_AUDIO_BAKE.md),
  [replay](PHYSICAL_AUDIO_REPLAY.md), [saklanan geçmiş](PHYSICAL_AUDIO_STORED_HISTORY.md)
