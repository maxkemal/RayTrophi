# Designer kaynak kimliği ve uçtan uca ses iş akışı

2026-10-19: kaynak incelemesine dayalı öneri; uygulama teslimi veya canlı kabul değildir.
Birleşik domain çalışması kapsam dışıdır. Bu not Designer, asset ve klip akışını kapsar.

## Mevcut durum

- Designer sentetik tarif veya asset aralığını işler. Input seçicisinde assetler vardır,
  ancak kapalı seçici yalnız Synthetic / Audio asset gösterir. Kaynak adı merkezde
  sürekli görünmez. Sound name kaydetme alanındadır.
- Save as asset, değişmiş tarifi önce render eder; başarılı sonuç design_save ile
  float32 48 kHz WAV olur ve asset_import üzerinden yeni assete dönüşür. Kaynak
  dosya korunur, dosya adı çakışması numaralandırılır, tarif/provenance saklanır.
- İsteğe bağlı Insert clip at playhead, yeni asset ID ile clip_create çağırır.
  Kaydetme ve klip oluşturma ayrı undo işlemleridir. Klip ekleme başarısız olursa
  başarılı asset kaydının sonucunu ayrıca göstermek gerekir.
- addTimelineAsset seçili track'i yalnız Designer Edit/Arrange açıkken kullanır.
  Effects/Generate üzerinden kaydetmek seçili track'e eklemeyi garanti etmez.
  Başlangıç scene timeline current_frame / fps değeridir; hedef kullanıcıya açık olmalı.
- design_reuse saklanan tarifi açar. Bu, işlenmiş WAV'ı yeni kaynak olarak açmaktan
  farklıdır. Mevcut klipleri yeni assete yeniden bağlamak save yolunun parçası değildir.
- Designer kaynak aralığı ve stretch/tail dahil işlenmiş çıktı 30 saniye ile sınırlıdır.
  Bu sınır tüm timeline dosyalarının süresine ilişkin bir sınır değildir.

İncelenen yollar: AudioSoundDesignerUI.cpp (saveAsset/pollRender/save kontrolleri),
AudioWorkspaceUI.cpp (addTimelineAsset), AudioSoundDesign.cpp (prepare/save),
AudioService.cpp (design_save). Kaynak bağlantısı canlı uçtan uca doğrulama değildir.

## Öncelik 1: görünür kimlik ve güvenli çalışma geçişi

Merkezde tek kompakt bağlam satırı göster:

- Sentetik: taslak adı, Sentetik taslak, değiştirildi/render bekliyor/kaydedildi durumu.
- Asset: görünen ad, kaynak dosya, işlenen zaman aralığı; tam yol ayrıntıda.
- Klip üzerinden giriş: ayrıca klip adı, track ve source offset/loop bağlamı.

Asset, klip ve üretim tarifi ayrı kimliklerdir. Sentetik taslak dosya üretmeden
adlandırılabilir; kaydedilen asset tarif kökenini taşır, klip bu asset ID'ye bağlanır.
Klip adı asset adından başlayabilir, bağımsız yeniden adlandırma kimliği değiştirmez.

Başlıktan kaynak seçilebilmeli. Başka çalışmayı açmak ile mevcut efekt zincirinde
kaynağı değiştirmek ayrı işlemler olmalı. Geçişte kaydedilmemiş taslak korunmalı veya
açık kaydet/at/iptal seçimi sunulmalı. Eski render/analiz/save sonucu yeni kaynağa
uygulanmamalı. Kalıcı taslak saklama mevcut kabul edilmiş özellik olarak sunulmamalı.

## Öncelik 2: anlaşılır tamamlanma ve klip bağlantısı

Kaydetme alanında iki açık sonuç sun:

1. Asset olarak kaydet.
2. Kaydet ve timeline'a ekle: hedef track ve başlangıç zamanı görünür/değiştirilebilir.

Sonuç özeti asset adı/dosyası ve varsa oluşturulan klip/track/zamanı göstermeli.
Asset kaydı başarılı, klip ekleme başarısız ise ayrı durum ve yalnız klip eklemeyi
yeniden deneme sunulmalı; tekrar deneme yeni WAV/asset üretmemeli.

Klipten Designer açıldığında varsayılan kaynak dosya korunmalı. Yeni asset oluşturma
sonrasında seçili klibi yeni sürüme bağlama açık bir işlem olmalı; diğer klipler
kendiliğinden değişmemeli. Bütün dosyayı işlemek ile klibin kullandığı aralığı işlemek
ayrılmalı. Aralık çıktısına bağlarken offset, süre, loop, fade, otomasyon ve track
korunumu doğrulanmalı; aynı offset körlemesine yeni kesilmiş dosyaya kopyalanmamalı.

Reuse recipe ve işlenmiş dosyayı yeniden işle ayrı adlandırılmalı; aksi halde kullanıcı
aynı EQ'yu ikinci kez uygulayabilir. Undo kapsamı ve dosyanın diskte kalması anlaşılır olmalı.

## Öncelik 3: 30 saniye sınırını aşamalı kaldırma

Ürün hedefi olarak uzun asset ve tam dosya EQ desteği mantıklıdır. Sabiti büyütmek
veya doğrulamayı silmek yeterli değildir: mevcut yol original/processed PCM'i bellekte
tutar, analiz kopyalar; eşzamanlı DSP ve dosya işlemleri ayrıca değerlendirilmelidir.

- Önce uzun asset EQ/uygun efektler için bloklar halinde okuma, işleme ve dosyaya
  yazma tasarlanmalı. EQ filtre durumu bloklar arasında korunmalı; sınırlar klik üretmemeli.
- Tam dosya işleme ile kısa seçimin ön dinlemesi ayrı olmalı. Preview aralığının
  sınırlı olması tam dosya işleme süresini sınırlamamalı; ekran kapsamı açıkça yazmalı.
- Bellek/decoded byte, çıktı frame/disk bütçesi, süre taşması, iptal ve ilerleme
  tanımlanmalı. Geçici çıktı ancak başarıda yayımlanmalı; hata/iptal kaynağı korumalı.
- Stretch, convolution, spectral işlemler ve sentetik katmanlar için ayrı kapasite
  incelemesi gerekli. Desteklenmeyen uzun işlem açık hata vermeli, sessizce kesmemeli.
- Sentetik üretimde açık sonlu süre seçimi kalmalı; uzun üretim kaynak bütçesiyle
  desteklenmeli. Sınırsız süre veya her efekt için aynı kapasite vaat edilmemeli.
- UI, Python ve IPC aynı core limit/capability/validation sözleşmesini kullanmalı.
  Uzun dosya kabulü ölçülmeden mevcut 30 saniye uyarıları kaldırılmamalı.

## EQ görünümü

Mevcut eğri, renkli bantlar ve ölçülmüş spektrum ana görünüm için yeterli temeldir.
İlk ek: aynı ölçekte üst üste Before/After spektrum; veri son renderdan eskiyse
görünür durum. Before tüm efekt zincirinin öncesidir, yalnız EQ girişi değildir.
Dalga biçimi ve spektrogram Edit/Analyze içinde kalmalı. Faz/grup gecikmesi sonraki
uzman seçenek olabilir; kaynak kimliği ve kayıt akışından önce gelmez.

## Teslim ve kabul

Yeni işlemler UI/script/IPC'de aynı servis üzerinden; hata ve undo anlamları birlikte
teslim edilmeli. Yeni odaklı modüller kullanılmalı, 2000 satır üzeri dosyalara yalnız
gerekli wiring eklenmeli. Birleşik domain dosyaları değiştirilmemeli.

Kullanıcı build'i sonrasında kısa kabul turu:

1. Sentetik taslak adlandır, dinle, kaydet; yeni assetten klip oluştur, tarifini tekrar aç.
2. Imported asseti işle; kaynak bytes değişmesin, yeni asset/recipe doğru olsun.
3. Effects'ten hedef track/zaman seç; başarı ve klip ekleme hatasını ayrı doğrula.
4. Kaynak değiştirirken taslak, eski worker sonucu ve iptal davranışını kontrol et.
5. Klipten işleme/yeni sürüme bağlama: trim/offset/loop/fade/eğriler ve undo/redo.
6. Projeyi kaydet/aç ve portable export: dosya, asset/clip ID ve tarif bağlantısı.
7. Uzun mono/stereo, 30 saniye sınır çevresi, efekt kuyruğu/stretch, blok sınırları,
   bellek/disk bütçesi, iptal ve UI/script/IPC hata eşliği; işitsel karşılaştırma.

Codex build veya uygulama başlatmaz. Canlı probe yalnız kullanıcı build'i sonrasında,
açık Studio'ya dış süreçten scripts/test/rt_ipc.py ile yapılır.

## 2026-10-19 ilk kaynak teslimi

Merkez başlıkta düzenlenebilir ses adı, gerçek asset adıyla kaynak seçici ve render
durumu var; tam dosya yolu tooltip'te. Kaynak seçici diğer assete geçebilir ve açık
bir seçenekle mevcut efekt rack'ini yeni kaynağa taşıyabilir. Varsayılan geçiş aynı
assetin oturum içi UI taslağını geri getirir. Bu taslak tamponları proje verisi değildir;
audio context generation değişiminde (restore/clear ve bazı transport resetleri dahil)
temizlenir. Kalıcı çalışma için asset kaydedilmelidir. Reuse recipe asset adını da açar.

Kaydetme, seçili track'i Designer sayfasından bağımsız başlangıç hedefi olarak alır;
track ve saniye kullanıcı tarafından değiştirilebilir. Use current playhead açık bir
eylemdir. Save as new asset / Save and create clip ayrımı ve başarı özeti gösterilir.
Klip ekleme başarısızsa asset korunur; Retry clip creation only yalnız clip_create
işlemini tekrarlar, yeni dosya veya asset üretmez. Bekleyen hata çözülene kadar yeni
save engellenir. Kaydetme ve klip ekleme mevcut ayrı authoring/undo işlemleridir.

Eski 10 saniye sentetik üretim ve 30 saniye render sınırları kaldırıldı. Yeni ortak
kapasite sözleşmesi: input/output (stretch ve tail dahil) en fazla 600 saniye ve
512 MiB tahmini PCM working set. Tahmin, input ve output için dörder float32 bankası
ayırır; stereo/layer ve stretch kullanılabilir süreyi azaltır. Bu bir process-memory
garantisi değildir; efektlere özgü ve mevcut asset decode/import sınırları ayrıca
geçerlidir (600 saniye / 128 MiB decoded PCM). Tamamen streaming veya sınırsız dosya
işleme teslim edilmiş değildir. Uzun uygun sesler mevcut in-memory core ile işlenir;
bütçe aşımı önceki geçerli tasarım değiştirilmeden reddedilir. UI source range
etkileşimleri aynı süre üst sınırını kullanır; açılan assette tam aralık seçilir
(en fazla 600 s), kapasiteyi aşan aralık için daha kısa seçim gerekir.

Python rt.audio.design_catalogue() / IPC audio.design_catalogue yeni
max_duration_seconds, working_pcm_budget_bytes ve processing_mode=bounded_memory
alanlarını sunar. Mevcut design_render / design_job_start / design_save ve clip_create
işlemleri kullanılır; yeni UI-only DSP veya ayrı IPC işleme yolu yoktur.

Mevcut rt_test_audio_design_tools_ipc.py; 31 saniye synth ve imported-source EQ,
save/reuse PCM eşliği, orijinal dosyanın korunumu, süre/bütçe aşımında tasarımın
korunması ve hatalı track sonrası dosya üretmeden klip ekleme akışıyla genişletildi.
Yeni kodun canlı kabulü kullanıcı build'ini bekler.

Bu ilk teslim mevcut klibi yeni assete yeniden bağlama/remap işlemini, kalıcı taslak
kütüphanesini, büyük dosya streaming'ini veya üst üste Before/After spektrumu
tamamlamaz. Öncelik bölümleri bu kalan işleri tanımlamaya devam eder.

Build sonrası: dar/geniş Designer'da kaynak/ad/range, assetler arası taslak geçişi,
efekt taşıma seçeneği ve Reuse recipe; Effects'ten hedef track/zamanla kaydetme;
31 s mono/stereo ses, bütçe hatası, iptal ve mevcut kısa-ses regresyonu kontrol edilmeli.
Test komutu (Studio açıkken dış terminal):

```powershell
python scripts/test/rt_test_audio_design_tools_ipc.py --output-dir docs/dev
```

## 2026-10-19 kapanis durumu

Bu planin tum maddeleri tamamlanmis sayilmiyor. Bu konusmada Designer kaynak/isim,
asset ve hedef track/zamanla klip kaydi, kapasite sozlesmesi, kompakt baslik ve canli
analiz alternatifleri kaynakta teslim edildi. Kullanici temel canli analizin ve son
build'de grafik Ctrl+wheel hareketinin calistigini bildirdi. Bu geri bildirim tum
kayit/yeniden acma/export, isitsel, uzun ses ve IPC kabul maddelerinin yerine gecmez.

Acik kapsamlar:

- Designer ile sahne gecmisinin odak odakli ana menu/kisayol yonlendirmesi.
  Gecmisler ayri; ana menu/kisayollar halen sahne gecmisini kullaniyor.
- Islenmis yeni asseti mevcut klibe acik ve guvenli baglama/remap; source offset,
  loop/fade ve otomasyon korunumu icin kabul.
- Buyuk dosyada streaming DSP; mevcut 600 s / bellek butcesi sinirsiz isleme degil.
- Ust uste Before/After spektrumu ve kalici taslak kutuphanesi.
- Uctan uca save/reopen/portable export ve ses esligi icin kalan kabul listesi.

Genel fiziksel audio yol haritasi ayridir: alti bantta dogrudan/yansiyan/iletilen
enerjinin ayri gosterimi bu canli mixer analizinde yoktur. Ileri yayilim, fiziksel
uretim ve akustik export/performance kabulunun tam kapanisi ilgili yol haritalarinda
izlenir. Bu durum notu diger ajanin birlesik domain calismasini tamamlandi saymaz.
