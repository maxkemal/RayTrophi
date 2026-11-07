# RayTrophi panel tasarımı ve yerleşim standardı

Tarih: 2026-11-03  
Durum: DEVAM EDİYOR — ilk sınırlı kaynak pilotu uygulandı; build/görsel kabul bekleniyor.

İlk dilim: ortak `PropertyLayout` ve Surface Core'un beş temel alanı eklendi.
Tam panel envanteri ve tasarım sözleşmesinin canlı ekranlarla kesinleşmesi henüz
tamamlanmadı. Pilot, Faz A/B/C'nin tamamlandığı anlamına gelmez.
Kapsam ve kullanıcı doğrulaması: [Pilot kontrolleri](PANEL_DESIGN_PILOT_CHECKS.md).

Güncel dilim: materyal editörünün yedi özellik bölümü odaklı modüllere çıkarıldı;
46 alanlık sunum envanteri ve ortak kontrol çizimi eklendi. UV/texture/volume
yayılımı ve eksik animasyon işlemleri henüz tamamlanmadı.
Bkz. [Materyal standardı ve animasyon audit'i](MATERIAL_INSPECTOR_STANDARD.md).

## 1. Amaç ve sınır

Mevcut çok disiplinli çalışma alanlarını ve viewport-first yerleşimi koruyarak
ana panellerin görsel kalitesini, bilgi hiyerarşisini ve yerleşim tutarlılığını artırmak.
ImGui korunur. Yeni çalışma alanı mimarisi veya kalıcı shelf tasarlanmaz.
Gaea, ZBrush, Blender, Houdini ve Substance referansları disiplinin anlamlı sunumu
içindir; bu uygulamaların görünüşleri ayrı temalar olarak kopyalanmaz.

Başarı, yalnız renk/köşe değişimi değildir: aynı tür kontroller farklı panellerde
aynı şekilde yerleşir; hedef, ana işlem ve ileri ayarlar kolay ayırt edilir.
Sadelik adına özellik, hassas sayısal giriş veya hata bilgisi kaybedilmez.

Bağlayıcı mevcut kararlar:

- [Template Hub ürün yönü](TEMPLATE_HUB_UX_ROADMAP.md): viewport, sol bağlam rayı,
  sağ bağlamsal dock ve alt editör yapısı korunur.
- [Layout guardrails](UI_LAYOUT_GUARDRAILS.md): splitter hit-area ve cursor kuralları.
- [Shared Substance UI](SHARED_SUBSTANCE_UI_ROADMAP.md): tek ortak editör, dar dock,
  sekme/arama ve düzenleme yaşam döngüsü.
- [Madde UI tek otorite](MADDE_UI_TEK_OTORITE.md): Domain/Madde/Kaynak ayrımı;
  sunum düzenlemesi veri sahipliğini değiştirmez.

## 2. İlk statik incelemede görülenler

Bu liste tam envanter veya canlı uygulama değerlendirmesi değildir.

| Kaynak | Gözlem | Planın karşılığı |
|---|---|---|
| `source/include/ui_modern.h`, `source/src/UI/ui_modern.cpp` | ThemeManager, ThemeStyle, UIWidgets ve PanelManager mevcut. | Yeni paralel tema sistemi yerine mevcut altyapı genişletilir. |
| `source/src/UI/ui_section_style.cpp` | Ortak bölüm başlıkları mevcut; açık bölüm çevresine de çerçeve çiziliyor. | Başlık ve içerik grubu ayrı roller olarak değerlendirilir; global görünüm birden değiştirilmez. |
| `source/src/UI/scene_ui_materials.cpp` | Ortak bölümler ile sabit 180px alan genişlikleri birlikte kullanılıyor. | Dar/geniş dock için özellik satırı sözleşmesi gerekir. |
| `source/include/Hair/HairUI.h` | Eski Columns, doğrudan CollapsingHeader ve yerel genişlik hesapları birlikte kullanılıyor. | Ortak bileşene geçiş envanterle ve küçük dilimlerle yapılır. |

Kaynak yolları `RayTrophiStudio/` altındadır. Bunlar hata teşhisi değil,
standardizasyon ihtiyacının somut örnekleridir.

## 3. Panel yerleşim sözleşmesi

### Ortak kontrol boyutları ve nötr eylemler — güncel karar

`UI/PanelControls.h` combo/button/heading ölçülerinin ortak girişidir.
Boyutlar font ölçeğinden türetilir; selector ve form kontrolü iki açık yerleşim
rolüdür. Panel/bölüm seçicisi içerik genişliğini doldurur. Özellik enum'u ise
ortak etiket-kontrol satırındaki ayrılmış genişliği kullanır; rastgele 130/180px
veya kalan genişlikten sabit bir sayı çıkarma yöntemi yeni kodda kullanılmaz.
Başlıklar, eylem yüksekliği ve padding aynı merkezden alınır. Yan yana iki
eylem eşit paylaşımlı genişlik kullanır. Uzun eylem grupları dar alanda reflow
edilir; tüm butonlar keyfi sabit genişliğe zorlanmaz.

Delete/Erase/Remove isimleri otomatik olarak kırmızı arka plan gerektirmez.
Normal eylemler nötr yüzey ve ortak hover/active davranışı kullanır. Gerçek hata
ve uyarı göstergeleri semantik renklerini korur; silme onayı/undo davranışları
görsel değişiklik gerekçesiyle değiştirilmez.

İlk entegrasyon: materyal atama ve bölüm seçicisi ortak full-width combo'ya,
özellik combo'ları aynı padding sözleşmesine bağlandı. Surface/Volume oluşturma
butonları eşit genişlikte; materyal keyframe ikonu ortak yüksekliktedir.
Mevcut UIWidgets Primary/Secondary/DangerButton çağrıları ortak nötr button'a
yönlendirilir; davranış ve enabled durumu korunur.

Henüz bütün paneller bu sözleşmeye taşınmadı. Yerel `PushStyleColor` ile
kırmızı boyanan eylemler, raw ImGui butonları ve hardcoded combo ölçüleri panel
envanterinde tespit edilip küçük dilimlerde taşınacak; özellikle birleşik domain
üzerinde çalışan diğer işin dosyaları bu dilimde değiştirilmez.

Kullanıcı doğrulaması: iki materyal seçicisinin genişlik/yüksekliği, dar dock,
özellik combo popup'ları, disabled butonlar ve eski DangerButton çağıranları;
silme/iptal/onay davranışlarının ve gerçek hata renklerinin korunması.

Her panel, içerik gerektiriyorsa şu sırayı izler; boş bölümler çizilmez:

1. **Kimlik ve hedef:** başlık, seçili nesne/sistem, gerekli durum bilgisi.
2. **Yerel gezinme:** yalnız aynı hedefin gerçekten farklı içerik grupları varsa sekme.
3. **Ana içerik:** sık kullanılan ayarlar ve ilgili görsel önizleme.
4. **İleri içerik:** seyrek/uzman ayarları; açık ve bulunabilir giriş altında.
5. **İşlem ve geri bildirim:** ilgili grubun yanında eylem; hata ilgili alana yakın.

Katman listesi, node ayarları ve fizik kaynağı aynı içerik biçimine zorlanmaz.
Ortak olan ölçüler, hizalama, kontrol davranışları ve görsel hiyerarşidir.

### Özellik satırları

- Normal genişlikte etiket solda, kontrol sağda; aynı grupta kontrol başlangıcı ortaktır.
- Etiket sütunu panel genişliğiyle sınırlanır; en uzun etikete göre sınırsız büyümez.
- Dar panelde etiket üstte/kontrol altta düzenine geçilir; metin kırpılarak saklanmaz.
- Sayılar hizalı, birimler görünür; slider yanında hassas klavye girişi korunur.
- XYZ gibi ilişkili değerler ortak etiket altında gruplanır, gerektiğinde satıra yayılır.
- Sıfırlama/override/kalıtım göstergesi için öngörülebilir alan ayrılır.
- Çok satırlı hata/açıklama satır yüksekliğini büyütebilir; komşu alanı örtmez.
- Kontrol türü işlevle seçilir: bool, enum, sürekli değer ve eylem birbirinin yerine geçmez.

### Görsel sözleşme

- Panel, grup, alan, hover, seçim, disabled ve hata için semantik tema rolleri.
- Nötr yüzeyler; vurgu seçim/aktif işlem için, uyarı renkleri gerçek durumlar için.
- Her grubun ayrı rengi, çerçevesi ve başlık dolgusu olması gerekmez.
- En fazla iki görsel grup seviyesi; daha karmaşık içerik yerel gezinmeyle düzenlenir.
- Ortak ikon ailesi/ölçeği; ikon-only eylemlerde tooltip ve anlaşılır erişilebilir ad.
- Önizleme yalnız karar vermeye yardım ettiği yerde: falloff, ramp, brush, material vb.
- Boş durum, yükleniyor, salt okunur ve geçersiz durumlar ortak kalıplarla sunulur.

### Ölçüler ve viewport bütçesi

Pilot için başlangıç adayları, nihai ölçüler değildir: 4/8/12/16 logical-px boşluk
ritmi, 8–12px panel iç boşluğu, 24–28px normal kontrol yüksekliği.
Ölçüler mevcut UI ölçeği/font/DPI ile birlikte hesaplanır; ayrı sabit-piksel sistemi kurulmaz.

- Dar/normal/geniş dock örnekleri: 240/320/440 logical px; gerçek minimumlar envanterde doğrulanır.
- Okunabilirliği korumak için font küçültülmez; içerik yeniden akar.
- Ana eylem ve hedef bilgisi yatay kaydırmaya bağımlı olmaz.
- Aynı ekran boyutu, UI ölçeği ve açık panel setinde viewport alanı başlangıçtan küçülmez.
- Büyük dokunma hedefleri/erişilebilirlik ihtiyacı görsel kompaktlığa feda edilmez.

## 4. Uygulama aşamaları ve çıkış koşulları

### Faz A — Envanter ve gerçek ekran temeli

Properties, Material, Edit Mesh, Sculpt, Paint, Terrain, Hair, Water ve Physics için
giriş dosyası/fonksiyonu, mevcut ortak bileşenler, yerel stiller, sabit genişlikler,
veri sahibi ve script/IPC yolları kaydedilir. Her kontrol için mevcut işlev listesi tutulur.
Temsilci dar/geniş ekranlar kullanıcı uygulamayı açıkça açmamızı istediğinde incelenir;
o zamana kadar kaynak incelemesi ile canlı gözlem birbirinden ayrı işaretlenir.

Çıktı: panel envanteri, tekrar eden sorunlar, başlangıç ölçüleri ve pilot kapsamı.
Çıkış: ilk pilotun gerçek kontrol listesi ve mevcut davranışları belli.

### Faz B — Tasarım sözleşmesi ve ortak bileşen kataloğu

Bu belgedeki kurallar gerçek panel üzerinde önce/sonra taslağıyla kesinleştirilir.
Tema rolleri, boşluklar, panel başlığı, özellik satırı, grup, eylem satırı, durum/hata,
katman/öğe listesi ve ileri ayar kalıpları örneklenir.
Her biri için dar/geniş ve normal/hover/active/disabled/error durumları tanımlanır.

Çıkış: yalnız güzel tek ekran değil, pilotun tüm durumlarını taşıyan sözleşme.

### Faz C — Sınırlı ortak altyapı ve pilot

Önerilen pilot: Material Inspector içindeki **Surface Core** ve komşu bir ileri ayar grubu.
Kesin sınır Faz A'da belirlenir. Bütün material/volume/UV editörünü ilk dilime dahil etme.
Bu örnek sayısal değer, renk/texture, koşullu kontrol ve ileri ayarı birlikte sınar.

Mevcut UIWidgets/ThemeManager yeniden kullanılır. Gerekli yeni davranış odaklı
`PanelLayout` / `PropertyRow` gibi küçük `.h/.cpp` modüllerinde geliştirilir;
isimler taslaktır. 2000 satırı aşan dosyalara yalnız minimum entegrasyon eklenir;
pilot gövdesi gerekiyorsa yeni modüle davranış korunarak taşınır.
Global widget görünümü bütün panelleri aynı anda değiştirecek biçimde uygulanmaz.

Çıkış: kullanıcı build ve görsel değerlendirmesiyle kabul edilmiş pilot;
kontrol envanteriyle işlev kaybı olmadığı doğrulanmış.

### Faz D — Kontrollü yayılım

Önerilen sıra, envanter maliyetine göre güncellenebilir:

1. Properties ortak başlığı ve sık kullanılan transform/nesne satırları.
2. Material Inspector'ın kalan grupları.
3. Sculpt ve Paint araç ayarları/katman listeleri.
4. Terrain parametre panelleri; graph çalışma mantığı korunur.
5. Hair üretim/groom panelleri.
6. Water ve Physics; mevcut Domain/Madde/Kaynak ve ortak Substance kararları korunur.
7. Edit Mesh ve kalan editör panelleri; alan bazlı istisnalar belgelenir.

Her dilimde aynı döngü: envanter → düzenleme → statik kontrol → kullanıcı build/canlı
kontrol → kabul. Bir alan kabul edilmeden toplu mekanik dönüşüm yapılmaz.

### Faz E — Kalıcı bakım

Yeni panel kontrol listesi geliştirme rehberine bağlanır. Yerel stil/genişlik
istisnaları gerekçesiyle belgelenir. Statik taramalar aday sorun gösterir;
özel viewport/graph kontrollerini kör biçimde yasaklamaz.
Eski helper ancak tüm çağıranları taşındığında kaldırılır.

## 5. Davranış, API ve durum güvenliği

- Görsel yerleşim değişiklikleri mevcut core/service işlemlerini çağırır.
- Yeni kullanıcı işlemi veya kalıcı authoring/UI tercihi eklenirse kapsam açıkça yazılır;
  ortak service, scripting ve IPC yüzeyleri, hata semantiği ve belgeleri birlikte teslim edilir.
- Mevcut script/IPC işlemleri ve hata kodları sunum gerekçesiyle değiştirilmez.
- Slider drag, commit-on-release, undo gruplaması ve cache invalidation korunur.
- Stable ImGui ID'leri hedef/caller/alan kimliğine dayanır; görünen etiket kimlik değildir.
- Seçim/panel değişirken yarım düzenleme yanlış hedefe yazılmaz.
- Sahne verisi UI-local kopyalarda otorite haline gelmez; geometri flat TriangleMesh/DNA yolundadır.

## 6. Her dilim için kabul ve doğrulama

- Önce/sonra kontrol listesinde tüm işlemler mevcut ve bulunabilir.
- 240/320/440 logical-px dock; desteklenen UI ölçeği/DPI ve uzun etiketlerde taşma yok.
- Mevcut temalar/custom theme ve alfa ayarlarında metin, seçim ve hata okunabilir.
- Klavye girişi, Tab odağı, numeric entry, reset ve disabled açıklamaları çalışıyor.
- Sekme/nesne/panel değişimi, daraltıp açma ve yeniden seçimde değer/undo korunuyor.
- Splitter, scroll, popup ve viewport input capture davranışı korunuyor.
- Aynı açık panel koşullarında viewport küçülmüyor; gereksiz yatay scroll oluşmuyor.
- Mevcut core işlemleri kullanıcı tarafından değiştirildiğinde script/IPC readback aynı sonucu veriyor.

Codex: build ve shader derlemesi çalıştırmaz, uygulamayı açık talep olmadan başlatmaz.
Kaynak/diff, ID ve push/pop dengesi, yerleşim hesapları ve mevcut non-build
kontroller incelenir. Derleme ve canlı kontrol listesi kullanıcıya verilir.
Kullanıcı uygulamayı açarak canlı doğrulama istediğinde named-pipe probları ayrı
process'ten `scripts/test/rt_ipc.py` ile çalıştırılır; sandbox error 5 için escalation kullanılır.
Geçici doğrulama ekran görüntüleri yalnız ignored `docs/dev/.captures/` altında tutulur
ve iş bitiminde silinir; depoya eklenmez.

## 7. İlk sonraki iş

Faz A envanterini hazırlamak ve Surface Core için mevcut gerçek kontrolleri kullanan
dar/geniş önce/sonra taslağı üretmek. Bu belge uygulama veya runtime doğrulaması tamamlandı
anlamına gelmez; her faz yalnız teslim edilen kanıtla ilerletilir.

## 8. Güncel yön — tek modern kontrol ve domain inceleme overlay'i

Kullanıcı kararı: Retro LCD kaldırılır; ayrı slider stili/paleti tutulmaz.
`DrawSmartFloat` mevcut çağıranlar için ince uyumluluk girişi olarak kalır ve
ortak `PropertyLayout::slider` üzerinden çizilir. Eski segment argümanı yalnız
kaynak uyumluluğu içindir. Yeni kontrollerde kullanılmaz.
Bu, tüm panellerin dönüşümü tamamlandı anlamına gelmez; doğrudan ImGui çağrıları
ve alana özel kontrol düzenleri envanterle taşınmaya devam eder.

### Domain gizmosunda inceleme/debug ikonu — PLANLANDI, uygulanmadı

Seçili domain gizmosunun yanında küçük, ekran yönüne bakan bir inceleme ikonu
gösterilir. İkon, görünümün yoğunluğunu kullanıcı istediğinde açar; sürekli
kutular ve çizgilerle viewport'u doldurmaz. Birleşik domain geliştirmesiyle
entegrasyon, o veri sözleşmesi kesinleştiğinde ayrı dilim olarak yapılır.

Önerilen inceleme katmanları:

| Katman | Gösterim | Düzenlenebilir görünüm ayarları |
|---|---|---|
| Hücreler | Domain hücre ızgarası ve aktif/dolu hücreler | Kesit düzlemi, seyrekleştirme, görünür bölge |
| Hız | Oklar veya mevcutsa akış çizgileri | Ok ölçeği, örnekleme yoğunluğu |
| Kuvvet | Gerçekte mevcut kuvvet alanları, ayrı adlandırılmış | Alan seçimi, ölçek, yoğunluk |
| Sıcaklık | Birimli renk skalasıyla kesit/hacim örnekleri | Aralık, otomatik/manüel ölçek |
| Hücre/örnek inceleme | Seçilen örneğin koordinatı, alan değerleri ve birimleri | İncelenen örneği seçme |

Hız ve kuvvet fiziksel olarak ayrı alanlardır. Sıcaklık skaler bir alandır;
vektör gösterimi ancak sıcaklık gradyanı/ısı akısı gerçekten sağlanıyorsa kendi
adı ve birimiyle sunulur. Desteklenmeyen alanlar üretilmiş veriyle doldurulmaz;
neden kullanılamadığı gösterilir.

Etkileşim sözleşmesi:

- Gizmo seçimi ikonu açığa çıkarır; ikon tıklaması inceleme modunu açar/kapatır.
- Küçük katman seçicisi görünümü yönetir. Simülasyon parametresi değiştirmez.
- Ölçek/yoğunluk/aralık değerleri değiştirilebilir; fizik sonucu read-only ve
  düzenlenebilir görselleştirme ayarından görsel olarak ayrıdır.
- Aktif katmanda küçük legend, birim, örnekleme ve snapshot/frame bilgisi görünür.
- Kesit/ROI ve çizim bütçesiyle büyük domain'de tüm hücreler çizilmez; GPU readback
  her UI frame'inde zorlanmaz. Son geçerli snapshot kullanılıyorsa zamanı gösterilir.
- Kamera hareketinde ikon/etiket okunabilir boyutta kalır; çizgilerin occlusion ve
  isteğe bağlı x-ray davranışı açıkça ayrılır.
- Normal nesne seçimi, brush ve gizmo drag olaylarını inceleme kontrolleri çalmaz.
- Seçim değişimi, domain silinmesi ve scene reset, eski hedefe erişimi sonlandırır.
- Görünüm state'i stable domain kimliğine bağlıdır; solver/scene otoritesi değildir.

Mimari ve teslim koşulu:

Ortak domain tanılama servisi, renderer'dan bağımsız alan/capability ve snapshot
sözleşmesi sağlar. UI, scripting ve IPC aynı servisten alan listesini, görünüm
ayarlarını ve seçili örneği okur/değiştirir. Geçersiz domain/alan, hazır olmayan
snapshot ve bütçe sınırı hataları tutarlıdır. Yeni public işlemler, ilgili
script/IPC yüzeyleri, testleri ve belgeleriyle birlikte teslim edilir.

Kabul: küçük/büyük domain, desteklenmeyen alan, boş snapshot, pause/step,
seçim/silme, kamera ve input capture; alan birimleri ve snapshot tutarlılığı;
UI/script/IPC ayar paritesi ve görselleştirmenin solver sonucunu değiştirmemesi.
