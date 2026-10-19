# Substance aile kataloğu ve ortak panel

> **Durum:** AKTİF — 2026-10-19 kullanıcı build'iyle hızlı IPC kabulü PASS; görsel UI/proje kabulü açık.

2026-10-19 kullanıcı build'i sonrası hızlı dış IPC kabulü PASS: altı temel profil,
temsilci alan eşliği, iki seviyeli aile mirası, shared-scope referansları, revert ve
atomik hata davranışı doğrulandı. Testin iki geçici profili silindi. Önce/sonra
mevcut profil isimleri, audio asset/source/clip/track verileri ve akustik ayarları aynı.
Mevcut üç akustik kaynak Vulkan'da ready/clear raporu verdi; her birinde altı specular
ve 2–3 diffuse yol, tam snapshot ve sonlu/negatif olmayan yol verileri gözlendi.
Bu okuma işitsel kalite veya çok sıçramalı fizik doğrulaması değildir. Panel görsel
kabulü ve proje round-trip hâlâ açık. Rapor:
[substance_catalogue_acceptance_20261019.json](substance_catalogue_acceptance_20261019.json).

## Amaç ve kapsam

Ana seçimde çok sayıda test/örnek malzeme yerine temel başlangıç profilleri ve proje
profilleri gösterilir. Aile, fizik çözücüsü veya faz değildir. Mevcut isimler,
alanlar, kayıtlı bağlamalar ve domain fiziği değiştirilmez. Birleşik domain'in
devam eden fizik/termal çalışması bu teslimin kapsamında değildir.

| Temel profil | Aile | Temsilci başlangıç değerleri |
|---|---|---|
| Generic metal | Metaller | Steel |
| Generic mineral | Mineral ve seramikler | Stone |
| Generic polymer | Polimer ve elastomerler | Plastic (PE) |
| Generic fibrous solid | Lifli ve gözenekli | Wood (Oak) |
| Generic liquid | Sıvılar | Water |
| Generic mineral grains | Mineral ve seramikler | Sand |

Bunlar ailelerin evrensel ölçümleri değildir. Temel profiller, mevcut temsilcinin
tam fizik/transport/solver değerlerini başlangıçta kopyalayan salt okunur built-in
profillerdir; kullanıcı aynı derive/set/revert yoluyla projeye özel profil üretir.
Temsilci adı UI ve API'de gösterilir. Temel profiller eski built-in sırasının
sonuna eklenir; eski isimler ve varsayılan seçimler yeniden adlandırılmaz.

Detailed examples / tests seçeneği eski ayrıntılı profilleri açar. Arama tüm
profilleri kapsar; seçili eski profil gizlenmez. Flesh test katmanındadır. Proje
profilleri temel alınan kökün ailesini miras alır; category değiştirmek aileyi
otomatik değiştirmez. category eski liquid/granular/solid/fuel sözleşmesini korur.
Gaz için katı/sıvı kategoriyle sahte bir profil eklenmez; gaz ailesi mevcut gaz
kimliği/ortam sistemiyle ayrı entegrasyon gerektirir. Ahşap/kâğıt gibi profiller
silinmez, ana listede ayrı bir zorunlu katalog oluşturmaz.

## Ortak UI

Domain, flow source, collider ve obje aynı SubstanceEditorUI seçicisini/editörünü
kullanır; simülasyon Substance node seçimi de aynı katalogdan geçer.

Kompakt özet aileyi, salt okunur/paylaşılan profil durumunu, miras alınan profili,
override sayısını ve kaydedilmiş etki referanslarının sayısını gösterir.
Profile details kapalıyken uzun form çizilmez. Açıldığında beş sekme vardır:

- General: kimlik, temsilci, akustik tanımlılık durumu ve etkilenen referanslar.
- Mechanical: sıvı, granular ve DEM malzeme özellikleri.
- Thermal: termal, faz değişimi, nem ve yanma özellikleri.
- Acoustics: yüzey tepkisi ve eşdeğer akışkan ortamı ayrı açılır bölümlerdir.
- Advanced: solver numerics, optik alanlar ve kullanılmayan bölümleri gösterme.

Alan alanı 14 metin satırı yüksekliğinde bağımsız kaydırılır; aktif olmayan sekmeler
uzun panel alanı tüketmez. Read-only built-in için yalnız düzenleme widget'ları
kilitlidir; sekmeler, bölüm açma ve kaydırma kullanılabilir. Domain/flow ve
obje olmayan collider ilk açılışta Mechanical sekmesini seçer.

Collider bir sahne objesine aitse ortak obje malzemesi paneli bu yüzeyi devralır;
aynı anda ikinci bir legacy picker/editör çizilmez. Legacy collider'dan gelen
bağlama Use as object material ile açıkça benimsenebilir. Objesiz collider
mevcut collider bağlamasını ortak editörde kullanır.

Obje başlığında hangi objenin bağlantısının değiştiği görünür. Paylaşılan profil
alanları bütün kullanıcıları etkiler; yalnız mevcut bağlantıyı ayırmak için Derive
ile yeni proje profili oluşturulur. Obje yanma override'ları ayrı kapalı bir bölümde
kalır; bunlar paylaşılan profil alanlarıyla karışmaz.

Tanımsız akustik veri artık Unconfigured olarak etiketlenir; ekranda sıfır görmek
ölçülmüş sıfır tepki anlamına gelmez. Bu teslim yeni ölçülmüş akustik tabloları,
katı ortam dalga modellerini veya kırılmış GPU yayılımını eklemez.

## Ortak servis / Python / IPC

Yeni authoring endpoint'i yoktur: UI, rt.substance ve substance.* mevcut
list/get/derive/set/remove işlemlerini kullanır. Katalog sınıflaması tek yeni
SubstanceCatalogue modülündedir. List satırlarına ve get sonucuna aynı catalogue
nesnesi eklenir:

- family, family_label: kökten çözülen aile.
- tier: foundation / example / test / project.
- root_profile, representative, values_kind=representative_starting_values.
- acoustic_surface_defined, acoustic_medium_defined.

get ayrıca edit_scope=read_only_builtin|shared_project_profile ve
affected_references döndürür. Referans listesi mevcut obje, collider, source,
domain, undo-history kullanımlarını ve türetilmiş profilleri içerir; obje sayısı
değildir. İncelenen servis tüketicilerinin envanteridir, her dış scriptin olası
kullanımını saydığı iddia edilmez. Bu metadata fizik alanı değildir;
substance.set(fields={family: ...}) mevcut unknown-field hatasıyla atomik reddedilir.

```python
profiles = rt.substance.list()
rt.substance.derive("My metal", "Generic metal")
rt.substance.set("My metal", {"density": 7800})
state = rt.substance.get("My metal")
# IPC: substance.list / substance.get / substance.derive / substance.set
```

## Kullanıcı build'i sonrası kabul

1. Boş sahnede dış terminalden:
   `python scripts/test/rt_test_substance_catalogue_ipc.py`.
   Altı temel profil, temsilci alan eşliği, iki seviye miras, shared-scope okuması,
   revert ve atomik hatalar geçmeli. Geçici profiller temizlenmeli.
2. Obje Properties'te temel seçim, arama, ayrıntılı örnekler, derive ve geri alma
   alanları; dar/geniş panelde sekmeler ve sabit yüksekliğin kaydırılması.
3. Collider'da tek obje malzemesi paneli; legacy adopt ve objesiz collider fallback.
4. Domain ve flow source'ta aynı katalog ve Mechanical ilk sekmesi; readonly
   profilde sekmeler erişilebilir, alanlar kilitli olmalı.
5. İki objeyi aynı proje profiline bağla, get/UI referanslarını karşılaştır;
   paylaşılan alan değişikliği ikisini etkilemeli. Derive edip yalnız birine ata;
   sonraki override diğer objeye uygulanmamalı.
6. Eski isimli sahneyi kaydet/aç; malzeme adı, alanlar ve bağlamalar aynı kalmalı.
   Önceki substance regression ve birleşik domain kontrolleri ayrıca korunmalı.

En sinsi hata: aile etiketi doğru görünürken Generic metal'in fizik değerlerinin
Steel'den farklı olması veya collider'da ikinci gizli/bağımsız malzemenin kullanılması.
Görsel benzerlik kabul değildir; get alanları ve object binding readback karşılaştırılır.

Codex build veya uygulama başlatmadı; kullanıcı build'i sonrası hızlı dış IPC kabulü
yukarıdaki kapsamla çalıştırıldı. Diğer kabul adımları ayrı kalır.

Statik doğrulama sırasında genel IPC audit'inin altı mevcut audio metodu için
Python capability mirror'ının C++ güvenlik tablosundan saptığı görüldü. Dört salt
okunur durum sorgusu ile play_all/transport_batch kopyaları eşitlendi; çalışma zamanı
izinleri değiştirilmedi. Descriptor çıktısı üreticiyle tekrar oluşturuldu.

Son statik kontrol: IPC capability/descriptor audit PASS (677 metot); Python AST
ve runtime test kopyası eşliği, VS XML/yeni modül kayıtları, C++ lexical delimiter,
modül boyutu, schema bölüm kapsamı ve diff whitespace kontrolleri PASS. Bunlar
C++ derlemesi, canlı panel kabulü veya fiziksel malzeme kalibrasyonu değildir.
