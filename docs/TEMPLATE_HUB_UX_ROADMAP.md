# RayTrophi Template Hub ve Yönlendirilmiş Sahne UX Yol Haritası

**Durum:** Kanonik ürün/UX yönü  
**Kapsam:** Açılış deneyimi, yeni proje akışı, template paketleri, yönlendirilmiş hazır sahneler ve bunların UI yerleşimi  
**Temel ilke:** RayTrophi viewport ağırlıklı, sade ve kendine özgü kalır. Template Hub bu yapının önüne eklenen ağır bir proje yöneticisi değil, kullanıcıyı doğru üretim bağlamına hızla yerleştiren giriş kapısıdır.

## İlerleme durumu

Son güncelleme: **2026-08-14**

| Faz | Durum | Tamamlanan çıktı |
|---|---|---|
| Faz 0 — Karar ve statik sözleşme | **TAMAMLANDI** | Manifest v1 JSON Schema, UI-state allow-list, keşif/kimlik kuralları, Start/Learn kararı ve ilk altı template içerik sözleşmesi yazıldı. Bkz. `docs/template_hub/`. |
| Faz 1 — Template registry ve doğrulama | **CONFIRMED** | Yeni `TemplateRegistry` ve manifest v1 validator; deterministik discovery/sıralama; duplicate ID, unknown field, enum, sürüm, güvenli path ve zorunlu asset kontrolleri; `rt.templates` script yüzeyi; `templates.*` IPC yüzeyi; capability sınıflandırması; offline contract, embedded script ve IPC testleri; API dokümantasyonu. **2026-08-14 doğrulama:** embedded `rt 0.5.0` testi PASS; statik IPC capability audit PASS; boş registry yolları ve ardından gerçek built-in paketle başarılı `refresh/list/get/validate` yolları canlı olarak doğrulandı. |
| Faz 1B — İlk built-in runtime paketi | **CONFIRMED** | `raytrophi.start.empty` paketi; English-only manifest/recipe/guidance; hafif SVG preview; zorunlu dosya contract testi; build sonrası `assets/templates` dağıtım hedefi; script ve IPC pozitif `prepare -> ready` kontrolleri. **2026-08-14 doğrulama:** registry 1 paket keşfetti; manifest valid; `templates.get`, `templates.validate` ve gerçek recipe için `templates.prepare -> ready` PASS; canlı named-pipe suite **167 PASS / 0 FAIL**. |
| Faz 2A — Mutasyonsuz preflight/load plan | **CONFIRMED** | `TemplateLoader::prepare`; project/recipe JSON preflight; gerekli v3 geometry binary ve opsiyonel auxiliary doğrulaması; açık `reject|discard` conflict policy; `rt.templates.prepare` ve `templates.prepare` IPC paritesi; makine-okunur durum/hata kodları. Aktif scene/project/UI state değiştirilmez. **2026-08-14 doğrulama:** embedded script testi PASS; missing-template ve invalid-policy yolları PASS; built-in Empty recipe için gerçek `ready` planı PASS. Son canlı named-pipe suite **167 PASS / 0 FAIL**. |
| Faz 2B.1 — Güvenli recipe commit ve UI-state adapter | **CONFIRMED** | Preflight sonrası yalnız hatasız commit sınırı bulunan Empty recipe açılır; `reject` conflict aktif sahneyi korur, `discard` açık kullanıcı niyetidir. Yeni `TemplateSession`, semantik `TemplateUiStateAdapter`, ortak `rtapi::openTemplate`, `rt.templates.open` ve `templates.open` yüzeyleri eklendi. Project template ve fallible recipe commit'leri staging olmadan reddedilir. **2026-08-14 doğrulama:** embedded script PASS; canlı named-pipe suite **172 PASS / 0 FAIL**. Reject mutasyonsuz sahne koruması, Empty commit, viewport kamerası, Solid/Rendered geçişi ve commit sonrası içerik ekleme manuel olarak doğrulandı; sahne kararlı kaldı. Boş Vulkan RT sahnesinin geometri eklenene kadar ürettiği gereksiz maliyet ayrı, düşük öncelikli backend performans borcu olarak kaydedildi ve template akışını bloke etmiyor. |
| Faz 2B.2a — Staged General Scene recipe | **CONFIRMED** (2026-08-15 kullanıcı doğrulaması) | `TemplateRecipeStager` recipe JSON ve tüm CPU içeriğini aktif sahneden bağımsız hazırlar; General Scene için kanonik flat `TriangleMesh` cube, kamera ve key light ancak staging başarıyla tamamlandıktan sonra commit edilir. Eski facade tabanlı default-scene üreticisi aktif açılış yollarında kullanılmaz; uygulama başlangıcı ve New Project artık General Scene template servisine yönlenir. English-only runtime paket, preview, guidance, script ve IPC pozitif yol testleri eklendi. İlk testte flat cube CPU BVH ve sol hierarchy cache'i yalnız deferred bırakıldığı için ilk seçim eksik kaldı; commit artık CPU BVH'yi ve `direct_mesh_nodes` cache'ini hemen kurup manifest `frame_target` nesnesini seçiyor. Son doğrulamada Open/Import yapılana dek viewport sol tık picking'in template ve Add nesnelerini yakalamadığı görüldü: ortak sahne mutasyon yolu artık devam eden eski async CPU-BVH snapshot'ını mutasyon anında nesil artırarak geçersiz kılıyor; hierarchy seçimi de temsilci facade yerine kanonik flat mesh kimliğini taşıyor. Sol tık nesne seçimi, hassas GPU/CPU surface hit bulunamadığında sağ box-selection'ın kanonik cache + projected-bounds çözümünü iki piksellik point-pick olarak yeniden kullanıyor; böylece template ve Add flat nesneleri bir Open/Import yenilemesini beklemeden seçilebiliyor. Template küpünün ilk sürümünde materyal ID 0 varsayılmış ve Add küpünden farklı bir UV düzeni kullanılmıştı; staged materyal artık commit sonrasında `MaterialManager` içine kaydedilip gerçek ID bütün corner'lara yazılıyor ve flat küp Add > Cube ile aynı 36-corner 4x3 UV cross atlasını kullanıyor. General Scene manifesti yol haritasına uygun olarak timeline'ı açık başlatıyor; kamera da 2x2x2 Add-küp ölçeğine göre geri alındı. Embedded testteki IPC-biçimi `rt.call` kullanımı gerçek Python yüzeyi `rt.material.list()` ile değiştirildi. Paint/sculpt gibi gerçek yüzey verisi isteyen araçların ilk fırça hiti için template commit sırası Open/Import ile eşitlendi: flat-mesh cache ve transformed vertices önce senkronize ediliyor, CPU BVH daha sonra kuruluyor ve picking state hazır işaretleniyor. Import finalization artık `rebuild=false` loader sonucunu yalnız lazy kuyruğa bırakmıyor: ana thread'de cache/transform hazırlığından sonra final flat sahnenin CPU BVH'sini senkron yayınlıyor ve eski snapshot'ları geçersiz kılıyor. Genel sahne mutasyon yolu ayrıca CPU moda erken geçilirse mevcut güvenli senkron rebuild kapısını kuruyor. |
| Faz 2B.2b-i — Dürüst reddetme (kanonik preset listesi + project preflight hizalaması) | **DERLENDİ (2026-08-15) — RUNTIME DOĞRULAMASI BEKLİYOR** (`docs/NEXT_BUILD_CHECKS.md` 2–6; ★5. madde project preflight'ı depoda project template olmadığı için hiç koşmuyor) | Preset desteği tek kanonik kaynaktan sorulur; preflight ile commit'in ayrışması yapısal olarak imkânsız. Project preflight'ı `openProject`'in gerçek kabul şartlarıyla hizalandı: `format_version` **tam** `"3.0"`, `has_geometry`/binary üçlüsü ve `.bin` header magic (`RTP3`–`RTP8`) mutasyondan önce doğrulanıyor. Project template'i hâlâ açılmıyor — açık hata veriyor, sahne silmiyor. |
| Faz 2B.2b-ii — Gerçek project staging | **ERTELENDİ** (bilinçli) | `newProject` sekiz singleton'ı temizlediği için geçici `SceneData`'ya yükleyip takas etmek mümkün değil; gerçek staging bu singleton'ların instance'lanabilir olmasını gerektirir. §4'teki altı template de recipe olduğundan hiçbir kullanıcı özelliğini bloke etmiyor. İlk project-backed template fiilen gerekene kadar başlatılmaz. |
| Faz 3 — Sade Template Hub | Bekliyor | 2B.2b-ii'yi beklemiyor; registry/loader/IPC hazır. |
| Faz 4 — Üretim template'leri | Bekliyor | 2B.2b-ii'yi beklemiyor; altı template de recipe. |
| Faz 5 — Recent, recovery ve tercihler | Bekliyor | — |
| Faz 6 — Kullanıcı template'leri | Bekliyor | — |

**2B.2a tanı notu (2026-08-14):** CPU render ile paint/sculpt yüzey hitinin flat `TriangleMesh` sahnelerde birlikte kaybolduğu doğrulandı. Sorunu çözmeyen Import ana-thread senkron rebuild, genel `g_cpu_sync_pending` zorlaması ve ek snapshot-invalidasyon servisi maliyet/risk oluşturmaması için geri alındı. Embree build aşamasına yalnız build başına bir kez çalışan vertex/index/AABB audit kaydı eklendi; kök neden bu ölçüm üzerinden kapatılacak.

Tamamlanma yalnız kod veya belge gerçekten teslim edildiğinde işaretlenir. Kısmi çalışmalar `DEVAM EDİYOR`, doğrulanmamış uygulamalar `UYGULANDI — KULLANICI DOĞRULAMASI BEKLİYOR` olarak kaydedilir; planlanan iş tamamlanmış sayılmaz.

## 1. Ürün kararı

RayTrophi; Maya, 3ds Max veya Houdini tarzı geniş, kalıcı shelf alanlarını kopyalamaz. Basit bir ışık, kamera veya geometri eklemek için viewport'tan büyük alan çalan ikon tablaları oluşturulmaz.

Mevcut UX omurgası korunur:

- Viewport ana ve kalıcı çalışma yüzeyidir.
- Sol ince ray üretim bağlamlarını seçer.
- Sol inspector seçilen bağlamın içeriğini sunar.
- Sağ bağlamsal dock aktif Paint, Sculpt veya Hair aracını taşır; dar durumda hızlı ikon, geniş durumda ad, açıklama ve ayrıntılı ayar gösterir.
- Alt ince bar özel editörleri açar ve gerektiğinde dock/float davranışı sunar.
- Global ve seyrek eylemler menü, `Add` girişi ve ileride komut araması üzerinden erişilir.

Template çalışması bu omurgayı değiştirmemeli; doğru sahneyi, doğru bağlamı ve doğru panel düzenini birlikte hazırlamalıdır.

## 2. UX hedefi

Kullanıcı açılışta boş ve yönsüz bir viewport'a bırakılmamalıdır. En fazla birkaç karar vererek gerçek üretime hazır bir sahneye ulaşmalıdır:

1. Başlangıç türünü seçer.
2. İsterse birkaç temel proje ayarını değiştirir.
3. Template sahneyi ve uygun UI durumunu yükler.
4. Viewport doğrudan çalışılabilir hâlde açılır.

Başarı ölçütü, güzel bir karşılama ekranı değil; ilk anlamlı üretim eylemine kadar geçen sürenin ve yanlış başlangıç kurulumlarının azalmasıdır.

## 3. Template Hub kapsamı

Template Hub uygulama içi, sade bir tam yüzeydir. Ayrı bir işletim sistemi penceresi veya sürekli açık kalan ana panel değildir. Proje açıldığında tamamen kaybolur ve viewport alanını geri verir.

Birincil eylemler:

- New Project
- Open Project
- Recent Projects
- Recover Autosave
- İsteğe bağlı: Restore Previous Session

Başlangıç tercihi:

- Always Show Template Hub
- Open Last Project
- Start Empty
- Restore Previous Session

`File > New Project` de aynı Template Hub akışını açmalıdır; açılış ve yeni proje için iki ayrı UX geliştirilmemelidir.

## 4. İlk kanonik template seti

İlk sürüm çok sayıda yüzeysel template yerine az sayıda üretim kalitesinde template sunar:

| Template | Amaç | Açılış bağlamı | İçerik kaynağı | Paket biçimi |
|---|---|---|---|---|
| Empty | Tamamen temiz proje | Scene/Layout | prosedürel kamera | recipe |
| General Scene | Genel modelleme ve sahne düzenleme | Scene/Modeling | prosedürel küp + key light + kamera | recipe |
| Product Lookdev | Ürün, materyal ve stüdyo ışığı | Material veya Scene + Material editörü | prosedürel primitive + area light rig + nötr zemin | recipe |
| Portrait & Groom | Karakter portresi ve hair grooming | Hair & Fur + Hair dock | prosedürel UV sphere scalp + tek groom layer | recipe |
| Character Paint | Çok kanallı karakter boyama | Paint + Paint dock | prosedürel UV sphere hedef + material slot + paint layer | recipe |
| Terrain Environment | Terrain, biome ve çevre üretimi | Terrain + Terrain graph | prosedürel terrain + sun/sky + prosedürel low-poly ağaç | recipe |

**★ Built-in template'ler binary asset taşımaz.** Altısı da prosedürel olarak
üretilir; `assets.required` her birinde boştur. Bunun üç gerekçesi var ve üçü
bağımsızdır:

1. **Ürün gerekçesi (kanonik):** Template'in verdiği şey model değil, **genel
   kullanım mantığı ve doğru UI düzenidir**. Kullanıcı kendi mesh'ini getirir.
2. **Performans:** §11 güvenli başlangıç yükü ister. Depodaki gerçek ağaçlar
   parça başına ~13 MB ve ~68.500 üçgen; birkaç yüz instance scatter edilince
   bir *başlangıç* sahnesi kabul edilemez hâle gelir.
3. **Lisans:** `assets/vegetation` PlantCatalog türevidir ve **satılamaz**
   (kalıcı kısıt). Zorunlu asset yapılırsa template ileride ücretli bir dağıtımda
   kırılır. Bkz. `RayTrophiStudio/assets/THIRD_PARTY_ASSETS.md`.

**Gerçek varlıklar opsiyonel zenginleştirmedir.** Terrain Environment,
`assets.optional` üzerinden `vegetation/trees/*`'ı arar: varsa scatter kaynağı
olarak bağlar, yoksa prosedürel ağaca düşer ve template yine çalışır (§9).

★ **İş hedefi primitive ≠ dekoratif gösteri.** UV sphere burada silinecek demo
içeriği değil, üzerinde çalışılan **yüzeyin kendisidir** — hair bir yüzey
olmadan, paint bir hedef mesh olmadan var olamaz. Kullanıcı onu ya groom'lar ya
kendi mesh'iyle değiştirir; §12.7'nin yasakladığı şey bu değil, sahneyi
süsleyen ve işle ilgisi olmayan içeriktir.

Animation ve VFX/Simulation template'leri ilk sistem sağlamlaştıktan sonra eklenir. Her template için kalite, doğruluk ve bakım maliyeti template sayısından daha önemlidir.

## 5. Template bir sahne dosyasından fazlasıdır

Kanonik template paketi aşağıdaki bileşenleri taşır:

```text
Template Package
├── manifest.json
├── preview.webp/png
├── scene.rtp veya scene recipe
├── ui_state.json
├── optional assets/
└── optional guidance.json
```

Manifest en az şunları tanımlar:

- Kalıcı ve benzersiz template kimliği
- Görünen ad ve kısa açıklama
- Template şema sürümü
- RayTrophi minimum uyumlu sürümü
- Kategori ve sıralama
- Preview yolu
- Scene/recipe yolu
- Başlangıç properties sekmesi
- Başlangıç alt editörü
- Sağ bağlamsal dock durumu ve önerilen genişliği
- Viewport shading/camera başlangıcı
- Render backend için tercih, zorunluluk değil
- Tahmini VRAM/performans sınıfı
- Gerekli ve isteğe bağlı asset listesi
- `Start` veya `Learn` türü

Template, kullanıcıya ait genel tema veya erişilebilirlik tercihlerini zorla değiştirmemelidir. Yalnız üretim düzeniyle ilgili izinli UI state alanlarını uygular.

## 6. Start ve Learn ayrımı

Üretim template'i ile öğretici/demo sahnesi aynı şey değildir:

- **Start:** Temiz, hafif ve doğrudan üretime uygun başlangıçtır. Kullanıcının silmek zorunda kalacağı gösteri içeriği içermez.
- **Learn:** Açıklamalı örnekler, hazır node graph'ları ve öğretici içerik taşıyabilir.

Örnek:

- `Terrain Environment`: temiz terrain çalışma başlangıcı
- `Learn Terrain`: biome, river, snow ve foliage örnekleri

Template Hub ilk görünümde `Start` içeriklerini öne çıkarmalı, `Learn` içeriklerini ayrı bir filtre veya bölümde sunmalıdır.

## 7. Yönlendirilmiş sahne sözleşmesi

Yönlendirilmiş sahne, sadece doğru nesneleri değil doğru çalışma durumunu da hazırlar.

### Character Paint

- Boyanabilir, açıkça isimlendirilmiş hedef mesh
- Doğrulanmış material slot ve texture set yapısı
- Başlangıç paint layer'ları
- Paint properties sekmesi aktif
- Sağ Paint dock görünür
- Material Preview veya uygun viewport modu
- Portreyi okunur kılan hafif ışık rig'i
- İlk eylemi anlatan, kapatılabilir kısa rehber

### Portrait & Groom

- Uygun scalp/karakter hedefi
- Örnek ama temiz groom layer yapısı
- Hair & Fur sekmesi aktif
- Sağ Hair dock görünür
- Portre kamerası ve ışık rig'i
- Groom üretimi ile material/shading ayrımı anlaşılır hâlde

### Terrain Environment

- Terrain nesnesi ve sun/sky başlangıcı
- Küçük ve okunabilir başlangıç graph'ı
- Terrain properties sekmesi aktif
- Alt `Terrain` graph editörü açık
- Sahne ölçeğine uygun navigation scale
- Performans sınıfına uygun güvenli başlangıç çözünürlüğü

### Product Lookdev

- Nötr stüdyo zemini ve ışık rig'i
- Kamera ve ürün yerleştirme hedefi
- Material editörüne hızlı geçiş
- Renk yönetimi ve render kalitesi için güvenli varsayılanlar

## 8. UI dili ve sade görünüm kuralları

- Template Hub kartları görsel, isim ve tek satır açıklamayla başlamalıdır.
- İleri ayarlar varsayılan olarak kapalı olmalıdır.
- Kart seçimi kullanıcıyı uzun bir forma götürmemelidir.
- Ayrı ışık/geometri ikonlarından oluşan büyük bir shelf eklenmemelidir.
- Sol rayın grupları kalıcı uzun başlıklarla genişletilmemeli; boşluk, ince ayraç ve tooltip diliyle okunur kılınmalıdır.
- Alt bardaki belirsiz `Graph` etiketi, alanı boğmadan `Terrain` olarak gösterilebilir; tooltip tam adı `Terrain Graph` ve kısa açıklamayı taşımalıdır.
- Template tarafından açılan rehber kapatılabilir olmalı ve kullanıcı tercihini hatırlamalıdır.
- Template yüklendikten sonra Hub viewport üzerinde kalmamalıdır.

## 9. Template yükleme ilkeleri

Template yükleme normal proje/sahne yollarını yeniden kullanmalıdır. Aynı veriyi farklı biçimde yükleyen ikinci bir gizli scene loader oluşturulmamalıdır.

- Scene tarafı mevcut Project Manager ve SceneSerializer sözleşmeleriyle uyumlu olmalıdır.
- Kanonik geometri flat `TriangleMesh` / DNA SoA yoludur; template sistemi per-face `Triangle` facade koleksiyonlarını authoritative scene geometry olarak üretmemeli veya saklamamalıdır.
- UI state scene verisinden ayrı ve allow-list tabanlı uygulanmalıdır.
- Eksik isteğe bağlı asset template'i tamamen bozmamalı; açıklanabilir fallback sunmalıdır.
- Eksik zorunlu asset kullanıcıya açık hata vermeli ve yarım yüklenmiş projeyi sessizce bırakmamalıdır.
- Template kaynakları salt okunur kabul edilmeli; proje oluşturulurken kullanıcı proje alanına güvenli kopya veya referans politikası uygulanmalıdır.
- Template yükleme tek bir işlem gibi davranmalıdır: **başarısızlıkta önceki geçerli sahne korunmalıdır.** "Ya da güvenli boş duruma dönülür" kaçamağı kaldırılmıştır; sahneyi silip sonra hata vermek bu sözleşmeyi karşılamaz, çünkü kullanıcı çalışmasını kaybeder. Bu garantiyi veremeyen bir yol, garanti veriyormuş gibi davranmak yerine **mutasyondan önce açık bir hata ile reddetmelidir** (bkz. Faz 2B.2b).
- Şema migrasyonu versioned ve test edilebilir olmalıdır.

## 10. Uygulama aşamaları

### Faz 0 — Karar ve statik sözleşme

- Template manifest şemasını yaz.
- UI state allow-list'ini tanımla.
- Template klasör keşfi ve kimlik kurallarını belirle.
- Start/Learn ayrımını sabitle.
- İlk altı template'in içerik sözleşmelerini onayla.

**Çıkış kriteri:** Kod yazmadan önce örnek manifestler ve her template için kabul kriterleri incelenebilir durumda.

### Faz 1 — Template registry ve doğrulama

- Built-in template klasörlerini tara.
- Manifestleri parse et ve doğrula.
- Uyumlu olmayan/bozuk template'leri açıklanabilir hata ile işaretle.
- Preview ve metadata sorgu API'si sağla.
- Registry template sıralamasını deterministik tutsun.

**Çıkış kriteri:** UI olmadan template listesi güvenilir biçimde üretilebiliyor ve bozuk paketler sahne yüklemeden teşhis edilebiliyor.

### Faz 2 — Transactional template loader

Bu faz uygulanırken **dörde bölündü**. Bölünmenin sebebi tasarım değişikliği
değil, bir mimari gerçekle çarpışmasıdır: `ProjectManager::openProject`
yıkımı (`newProject`, [ProjectManager.cpp:1854]) geç yükleme hatalarından
**önce** yapıyor, dolayısıyla project yolu için "başarısızlıkta önceki sahne
korunur" garantisi mevcut mimaride verilemiyor.

- **2A — mutasyonsuz preflight.** `TemplateLoader::prepare`; hiçbir aktif
  sahne/proje/UI durumu değişmeden plan ve makine-okunur hata kodu üret.
- **2B.1 — güvenli recipe commit.** Yalnız hatasız commit sınırı bulunan
  recipe'ler açılır; `reject` aktif sahneyi korur, `discard` açık kullanıcı
  niyetidir. `TemplateSession` + `TemplateUiStateAdapter`.
- **2B.2a — staged General Scene.** Bütün fallible CPU içeriği aktif sahne
  mutasyon sınırını geçmeden önce kurulur (`TemplateRecipeStager`).
- **2B.2b-i — dürüst reddetme.** Preset desteği **tek kanonik listeden**
  sorulur (preflight ile commit'in ayrışması imkânsız hâle gelir) ve project
  preflight'ı `openProject`'in gerçek kabul şartlarıyla hizalanır. Bu faz
  project template'i **açmaz**; sahne silen sessiz başarısızlığı kapatır.
- **2B.2b-ii — gerçek project staging.** Ertelenmiştir. `newProject` sekiz
  singleton'ı (`MaterialManager`, `TerrainManager`, `InstanceManager`,
  `VDBVolumeManager`, `RiverManager`, `WaterManager`, `HairSystem`,
  `g_project`) temizlediği için geçici bir `SceneData`'ya yükleyip takas etmek
  mümkün değil; gerçek staging bu singleton'ların instance'lanabilir hale
  gelmesini gerektirir. İlk project-backed template fiilen gerekene kadar
  başlatılmaz.

**Çıkış kriteri (2A–2B.2b-i):** Bütün recipe template'leri aynı loader
üzerinden güvenli açılıyor; hiçbir başarısızlık yolu aktif sahneyi silmiyor;
desteklenmeyen veya bozuk her giriş **mutasyondan önce** açık hata veriyor.

★ **Faz 3 ve Faz 4, 2B.2b-ii'yi beklemez.** §4'teki altı template de recipe
olarak paketlendiği için project staging hiçbirini bloke etmiyor.

### Faz 3 — Sade Template Hub

- New/Open/Recent/Recover girişlerini oluştur.
- Template kart grid'i ve preview detayını ekle.
- Klavye navigasyonu, Enter ve Escape davranışını ekle.
- İleri ayarları ikincil yüzeyde sun.
- `File > New Project` akışını aynı Hub'a bağla.

**Çıkış kriteri:** Kullanıcı fare veya klavyeyle birkaç eylem içinde template seçip viewport'a ulaşabiliyor.

### Faz 4 — Üretim template'leri

- Empty
- General Scene
- Product Lookdev
- Portrait & Groom
- Character Paint
- Terrain Environment

Her biri bağımsız statik içerik ve UX kabul kontrolünden geçmelidir.

**Çıkış kriteri:** Her template doğru sahne, doğru panel bağlamı, doğru dock/editör ve güvenli başlangıç performansıyla açılıyor.

### Faz 5 — Recent, recovery ve tercihlerin kalıcılığı

- Recent project metadata
- Kayıp dosya davranışı
- Autosave recovery
- Açılış tercihi
- Son kullanılan template
- Rehberlerin görüldü/kapatıldı durumu

**Çıkış kriteri:** Başlangıç deneyimi oturumlar arasında tutarlı ve bozuk recent/autosave girdilerine dayanıklı.

### Faz 6 — Kullanıcı template'leri

- Save as Template
- Kullanıcı preview seçimi veya üretimi
- Paketleme ve asset politikası
- Built-in ve user template ayrımı
- Import/export

Bu faz ilk Hub ve built-in template'ler üretim kalitesine ulaşmadan başlatılmamalıdır.

## 11. Kabul kriterleri — IPC test sözleşmesi

★★★ **Bu bölüm bir göz kontrol listesi değildir.** Altı template × on iki madde
elle doğrulanamaz; CLAUDE.md'nin birinci kuralı gereği kabul kriteri
**scriptten sürülebilir** olmalıdır. Aşağıdaki her madde bir assertion'dır ve
`templates.open` + mevcut sorgu yüzeyleri üzerinden koşar.

Template başına, `templates.open` sonrası:

| # | Assertion | Kaynak |
|---|---|---|
| 1 | `opened == true`, `ui_state_applied` manifest alanlarını kapsar | `templates.open` sonucu |
| 2 | Aktif properties context manifest ile aynı | UI state sorgusu |
| 3 | Sağ dock ve alt editör durumu manifest ile aynı | UI state sorgusu |
| 4 | `frame_target` nesnesi **seçili** | seçim sorgusu |
| 5 | Sahnede aktif kamera var ve `frame_target`'ı çerçeveliyor | kamera sorgusu |
| 6 | Sahne nesne sayısı manifest sözleşmesiyle birebir — fazlası demo kalıntısıdır | `rt.scene` listeleme |
| 7 | Her geometri flat `TriangleMesh` / DNA SoA; facade nesne **yok** | nesne listeleme |
| 8 | `assets.required` boş; opsiyonel asset yokken de `opened == true` | manifest + `templates.open` |
| 9 | `render.start` **boş olmayan** bir kare üretir | görüntü çıktısı |
| 10 | Kaydet → yeni oturumda aç → 2–7 yeniden geçer | proje round-trip |
| 11 | Bozuk/eksik zorunlu asset **mutasyondan önce** açık hata verir; aktif sahne değişmez | `templates.prepare` |
| 12 | Başka properties bağlamına geçişte Paint/Hair/Sculpt/Terrain state'leri güvenlik kurallarına uyar | UI state sorgusu |

★ **En sinsi başarısızlık 9'dur:** kamera yanlış yerdeyse veya materyal
bağlanmamışsa render **siyah ama hatasız** döner. Boş kare kontrolü olmadan
kalan on bir madde geçerken template kullanılamaz olabilir.

★ 6. madde §12.7'nin makine tarafından ölçülen hâlidir: beklenen nesne
kümesinden **fazlası** demo kalıntısıdır. İş hedefi primitive'i (UV sphere
scalp/paint hedefi) beklenen kümenin parçasıdır, fazlalık değil.

Ayrıca script edilemeyen, insanın bir kez bakması gereken tek madde:
template global tema/erişilebilirlik tercihlerini bozmamalıdır (allow-list
zaten bunu yapısal olarak engeller; bu kontrol allow-list'in doğruluğunu
sınar).

## 12. Ajanlar için uygulama kuralları

Template veya startup UX üzerinde çalışan ajanlar:

1. Bu belgeyi ürün yönü olarak kabul etmelidir.
2. Büyük kalıcı shelf veya viewport'u daraltan yeni ana toolbar önermemelidir.
3. Mevcut sol ray, bağlamsal sağ dock ve alt editör mimarisini yeniden kullanmalıdır.
4. Yeni bir paralel scene loader oluşturmadan önce mevcut Project Manager/SceneSerializer yolunu incelemelidir.
5. Scene state ile UI layout state'i birbirine kontrolsüz biçimde karıştırmamalıdır.
6. Template yüklemeyi başarısızlıkta yarım durum bırakmayacak şekilde tasarlamalıdır.
7. Built-in template içeriğini demo gösterisi değil gerçek üretim başlangıcı olarak değerlendirmelidir. ★ **İş hedefi primitive ile dekoratif gösteriyi ayırt etmelidir:** üzerinde çalışılan yüzeyin kendisi (groom'lanacak scalp, boyanacak hedef mesh, ürün yerleştirme primitive'i) yasak değildir — hair bir yüzey olmadan, paint bir hedef olmadan var olamaz. Yasak olan, işle ilgisi olmayan ve kullanıcının silmek zorunda kalacağı süs içeriğidir. Bu ayrımı bilmeyen bir ajan UV sphere'i kurala aykırı sanıp söker ve template'i çalışamaz hâle getirir.
8. Yeni template eklerken manifest, preview, içerik sözleşmesi ve kabul checklist'ini birlikte güncellemelidir.
9. Workspace talimatı gereği proje build'i veya uygulama çalıştırmamalı; kullanıcıya kesin build/test checklist'i teslim etmelidir.
10. Kalıcı manifest, schema, guidance ve diğer makine tarafından okunan JSON içeriklerinin tamamını İngilizce tutmalıdır; yerelleştirme JSON sözleşmesine gömülmemeli, ileride ayrı localization katmanından gelmelidir.
11. 2000 satırı aşan mevcut kaynak dosyalarına feature implementation eklememelidir. Özellik mantığı odaklı yeni `.h/.cpp` modüllerinde kurulmalı; büyük dosyalara yalnız zorunlu en küçük include, declaration, call veya registration bağlantısı yapılmalıdır.
12. Her yeni kullanıcı özelliğini hem scripting API'ye hem IPC'ye açmalıdır. UI, script ve IPC aynı kanonik servis/çekirdek işlemlerini çağırmalı; binding katmanlarında veya UI içinde iş mantığı kopyalanmamalıdır. Script ve IPC yüzeyleri, aynı doğrulama/hata semantiği ve ilgili test/dokümantasyon teslim edilmeden özellik tamamlanmış sayılmaz.

Template Hub özelinde `scene_ui.cpp` ve `ProjectManager.cpp` yeni registry, loader veya Hub implementasyonunun evi değildir. Bu dosyalar yalnız yeni modüllere minimal bağlama noktaları sağlayabilir.

Template Hub için bu kural en az şu dış yüzeyleri gerektirir:

- Template listeleme ve metadata sorgulama
- Kimlikle template doğrulama
- Kimlikle yeni proje/template açma isteği
- Yükleme durumu ve açıklanabilir hata sonucu
- İleriki fazlarda recent/recovery ve kullanıcı template yönetimi

IPC ve script çağrıları UI diyaloğunu taklit etmemeli; aynı headless-safe Template Registry/Loader servisini kullanmalıdır. Unsaved-changes gibi etkileşim gerektiren durumlar dış API'de sessiz onaylanmamalı, açık policy/sonuç olarak ifade edilmelidir.

## 13. Bilinçli olarak kapsam dışında

İlk geliştirme dalgasında şunlar yapılmaz:

- Cloud template marketplace
- Online asset indirme
- Otomatik AI sahne üretimi
- Çok sayıda yüzeysel template
- Template için ayrı renderer veya geometry pipeline
- Mevcut UX'i Houdini/Maya/3ds Max shelf kopyasına dönüştürme
- Kullanıcının bütün kişisel UI tercihlerini template içine gömme
- **Built-in template'lere binary asset (model, HDRI, doku) gömme.** Template'ler prosedüreldir; gerçek varlıklar yalnız `assets.optional` üzerinden, fallback'li biçimde referanslanır. Gerekçe §4'te.
- **Üçüncü taraf varlıkları kaynağı ve lisansı `THIRD_PARTY_ASSETS.md`'ye yazılmadan dağıtıma sokma.** `assets/vegetation` PlantCatalog türevidir ve satılamaz; `assets/volume` karışık kaynaklıdır ve şu an yalnız geliştirme/test içindir.

## 14. Son ürün tanımı

Template Hub tamamlandığında kullanıcı RayTrophi'nin yeteneklerini menülerde aramak zorunda kalmadan, yapmak istediği işe uygun ve gerçekten çalışmaya hazır bir sahneyle başlamalıdır. Hub yön gösterir; ardından geri çekilir. Viewport, RayTrophi deneyiminin merkezi olarak kalır.
