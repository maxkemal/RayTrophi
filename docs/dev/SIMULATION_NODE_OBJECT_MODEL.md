# Simülasyon node katmanı — nesne modeli raporu

> **Durum:** TASLAK — 2026-08-17. Temel mimari önerisi; uygulanmadı. Onaylanırsa
> `NODE_SIMULATION_ARCHITECTURE_PLAN.md` BÖLÜM D'nin depolama/kapsam kısmını
> bu belge yeniden yazar. N0'ın yürütme sözleşmesi değişmez.

Bu belge "node sistemi nasıl olmalı" sorusuna **mevcut veri modelinden yola
çıkarak** cevap arar. Konuşmada dile getirilen öneriler burada ayrıca
eleştirilmiştir; bir kısmı reddedilmiştir.

---

## 1. Elimizde ne var — ölçülmüş envanter

### 1.1 Node çekirdeği tektir, ve bu iyi haber

Geometri, materyal, terrain, animasyon ve simülasyon graph'larının hepsi
**aynı** `NodeSystem` V2 çekirdeğini kullanıyor: `NodeBase` (id, pin, x/y,
`dirty`, `metadata`), `GraphBase` (nodes/links/groups/portals), `NodeRegistry`
(typeId → factory), ve jenerik `NodeEditorUIV2::draw(GraphBase&)`.

**Tek çatı yapısal olarak zaten kurulu.** Eksik olan çatı değil, *kapsam*.

### 1.2 Graph depolaması varlığa göre anahtarlanıyor

```cpp
std::unordered_map<std::string, ...GeometryNodeGraphV2>  geometry_node_graphs;  // nodeName ->
std::unordered_map<std::string, ...MaterialNodeGraphV2>  material_node_graphs;  // materialName ->
```

Simülasyon graph'ı ise **tek global singleton** (`g_sim_graph`,
`RtApiSimNodes.cpp`). Bu, deponun kendi emsalinden sapan tek graph ailesi.

### 1.3 ★★★ Sahne varlıklarının kapsamları ASİMETRİK

Bu raporun en önemli ölçümü. Üç varlığın alanlarına bakıldığında:

| Varlık | `domain` alanı | `source_object` alanı | Gerçek kapsamı |
|---|---|---|---|
| `SimulationFlowSourceInfo` | **VAR** | var | **domain'e ait** |
| `SimulationColliderInfo` | **YOK** | var | sahne geneli |
| `ForceFieldInfo` | **YOK** | — | sahne geneli, **uzamsal** |

Yani **"kuvvet alanını domain'e bağla" bugün ifade edilemez.** Kuvvet alanı
konumu, şekli ve falloff yarıçapı olan uzamsal bir varlıktır; hacmine giren
her şeye etki eder. Collider da öyle.

★★★ Ve bu muhtemelen **doğru tasarım**, eksiklik değil. Rüzgâr fiziksel olarak
"şu domain'e" esmez, bir bölgede eser. Node'a o kabloyu çizdirmek, çözücüde
karşılığı olmayan bir ilişki uydurmak olur — D.0'ın yasakladığı şeyin ta
kendisi, ve deponun ayrıca test ettiği "ölü kadran" sınıfı.

**Sonuç:** emitter *beyan edilen* bir ilişkidir, force/collider *geometrik*
bir ilişkidir. Node katmanı ikisini aynı kablo gibi göstermemelidir.

### 1.4 Ortam bilgisi zaten iki kapsamda, ve override bayrağıyla

`WorldThermalState` global; `SimulationGridDomainDesc` ise
`thermal_override_enabled` / `thermal_ambient_kelvin` / `thermal_oxygen`
taşıyor. Yani **"ortam kendi bilgisini tutar, kapsam onu ezebilir"** deseni
kodda zaten var. Kapsam modeli uydurma değil; veri onu şimdiden yapıyor.

---

## 2. Bulgu: node'lar tek cins değil, BEŞ cins

Mevcut on beş node tipine cinsine göre bakınca tablo şu:

| Cins | Node'lar | Ne yapar |
|---|---|---|
| **A. Varlık yansıması** | Domain, Object, (Emitter, Collider, Solver) | Bir sahne varlığının alanlarını yansıtır |
| **B. İlişki beyanı** | Fluid→Gas, Gas Ignition, Foam | Varlıklar arası ilişkiyi beyan eder |
| **C. Parametre sürücüsü** | Set Parameter | Tek bir alanı ezer |
| **D. Ölçüm** | Field Inspect, Surface Inspect, Cache | Okur, yazmaz |
| **E. Alt sistem bağlaması** | Liquid Material, Volume Material | Render'a bağlar |

### ★★ Bu tablo bir kusuru açığa çıkarıyor: C, A'nın eksikliğidir

`Set Parameter` bugün bir domain'i yapılandırmanın **ana yolu**. Ama bu bir
tasarım değil, **A cinsinin tamamlanmamış olmasının belirtisi**. Domain node'u
domain'in alanlarını tam yansıtsaydı, `Set Parameter`'a yapılandırma için
gerek kalmazdı.

Olgun tasarımda `Set Parameter`'ın işi başkadır: **bir değeri/eğriyi/alanı
isimli bir parametreye sürmek** — yani animasyon ve ifade kancası. Bugünkü
"her ayarı bununla yaz" kullanımı geçici olmalı.

★ Pratik sonuç: **`Set Parameter`'a yeni özellik eklemeyin.** A cinsini
tamamlayın; C kendiliğinden asıl işine çekilir.

---

## 3. Simülasyon graph'ı NE ÜRETİR

Bu soru cevaplanmadan kapsam kararı verilemez.

- Geometri graph'ı bir **mesh** üretir → terminal gerekir (`Output`), çünkü
  birden çok dal varken hangisinin sonuç olduğu belirsizdir.
- Materyal graph'ı bir **gölgeleme ağı** üretir.
- Simülasyon graph'ı **değer üretmez.** Bir **yapılandırma + bir program**
  üretir: hangi varlık nasıl ayarlanacak, hangi ilişki hangi sırada.

★ Bu yüzden simülasyon graph'ına `Output` node'u eklenmemelidir. Kuyruğu
zaten `Cache` (bake durumu) ve `Liquid/Volume Material` (render bağlaması)
oluşturuyor, ve ikisi de ne yaptığını adıyla söylüyor. Geometri graph'ının
şeklini taklit etmek, orada gerçek bir belirsizliği çözen bir node'u burada
süs olarak kopyalamak olur.

---

## 4. Öneri: KAPSAM birinci sınıf kavram olsun

Graph'ın sahibi = graph'ın kapsamı. Üç kapsam, ve üçü de verinin zaten
yaşadığı yer:

| Kapsam | Sahibi | Ne beyan eder | Depolama |
|---|---|---|---|
| **Object** | sahne objesi | Neyden yapılmış, nasıl yanar/erir (MSF), katı gövde | `object_sim_graphs[nodeName]` |
| **Domain** | fluid/gas domain | Izgara, çözücü, emitter'ları, cache, görünüm, kuplajları | `domain_sim_graphs[domainName]` |
| **World** | sahne | Ortam ısısı, oksijen, yerçekimi, global kuvvet alanları | `world_sim_graph` (tek) |

### 4.1 Sahip örtük bir node'dur

Geometri graph'ı bunu zaten yapıyor: `Base Mesh: Crate`, graph başına bir
tane, başlığı sahibinin adını taşıyor, ikincisi eklenemiyor.

Simülasyonda birebir karşılığı: her domain graph'ı kendi **`Domain: Fuel`**
node'uyla açılır. **Kullanıcı artık "Domain node'u ekleyip neye bağlayacağım"
sorusunu sormaz** — graph zaten o domain'in graph'ıdır. İkinci domain
eklendiğinde cevap da kendiliğinden gelir: onun kendi graph'ı olur.

### 4.2 Bir node hangi kapsamda yaşar — YERLEŞİM KURALI

> **Bir node, alanlarını YAZDIĞI varlığın kapsamında yaşar.**
> Başka kapsamdaki varlıklara yalnızca **isimle referans** verir.
> Geometrik ilişkiler beyan edilmez, **ölçülür**.

Uygulaması:

- **Emitter** → domain kapsamı. `flow_source.domain` alanı var, yani ilişki
  gerçek ve beyan edilebilir.
- **Collider / Force** → world kapsamı (orada düzenlenir). Domain graph'ında
  **ölçüm** olarak görünür: "bu domain'in kutusuyla 3 kuvvet alanı kesişiyor",
  kesişme hesaplanır. ★ Beyan değil rapor — çünkü ilişki geometriktir.
- **Substance / Pyrolysis / Phase Change / Surface Inspect** → object kapsamı.
  MSF verisi zaten `source_object` ile anahtarlı; bir kez bu yüzden arıza
  çıktı (bkz. `bugfix_msf_keyed_by_source_object_not_collider`).
- **Kuplajlar** → **kaynak** domain'in kapsamı, hedef isimle. "Fuel yanınca
  Smoke'u besler" cümlesi Fuel'in graph'ında durur. Pinler zaten handle değil
  isim taşıdığı için iki graph'ın birleşmesine gerek yoktur.

### 4.3 Bütünü görme ihtiyacı bir GRAPH ile karşılanmaz

Kapsamlara bölünce "tüm simülasyonu tek ekranda göremiyorum" itirazı gelir.
Cevabı dördüncü bir graph açmak **değil** — o, dördüncü bir otorite olurdu.
Cevap zaten var: `rt.sim_graph.couplings()` beyan edilen ve gerçekte koşan
kuplajları birlikte raporluyor. Genel görünüm bir **rapor**tur, bir düzenleme
yüzeyi değil.

---

## 5. Konuşmadaki önerilerin eleştirisi

Bağımsız değerlendirme istendi; sonuçlar:

| Öneri | Karar | Gerekçe |
|---|---|---|
| Node varlığı yansıtır, sahiplenmez | ✔ **kabul** | Materyal tarafında zaten ödenmiş ders. Tek doğru cevap. |
| Her domain kendi node sistemini kurar | ✔ **kabul** | Deponun kendi emsali (geometry/material graph). Serileştirme, silme ve seçim doğal geliyor. |
| Emitter sahnedeki bir objeyi listeden seçer | ✔ **kabul** | `source_mode = mesh_surface` + `source_object` zaten var. |
| Çözücüler node olsun | ✔ **kabul, şerhli** | Yansıtıcı çerçevede meşru; hangi domain'in hangi ızgarayla koştuğunu **görünür** kılar (bu görünmezlik bake arızasına mal oldu). ★ Şerh: node bir **görünüm**, anahtar değil — silmek çözücüyü kaldırmaz. |
| Kuvvet alanları domain'e bağlanır | ✘ **RED** | `ForceFieldInfo`'da `domain` yok ve olmamalı: kuvvet uzamsaldır. Kablo çizmek çözücüde karşılığı olmayan ilişki uydurur. Yerine **kesişme ölçümü**. |
| Parametreleri haritalar sürsün | ⚠ **koşullu** | Yalnızca çözücünün gerçekten *alan olarak* örneklediği parametreler. Skaler okunan bir parametreye harita bağlamak ölü kadran üretir ("sahte kalınlık kadranları" tam buydu). Liste ölçülmeden bağlanmaz. |
| En son bir "çıktı" node'u | ✘ **RED** | Graph değer üretmiyor; terminal taklit olur. Bkz. §3. |
| MSF node'ları geometri graph'ına girsin | ✘ **RED** | Geometri graph'ı mesh üretir, fizik beyanı üretmez. Aynı tuval "bu graph ne üretir" sorusunu bulandırır. Object kapsamı ayrı graph olmalı. |

### ★ Kendi önerimin eleştirisi

Konuşmanın başında "solver node açmayalım, ikinci otorite olur" demiştim.
**Bu gerekçe yanlıştı** ve yansıtıcı çerçevesinde düşüyor: node otorite değil.
Bir varlığın birden çok *yönünü* (kimlik/sınırlar, ayrıklaştırma/çözücü,
cache, görünüm) ayrı node'lara bölmek meşrudur — materyal graph'ı bunu zaten
yapıyor.

---

## 6. Tek çatı: ne birleşir, ne birleşMEZ

### 6.1 Birleşen: kimlik, depolama, UI, ve ★ ATTRIBUTE HAVUZU

"Aynı havuzdan beslenen kümeler" fikrinin kodda karşılığı **attribute
katmanıdır** ve D.3 ile başlatılmıştır: isim sınırda çözülür, döngüde indeks
kullanılır. Geometri flat SoA attribute tutuyor, MSF texel başına attribute
tutuyor, parçacıklar attribute tutuyor.

★★★ **Ortak para birimi attribute'tur.** Üç graph ailesinin gerçek buluşma
noktası ortak bir *attribute/field referans tipi* — geometri bir attribute
yazar, sim onu okur, materyal onunla gölgeler. Havuz zaten var; eksik olan
ortak referans tipi ve keşif yüzeyi.

`Nodes ▾` sekmesindeki seçici de bu yüzden doğru çıktı: o seçici aslında bir
**kapsam seçicisidir**, ve tek çatının kullanıcıya görünen yüzüdür.

### 6.2 ★★★ Birleşmemesi GEREKEN: yürütme sözleşmesi

`GraphBase::evaluate()` saftır (`clearCache()` + `markAllDirty()`). Geometri ve
materyal için doğru. **Simülasyonda ölümcül**: orada durum birikmiş tarihtir,
"girdilerden yeniden hesapla" = "simülasyonu sıfırla", ve belirti hata değil
"sim hiç ilerlemiyor" olur.

Tek çatı **kimliği, depolamayı, UI'yi ve attribute'ları** birleştirir;
**değerlendirme semantiğini birleştirmez.** Bu satır raporun en pahalı cümlesi:
birleştirme hevesiyle `evaluateSimulation()`'ı `evaluate()`'e geri katan bir
yeniden yapılandırma, sessizce çalışmayan bir simülasyon üretir.

---

## 7. Node katmanının çekirdekte açığa çıkaracağı boşluklar

Kapsam modeli bunları görünür kılar; hiçbiri node ile kapatılamaz:

1. `WorldThermalState`'in **hiç** script yüzeyi yok → World kapsamı bugün boş.
2. Foam'un script yüzeyi yok (`foam_material_id`, aç/kapa).
3. `flow_source.delete` yok (create/update/get/list var).
4. `splat_material`'in `updateFluidDomain`'de yeri yok.
5. Collider/force **domain kapsamlı değil** — §1.3. Bu bir çekirdek kararıdır,
   node'un düzeltebileceği bir şey değil.

---

## 8. Göç sırası (onaylanırsa)

Doğrulanmış N0–N7'ye dokunulacak; testler duruyor, kırılırsa söylerler.

1. **Kapsam altyapısı** — `NodeSystem::ScopedGraph` (sahip kimliği + kapsam),
   `scene_data.h`'de üç depolama, örtük sahip node'u.
2. **`rt.sim_graph.*` kapsam argümanı alır.** ★ Argüman **opsiyonel
   yapılmayacak**: "aktif domain" varsayımı tam olarak deponun ısırdığı sessiz
   varsayım desenidir. Beş test dosyası buna göre güncellenir.
3. **A cinsini tamamla:** Domain node'u panel kadar kapsamlı olur, Solver
   node'u ayrılır, Emitter node'u eklenir (+ inspector'da `Create…`).
4. **World kapsamı** — önce `WorldThermalState`'in dört katmanı, sonra node'u.
5. **Object kapsamı** — MSF node'ları global graph'tan oraya taşınır.
6. **Ölçüm ilişkileri** — domain ile kesişen collider/force raporu.

★ 3'ten önce 1–2 yapılmazsa dört node yanlış temele yazılmış olur.

---

## 9. ★★★ Evrimsel yön: madde ve alan — "gerçek dünya gibi attribute"

Bu bölüm bir sonraki adım değil, **yönü** tarif eder. Değeri, bugünkü kararların
hangisinin o yöne bakıp hangisinin ona ters düştüğünü söylemesindedir.

### 9.1 Fizikte üç varlık tipimiz yok, ikisi var

Gerçek dünyada "emitter varlığı", "collider varlığı", "kuvvet alanı varlığı"
diye şeyler yoktur. Şunlar vardır:

- **Madde** — özellikleri olan nesneler: neyden yapılmış, sıcaklığı, kütlesi,
  yüzeyi.
- **Alan** — uzayın özellikleri: yerçekimi, rüzgâr, sıcaklık, basınç.

Ve **etkileşim beyan edilmez; özellikler uzayda buluştuğu için olur.**

Bizim üç varlığımız fiziğin değil, **DCC'nin eseri**. Ontolojiye oturtunca:

| Bugünkü varlık | Aslında ne | Doğal kapsamı |
|---|---|---|
| Collider | *bu obje sıvı çarpışmasına katılır, temsili mesh_sdf* | **madde özelliği** |
| Emitter | *bu obje S maddesini R debisiyle yayar* | **madde özelliği** |
| Force field | *uzayın bu bölgesinde şu kuvvet var* | **alan** (zaten doğru) |
| MSF | neyden yapılmış, nasıl yanar/erir | **madde özelliği** (zaten doğru) |
| Grid domain | alanların çözüldüğü uzay bölgesi | **alan taşıyıcısı** (zaten doğru) |

### ★★★ Ve bu model kodda ZATEN çalışıyor: termal kuvvet alanı

İlkenin tam hâli şudur: **alan etkiyi sunar, maddenin özellikleri tepkiyi
belirler.** Kanıtı uydurmaya gerek yok, `ForceFieldInfo`'da duruyor:

> *"A Thermal field exerts no force at all — it drives Material State Field
> surface heating (ignition, char, incandescence) only."*

Termal alan **hiç kuvvet uygulamaz**; yerel ortam sıcaklığını falloff'la
değiştirir, gerisini objenin MSF durumu belirler — maddesinin tutuşma noktası,
nemi, kömürleşme derecesi. Aynı alan, iki farklı objede iki farklı sonuç, ve
hiçbir yerde "bu alan şu objeyi etkiler" diye bir beyan yok.

★★ Yani madde/alan ontolojisi bu depoda bir hedef değil, **zaten uygulanmış**
bir model. Node katmanında eksik olan tek şey ona bir ad vermekti. Bu, §9'un
geri kalanını spekülasyon olmaktan çıkarıp mevcut davranışın genellemesi
yapıyor — ve "etkileşim beyan edilmez, ölçülür" kuralının gerekçesi de budur:
tepki alanın değil, **maddenin durumunun** fonksiyonu.

★ Bu tablo §1.3'teki asimetriyi **açıklıyor**: kuvvet alanının `domain` alanı
olmaması bir eksiklik değil, ontolojinin doğru tarafında durması. Collider'ın
`source_object` taşıyıp `domain` taşımaması da öyle — o bir madde özelliği.

### 9.2 ★ Ve tuhaf olan `flow_source.domain`

Ontolojiye göre emisyon bir madde özelliğidir; hangi çözücü bölgesini
beslediği **geometrik** olmalıydı — obje hangi domain'in kutusundaysa onu
besler. Açık `domain` alanı bir DCC kolaylığı.

Ama kaldırılmamalı, ve gerekçesi de gerçek: bir obje **iki çakışık domain'in**
içindeyse geometrik kural belirsizdir, ve bu depo çakışık hacim kutularının
bedelini zaten ödedi. Açık bağlama o belirsizliği kaldırıyor.

**Sonuç:** ontoloji hedeftir, açık bağlama ise belirsizliğin çözümüdür. İkisi
çelişmiyor — ama emitter'ın `domain` alanı bir *özellik* değil bir *çözüm*
olduğu için, node katmanında öyle sunulmalı: madde özelliği yukarıda, bağlama
ayrı ve gerekçeli.

### 9.3 Enerji zaten attribute ile taşınıyor — bu kısım BİTMİŞ

"Enerjiler attribute ile taşınır" fikrinin kodda karşılığı zaten var:
MSF `temperature` / `char` / `fuel_remaining` / `mass_loss`, grid `temperature`
kanalı, parçacık `temperature` / `combustible_fraction` / `substance_tag`.

Yani yön doğrulanmış durumda; icat edilecek bir şey yok, **bağlanacak** bir şey
var.

### 9.4 ★★★ Ve buradaki TUZAK: attribute ≠ dinamik özellik torbası

"Her şey attribute olsun" fikri doğal olarak string anahtarlı, dinamik bir
property bag'e davet eder. **Bir çözücüde bu felaket olur** — ve bu deponun
geometri modeli tam tersi gerekçeyle tipli flat SoA.

Koruma zaten D.3'te yazılı ve tekrar edilmeli:

> Attribute **depolama yaratmaz**; var olan diziye isimle bağdır.
> **İsim sınırda, indeks döngüde.**

Aynı şekilde **grid kanalları attribute'a çevrilmemeli**: onlar voksel
ızgarasında yaşayan ayrı bir temsildir, kopyalamak aynı verinin ikinci temsilini
üretir. Ortak **sorgu yüzeyi** onları *isimlendirebilir*; bu, onları attribute
*yapmaz*. Temsil ile isimlendirme aynı şey değildir.

### 9.5 Hem ajana hem insana yarayan somut ilk adım

İkisinin de ihtiyacı aynı: **keşfedilebilirlik, tek tip okuma/yazma, ölçüm.**

Ve bu yüzey zaten **iki kez** yazılmış durumda:
`sim_graph.attributes(domain)` ve `sim_graph.surface_attributes(object)`. İkisi
aynı şeyin iki özel hâli. Birleştirilmesi küçük bir iş, ve tek çatının ilk
gerçek tuğlası olur:

```
rt.attr.list(scope, id)      # scope: object | domain | world
rt.attr.stats(scope, id, name)
```

İnsana kazancı: node inspector'ında "bu kapsamda ne var" listesi tek yerden
gelir. Ajana kazancı: ölçüm tek imza. **Yeni depolama yok, yeni kavram yok** —
var olan iki fonksiyonun ortak adı.

### 9.6 Evrim sırası

| Aşama | İş | Ön koşulu |
|---|---|---|
| 0 | Kapsamlı graph'lar (§8) | — |
| 1 | `rt.attr.*` — iki özel hâli birleştir | 0 |
| 2 | Varlık node'ları attribute varsa attribute üzerinden okur/yazar | 1 |
| 3 | Etkileşim **ölçülür**: domain ile kesişen force/collider raporu | 0 |
| 4 | Emitter'ın açık `domain` bağlaması geometrik olabilir mi — **yalnızca** 3 çalıştıktan sonra tartışılır | 3 |

★ 4'ü şimdi tartışmak erken: geometrik etkileşim raporu (3) çalışmadan,
geometrik bağlamanın belirsizliği ölçülemez. **Önce ölç, teşhis sonra.**

---

## 10. Karar bekleyen tek şey

Yukarıdaki her şey ölçüme dayanıyor; **kullanıcı kararı gereken tek nokta
Object kapsamının ayrı bir graph tipi olup olmayacağı** (§5'te (a) önerildi,
(b) reddedildi). Reddin gerekçesi güçlü ama bu bir ürün kararı: obje başına
iki ayrı graph (geometri + fizik) kullanıcıya iki yer mi, yoksa net bir ayrım
mı olarak görünür?
