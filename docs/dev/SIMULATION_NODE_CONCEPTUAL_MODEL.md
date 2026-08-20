# Simülasyon node katmanı — kavramsal model

> **Durum:** REFERANS — 2026-08-19. Bu katmanın **ne olduğunu** tanımlar.
> Hangi kararın neden verildiği [SIMULATION_NODE_OBJECT_MODEL.md](SIMULATION_NODE_OBJECT_MODEL.md)'de;
> bu belge onun **öncesidir**. Bir tasarım kararı bu modelle çelişiyorsa karar
> yanlıştır, model değil.

---

## Tek cümle

> **Simülasyon graph'ı, sahnenin BEYAN EDİLMİŞ NİYETİDİR.
> Varlıklar durumu tutar; graph, o durumun hangi kısmının BİLEREK seçildiğini tutar.**

Bunun dışındaki her şey sonuçtur.

---

## 1. Sürekli dönen soru: "panel zaten yapıyorsa node niye?"

Bu itiraz haklı görünüyor çünkü node'a *parametre girme aracı* olarak
bakıldığında **panel her açıdan daha iyidir**. Daha hızlı, daha az tıklama,
daha az kavram.

Ama panelin yapısal olarak **söyleyemediği** bir cümle var:

> *"Bu değer hakkında bir fikrim yok."*

Bir panel her alanı yazar. Viskozite kutusunda `0.5` gördüğünde bunun anlamı
belirsizdir: kullanıcı 0.5'i **seçti** mi, yoksa 0.5 **varsayılan** mı? Panelde
bu iki hâl aynı piksellerdir.

★★★ Bu, bu deponun en pahalı hata sınıfının ta kendisi — *"varsayılan bir
ölçüm değildir"* — ama teşhis tarafında değil, **yazarlık** tarafında.
`bugfix_water_preset_kept_fake_thickness_dials`, `'Volume' varsayılanı`,
`gas shader preset'i sessiz no-op`: hepsi "birinin seçtiği değer" ile "orada
duran değer" ayrımının kaybolmasıydı.

**Node katmanının varlık sebebi budur.** Her alanın bir `use` bayrağı taşıması
bir uygulama detayı değil, **katmanın tanımıdır**: bayrak kapalıysa o node o
parametre hakkında *fikir beyan etmiyor*. `in_use = false` ile `value = 0`
farklı şeylerdir, ve node katmanı bu farkı temsil edebilen **tek** yüzeydir.

Panel niyeti gösterir; **graph niyetin kendisidir.**

---

## 2. Modelden çıkan dört sonuç

Aşağıdakiler ayrı kararlar değil, 1. bölümün zorunlu sonuçlarıdır. Bir tanesini
bozmak modeli bozar.

### 2.1 Node varlığı YANSITIR, sahiplenmez

Durum varlıkta yaşar (`SimulationGridDomainDesc`, `SimulationFlowSourceInfo`,
MSF alanları). Node o durumun bir **görünümü** ve üzerine yazılmış bir
niyettir.

Bu yüzden node silmek varlığı silmez, ve inspector'da `Create…` düğmesi
**yoktur** — varlık yaratmak `fluid.create_domain`'in işidir. Node'a yaratma
düğmesi koymak, tam da "node varlığı sahiplenir" yanılsamasını üretirdi.

### 2.2 Graph DEĞER üretmez, o yüzden `Output` node'u yoktur

Geometri graph'ı bir mesh üretir — birden çok dal varken hangisinin sonuç
olduğu belirsizdir, terminal gerekir. Materyal graph'ı bir gölgeleme ağı
üretir.

Simülasyon graph'ı bir **yapılandırma ve bir sıra** üretir. Terminali yoktur
çünkü çözülecek bir belirsizlik yoktur. Geometri graph'ının şeklini taklit
etmek, orada gerçek bir işi olan node'u burada süs olarak kopyalamak olur.

### 2.3 ★★★ Yürütme semantiği BİRLEŞMEZ

`GraphBase::evaluate()` saftır: `clearCache()` + `markAllDirty()`, yani
"girdilerden yeniden hesapla". Geometri ve materyal için doğrudur.

Simülasyonda **ölümcüldür**. Orada durum *birikmiş tarihtir*; "girdilerden
yeniden hesapla" cümlesinin karşılığı **"simülasyonu sıfırla"**dır. Ve belirti
bir hata değil, *"sim hiç ilerlemiyor"* olur — yani en pahalı türden.

Tek çatı **kimliği, depolamayı, UI'yi ve attribute'ları** birleştirir.
**Değerlendirmeyi birleştirmez.** Birleştirme hevesiyle `evaluateSimulation()`'ı
`evaluate()`'e geri katan bir refactor, sessizce çalışmayan bir simülasyon
üretir.

### 2.4 Etkinin KABLOYU izlemesi gerekir

Bağlanmamış bir Emitter node'u bir zamanlar yine de komut yayıyordu — kimliğini
kendi alanından aldığı için kabloya ihtiyacı yoktu. Sonuç: graph'ın *okunuşu*
ile *yürütmesi* ayrıştı, ve graph **dekoratif** oldu.

★★ Kural: **bir sim node'unun etkisi kablosunu izlemeli.** Kimliğini kendi
alanından bilen bir node bile zincire girmek için bağlanmak zorundadır. Aksi
hâlde tuvale bakan insan sebebi göremez, çözücü sonucu görür.

---

## 3. Kapsam: graph KİMİN graph'ı

> **Graph'ın sahibi, graph'ın kapsamıdır.**

| Kapsam | Sahibi | Ne beyan eder |
|---|---|---|
| **Object** | sahne objesi | Neyden yapılmış, nasıl yanar/erir, katı gövde |
| **Domain** | fluid/gas domain | Izgara, çözücü, emitter'ları, cache, görünüm, kuplajları |
| **World** | sahne | Ortam ısısı, oksijen, yerçekimi |

"Domain node'unu ekleyip neye bağlayacağım?" sorusu bu modelde **sorulmaz**:
graph zaten o domain'in graph'ıdır ve kendi `Domain: Fuel` node'uyla açılır.
İkinci domain eklendiğinde cevap kendiliğinden gelir — onun kendi graph'ı olur.

**Yerleşim kuralı:**

> Bir node, alanlarını **yazdığı** varlığın kapsamında yaşar.
> Başka kapsamdaki varlıklara yalnızca **isimle** referans verir.

Kuplajlar **kaynak** domain'in kapsamında durur: *"Fuel yanınca Smoke'u besler"*
cümlesinin bir **yazarı** vardır, o da Fuel'dir. Pinler handle değil isim
taşıdığı için iki graph'ın birleşmesine gerek yoktur.

★ "Tüm simülasyonu tek ekranda göremiyorum" itirazının cevabı dördüncü bir
graph **değildir** — o, dördüncü bir otorite olurdu. Genel görünüm bir
**rapordur**: `sim_graph.couplings()` beyan edileni ve gerçekte koşanı yan yana
verir.

---

## 4. Neden bazı şeylerin kablosu var, bazılarının yok

Bu, tuvale bakınca en kafa karıştırıcı görünen şey, ve cevabı fizikte.

Bizim üç varlığımız — emitter, collider, force field — **fiziğin değil,
DCC'nin** eseri. Fizikte iki şey vardır:

- **Madde** — özellikleri olan nesneler: neyden yapılmış, sıcaklığı, yüzeyi.
- **Alan** — uzayın özellikleri: yerçekimi, rüzgâr, sıcaklık.

Ve **etkileşim beyan edilmez; özellikler uzayda buluştuğu için olur.**

| Bugünkü varlık | Aslında ne | Node'da nasıl görünür |
|---|---|---|
| Emitter | madde özelliği (+ hedef bağlaması) | **kablo** — beyan edilen ilişki |
| Kuplaj | bir yazarı olan beyan | **kablo** |
| Collider | madde özelliği | **ölçüm** — kesişme raporlanır |
| Force field | alan | **ölçüm** — kesişme raporlanır |

★★★ Kuvvet alanına kablo çizmek, **çözücüde karşılığı olmayan bir ilişki
uydurmak** olur. Rüzgâr "şu domain'e" esmez; bir bölgede eser. `ForceFieldInfo`'da
`domain` alanının olmaması bir eksiklik değil, ontolojinin doğru tarafında
durmasıdır.

Ve bu model burada **zaten çalışıyor**: termal alan *hiç kuvvet uygulamaz*,
yalnızca yerel ortam sıcaklığını değiştirir; gerisini objenin MSF durumu
belirler — tutuşma noktası, nem, kömürleşme. Aynı alan, iki objede iki farklı
sonuç, ve hiçbir yerde "bu alan şu objeyi etkiler" beyanı yok.

> **Alan etkiyi sunar, maddenin özellikleri tepkiyi belirler.**

★ Tek istisna `flow_source.domain` ve gerekçesi dürüst: bir obje **iki çakışık
domain'in** içindeyse geometrik kural belirsizdir, ve bu depo çakışık hacim
kutularının bedelini zaten ödedi. Açık bağlama bir *özellik* değil, bir
**belirsizlik çözümüdür** — node'da da öyle sunulmalı.

---

## 5. Bugün ne var, ne yok (2026-08-19'da kodda ölçüldü)

| | durum |
|---|---|
| Üç kapsam depolaması (`scene_data.h:171-174`) | ✅ |
| 18 node tipi, `sim.solver` / `sim.domain_settings` / `sim.emitter` dahil | ✅ |
| Her alan opt-in (`use` bayrağı), panelde de öyle çiziliyor | ✅ |
| Kapsam argümanı **zorunlu** — "aktif domain" varsayımı yok | ✅ |
| Öksüz graph `owner_missing` ile raporlanıyor | ✅ |
| **World kapsamı** — `WorldThermalState`'in script yüzeyi **hiç yok** | ❌ |
| **Object kapsamı** — MSF node'ları hâlâ eski yerinde | ❌ |
| **Ölçüm ilişkileri** — domain ile kesişen force/collider raporu | ❌ |
| `rt.attr.*` — `sim_graph.attributes` + `surface_attributes` birleşimi | ❌ |

★ World kapsamı bugün **boş bir kutu**: kapsam altyapısı çalışıyor ama içine
koyacak tek bir script yüzeyi yok. Bu bir node eksiği değil, **çekirdek
eksiği** — ve node katmanının en yararlı işlerinden biri onu görünür kılmış
olması.

---

## 6. Kafan karıştığında sorulacak tek soru

Yeni bir node, yeni bir alan veya yeni bir kablo düşünürken:

> **Bu, birinin BEYAN ETTİĞİ bir niyet mi, yoksa dünyanın bir OLGUSU mu?**

- **Niyet** → graph'a girer, bir sahibi olur, opt-in olur, kablosu olur.
- **Olgu** → graph'a girmez. **Ölçülür** ve raporlanır.

Ve ikinci soru:

> **Bu değer "ayarlanmadı" hâlini temsil edebiliyor mu?**

Edemiyorsa bir bayrak gerekir. (`fluid_substance` bunun için ayrı tutuldu:
boş string **meşru bir değerdir**, yani boşluk "ayarlanmadı" anlamına gelemez.)

★★★ Ve asla tahmin edilmiş bir kadran ekleme. `granular_enabled` bilerek
açılmadı: açıp kapatmanın biriken durumu geçersiz kılıp kılmadığı **ölçülmedi**.
**Tahmin edilmiş bir kadran, eksik kadrandan kötüdür — çünkü tahmin görünmez.**
