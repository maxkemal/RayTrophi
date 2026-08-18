# Ajan destekli prodüksiyon pipeline'ı — karar kaydı

> **Durum:** TASLAK — Onlarca eşzamanlı uygulama örneği, tek proje, insan ve ajan
> işçilerin birbirinin yerine geçebildiği bir prodüksiyon düzeni. Hiçbiri
> uygulanmadı. Bu not bir yol haritası değil, **hangi kararın geri alınamaz
> olduğunun** kaydıdır.

Bu belge bilerek özellik listesi değil. Aşağıdaki kararların çoğu, yanlış
verilirse **sonradan düzeltilemez** — çünkü veri düzenine ya da kimliğe dokunur.
Görünür olan parça (sohbet, ajan rolleri) ise en kolay değiştirilebilen parça,
ve o yüzden en sona bırakılmıştır.

---

## 0. Problemin şekli

Hedef: onlarca uygulama örneği aynı anda açık, her biri **tek bir projenin farklı
sahnesini** hazırlıyor. Bir koordinatör kullanıcılarla konuşur, işi böler, geri
gelen işi birleştirir. İşçiler ajan **ya da** gerçek insan olabilir; koordinatör
açısından ikisi aynı arayüzdür.

★ **Bu bir "çoklu ajan" problemi değil, bir pipeline ve sürüm kontrolü
problemidir.** Ajanların varlığı yalnızca bazı işçilerin uyumadığı anlamına
gelir. Zor kısım ajan protokolünde değil, işin **dağıtılması ve geri
birleştirilmesindedir**.

Bunu erken görmek gerekiyor, çünkü aksi hâlde ajan katmanı kusursuz kurulur ve
proje birleştirmede batar. Tasarımı yanlış yerde optimize etmek buradaki en
pahalı hata olur.

---

## 1. Zaten elde olanlar

Bu kısım şans eseri değil; çoğu proje bu katmanlarda aylar harcar.

| Var olan | Nerede | Pipeline'daki karşılığı |
|---|---|---|
| Uzak IPC + kimlik doğrulama | `RtIpcTransportTls.cpp`, [REMOTE_IPC_GATEWAY.md](REMOTE_IPC_GATEWAY.md) | Onlarca makinedeki örneğe dışarıdan bağlanma |
| Yetki modeli, **fail-closed** | `RtIpcSecurity.cpp` | İşçi rolü = capability maskesi |
| Denetim kaydı | `RtIpcAudit.cpp` | "Bunu kim değiştirdi" |
| Yazmaların seri olması | `enqueueResult` / `enqueueQuery` ana döngüye kuyruklanır | Yarış yok; yalnızca **çakışma** var |
| Bake geçerliliği **hesaplanabilir** | `hashFluidDomainSolverConfig`, `computeFluidCouplingSignature` | "Bu işçinin yeniden pişirmesi gerekiyor mu" |
| Görsel doğrulama otomatikleştirilebilir | `render.start` görüntü üretir, görüntüler okunabilir | İş emri kendi kabul ölçütünü taşıyabilir |

★ Yerel pipe token istemez, **uzak istekler doğrulanır** — yani güvenlik sınırı
zaten doğru yerde duruyor ve çok makineli senaryo bu ayrımı bozmuyor.

---

## 2. Geri alınamaz kararlar (önce bunlar)

### 2.1 ★★★ Kimlik tahsisi

Bugünkü `.rtp` kökünde şunlar var: `next_object_id`, `next_texture_id`,
`next_model_id`, `next_particle_system_id`. Bunlar **tek bir uygulamanın
sayaçlarıdır**.

İki işçi paralel çalıştığında **ikisi de nesne 47'yi üretir**. Birleştirmede biri
sessizce diğerinin üstüne biner ve hiçbir yerde hata görünmez.

İki ağırlaştırıcı etken:

- **★ Material ID 0 geçerli bir kimliktir.** "0 ise yok say" temelli hiçbir
  birleştirme mantığı bu kod tabanında güvenli değildir.
- Çoklu materyal içe aktarımında **düğüm adı çakışması zaten bir kez ısırdı**
  (`bugfix_multimaterial_import_nodename_collision`). Ad temelli eşleme de
  güvenli değil.

**Karar gerekiyor:** işçi başına ayrık kimlik aralığı **ya da** içerik-adresli /
UUID kimlik.

**Neden geri alınamaz:** kimliğe dokunan her sistem (serileştirme, undo, node
grafiği, materyal bağlama, sim domain referansları, sidecar `.rtp.bin`) yeniden
açılmak zorunda kalır. Bu karar ertelenirse proje taşımak gerekir.

### 2.2 ★★★ Paylaşılan / sahne-yerel / türetilmiş ayrımı

Üç kova, ve üçüncüsü en çok yanlış anlaşılanı:

| Kova | İçerik | Kural |
|---|---|---|
| **Kütüphane** | materyaller, maddeler, yakıt profilleri, presetler, dokular, düğüm grafiği şablonları | Paylaşılır; yazma yetkisi **nadir ve tek elden** |
| **Sahne-yerel** | nesne örnekleri, dönüşümler, sim domain'leri, kameralar, ışıklar | Serbestçe paralel |
| **Türetilmiş** | bake'ler, cache'ler, `.rtp.bin` sidecar'ları, render çıktıları | **ASLA birleştirilmez** |

★★ Türetilmiş olanlar birleştirilmez — **geçersiz kılınıp yeniden üretilir**. İki
işçinin bake'ini birleştirmeye çalışmak, hiçbir yerde hata vermeyen ama sessizce
bozuk bir sim üretir. Neyse ki geçerlilik sorusu zaten cevaplanabiliyor (§1).

**Neden geri alınamaz:** bu ayrım veri düzenini ve dosya sınırlarını belirler.
Sonradan bölmek, projenin diskteki biçimini değiştirmek demektir.

### 2.3 ★★★★★ İmza **gerçekleşen** yolu içermeli, istenen yolu değil

Bu, farm ölçeğinde en sinsi arıza ve tamamen bu depoya özgü.

Vulkan birincil, CUDA ikincil; ve elde **sessiz geri düşüş** vakası zaten var —
Vulkan MGPCG boşluğunda çözücü fark ettirmeden başka bir yola düşüyor
(`project_vulkan_mgpcg_variational_gap`).

Tek makinede bu bir performans notudur. Otuz makinede şu demektir: **işçi A ile
işçi B aynı girdiden farklı simülasyon üretir**, ikisi de "başarılı" döner, ve
fark ancak kare 200'de gözle belli olur. Bake paylaşımı açıksa bu, **sahte cache
isabeti** üretir — yani yanlış bake doğru sanılıp dağıtılır.

Bu, deponun kendi kuralının doğrudan uygulaması:

> **★ Varsayılan bir ölçüm değildir.** "MGPCG istedim" ile "MGPCG koştu" aynı şey
> değildir.

**Karar:** bake/cache imzasına backend **ve fiilen yürütülen çözücü yolu** girer.
Farm'a çıkmadan önce, çünkü sonrasında paylaşılan bütün cache'ler şüpheli olur.

### 2.4 ★★ Varlık kökeni ve lisans, veride taşınmalı

Vegetation varlıklar PlantCatalog türevidir: **paylaşım serbest, satış kalıcı
olarak yasak.** Volume/VDB varlıkları ayrı bir kümededir ama depoda iç içe durur.

Onlarca kişinin çalıştığı ve teslim paketi üretilen bir pipeline'da, bu kısıt
**veri tarafından taşınmıyorsa** er ya da geç satılabilir bir pakete girer.
Kısıt, varlığın kendisinde bir alan olarak durmalı ve **birleştirme ile dışa
aktarma kapısında** kontrol edilmeli.

Şimdi bir alan; sonra bir hukuk problemi.

---

## 3. Geri alınabilir ama erken kurulması gereken

### 3.1 İş emri bir veri yapısıdır, sohbet değil

İnsan ve ajan işçinin birbirinin yerine geçebilmesi bu tasarımın en güçlü
tarafı — ama yalnızca iş emri ikisine de **aynı alanları** verirse çalışır:
insana kart, ajana JSON.

★★ Kritik olan: **kabul ölçütü makine tarafından kontrol edilebilir olmalı.**
Aksi hâlde koordinatör, bir ajanın işi bitirip bitirmediğini ajanın kendi
beyanından öğrenir. Bu, tek ajanla idare eder; onlarca ajanla çöker.

Bu depoda kabul ölçütü gerçekten yazılabilir: `render.start` görüntü üretir, IPC
sayısal ölçüm döndürür.

### 3.2 Ajan cevaplarının üç türü — ve zaten var olan karşılıkları

| Tür | Karşılığı | Not |
|---|---|---|
| **Proposal** | Kaydedilmiş ama **çalıştırılmamış** `SceneCommand` demeti | UI zaten record eder, execute etmez |
| **Action** | Çalıştırılmış demet + undo tutamağı | `undo` / `redo` IPC'de mevcut |
| **Observation** | Hiç komut üretmeyen okuma | Ama §3.3'e bak — tek başına yetmez |

★ Proposal'ı metin değil **komut demeti** yapmak onu diff'lenebilir, çakışma
kontrolü yapılabilir ve reddedilebilir kılar. Metin öneri, ajan sayısı arttıkça
denetlenemez hâle gelir.

**Eksik olan:** adlandırılmış checkpoint — bir Proposal demetini uygulayıp **tek
hamlede** geri alabilmek. `undo` var, demet kapsamı yok.

### 3.3 ★★★ Observation'lar bozulur, ve sessizce

Granüler parametreler `hashFluidDomainSolverConfig`'e bağlı: bir parametre
düzenlemesi bake'i düşürüp **kare 0'a geri sarar**. Tek kullanıcı için doğru
davranış.

Çok işçili dünyada anlamı şudur: Granular işçisinin tek bir Action'ı, Fluid
işçisinin az önce yaptığı **bütün gözlemleri geçersiz kılar** — ve Fluid işçisi
bunu bilmez.

**Karar:** Observation bir değer değil, bir **çifttir** — değer + hangi
jenerasyona karşı ölçüldüğü. `g_scene_geometry_generation` zaten var; sim tarafı
için imza hash'i zaten var. Koordinatör bayat gözlemleri **düşürmek zorunda**.

★★ Bu yapılmazsa ajanlar, artık var olmayan bir sahne hakkında son derece ikna
edici raporlar yazar. Hiçbiri yanlış olduğunu bilmez.

### 3.4 ★★★ Bağlam yükünde zor olan "ne" değil, "kim"

Kullanıcı viewport'ta bir bölge seçip "burayı biraz daha kırılgan yap" dediğinde,
ajanın "burayı" gerçekten seçili bölge olarak anlaması gerekir. Taşınacak alanlar
açık: seçili varlıklar, aktif panel, kare, aktif domain, hover'lanan nesne,
viewport bölgesi, kullanıcı modu.

Zor olan bunları saymak değil — **"burası"nı kimin çözdüğü.**

Ölçülmüş örnek (2026-08-16): `fluid.list_domains` ile `fluid.get` aynı domain'in
`granular_enabled` değeri için **farklı cevap** veriyordu; biri legacy editor
aynasını, diğeri grid domain'i okuyordu. İki ajan aynı anda "aktif domain" sorsa
iki farklı gerçeklik görürdü.

Deponun kendi kuralı da bunu söylüyor: **panel görünürlüğü seçim otoritesi
değildir.**

**Karar:** bağlam yükünü ajanlar toplamaz. **Tek bir çözücü** üretir, ve yük
üzerinde hangi otoriteden okunduğu ile hangi jenerasyonda çözüldüğü yazar.

### 3.5 ★★ API geneli değişmez: "ölçülmedi" ≠ "sıfır ölçtüm"

`render.volume_stats`'a `available` ve `enabled` bayrakları tam da bunun için
kondu: sıfırlarla dolu bir okuma "hacim bedavaydı" değil, "sayaçlar kapalıydı"
olabiliyordu.

Tek ajanla bu bir tuzak. Onlarca ajanla bu bir **salgın**: bir ajan varsayılanı
ölçüm diye raporlar, koordinatör ona göre karar verir, ve sonuç **makul
görünür**. Kimse bunu bug diye raporlamaz.

Bu bir özellik değil, **ajanların okuduğu her metot için bir değişmezdir.**

---

## 4. Koordinatör

### Uygulamanın **dışında**

İki gerekçe:

1. Test edilen şey uygulamanın kendisi. İçeride yaşayan bir koordinatör, uygulama
   takıldığında kurtaramaz — çünkü o da takılmıştır.
2. Uzak IPC zaten var; dışarıdan N örneğe konuşmak mevcut yolla mümkün.

### Mesaj yönlendirici **olmamalı**

Öyle olursa hem darboğaz hem tek karışıklık noktası olur. Üç şeyin sahibi olsun,
gerisini devretsin:

1. **İş dağıtımı ve kiralama** — hangi işçi, hangi kapsamda, ne kadar süreyle
   yazma hakkına sahip
2. **Kütüphaneye yazma yetkisi** — paylaşılan materyal/preset değişimi tek elden
3. **Birleştirme ve doğrulama kapısı** — geri gelen her delta buradan geçer

★ Kiralama teorik bir ihtiyaç değil: 2026-08-16'da tek bir ajan (ben), kullanıcı
aynı uygulamada çalışırken **bir proje açıp kullanıcının sahnesini değiştirdi.**
Boş sahneyle başlandığı için kayıp olmadı. Bu, olayın **tek işçili** hâli.

### İşçi rolleri = capability maskesi

Yeni bir izin sistemi yazmaya gerek yok; `RtIpcSecurity` bitleri zaten var ve
`authorize()` fail-closed:

| Rol | Maske | Yapabildiği |
|---|---|---|
| Observer | `Read` | Yalnız gözlem |
| Proposer | `Read \| Render` | Gözlem + öneri + kendi doğrulaması |
| Actor | `+ SceneWrite` | Sahneyi değiştirir |
| Publisher | `+ FilesWrite` | Teslim paketi üretir (§2.4 kapısı burada) |

★ Fail-closed olması doğru tarafta hata veriyor: unutulan bir yetki sessiz bir
kaza değil, sessiz bir **ret** üretir.

---

## 5. Ajanları hangi eksende bölmeli

Sezgisel bölme alt sistem başınadır: Terrain / Fluid / Gas / Granular / Material
/ Lighting. **Bu eksene ihtiyatla yaklaşılmalı.**

Bu depoda en pahalı hatalar alt sistemlerin *içinde* değil, **dikişlerinde**
çıktı:

- yangın → yapı blast kuplajı
- gaz ↔ parçacık sözleşmesindeki **dört sessiz kapı**
- sıvı görünüm düzenlemesinin render'a hiç ulaşmaması
- sim çıktısının `g_*_dirty` kapılarına asılması
- 2026-08-16 splat arızası: gaz yolu ile sıvı yolu **aynı hatayı** paylaşıyordu,
  biri düzeltilmiş diğeri maliyet gerekçesiyle bilerek ertelenmişti — ve o
  gerekçe çoktan geçersizleşmişti

Alt sistem başına ajan koymak, **hatanın yaşadığı sınırları birebir yeniden
üretir** ve dikişin sahibi kimse olmaz. "Gas'ta her şey normal", "Fluid'de de
normal" — ve arıza aradadır.

**Karar önerisi:** Validation kardeş bir ajan değil, **dikişlerin sahibi** olsun,
ve istendiğinde değil **her Action'dan sonra** koşsun.

---

## 6. Ölçek üzerine dürüst not

Onlarca eşzamanlı örnek, her biri GPU simülasyonu koşan ~1M satırlık bir
renderer — darboğaz koordinasyon mantığı olmayacak, **GPU belleği ve bake
depolaması** olacak.

Ve bu katmanı sürdüren tek kişi var.

★ Bu yüzden koordinatör katmanı olabildiğince **ince** ve var olanın üstüne
kurulmalı. Yeni protokol, yeni durum deposu, yeni kimlik sistemi yazmak —
üçü de sürdürülmesi gereken yüzeyi büyütür.

Yukarıdaki maddelerin çoğu *yeni sistem* değil, mevcut yapılara **birer alan
eklemektir**: imzaya koşan yol, denetim kaydına işçi kimliği ve iş emri, varlığa
lisans bayrağı, kimliğe işçi aralığı.

Ayrıca ölçek hedefi kendisi sorgulanmalı: yüz ajan bir yetenek kazancı değil, bir
**koordinasyon maliyetidir**. Az sayıda gerçekten yetkili ajan + çok sayıda
gözlemci daha iyi bir başlangıç. **Yazma yetkisi pahalı ve nadir olmalı; okuma
bedava.**

---

## 7. Sıra

Ucuzdan pahalıya değil — **sonradan değiştirilmesi imkânsızdan mümküne**:

| # | Karar | Neden bu sırada |
|---|---|---|
| 1 | Kimlik tahsisi (§2.1) | Sonra değiştirilemez; projeyi taşımak gerekir |
| 2 | Paylaşılan/sahne-yerel/türetilmiş (§2.2) | Veri düzenini ve dosya sınırlarını belirler |
| 3 | İmzaya gerçekleşen yol (§2.3) | Sessiz bozulmayı ve sahte cache isabetini kapatır |
| 4 | Varlık lisans alanı (§2.4) | Teslim paketi üretilmeden önce |
| 5 | İş emri + makine-kontrol edilebilir kabul (§3.1) | Koordinatörün "bitti mi" sorusu buna dayanır |
| 6 | Observation jenerasyonu + tek bağlam çözücü (§3.3, §3.4) | Ajanların birbirini yanıltmasını engeller |
| 7 | Koordinatör süreci (§4) | Yukarıdakiler olmadan yazılırsa yeniden yazılır |
| 8 | **Ajan rolleri ve sohbet katmanı** | **En son** |

★★ Sohbet katmanının en sonda olması bilinçli. En görünür parça o, ve o yüzden
ilk yapılmak istenir. Ama yanlış kurulursa kaybedilen birkaç haftalık iştir.
Kimlik yanlış kurulursa proje taşınır.

---

## Açık sorular

- Kimlik: ayrık aralık mı, UUID mi? UUID `.rtp` boyutunu ve okunabilirliğini
  etkiler; aralık ise işçi sayısına tavan koyar.
- Kütüphane yazımı tek elden olacaksa, bir materyal düzenlemesi için bekleme
  süresi ne olur? Bu, kütüphanenin ne kadar ince taneli bölüneceğini belirler.
- Bayat Observation düşürülüyorsa, koordinatör onu **yeniden mi ister** yoksa
  ajana mı bildirir? İkisi farklı ajan sözleşmesi demek.
- Kiralama kapsamı ne (sahne mi, domain mi, nesne mi)? Çok ince tanecik
  koordinasyon maliyetini, çok kaba tanecik paralelliği öldürür.
