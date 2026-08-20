# Fizik doğrulaması — çözücü KOŞUYOR mu, DOĞRU mu?

> **Durum:** AKTİF — 2026-08-19. Çözücü analitik olarak DOĞRULANDI; altı vaka
> da yeşil, biri (6) yapısal olarak kör. Rig
> `scripts/test/rt_test_physics_validation.py`; sonuç
> `scripts/test/_physics_validation_result.txt`.

## Neden

Bu depodaki bütün ölçüm aletleri *"oldu mu?"* sorusunu cevaplıyor — çağrı
ulaştı mı, kare değişti mi, özet şemayla uyuşuyor mu. **Hiçbiri "doğru mu?"
sorusunu cevaplamıyor.**

Bir render aracı için bu yeterli: göz karar verir. Ama hedef *"ajanların
kullanacağı ve kendini geliştireceği simülasyon"* olduğunda taşıyıcı boşluk
budur — yanlış bir çözücüye güvenen ajan yanlış fiziği kendinden emin öğrenir,
ve bütün alet yığını **başarı raporlar**.

Kuralın bir kat yukarısı: *varsayılan bir ölçüm değildir* →
**koşan bir çözücü bir doğrulama değildir.**

## Kural: bağımsız olarak bilinen sayı

Her vaka, **bu programı çalıştırmadan** bilinen bir sayıyla karşılaştırılır:
analitik çözüm ya da bir korunum yasası. Bağımsız doğrusu olmayan bir vaka bu
dosyaya girmez.

| # | vaka | bilinen doğru |
|---|---|---|
| 1 | serbest düşüş | `y(t) = y₀ − ½gt²` |
| 2 | kütleden bağımsızlık | 1 kg ile 100 kg aynı düşer (Galileo) |
| 3 | yerçekimi bir DEĞER | Ay/Dünya mesafe oranı = 1.62/9.81 |
| 4 | kütle korunumu | emitter yok ⇒ parçacık sayısı sabit |
| 5 | tohum dürüstlüğü | 0 parçacık yaratan çağrı başarı raporlamamalı |
| 6 | domain izolasyonu | ikinci domain birincinin parçacıklarını silmemeli |

★ 2. vaka bu deponun iki kez ödediği sınıfı hedefliyor: kütleye bölünmeden
uygulanan kuvvet, ve hız alanına yazılan impulse. İkisi de "biraz fazla güçlü"
gibi görünür ve bir kalibrasyon turuna gömülür. İkisi de bu karşılaştırmadan
sağ çıkamaz.

## ★★ Node katmanından GEÇMİYOR, ve bu bilinçli

Bu vakaları simülasyon node graph'ıyla yazmak doğal görünüyor ve plan da bu —
ama **enstrüman, ölçtüğü şeyle aynı arıza kipini paylaşmamalı** (CLAUDE.md).
Uygulanmayan bir opt-in bayrağı, bir çözücü vakasını kırmızıya düşürür; daha
kötüsü, varsayılan parametrelerle **yeşile**.

Sıra şu: **bu dosya referanstır**; node ile yeniden ifade edilmiş hâli ona karşı
koşan **diferansiyel testtir** (*"graph, doğrudan yolun ürettiği sayının
aynısını üretiyor mu?"*). O da sim node katmanına bugün eksik olan **bitti
tanımını** verir.

## İlk koşunun bulduğu (2026-08-19)

### ★★★ 1. Çözücünün hareketi dışarıdan GÖRÜLEMİYORDU

Yerçekimi + rigid gövde + **240 adım**, ve `scene.get_transform` hâlâ spawn
pozunu döndürüyordu — başarıyla, tamamen makul bir sayıyla.

★★★ **İlk teşhis yarım, ilk düzeltme YANLIŞTI, ve bunu yazmak önemli.**
"Çözücüler `Transform::current`'a yazar" varsaydım ve `final = current * base`
döndüren `scene.get_world_transform`'u ekledim. Derlendi, koştu, ve **yine 0.0
metre düşüş** ölçtü.

Gerçek şu: **rigid çözücü hiçbir transform'a yazmıyor.**
`RigidBodySystem::step` dünya deltası `D = B(t)·inv(B0)`'ı doğrudan **mesh vertekslerine BAKE ediyor** (`rigid_baker_`) ve transform tutamağına bilerek
dokunmuyor — çünkü import edilmiş bir objenin transform'unu oynatmak render'da
onu bozuyordu. Yani `base` de `current` de spawn pozunda kalıyor; kompozisyon
hiçbir şey değiştirmiyor.

Poz otoritesi `RigidBodyObject::last_rigid_delta` — ve `RigidBodySystem.h` bunu
zaten açıkça yazıyor: *"Compose this onto the object's own (unchanged) spawn
transform instead."* Not oradaydı, okuyucu tarafta yoktu.

**Kapatıldı:** `getObjectWorldTransform` artık rigid deltayı spawn transform'una
kompöze ediyor, ve sonuca **`simulated`** bayrağı ekleniyor (IPC + `rt.scene`).

★★ `simulated` süs değil: onsuz *"0 m düştü"* iki farklı şeyi aynı gösteriyor —
gövde durdu, ya da gövdeyi süren hiçbir şey yok. İkincisi bir fizik sonucu
değil, **eksik bir önkoşul.**

### ★★★ 1b. Ve rig'in kendisi bu yüzden YALAN SÖYLEDİ

İlk koşuda 1. vaka kırmızı (0 m düştü), ama **2. vaka YEŞİL**: *"100 kg'lık
gövde 1 kg'lık gibi düşüyor"*. Çünkü ikisi de **sıfır** düşmüştü ve sıfır
sıfıra eşittir.

★★★ **Hareketin tamamen yokluğuyla sağlanan bir eşitlik, hiçbir şey
kanıtlamaz.** Bu tam olarak bu dosyanın önlemek için var olduğu arıza sınıfının
kendisiydi — ve dosyanın *içinde* gerçekleşti. Her karşılaştırmalı vaka artık
önkoşulunu önce doğruluyor (`simulated`, ve referans ölçümün sıfır olmaması).

### ★★ 2. `fluid.seed` sıfır parçacık yaratıp başarı dönüyordu — KAPATILDI

İki ayrı yarısı vardı, ve **sessiz olanı ikincisiydi**:

1. Tohum bölgesi domain'e kırpılıyordu (doğru), ama örtüşme boşken çağrı yine
   `true` dönüyordu. Üç bağımsız durumda ölçüldü: domainin üstünde, altında, ve
   tam üst yüzeyden başlayan bölge.
2. Varsayılan bölge **hiçbir şeyden türetilmiyordu**: `(-0.5, 1.0, -0.5)` →
   `(0.5, 1.5, 0.5)`, hem IPC hem Python tarafında aynı sabit kutu. Bu kutuyu
   içermeyen her domain varsayılan tohumla boş kalıyordu.

İkisi birleşince bir ajan için **sessiz kayıp iş**: tohumlar, adımlar, hiçbir
şey ölçmez, ve çözücüyü suçlar.

**Kapatıldı:**

- Örtüşme boşsa çağrı **reddediliyor**, ve hata mesajı **iki kutuyu da** basıyor.
  ★ Yalnızca "bölge domain dışında" demek okuyucuyu yanlış uca baktırır — gerçek
  hata çoğu zaman domainin sanılan yerde olmamasıdır.
- `seedFluidParticles` artık bölgeyi **pointer** alıyor; `nullptr` = *"domain'den
  türet"* (tabanın bir voksel içeri, alt yarı). Sabit kutu her iki kanaldan da
  **kaldırıldı** — geriye uyum yükü yok.
- Rig'e **5b** eklendi: y 20-22'de duran bir domain'e bölgesiz tohum. Eski
  varsayılan bu domain'i ıskalıyordu. ★ Domain bilerek eski kutudan uzağa
  kondu; onu içeren bir domain seçmek her iki halde de geçerdi — sıfır düşen iki
  gövdeyi karşılaştırmakla aynı boşluk.

### ★★★★ 3. Ve asıl ders: script testi kare döngüsüne KÖR

İkinci bir fluid domain yaratmak, birincinin parçacıklarını siliyor — **ama
yalnızca IPC yolunda**:

| aynı çağrı dizisi | tohum sonrası | ikinci domain sonrası |
|---|---|---|
| **IPC'den** (aralarda uygulama döngüsü döner) | 22932 | **0** — 5/5 kayıp |
| **script içinden** (döngü hiç dönmez) | 6760 | 6760 — 2/2 korundu |

Aynı dizi, iki farklı sonuç. Silen şey `create_domain` değil, **kare
döngüsünün** yeni domain'i görünce yaptığı bir şey. Kök henüz bulunmadı.

★★★ Bunun test stratejisi için sonucu, buggan daha önemli:
**`script.run_file` ile koşan her test, kare döngüsünün yaptığı hiçbir şeyi
göremez** — script ana thread'i tutar, döngü hiç dönmez. Aynı ders
`viewport.render_frames`'in kareyi yayımlamamasında da çıkmıştı: **üretici ve
tüketici ayrı döngülerde.**

Yani bu depo iki test kanalına ihtiyaç duyuyor ve **birbirinin yerine geçmezler**:

- **script içi** — çekirdek mantığı, hızlı, döngüye kör
- **IPC'den** — uygulamayı gerçekten sürer, döngü arızalarını görür

Rig'in 6. vakası bu körlüğü **açıkça yazıyor**: yeşili "motor sağlam" değil,
"script yolu sağlam" demek.

## Ölçülen sonuçlar (2026-08-19, ikinci derleme)

| # | vaka | sonuç |
|---|---|---|
| 1 | serbest düşüş | **4.84370 m** — vaküm cevabı 4.905 m, sapma %1.25 |
| 1b | sapma AÇIKLANDI mI | **4.84370 vs 4.84470 — %0.02** |
| 2 | Galileo | 1 kg ve 100 kg **birebir aynı**: 3.11346 m, hata 0.00000 |
| 3 | yerçekimi bir DEĞER | oran **0.16513**, beklenen 0.16514 |
| 4 | kütle korunumu | 6760 → 6760, reseed +0/-0 |
| 5 | tohum dürüstlüğü | sıfır örtüşme artık **reddediliyor** |
| 5b | varsayılan bölge | y 20-22'deki domain, bölgesiz → **3200 parçacık** |
| 6 | domain izolasyonu | yeşil, ama **script yoluna özgü** |

**ALL PASSED** (2026-08-19, üçüncü derleme). ★ 6. vakanın yeşili "motor sağlam"
demek değil, "script yolu sağlam" demek — aşağıya bak.

★★★ **1b asıl vaka.** %1.25'lik sapma tam olarak "entegratör hatası" diye
geçiştirilecek büyüklükte — ve bu depo o büyüklükte geçiştirilen sayılar için
iki kez ödedi. Değil: iki terim onu tamamen açıklıyor, ve **ikisi de fit
edilmiş değil**:

1. `RigidBodySystem.h`'de **beyan edilmiş** `linear_damping = 0.05`, kapalı
   formu `y = (g/c)(t - (1 - e^{-ct})/c)`;
2. semi-implicit Euler'in yarım adım sapması, `+g·dt·t/2`.

İki farklı aralıkta (1.0 s ve 0.8 s) **%0.02**'ye kadar tutuyor. Yani artık
"yaklaşık doğru" değil, **kalanı hesaplanmış** bir çözücümüz var.

★ Bu karşılaştırma söndürme varsayılanı ya da entegratör değişirse kırmızıya
döner — ve depoda bunu söyleyecek başka hiçbir şey yok.

## Rig'in kendi ayak bağı: ad tekrarı

İkinci koşu `object not found: PV_Fall` ile çöktü — `add_primitive` başarılı
döndükten hemen sonra. Bu bilinen açık hata
([BUG_DELETED_NAME_REUSE_GHOST.md](BUG_DELETED_NAME_REUSE_GHOST.md)):
`scene.delete` yalnızca **pending-delete** işaretler, fiziksel kaldırmayı kare
döngüsü yapar — ve script ana thread'i tuttuğu için o döngü **hiç dönmez.**
Aynı adı hemen geri eklemek bir cesetle çakışıyor.

Rig artık her varlığa **koşu-benzersiz** ad veriyor. ★★ Açık bir hatanın
etrafından dolaşmak yalnızca dolaşma **görünürse** dürüst: bu süit ad tekrarını
**test etmiyor**, ve yeşil bir koşu o konuda hiçbir şey söylemiyor.

## ★★ Yan bulgu: Türkçe locale ve ONDALIK VİRGÜL — sınırı ÖLÇÜLDÜ

Yeni `fluid.seed` ret mesajı ilk halinde şöyle çıktı:
`region (-0,400 5,000 ...` — **ondalık virgülle**. Sebep `Main.cpp`'deki
`setlocale(LC_ALL, "Turkish")`: printf ailesi `LC_NUMERIC`'i izler.

Panikleyip locale'i değiştirmeden önce blast yarıçapı ölçüldü, ve **beklenenden
çok daha dar çıktı**:

| alan | durum | neden |
|---|---|---|
| JSON / **bütün IPC** | ✅ güvenli | nlohmann locale'in ondalık ayracını hem parse'ta hem serialize'da değiştiriyor (`json.hpp` 8424 / 18863) |
| `atof`/`strtod`/`sscanf` | ✅ yok | depoda tek kullanıcı json.hpp, o da locale-farkında |
| `printf %f` (33 yer) | ✅ zararsız | hepsi ImGui etiketi — görüntü, ve Türkçe arayüzde virgül zaten DOĞRU |
| **makine okunur düz metin** | ❌ tek delik | benim eklediğim ret mesajı |

★ `LC_CTYPE` korkusunu da ölçtüm, çünkü API her yerde enum string'lerini
`std::tolower` ile normalize ediyor ve Türkçe'nin noktasız ı'sı `"KINEMATIC"`→
`"kınematıc"` yapabilirdi. **Yapmıyor:** IPC'den `KINEMATIC`, `STATIC`,
`DYNAMIC` üçü de doğru çözüldü. MSVC'nin tek-baytlık CP1254 tablosu ASCII
`I`→`i` eşliyor; noktasız-ı özel durumu Unicode/ICU davranışı, CRT'de yok.

**Kural:** locale sorunu değil, **kanal** sorunu. JSON sayısı güvenli, arayüz
etiketi zaten yerelleşmeli; tehlikeli olan tek şey **script'in okuyacağı düz
metne printf ile sayı basmak.** Ret mesajı artık `std::locale::classic()` ile
imbue edilmiş bir stream kullanıyor.

## Sıradaki

1. **Derle ve 5/5b'yi koş.** 5 yeşile dönmeli (sıfır örtüşme artık reddediliyor),
   5b yeşil gelmeli (bölgesiz tohum y 20-22'deki domain'i dolduruyor).
2. 3. maddedeki kökü bul: kare döngüsünde yeni domain'i gören ve parçacıkları
   düşüren yer. **IPC yolundan** koşan ayrı bir probe gerekiyor — script bu
   hataya asla ulaşamıyor.
3. Vakaları node graph'ı olarak yeniden ifade et → diferansiyel test → sim node
   katmanının bitti tanımı.
4. Ad tekrarı hayaleti (BUG_DELETED_NAME_REUSE_GHOST) hâlâ açık; rig onu
   dolaşıyor, çözmüyor.

## Genişletmeye açık vakalar

Bu dosyaya girme ölçütü değişmedi: **bağımsız olarak bilinen bir sayı.**
Sıradaki adaylar, hepsi bu ölçüte uyuyor:

- **Kaldırma (Archimedes):** yüzen bir cismin batma derinliği = ρ_cisim/ρ_sıvı.
  Analitik, ve `scene.get_world_transform` artık ölçebiliyor.
- **Geri sıçrama (restitution):** e=0.5 ile bırakılan bir top h₁ = e²h₀'a çıkar.
- **Eğik atış menzili:** v²sin(2θ)/g — yerçekimi + başlangıç hızını birlikte sınar.
- **Açısal momentum korunumu:** tork yokken ω sabit.
- **Hidrostatik denge:** durgun bir tankın tabanındaki basınç ρgh.
