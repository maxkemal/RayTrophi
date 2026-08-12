# Yangın bir yapıyı yıkar — elle kurulum tarifi

Su kulesi senaryosu: ahşap bir yapı tutuşur, yandıkça zayıflar, ve kendi
yangınının basıncı onu yıkar. Hiçbir script gerekmez; aşağıdakilerin hepsi
panellerden kurulur.

Zincir şu, ve her halkası ayrı bir panelde yaşıyor:

    tutuşma → MSF malzeme kaybı → integrity düşer → fracture eşiği düşer
            → yanma aşırı basıncı → blast → vurulan küme kopar

★ Bir halka eksikse **hiçbir hata görmezsin**. Yapı yanar, kararır ve dimdik
durur. Aşağıdaki her adımın sonunda "ne görmelisin" yazıyor; atlama.

---

## 1. Gaz domain'i

Simulation panelinden bir **Gas** domain ekle, yapının yanmasını istediğin
bölümü kapsasın.

- **Fuel Grid (Combustion/Fire)** kanalını aç. Yakıt kanalı yoksa yanma yok.
- **Combustion & Fire Physics** → *Enable Combustion Physics*.
- `Ignition Temperature`, `Burn Rate`, `Heat Release` varsayılanlarıyla başla.

> Domain'i yapının TAMAMINI kapsayacak kadar büyük yapma. Her yeri eşit yakar,
> ve o zaman "yangının vurduğu yer koptu" ile "obje tümden gitti" arasındaki
> farkı ne sen görebilirsin ne de test.

## 2. Tutuşturucu — Flow Source

Aynı panelden bir **Flow Source** ekle, domain'e bağla.

- `Temperature` tutuşma sıcaklığının belirgin üstünde,
- `Combustion Fuel Rate` > 0.

**Görmen gereken:** oynatınca alev ve duman. Yoksa yakıt/sıcaklık yetersiz veya
Fuel kanalı kapalı.

## 3. Objeyi yanıcı yap — Collider + MSF

Objeyi seç, **Collider** oluştur:

- `Gas Interaction` açık,
- `Ignite on Contact` açık,
- `MSF Substance` = **Wood (Oak)** (ya da uygun malzeme),
- `MSF Mask Resolution` 96 civarı yeter.

Bu adım objeye bir **malzeme durum alanı** verir: yüzeyin her texel'i piroliz,
erime ve transferle ne kadar kütle kaybettiğini ayrı ayrı bilir.

**Görmen gereken:** yandıkça objede kararma (char) ve MSF panelinde
`mean_integrity`'nin 1.0'dan aşağı düşmesi. **Düşmüyorsa durup burayı çöz** —
sonraki her şey bu sayıya dayanıyor.

## 4. Parçalanmayı hazırla — Fracture

Objeyi seç, Physics → **Fracture (Destruction)**:

- `Shards` = 40–50,
- `Pattern` = **Thermal (burn-guided)** — tohumlar yanan bölgede yoğunlaşır,
  yani obje char hattından kırılır, rastgele yerden değil,
- `Structural Clusters` = 6 (ya da yapının kaç bağımsız parçaya ayrılmasını
  istiyorsan),
- `Exact Surface` açık (boşluklar, UV'ler ve malzemeler korunur),
- **Generate Shards**.

★ Bunu yangın BİR MİKTAR İLERLEDİKTEN sonra yap. Thermal desen o anki hasarı
okur; hiç yanmamış objede uniform'a düşer ve termal tohumlamanın anlamı kalmaz.

Sonra `Break Toughness` ver (varsayılan 5 m/s iyi bir başlangıç) ve
**Make Breakable**. Panel hemen altında grubun kütlesini ve bunun karşılık
geldiği N·s eşiğini yazar — asıl bakman gereken sayı odur.

**Görmen gereken:** "N clusters" mesajı ve `Registered Bodies` listesinde grup
satırı.

## 5. Son halka — blast kuplajı

Gaz domain'ine dön, Combustion & Fire Physics'in altında:

- **Blast Damages Structures** → aç.

Bu kutu kapalıyken yangın malzemeyi zayıflatır, eşiği düşürür ve **yükü asla
teslim etmez**. Yapı kömürleşir ve ayakta kalır. Uzun süre böyleydi.

---

## Kuvveti ayarlamak (senin "fazla kuvvet" durumun)

Üç knob var, üçü farklı soruya bakıyor:

| Knob | Ne yapar | Nasıl ayarlanır |
|---|---|---|
| `Blast Pressure Scale` | Yanma yoğunluğunu kPa'ya çevirir | **Asıl knob bu.** Yarıya indir, yeniden dene |
| `Break Toughness` (fracture) | Kümenin dayandığı **hız değişimi** (m/s) | 1–2 kırılgan · 5 sıradan · 20+ sağlam |
| `Minimum Blast Intensity` | Altında yangın yük bindirmez | Küçük alevler yapıyı sarsıyorsa yükselt |

★ **Break Toughness bir impulse DEĞİL, hızdır.** Eşik = bu değer × kümenin
kütlesi, ve panel sliderın altında ikisini de yazar. Aynı N·s bir tahta için
yıkım, bir kule ayağı için dokunuştur; o yüzden impulse eşiği her obje için
ayrı ayarlanmak zorundaydı ve hiçbiri taşınmıyordu. Hız her iki ölçekte de aynı
şeyi ifade ediyor. Karşılaştırmayı **panelin yazdığı N·s ile** yap, toughness
değeriyle değil.

★ **Pressure Scale bir KALİBRASYONDUR, fizik sabiti değil.** Bu çözücüde yakıt ve
sıcaklık normalize birimdir; onları kilopaskala çeviren dürüst bir formül yok.
Fizikmiş gibi bir sayı türetmek, fiziksel görünen ama hiçbir şeye hesap vermeyen
bir değer üretirdi. Doğru yöntem: yapının kırılmasını istediğin ana kadar
yükselt/indir.

**Ayar döngüsü:** `Break Toughness`'ı sabit tut, `Blast Pressure Scale` ile oyna.
Ters yönde çalışmak daha zordur, çünkü eşik aynı zamanda temas kırılmalarını da
etkiler.

Fazla kuvvet belirtisi: yangın başlar başlamaz yapının TAMAMI dağılır. Doğru
ayarın belirtisi: alevin olduğu uçtaki kümeler kopar, uzak uç ayakta kalır.

## Yıkılma hissi

Bir su kulesinin çökmesi tek bir blast değildir: ayak zayıflar, kopar, sonra
ağırlık gerisini yapar. Onun için:

- `Blast Interval`'ı çok küçültme (0.25 s makul) — yangın ardışık darbeler verir,
- kuleyi bir seferde yıkacak kadar kuvvet verme; **taşıyıcı ayağı** kıracak kadar
  yeter, gerisini yerçekimi halleder,
- `Structural Clusters`'ı ayaklar ayrı kümelere düşecek şekilde seç.

## Kontrol listesi — yanıyor ama yıkılmıyor

Sırayla bak, ilk "hayır"da dur:

1. MSF `mean_integrity` 1.0'ın altına düşüyor mu? → Hayırsa: collider'da MSF
   substance/ignite ayarları.
2. Fracture grubu `integrity_regional` = true mu? → Hayırsa: küme bölgesinde MSF
   elemanı yok, `source_object` eşleşmiyor olabilir.
3. `Blast Damages Structures` açık mı?
4. Blast olayı üretiliyor mu? → SceneLog'da yapısal impulse sayaçları; `queued`
   artmıyorsa `Minimum Blast Intensity` fazla yüksek ya da yangın zayıf.
5. `queued` artıyor ama `consumed` artmıyor mu? → tüketici pompalanmıyor
   (bu bir kod hatasıdır, ayar değil).
6. Impulse eşikten küçük mü? → Panelin yazdığı N·s eşiğine bak. Küme beklediğinden
   ağırsa sorun toughness değil **kütle**: shard'lar `density`'den otomatik
   ağırlık alıyor, ve yanlış bir density eşiği sessizce onlarca kat büyütür.
