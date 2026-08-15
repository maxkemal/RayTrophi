# Malzeme koordinatı (UVW) — kalite notları

> **Durum:** ARSIV — 2026-08-13'te kapandi; kalan fark yontemin tanimi.

> ## ✅ KAPANDI — 2026-08-13. Kalan fark BUG DEĞİL, yöntemin tanımı.
>
> **Material, Domain/World ile aynı düzgünlükte OLAMAZ ve olmamalı.** Domain ve
> World sabit bir uzaya çakılıdır, dolayısıyla haritaları hiç bozulmaz. Material
> maddeye bağlıdır; madde gerilir, yayılır, damlalara ayrılır — ve haritanın
> onunla deforme olması **istenen davranışın ta kendisidir**. Dokuyu sıvıyla
> taşıyan mekanizma ile haritayı bozan mekanizma aynı mekanizmadır.
>
> ★ Yani "Domain kalitesinde **ve** akışı takip eden koordinat" isteği kendi
> içinde çelişiktir. Ulaşılabilir hedef gerilmeyi **sınırlamaktı**; iki kuşaklı
> yenileme onu yapıyor ve sonuç tatmin edici bulundu.
>
> **Bu dosyayı yeniden açmadan önce:** aşağıdaki üç kök neden (mutlak saklama,
> quintic ease'in gradyanı modüle etmesi, ızgara adresleme) çözüldü ve kaydı
> aşağıda. Kalan görünüm farkı bunlardan biri DEĞİL. Yeni bir kazı ancak
> gerilmeden bağımsız, tarif edilebilir bir semptom varsa anlamlıdır.
>
> Hâlâ açık olan tek gerçek kusur: **tri-planar ayrışması** (aşağıda) — ki o
> gerilmeyle ilgisiz ve üç modun hepsini eşit etkiler.


Durum: 2026-08-13. Normal map bağlandı ve **Domain / World** modunda doğru
çalışıyor. **Material** modunda kalite yetersizdi.

> **UYGULANDI (aynı gün).** Artık saklama kuruldu; aşağıdaki teşhis kaydı
> olduğu gibi duruyor. İki düzeltme birden çıktı ve ikincisi ilkinden büyük:
>
> 1. **Artık saklama.** Grid `mean(uvw − position)` tutuyor, tüketici
>    `worldPos + trilinear(d)` kuruyor. Alan adı `uvw_residual_*` oldu ki eski
>    anlama göre yazılmış bir tüketici **derlenmesin**.
> 2. **★★★ Quintic ease koordinatın kendisini bozuyormuş.** Dikiş gidersin diye
>    konan yumuşatma, MUTLAK koordinata uygulanınca kimlik gradyanını modüle
>    ediyordu: `d(koordinat)/d(dünya) = s'(f)`, hücre yüzünde **0.0**, hücre
>    merkezinde **1.875**. Yani doku her hücre sınırında donuyor, ortasında
>    sıkışıyor — hücre başına tam bir karo. **Birinci görüntüdeki kare-dalga
>    deseni budur.** Dikişi silmek için konan araç, dikişin kendisiydi.
>    Artık alanda aynı ease zararsız: kimlik gradyanı bu tampondan geçmiyor.
>
> 3. **★★★ IZGARA YANLIŞ BÖLGEYE SERİLİYORDU (asıl "warp").** Tüketici alanı
>    `inv_transform` + `aabb_min/extent` ile adresliyordu — yoğunluk
>    örnekleyicisine benzetilerek. O benzetme yanlış: canlı sıvıda `aabb`,
>    **dense/SDF ızgarasının dar AKTİF kutusu**dur (bir hücre paylı, her kare
>    yeniden hesaplanır), oysa bu tampon **tüm SİM ızgarasını** sim
>    çözünürlüğünde kaplar. Biri diğerine eşlenince alan, kutuların oranı kadar
>    **gerilir**, originler farkı kadar **kayar**, ve aktif kutu sıvıyı takip
>    ettiği için **her kare değişir**. Düzeltme: ızgaranın kendi `uvw_origin` +
>    `uvw_voxel` değerleri yayınlanıyor ve arama **dünya uzayında** yapılıyor.
>    ABI 608 → **624**.
>
>    ★★★ **Bu hatayı durgun-kimlik testi YAKALAYAMAZ** ve yakalamadı: durgunda
>    `d ≡ 0`, sıfır bir alana yanlış adresten bakmak da sıfır verir. Kullanıcının
>    "durgunda Domain ile neredeyse aynı, **akışta** bozuluyor" gözlemi teşhisi
>    veren şey oldu — hata `d` ile orantılı. Artık render'sız yakalanıyor
>    (`phase_numeric` ızgara yerleşimini doğruluyor).
>
> Ek: gather'da fark **parçacık başına** alınıyor (`uvw_i − p_i`), sonradan
> hücre merkezi çıkarılarak değil. İkisi aynı şey değil — ikincisi parçacık
> dağılımının ağırlık merkezini geride bırakır. Bu haliyle durgunda `d ≡ 0`
> **tam olarak**, dağılım ne olursa olsun; test bunun üzerine kuruldu
> (`phase_still_identity`).

---

## Sahadan gelen tarif (kullanıcı gözlemi, birebir)

- Material modu hâlâ **"her hücreye bir pixel gibi"** kaba, ve **sarılıyor**
- **Yüzeyler ayrışmadan projekte edilmeli**
- **Domain koordinatı çok kaliteli**, ama tabii akışı takip etmiyor —
  **tek kare için mükemmel**

★ Son madde teşhisin anahtarı. "Domain mükemmel ama akmıyor" ile "material akıyor
ama kaba" **aynı problemin iki yarısı**: biri çözünürlüğü, diğeri hareketi
veriyor, ve şu an ikisini birden veren bir temsil yok.

---

## ★★★ KÖK: koordinatın TAMAMI grid'de saklanıyor

Şu anki temsil, her hücrede mutlak `uvw` değerini tutuyor. Dolayısıyla grid'in
çözünürlüğü **koordinatın tamamını** sınırlıyor — sim voksel boyutu (tipik 5 cm)
doğrudan dokunun efektif çözünürlüğü oluyor. "Her hücreye bir pixel" bunun
tarifi.

Oysa koordinat iki bileşene ayrılıyor ve **frekansları taban tabana zıt**:

```
uvw(x) = x + d(x)
         ^   ^
         |   yer değiştirme: DÜŞÜK frekanslı, gerçekten grid'e ait
         pozisyon: TAM çözünürlüklü, sürekli, ve shader'da zaten bedava var
```

Mutlak değeri saklamak, tam çözünürlüklü olan yarıyı da grid'in kabalığına
mahkûm ediyor. **Artık (residual) saklamak** onu kurtarıyor.

---

## Sıradaki iş: artık saklama (residual UVW)

**Üretici** (`buildMaterialCoordinateGrid`): hücreye `uvw_gathered − p_hücre_merkezi`
yaz, mutlak `uvw` yerine.

**Tüketici** (`sampleMaterialCoord`): `return worldPos + trilinear(d);`

Kazanç:
- Birim eşleme **tam olarak** korunuyor — dünyada 1 mm, koordinatta 1 mm, hücre
  içi dâhil. Kabalık yalnızca deformasyona kalıyor, ki o gerçekten yumuşak.
- **Durgun sıvıda `d ≡ 0`**, yani Material modu Domain/World ile **bit düzeyinde
  aynı** kaliteye çıkıyor. Kullanıcının "domain çok kaliteli" dediği kaliteyi
  Material modu doğrudan devralıyor.
- Bellek ve format aynı: üç float, aynı tampon. `SimCache` etkilenmiyor
  (o partikül başına `uvw` tutuyor, değişmiyor).

> **★ Neden bu düzeltme öncekinden farklı.** Bir önceki tur ekstrapolasyonun
> gradyanı sıfırlamasını düzeltti — o, alanın **desteklenen bölge dışında**
> bozulmasıydı. Bu ise alanın **her yerde** çözünürlükle sınırlı olması. Aynı
> semptomu ("kaba, sarılıyor") iki farklı sebep üretiyordu; birincisi
> damlalarda, ikincisi her yerde.

**Doğrulama:** durgun bir tank, aynı normal map, üç mod. Material ile Domain
**ayırt edilemez** olmalı. Ayırt edilebiliyorsa artık saklama bir yerde
kaçmıştır — o ölçüm, kalite tartışmasını yargı olmaktan çıkarır.

---

## Ayrı konu: "yüzeyler ayrışmadan projekte edilmeli"

Tri-planar üç projeksiyonu harmanlıyor. Whiteout blend dikişi büyük ölçüde
gizliyor ama **ayrışmayı ortadan kaldırmıyor** — eğik yüzeylerde üç projeksiyon
farklı yönde gerilir ve desen bölünür.

Bu, çözünürlükten **bağımsız** bir konu ve ayrı çözümü var:

1. **Bi-planar** (en iyi iki eksen, üçüncüsü atılır) — dikiş sayısı üçten ikiye
   düşer, maliyet de düşer. Ucuz iyileştirme.
2. **Stokastik / dithered projeksiyon seçimi** — örnek başına tek düzlem seç,
   path tracer zaten ortalıyor. Dikiş harmanlanma yerine gürültüye dönüşür ve
   yakınsayınca kaybolur. Bu render'a çok uyuyor.
3. **Gerçek yüzey parametrizasyonu** — izoyüzey için pratik değil, kaydediliyor
   ki bir daha değerlendirilmesin.

★ (2) bu projeye özellikle uygun: tri-planar'ın harman maliyeti üç doku
okumasıyken, stokastik seçim **tek** okuma yapar. Yani hem dikişi kaldırır hem
normal map'i ucuzlatır — ki normal map şu an vuruş başına üç okuma yapıyor.

---

## Hâlâ açık, ve karıştırılmaması gereken

Bu üç şey benzer görünüp farklı sebeplerden gelir:

| Görüntü | Sebep | Durum |
|---|---|---|
| Damla etrafında kıvrılan girdap | Ekstrapolasyon gradyanı sıfırlıyordu | **Düzeltildi** (offset taşıma) |
| Her yerde hücre-ölçekli kabalık | Koordinatın tamamı grid'de **+ quintic ease'in kimlik gradyanını modüle etmesi** | **Düzeltildi** — artık saklama |
| Eğik yüzeyde desenin bölünmesi | Tri-planar ayrışması | **Açık** — bi-planar / stokastik |
| Uzun dökülmede yön boyunca şeritlenme | **Gerilme** — bug değil | Açık: iki-tohumlu blend |
