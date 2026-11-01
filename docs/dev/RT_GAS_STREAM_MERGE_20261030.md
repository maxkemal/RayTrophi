# Gaz akışlarını render'da birleştirme

## Panel bağı ve ilk yayın incelemesi (2026-10-31, derlenmedi)

- Önceki `getShader()` ayrımı tek başına yeterli değildi: VDB panelinin asıl malzeme
  bölümü `getOrCreateShader()` çağırıyordu ve birleşik gazın nötr render kopyasını
  düzenliyordu. Bu getter da artık domain'deki yazılı malzemeyi döndürür.
- `setPublishedShader` canlı alanın yazılı pointer'ını ayrı tutar; `setShader` ataması
  bu malzemeyi yerinde günceller. Böylece sonraki domain yayını ayarı geri almaz.
  Panel hangi kaynağın düzenlendiğini ve optik değişimin birleşmeyi ayırabileceğini açıklar.
- İlk yayın kodunda kırpma zaten `registerOrUpdateLiveVolume` öncesindedir; cutoff'un
  önce bir kez değiştirilmesini gerektiren bir dal yoktur. Bu kırpma önceki notta
  henüz derlenmemişti. Eski binary veya bayat yayın kutusu ile kaynak davranışı ayrılmalıdır.
  İnce gaz simülasyonda korunur; bu işlem yalnız render kopyasını kırpar, snapshot'ın
  yoğun grid taramasını veya simülasyon maliyetini kaldırmaz.
- Kırpma sigma ve emisyon türüne de bağlıyken optik anahtar bunları izlemiyordu.
  Saçılma, soğurma ve emisyon türü anahtara eklendi. İlk yayın/görsel kabul canlıda açık.
- 8³ blok için τ=0.01 sezgisel bir eştir; birçok ince blok boyunca biriken toplam
  katkının veya eğik ışının görünmezliğine matematiksel garanti değildir. Görsel katkının
  sıfır olduğu varsayımı canlı A/B ile doğrulanmalıdır.

> **Durum:** AKTİF — aşama 1 canlı: aynı süreçte 262 → ~135 ms; görüntü kabulü ve açılış tetikleme sorunu açık.

★ **DEVİR NOTU (2026-10-30):** Kök ölçüldü (`RT_GAS_OVERLAP_COST_20261030.md` §11.1): aynı toplam
gaz tek akışta ~131 ms, iki akışta 240–280 ms. Sim tarafı akışları korur, sorun render'da her
akışın tam domain boyutunda ayrı hacim olmasıdır. Kodun durumu için aşağıdaki "İlerleme"ye bak.

## Hedef

Çoklu kaynaklı gaz, görüntüsü aynı kaldığı sürece render'a TEK hacim olarak gider. March, boş alan
atlama, gölge, ışık önbelleği, radyans önbelleği ve sekme tek gaz yolunu kullanır; karışım yolu
(`shadeFogMixture`) çalışmaz.

## Değişmezler

- Sim akışları ve `fluid.inspect_fog_fields` değişmez. Birleştirme yalnız render yayınıdır.
- Toplam yoğunluk toplanır, sıcaklık zaten ortak tek alandır.
- Yoğunluk eğrisi ve çarpan alan başınadır: ham toplam alınmaz. Önce her akış kendi eğrisinden geçer,
  sonra toplanır. Aksi halde doygun eğride görüntü değişir (`f(a)+f(b) != f(a+b)`).
- Malzemeler optik olarak uyumlu değilse birleştirme YAPILMAZ, karışım yolu çalışır. Sessiz görüntü
  değişimi kabul edilmez.

## Aşamalar

1. **Uyumlu akış:** saçılma, soğurma, emisyon ve gölge ayarları aynı, yalnız yoğunluk eğrisi, çarpan
   ve cutoff farklıysa birleştir. Bu ayarlar yayında yoğunluğa işlenir.
2. **Farklı emisyon:** akış oranı alanı eklenir, malzeme farkı yalnız dolu örnekte harmanlanır.
   Yeni grid ve shader bağı ister, ayrı parti.

## Ölçüm

Küplü sahnede aynı süreçte kıyas mümkün değil, açılış alanları değiştirir. Oran kullan: gaz gizli
süresine göre, küresel ~%25 bant nedeniyle aynı banttaki değerlerle.

## İlerleme

- [x] Yayın yolu okundu (`SceneData::syncMatterFogStreams`)
- [x] Aşama 1 kodu: `MatterFogPublisher.cpp`, `MatterFogStreams.h` (derlenmedi)
- [x] Derleme ve ölçüm (aşağıda)
- [ ] Görsel kabul
- [x] Açılışta birleşmeme / panel düzenlemesi: `matterFogMergeChanged` (derlenmedi)

Nasıl çalışır: `gasMergeKey` görünür ve uyumlu gaz akışlarını sınar, `mergeGasFields` her akışı kendi
yoğunluk yanıtından geçirip toplar. Tek akış nötr gölgelendiriciyle (eğri kapalı, çarpan 1, cutoff 0)
yayınlanır, diğerleri mevcut emeklilik yoluyla kalkar, görünümleri domain'de saklı kalır.

Bilinen sınırlar:
- Yanıt voksel başına işlenir, shader ise trilineer değerde işliyordu: cutoff kenarı hafif değişir.
  Ayrıca `edge_falloff`, görünürlük ve `density_scale` farkları için birleştirme denetimi yalnız
  kimlik JSON'una bakar; `density_scale` ayrıca denetlenmiyor.
- Emisyon, saçılma veya soğurması farklı akışlar birleşmez, karışım yolunda kalır. Aşama 2 bunu kaldırır.
  Canlı sahnede akışlar yalnız `density_curve` ve `density_multiplier` farkındaydı, yani uyumluydu.
- ★ Regresyon, düzeltildi (derlenmedi): panel ve `vdb:N` nötr kopyayı düzenliyor, yeni kare ayarı
  sıfırlıyordu; panel de çarpan 1 gösteriyordu. `VDBVolume` artık yazılı malzemeyi ayrıca tutar:
  `getShader` onu, render `volume_shader` ile nötr kopyayı okur. Bir akışın yalnız renk/emisyonu
  düzenlenirse akışlar uyumsuzlaşır ve birleşme bozulur: bu doğru davranıştır, maliyet geri gelir.

## Canlı sonuç (2026-10-31)

Aynı süreç, aynı alanlar, küplü sahne, frame 99, sample 8–23 medyanı:

| Durum | GPU ms | Gaz slotu |
| --- | ---: | ---: |
| İki akış (başlangıç) | 262.4, 299.7, 251.1 | 2 |
| Birleşik | 131.6, 138.6, 149.4 | 1 |
| B gizli | 132.8 | 1 |
| B yeniden görünür | 134.8 (birleşik) | 1 |
| Uyumsuz (B saçılma +0.01) | 262.4 | 2 |

~%48 düşüş, ilk kez aynı alanlarla doğrudan kıyas. Geçişler çalışıyor, uyumsuzluk iki slota dönüyor.
Görüntü (128 örnek, 8-bit JPEG, birleşik vs iki akış): ortalama fark 0.639, p99 19, en çok 111/255;
ortalama parlaklık 200.24 vs 199.67, gaz bölgesi ortalaması 0.884. Küçük değil: p99 19 görünür olabilir,
gözle kabul gerekiyor. İki akış karşılaştırması B saçılma +0.01 ile alındı, fark kaba bir üst sınırdır.

★ Açık: açılışta üç slot kalıyor, malzemeler uyumlu olsa da birleşme ilk yeniden yayında tetiklendi
(boş `volume.set_material` yetti). Oynatmada her kare yayın olduğu için büyük olasılıkla kendiliğinden
düzelir, doğrulanmadı. Muhtemel neden: duraklatılmış açılışta yayın kapısı; incelenmedi.

### Açık sorunlar (2026-10-31)

- Duraklatılmışken panel düzenlemesi yayın istemiyordu; açılıştaki gecikmenin de kökü buydu. Düzeltildi,
  derlenmedi.
- `nuke` dışı sahnede Solid + Play: `gas inventory backend switch readback failed` istisnası
  (`GasInventoryDevice.cpp:575`, `ParticleDeviceResidency.cpp:327`) uygulamayı kapatıyor. Birleştirme
  bu yola girmiyor, ama hangi işle geldiği araştırılmadı. Bekleyen cihaz birikimi şüpheli.
- Aynı denemede kayıtta splat beklenirken SDF su çizildiği bildirildi, incelenmedi.
- "Ayrı akış 100 ms, gizle/göster sonrası pahalı" gözlemi yeniden üretilemedi. Kare 54'te iki slot:
  128.8/148.6 ms, gizli 95.2/95.4, yeniden görünür 120.7/120.1 ms; sayaçlar gizle/göster sonrası birebir
  aynı (kamera 6.058). Maliyet gizle/göster ile değil kareyle değişiyor: kare 37'de kamera örneği 25.96,
  190–214 ms. Kutular kare 54'te yalnız kısmen örtüşüyor (x 3.26–4.65), kare 37 kutuları kaydedilmedi.
  Ayrıca kutular farklı olduğu için `mfSharedWalk` burada çalışmaz.

## İnce buhar: maliyetin ikinci büyük kökü (2026-10-31)

`Physics Domain 1`, frame 52, B gizli, tek gaz slotu. A: 54248 aktif hücre, yoğunluk toplamı 172;
B: 5679 hücre, 297. Üç sıvı kaynağının buharı A'ya gidiyor, domain'e ince pus olarak yayılıyor.

| A `density_cutoff` | kutu üstü y | GPU ms | kamera örneği/yol | sekme yürüyüşü |
| ---: | ---: | ---: | ---: | ---: |
| 0.01 (taban) | 4.19 | 209–240 | 30.83 | 0.647 |
| 0.02 | 4.19 | 179–212 | 24.37 | 0.613 |
| 0.05 | 0.93 | 78–80 | 3.62 | 0.217 |
| 0.1 | 0.93 | 72 | 3.13 | 0.195 |
| 0.2 | 0.47 | 55 | 1.57 | 0.130 |

0.02 → 0.05 arasında kamera örneği 7 kat, süre 2.5 kat düşüyor. Kullanıcının "ucuz" durumundaki kutu
üstü de 0.93'tü. Yani ucuz durum büyük olasılıkla ince pusun çizilmediği durumdur. UI'dan gizleyince neden
geçiş olduğu ve o anda hangi cutoff'un etkin olduğu kanıtlanmadı. Cutoff bir görüntü ayarıdır, görsel kıyas
yapılmadı. Öneri: sabit kullanıcı değeri yerine görünmez optik derinliğe göre kırpma, tasarımı yazılmadı.

### Yazıldı (derlenmedi)

- Görünmez blok kırpma: 8³ blokta hiçbir eksen sütunu τ=0.01'e ulaşmıyorsa blok yayında sıfırlanır.
  Emissive akışlara uygulanmaz, sim verisi değişmez. Eşik tek sabit, ölçümle ayarlanacak.
- Durum: panelde "Gas render: ..." satırı, IPC'de `fluid.inspect_fog_fields.gas_render_merge`.
- Solid + Play çökmesi: arka uç değişiminde geri okuma başarısızsa domain "reset required" ile bekler,
  uygulama kapanmaz. Okumanın NEDEN başarısız olduğu hâlâ bilinmiyor, bu kök düzeltme değildir.

### Kullanıcının yakaladığı kök: bayat kutu

Panelde cutoff değişince gaz kutusu küçülmüyordu, ancak gizle/göster yayın zorlayınca değer etkili oluyordu.
Kutu yalnız yayında kesilir, duraklatılmış sahnede panel yayın istemez. "Ucuz ↔ pahalı" zıplamaları büyük
olasılıkla budur: kutu hangi cutoff ile son yayınlandıysa onu taşıyordu. `matterFogMergeChanged` artık birleşme
anahtarı ile optik kutu anahtarını birlikte izler (derlenmedi). Yeni kırpmanın eşiği cutoff'a bağlıdır, aynı yolla yenilenir.
