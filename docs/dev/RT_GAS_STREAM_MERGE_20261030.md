# Gaz akışlarını render'da birleştirme

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
- [ ] Açılışta birleşmeme sorunu

Nasıl çalışır: `gasMergeKey` görünür ve uyumlu gaz akışlarını sınar, `mergeGasFields` her akışı kendi
yoğunluk yanıtından geçirip toplar. Tek akış nötr gölgelendiriciyle (eğri kapalı, çarpan 1, cutoff 0)
yayınlanır, diğerleri mevcut emeklilik yoluyla kalkar, görünümleri domain'de saklı kalır.

Bilinen sınırlar:
- Yanıt voksel başına işlenir, shader ise trilineer değerde işliyordu: cutoff kenarı hafif değişir.
  Ayrıca `edge_falloff`, görünürlük ve `density_scale` farkları için birleştirme denetimi yalnız
  kimlik JSON'una bakar; `density_scale` ayrıca denetlenmiyor.
- Emisyon, saçılma veya soğurması farklı akışlar birleşmez, karışım yolunda kalır. Aşama 2 bunu kaldırır.
  Canlı sahnede akışlar yalnız `density_curve` ve `density_multiplier` farkındaydı, yani uyumluydu.
- Panelden doğrudan `VDBVolume` gölgelendiricisi düzenlenirse nötr kopya değişir ve sonraki
  yayında kaybolur. Malzeme `matter:<domain>:gas*` hedefleriyle düzenlenmeli; panel yolu doğrulanmadı.

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
