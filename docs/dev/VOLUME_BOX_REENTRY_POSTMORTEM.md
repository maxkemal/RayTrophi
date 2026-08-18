# Hacim kutusuna geri giriş — siyah bant ve maliyet patlaması

> **Durum:** REFERANS — 2026-08-16'da çözüldü. Kural hâlâ bağlayıcı: hacim
> kutusundan çıkışta ilerleme MESAFEYLE değil MASKEYLE garanti edilir.

Aylarca açık kalmış, birden fazla kez yanlış teşhis edilmiş bir arıza. Kökü tek,
belirtisi üç taneydi ve üçü farklı alt sistemlere işaret ediyor gibi görünüyordu.

## Belirtiler

1. Bir fluid domain ile bir gas domain aynı hacmi paylaşınca **domain
   sınırlarında siyah bant**.
2. Aynı anda **pathtrace maliyetinin patlaması** — ama VRAM/CPU RAM artmadan.
3. "Domainin hemen altına bir plane koyunca 50x yavaşlıyor, biraz uzaklaştırınca
   düzeliyor."

Ve iki tane sahte çözüm: domain içine **katı bir kutu** koymak, veya render
modunu **splat**'a almak. İkisi de bandı kaldırıyordu.

## Kök neden

Gaz marşı bir şey bulamayınca ışını kutu çıkışının ötesine taşıyıp yeniden iz
sürüyor. İlerleme sabit **`tFar + 0.002`** dünya birimiydi, ve `skipGasVolumes`
bu yolda `false` bırakılıyordu.

Yeni başlangıç noktası hâlâ bir kutunun içinde veya yüzeyindeyse ışın **aynı
kutuya geri girer**. İki tetikleyici:

- İki domain aynı sınırlarla yazılmışsa yüzeyler **aynı düzlemdedir**. Ölçülen
  sahnede fluid ve gas kutuları X ve Z'de birebir aynıydı, Y'de 0.0029 arayla —
  yani ilerlemeden küçük.
- Işın yüzeye **yalayarak** giriyorsa ilerlemenin yüzeye dik bileşeni ~0'dır.
  3. belirti (teğet plane) budur.

Her geri giriş raygen'in serbest geçiş bütçesinden bir tane yer ama `bounce`'ı
artırmaz. Bütçe bitince yol **hiç ışık toplamadan** kapanır — ekranda siyah. Ve
her geri giriş tam bir `traceRayEXT`'tir — maliyet. Hiçbir şey tahsis edilmediği
için bellek artmaz, bu yüzden sızıntı aramak boşunaydı.

## Ölçüm

Aynı sahne, bitişik kutular, alev yanarken:

| | sıkışmış | düzeltilmiş |
|---|---:|---:|
| `volume_rays` | 419.661.625 | 72.190.640 |
| `density_samples / volume_rays` | 0,0196 | 0,5594 |
| `layered_handoffs` | 0 | 2.213 |
| `temporal_accepted` | 0 | 19.250 |

## Çare

Marş segmenti **şeffaf** geçtiyse (`volumeOpacity <= 0.04 &&
volumeContribution <= 5e-4`) ışın bir sonraki izde o gaz kutusuna geri
sokulmaz — `payload.skipGasVolumes = true`. Şeffaflık kapısı bunu görsel olarak
etkisiz kılar: ışın o gazı zaten göremediğini ölçmüştür.

Kod: `volume_closesthit.rchit`, teleport dalı. Uzun postmortem yorumu orada.

### ★★★ Maskenin İLK hâli fazla genişti — üretilen regresyon

Maske önce yalnızca `transparentSegment` koşuluna bağlandı. Bu, yazıldığı
arızadan **çok daha geniş**: alevin kenarları, ince duman ve sönmekte olan
bölgeler meşru olarak şeffaftır. Sonuç: neredeyse her ince segmentten sonra gaz
bir iz boyunca kapatıldı ve gaz **çok segmentli birikimini kaybetti**.

Belirti: örtüşen bir fluid domain içinde gaz, kutu düzleminde **ince bir kabuğa**
çöktü; altta ve üstte farklı görüntü oluştu. Örtüşme bölgesi, hakemin `tFar`'ı
kırptığı yer olduğu için yanlış kapı orada en sert vurdu.

**Ayıran ölçüt ŞEFFAFLIK DEĞİL, DEJENERE ARALIKTIR:**

| | sıkışmış ışın | meşru ince gaz |
|---|---|---|
| kat edilen mesafe | bir voksel bile değil (çakışık kutunun çıkış yüzeyine düşmüş) | kutuyu **baştan sona** geçer |
| yapılan iş | yok | az yoğunluk bulur |

Kapı artık **ikisini birden** istiyor: `degenerateSpan && transparentSegment`.
Ve aralık, hakem/solid probe tarafından kırpılmış `tNear/tFar` yerine
intersection shader'ın ham `volumeHitAttrib`'inden okunur — kırpılmış bir
aralık, sağlıklı bir tam geçişi dejenere gösterirdi.

★ Ders: bir kapıyı, düzeltmeye çalıştığın arızanın **imzasına** bağla; arızayla
birlikte görülen ama ondan çok daha yaygın bir özelliğe değil.

### ★★★ Epsilon'u SİLME — maskenin kapsamadığı bir aralık var

Maske yalnızca **şeffaf** segmentte kurulur. Gaz orta yoğunluktaysa
(`volumeOpacity > 0.04` ama saçılma olayı örneklenmemişse) `skipGasVolumes`
kurulmaz ve ışın **yalnızca mesafeyle** ilerler. O aralıkta tek koruma
`exitEpsilon`'dur.

Yani epsilon **ölü bir yol değil**, yetersiz olduğu anlaşılan bir yoldur —
ikisi farklı şeydir ve "ölü yolu sök" kuralı buna uygulanmaz. Alev sönerken
sahne tam bu aralıktan geçer.

Zararı düşük: çeyrek voksel, yani hacmin temsil edebildiği en küçük özellikten
küçük; atlayabileceği görünür içerik yok.

## Bir daha düşmemek için — kurallar

★★★★★ **İlerleme mesafeyle garanti edilmez.** Önce epsilon `0.002` →
`voxel_size * 0.25` yapıldı. Bant **kaybolmadı, içeri taşındı**. Sabit bir
mesafeyi bilinmeyen bir boşluğa karşı yarıştırmak; her zaman daha dar bir
çakışma bulunur. Epsilon kodda duruyor ama tek başına çare değildir.

★★★★ **İlerlemesi sıfır olan döngü bütçeye duyarsızdır.** "64 bounce'da da aynı"
sonucu, pass-bütçesi tezini eleyen kanıt sanıldı. Elemedi — bütçeyi büyütmek
yalnızca daha çok israf üretir. Bir teoriyi elemeden önce, testin o teoriyi
gerçekten ayırt edip etmediğini sor.

★★★★ **Sağlıklı durumda DA ölç.** `arbiter_no_crossing` patolojik durumda %99,99
okuyordu ve saatlerce suçlu sanıldı — sağlıklı durumda da %99,99 okuyor. Bir
sayacın bozuk durumda yüksek olması onu ayırt edici yapmaz. Ayıran sayı
`density_samples / volume_rays` idi.

★★★ **Kaçış çare değildir.** Katı kutu (solid probe ışını gerçek geometriye
devrediyor) ve splat modu (ikinci çakışık kutu yok) semptomu gizledi ve teşhisi
çok geciktirdi. "Şunu yapınca düzeliyor" bir çözüm değil, bir **ipucudur**.

★★ **A/B'de tek değişken.** Kutu sınırı değiştirilince sim de sıfırlandı;
"ateş var/yok" ile "çakışık/ayrık" iç içe girdi ve 10.000x'lik yanlış bir sonuç
çıkarıldı. Alev varken tekrar ölçmek düzeltti.

★★ **Aynı sınırlarla iki domain yazmak NORMAL kullanımdır.** Yanan bir sıvı tam
olarak böyle kurulur. Çakışık kutuyu "kullanıcı hatası" saymak bu arızayı
kalıcılaştırırdı.

★ **Teşhis eşiği:** hacim yolunda sıkışma şüphesi varsa önce
`density_samples / volume_rays` bak. ~0,5+ sağlıklı; ~0,02 ve altı ışın giriyor
ama iş yapmıyor demektir.

## Açık kalan

Belirti 3 (teğet plane, "kamerayı geri çekince düzeliyor") aynı kök olarak
**ölçülmedi**, yalnızca mekanizma eşleşmesine dayanıyor. Tekrarlarsa v8 volume
metrics dump'ı al ve yukarıdaki eşiğe bak.
