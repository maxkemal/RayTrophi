# Postmortem: CPU seçimi — `TriangleMesh::hit` ile `EmbreeBVH` farklı tampon okuyordu

> **Durum:** ARŞİV — 2026-09-16 çözüldü ve ölçümle doğrulandı.

## Değişmez (bunu bir daha kırmayın)
**Facade'sız (flat SoA) bir mesh için yerel konum kaynağı `P_orig`'dir, `"P"`
değil.** `"P"` ayrı basılan bir önbellektir ve yalnızca mesh'in transform'u
değiştiğinde **güvenilir biçimde yeniden basılmaz** — flat mesh'in dinamik
üçgen refit listesinde bir facade'ı yoktur.

Aynı geometriyi okuyan her tüketici aynı tamponu okumak zorunda:

| tüketici | matematik |
|---|---|
| `EmbreeBVH::build` | dünya = `getFinal() * P_orig` |
| `TriangleMesh::hit` | ışını `inv(T)` ile yerele çevirir, **`P_orig`** ile test eder |
| `TriangleMesh::bounding_box` | kutuyu `P_orig`'den kurar, sonra `T` uygular |

Skinli mesh istisnadır ve üçünde de aynıdır: canlı skin çıktısı `"P"`de yaşar,
`P_orig` bind pozudur.

## Ne oldu
BVH yolu bir önceki partide `P_orig`'e geçirilerek düzeltildi (`EmbreeBVH.cpp`
içindeki yorum bunu anlatıyor). `TriangleMesh::hit` ve `bounding_box` aynı
düzeltmeyi **almadı**. Sonuç: CPU BVH varken seçim doğru, **BVH yokken** her
şey düzeltilmemiş doğrudan yola düşüyor ve yanlış obje seçiliyor.

Belirti: *"işaret edilen obje ile seçilen obje konumları farklı, çok nadir
tutarlı"*, *"Vulkan RT'ye geçince düzeliyor"*, *"bazı açılışlarda sorunsuz"*.

Ölçüm (ekranı tarayan 77 ışın, kaç benzersiz nesne çarpılabiliyor):
- bedroom.rtp, 710 nesne: **13 → 25**
- saat.rtp, 43 nesne: **1** (siyah boşluğa atılan ışınlar bile "isabet"
  dönüyordu, yani bir mesh kamerayı sarmıştı)

## Teşhisi geciktiren üç şey — asıl ders bunlar

1. **★★★★★ Enstrüman, ölçtüğü şeyle AYNI HATAYI paylaşıyordu.** Teşhis için
   yazılan `scene.pick_ray` de aynı düz taramayı kullanıyordu, bu yüzden bozuk
   yolu "sağlıklı" raporladı: `paths_agree: True`, `ray_divergence_deg: 0`,
   doğru adda isabet. `scene.raycast` ile yapılan çapraz kontrol de **bağımsız
   değildi** — yön ona dışarıdan veriliyordu. Bir ölçüm aleti, ölçtüğü yolun
   bir kopyasını kullanıyorsa hiçbir şey kanıtlamaz.

2. **★★★★ `bvh_present: False` iki kez küçümsendi.** "BVH yoksa doğrusal tarama
   zaten koşuyor" diye elendi. Asıl önemi hız değildi: BVH'nin yokluğu, her
   şeyi **düzeltilmemiş yola** zorlayan anahtardı.

3. **★★★ Sabit bir geometri hatası moda ve açılışa göre DEĞİŞEMEZ.** Kullanıcı
   bunu söyledi ("bu olsa RT'ye geçince düzelmemesi gerekmez mi") ve haklıydı;
   ölçek/taban hipotezlerini bu tek cümle elemeliydi.

## Kullanıcının verdiği ayırt edici kökü buldu
*"Edit mesh modu bu sahnede bile doğru alt alanları seçebiliyor."* Aynı sahne,
aynı kamera, doğru seçim ⇒ geometri de ışın da yerinde; ayrılan tek şey hangi
çarpma yolunun koştuğu. Sculpt/edit `raycastViewportHit` → `scene.bvh->hit()`
(düzeltilmiş yol); nesne seçimi BVH yokken `obj->hit()` (düzeltilmemiş yol).

## Denenip ölçümle GERİ ALINAN iki düzeltme
İkisi de varsayıma dayanıyordu ve ikisi de sayıyı düşürdü:
- "flat mesh'in P'si yerel kalsın" kapısı → 13 **→ 10**
- `P_orig` yoksa `P`'den kur → 13 **→ 5**

Ders: bu kod tabanında seçim yolunu varsayımla değiştirmek pahalı. Doğru hamle,
**çalıştığı kanıtlanmış kardeş yoldaki matematiği kopyalamaktı.**

## Açık kalan
CPU BVH hiçbir sahnede kurulmuyor (`bvh_present: False`, hem 710 hem 43
nesnede). Bu düzeltme onu gerektirmiyor ama BVH olsaydı arıza hiç görünmezdi —
ikinci bir kök, ayrıca kovalanmalı.
