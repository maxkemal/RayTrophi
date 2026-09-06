# Timeline oynatımı kare hızına kilitleniyordu

> **Durum:** REFERANS — kök neden bulundu ve düzeltildi (2026-09-05), RUNTIME
> DOĞRULANMADI. Doğrulama adımları `NEXT_BUILD_CHECKS.md` §2'de.

## Belirti (kullanıcı raporu, 2026-09-05)

> "raster mod animasyonu hesaplama hızına göre oynatıyor yani ne kadar hızlı ise
> o kadar hızlı oynatıyor, bu bir hata mı"

**Evet, hata.** Ama hata sanılan yerde değil.

---

## ★★★★ Kök neden: ÖDENEMEYECEK bir zaman borcu

`TimelineWidget.cpp` oynatım saati duvar saatine dayanıyor:

```cpp
float elapsed = duration(now - last_time);
float frame_duration = 1.0f / animation_fps;
if (elapsed >= frame_duration) {
    int frames_to_advance = (int)(elapsed / frame_duration);
    if (frames_to_advance > 1) frames_to_advance = 1;      // TAVAN
    current_frame += frames_to_advance;
    ...
    last_time += duration(frames_to_advance * frame_duration);   // TAŞIMA
}
```

Tavan ile taşıma **birbiriyle çelişiyor**, ve kodda ikisi de ayrı yorumlarla
savunuluyordu (biri "yavaş çekimi düzeltir", öteki "ileri sarmayı önler").

Viewport `animation_fps`'ten yavaş çiziyorsa — diyelim 24 fps hedefte 10 fps —
her tick'te `elapsed ≈ 100 ms`, `frame_duration ≈ 41.7 ms`, yani gerçekte
**2.4 kare** geçmiş. Tavan bunu 1'e kırpıyor, ama taşıma satırı `last_time`'ı
yalnızca **1** kare kadar ilerletiyor.

⇒ Her tick'te ~1.4 karelik bir borç birikiyor ve bu dal borcu **asla
ödeyemiyor**, çünkü tavan geri ödemeyi tick başına 1 kareyle sınırlıyor.

★ Sonuç, belirtinin tam olarak kendisi: `elapsed >= frame_duration` bir kez
kalıcı olarak doğru olduktan sonra timeline **çizilen kare başına tam bir
animasyon karesi** ilerliyor. Oynatım hızı = çizim hızı. Sahne yavaşken yavaş
çekim; sonra sahne hızlanınca **gerçek zamandan hızlı** oynatıyor — ve oynatımı
durdurup yeniden başlatmadan geri dönüşü yok (`last_time` yalnızca orada
sıfırlanıyor).

**Yani asıl kusur tavan değil, tavanın ALTINDAKİ taşıma.** Tavan bilinçli bir
tercih: yavaş bir viewport'ta kare atlamak ileri sarma gibi görünür ve kare
başına sorunları gizler.

---

## Düzeltme

Tavan devrede kaldı; kırpma olduğunda `last_time` **yeniden senkronlanıyor**
(`last_time = now`), yani fazlalık bankaya yazılmak yerine gerçekten düşürülüyor.

Böylece tavan söylediği şeyi ifade ediyor:

- viewport yetişiyorsa → **gerçek zaman**, `animation_fps` hızında,
- yetişmiyorsa → **yavaş çekim**, tick başına en fazla bir kare,
- her iki durumda da **kalıcı kayma yok**, kaçış yok.

---

## Ne DEĞİŞMEDİ (ve bu bir tercih)

Yavaş bir sahnede oynatım hâlâ gerçek zamandan yavaştır. Alternatif — kare
atlayıp gerçek zamanı korumak — çoğu DCC'nin yaptığı şey ve kod bir zamanlar onu
denemiş (üstteki ilk yorum ondan kalma). Bu partide o tercih **yeniden
açılmadı**: değiştirilen tek şey, hangi tercih seçilirse seçilsin yanlış olan
sınırsız borçtu.

★ Bu ayrım önemli: bir hatayı düzeltirken bir davranış tercihini de sessizce
çevirmek, sonradan bakan kişiye hangisinin kasıtlı olduğunu söyleyemez hale
getirir.

---

## Nasıl doğrulanır

Sayı ile, göz kararıyla değil: bilinen uzunlukta bir klip oynat ve duvar saati
süresini ölç. 24 fps'te 96 karelik bir klip, viewport yetişiyorsa 4.0 saniyede
bitmeli. Eskiden aynı klip, çizim hızı ne olursa olsun `96 / render_fps` saniye
sürerdi.
