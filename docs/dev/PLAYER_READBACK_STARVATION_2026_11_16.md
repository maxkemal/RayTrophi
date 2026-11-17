# Player yoğun sahne: tamamlanmış kare kaybı — 2026-11-16

Durum: canlı eski binary üzerinde yeniden üretildi; ortak kaynak düzeltildi.
Kullanıcı yeni derleme sonrası yoğun alandaki kamera takılmasının çözüldüğünü
onayladı. C++ build Codex tarafından çalıştırılmadı. Aşağıdaki otomatik özel
delivery assertion yeniden çalıştırılmadı; sonraki yoğun boya matrisinde
submit/consume ilerledi.

İki uygulama oyuncak.rtp sahnesinde aynı Vulkan raster backend'ini kullanıyor.
Bu ölçüm Player kamerasını sabit tuttu; Studio kamerası birebir eşit olmadığından
Studio/Player FPS karşılaştırması değildir. Player: 1680×945, 1627 draw,
3.710.846 görünür üçgen. Proje yükleme/kaydetme, kalite ve shader değişikliği yok.

| Player, yaklaşık 3 saniye | Submit | Tüketilen yeni kare | Eski görüntü | GPU ortalama |
|---|---:|---:|---:|---:|
| Asenkron, sınırsız | 114 | 0 | 114 | 26,82 ms |
| Capture ile senkron kontrol | 87 | 87 | 0 | 27,08 ms |
| Asenkron, 10 Hz sınır | 31 | 31 | 0 | 33,50 ms |

Capture kontrolü bir FPS optimizasyonu değildir. GPU çizimi sürerken yeni görüntü
sunulamaması sorunu izole edildi. Her aşamada resource_drains=0; kamera aynı,
sonunda continuous/pacing/capture geri alındı, restoration_errors boş.

## Kaynak neden ve düzeltme

RasterViewportFrameRing::consumeNewestReady yalnız fence'i tamamlanan inFlight
slotları okur. GPU yetişemediğinde ilk tüketim denemesi boş döner. beginFrame
slotu yeniden kullanmak için mevcut fence'i bekler; bekleme sonrası tamamlanmış
ama okunmamış görüntüyü korumadan fence'i sıfırlayıp slotun üstüne yazıyordu.
Yoğun sürekli üretimde bu her tur tekrarlanabilir. Görülen takılma geometri cache
tekrar oluşturma kanıtı değildir. Uzaktaki düzelme daha kısa GPU işiyle uyumludur;
uzak kamera ayrıca ölçülmedi. Studio'nun akıcılığını gizleyen tam zamanlama farkı
bu deneyde ayrıca izole edilmedi; farklı renderer kullanıldığı sonucu çıkmaz.

Yalnız RasterViewportFrameRing.cpp değişti: zaten gereken slot beklemesinden
sonra tamamlanan okunmamış görüntü tek CPU tamponunda korunur; tüketici hazır
GPU slotları ve korunmuş görüntü arasından en yenisini sunar. Sayaç yalnız gerçek
tüketimde ilerler. Pending kontrolü korunan görüntüyü de görür. Yeni GPU bekleme,
kalite düşürme, shader veya public API/ABI değişikliği yok. Basınç altında ek CPU
kopyası ve bir kare tamponu (~6,35 MB) maliyeti vardır; bu kopya toplam CPU frame
maliyetine dahildir, mevcut host_read_ms aşamasında ayrıca görünmez.

## Doğrulama ve sonraki adım

- Player ve Studio derleyici seçenekleriyle cl /Zs sözdizimi: PASS (build değil).
- Odaklı git diff --check ve dış probe --help: PASS.
- Player'ı kullanıcı artımlı derlemeli; tek ortak .cpp etkilenir. Studio da yeniden
  derlendiğinde ortak düzeltmeyi alır. Yeni binary başarısı henüz ölçülmedi.
- Aynı yoğun bakışta, Player açıkken dış terminalden:

```powershell
python scripts/test/rt_player_readback_probe.py --assert-delivery --output tools/perception_bridge/out/readback-after-fix.json
```

Probe Studio'yu salt okunur inceler, Player görüntü bayraklarını geçici değiştirip
geri alır; sahne/kamera değiştirmez. Kabul: asenkron aşamada yeni kare tüketimi
ilerlemeli, sürekli eski görüntü dizisi kesilmeli, device loss olmamalı. Ardından
kullanıcı aynı yoğun alanda kamera hareketinin akıcılığını doğrulamalı. Bu sayaç
kontrolü tek başına kusursuz frame pacing veya tüm performans sorunlarının çözümü
anlamına gelmez.
