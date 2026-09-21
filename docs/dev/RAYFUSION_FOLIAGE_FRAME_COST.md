# RayFusion — foliage sahnesinde kareyi CPU yiyor

> **Durum:** AKTİF — bir kök neden bulundu ve düzeltildi (YAZILDI, DERLENMEDİ,
> 2026-09-08); ikinci aday **ölçülmedi**. Kaynak denetimleri geçti.

## Belirti ve onun yanlış okunuşu

Kullanıcı: 1000 foliage + terrain + river network olan sahnede, foliage
üretildikten sonra RayFusion (raster viewport) aşırı yavaşlıyor. Aynı sahne
**RT'de çok hızlı**. Kamera hareket ederken **CPU ~%6, GPU ~%17**.

★★★★ **%6 "boşta" demek değil.** Makine 8 çekirdek / **16 mantıksal**; bir
çekirdeğin tamamı toplamın **%6.25**'i. Yani ölçülen şey *tek bir thread'in
%100'e yapışması*, GPU'nun da onu beklemesi. Görev yöneticisinin toplam yüzdesi
tek iş parçacıklı bir darboğazı **yokmuş gibi** gösterir; bu yüzden aranacak şey
"CPU doluysa" değil, "hangi thread serileşti" olmalı.

## Kök neden 1 — BULUNDU: tablo kurulumu tüm sahnenin materyal akışını her kare hash'liyordu

`prepareRayFusionBounce()` **her raster karesinde**, bounce açık olsun olmasın
çalışıyor (`serviceMaterialPreviewProbeField` içinde koşulsuz çağrılıyor). O da
`getRayFusionHitInstances()` çağırıyor, ve orada:

```cpp
for (const auto& key : state->hitMeshKeys) {      // TLAS instance BAŞINA bir kayıt
    ...
    hit.contentHash = hashMix(seed, mesh.cpuMatIds.data(),
                              mesh.cpuMatIds.size() * sizeof(uint32_t));
}
```

`cpuMatIds` **köşe başına** bir `uint32`. Yani her kare, sahnedeki her TLAS
kaydı için o mesh'in **bütün** materyal-ID akışı bayt bayt hash'leniyordu:

- 22M flat üçgen = 66M köşe = **264 MB**, kare başına, tek thread'de.
- `hashMix` bayt başına bir `xor` + bir 64-bit çarpma, ve bir sonraki bayt o
  çarpmanın **gecikmesini** bekliyor. Yani bayt başına ~3-5 çevrim: 264 MB için
  ~0,2-0,3 **saniye**.
- Aynı mesh birden çok kayıttan referanslanıyorsa maliyet o kadar **tekrar**lanıyor.

Bu maliyet hiçbir yerde raporlanmıyordu: `trace_ms` GPU dispatch'ini,
`signature_ms` ise AS kapısını ölçüyor — bu ikisinin arasında kalıyordu.
Bu deponun tekrar eden dersi: **ölçülmeyen kare payı %95 olabilir.**

### Düzeltme

1. **Hash önbelleğe alındı.** `RasterMeshBuffer::matIdsHash` + `matIdsHashValid`;
   yalnız `cpuMatIds`'e yazıldığında yeniden hesaplanır. Steady state O(1).
2. **Her mesh bir kez çözülür**, kayıtlar indeksle çoğaltılır. Yayınlanan dizi ve
   `customIndex` sırası **bayt bayt aynı**; tüketici hiçbir fark göremez.
3. **Karıştırıcı sözcük tabanlı oldu** (8 baytta bir adım). Aynı özellik korunur:
   *her girdi baytı sonucu değiştirir*, yani kapı bir değişikliği kaçıramaz.
   Denetim script'i bunu 192 baytın her biri için tek tek doğruluyor.
4. **Maliyet artık ölçülüyor:** `bounce_prepare_ms` (IPC/Python/panel).

★ Önbelleğin sinsi tarafı: `cpuMatIds`'e yazan bir yeri unutmak **gürültüsüz**
bozar — bounce eski materyal atamasıyla ışıklandırmaya devam eder, görüntü
yalnızca "biraz yanlış" görünür. Bu yüzden `audit_rayfusion_bounce.py` artık
`cpuMatIds`'e yazan **her satırın** yakınında `matIdsHashValid = false`
aramaktadır; yeni bir yazma yeri denetimi düşürür.

Yerinde değiştiren tek yer `updateInstanceMaterialBinding()` (materyal yeniden
atama GPU tamponunu adres değiştirmeden günceller) — orada da bayrak düşüyor.

## Kök neden 2 — ADAY, ÖLÇÜLMEDİ: kamera hareketi TLAS'ı her kare yeniden kurduruyor olabilir

Belirtinin **kamera hareketine** bağlı olması bunu işaret ediyor. AS kapısı
`m_rasterInstances`'ın içeriğini (meshKey + transform + mask) hash'liyor. Scatter
LOD / impostor geçişi kamerayla değişiyorsa instance imzası her kare değişir ve:

```
imza değişti -> rebuildRayFusionTLAS() -> drainInteractiveViewportInFlight()
             -> tüm TLAS yeniden kurulur -> her karede tam boru hattı drenajı
```

`drainInteractiveViewportInFlight()` = `m_rasterFrameRing->waitAll(true)`, yani
kare halkasının **bütün paralelliğini** sıfırlar. CPU GPU'yu, GPU CPU'yu bekler
— kullanıcının tarif ettiği "gereksiz birbirini bekleme" tam olarak budur.

**Bu ölçülmeden düzeltilmemeli:** instance'lar gerçekten değişiyorsa TLAS'ı
yenilemek DOĞRU davranıştır. Ayırt eden sayı `rayfusion.scene_as` içinde hazır:
`tlas_only_refreshes` ve `builds`. Kamera hareket ederken artıyorlarsa neden bu;
sabit kalıyorlarsa değil.

## Ölçüm sırası (build sonrası)

Kontrol listesi: [NEXT_BUILD_CHECKS.md](NEXT_BUILD_CHECKS.md) madde 1-3.

## Kapatılmamış

- Kök neden 2 ölçülmedi.
- AS imza döngüsü hâlâ kare başına bütün instance'ları geziyor. Sözcük tabanlı
  karıştırıcıyla ~8 kat ucuzladı ama O(instance) kaldı; `signature_ms` bunu
  raporluyor ve bir sonraki adayı **o sayı** seçmeli.
- `prepareRayFusionBounce` bounce kapalıyken de tabloyu kuruyor. Önbellekten
  sonra ucuz, ama hâlâ gereksiz; ölçüm bunu üste taşırsa kapı eklenmeli.

## İlgili

- [RAYFUSION_PROBE_GRID.md](RAYFUSION_PROBE_GRID.md) — aynı partide açılan ızgara kadranı
- [RAYFUSION_PROBE_BOUNCE.md](RAYFUSION_PROBE_BOUNCE.md) — tablonun ne işe yaradığı
- [REALTIME_CAMERA_MOTION_PERF.md](REALTIME_CAMERA_MOTION_PERF.md) — aynı hata sınıfının önceki turu
