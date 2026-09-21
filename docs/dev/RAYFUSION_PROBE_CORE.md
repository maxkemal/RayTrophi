# RayFusion 0.1 — probe kontrol çekirdeği

> **Durum:** AKTİF — yazıldı, DERLENMEDİ (2026-09-07). GPU veya görüntü doğrulaması yapılmadı.

## Bu partinin gerçek kapsamı

`source/include/RayFusion/ProbeField.h` ve `source/src/RayFusion/ProbeField.cpp`
GI için CPU kontrol katmanını kurar. Canlı scene, Triangle facade veya Vulkan
handle'ı okumaz. Üçgenleri taramaz; sınırlı probe ızgarasını yönetir. Flat
TriangleMesh/DNA kaynak adaptörü ve GPU kaynak sahibi sonraki dilimdedir.

- Dünya koordinatlı tamsayı hücreler toroidal slot'lara yerleşir. Kaydırmada
  örtüşen hücreler korunur; yeni hücre eski slotun ışığını devralmaz. Negatif
  hücre koordinatları da desteklenir. `spacing` dünya birimidir.
- İş biletinde field kimliği, generation, device/scene epoch, geometry ve
  lighting revision, dünya hücresi ve tek kullanımlık serial bulunur. Başka
  field'ın veya önceki proje/cihaz/ışık/geometri sürümünün sonucu reddedilir.
- Her slot'un en fazla bir işi uçuşta olabilir. Round-robin planlayıcı hem
  probe hem ray bütçesine uyar. `cancel` başarısız işi serbest bırakır;
  iptal edilmiş işin geç completion'ı kabul edilmez.
- Önce paket tamamen doğrulanır, sonra yayınlanır. NaN/Inf, negatif ışık ve
  imkânsız mesafe momentleri reddedilir. Başarısız publish geçerli cache'i
  değiştirmez; producer yeniden deneyebilir veya bileti iptal edebilir.
- İlk geçerli örnek boş/siyah history ile karıştırılmaz. Sonrakiler aynı
  revision içinde sınırlı history ağırlığıyla birleşir. `targetUpdates` kadar
  başarılı örnekten sonra planlayıcı iş üretmez; yeniden örnekleme/yeniden
  ışıklandırma kararı gelecekteki scene service tarafından verilir.
- Bilinmeyen probe `lookup=nullptr` döner. Bu, ölçülmüş siyah ışık değildir.
  Moment görünürlüğü bir tahmindir; geometrik tam görünürlük garantisi vermez.

Kontrol katmanı tek owner thread'de çalışır. GPU worker/completion kuyruğu
sonuçları bu threade taşımalıdır; canlı sahneye paralel güvenli erişim bu
sınıfın sorumluluğu değildir. Kuyruk iptali GPU işinin fiziksel olarak bittiği
anlamına gelmez; GPU kaynak ömürleri ayrı fence/retirement mekanizması ister.

## Veri ve bütçe sözleşmesi

Probe başına 8×8 yönlü texel; texel başına 32 byte: scene-linear
irradiance/PI RGB ve yönlü mesafe ortalaması/ortalama karesi. Mesafe dünya
birimi, ikinci moment dünya biriminin karesidir. Albedo ve pozlama pakete
dahil değildir. GPU projeksiyon shader'ı henüz yazılmadı; C++ ABI sürümü 1.

| Mevcut viewport kalite preset'i | Ray / probe | En fazla probe / update | En fazla ray / update |
|---|---:|---:|---:|
| Performance | 32 | 8 | 256 |
| Auto / Balanced | 64 | 16 | 1024 |
| Quality | 128 | 32 | 4096 |
| Full | 256 | 32 | 8192 |

Bunlar ilk **planlama tavanları**, ölçülmüş optimumlar değildir. GPU süreleri,
visibility ray'leri ve secondary/shadow ray maliyeti henüz dahil edilmedi:
bu alanlar primary probe-ray bütçesidir; gelecekte toplam GPU iş sayacı ayrıca
olmalı. Full preset'inin mevcut tam geometri anlamı değiştirilmedi.

Izgara 32768 probe ile sınırlı; yönlü paketler en fazla 64 MiB, CPU slot
metadata ve geçici üretici paketleri ek maliyettir. Varsayılan 256 probe'nin
payload'ı 512 KiB. Bu değerler canlı GPU allocation değildir.

## Tek API, üç erişim

- C++: `rtapi::rayFusionCoreStatus()` / `rtapi::validateRayFusionCore()`.
- Python: `rt.rayfusion.core_status()` / `rt.rayfusion.validate_core()`.
- IPC: `rayfusion.core_status` / `rayfusion.validate_core` — `Read` capability.
- UI: Render Settings > Realtime PBR Quality > RayFusion development >
  Validate core. Testler yalnız düğmeye basılınca koşar, popup açıkken tekrarlanmaz.

İki IPC metodu parametre almaz; fazladan parametre reddedilir. Mevcut quality
ayarının raporu kullanılır, yeni authoring ayarı veya proje verisi eklenmedi.
Durum `core_available=true`, **`renderer_available=false`, `gi_active=false`**
ve sebep bildirir. `planned_*` alanları gerçekleşmiş iş değildir.

`validate_core`, gerçek C++ sınıfına 34 izole kontrol uygular. Canlı scene'i,
kamera/modu veya üretim cache'ini değiştirmez. Sonuçtaki `gpu_tested=false`
daima korunur; testlerin geçmesi GI görüntüsünün hazır olduğu anlamına gelmez.

## Sıralı build ve kabul

1. **C++ build** al. Beş yeni `.cpp` vcxproj'da kayıtlıdır. Bu parti yeni
   shader eklemez ve shader derlemesi gerektirmez. Kullanıcı uygulamayı açar.
2. `rayfusion.core_status` oku. Yukarıdaki unavailable durumları açık olmalı.
   **Sinsi hata:** yalnız core_available'ı görüp renderer/GI hazır saymak.
3. `scripts/ipc/Probe-RayFusionCore.ps1` çalıştır. Kopyası
   `x64/Release/scripts/ipc/Probe-RayFusionCore.ps1` içindedir. Betik uygulama
   başlatmaz/derlemez ve sahne değiştirmez. 34 native kontrol geçmeli; ABI,
   bütçe, capability, bilinmeyen parametre ve aggregate ayrıca denetlenir.
4. UI düğmesi ve `rt.rayfusion.validate_core()` aynı isimli kontrolleri aynı
   sonuçla raporlamalı. Bozuksa binding/core farkı veya eski binary söz konusu.
5. Mevcut viewport quality değişince `planned_*` tabloyu izlemeli. RayFusion
   modu/ışığı açılmamalı; bu dilimde producer ve raster consumer yoktur.
6. Mevcut Realtime/Rendered, sahne açılış/TDR ve template HUD testleri kendi
   listelerinden sürer. Bu parti onların doğrulandığı iddiasını taşımaz.

## Bu turda çalıştırılan denetimler

IPC capability sınıflandırması ve Python mirror karşılaştırması geçti;
descriptor tablosu üretildi. Vcxproj XML'i ve beş source kaydı, Python/IPC'nin
aynı rtapi operasyonlarına bağlı oluşu statik kontrol edildi. Native C++
kontrolleri yazıldı fakat **çalıştırılmadı**; build kullanıcıya aittir.
