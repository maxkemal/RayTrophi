# Kullanıcı derlemesi: SDF kalite ve cache maliyeti

Mevcut açık sahne dış IPC ile incelendi; build/uygulama başlatılmadı. Başlangıç
playhead kare 0, runtime parçacık sayısı 0 ve eski surface stats mevcuttu;
bu stale sayılar başlangıç performans örneği olarak kullanılmadı. Mevcut RAM
cache'ten kare 61 açıldı, yeni fizik adımı hesaplanmadı. 25,416 parçacığın
tamamı Fluid modelinde, frozen/granular/elastic sayıları sıfır.

## Aynı karede kalite A/B

Default m1/anisotropy off, m2/off, m1/on karşılaştırıldı. Parçacık sayısı ve
playhead değişmedi; sonda kalite=1/anisotropy=false ve capture=false geri
yüklendi. Cache 1..61/61 kare, 2,229,130,137 B olarak korundu. Sahne kare
61'de durmuş bırakıldı. Görüntüler capture açıkken alındı; capture render
yolunu senkronlaştırdığı için capture viewport süreleri normal playback
performansı olarak kullanılmadı.

| Ayar | SDF producer (iki ölçüm) | Render sync (capture kapalı ölçüm) |
|---|---:|---:|
| Varsayılan 128³ | 8.2–8.6 ms | 82.9 ms |
| Kalite x2 / 256³ | 35.1–43.0 ms | 123.3 ms |
| 128³ + anisotropi | 88.9–94.7 ms | 160.2 ms |

Eski binary'nin aynı frame/state/parametrelerle ölçümü yok; hızlanma yüzdesi
verilemez. Yeni producer'ın geçerli default cache örneğinde hızlı olduğu
ölçüldü. Default plan 224,256 tile lane / 2,097,152 domain hücresi (%10.7),
CSR 262,144 bin. Kalite x2'de CSR yine 262,144, tile işi 1,145,856 lane.
Beklenen refinement-independent bin sözleşmesi canlıda doğrulandı.

60↔61 cache geçişlerinde default SDF producer 7.76–9.20 ms; render_sync
76.73–80.55 ms. Alt aşamalar: material_coordinates 46.51–49.27 ms,
fog_volume 12.05–12.18 ms, volume.publish 9.25–9.80 ms, level_set
profiler 7.99–9.44 ms. Cache restore 40.63–41.87 ms ayrıca raporlanıyor.
Parent ve child süreler birbirinin üzerine eklenmez. SDF production artık
bu surface hattının başlıca maliyeti değildir; UVW üretimi yaklaşık %60 pay
alıyor. Bu turn'de UVW veya VDB kodu değiştirilmedi.

## Geometrik görünüm

Solid capture'da jet sütunu pütürlü, şişkin/loblu, yer yer parçacık izli
görünüyor. m2 görüntüsü daha ince örnekliyor fakat aynı lobları taşıyor;
temel şekil sorunu çözünürlükle düzelmiş değil. Anisotropi bu karede maliyetini
haklı çıkaracak belirgin bir görsel iyileşme göstermedi. Tek kare görsel
değerlendirmesidir; temporal kararlılık ve sıvı optiği kabulü sayılmaz.

Domain 5 m/128 → voxel=0.0390625 m. Kernel radius=2 voxel (7.8125 cm),
particle radius=.55 voxel (2.1484 cm), offset=.65 voxel (2.5391 cm),
smoothing=2. Source radius=.35 m (nominal çap yaklaşık 18 voxel), velocity
(0,-3.7,0) m/s, spread=.15. Böyle bir nominal kaynak ölçeğinde pütürleri
yalnız “128 yetersiz” diye açıklamak uygun değil. m2 karşılaştırması,
reconstruction/normal alanını ve gerçek parçacık dağılımını hedeflemek
gerektiğini destekliyor; tek başına solver fiziğinin yanlışlığını kanıtlamaz.

Öncelik: gereksiz/yoğun UVW üretimi ve extrapolation maliyetinin ayrı audit'i;
Zhu-Bridson weighted-centroid/dilation/normal davranışının düzgün jet, tank,
izole damla ve sheet referanslarıyla kalibrasyonu. Yeni tile kodu yüzey
formülünü değiştirmiyor (anisotropic eksik support düzeltmesi hariç); kalite
iyileşmesi esas olarak bundan sonraki reconstruction çalışmasına bağlı.

## Kanıt

- Varsayılan (temporary capture removed)
- x2 kalite (temporary capture removed)
- Anisotropi (temporary capture removed)
- [A/B ham sayaçlar](sdf_quality_ab_frame61_2026_10_10.json)
- [Capture kapalı aynı-kare profiler](sdf_same_frame_perf_2026_10_10.json)
- [Cache geçiş ölçümleri](sdf_cache_cost_live_2026_10_10.json)
