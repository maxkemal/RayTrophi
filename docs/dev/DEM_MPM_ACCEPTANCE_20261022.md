# DEM kapanış ölçümü ve MPM–DEM başlangıç sahnesi

## Mevcut DEM sahnesi

Kullanıcının derlediği uygulamada dış named-pipe istemcisiyle ölçüldü.
Yeni derleme veya uygulama başlatma yapılmadı.

- Domain: `Physics Domain 1`.
- 100.000 fiziksel tane; mevcut boyut/malzeme ayarları korundu.
- 6 ardışık adım, dt = 1/24 s; toplam 0,25 s ek simülasyon.
- Hiçbir adım tutulmadı; 1162–1260 alt adım.
- Ölçülen en yüksek temas sayısı: 33.
- Host `step_total`: 1250–1476 ms; ortalama 1358 ms.
- Son bildirilen çalışma belleği: 430.100.992 bayt (~410 MiB).

Bu, bildirilen durma senaryosu için kısa devam kabulüdür. Uzun süreli kararlılık,
fiziksel yakınsama veya milyon tanelik performans sonucu değildir.

Kurulum değişikliğinden önce `project.autosave_now` başarıyla çalıştı:
`assets/config/autosave.rtp`, 30.918 bayt. Bu bir proje kurtarma kaydıdır;
canlı parçacık durumunun kalıcı checkpoint'i olarak değerlendirilmemelidir.

## Aynı sahnede yeni ortak fizik düzeneği

Eski domain ve `Point Flow Source` silinmedi, pasifleştirildi. Yapılandırma
değişiklikleri sırasında eski canlı parçacık durumu sıfırlandı; eski domain son
okumada 0 parçacık taşıyor. İlk kurulum ve ölçümler JSON kaydında bulunuyor.

Yeni domain: **MPM Sand + DEM Stones**.

- Sınırlar: (-0,6; 0; -0,6) – (0,6; 1,6; 0,6) m.
- İstenen voxel: 0,06 m; motorun raporladığı etkin voxel: ~0,05926 m.
- `Acceptance MPM Sand`: Sand'den türetilmiş, `granular_transport=mpm`.
- Kalıcı yatak dolumu: (-0,45; 0,06; -0,45) – (0,45; 0,36; 0,45), ppc=8.
- 12.288 MPM taşıyıcısı.
- `Acceptance DEM Stone Drop`: Gravel, (0; 0,95; 0), kaynak yarıçapı 0,20 m.
- DEM taş yarıçapı 0,045 m, sabit boyut; hassasiyet 24, alt adım sınırı 2048.
- Kaynak toplam doğum sınırı 12; sahiplik hazırlığı/reseed sırasında iki doğum
  silindiği için ölçülen son sahnede **10 canlı DEM taşı** bulunuyor.

Sahiplik sırası önemlidir: yatağı doldurmadan önce boş domain'de bir adım
çalıştırılarak DEM kapısı kaynaklardan türetildi. Aksi halde canlı MPM taşıyıcıları
varken sonradan DEM kapısına geçiş `reset_pending` ile engelleniyor. Bu gözlem
kurulum sırasıyla giderildi; otomatik authoring davranışına ilişkin ayrı bir
iyileştirme adayıdır.

## Ortak çözücü ölçümü

Hazırlık sonrasında 6 + 80 adım, dt = 1/120 s (~0,717 s):

- Ortak durum: `MPM + grain Vulkan: shared geometry/contact clock, CFL grid refresh`.
- Taşıyıcılar MPM/DEM olarak ayrı sahiplerde kaldı; birbirine dönüştürülmedi.
- Adım tutulması görülmedi.
- Taşlar yatağa ulaşınca temas olayları ve karşılıklı impuls ölçüldü.
- Son örnekte temas olay sayısı 6120; bu eşsiz çift sayısı değil, alt adımlar
  boyunca biriken olay sayısıdır.
- Örneklenen temas momentum residual değerleri ~5,7e-9 – 5,6e-8 N·s.
- Son durumda 12.298 toplam taşıyıcı: 12.288 MPM + 10 DEM.

Bu ölçüm ortak temas yolunun çalıştığını gösterir. Doğru batma derinliği,
uzun süreli destek, hacim/yoğunluk yakınsaması ve taş boyutuna göre cevap henüz
fiziksel kabulden geçirilmedi.

## Kullanıcıya bırakılan durum

- Uygulama duraklatılmış, timeline kare 0.
- Yeni domain etkin ve görünür; eski domain ve kaynak pasif.
- Kalıcı MPM seed'i ve DEM emitter'i reset sonrası tekrar kurulum için mevcut.
- Baştan izlemek için Reset, ardından Play kullanılabilir. Kaynak sayacı da
  resetlenirse yeni koşuda 12 taşlık doğum sınırı geçerli olur.
- Değişiklik sonrası proje kaydı otomatik olarak zorlanmadı; kullanıcı normal
  Save ile bu kabul sahnesini saklayabilir.

Ham ölçüm ve önceki ayarlar: `dem_mpm_acceptance_20261022.json`.

Sonraki fizik kabulü: kum yatağı tek başına / taşlar tek başına / birlikte
kontrollü karşılaştırma; taş konumu, desteklenme, yatak deformasyonu ve
momentum aktarımı; dt ve voxel yakınsaması.

## Seed substance clarity and visual follow-up

User reported fluid-like appearance. Read-only IPC audit confirmed zero fluid
owners, 12,288 MPM + 10 DEM, domain granular_enabled=true, material category
Granular, transport MPM, Young modulus 200 kPa and friction angle 35 degrees.
Source audit confirms runMatterGpuStep enables the granular lane even though
its enclosing continuum input disables the legacy aggregate granular flag.
This is not proof of correct repose/penetration; the initial broad rectangular
bed is inadequate for that acceptance. Visible spheres represent MPM parcels,
not individual physical sand grains. Direct viewport observation was performed
with computer-use; no screenshots were saved as repository files.

Pending C++ build: Seed section exposes the existing domain-default substance
picker plus model/transport labels and inherited-authority/reset explanation.
It calls the same updateFluidDomain service as Matter, IPC fluid.set_param and
Python fluid.set_param(default_substance=...). No independent seed-substance
state or retagging path was introduced. Emitters retain their own substances.

Verify picker synchronization with Matter, granular MPM/DEM/liquid labels,
rejected-update feedback, persistent reset/seed material, and save/load. Existing
fluid.seed has no independent substance argument: it uses the domain default.
No physical material values, particle tags or scene setup were changed in this
follow-up. A free pile/slope/settling comparison remains the next physics gate.
