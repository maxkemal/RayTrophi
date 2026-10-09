# Diğer ajan — S1b kalan hücre alanları

2026-10-09. Ana ajan DEM uyku aktarımının revision-24 shader/host/test fazını
çalışıyor. Paralel, bağımsız kaynak işi için bu devir kullanılabilir.

Başlangıç: `MATTER_SPARSE_S1_SIVI_GPU.md` → S1b ikinci parti ve "Sıradaki parti";
üst mimari sözleşme `MATTER_SPARSE_FULL_STORAGE_IMPLEMENTATION.md`.
Devir notuna göre P2G ve katı MAC yüz ağırlıkları kaynakta taşındı; **hücre alanları
(mask/divergence/pressure publication/solid velocity) ve bütün tüketicileri açık**.
Mevcut kaynakta bunlar önceki devirden beri taşınmışsa ikinci bir yol ekleme:
önce git/worktree ve gerçek caller/allocation durumunu denetle, tamamlananı koru.

## Kapsam / sınırlar

- Mevcut owner kapısı, transactional fallback ve net allocation muhasebesi aynı
  kalacak. Mask/solid/fraction/porous, pressure/RHS/gradient ve projection/GFM
  tüketicileri tek otoriteyi okuyacak; unlinked ikinci canonical state oluşturma.
- Cell slot/512 ve MAC slot+1/576 ayrımı, clipped/padding yüzleri ve scatter+halo
  desteği korunacak. Boş hücre/ambient reading kendi başına value page ayırmamalı.
- S1/S1b ağırlık kapılarının build/IPC sonucu teyit edilmeden yeni faz kabulü
  ilan etme. Source audits ve mevcut `rt_test_sparse_pressure_ipc.py --transfer
  --solid-weights` checklist'ini temel al. Shader hazır değilse mevcut fallback
  kalır; CPU/dense yolu compact GPU başarı diye raporlama.
- Saf sıvı `compact_owner=false` satırları S1.5 cihazda kalıcılık ön koşulu
  sağlanmadan kaldırılmaz. S1.5/S2/S3 ayrı gate'ler; bu göreve karıştırma.
- `GasSimulator.cpp` değişmeyecek. Build ve shader compilation kullanıcıya ait.
  Uygulama açma; live test gerekiyorsa açık uygulamada ayrı external IPC süreç.
- Over-2000-line dosyalara feature gövdesi ekleme; focused modüller ve minimal
  wiring. Flat TriangleMesh/DNA tek scene geometry yolu. Script/IPC/UI aynı core.
- Test script'leri `scripts/test` ve `x64/Release/scripts/test` kopyalarında aynı.

## Ana ajanın ayırdığı dosyalar

`sim_matter_grain.glsl`, `sim_matter_grain_sleep*.glsl`, `MatterGrain.h`,
`MatterGrainGpu.cpp`, `MatterGrainControls.cpp`, grain sleep/transfer tests ve
`DEM_SLEEP_TRANSFER_IMPLEMENTATION.md` ana ajan kapsamı: değiştirme.
Shared `NEXT_BUILD_CHECKS.md` düzenlenecekse kendi S1b bölümüne küçük ek yap;
DEM revision-24 bölümünü koru. Core grain dosyasına wiring gerekirse yalnız
somut ihtiyaç/çağrı yerini devirde belirt; aynı anda grain ownership değiştirme.

Çıkış: taşınan alan/tüketici/ABI listesi, kalan yoğun banklar ve sebepleri, gerçek
capacity bytes, source audit sonuçları ve kullanıcı build/live checklist'i.

## Kaynak denetimi devri

Uyku rev24 denetiminde 10 kontrolden 8'i geçti. İki eski kontrol ilk kopyalama
assertion'ında durdu: `check_matter_transfer_contracts.py` literal
`copy(particle_id, other.particle_id)`, `check_matter_pore_contracts.py` literal
`copy(pore_water_mass_kg, other.pore_water_mass_kg)` arıyor. `FluidParticles.h`
kopyalamayı `forEachColumnPair` üzerinden yürütüyor; particle_id ve dört pore
alanı ortak sütun listesinde mevcut. Uyku işi bu dosyayı değiştirmedi.
Bu kontrolleri ortak sütun listesi ve gerçek copy/gather yollarını doğrulayacak
şekilde güncelle; kapsamı azaltma. İlk assertion sonrasındaki kapılar henüz bu
koşuda doğrulanmadı, yeniden çalıştırıp sonucu raporla.
