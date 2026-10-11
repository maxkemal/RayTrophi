# Korunum ve Matter RAM cache optimizasyonu — 2026-10-10

Kaynak hazır; yeni C++ ve sim_inventory_conserve SPV kullanıcı derlemesi,
canlı karşılaştırma ve rewind kabulü bekliyor. Ajan derleme/uygulama açma yapmadı.

## Yapılan değişiklik

- Closed/periodic inventory advection bütün alanı sıfıra taşırsa GPU ve CPU
  referansı önceki alanı korur. Normalizasyonun sıfır paydası artık kütle/enerjiyi
  silmez. Open outflow aynı kalır. Çok ince deposit'in backtrace örnekleri arasında
  kaybolması somut bir kaynak açığıdır; kare 54'teki ledger/grid farkının yalnız
  bu nedenle oluştuğu henüz canlı olarak kanıtlanmadı.
- Boiling kg miktarı thermal evaporated toplamına eklenir. Surface exchange,
  gas heat/particle/evaporation sayaçlarına katkı ekler; önceki boiling katkısını
  kendi sıfır sonucu ile ezmez. Fizik katkıları ile API/IPC/script telemetry aynı
  runtime alanından gelir. Her sıvı adımı mevcut ThermalLiquidStats resetini korur.
- Matter'ın ikinci grid'inin 33 vector alanı ayrı kayıpsız frame payload'una taşınır.
  Tekrarlanan değer + indeks/değer kodlaması yalnız dense storage'dan küçükse
  kullanılır. Yoğun alan dense kalır. Signed zero ve NaN/Inf bit desenleri sayısal
  eşitlik ile birleştirilmez. Pressure, divergence, solid/moving-solid geçmişi,
  ağırlıklar ve MAC velocity dahil tüm alanlar restore edilir; hiçbir fiziksel
  state silinmez veya quantize edilmez. Scalar layout metadata State'te kalır.
- Kg/J CPU cache snapshot alanları aynı kayıpsız payload'la saklanır.
- Cache capture bağımsız snapshot'ı move ile tüketir. Önceki `c.meta = st`
  büyük grid/parçacık kopyası artık ikinci defa yapılmaz. Canlı state taşınmaz.
- Cache bellek hesabı primary/secondary grid'in bütün vector kapasitesini ve yeni
  payload'ların indeks/değer kapasitesini kapsar. Budget bu kez Matter grid'ini
  atlayamaz. Scripting/IPC sim_cache.status aynı core budget hesabını raporlar.

## Kapsam sınırı

Bu değişiklik RAM saklama ve CPU kopya maliyetini hedefler. Timeline'ın her kare
16 MiB kg/J snapshot okuması hâlâ vardır; GPU snapshot residency uygulanmadı.
~175 MB/adım liquid grid/particle IO bu partide kaldırılmadı. CPU tüketicilerin
kaynağını değiştirmeden bu IO'yu atlamak stale veri ile fiziği bozabilir.
Bellek veya FPS kazancı henüz ölçülmedi; kaynak testleri bunu kanıtlamaz.

Havada/yerde boiling gözlemi ayrı model incelemesi olarak açık. Termal fizik
kapıları gevşetilmedi; latent ödeme, madde özellikleri ve boundary değiştirilmedi.

## Kontrol ve tek kullanıcı derlemesi

Kaynak/matematik kontrolleri:

```powershell
python scripts/test/check_matter_frame_cache.py
python scripts/test/check_gas_inventory_math.py
python scripts/test/check_gas_inventory_gpu_contracts.py
python scripts/test/check_matter_phase_contracts.py
python scripts/test/check_matter_solver_stages.py
python scripts/gen_ipc_descriptors.py --check
```

Packing oracle C++ header çalıştırmaz; source tests compiler/GPU yerine geçmez.
Yeni iki header vcxproj/filters'a kayıtlı. Kullanıcı eşleşen C++ ve simulation
shader derlemesini alır; özellikle değişen sim_inventory_conserve SPV yenilenir.

Canlı kabul, ayrı terminal ve durmuş disposable sahne:

1. `python scripts/test/rt_test_gas_inventory_gpu_ipc.py` — mevcut boiling/heat
   fixture'larının ledger/mass/latent kapıları, resident yol, 0 fizik grid transferi,
   summary cache ve **toplam inventory >= son-adım vaporization katkısı** şartı.
2. `python scripts/test/rt_test_contact_heat_ipc.py` — temas regresyonu.
3. Aynı proje kare 54 evresinde ledger/grid toplamı ve thermal evaporated ölçümü.
   Yaklaşık .1287 kg katkı karşısında .000935 kg toplam tekrarlarsa kabul FAIL;
   deposit/surface commit/advection/snapshot/summary zincirinde ek attribution gerekir.
4. Timeline 0..54 RAM/cache/solver probe: eski baseline private RAM artışı
   ~104,8 MB/kare; kare 54 private 8,75 GB, cache raporu 2,81 GB, solver toplamı
   218,78 ms. Yeni aynı koşulda gerçek RAM, doğru cache raporu ve solver maliyeti
   birlikte alınır. Aynı kare ve parçacık sayısı olmadan hız oranı hesaplanmaz.
5. Cache keyframe ve ara kareye scrub, sonra devam: kg/J toplam/centroid, parçacık
   state digest ve closed mass karşılaştırılır. Particle legacy quantization
   politikası değişmedi; yeni secondary-grid/kgJ payload'ı bit-exact roundtrip ister.
   Disk bake v12 formatı değişmedi; save/bake/reopen kapısı tekrar çalıştırılır.

Kabul kapanmadan “hızlı ve fiziksel olarak doğru” sonucu verilmez.
