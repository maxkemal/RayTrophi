# 2026-10-11 parti 3 — yağmur: otokonversiyon, sis tavanı, düşüş, iniş

Parti 2 (sis suyu) canlı 6/6 PASS, ama sis tutan hücreler ortalama 41,7 kg/m³
taşıyordu (gerçek sis 1e-4–1e-3). Bu parti sisi tavana bağlar, fazlasını yağmura
çevirir, yağmuru düşürür ve sıvıya indirir. Ayrıntı:
[BIRLESIK §4.8b](BIRLESIK_MADDE_DOMAIN_TASARIMI.md).

Yeni shader: **`sim_inventory_rain_fall.comp`**. Değişenler:
`sim_inventory_saturation.comp` (11 bağlama, 112 B itme sabiti),
`sim_inventory_surface_exchange.comp` (11 bağlama, 36 B), `sim_inventory_surface_commit.comp`
(8 bağlama) → `compile_sim_shaders.bat`. Disk önbelleği **v14** (v11–13 okunur, yağmur boş).

1. **Derleme.** Hata çıkarsa: `GasInventoryDevice::fields` artık 11;
   `gasInventoryScratchBuffers` 3 eleman döndürüyor; `GasSurfaceResult`'ın
   `padding[4]`'ü `rain_kg, rain_energy_j, padding[2]` oldu (64 B aynı);
   `MatterExchangeKind::Autoconversion` yeni; `gas_fog_present` →
   **`gas_liquid_water_present`** (adı değişti: artık sis VEYA yağmur).
2. **Exe/.spv zaman damgası.** Eski `.spv` ile itme sabiti/bağlama sayısı uyuşmaz →
   `inventory_step_held` "gas saturation dispatch/readback failed" veya
   "gas surface exchange dispatch failed". Önce shader derlemesini doğrula.
3. **Panel:** Thermal Liquid altında "Fog water in the air" — Rain Onset, Rain
   Formation (tau gösterir), Fog Ceiling. Gas Step → Physical gas inventory:
   "fog -> rain (autoconversion)" ve "rain -> liquid (… absorbed, … parcels)".
4. **IPC:** `fluid.set_param` `fog_autoconversion_kg_m3`, `fog_autoconversion_rate`,
   `fog_ceiling_kg_m3` (aralık dışı → hata, kırpma yok); `fluid.get` bunları ve
   `gas_rain_kg`, `gas_rain_cells`, `fog_autoconverted_kg`, `rain_absorbed_kg` döner.
5. **Canlı kabul:** `python scripts/test/rt_test_fog_ipc.py` (9 hüküm). Yeni üçü:
   - `fog_capped` FAIL (ortalama sis > 2× tavan) → otokonversiyon çekirdeği çalışmıyor
     ya da tavan kg/m³ → kg/hücre çevriminde yanlış. Önce `fog_autoconverted_kg`'ya bak.
   - `rain_forms` FAIL → defterde `autoconversion` kaydı yok.
   - `rain_reaches_liquid` FAIL, `gas_rain_kg` büyüyor → yağmur iniyor ama emilmiyor:
     iniş damgası (yüzey değişimi) ile düşüş aynı damgayı görmüyor, ya da commit
     yağmuru geri yazmıyor. `gas_rain_cells` küçük ve `gas_rain_kg` büyükse yağmur
     tabanda birikmiş (sıvının içinden geçmiş).
   - `water_conserved` artık buhar + sis + **yağmur** sayar; FAIL → yağmur yuvaları
     (9/10) snapshot/önbellek/arka uç değişiminde düşüyor.
   Kanıt: `docs/dev/fog_live.json`.
6. **Regresyon:** `rt_test_boiling_ipc.py` (`inventory_holds_steam` yağmuru sayar),
   `rt_test_gas_inventory_gpu_ipc.py` (havada = buhar+sis+yağmur, dönen = yoğuşma +
   precipitation), `rt_test_fusion_condensation_ipc.py`, `rt_test_fluid_mist_ipc.py`,
   `rt_test_heat_exchange_ipc.py`.
7. **CPP regresyon:** `scripts/test/gas_saturation_test.cpp` (otokonversiyon, uç hızı
   5,52 m/s @ 1 g/m³, sütun düşüşü korunumu ve iniş damgasında durma).
8. **Önbellek:** geri sar/ileri oynat — `gas_rain_kg` kareyle geri gelmeli; diske
   bake edip aç (v14).

★ **Sinsi — yağmur tabanda birikip "korunum PASS" der.** Yüzey hücresini atlayan
yağmur sıvının içinden tabana iner, orada bir parsel kütlesine kadar birikir. Kütle
korunur, test geçer, ama yağmur su yüzeyine değil tank dibine katılır. Belirtisi:
`rain_absorbed_kg` ≈ 0 iken `rain_parcels` > 0, ve `gas_rain_cells` taban satırında.

★ **Sinsi — sis hâlâ görünmüyor**, yağmur da (render bağlı değil; bulut işi).

---

# 2026-10-11 parti — T2: gizli ısılı erime/donma + yüzey yoğuşması

Bu parti, aşağıdaki 2026-10-10 partileriyle **aynı build'e** girer (onların
C++/shader'ları da henüz derlenmedi). Önce bu listeyi uygula; sonuçlar
aşağıdakileri maskelemez, aşağıdakiler bunu maskeleyebilir (gaz yüzey GPU
yolu iki partide ortak). Ayrıntı: [MADDE_TIPLERI_TASARIMI.md §6f](MADDE_TIPLERI_TASARIMI.md),
[BIRLESIK_MADDE_DOMAIN_TASARIMI.md §4.7](BIRLESIK_MADDE_DOMAIN_TASARIMI.md).

Değişen shader: **`sim_inventory_surface_exchange.comp`** (yoğuşma dalı) →
`compile_sim_shaders.bat`. Yeni dosya yok, vcxproj değişmedi.

1. **Derleme.** Hata çıkarsa ilk bakılacak yerler: kaldırılan
   `thermal_freeze_kelvin / thermal_viscosity_range / thermal_cold_viscosity`
   (APICSolverParams + `FluidDomainInfo`) alanlarına kalmış bir referans;
   `substanceMeltReleaseKelvin` (söküldü); `drawBirthTemperatureWarning`
   (yeni `substance_tag` argümanı); `updateThermalFreeze` (yeni
   `solid_change` argümanı). Shader: `max/min` double/float karışımı
   (diğer satırlarla aynı desen).
2. **Exe zaman damgası** yeni mi? Değilse aşağıdakilerin hepsi eski binary'yi ölçer.
3. **Panel (Domain → Thermal):** "Water: melts at 273.15 K, latent 334 kJ/kg …"
   satırı görünür; "Phase thresholds: set below … release above …" satırı
   **yok**. Domain "Resolved:" satırı "melts at 273.1 K" der (Sand'de "does not
   melt"). Bozuksa: UI hâlâ domain kopyasını okuyordur (derlenmemiş dosya).
4. **IPC okuma:** `fluid.get` içinde `thermal_mushy_particles`,
   `thermal_fusion_released_j`, `thermal_fusion_absorbed_j`,
   `thermal_condensed_kg`, `thermal_condensation_splits` var;
   `thermal_freeze_kelvin` **yok**. `fluid.set_param thermal_freeze_kelvin=…`
   "property of the domain's substance" hatasıyla reddedilir (beklenen).
5. **Canlı kabul:** `python scripts/test/rt_test_fusion_condensation_ipc.py`
   (uygulama açık, duraklatılmış). 8 hüküm:
   - `mushy_seen` / `latent_released` FAIL → füzyon geçişi çalışmıyor
     (gizli ısı yolu bağlanmadı); `water_freezes` FAIL ama ikisi PASS →
     150 karede yarı donmaya yetecek ısı çekilmedi (soğuk alanı kontrol et).
   - `freeze_ledger_matches_latent` FAIL → defter ile sıcaklıktan gerçekten
     düşülen ısı ayrıştı: **en önemli hüküm**, eskiden defter ödenmeyen ısıyı
     "ödendi" yazıyordu.
   - `boils` FAIL → sıcak köşe kaynamadı (bu parti değil, Ek 10 yolu).
   - `condenses` FAIL, `boils` PASS → soğuk taraftaki hava hücresinde buhar
     doygunluğu geçmedi veya GPU shader eski (.spv derlenmedi).
   - `condense_stats_match_ledger` FAIL → istatistik sıfırlama sırası
     (sıvı termal geçişi gaz adımından SONRA koşuyorsa okunan değer 0'dır).
   Kanıt dosyası: `docs/dev/fusion_condensation_live.json`.
6. **Regresyon (davranış değişti, eşikler yetmeli):**
   `rt_test_thermal_wax_ipc.py` (mum artık 100 K eşdeğeri gizli ısı öder;
   240 karede donmalı), `rt_test_snow_ipc.py` (`heat_field_melts`),
   `rt_test_boiling_ipc.py`, `rt_test_heat_exchange_ipc.py`. Mum FAIL ederse
   önce donma süresini oku: gizli ısı gerçek, hız sanatsal oranlardan gelir.
7. **CPP regresyon (derliyorsan):** `scripts/test/matter_substance_state_test.cpp`
   — yeni gizli ısı assert'leri (Water 10 K altında 0.125 donar ve 273.15 K'e
   döner; Ice 300 K'de 0.169 erir ve hâlâ Granular).
8. **Görsel:** soğuk alanda su tankı yüzeyden aşağı doğru, saniyeler içinde
   donar (eskiden 273 K'e değdiği karede donuyordu).

★ **Sinsi başarısızlık — karışık domain'de kar hiç erimez.** Snow + Water
karışık Matter domain'i ısı alanı altında: `thermal_mushy_particles` > 0
olmalı ve kar zamanla suya dönmeli. Mushy hep 0, kar 273.15 K'de takılı
duruyorsa `fusion_solid_fraction` sütunu karışık adımda (`MatterModelBatch`
emit+copy) düşüyor demektir; her adım kısmi erime sıfırlanır. Hata vermez,
"kar soğuk kalıyor" gibi makul görünür.

★ **Sinsi — yoğuşan parsel bölünmesi.** `thermal_condensation_splits` artarken
`particle_count` de artmalı; kütle defteri dengeli kalmalı. Yoğuşma çok
uzun sürerse yüzey parselleri 2× kütleye ulaşıp bölünür — bu beklenen.

---

# 2026-10-10 güncel parti — yüzey fiziği, momentum ve whitewater

**Project timeline reset ek düzeltmesi:** widget/model/settings reset debt,
UI layout restore sonrası tüketim ve authored start>0 rest-frame kapısı.
C++ build/live load kabulü bekliyor; shader değişmedi. UVW yeni binary'de
azaldı; yüzey görsel kalite kabulü açık.
[UVW_LIVE_AND_PROJECT_TIMELINE_FIX_2026_10_10.md](UVW_LIVE_AND_PROJECT_TIMELINE_FIX_2026_10_10.md).

**UVW/yüzey filtre ek partisi:** UVW gather/extrapolation/smoothing sparse tile,
fiziksel CSR, spherical+anisotropic render-centre filtresi ve centre sonrası
output CSR. C++ build/live kalite-hız kabulü bekliyor; shader değişmedi.
[UVW_SURFACE_FILTER_OPTIMIZATION_2026_10_10.md](UVW_SURFACE_FILTER_OPTIMIZATION_2026_10_10.md).

**SDF yüzey ek partisi:** fiziksel CSR bin, sparse çalışma tile'ları ve halo
yumuşatma; anisotropi destek sınırı düzeltmesi ve ortak API aşama sayaçları.
C++ build/live kabul bekliyor, shader değişmedi. Final dense SDF ABI korunuyor.
[SDF_SPARSE_WORK_OPTIMIZATION_2026_10_10.md](SDF_SPARSE_WORK_OPTIMIZATION_2026_10_10.md).

**Solid background ek partisi:** yalnız Solid viewport çizimine fokus/görünürlük
kapısı, background Play/cache için kısa event beklemesi. C++ build/live kabul
bekliyor; shader değişmedi. [SOLID_BACKGROUND_GPU_FIX_2026_10_10.md](SOLID_BACKGROUND_GPU_FIX_2026_10_10.md).

**İlk-kare/kapanış ek düzeltmesi:** kullanıcı derlemesinde gaz hold canlı
doğrulandı. Surface/summary readback komut sırası ve logger static-destruction
kapısı düzeltildi; C++ yeniden derlemesi ve canlı kabul bekliyor.
[GAS_READBACK_SHUTDOWN_FIX_2026_10_10.md](GAS_READBACK_SHUTDOWN_FIX_2026_10_10.md).

Önce bu checklist uygulanır:
[GAS_WHITEWATER_PHYSICS_COST_2026_10_10.md](GAS_WHITEWATER_PHYSICS_COST_2026_10_10.md).
C++ ile momentum/yüzey/dört whitewater shader'ı birlikte kullanıcı tarafından
derlenmeli. Kaynak kontrolleri PASS; bu partide build ve canlı kabul yapılmadı.
WW kapalı/açık maliyet, temas öncesi jet buharı, yerel momentum, ledger-grid
korunum ve rewind kabulü bekliyor. Aşağıdaki partiler de aynı build'e dahildir.

# 2026-10-10 son düzeltme — korunum ve Matter cache

Güncel kaynak partisi: [GAS_RESIDENCY_OPTIMIZATION_2026_10_10.md](GAS_RESIDENCY_OPTIMIZATION_2026_10_10.md).
Closed zero-advection inventory koruması, boiling telemetry toplamı, kayıpsız
Matter ikinci-grid/kgJ RAM payload'u, eksiksiz budget ve snapshot move birlikte
C++/simulation shader derlemesine girer. Özellikle sim_inventory_conserve SPV
eski kalmamalı. Canlı kütle/rewind/performans kabulü bekliyor; build yapılmadı.

# 2026-10-10 — kg/J GPU sahipliği: tek kullanıcı derlemesi

Kullanıcı önce kaynak değişikliklerinin bitmesini, sonra tek derleme ve test
istedi. Güncel kapsam/checklist: [GAS_INVENTORY_GPU.md](GAS_INVENTORY_GPU.md).
Sekiz inventory shader, iki ortak scalar wrapper, C++ owner/surface/cache/API
kayıtları ve proje açılışında kare 0 düzeltmesi aynı derleme partisine dahildir.
Codex build çalıştırmadı. Kaynak sözleşmesi/matematik/phase/solver kontrolleri PASS;
canlı performans ve fizik kabulü bekliyor. Fizik kg/J tam-grid roundtrip kaldırıldı;
CPU timeline/bake snapshot transferi ve compact parcel IO kapsam sınırıdır.

Dış terminalde, kullanıcı derlemesinden sonra durmuş disposable sahnede sırayla:

```powershell
python scripts/test/rt_test_gas_inventory_gpu_ipc.py
python scripts/test/rt_test_contact_heat_ipc.py
```

Önceki checklist'ler aşağıda tarihsel bağlamıyla korunur.

# Güncel kabul: MPM + grain GPU temas devamı

> **DEM uyku güncellemesi:** canlı revision=23. Dört repose A/B paritesi PASS;
> açık tane boyutu kapısı 4.04849° > 4° FAIL. 1.3M ölçek PASS, hızlanma 1.125x.
> Tam kabul açık; [son rapor](grain_sleep_ab_rev23/REPORT.md) ve
> [uyandırma incelemesi](grain_sleep_ab_rev23/WAKE_AUDIT.md). Eski checklist'ler tarihsel.
> **Kaynak revision=24 hazır, canlı kabul bekliyor.** Yerel itki audit'i ve fiziksel
> uyku saati: [uygulama/kabul planı](DEM_SLEEP_TRANSFER_IMPLEMENTATION.md).
> Aşağıdaki ortak build bekletme kuralı korunuyor; rev24 testleri sonraki eşleşen
> C++/shader build'inde koşulacak.

> **2026-10-09 yeni kapsam: build bekletiliyor.** Kullanıcı sıvı ve gaz sparse
> sisteminin tamamını kaynakta bitirip tek build/test istedi. Aşağıdaki eski
> parti checklist'leri son build için saklıdır; consumer migration kapanmadan
> build istenmez. Güncel çalışma:
> [MATTER_SPARSE_FULL_STORAGE_IMPLEMENTATION.md](MATTER_SPARSE_FULL_STORAGE_IMPLEMENTATION.md).

## Son kaynak partisi: compact MAC transfer + FLIP (revision 8)

1. Kullanıcı `RayTrophiStudio/source/shaders/compile_sim_shaders.bat` ve C++
   build alır. Yeni `SparseMacTransferGpu.cpp` proje kayıtlarında. On bir yeni
   `sim_sparse_mac_*` SPV ve ortak gövde kullanan dense/indexed P2G/G2P SPV'leri
   aktif runtime shader dizinine birlikte deploy edilmelidir. Codex derlemedi.
   Bu çalışma ağacında root `compile_shaders.bat`'ın çağırdığı
   `compile_shaders.ps1` mevcut değil; burada mevcut simulation batch yolu
   veriliyor. Root driver eksikliği bu aktarım partisinde değiştirilmedi.
2. Kaynak denetimleri:
   `python scripts/test/check_sparse_mac_transfer_contracts.py` ve
   `python scripts/test/check_sparse_mac_transfer_math.py` PASS. İkinci test
   CPU oracle'dır, GPU/GLSL testi değildir. Existing dispatch/pressure/viscosity,
   Matter GPU/grain ve domain panel source denetimleri de PASS.
   Capability/descriptor freshness PASS (663 method). Genel
   `verify_descriptor_claims.py` enum list/string uyumsuzluğunda durdu; sonucu
   PASS değildir. Yeni transfer alanları ayrı source contract ile denetlendi.
3. Build sonrası uygulama kullanıcı tarafından açılır. Boş/durmuş sahnede,
   başka IPC testi koşmazken ayrı terminalden sırayla:
   `python scripts/test/rt_test_sparse_pressure_ipc.py --transfer`
   `python scripts/test/rt_test_sparse_pressure_ipc.py --transfer --viscosity`
   Her iki probe dense/sparse reset parity, mass/momentum, centroid/hız ve
   `transfer_sparse_used`/`flip_sparse_used` seçimini ister. Silent dense
   fallback FAIL'dir. Geçici domain/source/madde cleanup edilir.
4. Tek geniş paket iki yeni probe'u da içerir:
   `python scripts/test/rt_test_matter_acceptance_ipc.py --extended`.
   Named-pipe Windows error 5 sandbox kaynaklıysa dış probe escalation ile
   tekrar koşulur; embedded script workspace kullanılmaz.
5. Tam sparse MAC kabulü henüz yok. Transfer havuzu compact, fakat host MAC,
   projection/contact device publication, weight ve FLIP fallback scratch
   hâlâ dense. Pool bytes ilave capacity'dir; VRAM/FPS kazancı raporlanmaz.
   Moving wall, topology değişimi, büyük 2D GPU, tüm sparse consumer'lar ve gaz
   depolaması sonraki kapılardır. Son iki probe henüz canlı koşulmadı.

## 2026-10-09 tamamlanan canlı sparse probe'lar

rt_test_sparse_pressure_ipc.py ve --viscosity dış süreçte sıralı PASS.
Dense/sparse mass/momentum ve centroid/hız eşitliği geçti. Son sahne boş/durmuş;
geçici domain/source/madde kaldırıldı. matter_sparse_live_2026-10-09.json raporu.
Pressure pool 13,364 byte; viscosity fixture pressure 99,492 + RHS 56,300 byte.
Host süreleri karışık (pressure 17.26/35.95 ms, visc 22.67/19.95 ms dense/sparse);
native GPU/FPS/cinematic kabulü yapılmadı. Full MAC/gas ve büyük 2D gate açık.

## Shader çıktı yazma devamı — kullanıcı build aracı

1. Hata footer'ı RayTrophiStudio/compile_shaders.bat'ten geliyor. Script artık
   compile_shaders.ps1'e delege eder; shader önce GUID geçici dosyaya derlenir,
   başarılı SPIR-V header kontrolünden sonra mevcut output değiştirilir.
   Compiler hatası eski SPV'yi truncate etmez; publish kısa I/O kilidini retry eder.
2. Aynı root için yeni ana script'in eşzamanlı ikinci koşusu named mutex ile
   reddedilir. Doğrudan glslc/diğer legacy batch'ler bu mutex'e katılmaz.
3. PowerShell 5 -ValidateOnly source plan PASS (275 entry, compiler çağrılmadı).
   check_shader_output_publish.ps1 yalnız AST publication fonksiyonunu çalıştırır;
   valid replace, invalid çıktı ve kilitli target korunması, unlock sonrası retry.
   Bu shader derlemesi veya GPU test değildir. Build kullanıcıda.
4. Kullanıcı aynı RayTrophiStudio/compile_shaders.bat dosyasını tekrar çalıştırır.
   Publish kalıcı olarak başarısızsa artık compiler syntax hatası diye gizlenmez;
   target yolu ve işletim sistemi hatası görünür. Runtime deployment da aynı
   güvenli publication yolunu kullanır. Her SPV atomiktir; bütün batch tek
   transaction değildir ve derleme sonuçları ancak kullanıcı koşusunda doğrulanır.

## Ek kaynak partileri: sparse tile temel ve GPU pressure

### Son devam: P2G/G2P/FLIP 2D dispatch

1. Kullanıcı son shader batch + C++ build: sim_dispatch.glsl kullanan P2G/G2P
   (normal + indexed), clear/normalize/window, sim_matter_copy/clear/contact
   SPV'leri birlikte güncel olmalı. Shader ABI'leri değişmedi; old SPV + yeni
   C++ 2D planı birlikte kullanılmaz. Core revision 7.
2. `check_fluid_gpu_dispatch_contracts.py` PASS; C++ test hedefinde
   fluid_gpu_dispatch_test.cpp beklenen çıktı:
   `PASS balanced GPU dispatch coverage, padding and 32-bit boundary`.
   Codex C++ testini derlemedi/çalıştırmadı.
3. İlk iki-satır sınırı 65535*256 + 1 eleman: 32768x2 logical workgroup; ikinci
   satır kaybolmamalı, aynı yüz/particle iki defa işlenmemeli. Sinsi: küçük
   fixture PASS olup >16.7M elementli gerçek dispatch'in hâlâ bozuk olması.
   C++ shape testi GPU kernel kabulü değildir; bu büyük-case GPU gate'i açık.
4. Diğer ajan DEM dosyalarında devam eder; bu kaynak partisi o dosyaları
   değiştirmez. Son IPC paketi iki tarafın source/build partileri tamamlandıktan
   ve diğer ajanın mevcut canlı testleri bittikten sonra bir kez sıralı koşulur.
5. Ana MAC/FLIP depolaması hâlâ dense. Bu kaynak hazırlığı memory/FPS veya tam
   sparse solver kabulü değildir; native timestamp ve gerçek resident byte gerekir.

### Son devam: sparse viscosity MAC RHS

1. Kullanıcı shader + C++ build: sim_sparse_viscosity clear/mark/capture/sweep
   ABI 13/68 ve dense sim_fluid_viscosity_rbgs ABI 11/52 ortak GLSL gövdesinden
   yeniden derlenir. SparseViscosityGpu.cpp/.h proje kayıtlarında; Codex derlemedi.
2. `python scripts/test/check_sparse_viscosity_contracts.py` PASS. C++/shader
   runtime testi değildir; fizik eşitliği sonraki gate'te ölçülür.
3. Son sıralı runner sparse_viscosity işini de içerir. Doğrudan probe:
   `python scripts/test/rt_test_sparse_pressure_ipc.py --viscosity`.
   Boş durmuş sahne, diğer agent testleri bitmiş ve son build alınmış olmalı.
   Nu=0.02 geçici maddeyle dense/sparse mass/momentum, hız/centroid ve pool
   resident byte eşitlik/savings aranır; sahne ve türetilmiş madde cleanup edilir.
4. Sinsi: tile sınırındaki pozitif MAC yüzünün sahibi dahil edilmezse viscosity
   yalnız bazı yüzlere uygulanır. Sparse bayrağı tek başına kabul değildir.
   Hareketli duvar ve yüksek nu/dt yakınsaması son fizik kapısında kalır.
5. Ana velocity, P2G/FLIP ve gas grid hâlâ dense. Yeni ölçü sadece viscosity RHS
   pool'udur. Full sparse solver/FPS başarısı iddia edilmez. Core revision 6.

### Güncel devam: Vulkan sparse pressure bağlı, tam sparse MAC/gaz açık

1. Kullanıcı `source/shaders/compile_sim_shaders.bat` + C++ build: yeni on
   sim_sparse_pressure_* shader ABI 16/80; değiştirilmiş divergence/gradient ve
   CG/window reduction SPV'leri birlikte güncel olmalı. SparsePressureGpu.cpp
   proje kayıtlarında. Codex build veya shader derlemesi çalıştırmadı.
2. Aşağıdaki S0 C++ testi ve `check_sparse_pressure_contracts.py` önce gelir.
   Son statik audit PASS; C++ testi henüz derlenip çalıştırılmadı.
3. Diğer ajanın canlı testleri bittikten sonra tek son paket:
   `python scripts/test/rt_test_matter_acceptance_ipc.py --extended`.
   Yeni sparse_pressure probe dense/sparse aynı fixture, mass/momentum,
   centroid/hız ve pressure resident memory karşılaştırır; scene cleanup yapar.
4. Kullanılınca UI/script/IPC pressure_sparse_used=true; aktif/allocated tile ve
   gerçek resident byte görünür. Sessiz sinsi sonuç: CPU/dense fallback başarılı
   görünüp sparse diye kabul edilmesi. Probe sparse flag ve pressure_on_gpu ister.
5. Bu devam yalnızca pressure scratch'ı sparse yaptı. MAC/viscosity/host grid,
   gaz storage/pressure ve GFM sparse portu açık. Full sparse/FPS kabulü sayılmaz.

### Önceki S0 kaynak kontrolü

1. Kullanıcı C++ test hedefinde `scripts/test/sparse_tile_grid_test.cpp`:
   `PASS sparse tile storage, MAC ownership, remap and swept support` beklenir.
   FAIL adres sahipliği, tile remap veya seyahat halo'su sorununu gösterir.
   Codex C++ derleme/test çalıştırması yapmadı.
2. `python scripts/test/check_domain_panel_fields.py`: PASS beklenir. Initial
   Model/Phase/Constitutive Model madde şemasına taşındı; panelde geri eklenmez.
3. S0 host temelinin aktif solver'da bellek/FPS kazancı göstermesi beklenmez.
   En sinsi yanlış kabul: eski dense storage + window dispatch'i gerçek sparse
   bellek sonucu saymak. S1–S4 açık; takip MATTER_SPARSE_TILE_GRID.md.
4. Aşağıdaki ortak saat/dinamik destek kabulü aynı son test paketinde korunur.

> **Durum:** AKTİF — 2026-10-08. Önceki ortak state partisi kullanıcı build PASS.
> Bu devamın C++/shader ve canlı kabulü açık. Ana plan kapanmadı; build kullanıcıda.

## Son kaynak partisi: dinamik destek + tek kabul paketi

Kaynak partileri tamamlanınca kullanıcı shader + C++ build alır; devam eden
testlerle paralel yeni IPC koşusu açılmaz. Bu devam derlenmedi/canlı kabul edilmedi.

1. Sekiz `sim_grain_fluid_*` shader (clear/hash/cells/solid/refresh/delta/reaction/apply)
   ABI 21/96; eski 13/32 SPV ile yeni kaynak test edilmez. C++ coupling pool release
   ve yeni prepare params/motion imzası kaynakta. Proje build'i kullanıcıda.
2. Kullanıcı C++ hedefinde matter_common_clock_test.cpp ve matter_grain_coupling_test.cpp.
   Güncel konumda Fluid destek edinir, uzaklaşınca bırakır, geri dönünce edinir.
   Partikül kimliği/kütlesi değişmeden trilinear lump mass partition korunmalı.
3. Boş durmuş sahnede tek dış komut:
   `python scripts/test/rt_test_matter_acceptance_ipc.py --extended`.
   Tek rapor `docs/dev/matter_acceptance_<UTC>.json`; H1 detayları koşu başına
   ayrı snapshot. Extended H1 fixture'ları disabled olarak incelemeye bırakılır.
4. Dinamik probe: uzaktaki Water grain desteğine GIRER, sonra ÇIKAR; destek
   device_rebin_each_tick, ortak saat enabled, held=false, fluid/grain/mpm=1/1/1.
   Liquid impulse residual <1e-6 N s. Sinsi: sıvı hareket eder fakat eski hücre
   eşlemesi yüzünden coupling hep 0 ya da hep aktif kalır. İkisi FAIL.
5. Native GPU kernel timings ayrı; host stage wall time, dispatch/transfer byte,
   working set ve alt adımlar ayrı ölçülür. Profiler unsupported ise maliyet
   ölçülmemiştir; 24 fps PASS sayılmaz. Native profiling instrumentation açık diye
   raporlanır, production performance proof değildir.
6. Tam kapanışa eksik kaynak/kabul: üç-owner porous pressure reaction, MPM/grain
   deformasyon-angular ve yüksek hızlı penetration/dt yakınsaması; 1.3M+ cinematic
   ölçek/boundary gerçek GPU maliyeti; termal enerji/elastik/gaz son kapıları.
   Runner implemented_gates_passed dönebilir, all_unified_matter_gates_closed=false
   kalır. Fizik terimlerini kapatarak veya hidden cap ekleyerek maliyet kapısı geçilmez.

## Önceki kaynak partisi: ortak saat

1. Kullanıcı shader + Release x64 build: altı `sim_grain_mpm_*` (count dahil,
   ABI 13/32, revision 2), beş `sim_grain_fluid_*` (ABI 13/32). Yeni
   MatterGrainFluidGpuCoupling.cpp/.h ve MatterCommonClock.h proje kayıtlarında.
   Codex build/uygulama başlatmadı; eski SPV'lerle yeni kaynak kabul edilmez.
2. Kullanıcı C++ test hedefi: `scripts/test/matter_common_clock_test.cpp`.
   Even ping-pong, authored cap, 100000 adımın keyfi 4096 ret almaması,
   NaN/Inf/index width ve grid interval'in CFL üst sınırını aşmaması.
3. Boş durmuş sahnede dış terminal: `python scripts/test/rt_test_matter_transport_ipc.py`.
   Common clock enabled, held=false, contact impulse >0; transport_steps >=
   continuum_grid_steps>0. Water eklenince sahipler 1/1/1, liquid_reaction_on_gpu=true,
   liquid momentum residual <1e-6 N s. Sinsi hata: host ikinci tepki ekler veya
   fluid adveksiyonu contact-corrected canonical hızı kullanmaz. İkisi FAIL.
4. MPM bloğu→kum ve kum→MPM havuzu: 24/60/120 fps, penetration/mass/momentum;
   saf DEM ve grain'siz MPM regresyonu. CFD-DEM destek weights hâlâ frame sampled.
5. Sinematik ölçek: 1.3M+ sahne, authored budget ve gerçek GPU buffer kapasitesi.
   Coordinator'da 1M particle/64-neighbour ret kapısı olmamalı. GPU kernel timestamp
   ile maliyet ölç; host_wait tek başına 24 fps PASS değildir. Porous owner izolasyonu,
   deformasyon support/angular yakınsama ve ana plan final kapıları açık.

Kaynak denetimi: grain/common-clock ABI, GPU indexed kontratları, IPC descriptor ve
capability, Python AST/proje/filters/JSON PASS. C++/shader/canlı kabul değildir.

## Önceki kaynak partisinin listesi (tarihsel)

1. **Final kaynak partisi sonunda shader + Release x64 build (kullanıcı):**
   `RayTrophiStudio/source/shaders/compile_sim_shaders.bat` beş `sim_grain_mpm_*`
   SPIR-V üretmeli. C++ projesinde `MatterGrainMpmContact.cpp/.h` kayıtları var.
   Eksik/bayat shader teması hazır gösterip geçiştirilmez, adım açık hatayla tutulur.
2. **Hızlı sahibi/modeli kontrol et:** Substance → granular_transport: Sand/Gravel/Ice
   dem, Soil mpm. Grain açıkken MPM material satırları görünmeli. Active Matter sayıları
   fluid/grain/MPM olarak ayrı. Yanlışsa solver değil doğum/transport çözümlemesi bozuk.
3. **Boş durmuş sahnede dış IPC smoke:**
   `python scripts/test/rt_test_matter_transport_ipc.py`. Beklenen grain=1, mpm=1,
   ready=true, held=false; GPU contact events ve grain impulse >0, residual
   <max(1e-7, impulse*1e-4). Ardından Water eklenince üç sayının hepsi 1 kalmalı.
   Test kaynaklarını temizler, kaydetmez. ★ En sinsi hata: ready=true/held=false ama
   temas dürtüsü sıfır; ya da Soil sessizce ikinci DEM tanesine dönüşür. İkisi FAIL.
4. **Core kaynak regresyonları (kullanıcının test hedefi):**
   `matter_substance_state_test.cpp`, `matter_grain_coupling_test.cpp`,
   `matter_grain_params_test.cpp` (MPM spin/pile dışında). MPM, sıvı
   mass/volume/momentum bin'ini değiştirmemeli; parcel_cell=-1, drag/su exchange
   yalnız Fluid sahibine gitmeli. Kimlik ve doğum kütlesi regresyonlarını koru.
5. **Regresyon ve fizik yakınsaması:** saf DEM H1 suite; grain kapalı Soil MPM;
   kum→Soil havuzu ve hareketli Soil bloğu→kum. 24/60/120 fps karşılaştır, kütle/
   momentum ve en derin penetration ölç. Normal temas var diye şekil/maliyet PASS
   sayma: MPM pozisyonu DEM frame'i boyunca sabit olduğundan yüksek hızda risk var.
6. **Üç sahip + collider:** Fluid/MPM/grain aynı domain. Fluid drag+kaldırma açık,
   porous projection MPM varken kapalı olduğu UI/IPC raporunda görünür. Ayrı owner
   weight'leri yapılana kadar B5'in üç-sahip eşitliği kapanmaz. Disk cache/render
   hazırlığı/bridge başka ajanın alanı; bu partide değiştirilmedi.
7. **Ana plan final kapıları:** ortak üç-sahip scheduler; deformasyonla temas
   desteği/angular momentum; per-substance MPM parametre tüketimi; termal enerji;
   karma elastik MPM/gaz-grain; save/open/legacy/numerical tuning; T0 uzman/Matter
   eşitlikleri. T6 uzman domain sökümü bu kabul sonuçları olmadan yapılmaz.

Kaynak denetimi: grain/MPM ABI ve dispatch sözleşmesi PASS; IPC capability ve
663 metot descriptor eşliği PASS; AST, Release aynaları, proje/filters ve JSON PASS.
Bunlar C++/shader derlemesi veya canlı fizik kabulü değildir.

## Ek parti (ayrı ajan): gizli tavanlar söküldü — grid 512, DEM 100k/1M, çarpıştırıcı 4096, Max Particles 10M

Kural: kodda ayar tavanı yok; panelde yalnız sürükleme aralığı (Ctrl+tık ile
üstü yazılır). Kalan tek tavanlar **indeks genişliği** (doğruluk sınırı) ve
kendi mesajlarıyla raporlanan bellek/cihaz sınırları.

0. **Shader derle:** `compile_sim_shaders.bat` (sim_matter_grain*.spv). Shader
   revizyonu 16→17. Eski .spv ile adım **held** olur ve "revision" der — bozuk
   değil, derlenmemiş demektir.
1. **Grid tavanı (hızlı, bağımsız):** IPC `fluid.set_param {domain, max_auto_resolution: 768}`
   → `fluid.get` `max_auto_resolution` 768 kalmalı (önceden 512'ye geri yazılıyordu).
   Enforce budget kapalıyken `resolution` en büyük eksende ~768'e çıkmalı.
   Panelde Max Auto Resolution'a Ctrl+tık 3000 yazılabilmeli.
   Bozuksa: değer 512'ye dönüyorsa eski binary.
2. **Max Particles:** `fluid.set_param max_particles: 50000000` → `fluid.get` 50M;
   panel 50M göstermeli (önceden 10M gösteriyordu).
3. **DEM >1M (2D dispatch):** Sand emitter'lı sahneyi Max Particles 3M ile oynat.
   1M'yi geçince taneler hareket etmeye devam etmeli; `fluid.matter_models`
   `step_held` false. Durursa mesajı oku: bütçe / storage-buffer / 2^24 indeks
   sınırından hangisi olduğunu ayrı söyler.
   ★ Sinsi: 1M üstünde **taneler hareket ediyor ama bir kısmı çift/titrek** —
   bu 2D katlamanın yanlış olduğu anlamına gelir (y satırları aynı indeksi
   okuyor). Tane sayısı ile `pile.grains` eşit, `scattered_fraction` 100k
   altındakiyle benzer olmalı.
4. **Çarpıştırıcı >4096 yüz:** yoğun bir mesh (≥50k üçgen, ör. subdivided zemin)
   çarpıştırıcı yap. Adım held olmamalı; `runtime.collider_faces` gerçek sayı.
   `host_ms` ilk karede BVH kurulumunu öder, statik çarpıştırıcıda sonraki
   karelerde ödemez. Hareketli büyük mesh her kare yeniden kurar (beklenen).
   Bozuksa: ilk kare saniyeler sürüyorsa patch birleştirme hâlâ O(N²).
5. **C++ test:** `scripts/test/matter_grain_collider_bvh_test.cpp` PASS
   (patch birleştirme union-by-size'a geçti; eşitlik/eşitsizlik iddiaları aynı).

6. **Render instance tavanları kalktı:** fluid splat 1M ve klasik parçacık
   toplamı 300k (`ParticleRenderBridge.cpp`) söküldü. Tek tavan cihazın TLAS
   `maxInstanceCount`'u (genelde 16.7M): `VulkanDevice::createTLAS` fazlasını
   kırpar ve bir kez ERROR loglar. 1M+ splat'lı sahnede Rendered modda tüm
   parçacıklar görünmeli; log'da `exceed the device maxInstanceCount` yoksa
   sınıra gelinmemiştir. ★ Sinsi: parçacıkların bir kısmı RT'de yok ama
   raster'da var → bu log'a bak.

Kalan bilinen tavanlar: SurfaceSDF modunun raster uyumluluk proxy'si 32768
(seyreltme/LOD, parçacık sınırı değil), eski `GasSimulator` 512/256³ (ayrı iş).
Diğer ajanın `sim_matter_contact.comp`'u da 1D dispatch: ~16.7M thread'de aynı
sınır; aynı `setLinearGroups` + 2D katlama kalıbı uygulanmalı.

### Canlı sonuç (2026-10-08, RTX 3060 12 GB, build 13:54) — `scripts/test/rt_test_scale_ceilings_ipc.py`

- **A grid PASS:** `max_auto_resolution` 600 ve `max_particles` 50M yazıldığı gibi
  kaldı; voxel .007 ile grid **572×29×572** kuruldu, düğme 512'ye geri yazılmadı.
- **B DEM:** 32 258 yüzlü düz terrain çarpıştırıcı (eski 4096) ve **1 009 066 tane**
  held olmadan adımlandı (eski 100k / 1M). r=8 mm'de 2622 substep, ~40 s/kare.
  Enerji sınırda, taneler domain içinde → 2D katlama tutarlı.
- **Buldu, düzeltildi (derleme bekliyor):**
  1. 1.09M'de held: "grain buffer of 2224 MiB exceeds the device storage-buffer
     limit of 2048 MiB". Sayının kendisi sığıyordu (history 1.09M×1536 B ≈ 1.6 GB);
     taşıran büyüme payıydı (1.5×). Pay artık cihaz sınırına göre kırpılıyor
     (`MatterGrainGpu.cpp`). Yeniden: `--grains 1300000` held olmamalı.
  2. Tane doğumu kaynak başına kare başına **262 144 deneme** ile sınırlıydı
     (`MatterDomainSources.inl`) → kare başına ~200k doğum. Artık istekle ölçekleniyor
     (büyük isteklerde 4×). Yeniden: `--window-frames 2 --grains 600000 --source-radius 1.7
     --grain-radius .008` ilk karede ~600k doğurmalı.
- **Kalan gerçek tavan (bu GPU):** contact history tek buffer'da tane başına 1536 B;
  Vulkan storage-buffer sınırı 2 GiB → **~1.39M DEM tanesi**. Aşmak için history'yi
  birden çok buffer'a bölmek gerekir (ayrı iş). Mesajı artık bunu söylüyor.

### Canlı sonuç 2 (kullanıcı build'i, aynı gün)

- Doğum denemesi düzeltmesi tuttu (kare başına 200k → 329k), ama her kare **tam
  329 272** doğdu: APIC sıvısının aşırı paketleme koruması (`spawn_cells × 16`,
  1.7 m küre / 0.1 m voxel) tane doğumuna da uygulanıyordu. Tanede atlanıyor
  (`MatterDomainSources.inl`; tane çakışma filtresi zaten var). Sıvı doğumu
  değişmedi. Yeniden: aynı komut, 3 karede 1.3M'ye ulaşıp held olmadan adımlamalı.
- Kalıcı değil, yalnız bu sahnede: 987 816 tanede 6 kare, enerji sınırda.

### Canlı sonuç 3 — PASS
1.3M tane 3 karede doğdu, 6 kare held olmadan adımlandı (büyüme payı kırpması
dahil), enerji sınırda, 32k yüzlü çarpıştırıcı. ~50 s/kare: 2622 substep, sınır
`damping` (r=8 mm'de mutlak 4 N·s/m kayma sönümü × 24 temas slotu); son substep'te
yalnız 15 702 temas, tane başına en çok 3. Substep başına maliyet ~19 ms/1.3M tane.

### Parti 4: DEM substep — kayma sönümü kararlılık sınırından çıkarıldı (yalnız C++, shader değişmedi)

`MatterGrainGpu.cpp`: host CFL kayma dashpot'unu (7 c_t × 24 temas) sayıyordu;
shader o dashpot'u zaten temas başına kesiyor (bütçe bir substep'te kaymayı
ters çeviremez) → koşulsuz kararlı. Sınırda artık yalnız normal (restitution) dashpot.

1. **Maliyet (hızlı):** aynı ölçek komutu (1.3M, r=8 mm). Beklenen: substep
   2622 → **~712**, `substep_limit` yine `damping` (artık normal dashpot),
   kare ~50 s → **~14 s**. Hâlâ 2622 ise eski binary.
2. **Fizik regresyonu (asıl kabul):** `python scripts/test/rt_h1_grain_suite.py`
   — settle, coexist, repose (yığın açısı), G2 kollarının hepsi önceki gibi PASS.
   Kayma sönümü artık daha seyrek substep'te kesmeye takılabilir (r=8 mm'de etkin
   ~1.45 N·s/m; kayma ~0.5 ms'de söner, kare 16 ms) — görsel fark beklenmez.
   ★ Sinsi: yığın açısı birkaç derece **düşerse** (taneler daha uzağa kayıyorsa)
   kesme sönümü fazla zayıflatıyor demektir; sayı PASS bandında kalsa da
   önceki koşudaki repose_angle_deg ile karşılaştır.
3. Sonraki kaldıraç (yapılmadı): normal dashpot da 24 temasın hepsi aynı yönde
   varsayımıyla sayılıyor; doğruluk sınırı ~340 substep. Ölçüm 2'den sonra.
- **Parti 4 madde 1 PASS (16:18 build):** 1.3M tane, r=8 mm: substep 2622 → **712**,
  kare ~50 s → **~15.3 s** (3.3×). Serbest düşüş eğrisi öncekiyle aynı
  (com_y 2.1661 / 2.1656, KE 2540 / 2541 J). Madde 2 (süit) koşuyor.
- **Parti 4 madde 2 — süit (16:18 build): 14/17, kalan üçü bu partiyle ilgisiz.**
  settle: substep 164 → 138, sınır artık `accuracy`; kalan enerji 4.9e-13 J/tane (önce 4.5e-13).
  repose (önce → şimdi): μr .05 18.2→24.7 (tepe alçaldı: açı uydurması bu kolda gürültülü),
  .1 23.1→22.3, .3 34.8→35.3, .1 küçük tane 24.4→22.2. Tane boyutu bağımlılığı
  1.3° → 0.1°. Hesap: repose ve motion sahnelerinde kayma kesmesi **hiç devreye
  girmiyor** (dt eşiğin 4-6× altında) → fark yalnız dt (3.4e-5 → 4.1e-5 s) ve
  koşudan koşuya gürültü; "sönüm zayıfladı" etkisi yok.
  Düşenler: contracts (bayat 4096 dizesi — düzeltildi, tek başına PASS);
  panel_fields (diğer ajanın 493995a'sında kalkan 'Initial Model', 'Phase',
  'Constitutive Model' — SUBSTANCE_MIGRATIONS'a girmeli); motion (önceden de
  düşüyordu, 493995a kaydı sweep −0.015 m: küre çapı 0.15, merkez y .15 → tek
  katman yığının 2.5 cm üstünden geçiyordu. Merkez .07 yapıldı → **PASS**, sweep +0.067 m).

### Parti 5: DEM kararlılık sınırı ölçülen temas sayısıyla (yalnız C++, shader değişmedi)

`MatterGrainGpu.cpp`: Gershgorin sınırı 24 temas varsayıyordu; normal dashpot
yaylarla orantılı olduğundan toplam ζ √temas ile büyür → 712 substep'i o
belirliyordu. Artık temas = geçen karenin ölçülen maksimumu + 4, [12, 24]
(12 = eşit kürelerde öpüşme sayısı; süitte en yoğun yığın 8-10). Karede daha
fazlası ölçülürse kare yayımlanmaz, 24 ile yeniden koşulur (temas geçmişi o
karede sıfırlanır; `cfl_contact_budget_retry`). MPM/ortak saatle bağlı adım 24'te kalır.
Restitution / sönüm katsayıları değişmedi.

1. **Maliyet:** 1.3M komutu. `cfl_contacts_per_grain` 12 (ilk kare 24), substep
   712 → **~384**, kare ~15 s → **~8 s**. `fluid.matter_models grain_diagnostics.runtime`.
2. **Fizik:** `python scripts/test/rt_h1_grain_suite.py --only settle repose dense static convergence`.
   repose açıları ve settle kalan enerjisi Parti 4 değerleriyle ±2° / aynı mertebe.
   `cfl_contact_budget_retry` dense ve repose kayıtlarında **nadiren** true olmalı
   (her karede true ise +4 payı yetmiyor → hint sürekli 24'e döner, hız kazancı kaybolur).
3. ★ Sinsi: yığın kayıp patlarsa / KE tane başına yükselirse ama step held değilse,
   bir karede temas sıçraması yakalanmadı demektir — `max_contacts_per_grain` >
   `cfl_contacts_per_grain` olan ve retry=false bir satır arayın (olmamalı).
- **Parti 5 canlı (21:56 build): etkisiz** — `cfl_contacts_per_grain` 24 kaldı, substep 712.
  Sebep: tane adımı `mpm_contact` nesnesini **her zaman** alıyor (MPM parseli yokken
  de); koşul nesnenin varlığına bakıyordu. Artık `mpm_contact->active()`. Yeniden derle.

### Parti 6: DEM maliyet sayaçları (SHADER DEĞİŞTİ → compile_sim_shaders.bat; revizyon 17→18)

`grain_diagnostics.runtime.cost_last_substep` (son substep, tüm tanelerin toplamı):
`scanned_bucket_slots`, `grain_pairs`, `history_probes`, `contactless_grains`,
`collider_nodes_visited`. Tane sayısına bölerek okunur.
1. Shader derlenmeden koşarsa adım held: "shader revision mismatch" — beklenen, derle.
2. 1.3M komutu: Parti 5 beklentileri (cfl 12, ~384 substep) + sayaçlar dolu.
   Okunacak oranlar: scanned/grain_pairs (komşu listesi kazancı), history_probes/
   temas (geçmiş arama), contactless_grains/grains (havadaki pay → uyutma/multirate).
- **Parti 5+6 canlı (22:15 build, 22:17 shader):** 1.3M dökülen tane: `cfl` 12,
  substep **384** (ilk kare 712, ipucu 24'ten başlıyor), kare ~15 s → **~9.4 s**,
  yörünge aynı (KE 2539.6/2540.3 J), retry yok.
  Sayaçlar (son substep): bucket slotu 15.04M (11.6/tane), tane-tane temas 15 064 →
  **bulunan temas başına ~1000 slot okuması**; temassız tane %98.9; geçmiş araması
  temas başına ~1; BVH 1 düğüm/tane. GPU 7.2 s/kare (~19 ms/substep), host 1.4 s.
  → Komşu listesi (Verlet) en büyük kaldıraç. Yığın oranları süit kayıtlarında.
- **Parti 5 fizik (alt küme 5/6):** static, convergence, settle, dense, repose PASS.
  Uyarlamalı temas: dense cfl 12-14 (maks ölçülen 10), repose 12-13 (maks 9),
  **retry 0** (dense 0/246, repose 0/300). motion FAIL: küre itiyor (+0.022 m) ama
  eşik .03; önceki koşu +0.067 — tek katman yığının yerleşimi koşudan koşuya
  değişiyor, kütle merkezi ölçüsü bu test için kararsız (dt'den bağımsız).
- **Repose A/B (aynı build, `--repose-arms mu_r_.05 mu_r_.1 --repose-contact-resolution 51`
  = Parti 4 dt'si, 290 substep):** tepe .05: 0.255 (Parti 4: 0.254, Parti 5: 0.237);
  .1: 0.273 (Parti 4: 0.286, Parti 5: 0.267). Aynı dt'de iki koşu arası fark
  (0.286 / 0.273) gözlenen kaymayla aynı büyüklükte → dt etkisi tek koşuyla ayırt
  edilemiyor; Parti 5 kabul. Açı aynı dt'de 24.7° / 17.3° → **açı uydurması ±4°'lik
  kapıya göre fazla gürültülü**; fiziği değiştiren partiler için çok tohumlu koşu
  veya tepe/profil ölçüsü gerekir.
- settle: tanelerde kalan dönme enerjisi dünkü koşudan çok daha yavaş sönüyor
  (kare 480: 1.2e-4 J / dün 1.9e-16 J); bu sahnede dt değişmedi (138 substep).
  Dikey eksende dönme yalnız burulma sürtünmesiyle söner (varsayılan 0) — kaotik
  olabilir. Kapı yalnız "büyüyor mu"ya bakıyor. Açık gözlem, regresyon değil.

### Parti 7: DEM Verlet komşu listesi (SHADER + C++; revizyon 18→19) — `docs/dev/DEM_VERLET_LISTESI.md`

Önce `compile_sim_shaders.bat` (yeni: `sim_matter_grain_list_clear`, `sim_matter_grain_list_build`),
sonra build. Üç dönen bucket tablosu → tek tablo + tane başına 32'lik liste;
liste GPU'da bayrakla yeniden kurulur (pay 0.5 r), kare başında zorunlu kurulum.
Temas kümesi aynı olmalı (yalnız toplama sırası değişir) → fizik farkı float gürültüsü kadar.

1. **Revizyon:** shader derlenmeden koşarsa "shader revision mismatch" → derle.
   Kernel tablosu 17 bağlama bekliyor; eski .spv ile pipeline kurulmaz.
2. **Maliyet (1.3M):** `cost_last_substep.neighbour_candidates` ≈ taneler × liste boyu
   (dökmede çoğu 0) — önceki 15.04M bucket slotuna karşı. `neighbour_list_builds`
   kare başına ≥1; serbest düşüşte birkaç-onlarca, durgun yığında ~1.
   Kare süresi ~9.4 s'den aşağı. Substep 384 aynı kalmalı.
3. **Fizik (kısa):** `python scripts/test/rt_h1_grain_suite.py --only base history static settle dense wet`
   — hepsi PASS; settle kalan enerji ve dense sayıları Parti 5 ile aynı mertebe.
   ★ Sinsi: liste bir çifti kaçırırsa taneler birbirinin İÇİNDEN geçer ama adım
   held olmaz. Belirti: yığın normalden alçak/yayvan, `grain_pairs` aynı sahnede
   Parti 6'dakinden belirgin az. dense ve settle bunu yakalar.
4. Taşma: "grain neighbour list full" → 0.5 r kesmede 32'den fazla komşu (aşırı
   sıkışma); normal yığında olmamalı.

**Parti 7 sonuç (00:03 build, 2026-10-09 09:45–09:55):** madde 2 PASS (önceki oturum).
Madde 3: `base history static settle dense wet` → **6/6 PASS**, taşma mesajı yok.
- base: g = 9.8101 / 9.8100 (iki dt).
- static: tutma kolu sürünme 8.8e-10 m; statik kapalı kol 0.146 m kayıyor (ayırt ediyor).
- settle: kalan enerji 5.6e-13 J/tane (Parti 4: 4.9e-13). Dönme enerjisi kare 480'de
  1.9e-4 J (Parti 5: 1.2e-4; dün gece aynı build'le yarım kalan koşuda 8e-19) →
  koşudan koşuya kaotik, listeyle ilgisiz; açık gözlem olarak duruyor.
- wet: kuru rms 0.324 m / ıslak 0.205 m (2652 köprü); su kayması 6.5e-7 kg.
- dense: 4096 → 17.3 ms (p95 27.5), 16384 → **34.0 ms** (p95 46.6; Parti 3 kapısı ≤41.1);
  substep 194, sınır accuracy, maks temas/tane 9, kuru kütle sapması 0.
  Önceki koşuların dense temas sayısı kayıtlı değildi (log üzerine yazılıyor) — bu koşunun
  4873 / 72659 temas değerleri bundan sonrası için referans.

### Parti 8: DEM temas geçmişi tek bank, yerinde (SHADER + C++; revizyon 19→20) — `docs/dev/DEM_VERLET_LISTESI.md` "Sonraki 1"

Geçmiş tane başına **1536 B → 672 B** (+ 12 B blok haritası/sahip/maske; 1544 → 684).
İki bank + her yeniden sıralamada cihazda toplama (permute) yerine: her tanenin kalıcı
bir **geçmiş bloğu** var, host her kare tane→blok haritasını tutuyor (yalnız sıra
değişince / doğumda 4 B/tane yüklenir). Shader kayıtları yerinde günceller; bulunan kayıt
kendi slotuna, yeni temas alt adım başında boş olan slota yazılır. `sim_matter_grain_clear`,
`_permute`, `_permute_copy` söküldü (.comp/.spv, kernel tablosu, bat, vcxproj).
Fizik birebir aynı olmalı (yay değerleri aynı, yalnız saklama yeri değişti).
Not: plandaki "liste konumuna bağla + çift başına tek kayıt" yapılmadı — tek bankta j'nin
i'nin kaydını okuması i'nin yazımıyla yarışır; liste 32 slot > 24 geçmiş slotu.

1. **Revizyon:** önce `compile_sim_shaders.bat`. Eski .spv ile "shader revision mismatch"
   (19 ≠ 20) → derle. Eski `sim_matter_grain_clear/permute*.spv` dosyaları
   `x64/Release/shaders`'ta kalabilir; artık yüklenmiyor, zararsız.
2. **Kontrat (derlemesiz, PASS):** `python scripts/test/check_matter_grain_contracts.py`.
3. **Fizik (asıl kabul):** `python scripts/test/rt_h1_grain_suite.py --only base history static settle dense wet`
   — 6/6 PASS ve sayılar Parti 7 sonucuyla aynı mertebe (settle 5.6e-13 J/tane,
   dense 4873 / 72659 temas, 16384 ≤ 41 ms).
   ★ Sinsi: geçmiş her kare kaybolursa hiçbir şey çökmez, yığınlar yine durur.
   **static** bunu yakalar: tutma kolunda sürünme 8.8e-10 m'den mm'lere çıkar (kapı
   5e-4). history arm'ında `history_remapped` true iken static'in hâlâ tutması = blok
   haritası sıralamayı doğru izliyor. FRESH bayrağı temizlenmezse de aynı belirti.
4. **Maliyet (1.3M, `scripts/test/rt_test_scale_ceilings_ipc.py --grains 1300000`):**
   `working_set_bytes` ~1.1 GB düşmeli; substep süresi Parti 7 ile aynı. `history_probes`
   temas başına ~1'den birkaça çıkabilir (artık sabit sıra yok, dolu slot maskesi
   taranıyor) — oranı not et; maliyet farkı görünmüyorsa sorun değil.
5. **Tavan (opsiyonel, VRAM izin verirse):** `--grains 2500000` storage-buffer
   sınırına takılmadan adımlamalı (eski tavan ~1.39M; yeni hesap ~3.1M, en büyük
   buffer artık geçmiş 672 B × N). Takılırsa mesaj hangi buffer olduğunu söyler.
6. Taşma: "…or 24 history records were held in one substep" yeni mesaj — eski temaslar
   kaybolurken aynı alt adımda yenileri gelip toplam 24'ü aşarsa. Normal yığında görülmemeli.

### Parti 9: DEM host kare maliyeti (yalnız C++; shader değişmedi) — Parti 8 ile aynı build'de

**Ölçüm (2026-10-09, 00:03 build, 1.3M dökülen tane r=8 mm, 384 substep):** kararlı karede
duvar ~3.3 s, bunun yalnız **~0.95 s'i `gpu_wait`**. Tane host aşamaları ~1.7 s:
merge ~670, order ~525, prepare ~475, publish ~55 ms. Kalan ~0.6 s tane adımı dışında.
Komut: `python scripts/test/rt_test_scale_ceilings_ipc.py --skip-grid --grains 1300000
--grain-radius .008 --source-radius 1.7 --window-frames 2 --steps 6` (satırlar artık
`host_ms` ve `working_set_mb` basıyor).

Kaldırılan işler (hepsi gereksiz iş; fizik ve sıra bit düzeyinde aynı kalmalı):
- **order:** bölmedeki kimlik `stable_sort`'u (hemen ardından (hücre, kimlik) toplam
  sırasıyla yeniden sıralanıyordu); hücre sıralamasının eşitlik bozucusu artık anahtarda
  (sort içinden `particle_id`'ye rastgele erişim yok); `selectMatterParticles` tam kopya +
  compact + parçacık başına 31 dizi kopyası yerine sütun bazlı `gatherFrom`.
- **merge:** 1.3M düğümlü `unordered_set` kimlik kontrolü → düz `MatterParticleIdIndex`;
  parçacık başına kopya → sütun bazlı `copyRangeFrom`.
- **prepare:** yayımlanmış kimlik eşlemesi `unordered_map` → aynı indekste kalan tane için
  arama yok, gerekirse düz indeks. Kuru karede (ıslak/kuplaj/kuvvet alanı/ortak saat yok)
  48 B/tane sıfır kuplaj satırı artık kurulmuyor/yüklenmiyor (62 MB/kare); cihazda zaten sıfır.
- `FluidParticles` kopya yolları tek sütun listesinden (`forEachColumnPair`) geçiyor.

1. **Derleme:** `FluidParticles.h` şablon değişikliği birçok dosyaya dokunur; hata çıkarsa
   ilk bakılacak yer `forEachColumnPair` / `copyRangeFrom` / `gatherFrom`.
2. **Birim (hızlı):** `scripts/test/matter_grain_coupling_test.cpp` (bölme/seçme/birleştirme,
   değişen nüfus reddi) — PASS kalmalı.
3. **Maliyet (asıl):** yukarıdaki 1.3M komutu. Beklenen kaba hedef: merge ≤ ~150 ms,
   order ≤ ~250 ms, prepare ≤ ~350 ms; duvar ~3.3 s → **~2.2 s**. `gpu_wait` değişmemeli
   (~0.95 s). Hangi aşama beklenenden yüksek kalırsa bir sonraki kaldıraç odur.
4. **Fizik:** Parti 8'in süit komutu (base history static settle dense wet). Sıra aynı
   toplam sıralama olduğundan sayılar Parti 8 ile aynı mertebe.
   ★ Sinsi: sütun kopyasında bir dizi yanlış hizalanırsa (örn. `granular_*` ya da
   `pore_water_mass_kg` başka taneye kayarsa) hiçbir şey çökmez. **wet** arm'ı su
   kütlesi kaymasını (6.5e-7 kg mertebesi) ve köprü sayısını ölçer; o sayı büyürse buradan.

**Parti 9 ek (aynı build):**
- Yeni zaman alanları (`grain_diagnostics.runtime.host_ms`): `upload` (durum yükleme
  partisi), `download` (geri okuma; artık önüne bir `synchronize()` kondu, yani yalnız
  okuma — GPU hesabı `gpu_wait` içinde kalır), `step_total` (tüm tane-domain adımı;
  ölçek testindeki `wall_s` eksi bu = tane adımı dışında geçen süre, 1.3M'da ~0.65 s
  açıklanmamıştı). `cell_order: {disorder, resorted}`.
- **Koşullu hücre sıralaması:** popülasyon geçen yayımla aynıysa ve komşu çiftlerin
  ≤ %2'si sıra dışıysa (`kMatterGrainResortDisorder`) yeniden sıralanmıyor → durum
  cihazda kalıyor (`state_resident` true, ~88 MB yükleme yok). Doğum/emme varsa her zaman
  sıralanır. Durgun yığın zaten sıralıydı (önceden de resident); kazanç yavaş akan yığında.
5. Ölçek komutunda satır başına `resident` ve `cell_order` okunur. Dökme sırasında
   `resorted` true (doğum + düşüş) beklenir. ★ Sinsi: `resident` true iken yığın
   donuyorsa (taneler kare kare aynı) host durumu cihazla ayrışmış demektir —
   settle'ın `tail_rms_range_m` (≈5e-12) ve dense yığın sayıları bunu yakalar.

### Parti 10: DEM uyuyan taneler (SHADER + C++; revizyon 20→21, push constant 112→128 B) — `docs/dev/DEM_UYUYAN_TANELER.md`

Parti 8–9 ile aynı build. **Varsayılan AÇIK** (`sleep=true`, 2 mm/s, 0.2 s); A/B için
`fluid.set_grain_settings(domain=..., sleep=False)`. Blok başına 3 kelime (sahip/maske/
dinlenme); geçmiş haritası buffer'ı 4 kelime/tane. IPC tarifi yeniden üretildi.

1. **Revizyon:** `compile_sim_shaders.bat` (dört grain kernel'i). Eski .spv → "shader
   revision mismatch" (20/21). Push constant 128 B ile pipeline kurulamıyorsa kernel
   tablosu (`SimulationComputeVulkan.cpp`, 17, 128) ile shader bloğu uyuşmuyor demektir.
2. **Kontrat (derlemesiz, PASS):** `check_matter_grain_contracts.py`.
3. **Fizik, uyku AÇIK:** süit alt kümesi (Parti 8 komutu) + `repose`.
   Beklenen: settle/static/dense PASS; settle'da `sleeping_grains` kuyrukta 256'ya
   yakın, kalan enerji daha da düşük (uyuyanın hızı tam 0). static'in **sürünen** kolu
   (statik sürtünme kapalı) hâlâ ≥1e-2 m kaymalı — kaymıyorsa tane uyumuş demektir
   (eşik/süre fazla gevşek). repose açıları Parti 5/7 ile ±2° içinde.
   ★ Sinsi: **yığın havada donar** (uyuyan tane, altı çekilince düşmez) ya da
   **çığ erken durur** (repose açısı birkaç derece yükselir) — hiçbir şey çökmez,
   yığın sadece "biraz dik" görünür. repose'u `sleep=False` ile bir kez daha koşup
   açıları karşılaştır; fark ±2°'yi aşarsa eşik/süre ayarı ya da uyandırma kuralı yanlış.
4. **Maliyet:** 1.3M komutunu `--steps 30` ile (yığın otursun). `sleeping_grains`
   artmalı, `gpu_wait` düşmeli. Aynı komutu `--no-sleep` ile koşup karşılaştır.
5. Kapalı durumlar: hareketli çarpıştırıcılı sahne (`rt_grain_motion_ipc.py`) ve sıvı
   kuplajlı coexist kolları uyku yüzünden değişmemeli (`sleeping_grains` 0 ya da
   yalnız kuplajsız tanelerde).

### Sparse S1: karışık sıvı şeridinde compact kanonik MAC hızı (SHADER + C++) — `docs/dev/MATTER_SPARSE_S1_SIVI_GPU.md`

Parti 8–10 (DEM) ile **aynı build**. Diğer ajanın son kaynak partisi (compact MAC
aktarım + FLIP, sparse grid storage/GPU transaction) de bu build'de ilk kez derleniyor.

1. **Shader derlemesi:** `compile_sim_shaders.bat` + `compile_fluid_window_shaders.bat`.
   13 yeni `sim_sparse_mac_*` girişi; gövdesi `.glsl`'e taşınan yoğun ikizler
   (divergence ×3, subtract_gradient ×2, matter_contact, zero_solid_faces, advect_tail,
   viscosity) da yeniden derlenmeli. glslc hatası `sim_mac_lane.glsl`'i gösterirse:
   ortak adres yardımcısı, `pc.nx/ny/nz` isteyen her gövde onu `pc`'den sonra include eder.
2. **Kontratlar (derlemesiz, hepsi PASS):** `check_sparse_mac_canonical_contracts.py`,
   `check_sparse_mac_transfer_contracts.py`, `check_fluid_gpu_dispatch_contracts.py`,
   `check_fluid_window_contracts.py`, `check_matter_gpu_contracts.py`,
   `check_sparse_viscosity_contracts.py`, `check_sparse_pressure_contracts.py`.
3. **Yoğun yol regresyonu (önce bu):** sparse KAPALI bir sıvı sahnesi ve karışık
   Su/Toprak — davranış öncekiyle aynı olmalı. Yoğun kernel'ler artık ortak gövdeden
   derleniyor; adres formülü aynı (kâhin doğruladı) ama derlenmiş kodun ilk koşusu bu.
4. **Canlı kabul (boş/durmuş sahne, dış process, sıralı):**
   `python scripts/test/rt_test_sparse_pressure_ipc.py --transfer`
   `python scripts/test/rt_test_sparse_pressure_ipc.py --transfer --viscosity`
   Beklenen: sparse kolda `transfer_sparse_canonical` true, `transfer_sparse_blocked`
   false, durum metninde "canonical"; dense/sparse centroid/hız farkı ≤2e-5, model
   kütle/momentum assertion'ları PASS (sınır değişmedi).
   ★ Sinsi: `transfer_sparse_blocked` true ise compact yol hiç koşmamış, şerit yoğun
   kalmıştır — prob bunu FAIL eder; panelde "Dense (compact blocked: neden)" görünür.
   ★ İkinci sinsi: parite PASS ama `transfer_sparse_canonical` false — eski binary
   ya da eski .spv (sahiplik hiç kurulmadı). Exe/spv zaman damgasına bak.
5. **Bellek (bilgi):** sparse kolda yoğun `GridDomainVelX/Y/Z` ve `ScratchVel*` tamponları
   yok. `perf.get_gpu_memory` ile iki kolu karşılaştır; kazanç yüz bankalarının
   yarısından azdır (hesap S1 notunda), sayfa havuzu ekstradır.

**Canlı sonuç (2026-10-09, exe 11:50) — S1 PASS:** 8 kontrat PASS. `--transfer` ve
`--transfer --viscosity` PASS: sparse kolda `canonical=true`, `blocked=false`, sayfa havuzu
208 KB; dense/sparse centroid 4.7e-10, hız 8e-9 / 2.0e-8 (sınır 2e-5). Küçük sahnede
sparse kol 40 ms, dense 21 ms (sabit ek yük; ölçek ölçümü yapılmadı). Madde 3 (yoğun
regresyon) ve 5 (bellek) ayrıca koşulmadı.

**Aynı build, DEM (Parti 8–10):** probun `dispatches == substeps+2` varsayımı Parti 9'un
gerçek sayısına (`4 × substeps`) göre düzeltildi. Uyku AÇIK: base/history/settle/dense
PASS; **static** (son alt adımda `sticking_contacts` 0, hold ratio 0.29) ve **wet**
(`liquid_bridges` 0) FAIL; **repose** 590 sn'de ilk kolu bitirip kaldı. Uyku KAPALI
(`RT_GRAIN_SLEEP=0`, probe anahtarı): static ve wet PASS, repose iki kolu bitirdi
(mu_r .05 → 19–21°, .3 → 37.9°). Yani kırılma Parti 10 uykusundan: uyuyan tane temas
ve köprü sayaçlarına girmiyor ve/veya uyandırma kuralı eksik. Düzeltilene kadar uyku
varsayılanı kapatılmalı ya da sayaç/uyandırma düzeltilmeli (`DEM_UYUYAN_TANELER.md`).

### Sparse S1b ilk parti: yoğun P2G ağırlık bankalarını kaldırma (yalnız C++)

2026-10-09: kullanıcı S1/DEM testleri devam ederken kaynak çalışmasına izin verdi.
Bu parti çalışan S1 binary'sinde yok; mevcut S1 testlerinin sonucu ayrı kaydedilir.
S1b'nin katı yüz ağırlıkları ve hücre alanları henüz bu partide taşınmadı.

1. Kullanıcı sonraki C++ build'ini alır; bu parti shader veya ABI değiştirmedi.
2. Derlemesiz: `python scripts/test/check_sparse_mac_canonical_contracts.py`;
   ağırlık bankalarının compact sahipte serbest bırakılması, ayrılmaması, yoğun
   yolda ayrılması ve compact başarısızlık sonrası yeniden deneme PASS beklenir.
3. Boş/durmuş sahnede, mevcut testler bittikten sonra dış process'ten sıralı:
   `python scripts/test/rt_test_sparse_pressure_ipc.py --transfer` ve
   `python scripts/test/rt_test_sparse_pressure_ipc.py --transfer --viscosity`.
   S1'in canonical/blocked, fizik paritesi ve kütle/momentum kapıları aynen geçmeli.
4. Sparse kapalı karışık Su/Toprak ve saf sıvı regresyonunu tekrarla. Aynı sahnede
   sparse açık → kapalı → açık geçişlerinde sıvı hareketi sürmeli, canonical alanı
   sparse seçimini izlemeli. Saf sıvı compact sahipliğe geçirilmedi.
5. Bellek: aynı çözünürlükte eski S1 ile bu parti karşılaştırılırsa ek düşüş
   `3 * max(face_count) * 4` B olmalı (40³ hücrede 787200 B); yalnız sıvı şeridi.
   Toplam VRAM ölçümünde driver/havuz etkileri olabilir. `var_*_weight` ve hücre
   bankaları hâlâ yoğun; bu sonucu tam sparse storage kabulü sayma.

### Sparse S1b ikinci parti: compact katı yüz ağırlıkları (SHADER + C++)

2026-10-09, kaynakta; mevcut canlı test binary'sine dahil değil. Hücre alanları
henüz taşınmadı. İlk partide kalan `var_*_weight` artık compact sıvı sahibinde yok.

1. Kullanıcı `compile_sim_shaders.bat` + `compile_fluid_window_shaders.bat`, ardından
   C++ build alır. Yeni basınç girişleri `init_mac_weights` / `spmv_mac_weights`
   18/80; var/porous divergence ve var gradient dense/compact gövdeleri de yeniden
   derlenmeli. Header `SparseMacSolidWeightsGpu.h`; projeye
   `SparseMacSolidWeights.cpp`, `SparseMacSolidWeightsGpu.cpp` ve bu header kaydedildi.
2. Derlemesiz kontroller: `check_sparse_mac_solid_weight_contracts.py`, canonical,
   transfer, pressure, dispatch, window, Matter ve viscosity kontratları PASS.
3. Kullanıcı C++ test hedefinde `scripts/test/sparse_mac_solid_weights_test.cpp`
   ile `source/src/Physics/Fluid/SparseMacSolidWeights.cpp` paketleme testini koşar
   (source/include include yolu). `SparseMacSolidWeightsGpu.cpp` bu saf paketleme
   testine gerekmez. Kesirli ağırlıklar, 8 sınırı/son yüz/clipped tile, slot sırası,
   retired/boş topology, duplicate/bozuk layout ve atomik reddetme PASS beklenir.
4. Devam eden testler bittikten sonra boş/durmuş sahnede dış process'ten sıralı:
   `python scripts/test/rt_test_sparse_pressure_ipc.py --transfer --solid-weights`
   ve `python scripts/test/rt_test_sparse_pressure_ipc.py --transfer --solid-weights --viscosity`.
   Başka etkin collider olmamalı; prob kendi kesirli AABB collider'ını yaratır/siler.
   Canonical true, blocked false; centroid/hız ≤2e-5 ve kütle/momentum PASS.
   Resident muhasebesi ≥ `active_tiles * 576 * 12 * 4` B olmalı. Bu alt sınır yoğun
   bankaların yokluğunu tek başına kanıtlamaz; ayırıcı kaynak kontratı da geçmeli.
5. İlk partinin collider'sız `--transfer` / `--transfer --viscosity`, sparse kapalı
   karışık + saf sıvı, açık/kapalı geçişleri ve DEM wet/coexist regresyonları korunmalı.
   Porozite ağırlıkları da aynı paketleme yolundan gelir; ıslak DEM kollarını atlama.
6. Maliyet: ek tile listesi indirme/senkronizasyonu + CPU paketleme var. Aynı sahnede
   alt-adım/kare süresini önceki S1 ile karşılaştır. Ek kaldırılan yoğun katı ağırlık
   bankası `sum(face_count) * 4` B (40³ hücrede 787200 B); karşılığında üç compact
   sayfa alanı eklendi, net kazanç etkin/ayrılan tile sayısına bağlıdır.

### DEM uyku düzeltmesi — revizyon 22 (SHADER + C++; kabul bekliyor)

Uyku kapatılmadı. Revizyon 21'in static/wet kırılması yukarıdaki canlı sonuçta
kayıtlıdır. Revizyon 22: uyuyan tane için gerçek son-substep contact/bridge audit'i,
kuvvet/tork dengesi + float konum hassasiyeti toleransı, bridge/denge/context wake
ve rest sayacına CAS yazımı. 17 descriptor / 128 B push / 4 dispatch-substep ve
tane başı GPU bankası boyutları aynı. Ayrıntı: `DEM_UYUYAN_TANELER.md` son bölüm.

1. **Kullanıcı build'i:** `compile_sim_shaders.bat` (dört grain girişi yeni
   `sim_matter_grain_sleep.glsl`'i içerir), sonra C++ build. Shader revizyonu 22
   olmalı; eski 21 SPIR-V ile host yayını revision mismatch nedeniyle reddeder.
   Bu partinin build/GLSL derlemesi ajan tarafından çalıştırılmadı.
2. **Derlemesiz:** `check_matter_grain_contracts.py` ve
   `check_matter_grain_sleep_contracts.py` PASS. İkinci dosya kaynak/spec kontrolü;
   shader'ı sayısal çalıştırmaz, GPU kabulünün yerine geçmez.
3. **Önce hızlı canlı kapı**, boş/durmuş sahne, başka collider/force-field yok,
   dış terminalden: `python scripts/test/rt_test_grain_sleep_ipc.py`.
   Beklenen: gevşek `1 m/s + .01 s` uyku ayarı bile desteksiz taneyi donduramaz;
   sleep açık/kapalı düşüş paritesi ≤2e-5. Yerleşmiş tek tane sleep açıkta gerçekten
   uyur (`sleeping_grains=1`) ama contacts/sticking/max_contacts ≥1,
   contactless=0; uyku kapalıda sleeping=0. Prob kendi kaynak/domain'ini siler.
4. **Asıl fizik:** `python scripts/test/rt_h1_grain_suite.py --only base history static settle wet dense`.
   Static tutan kolun sayaçları pozitif, kayan kol ≥1e-2 m; wet köprüleri pozitif,
   su bilançosu ve cohesion RMS kapıları aynı. Eşikler gevşetilmedi. Hareket/coupling
   regresyonu için `--only motion coexist` ayrıca PASS kalmalı.
5. **Repose A/B:** önce `python scripts/test/rt_h1_grain_runtime_ipc.py --repose-only --repose-arms mu_r_.05 mu_r_.3`
   uyku açık; sonra aynı komutu `RT_GRAIN_SLEEP=0` ile kapalı koş. Dört kola giden
   tam repose kabulü daha sonra: `mu_r_.1`, `mu_r_.1_r_.0175` de koşulmalı. Açı farkı
   ±2° içinde, kaynak sayısı tamamlanmış ve kuyruk fizik kapıları PASS olmalı.
   Her kolun toplam süresini, GPU wait/substeps/list rebuild/history reset ve
   sleeping fraction'ı kaydet. Önceki 590 s farkının nedeni henüz kanıtlanmadı.
6. **1.3M maliyet A/B:** `python scripts/test/rt_test_scale_ceilings_ipc.py --skip-grid --grains 1300000 --grain-radius .008 --source-radius 1.7 --window-frames 2 --steps 30`
   ve aynı komut `--no-sleep`. Yığın oturmuş son pencereleri karşılaştır;
   sleeping fraction artarken `gpu_wait` düşmeli, toplam duvar/host aşamaları
   ayrı okunmalı. Geçerli popülasyon/durum, destek kaybı/çarpıştırıcı hareketi
   görülmeli; hızlanma iddiası ölçümden önce yapılmaz. Yeni kuvvet toleransı
   bütün taneleri uyanık bırakıyorsa veya açı/creep'i bozuyorsa kabul kapanmaz.
7. **Sparse partileriyle aynı build:** S1b'nin collider/viscosity dense-compact
   kapılarını da sonraki build'de koş; bu uyku düzeltmesi sparse bankaları değiştirmedi.

**Kullanıcı canlı sonucu (2026-10-09), revizyon 22: 4/4 PASS.**
`--only sleep static wet settle`: sleep 33 s, static 250 s, settle 12 s, wet 64 s.
Desteksiz düşüş ve gerçekten uyuyan tanenin contact/sticking/CFL muhasebesi PASS.
Static kayan kol creep 0.146423115 m; settle mass/COM/RMS drift 0,
enerji 6.690211889e-15 J/tane (resident tail 5). Wet köprü 2680,
cohesion RMS 0.204118119 m, grain water 0.837685364 kg, su drift 5.488811610e-7 kg.
Sonraki kapılar: madde 5 repose A/B ve madde 6 büyük-N maliyet; ayrıca bu build'de
base/history/dense ve motion/coexist henüz bu kullanıcı çıktısında raporlanmadı.


### Revizyon 23 — gereksiz komşu wake/atomik maliyeti (kaynak hazır; build bekliyor)

Revizyon 22 canlı A/B kayıtları `docs/dev/grain_sleep_ab_rev22/` altında.
Repose uyku açık `mu_r_.05` saçılma 0.317333 > 0.30: FAIL. `mu_r_.3` açıları
30.03675° açık / 32.83456° kapalı, fark 2.79781° > 2°: FAIL. Kapalı iki kolun
kendi fizik kapıları PASS. Bu yüzden dört hızlı PASS, tam uyku kabulü değildir.

1.3M açık koşuda 90. kare sleeping=30, KE=155.0077 J; 180. kare sleeping=420072
(%32.31), KE=0.0627299 J, mass=7434.8235 kg. Son enerji öteleme RMS hızını
4.108 mm/s yapar; görünürde sabit olması 2 mm/s + yüzey dönme kapısını geçtiği
anlamına gelmez. Eşik veya repose kabul sınırları gevşetilmedi.

Kaynakta bulunan somut sorun: hızlı/dengesiz tane zaten uyanık her komşuya WAKE
bırakıyor; bu adayın dinlenme sayacını tekrar başlatıyor ve aktif yoğun yığında
çok fazla atomik yazı üretiyordu. Revizyon 23 coherent rest okumasından sonra
sadece uyuyan komşuya atomik WAKE bırakır. Uyanık aday zaten her alt adımda gerçek
relative motion/kuvvet/tork uygunluğunu hesaplar. Pozitif rest CAS yazımı WAKE'i
korur; uyuyan sahibin sayacı geçici 0 göstermez. Sıfır rest yazımı zaten uyanık
sahipte gereksiz atomik işlemi atlar. Wake okuması yalnız bit varsa atomicAnd
kullanır; okumadan sonra gelen wake sonraki alt adıma korunur. Uyku kapalıyken
rest atomikleri/denge hesabı atlanır; yeniden açılınca host context wake geçerlidir.
Descriptor/banka/dispatch/push ABI değişmedi. Shader ve host revision=23.

Kaynak/spec kontrolleri PASS; bu sonuç derleme veya GPU kabulü değildir. Çalışan
uygulama revizyon 22 kalır; revizyon 23 için kullanıcı shader + C++ build'i gerekir.
Sonrasında sleep/static/wet/settle, motion/coexist, dört kollu repose A/B ve aynı
1.3M 180-kare maliyet/enerji/bounds A/B yeniden koşulmalı. Repose hatasının bu
maliyet düzeltmesiyle giderildiği henüz iddia edilmiyor.

DEM + granular MPM aynı Matter domain'de çalışıyor: Sand DEM + Soil MPM dış IPC
`rt_test_matter_transport_ipc.py --owners-only` PASS; 17 temas, impulse=0.13090269
Ns, residual=1.16415e-10 Ns, ortak saat 130 transport / 7 continuum grid adımı.
Bu elastik MPM veya üç-sahipli sıvı kabulü değildir: varsayılan tam probda 12
örnekte liquid support hiç aktif olmadı ve prob FAIL. Ayrı teşhis açık kalır.


**Tamamlanan 1.3M revision 22 A/B:** 180 adım; iki koşuda ölçek kapıları PASS.
Son 20 medyan duvar 5.225 / 6.530 s,
GPU wait 4660.9 / 5451.4 ms (açık/kapalı).
Açık sleeping 420072/1.3M; duvar hızlanması 1.250x.
Repose FAIL nedeniyle tam uyku kabulü kapanmadı.
Rapor: [grain_sleep_ab_rev22/REPORT.md](grain_sleep_ab_rev22/REPORT.md).


**Revizyon 23 kısa canlı uyku kapısı — 2026-10-09: PASS.**
`python scripts/test/rt_test_grain_sleep_ipc.py`, dış IPC; shader revision=23.
Gevşek uyku ayarında desteksiz düşüş açık/kapalı paritesi PASS, sleeping=0.
Yerleşmiş tek tane: kapalı sleeping=0, açık sleeping=1; her ikisinde son-substep
contacts=1, sticking=1, max_contacts_per_grain=1 ve contactless=0. Prob kendi
source/domain'ini temizledi. Wet/motion/repose/large-N revizyon 23 kapıları bu kısa
koşuda çalıştırılmadı. Log: `grain_sleep_ab_rev22/sleep_rev23_quick.log`.


**Revizyon 23 tam repose + 1.3M A/B (2026-10-09):** Dört açık/kapalı açı paritesi
PASS; açık tam repose tane boyutu kapısı 4.048489° > 4° FAIL,
kapalı 3.323032° PASS. İki koşunun kendi kol kapıları PASS.
1.3M 180 adımlık ölçek kapıları iki koşuda PASS; son 20 medyan duvar
5.750 / 6.470 s (açık/kapalı),
hızlanma 1.125x; açık sleeping 490007/1.3M.
Tam kabul kapanmadı. Kaynakta hareket veto'su `balanced` üzerinden kuvvet dengesizliği
gibi wake yayıyor; enerji/itki/destek ayrımı henüz uygulanmadı. Bu tur core kaynak
şartları değiştirilmedi. Rapor: [grain_sleep_ab_rev23/REPORT.md](grain_sleep_ab_rev23/REPORT.md).

### Taneler maddeyi izler — "Enable discrete grains" söküldü (yalnız C++ + UI; shader yok)

Kullanıcı bulgusu (2026-10-09): Sand kaynağı "DEM" görünüp bağlı MPM akışı üretiyordu;
domain'deki gizli anahtar (varsayılan kapalı) sessizce MPM'e düşürüyordu. Artık anahtar yok:
**bir domain'de granüler ve `granular_transport=dem` bir madde varsa (Default Substance,
bağlı madde ya da o domain'e akan kaynak) DEM koşar.** Karar `Fluid::matterGrainOwnership`
(MatterSubstanceState.cpp), her adımda kaynaklar emit etmeden önce `grain.enabled`'a yazılır;
canlı parçacık varken değişmez (`reset_pending`, kütle/sahiplik doğumda sabit). DEM istenip
bloker varsa MPM koşar ama **söyler**: Solvers'da kırmızı satır + kaynak panelinde tek satır.
`fluid.set_grain_settings(enabled=…)` artık hata; eski kayıtlardaki `enabled` sessizce düşer.
IPC: `fluid.matter_models` → `grain_ownership {wanted, enabled, reset_pending, substance, blockers}`.
Ayrıca substance düzenleyicide sürükleme düzeltildi (değer her kare geri okunuyordu).

1. **Kontrat (derlemesiz, PASS):** `check_matter_grain_contracts.py`, `check_domain_panel_fields.py`,
   `audit_ipc_capabilities.py`. Birim: `matter_grain_params_test.cpp` (enabled reddi + eski kayıt).
2. **Elle (asıl):** yeni Matter domain + Sand preset kaynak → Solvers: "Solver: DEM grains (Sand)",
   kaynak panelinde "Solver here: DEM grains". Oynat → tek tek taneler (bağlı akış değil).
   Sınırı Open yap → reddedilmeli (taneler istendiği için). Sand'den Derive + `granular_transport=mpm`
   ata → reset → "MPM continuum" ve akış bağlı.
   ★ Sinsi: yeni madde düzenleyicide sürüklenip bırakılınca değer kalmalı; geri dönüyorsa eski exe.
3. **Süit:** `python scripts/test/rt_h1_grain_suite.py --only base history static settle dense wet`
   — script'lerden `enabled=True` kaldırıldı; readiness kolu artık `grain_ownership.wanted` okur.
   ★ Sinsi: bir kol FAIL değil ama `transport_owners.grain` 0 → madde DEM istemiyor (kaynak
   maddesi/ faz Gas) ya da `reset_pending` — `grain_ownership`'e bak.
4. Bilinen sınır: cache'ten ortadan devam eden sahnede (parçacık var, adım yok) `enabled`
   kaydedilmediği için `reset_pending` görünür; reset gerekir.


### DEM uyku aktarımı — revizyon 24 (kaynak; build/canlı kabul bekliyor)

Hareket veto'su/gerçek kuvvet dengesi ayrıldı. Komşu bildirimi audit talebi;
dengeli hedef yaşını korur. Post-damping/Coulomb reaksiyon değişimi + bounded
forecast hedef kütle/ataletine göre süzülür. Rest fiziksel saniye, CFL dt context
wake nedeni değil; skipped history süresi tutulur. Metadata +4 B/capacity;
17/128 ve 4 dispatch/substep aynı. Aktif frontier E2 henüz yazılmadı.
Sıra ve sınırlar: [DEM_SLEEP_TRANSFER_IMPLEMENTATION.md](DEM_SLEEP_TRANSFER_IMPLEMENTATION.md).
İlk kullanıcı kapısı: `python scripts/test/rt_h1_grain_suite.py --only sleep sleep_transfer`.
Rev23 canlı rakamları rev24 kabulü değildir. Bağımsız diğer ajan görevi:
[HANDOFF_S1B_CELL_FIELDS.md](HANDOFF_S1B_CELL_FIELDS.md).

### Madde UI tek otorite — U1–U5 (SHADER + C++ + UI + script) — `docs/dev/MADDE_UI_TEK_OTORITE.md`

"Taneler maddeyi izler" (U0, yukarıda) ile **aynı build**. Diğer ajanın DEM uyku
revizyonlarıyla da aynı build; grain shader'ına (`sim_matter_grain.glsl`) dokunulmadı.

Ne değişti:
- **U1 madde düzenleyici:** okunur etiketler ("Friction (deg)", "Melt temperature (K)"; üstüne
  gelince script anahtarı), bölümler maddenin kullanabildiğine göre (anahtarı — `combustible`,
  `meltable` — her zaman görünür), solver ayarları kapalı "Advanced: solver numerics",
  "Show unused sections", seçicide arama. Sürükleme düzeltmesi (U0) burada.
- **U2 tane malzemesi maddede:** `grain_friction/rolling/twisting/restitution/tangential_ratio/
  packing_fraction/real_radius_m/water_capacity_fraction/absorption/drying/contact_angle_deg`
  maddede; köprü yüzey gerilimi sıvının `liquid_surface_tension_n_m`'i. Adım her kare DEM
  maddesinden kopyalar (`applyMatterGrainSubstance`). Sahiplik sırası: **kaynak > bağlı madde >
  Default Substance** (dökülen kazanır). Bir domain'de bir tane malzemesi; ikinci DEM maddesi
  not olarak söylenir. `wet_grains` türetilir (kapasite > 0 ve sıvı / ıslak doğum).
  Doğum nemi kaynakta: `flow_source.grain_birth_saturation`. `set_grain_settings` eski
  anahtarları **yeni yerini söyleyerek** reddeder; eski kayıtlar düşer ve log'a yazılır.
  Matter sekmesi malzemeyi salt okunur gösterir.
- **U3:** `stiffness_n_m` → `stiffness_scale` (k = scale × 8e5 × r; 1 = 2e4 N/m @ 25 mm).
  Eski kayıt ölçeğe çevrilir. Panelde sertlik "from the radius" satırı, ölçek İleri'de.
- **U5 kar:** MPM gerilmesine sıkışma sertleşmesi (`granular_compaction_hardening` ξ,
  `granular_compaction_limit` θc; 0 = kapalı, diğer maddeler birebir aynı) — GPU shader +
  CPU ikizi, push constant boşlukları kullanıldı (64 B değişmedi). Yeni yerleşik **Snow**
  (MPM, geçici değerler: E 1.4e5, ν .2, ξ 10, θc .025). `attr.stats` artık
  `granular_plastic_volume` okur.
- **Testler:** `rt_grain_material.py` çevirici — eski grain anahtarlarını türetilmiş test
  maddesine taşır; tüm grain script'lerine tek satır (`install_grain_material(client)`),
  diğer ajanın iki uyku script'i dahil. Yeni `rt_test_snow_ipc.py`.

Sıra:
1. **Shader:** `compile_sim_shaders.bat` (`sim_fluid_granular_stress_update` +
   `sim_matter_stress_update`). Push constant 64 B aynı; kayıt değişmedi.
2. **Kontratlar (derlemesiz, 21/21 PASS):** `for s in scripts/test/check_*.py`. Birim:
   `matter_grain_params_test.cpp` (taşınan anahtar reddi, eski kayıt ölçeği, madde uygulaması).
3. **Madde kütüphanesi:** `python scripts/test/rt_test_substance_profiles_ipc.py` — Snow,
   grain_* ve yüzey gerilimi alanları.
4. **Grain süiti (asıl regresyon):** `python scripts/test/rt_h1_grain_suite.py --only base history static settle dense wet repose`
   — sayılar önceki kabulle aynı mertebe olmalı (varsayılanlar birebir taşındı).
   ★ Sinsi: süit PASS ama malzeme hiç uygulanmıyor → `fluid.matter_models(...)['grain_ownership']['material']`
   test değerlerini (ör. friction .5, restitution) göstermeli; `substance` "T: <domain> Sand grains"
   olmalı. Sand gösteriyorsa çevirici kaynağı yeniden hedeflememiş.
   ★ İkinci sinsi: **wet** kolunda köprü 0 → `wet_grains` türetilmemiş (kapasite 0 ya da sıvı
   görünmüyor); `grain_ownership.wet_grains` / `liquid` alanına bak.
5. **Kar (U5, yeni):** `python scripts/test/rt_test_snow_ipc.py` — compaction (Snow pv_max > 1.005,
   kapalı kol ≤ 1.005), cohesion (Snow RMS < 0.8 × gevşek), rest. Değerler geçici: FAIL bir
   kalibrasyon bilgisidir, kodu geri alma sebebi değil. `plastic_volume_readable` FAIL ise
   pv host'a indirilmiyor demektir (karma MPM yolu).
6. **Elle (panel):** Sand kaynağı → Matter sekmesinde "Grain material: Sand" salt okunur;
   Solvers'ta "Contact stiffness … (from the radius)". Sand'den Derive → `grain_friction` değiştir
   → kaynağa ata → Matter sekmesi yeni değeri gösterir. Islak kum: türetilmiş maddede
   `grain_water_capacity_fraction` > 0 + kaynakta "Wet at birth".
7. Bilinen sınırlar: bir domain'de tek tane malzemesi (ikinci DEM maddesi not olarak);
   MPM malzemesi (kar dahil) **Default Substance**'tan okunur — kar sahnesinde domain'in
   Default Substance'ı Snow olmalı; `melt_product` (eriyen kar Water etiketi) yazılmadı.

**Sonuç (2026-10-09, derlendi):** 1–5 PASS. Kontrat 21/21, kütüphane, grain süiti (base history static settle wet repose; dense fiziği tamam, yalnız görsel jpg yazılamadı — dosya kilitli), kar 4/4 (pv_max 1.89, kapalı kol 1.0). `grain_ownership.material` çevrilmiş test değerlerini gösteriyor. Düzeltme: çevirici artık yalnız değişen alanları yazıyor (canlı tane varken packing yeniden gönderilince reddediliyordu). 6 (elle panel) bekliyor; kar değerleri hâlâ geçici.

**Ek (derlenmedi): kar ambient'te su doğuyordu.** Özel sıcaklık girilmemiş bir kaynak domain ambient'inde (293 K) doğuruyordu; bu Snow'un erime noktasının (273.15) üstü, yani her parçacık sıvı modelinde koştu (kullanıcı sahnesi: granular 0 / fluid 10416). Test bunu kaçırdı çünkü 263 K'yi zorluyordu. Düzeltme: `Fluid::substanceAmbientBirthKelvin` — erir granular/elastic madde (Snow, Ice) ambient daha sıcaksa erime noktasının 10 K altında doğar, sonra zamanla erir. Emitter, kaynak paneli ("… poured 10 K below its melt point") ve `flow_source.list` (`fluid_birth_kelvin`, `fluid_born_solid`) aynı fonksiyonu kullanır. Kontrol: `python scripts/test/rt_test_snow_ipc.py` → yeni `ambient_birth_is_snow` PASS (birth < 273.15, granular > 0, fluid 0). Bozuksa: emitter eski ambient yolunda.

Ambient doğumu derlendi ve elle doğrulandı (kullanıcı: domain sıcaklığı yükselince kar eridi). **Ek 2 (derlenmedi): termal kuvvet alanı karı ısıtmıyordu.** Termal alan yalnız MSF'yi (mesh) ısıtıyordu; madde parçacığına hiç bağlı değildi. Şimdi `coolThermalLiquid` aynı alan listesini alır (MSF profili, ambient'e eklenerek): alanın içindeki her parçacık yerel sıcaklığa `thermal_contact_cooling_rate` ile gevşer, **thermal liquid kapalıyken bile** (yerleştirilen ısı kaynağı açık bir istektir; kapalıyken yalnız alan içindekiler değişir). Erime mevcut model yenilemesiyle olur. IPC: `thermal_field_heated_particles`; panel: "Heated by thermal fields: N". Kontrol: `rt_test_snow_ipc.py` → `heat_field_melts` PASS (heated > 0, max K > 273.15, fluid > 0). Test artık sahnedeki kuvvet alanlarını kapatıp geri açar. ★ Sinsi: heated > 0 ama fluid 0 → sıcaklık host'ta değişiyor ama GPU-yerleşik parçacıkta model yenilemesi görünmüyor. Bozuksa ne: host→GPU sıcaklık/model senkronu.

Ek 2 derlendi ve elle doğrulandı (ısı alanındaki kar eriyor). **Ek 3 (derlenmedi): Seed Fluid Now / Fill Domain karı su doğuruyordu.** İkisi aynı tüketiciden (`MatterDomainSynchronization.inl`) düz ambient'le doğuruyordu; kaynak kuralı (`substanceAmbientBirthKelvin`) oraya da uygulandı — seed edilen parçacık domain'in Default Substance'ıdır. Ayrıca panelin "Seed Fluid Now" düğmesi Fill modunda da modu SeedBox'a zorluyordu; artık seçili modu iletir. Fill Domain IPC'de yoktu: `fluid.seed fill_level=0..1` (API/IPC/Python/overlay). Kontrol: `rt_test_snow_ipc.py` → `seed_and_fill_are_snow` PASS (iki kolda granular > 0, fluid 0). Elle: Snow domain'de Seed Box ve Fill Domain → kar gibi durmalı.

Ek 3 derlendi ve elle doğrulandı. **Ek 4 (derlenmedi): kaynama.** Madde domain'inde `boiling_kelvin` üstündeki sıvı parçacık aşırı ısısını gizli ısıya harcar: `dm = min(m, m·c·(T−Tb)/L)` buhar olur, parçacık Tb'ye sabitlenir (hız sabiti yok; ısı kaynağı hızı belirler). Buhar aynı domain'in gaz fazına kg/J envanter + yoğunluk/sıcaklık izi olarak girer (izleyici = 1 atm ideal gaz hacmi / hücre, hücre boşluğuyla sınırlı); kayıt `vaporization`. Yalnız akışkan modelli, donmamış, yanıcı olmayan sıvılar (yanıcılar yanma yolundan). Kaynamaya hazır parçacık boşta gaz adımını uyandırır. Yeni madde alanı `vapor_molar_mass_kg_mol` (Water 0.018). Kaynama altı buharlaşma ve yoğuşma YOK. Sıra: (1) kontratlar; (2) `python scripts/test/rt_test_boiling_ipc.py` → boils, exchange_balanced, inventory_holds_steam, steam_visible PASS; (3) elle: Water dolu madde domain'i + termal alan → buhar fog olarak yükselmeli. ★ Sinsi: boils PASS ama steam_visible FAIL → envanter dolu ama izleyici GPU'ya yüklenmemiş (markHostWrote yolu). Bozuksa: gaz adımı boşta kapısı ya da gridsOverlap.

Ek 4 derlendi: kaynama çalışıyor (kullanıcı). Ama iki kusur görüldü, **Ek 5 (derlenmedi, SHADER dahil)**:
- **Termal alan şekli:** ısı her zaman küresel profil kullanıyordu (kutu alan suda daire kaynattı). Artık tek `ThermalSource` (96 B, hepsi float) alanın yerel dönüşüm matrisini ve şeklini taşır; Box = Chebyshev mesafesi, Cylinder = max(radyal, |y|), Cone = koni içi — kuvvet bölgesiyle aynı. `collectThermalSources` hem MSF hem madde parçacığı için tek liste kurar; `thermalSourceWeight` CPU'da, `sourceWeight` `sim_msf_ambient.comp`'ta birebir. ★ Ayrıca eski hata: `falloff_type`/`infinite` int olarak ham yükleniyor, shader float okuyordu → MSF'de falloff hep "None", Infinite hiç görülmüyordu. **`compile_shaders.bat` gerekli** (sim_msf_ambient).
- **Buhar zayıf yükseliyordu:** gaz sıcaklık izi `(Tb−ambient)/1500 ≈ 0.05` idi ve molar kütle yoktu. Artık buhar hücresi sanal sıcaklığa karışır: `Tv = Tb·M_hava/M_buhar` (su: ~600 K), çözücünün ölçeğinde (350 K/birim) ≈ 0.88.
Kontrol: (1) shader + build; (2) kutu termal alan → kaynama kutunun izdüşümünde (daire değil); MSF mesh ısınması artık falloff'a uyar (önceden tüm yarıçapta tam güç — **görünür davranış değişikliği**, sinsi değil); (3) buhar belirgin yükselmeli; (4) `rt_test_boiling_ipc.py` yine PASS. Bilinen: ortam sıcaklığı sıvıyı yalnız yüzeyden ve thermal liquid açıkken ısıtır; sıvı içi iletim/konveksiyon YOK.

**Not:** kullanıcının son build'i (exe 00:38) Ek 5'ten ESKİ — küre hâlâ görüldü çünkü şekil kodu derlemede yoktu. **Ek 6 (derlenmedi):** (a) Kaynama maddeye bağlı, suya özel değil: yanıcılar da kaynar (buharı gaz yakıtına), molar kütle bilinmiyorsa buhar hava kadar (nötr). Yeni/girilen değerler: `vapor_molar_mass_kg_mol` ve `thermal_expansion_per_k` (Water/Snow/Ice/Mud, Alcohol, Gasoline, Oil, Wax, metaller, Stone, Honey, Chocolate); Mud artık su gibi kaynar. (b) Doğal konveksiyon (Boussinesq): `dv = −g·β·(T−Tort)·dt`, akışkan modelli donmamış parçacıklara, thermal liquid açıkken ya da ısı alanı varken; host hızları yeniden yüklenir (donma yolu gibi, ayrı bayrakla — katı örtüsünü tetiklemez). IPC `thermal_convecting_particles`. Kontrol: shader + build; `rt_test_boiling_ipc.py` → `convects` dahil PASS; elle: thermal liquid AÇIK + domain ortamı 450 K → duvar/dipten ısınan su yükselir, zamanla tüm tank kaynar. Ortam yalnız thermal liquid açıkken etki eder (bilinçli).

Ek 6 derlendi: şekil doğru, kaynama sıcaklığa göre. Canlı ölçüm (80k su, adım başına 110–165 ms duvar saati, IPC dahil): konveksiyon 79996 parçacık; test `convects` FAIL'i testin hatasıydı (alan tüm suyu eşit ısıtıyordu → fark yok). Ortam 450 K'ye çekilince Fill tarifi yeniden seed ediyor ve su 450 K'de doğup tek adımda tüm tank kaynıyordu (1518 kg). **Ek 7 (derlenmedi):** (a) doğum kuralı: sıvı kaynama noktasının üstünde doğmaz (Tb'de doğar); panel metni genel. (b) çekirdekli kaynama: kaynayan parçacık boşluk oranıyla kaldırılır `dv = −g·Vv/(Vv+Vl)·dt` (fokurdama) ve boşluk > 0.2 ise whitewater Bubble doğar (adım başına ≤4096; görmek için whitewater açık olmalı). (c) test alanı alttan ısıtır. Kontrol: `rt_test_boiling_ipc.py` hepsi PASS; elle: thermal liquid AÇIK + domain override 450 K → tank tek adımda patlamaz, dipten/duvardan fokurdar, kabarcık yükselir. ★ Sinsi: kabarcık yok ama kaynama var → whitewater kapalı ya da foam taşıyıcısı host emit'ini görmüyor.

Ek 7 derlendi: tüm testler PASS, kaynama davranışı doğru (kullanıcı). RT'de buhar çıkınca su süt gibi görünüyordu. Kanıt (iki viewport görüntüsü): bulanık kenarlı beyaz levha = HACİM, dielektrik değil. Sebep: kaynama buhar izini parçacığın kendi hücresine, yani sıvının içine yazıyordu; gaz sisi sıvı hücrelerini doldurup yüzeyi örttü. **Ek 8 (derlenmedi):** buhar sütundaki en üst sıvı hücresinin bir üstüne bırakılır. Kontrol: RT'de kaynayan su dielektrik kalır, buhar yüzeyin üstünden yükselir; `rt_test_boiling_ipc.py` PASS. ★ AÇIK: kare 0'dan 3 adımda parçacık max 702 K okundu (alan 293+410). Kaynama her adım Tb'ye sabitlemeli; 702 K kalan parçacıklar kaynamanın bir kapısından düşüyor (gaz ızgarası dışı? mass_fraction 0? model≠Fluid?). Ayrıca 3 adımda ~32k parçacık kayboldu — sıfırlama sonrası doğum sıcaklığını ve ortam override'ını kontrol et.

Ek 8 derlendi, süt görüntüsü sürdü. Kare 0 görüntüsü: mavi su üstünde IZGARA HİZALI beyaz bloklar (gaz yok). **Ek 9 (derlenmedi):** (a) `registerOrUpdateLiveVolume` sıcaklık işaretçisi null gelince ESKİ sıcaklık ızgarasını tutup GPU'ya yüklüyordu ve `getGPUTemperatureGrid` `has_temperature`'a bakmıyordu → su yüzeyinin köpük kanalı önceki koşunun köpüğünü sıfırlamadan sonra da beyaz blok olarak çiziyordu. Artık eski ızgara silinir, getter bayrağa bakar (tüm hacimler, OptiX dahil). (b) Süt = Ek 7 kabarcıkları: her kaynayan parçacık her adım kabarcık doğuruyor, whitewater yüzey hacminin sıcaklık kanalını beyaz ortamla dolduruyordu. Artık seyrek: olasılık boşluk·0.05/adım, ≤512/adım, ömür 0.75 s. Kontrol: kare 0 temiz mavi su (blok yok); kaynarken su şeffaf kalır, seyrek kabarcık + yüzeyde köpük; beyazlık whitewater volume_density ile ayarlanır. ★ Sinsi: (a) değişikliği başka hacimlerin emisyonunu kapatırsa — sıcaklıklı ateş/duman VDB'si hâlâ yanmalı.

**Ek 10 (derlenmedi, shader YOK): sıvı ısı taşınımı, her madde kendi c/k/ρ/L/M/Tb'siyle.** (a) İletim: hücre başına ısı sığası (m·c) ve kapasite ağırlıklı k toplanır, yüz başına harmonik ortalama k ile kesin iki cisim çözümü (taşmaz, enerji korunur), parçacığa yalnız değişim yazılır. (b) Madde domain'inde hava = gaz ızgarası: yüzey parçacığı üstündeki gaz hücresiyle 20 W/m²K film üzerinden iki cisim dengesine gider (C_hava = ρ_hava(T)·V·1005); ortama sonsuz rezervuar soğuması madde domain'inde kapatıldı. Hava T = domain ortamı + (gaz.T − dünya ortamı)·K/birim. (c) Kaynama altı buharlaşma: Hertz–Knudsen + Clausius–Clapeyron (maddenin Tb, L, M), nem = atmosfer + hücredeki buhar; gizli ısı parçacıktan, buhar gaza, defter `vaporization`; etkin katsayı 4e-5 (20 °C kuru havada açık tava ~1e-4 kg/m²s). Thermal liquid açık madde domain'i gaz adımını uyandırır (maliyet: gaz çözümü her kare). IPC: `thermal_conducting_cells`, `thermal_gas_exchange_particles`, `thermal_heat_to_gas_j`, `thermal_evaporated_kg`. Kontrol: `python scripts/test/rt_test_heat_exchange_ipc.py` (6 karar) + `rt_test_boiling_ipc.py` yine PASS. ★ Bilinçli fizik: suyun havayla soğuması artık YAVAŞ (gerçekçi; eski 0.15/s abartılıydı); duvar teması hâlâ parametreli. ★ Sinsi: heat_to_gas > 0 ama gaz ısınmıyor → host gaz sıcaklığı GPU'ya yüklenmiyor (markHostWrote).

Ek 10 derlendi: boiling testi tam PASS; heat_exchange 5/5 PASS (iletim ~11k hücre, ~3k yüzey parçacığı, ısıtırken gaza ~+7 J/adım, buharlaşma ~1e-5 kg/adım, defter dengeli). İlk koşudaki 2 FAIL testin hatasıydı: kuvvet alanını kaldırmak `touchForceFields → invalidateRigidBodySimulationCache` ile madde domain'ini geri sarıp yeniden seed ediyor (310 K → taze 293 K). Soğuma kolu kaldırıldı; `end_frame` simde okunmuyor. ★ AÇIK karar: koşu sırasında alan düzenlemesi simülasyonu sıfırlamalı mı? Ayrıca ikinci koşuda adım 60'ta ortalama 305→299 K ve gaz kütlesi düştü (başka bir sıfırlama ya da kullanıcı etkileşimi — doğrulanmadı).

**Ek 11 (derlenmedi, shader YOK):** (a) Kuvvet alanı düzenlemesi artık koşuyu SIFIRLAMAZ: `invalidateSimCacheAfterCurrentFrame()` canlı durumu ve geçmiş kareleri tutar, yalnız şimdiki kareden sonraki önbelleği (sim/rigid/soft/particle/MSF) atar, disk bake'ini ayırır ve yeni ayarı benimser. API (`touchForceFields`) ve kuvvet alanı paneli bunu kullanır; rigid body/collider düzenlemeleri eskisi gibi tam sıfırlar. (b) Donma: kapı artık thermal liquid VEYA ısı/soğuk alanı; domain `granular_enabled` donmayı kapatmaz (kar görmüş madde domain'inde su hiç donmuyordu). Ayrıca serbest yüzey (üstü hava) destek sayılır: göl yüzeyden donar (canlı sahne: 13961 parçacık donma altında ama hepsi desteksiz, donan 0). Kontrol: `rt_test_heat_exchange_ipc.py` → yeni `field_edit_keeps_run`, `cools_after_source_off`, `freezes_from_surface` dahil hepsi PASS. ★ Sinsi: alan kaldırınca kare atlaması/titreme — kesilen önbellek karesi oynatılıyor olabilir.

**Ek 12 (derlenmedi, SHADER VAR: `sim_msf_ambient.comp` — `compile_shaders.bat` gerekli):**
(a) Sıvı ↔ collider temas ısısı: MSF kabuğu (alan·1 mm·ρ·c) ↔ temas parçacığı, film 500 W/m²K,
enerji korunur; sıvı tarafı bir readback geriden okur, kabuğa giden ısı yeni binding 4 ile
cihaza EKLENİR (kernel tablosu 4→5 tampon). Kabuklu collider'a temas artık ortama sonsuz
soğuma değil. (b) Ayrı Fluid + Gas domain'i üst üste binince sıvı havası gaz ızgarası
(çift soğuma yok) ve thermal liquid açık sıvı gaz domain'ini uyandırır. (c) Yanan sıvı:
görsel ısı eki söküldü, buharlaşan yakıtın gizli ısısı parçacıktan. IPC:
`thermal_solid_contact_particles`, `thermal_heat_to_solid_j`.
Sıra: (1) shader + build; (2) `rt_test_heat_exchange_ipc.py` (8 karar) ve `rt_test_boiling_ipc.py`;
(3) `python scripts/test/rt_test_contact_heat_ipc.py` → demir küp alttan ısıtılır, alan suya
değmez, su küpten ısınır (4 karar). ★ Sinsi: `heat_flows_solid_to_liquid` PASS ama
`water_warms_through_cube` FAIL → MSF host durumu güncellenmiyor (readback) ya da kabuk
hücresi parçacığın komşuluğuna düşmüyor. ★ Sinsi 2: MSF elemanları su içinde değilken
binding 4 kayarsa tüm MSF sıcaklıkları sapar — ateş/odun yanma sahnesi hâlâ normal yanmalı.
Kontratlar 21/21, audit OK.

**Ek 13 (derlenmedi, yalnız C++; shader yok):** `msf.fields` satırına eleman sıcaklığı
eklendi: `temperature_valid`, `mean_kelvin`, `min_kelvin`, `max_kelvin` (host readback'ten;
readback yoksa valid=false, 293 K diye uydurmaz). Kontakt testi fiziksel senaryoya çevrildi:
"kızgın demir suya" — kuru küp alanla ısıtılır, alan kalkar, tank doldurulur. Eski "ocak
alttan" kurgusu MSF'de katı içi iletim olmadığından (ve demirde 0.3 m için ~4000 s
gerektiğinden) geçemezdi. ★ Test collider'da `gas_interaction_enabled` + `gas_ignite_on_contact`
açıyor: MSF kabuğu bugün yalnız bu iki kapının arkasında var (gaz oranları 0, enjeksiyon yok).
Sıra: (1) build; (2) `rt_test_contact_heat_ipc.py` → 5 karar; `shell_hot_before_fill`
max_kelvin > 350 olmalı. ★ Sinsi: 5/5 PASS ama su yalnız ~0.0x K ısınıyorsa → kabuk ısısı
MSF'nin sabit `cooling_rate` gevşemesiyle havaya kaçıyor (bkz. açık iş: kütleden türeyen
kapasite).

**Ek 14 (derlenmedi, yalnız C++):** Temas ısısında kabuk kalınlığı artık cisimden: kapalı mesh'te
hacim/alan (iki orijinli diverjans hacmi eşitse kapalı), üst sınır Biot=1 derinliği k/500;
açık mesh 1 mm. Demir küp 0.6 m → 0.1 m (eskiden 1 mm; ölçüm: 1200 kg suyu +0.02 K ısıttı,
kabuk 347/394 K iken — tutarlı ama içi boş sac). Test kararları: `shell_is_a_body`
(adım 200'de heat_to_solid_j < −100 J; beklenen ~−400) ve `water_warms_through_cube`
(max > 293.1 K; 5 cm parsel duvar dibinde ~+0.5 K). ★ Sinsi: su ısınıyor ama odun/taş
collider'lı yanma sahnesinde değişiklik olmamalı — kalınlık yalnız sıvı temasında kullanılır.
Bilinen açık: MSF'nin havaya gevşemesi (`cooling_rate`) kapasiteden bağımsız; kalın demir
blok da sac kadar hızlı havaya soğur.
