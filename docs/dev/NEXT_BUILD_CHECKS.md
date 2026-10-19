# 2026-10-19 (3. build): P5 söküm + P6 bayat testler

> **SONUÇ (canlı):** madde 1/1, FG1 8/8, `g2_dry_wet` (Soil) dört kol PASS. `sleep_transfer` FAIL, beklendiği gibi: "bed collapsed" (COM y 0.028 m, 3 kat ~0.075). Not: `g2_dry_wet` domain'ini ve iki kaynağını sahnede bırakır; ardından koşan "boş sahne" testleri onlar silinmeden başlamaz.

**Derleme:** yalnız C++ (shader değişmedi). Emekli madde alanları:
`fluid_extinguishing`, `cooling_power`, `oxygen_dilution`; domain aynaları
`fluid_extinguishing`, `fluid_cooling_power`, `fluid_oxygen_dilution`.

Sıra (hepsi kısa):

1. `python scripts/test/rt_matter_smoke.py --only substance fire_steam` —
   madde 1/1 (Water'da üç alan yok, `fluid_extinguishing` set reddedilir),
   FG1 8/8. **Bozuksa:** FG1 düşerse söndürme hâlâ bu alanlardan birini
   okuyordu (grep temizdi; olmaması gerekir).
2. `python scripts/test/rt_g2_dry_wet_compare_ipc.py` — Soil (MPM) ile dört kol,
   kütle sürüklenmesi kapıları. **Bozuksa:** gözenek değişimi MPM'de de engel.
3. `python scripts/test/rt_test_grain_sleep_transfer_ipc.py` — ★ beklenen sonuç
   büyük olasılıkla **'bed collapsed; projectile birth site is occupied'**. Bu
   artık doğru teşhis; 3×3×3 basit kübik yığın sürtünmeyle bile kararsız olabilir.
   Ya yatak kararlı dizilime (piramit / duvarlı kutu) geçer, ya da çöküş bir
   regresyondur (revizyon 24'te geçiyordu). Karar bir canlı ölçüm ister.

Sessizce makul görünen: eski bir proje dosyası açılır, su domain'i hatasız
yüklenir. Bu doğru: alanlar okunmuyor, davranış aynı (su zaten yanmıyordu).

# 2026-10-19 (2. build): P0–P4 — TEK DERLEME, TEK DUMAN KOŞUSU

> **SONUÇ (2026-10-19, canlı):** `rt_matter_smoke.py` **10/10 PASS** — madde 1/1, filmler 5/5 (düşüş 0.069 K/s, ortama 67.7 kJ), phase3 donma PASS, gaz↔tane 6/6, Ek 11 7/7 (iki koşu), yüzey suyu 6/6, FG1 8/8, tane ısısı 8/8 (artık ~1e-10 J, 400 tanede 0.5 ms), readiness 1/1. Not: tane sahnesinde `liquid_max_k` tane sıcaklığını gösteriyor (sıvı istatistiği taneleri de sayıyor olabilir; raporlama, fizik değil).

**Derleme:** C++ **ve shader paketi**. Değişen shader'lar: `sim_msf_ambient`
(film + alan hızı), `sim_msf_gather` (7 bağlama, 112 bayt: buhar). Yeni dosya
yok (`MatterGrainThermal` 1. build'deydi). Emekli alanlar: `cooling_rate`,
`solver_air/contact_cooling_rate` (madde), `thermal_air/contact_cooling_rate`
(domain, IPC, Python, panel). Kural: [MATTER_ROADMAP_2026_10.md](MATTER_ROADMAP_2026_10.md).

**Tek komut:** `python scripts/test/rt_matter_smoke.py`. Test başına tek satır
basar; yalnız FAIL olanın logu açılır (`%TEMP%/rt_matter_smoke/`). Test
sırasında ayıklama yapılmaz, bulgu nota girer.

| Kol | Bekleme | Bozuksa |
|---|---|---|
| 0 açılış | log'da `sim_msf_*` pipeline hatası yok. Eski projede `retired field 'cooling_rate' dropped (...)`; proje açılır. | Pipeline hatası = `.spv` eski. |
| substance | PASS | `thermal_response` ve `cooling_rate` reddedilmezse exe eskidir. |
| thermal_films | 5/5. Su duvar filmiyle YAVAŞ soğur: düşüş < 1 K/s, `thermal_heat_to_ambient_j` > 0. Alan `thermal_rate_per_s` (0,5) okunur. | Düşüş ~25 K/s ise eski hız yolu duruyor. ★ Sinsi olan: 0 K/s ve `to_ambient` 0 ise duvar yüzleri sayılmıyor (`s_count` / kapalı sınır). |
| phase3_freeze | PASS (emekli hızlar testten çıktı) | Donma defteri bozulduysa freeze yolunda bir şey kaydı. |
| gas_drag | 6/6 | – |
| contact_heat (+ tekrarı) | 7/7, ikisi de. Kabuk dolumdan önce ısınmış ve `temperature_fresh`. Alan artık 2/s ile ısıtıyor (eskiden demirin `cooling_rate`'i), yani kabuk ~690 K'e çıkabilir; su→katı ısısı 1. build'den büyük olur. | İkinci koşu FAIL ise `add_primitive` düzeltmesi yok. |
| surface_water | 6/6. Meşe üstü daha hızlı ısınır (alan 2/s), `evaporated_kg` artar. | `water_mass_closes` asıl kapı. |
| fire_steam | 8/8 | ★ Sinsi olan: `steam_reduces_burned_fuel` yine PASS ama ıslak koldaki MSF yok; P2 burada görünmez. |
| grain_heat | 8/8: `cools_at_physical_rate`, `water_takes_most`. Ortalama koşu sonunda ~345–355 K. `ms` 400 tanede < 1 ms. | Hızlı soğuyup `to_walls` küçükse başka bir kapasitesiz yol var. |
| grain_readiness | PASS: `backend=cpu`/`boundary=open` reddedilir. | – |

Ölçülmeyenler (bilinçli, sonraki tura): P2'nin buharlı hücredeki etkisini
gösteren ayrı bir kapı yok. P3 (tane domain'inde donma) ve P4'ün 1e5 tane
maliyeti için test yazılmadı. İlk ihtiyaçta tek kısa kol eklenir.

---

# 2026-10-19: yüzey↔hava filmi (Ek 11 kökü) + taneler termal zincirde + gaz↔tane sahnesi — TOPLU TEST

**Derleme:** C++ **ve shader paketi** (`compile_shaders.bat`). Değişen shader'lar:
`sim_msf_gather` (6 bağlama, 96 bayt push constant), `sim_msf_resolve` (9 bağlama).
Yeni dosya: `Fluid/MatterGrainThermal.cpp/.h` (vcxproj + filters'ta). Emekli madde
alanı: `thermal_response`. Sıra: bağımsız ve hızlı olan önce; diğerlerini
maskeleyen sonra.

0. **Açılış:** log'da `sim_msf_gather` / `sim_msf_resolve` pipeline hatası YOK.
   Varsa `.spv` eski: kernel tablosu 6/9 bağlama ve 96 bayt bekliyor, eski `.spv`
   4/8 bağlamalı. Belirti: MSF adımı her karede başarısız, yüzeyler hiç ısınmıyor.
   Eski bir projede `thermal_response` override'ı varsa log'da `retired field
   'thermal_response' dropped (...)`. Proje AÇILMALI.
1. **Madde alanı (hızlı):** `python scripts/test/rt_test_substance_profiles_ipc.py`
   PASS olmalı. Yeni satır: `substance.set fields={thermal_response:3}` reddedilir.
   Kabul edilirse exe eskidir.
2. **Gaz↔tane (hızlı, bağımsız):** `python scripts/test/rt_test_grain_gas_drag_ipc.py`,
   **6/6** bekleniyor. Rüzgâr kaynağı artık düşüş yolunun üstünde (0, 1,1, 0;
   r 0,35); x kapısı düşüş penceresinde ölçülüyor (ilk 45 adım).
   `reported.fall_drag_n_s[0]` > 0 ve `fall_reaction_n_s[0]` bunun eksisi olmalı.
   x yine ≤ 0 ise: kaynak hızı hacimde dayatmıyor (yalnız yoğunluk basıyor) ya
   da gaz şeridi yalnız kaynakta hareket ediyor. `max_relative_speed_m_s` < 1
   ise taneler rüzgârı hiç görmüyor demektir.
3. **Ek 11, temas ısısı:** `python scripts/test/rt_test_contact_heat_ipc.py`,
   **7/7** bekleniyor. Yeni kapı `wetted_shell_is_hot` (ilk örnekte
   `shell_seen_kelvin` > 330).
   - Beklenen: `shell_before_fill.mean_kelvin` ≈ `max_kelvin` (~390 K). Eskiden
     ortalama 348 / en yüksek 396 idi: domain içindeki yüzler havaya çivileniyor,
     yalnız zeminin altındaki yarı ısınıyordu.
   - İlk örnekte `heat_to_solid_j` ≈ −400…−700 J/adım ve `shell_seen_kelvin`
     ≈ 380–395 K olmalı.
   - `shell_seen` hâlâ ~294 K ise gather `.spv`'si eski (0. madde).
   - Kabuk sıcak ama ısı küçükse sorun film/alan tarafında (`exchangeLiquidSolidContact`).
   - ★ Sinsi olan: 7/7 ama `water_warms_through_cube` zar zor geçiyorsa (293,1
     sınırında), bu doğrudur, hata değildir. 5 cm'lik bir parsel duvar dibinde
     adım başına ~0,002 K ısınır. Onlarca kelvin görürsen film yanlış.
4. **Yüzey suyu, regresyon:** `python scripts/test/rt_test_surface_water_ipc.py`,
   **6/6** beklenir ama SAYILAR DEĞİŞİR. Meşe küpün üstü eskiden gaz çivilemesiyle
   636 K'e çıkıyordu. Artık ~0,3 mm'lik meşe kabuğu sıcak havadan 20 W/(m²K)
   filmle ve alanın gevşemesiyle ısınıyor: `max_kelvin` belirgin düşmeli,
   `evaporated_kg` küçülmeli. Kapı yalnız > 0 istiyor.
   ★ Sinsi olan: `hot_surface_dries_into_domain` FAIL ise üst yüz kaynamaya hiç
   yaklaşmıyordur. Bu fizik, kusur değil; alanı (delta 300 K) büyütmeden önce
   gerçek gözlemi not et.
5. **FG1, regresyon:** `python scripts/test/rt_test_fire_steam_quench_ipc.py` →
   8/8 (MSF'siz gaz; yüzey değişikliği etkilememeli).
6. **Taneler termal zincirde (yeni):** `python scripts/test/rt_test_grain_heat_ipc.py`,
   **7/7** bekleniyor. Kapalı Matter domain, 293 K su havuzu, 360 K Gravel
   (r 1 cm) havadan suya düşüyor.
   - `thermal_not_a_blocker` FAIL ise exe eskidir (`grain_readiness` hâlâ
     `thermal_liquid` diyor).
   - `grains_measured` FAIL ama taneler doğmuşsa: tane taşıyıcısının modeli
     Granular değil ya da sahibi Grain değil (`roleOf`).
   - `water_takes_heat` FAIL, `wetted` = 0: tanelerin hücresi sıvı havuzuyla
     eşleşmiyor (ızgara/hücre eşlemesi).
   - Yalnız `air_takes_heat` FAIL: gaz şeridi adım atmıyor (`gas_stats.stepped`
     yanlış). Bu, gaz↔tane sürüklemesinin de aynı sahnede kapalı olduğu
     anlamına gelir.
   - `energy_closes` FAIL ise geri yazmada bir taraf eksik.
   - Ölçek: ortalama koşu boyunca ~10 K düşmeli (τ ≈ 36 sn). ★ Sinsi olan:
     birkaç saniyede 293 K'e inmesi. Hiçbir kapı onu yakalamaz, ama film ya
     da alan (süper tane kütlesi) yanlış demektir. `grain_mean_k` eğrisine bak.
   - `ms` (`thermal_grain_ms`): seri döngü. 400 tanede < 1 ms beklenir. 1e5
     tanede ölç ve not et (OpenMP'ye bölünmedi).
7. **H1 regresyonu (uzun, en son):** `python scripts/test/rt_h1_grain_suite.py`.
   Termal zinciri kapalı tane sahneleri bit bit aynı kalmalı. Tek yeni yol,
   `thermal_liquid_enabled=True` olan tane domain'i.
8. **Panel (gözle, 6'dan sonra):** Thermal Liquid bölümünde, taneli domain'de
   onay kutusu artık kilitli değil. "Grains: … min/mean/max K" ve "Grain heat …
   residual" satırları görünüyor; değerler IPC'deki `thermal_grain_*` ile aynı.

---

# 2026-10-15: ORTAK atmosfer sıcaklığı — SON kaynak teslimi

**Yeni derleme / RT Play canlı ölçümü:** Kullanıcı derleyip açtı. Dört dış IPC
ölçümünde timeline 2740→2752, cloud time 114.083→114.583 s, render backend
state_updates 2440→2452 ilerledi. RT samples her ölçümde 1 ve complete=false:
değişen atmosfer için accumulation yeniden başlıyor. Güncelleme akışı çalışıyor;
bu sayaçlar tek başına sunulan görüntünün her kare değiştiğini kanıtlamaz.
Kullanıcı benzer seed/desenin donma algısı oluşturmuş olabileceğini bildirdi.

**RT playback ek düzeltmesi:** Atmosfer time/revision push'u RT görünümünde artık
aynı döngünün render bloğundan önce `start_render` ister. Önceki uyanma, render
bloğundan sonraki ayrı `animation_playback_frame` takibine bağlıydı; clouds-only
fast path de yalnız o değişken değişince çalışıyordu. Perdenin sabit quadrature'ı
zamanı/rüzgârı sabitlemez. Yeni wiring C++ derlemesine dahildir, bu ek için shader
değişmedi. Açık eski binary salt okunur ölçümde RT, frame=2328/time=97 s,
128 samples/converged ve sabit cloud-update sayaçları verdi; Play sorunu canlı
yeniden üretilemedi. Yeni derlemede Studio öndeyken RT Play'de playhead/time,
gerçek görüntü hareketi ve Stop sonrası yakınsama birlikte kontrol edilmeli;
birkaç kare görüntü almadan hareket var diye kabul edilmemeli.

1. **C++ + bütün shader paketi**, Studio'yu yeniden aç. Ortak sıcaklık include'ı,
   Weather Surface packet'i ve lens CPU forcing değişti; yalnız lens shader'ı yetmez.
2. Atmosphere/Climate sıcaklığını değiştir: kamera camında ayrı sıcaklık girişi
   artık yok; **Atmosphere at camera** yerel değeri gösterir. Kar aynı sıcaklıkla
   erir/donar; kamera yüksekliği ve atmosphere altitude offset'i hesaba katılır.
3. Weather Surface kar kaplaması: soğukta tutunma, 0..2 °C'de geçiş, sıcakta
   ıslak görünüm. RT/RayFusion ve farklı yüksekliklerde aynı davranışı gözle.
   Bu termal görünüm kapısıdır; kalıcı yüzey kütle simülasyonu değildir.
4. Dış `python scripts/test/rt_test_lens_weather_ipc.py --backend viewport`:
   emekli cam-sıcaklık alanının atomik reddi, ortak yerel sıcaklık ve termal
   uygunluk sorgusu. Kayıtlı kar test sahnesinde global timeline duraklatılmışken
   ek `--snow`: atmosfer sıcaklığını geçici değiştirir, birikim/erime/donmayı
   test eder ve eski sıcaklığı geri yükler; otomatik bulutlar da sıcaklığı izler.
5. Eski `glass_temperature_c` içeren proje yükle/kaydet: alan yüklemede atılır,
   diğer lens ayarları kalır. Ayrıntı [Lens §12](ATMOSPHERE_LENS_WATER.md#12-tek-atmosfer-sıcaklığı--kamera-ve-yüzeyler).

Statik API/descriptor, profil eşliği ve include/syntax kontrolleri geçti;
derleme ve canlı/görsel kabul bekliyor. Perde gürültüsü ve hafif eliptiklik
kontrollerini de aynı derlemede sürdür.

---

# 2026-10-15: Yağış perdesinin içindeki kamera — gürültü düzeltmesi

Kullanıcı duraklatılmış sahnede gürültüyü yağış perdesine ayırdı. `cloud_rt.glsl`:
kamera perdesinde rastgele örnek/gölge jitter'ı yerine sabit orta nokta; aralık
sınırları artık tam t0..tEnd'i kapsıyor (eski kaydırma ilk len*(jitter/n)^2 metreyi
atlıyordu). Yakın ince perde deseni 240..1200 m arasında yumuşak geçişle açılır;
ortalama yoğunluk bileşeni ve yakın partikül alanı korunur. İkincil ışınlar aralık
içi rastgele örneklemeye devam eder. Örnek bütçesi artırılmadı.

**Yalnız shader derlemesi**, ortak include nedeniyle shader paketini birlikte
yenile ve Studio'yu yeniden aç. Etkilenen girişler: `cloud_callable.rcall`,
`cloud_raster.comp`, `cloud_scene.comp`, `cloud_shadow.comp`, `raygen.rgen`,
`miss.rmiss`, `closesthit.rchit`, `hair_closesthit.rchit`, `volume_closesthit.rchit`.
Codex derlemedi/başlatmadı. Sayısal tam aralık ve homojen Beer-Lambert kontrolü
geçti; gerçek sahne görsel kabulü bekliyor.

**Göz:** Aynı duraklatılmış sahnede perde içinde/dışında RT ve RayFusion; yakın
partikülleri kapat/aç ile perdeyi ayrı gözle. Sabit benekler ve kareden kareye
oynama azalmalı; ufuk perdesinde yeni bantlaşma, yoğunluk/dikiş/parlaklık değişimi
olmamalı. Oynatırken yağış hareketini, şimşek ve güneş yönünü ayrıca kontrol et.

---

# 2026-10-15: L2 kamera camında kar/buz — kaynak teslimi

1. **C++ + dört shader:** `lens_water_sim.comp`, `lens_water_draw.comp`,
   `tonemap.comp`, `raster_post.comp`; Studio'yu yeniden aç. Katı optik alanı için
   descriptor binding 2 eklendi, eski shader'larla karıştırma.
2. Kamera seç → **Camera glass weather (Lens weather)** → **Precipitation on
   camera glass**; Glass temperature 0 veya -5 °C. Kamerada atmosferik kar olmalı.
3. Timeline oynarken biriktir, duraklat: kar lekesi görünür kalmalı. Camı 20 °C
   yapıp oynat: kar azalıp suya dönüşmeli. Negatif sıcaklıkta donan yağmur/buz pusu.
4. Kayıtlı test sahnesi, global timeline duraklatılmış:
   `python scripts/test/rt_test_lens_weather_ipc.py --backend viewport --snow`.
   Yağmur sahnesinde `--wet` regresyonu; Rendered için `--backend render`.
5. Probe target/frame/ayarları geçici değiştirir; lens geçmişini temizler, seek'in
   diğer simülasyon yan etkilerini geri almaz. Gerçek kar optik alanı, pause/replay,
   erime, yeniden donma/buz optik alanı ve %2 su kütlesi bütçesi denetlenir.
   RT/RayFusion göz kabulü ayrıca gerekir.

Sözleşme: [Lens L2 §11](ATMOSPHERE_LENS_WATER.md#11-l2-karbuz-kaynak-teslimi--gerçek-sözleşme).
Kaynak teslim edildi; L2 derleme/canlı kabul bekliyor. Hafif eliptiklik açık not.

---

# 2026-10-15: L1 kamera camında sıvı yağmur — önceki teslim

**En-boy oranı düzeltmesi (son değişiklik):** Kare UV damla hesabı geniş görüntüde
elipse uzuyordu. Gerçek çıktı width/height oranı lens draw push'una iletiliyor;
alanı koruyan metrik damla şekline/kırılmaya, ters metrik composite blur tap'lerine
uygulanıyor. **C++ + lens_water_sim.comp, lens_water_draw.comp, tonemap.comp,
raster_post.comp** derle ve yeniden aç (ortak push include açıklaması da güncellendi).
Sayısal oran kontrolü 16:9, 9:16, 1:1 ve ultrawide için geçti; kullanıcı görsel
kabulü bekleniyor. Camı ıslatıp duraklat, farklı viewport oranlarında damla
genişliği/yüksekliğini ve blur'u RT/RayFusion'da karşılaştır. Önceki C++ DoF
kapısı/gerçek optik alan ölçümü düzeltmesi de bu teslimin içinde.

**Sonraki C++ düzeltmesi:** Önceki shader düzeltmesinden sonra kullanıcı 30
damlada hâlâ etki görmedi; IPC 30 damla ve %4,58 simülasyon kaplamasını doğruladı.
`LensWeatherCamera.cpp` artık DoF/pinhole için `effectiveLensRadius()` kapısını
kullanır; DoF kapalı kamerada fallback f/2 ile yapay geniş CoC oluşturulmaz.
`camera.sample_lens_weather` gerçek RGBA16F optik alanını yalnız açık ölçüm
isteğinde okur: `field_max_coverage`, `field_mean_coverage`, `field_max_shift_uv`,
`field_wet_pixels`, `optical_blur_uv`. Islak probe bu alanın sıfır olmadığını da
ister. **Yalnız C++ derlemesi + yeniden açma** gerekir; yeni shader değişikliği yok.
Sayısal/görsel kabul bekleniyor; damla sayısı tek başına görünürlük kanıtı değildir.

**Görünürlük düzeltmesi (sonraki derleme):** Kullanıcı 10 mm/saat ve 160 damlada
camı göremedi. `lens_water_draw.comp` içinde defocus coverage'ın kırılma kaymasına
ve son composite'e iki kez uygulanması giderildi: kayma artık toplam ağırlığa
normalize edilir, coverage son karışımda bir kez uygulanır. Kamera özelliklerinde
başlık `Camera glass rain (Lens weather)` oldu ve varsayılan açık. Kullanıcı
**C++ + lens_water_draw.comp** derlemeli ve Studio'yu yeniden açmalı. Islak camda
kontrastlı arazi/siluet üzerinde kırılmayı, pause'da görünümün kalmasını ve Clear
sonrası kaybolmasını RT/RayFusion'da kontrol et. Görsel sonuç henüz doğrulanmadı.

**Canlı kontrol:** Kullanıcı derledi ve Studio'yu açtı. Dış IPC viewport probu
geçti: ayar round-trip, atomik geçersiz girdi reddi, GPU pipeline/kaynakları ve
kapasite sınırları. Kamerada ölçülen yağış 0 mm/saat (`dominant_form: none`);
`--wet` ve RT/RayFusion görsel kabulü henüz yapılmadı.

**Devam canlı teşhis:** Kullanıcının yağmur sahnesinde kamera lens ayarı kapalıydı;
IPC ile etkinleştirildi. Görünüm/timeline çalışırken viewport GPU durumu başlatıldı,
kamerayla eşleşti ve timeline adımını yakaladı. 4–5 damla, %0,63–0,79 kaplama ve
pozitif su hacmi ölçüldü; iki ölçüm arasında state hash değişti. Son kamera yağışı
0,0095 mm/saat, yarıçap 1 mm: bu ölçüm güçlü akış/görsel kabulü sağlamaz.

1. **Studio C++** ve dört shader: `lens_water_sim.comp`, `lens_water_draw.comp`,
   `tonemap.comp`, `raster_post.comp`. Ortak `lens_water_*.glsl` include'ları bunlara
   dahildir. Codex derlemedi; yeni binary/shader'larla Studio yeniden açılmalı.
2. Kamera özellikleri → **Lens weather → Rain on camera glass**. Varsayılan kapalı.
   Atmosferik yağmur/çisenti altında timeline'ı oynat; başlangıçta cam kurudur.
3. Dış terminal: `python scripts/test/rt_test_lens_weather_ipc.py --backend viewport`.
   Kayıtlı yağış test sahnesinde, kamera ıslak noktadayken ek `--wet`; bu probe
   kamera target'ını/timeline'ı geçici değiştirir, lens geçmişini temizler. Mevcut
   fizik/animasyon seek yan etkilerini geri almaz. Rendered açıkken backend `render`.
4. RT/RayFusion göz: gerçek arka plan kırılması, yapışma/kayma/iz/birleşme;
   yukarı/rüzgâra bakış, coating/hood, Clear/off/pause. **Drops around the camera**
   kapalıyken lens yine atmosfer yağışını almalı. RT accumulation'a ek reset gelmemeli.
5. Son 10 s içinde geri sarma; daha uzak seek kuru restart ve history_misses verir.
   DoF + damla, resize, scene değişimi, proje save/reopen, final çıktı ve GPU süre/bellek.

İlk L1: 256 sıvı damla, 1/60 s, son 601 adım history; kar/buz L2'de.
Tam checkpoint/ısınma, barınak, film çözücüsü ve sınırsız offline seek teslim edilmedi.
Sözleşme/sınırlar: [ATMOSPHERE_LENS_WATER.md §10](ATMOSPHERE_LENS_WATER.md#10-l1-ilk-kaynak-teslimi--gerçek-sözleşme-2026-10-15).

---

# 2026-10-15: audio arka plan üretim/analiz (Studio C++)

Audio için yeni shader yok. AudioDesignJobs.cpp/h VS projesine kayıtlı.

1. Studio C++ derlemesi; mevcut boş sahneyi yeni binary ile aç.
2. Dış terminal: `python scripts/test/rt_test_audio_jobs_ipc.py`.
3. Designer: Play/Refresh ve kontrol bırakma sırasında viewport gezin; art arda
   düzenle/Stop/Stop All yap, eski işin yeniden ses başlatmadığını kontrol et.
4. Analyze FFT/kanal değişimi ve Save asset; scene değişimi ve kapanışta worker ömrü.
5. Kurgu M4: çoklu seçim undo/redo/iptal; proje save/reopen sonrası track/clip/curve
   kimlikleri ve routing. Sözleşme: [Audio jobs](AUDIO_BACKGROUND_JOBS.md).

Mevcut binary'de track/export ve dar M2/M3 IPC kontrolleri geçti; yeni worker kodu
henüz derlenmedi/canlı test edilmedi. İşitsel kalite ve ölçülmüş performans açık.

---

# 2026-10-15 (5): ışıksız sis araziyi gizlemiyordu (yalnız shader)

Kullanıcı: noktacıklar azaldı. Güneş ışığı yokken sis zayıf, arazi sisin
içinden belirgin. Fiziksel beklenti: kayıpsız saçan sis ortam ışığında
~gök parlaklığında, araziden (~albedo × gök) PARLAK gri perde. Kök: sis
örneğinin ortam örtme tap'leri sisin KENDİ yoğunluğunu da sayıyordu
(bulut kuralı) → güneşsiz sis ~0.4 × gök, arazi kadar koyu. Düzeltme
(`cloud_rt.glsl`): sis örneğinde yalnız üst/alt BULUTLAR örter.

1. **Derleme:** yalnız shader.
2. **Göz:** güneşi kapat (ya da fırtına gölgesinde bak): sis açık gri bir
   perde olmalı, arkadaki arazi mesafeyle kaybolmalı; güneşli sis bir
   miktar parlaklaşabilir (beklenen).
   - ★ Sinsi olan: sis şimdi gökten parlak / beyaz parlıyor → `below`
     (zemin) payı + albedo fazla; zemin ışığına bak.
3. Bilinen: RayFusion'da sis İÇİNDEKİ arazinin kendi ortam ışığı (gök IBL)
   sisi bilmiyor; kayıpsız sis için fark küçük, açık iş olarak not edildi.

---

# 2026-10-15 (4): güneşli sisteki parlak noktacıklar (yalnız shader)

Kullanıcı: sis maliyeti gitti, elle sis kuruluyor. Şimşek ışığında sis
kaliteli; GÜNEŞ ışığında aşırı gürültülü (RayFusion karesi: solda sarımsı
noktacıklı alan, ortada/sağda şimşekle temiz sis). Görüntü: sunlit sis
yüzeyinde (su ve sırt üstü) tek tek parlak pikseller.
Kök: sis örneklerinin güneş ışığı yürüyüşü rastgele tap konumlarıyla;
alçak güneşte yol sisin içinden kilometrelerce (τ ~ 7) → exp(-τ) tap
konumuna aşırı duyarlı. 150 m ışık yeniden kullanımı bu tek tahmini
yayıyordu; RayFusion'ın opak piksel pass'i (`cloud_scene.comp`) zamansal
birikim yapmadığı için noktacıklar sabit kalıyor. Düzeltme: sis
örneklerinde ışık tap'leri sabit (orta nokta) ve detaysız.

1. **Derleme:** yalnız shader (`cloud_rt.glsl`).
2. **Göz, aynı kare (RayFusion):** soldaki güneşli sis noktacıksız, düz ve
   yumuşak olmalı; şimşekli kısım aynı kalmalı.
   - Hâlâ noktacık varsa: kaynak ışık değil march başlangıç jitter'ı
     (piksel başına sabit tohum) ya da sis DIŞINDAKİ bir katman/yağmur
     damlası. Sisi kapatıp bak: noktacık gidiyorsa sis.
   - ★ Sinsi olan: noktacık gider ama sisin içinde güneş yönünde **sabit
     çizgisel bantlar** belirir → deterministik tap'lerin (40/80/160… m)
     kuadratür bandı; sis kenarı keskinse olur, tap'e küçük piksel-sabit
     kaydırma gerekir.
3. RT'de de aynı sahne: birikimle zaten temizleniyordu; daha hızlı
   yakınsamalı.

---

# 2026-10-15 (3): sis adımı + elle vadi sisi (C++ + shader)

1. **Derleme:** C++ (`AtmosphereClouds`, `AtmosphereLocalWeather`,
   `CloudMaterialUI`, `RtApiClouds`) + bulut shader'ları (`cloud_rt.glsl`).
2. **Maliyet (sayı), aynı durdurulmuş kare:** sis aç/kapa farkı ~200 ms'den
   belirgin düşmeli (beklenti ≤ ~50 ms). Asıl ölçü `shadow_atlas`.
   - Hâlâ yüksekse: `realtime_steps` 256 → bütçe 1024; sis dışı bulut
     katmanları da pahalı olabilir — sisi kapatıp tabanı ayrıca oku.
3. **Görünüm:** sis kenarı (25 m'lik yumuşak kapak) 40 m adımda
   **bantlanabilir/titreyebilir**. ★ Sinsi olan: durmuş karede yatay
   katmanlı çizgiler → adım jitter'ı yetmiyor; sis min adımı 30 m'ye indir.
4. **Elle vadi sisi:** yağışı kapat (rate 0), Clouds > Local weather / fog >
   "Author the fog" aç, Valley fog 0.8 → vadiler sisle dolmalı, tepeler
   açık kalmalı. Lift'i 0 → 1 sürükle: kapak yükselmeli ve incelmeli.
   `world.local_weather_stats` → `fog_map.source = "authored"`,
   `fogged_fraction > 0`.
   - Bozuksa sis yok → harita yeniden kurulmuyor (anahtar `cloudRevision`);
     ya da `fog_strength` 0.
   - ★ Sinsi olan: elle kapalıyken de sis değişiyor → `fogTexel` elle
     değerleri koşulsuz okuyor.
5. Keyframe: iki kare arasında Valley fog 0.8 → 0 ve lift 0 → 1 → sabah
   dağılan sis animasyonu.

---

# 2026-10-15 (2): yerel sis maliyeti — ışık mesafeyle yeniden kullanılıyor (yalnız shader)

Ölçüm: sis ~80–100 ms/kare (RayFusion), çoğu gök pass'inde. Kök: her sis
örneği tam bulut ışık yürüyüşü yapıyordu. Not: `ATMOSPHERE_LOCAL_WINDS_FOG.md`
devir notu.

1. **Derleme:** yalnız shader (`cloud_rt.glsl` → bulut callable, cloud_raster,
   cloud_scene).
2. **Maliyet (sayı):** aynı sahne, `local_weather.fog_enabled` aç/kapa,
   `viewport.frame_timings`: fark ~80–100 ms'den belirgin düşmeli
   (beklenti: ~20 ms'nin altı). `shadow_atlas` aşaması asıl ölçü.
3. **Görünüm:** sis eskisiyle aynı görünmeli. ★ Sinsi olan: sisin içinde
   bulut gölgesi kenarlarında ~150 m'lik **basamaklar/yamalar** → yeniden
   kullanım mesafesi büyük; `max(150, 0.05 t)` küçültülür.
4. Kamerayı sisin İÇİNE sok: yakın sis düz görünmeli, kare kare titremesin.

---

# 2026-10-15 (öğlen): kamera önü yağış N1c — sıvı damla mercek, yüzey rüzgârı

Not: `ATMOSPHERE_NEAR_PRECIP.md` devir notu N1c. İki açık gözlem:
- **Koyu:** sıvı damla artık kapladığı arka planın %47'sini geçirir
  (mercek, Garg–Nayar), kalanı ortam + güneş parıltısı
  (`near_precip.glsl` `nearParticleLight`). Buz formları değişmedi.
- **Fazla hareketli:** yakın damla rüzgârı = iklim rüzgârı × 0.6
  (`CloudParams.h`, perdeyle aynı `kSurfaceWindFraction`).
Lens/kamera camı tasarımı: `ATMOSPHERE_LENS_WATER.md` (kod yok).

## Kontrol listesi (sırayla)
1. **Derleme:** C++ (`CloudParams.h` başlığı → bulut parametresi kullanan
   her çeviri birimi) + shader'lar: `near_precip.glsl` içerenler
   (`cloud_callable.rcall`, `cloud_scene.comp`, `cloud_sample.comp`).
   - Bozuksa "nearPrecipLayer: no matching overload" → eski imzalı bir
     çağrı kalmış (yalnız `nearPrecipComposite` çağırmalı).
2. **Probe:** `.\scripts\ipc\Probe-NearPrecip.ps1` → ALL PASS.
   - Yalnız 4 FAIL ve beklenen ≈ ölçülen/0.6 → C++ tarafı eski derleme
     (rüzgâr hâlâ tam değerde yazılıyor).
3. **Göz — aynı fırtına sahnesi, RT + RayFusion:**
   - Koyu zemin/ağaç önünde damlalar artık **parlak/gri çizgi**, sahne
     genel olarak kararmıyor; parlak bulut tabanı önünde hafif gri.
   - ★ Sinsi olan: yağmur neredeyse **görünmez** oldu → mercek payı fazla
     (`kNearLensKeep`); ya da hâlâ koyu → sahnedeki form yağmur değil
     (Measure at camera: dominant form) ya da shader eski.
   - Kar (Form = Snow): görünüm eskisiyle aynı olmalı (buz modeli).
4. **Rüzgâr:** aynı sahne, aynı `shutter_s` (0.135) → çizgiler belirgin daha
   dik ve kısa. Hâlâ çok yatay → `shutter_s`'i 1/30'a çekip tekrar bak
   (uzunluk = hız × enstantane, bilinçli).
5. **Dikiş:** "Drops around the camera" aç/kapa → uzak pus seviyesi yakın
   kalmalı. Açınca sahne belirgin **daha berrak** → sıvı damla yakın
   alanda perdeye göre az söndürüyor; söyle, `kNearLensKeep` ayarlanır.

---

# 2026-10-15: kamera önü yağış — derlendi, açık gözlemler

Kullanıcı N1 + N1b'yi derledi: **performans RT ve RayFusion'da arttı**
(gece 8/9 maliyet maddeleri geçti sayılır; sayı ölçülmedi). Açık: yağmur
**koyu** ve **rüzgârda fazla hareketli** — kök adayları ve düzeltme yönü
`ATMOSPHERE_NEAR_PRECIP.md` devir notunda (damla opak yayıcı gibi
çiziliyor, kırılan ortam değil; yakın damla tam rüzgârı, perde %60'ını
kullanıyor; sahnede shutter_s 0.135). Kar ve lens başka ajanda.

---

# 2026-10-14 (gece, 9): yağış maliyeti — perde örnek sayısı + kamera durumu kare başına bir kez

Ölçüm (canlı, RayFusion, fırtına sahnesi 178 mm/h, 1680×945, ısıtmalı, iki tur):
`ground_fraction 0` → ~122–135 ms; `1` → ~336–358 ms; yakın damla açık/kapalı
farkı kamera kuruyken ~0–10 ms. Yani patlamanın ~210 ms'si **uzak perdeden**
(`cloudPrecipMarch`; bu batch'ten önce de vardı). Kök: (a) her perde örneğinde
(piksel başına 48'e kadar) güneş gölgesi için `cloudOpticalDepth` (her bulut
aralığında 4 adım) → piksel başına yüzlerce yoğunluk değerlendirmesi;
(b) `cloud_scene` 200 m'lik bir yer ışınına da 48 örnek veriyordu.
Ayrıca yakın damlalar kameradaki durumu (oran, tip, gölge) HER pikselde
yeniden hesaplıyordu — kamera yağışın içine girince aynı patlama.

Düzeltme:
- Perde: örnek sayısı segment uzunluğuna bağlı (~80 m'de bir, 6..48);
  güneş gölgesi örnek başına değil, mesafeyle yenileniyor (max(250 m, 0.4 t)).
- Yakın damla: kamera durumu `cloud_sample.comp` mode 7 ile **tek iş
  parçacığında**, bulut parametre yüklemesinin aynı gönderiminde hesaplanıp
  parametre bloğunun sonuna (`CloudParamsGPU::nearState`) yazılıyor; RT ve
  RayFusion yalnız okuyor. Kamera kuruysa RT callable'ı hiç çağrılmıyor.
  Kamera hareketi parametreleri yeniden gönderiyor (yalnız damlalar
  çizilebilirken).

## Kontrol listesi (sırayla)
1. **Derleme:** C++ + tüm bulut shader'ları (`cloud_common.glsl` bloğu 128 B
   büyüdü: `CloudParamsGPU` ile birlikte; `cloud_sample.comp`, `cloud_rt.glsl`,
   `near_precip.glsl`, `raygen.rgen`, `cloud_scene.comp`, `cloud_callable.rcall`).
   - Bozuksa (bulutlar tamamen bozuk/siyah) → GLSL blok ile C++ struct sonu
     uyuşmuyor; `nearState` iki tarafta da EN SONDA olmalı.
2. **Maliyet (sayı):** aynı sahne, RayFusion. `viewport.frame_timings`
   → `ground_fraction 1` iken kare ~130–160 ms civarına inmeli (önce ~340).
   Bulut maliyeti `shadow_atlas` (gök pass'i) ve `post` (`cloud_scene`)
   aşamalarında görünür. RT'de örnek süresi de düşmeli.
   - Hâlâ yüksekse: `post` yüksek → `cloud_scene`'deki perde; `shadow_atlas`
     yüksek → gök pass'i (aralıklı, daha az etkilenir).
3. **Kamera durumu gerçekten kare başına mı:** `world.cloud_stats` →
   `backends[].near_state_dispatches`; kamera yağış altında ve yörüngede
   dönerken artmalı, damlalar kapalıyken DÜZ kalmalı.
   - ★ Sinsi olan: sayaç artıyor ama damla yok → state yazılıyor ama yanlış
     ofsete (`kCloudNearStateFloatOffset`); `Probe-NearPrecip` yine PASS
     verir (mode 6 kendi hesaplar), göz kontrolü şart.
4. **Kamerayı yağışın içine sok** (Measure at camera > 0) ve maliyeti tekrar
   ölç: damlalar açık/kapalı farkı RayFusion'da birkaç ms olmalı (tam
   çözünürlükte hücre adımları), yüzlerce değil.
5. **Görünüm:** perde eskisiyle aynı görünmeli. Yakın segmentlerde daha az
   örnek → çok yakın bir yamacın önündeki perdede hafif bant/gürültü
   artarsa söyle (80 m'de bir örnek eşiği).
   - ★ Sinsi olan: gölgeli bir perde ışınının uzak ucunda güneşli bulut
     kenarı ile gölge arası geçiş sertleşir/kayar (gölge 0.4 t mesafede
     yeniden kullanılıyor).
6. **Probe:** `.\scripts\ipc\Probe-NearPrecip.ps1` → ALL PASS (değişmedi).
7. Önceki partinin (gece, 8) göz kontrolleri (damla davranışı) aynen geçerli.

## Bilinen
- Kamerayı sürüklerken her hareket bir bulut parametre gönderimi yapar
  (4'lük halka, CPU nadiren bekler). Takılma görürsen söyle.

---

# 2026-10-14 (gece, 8): kamera önü yağış N1 — atmosferden damla alanı, eski overlay söküldü

Not: `docs/dev/ATMOSPHERE_NEAR_PRECIP.md`. Kamera önündeki yağış artık
atmosferin yağışı (`cloudPrecipSampleForm(kamera)`): oran, tip karışımı,
düşüş hızı, yüzey rüzgârı, perdeyle aynı ışık. Eski elle yağmur/kar/toz/sis
overlay'i, gök tonu ve duvar saatli "Realtime Weather Preview" CPU, OptiX,
Vulkan RT ve RayFusion'dan söküldü. `WeatherParams` = yalnız elle yüzey
kaplaması. Enstantane tek: `world.set_clouds {shutter_s}` (şimşek + çizgi).

## Kontrol listesi (sırayla)
1. **Derleme — C++ + TÜM shader'lar** (`compile_shaders.bat`).
   - vcxproj'dan `VulkanWeatherClock.cpp` çıktı; `WeatherClock.h` ve
     `weather_fast.glsl` silindi. Yeni include: `near_precip.glsl`.
   - Shader'lar: `raygen.rgen` (callable çağrısı), `cloud_callable.rcall`
     (yeni op 2), `cloud_scene.comp`, `cloud_sample.comp` (mode 6),
     `cloud_rt.glsl` / `cloud_common.glsl` (her bulut shader'ı). WorldData
     kopyalarında iki alan yalnız yeniden adlandı (`_weatherPad0/1`),
     yerleşim aynı (`audit_shader_struct_layout.py` OK).
   - Bozuksa "undeclared weatherVisualActive / WEATHER_TIME" → eski bir
     include kalmış; "cloudCall redefinition" → raygen iki kez
     `cloud_call.glsl` içeriyor.
2. **Probe:** `.\scripts\ipc\Probe-NearPrecip.ps1` → ALL PASS.
   - 2 FAIL (oran ≠ perde oranı) → yakın durum başka fonksiyondan geliyor:
     tek kaynak bozuk, en ciddi hata.
   - 4 FAIL (çizgi boyu yanlış) → rüzgâr paketi (`cirrusDirection.w`,
     `emissionSource.w`) ya da `precip2.x` = shutter yazılmıyor.
   - 6 FAIL → `precip3.w` kapalıyken 0 değil: perde kameradan başlamaz,
     yakın 15 m hiç sayılmaz.
3. **Göz — RT viewport, kamera yağışın ALTINDA** (Clouds > Precipitation >
   "Measure at camera" > 0 mm/h; değilse kamerayı çekirdeğin altına taşı):
   - arkadan alçak güneş → parlak, ince çizgiler; kapalı gök + koyu zemin →
     soluk çizgiler (bu doğru, gerçek yağmur böyle; "oyun yağmuru" değil);
   - Form = Snow → yavaş, salınan, rüzgârla eğik taneler; Hail → seyrek,
     uzun, parlak çizgiler.
   - Bozuksa hiç yok ve oran > 0 → RT: callable op 2 çağrılmıyor;
     RayFusion: `cloud_scene` dispatch'i yok (bulutlar kapalı mı?).
   - ★ Sinsi olan: damlalar var ama kamerayı **döndürünce/yürütünce kamerayla
     birlikte kayıyor** ya da timeline DURURKEN kaynıyor → kafes dünyaya
     sabit değil (`q0`) ya da zaman duvar saatinden geliyor. Doğru
     davranış: durmuş timeline'da damlalar donar, birikim temiz bir kareye
     yakınsar.
4. **RT ↔ RayFusion:** aynı kamera, aynı kare → aynı damlalar aynı yerde
   (aynı kafes, aynı hash). RayFusion'da DoF damlayı arka yüzeyin
   derinliğiyle bulanıklaştırır (bilinen, N2).
5. **Timeline:** oynat → yağmur ~6.5 m/s, kar ~1 m/s düşer; geri sar → aynı
   kare birebir aynı (zaman-saf).
6. **Dikiş (perde ↔ damla):** "Drops around the camera" aç/kapa → uzak
   pus seviyesi DEĞİŞMEMELİ, yalnız yakın çizgiler gelip gitmeli.
   - ★ Sinsi olan: açınca sahne belirgin şekilde **daha az puslu** → perde
     15 m'den başlıyor ama yakın peçe eklenmiyor (ortalama kayıp).
7. **Eski projeler:** elle "Rain" overlay'li sahnede havada yağmur YOK
   (beklenen: yağış artık Clouds > Precipitation'dan). Islaklık > 0 ise
   yüzeyler hâlâ ıslak. Weather bölümü "Surface weather (manual)"
   Wet/Snow/Dust gösterir; Mist'li eski sahne → None.
8. **Şimşek:** Exposure kaydırıcısı şimşek bölümünden Precipitation >
   Near camera'ya taşındı; varsayılan 1/30 s, flaşlar eskisi gibi.
9. **Maliyet:** kamera yağış altındayken RT örnek süresi ve RayFusion kare
   süresi (yakın alan tip başına ≤ ~100 hücre adımı, toplam bütçe 200).
   Pahalıysa `near_precip.glsl` bütçe 200 / bant hücre oranı 0.2 ayarı.

## Bilinen (N2'ye)
- Barınak yok: çatı altında da yağar. Kamera hızı, sıçrama, yerel ışıkla
  parlama yok. Ses köprüsü kameradaki oranı henüz okumuyor.

---

# Latest checkpoint — 2026-10-14

User confirmed successful Studio compilation after the Sound Designer MSVC fixes.
New Designer listening/UI/external IPC results are pending. The cloud V2d shader/
visual checklist below keeps its separate acceptance gate; Studio compilation does
not establish that a shader rebuild or visual/physics test passed.

# 2026-10-14 (gece, 7): şimşek yüksekliği hücrenin kolonuna (C++ başlığı)

V3b derlendi (kullanıcı). Gözlem: storm peak düşünce kolon kısalıyor ama
şimşekler hâlâ tam kolon yüksekliğinde → açık gökte çakıyor.
- Kök: `CloudLightningParams.h` yük bölgelerini (başlangıç %30–45, üst
  pozitif %72–87, örs %72–88, yere iniş açıklığı) TAM kolonun
  (taban→tropopoz) kesirleri olarak alıyordu; hücrenin kendi tepesi
  `min(tower/0.75, 1)` ile kısalıyor.
- Düzeltme: kesirler `cellDepth = kolon × clamp(tower/0.75, 0.15, 1)`
  üzerinden. Rastgele çekim sırası aynı (diğer şimşekler değişmez).

1. **Derleme** (C++; yalnız başlık).
2. **Göz:** peak'i düşür → şimşekler kısalan kulenin İÇİNDE kalmalı; peak
   yüksekken eskisi gibi.
   - ★ Sinsi olan: örs gezgini (tür 3) zayıf hücrede örsün değil kule
     tepesinin hizasına iner — örsü olmayan hücre için doğru.

---

# 2026-10-14 (gece, 6): bulut hacimleri V3b — ufuk renk çizgisi (yalnızca shader)

V3 derlendi (kullanıcı). Gözlem: yukarıdan bakınca GÖLGEDEKİ bulutlar ufuk
çizgisinin altında kahverengi, üstünde gök renginde; iki backend'de de.
- Kök: hava perspektifi terimi (`inscatter·Tap + sky(d)·(1−Tap)`) bakış
  yönünün gök değerini kullanıyordu; ufkun altında o değer ZEMİN rengi.
  Gölgede bulut karanlık olduğu için bu pus terimi baskın. Ortak marcher'da
  (`cloud_rt.glsl`), bu yüzden RT ve RayFusion aynı.
- Düzeltme: `cloudAerialSky(d)` — yön ufkun hemen üstüne kısılıyor (4 yer:
  katman, sirrus, yağış sütunu, referans izleyici).
- Ek: `cloudSegmentMayContain` alçaltılmış kutu tabanını bilmiyordu
  (RayFusion sahne geçişi + RT isabet parçası): kutu tabanı ile kolon tabanı
  arasındaki parça bulut yok sayılabilirdi.

## Kontrol listesi
1. **Derleme** (yalnızca shader).
2. **Ufuk çizgisi (göz, yukarıdan, iki backend):** gölgedeki bulutların
   rengi ufuk çizgisinde değişmemeli; uzaklaştıkça pus rengine yumuşakça
   geçmeli.
   - Bozuksa hâlâ kahverengi → kaynak pus değil; ortamın `below` (zemin)
     payı. Normalin aşağı baktığı yüzlere bak (alt yüzler kahve OLABİLİR,
     ama ufukla değişmemeli).
3. **Sıra (sahne objeleri):** kod yolu doğru görünüyor — RT'de her
   isabette kamera→isabet parçasına bulut uygulanıyor, RayFusion'da
   `cloud_scene.comp` derinlikten aynısını yapıyor. Bulutun önünde
   görünmesi için objenin bulutun ARKASINDA (km ötede) olması gerek; yerde
   kameraya yakın objeler fiziksel olarak zaten bulutun önünde.

---

# 2026-10-14 (gece, 5): bulut hacimleri V3 — taban altı, içe akış, eğim

V2e CANLI ("çok daha doğal"). Bu parti: C++ başlığı (`CloudParams.h`) +
shader'lar (`cloud_volume_gen.comp`, `cloud_rt.glsl`); probe güncellendi.

1. **Taban altı yeri:** hacim kutuları kolonun 900 m altına iniyor (en
   fazla, 100 m irtifanın altına inmez; bu sahnede ~440 m). Marcher kolonun
   en alt aralığını kutu tabanına kadar uzatıyor.
2. **Sarkan taban:** güncelleme bölgesinde (çekirdeğin ~R çevresi) yuvarlak
   lob kubbeleri, düz tabanın en fazla 420 m altına; duvar bulutu 450 m'ye
   kadar.
3. **Scud:** 28 parça, inflow yarısında doğuyor, ömrü boyunca çekirdeğe ve
   tabana doğru çekiliyor, büyüyüp sıfıra küçülüyor (sarmada sıçrama yok).
4. **İçe akış:** besleyici kümülüsler de aynı ömür döngüsüyle fırtınaya
   doğru kayıyor ve doğup sönüyor.
5. **Eğim yarıya** (1.6 → 0.8): kuleler duman sütunu gibi değil.
6. **Örs alt kalınlığı** 60 → 250 m: lifli oyuk kenarı havada adacıklara
   bölüyordu.

## Kontrol listesi (sırayla)
1. **Derleme:** C++ (CloudParams.h, başlık) + shader'lar.
2. **Probe:** `.\scripts\ipc\Probe-CloudVolumes.ps1` → PASS. 3b artık
   "taban kutu tabanı ile LCL arasında" ölçüyor ve kaç hattın LCL altına
   sarktığını yazıyor (≥ 1 beklenir).
   - Bozuksa 3b bir taban LCL'nin üstünde → lob/kesme işareti ters.
3. **Taban (göz, alçak açı):** çekirdeğin altında sarkan yuvarlak loblar,
   kenarda düz taban; altında dağınık, yırtık küçük parçalar.
   - Bozuksa taban hâlâ bıçak gibi düz → kutu inmedi (ekran: `cloud_stats
     .volumes.boxes[].min[1]` kolon tabanının altında olmalı) ya da
     marcher aralığı uzamadı.
   - ★ Sinsi olan: loblar görünür ama kutu tabanında **dümdüz kesilir**.
     Kutu tabanı = taban − 440 m; lob en fazla taban − 420 m, scud
     `room × 0.9`: kesik görülürse oda hesabı yanlış.
4. **Zaman (oynat):** scud ve besleyiciler yavaşça fırtınaya akmalı, ani
   belirip kaybolmamalı. Her 4 sn'lik yeniden boyamada küçük adım normal.
5. **Eğim ve örs (göz):** kuleler daha dik; örs çevresinde havada kopuk
   parça kalmamalı.
6. **Maliyet:** üretim voksel başına +1 Worley, +28 elipsoit (yalnızca
   taban altı). Takılma varsa `repaints_last_update`'e bak.

## Bilinen
- Uzak kule tabanlarındaki soluk şerit bu partide ele alınmadı.
- Bu sahnede yağış görünmüyor: `ground_fraction = 0`.

## 2026-10-14 Audio sound designer v2 — Studio build only

No additional shaders for this delivery. Source contracts: AUDIO_SOUND_DESIGN.md.
1. Build Studio (new AudioSoundDesign, AudioDspModules, AudioSoundDesignerUI modules).
2. Audio > Assets > Design sound: Designer bottom tab, controls in Audio Inspector.
3. Render/listen without selecting a folder: no WAV/asset until Save asset.
4. Adjust preset, noise, ADSR handles, filters/drive; release previews the new sound.
5. Import mono/stereo; Design / process; compare original and processed. Change
   pitch and duration independently; try loop seam crossfade, loop and stop.
6. Save by name twice: project/assets/audio/generated (or existing asset-library
   root for unsaved scenes), suffix _2, preserved existing files. Folder picker.
7. Timeline insertion, undo, scene save/reopen and portable asset packaging.
8. Narrow/high-DPI inspector, short bottom panel scrolling, waveform/graphs/meters.
9. Run externally: python scripts/test/rt_test_audio_sound_design_ipc.py
   --output-dir E:\YOUR_EXISTING_TEST_FOLDER (add --audition only to hear preview).
10. Check assets/licenses/audio_dsp notices in the built distribution.

Codex performed source/static checks only; build/listening acceptance is pending.
