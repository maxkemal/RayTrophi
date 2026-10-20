# 2026-10-19: dinleme geri bildirimi — karakter revizyonu 2

> **Durum:** Kaynakta; bu revizyon derlenmedi veya dinlenmedi. Önceki teknik PASS
> sonuçları revizyon 1'e aittir. Birleşik domain dosyaları değiştirilmedi.

1. Kullanıcı Studio C++ build'i. AudioCharacter DSP yeni .cpp/.h proje kayıtlı;
   bu audio işi shader değişikliği istemez.
2. Source / Sound / Synthetic generator / v2 / Sound character -> Tuned.
   Su–gaz–sürtünme artık ayrı karakter hedefliyor. Friction surface dört aile sunar.
3. Dış terminal: `python scripts/test/rt_test_audio_natural_ipc.py`;
   yalnız boş/geçici audio sahnesinde aynı test `--exercise` ile.
4. Metal ve cam darbe revizyon 1/2 PCM hash'leri aynı kalmalı; ahşap/taşın kuyruğu
   belirgin kısalmalı. Dört sürtünme yüzeyi ayrı çıktı vermeli. History/seed aynıysa
   her revizyonda bake deterministik olmalı.
5. Yeniden dinleme: `rt_listen_audio_natural_ipc.py` artık revizyon 2 seçer;
   karşılaştırmada `--character-revision 1`, sürtünmede `--surface mineral` vb.
   Model adını her aşamada ayrıca bildir. Kırılmanın tek tip tık yerine kısa kümeler
   vermesini dinle. Statik kontroller doğallık kanıtı değildir.
6. İsteğe bağlı C++ harness üç DSP .cpp'siyle kullanıcı tarafından derlenir:
   block partition, sessizlik, metal/cam koruma, ahşap/taş sönümü ve yüzey farkı.

# 2026-10-19: kod partisi — odaklı Undo/Redo ve fiziksel üretici v2

> **Durum:** Kaynakta teslim. Codex derlemedi, Studio açmadı, canlı probe/dinleme yapmadı.
> Birleşik domain solver/shader/test dosyaları değiştirilmedi. Önceki liste aşağıda.

1. Kullanıcı Studio C++ build'i. Bu audio partisi shader değişikliği gerektirmez;
   diğer ajanın shader gereksinimleri kendi kapsamındadır.
2. Edit -> Designer'da Undo Sound Designer / Redo Sound Designer; Ctrl+Z/Y/Shift+Z.
   Viewport'a tıkla: scene geçmişi. Arrange de scene geçmişi. Menü açılması hedefi
   değiştirmemeli; text input kendi Ctrl+Z'sini almalı; boş Designer history sahneyi etkilememeli.
3. Kaynak Sound -> Synthetic generator: model seç, v1/v2 değiştir, sayıları Enter ile
   uygula. Sürekli kaynakta Activity connection ile uygun domain/signal bağla;
   impact'ta Excite impact dene. Eski sahnede v1 kendiliğinden değişmemeli.
4. Dış terminal: `python scripts/test/rt_test_audio_natural_ipc.py`.
   Katalog ve geçici history_scope doğrulaması; çalışma görünümü sonunda geri yüklenir.
5. Yalnız BOŞ/GEÇİCİ audio sahnesinde aynı komutu `--exercise` ile çalıştır:
   sekiz model/v1-v2, dry bake hash/seviye, parametre farkı ve atomik hata/history testleri.
   Sahne undo geçmişi oluşturur; çalışma sahnesinde koşturma.
6. Her v2 aileyi düşük/orta/yüksek girişte dinle; başlangıç/duruş, farklı seed,
   body/texture ve malzemede klik/ton/tekrar kontrolü. Önceki fizik/echo probe'ları
   ayrı regresyon kapısı. Doğal ses kalitesi FFT veya kaynak incelemesiyle kabul edilmez.
7. İsteğe bağlı bağımsız C++ harness: `scripts/test/physical_audio_dsp_test.cpp`;
   PhysicalAudioDsp.cpp ve PhysicalAudioNaturalDsp.cpp ile kullanıcı derler.
   Block-size, sıfır activity, seed ve darbe enerji sırası testleri içerir.

# 2026-10-19: Analyze durum satiri yerlesim duzeltmesi

> **Durum:** Kullanici graph Ctrl+wheel calismasini dogruladi. Bu duzeltme kaynakta;
> yeni build/gorsel kontrol bekliyor. Onceki kontroller asagida korunmustur.

1. Live Analyze acikken sesi baslat/durdur veya kanal mute yap: Signal active /
   Channel silent degisirken Peak/RMS, graph ve alt kanal tablosu yer degistirmemeli.
2. Output mute ve full-scale uyarisinda da ayni tek satir korunmali.
   Dar panelde metin satir sarmamali; ayrintilar hover tooltip'te okunmali.
3. Gecis sirasinda Ctrl+wheel ile zoom ve orta tus pan yap: graph konumu ve font
   boyutlari sabit kalmali. Codex bu turda build veya uygulama testi yapmadi.

# 2026-10-19: Analyze alternatif gorunumler ve mouse gezinme

> **Durum:** Temel canli analiz kullanicida calisti. Bu genisletme kaynakta hazir;
> yeni build/gorsel kabul bekliyor. Codex derleme veya uygulama baslatma yapmadi.

1. Dis terminal: `python scripts/test/rt_test_audio_live_analysis_ipc.py`.
   Beklenen: dort gorunum ve viewport sinirlari, atomik hata reddi,
   details/history sozlesmesi; workspace sonunda geri yuklenir.
2. Live Analyze / View: cizgi, dolu alan, alti sutun, spektrogram arasinda gec.
   Spektrogram zaman/Hz gostermeli; renkler -96..0 dBFS, bosluklar orneklenmeyen zaman.
3. Yalniz grafik uzerinde Ctrl+tekerlek imlec merkezli zoom; orta surukleme pan.
   Dugme yok, orta cift tik reset yok. 1x'e uzaklastirinca tam gorunum gelir.
   Kanal gostergeleri ve fontlar buyumemeli; duz tekerlek paneli kaydirmali.
4. Kanal degistir, sesi durdur/baslat: veri akmali, history karismamali,
   yeni pencere yoksa son snapshot uyarisi kalmali. Designer son-render analizi korunmali.
5. Istege bagli bos/gecici audio sahnesinde ayni dis test `--signal` ile gercek
   stereo FFT/bant/gain/mute ve history sozlesmesini kontrol eder.

# 2026-10-19: Designer kompakt üst bant

> **Durum:** Kaynakta hazır; build ve görsel kabul kullanıcıda. Canlı uygulamaya dokunulmadı.

1. Designer: geniş panelde ince sekme satırı ve tek kontrol satırı; ad/kaynak,
   Play/Stop, workflow, Save/Options görünür. Ana tema renkleri korunmalı.
2. Daralt: kontroller alta taşmamalı; yatay kaydırma ile erişilebilmeli.
3. Options: Loop, Original/Processed, Refresh, Undo/Redo, ölçek/Fit ve dock
   aç/kapat çalışmalı. Save penceresinde asset/klip hedefleri korunmalı.
4. Generate/Edit/Effects ve canlı Analyze arasında geçişte ana alan taşmamalı;
   araç çubuğunun küçük boyutları alt editör kontrollerine sızmamalı.

# 2026-10-19: canlı Audio Analyze

> **Durum:** CANLI — kaynak teslimi; yeni C++ build'i ve canlı kabul bekliyor.
> Önceki Substance ve birleşik domain kontrolleri aşağıda korunmuştur.

1. **Hızlı mevcut sahne:** dış terminalden
   `python scripts/test/rt_test_audio_live_analysis_ipc.py`.
   Beklenen: beş kanal, sonlu spektrum/bantlar (cihaz açıksa), geçersiz bus/mode
   reddi ve workspace'in geri yüklenmesi. Unknown operation ise exe eskidir.
2. **Görsel hızlı:** Master signal'e tıkla: Live açılmalı; Inspector Channel Peak'e
   tıkla: o kanal seçilmeli. Canlıda Designer Play/Save/Before/After görünmemeli.
   Kanal sesi/gain/mute değişirken spektrum güncellenmeli; render istememeli.
   Output mute öncesi ölçüm uyarısı doğru, sessizlik ve bekleme ayrı olmalı.
3. **Bağımsız PCM kanıtı:** boş disposable audio sahnesinde aynı scripti `--signal`
   ile çalıştır. 1 kHz ters polariteli stereo iptal olmamalı; yarım kanal gain'i
   -6.02 dB, Master gain değişimi yalnız Master grafiğini etkilemeli. Geçici
   kaynak/asset/dosya temizlenmeli; mixer ve output mute geri yüklenmeli.
4. **Son render ayrımı:** Designer: last render, kayıtlı render FFT/spektrogramını
   göstermeli. Sahne sesleri bunu değiştirmemeli. Live'a dönüş güncel miks olmalı.
   En sinsi hata: eski Designer spektrumunu canlı gibi göstermek; `--signal`
   sırasında Designer revision değişmeden gerçek kanal kazancını ölçerek yakalanır.

Ayrıntı: [AUDIO_LIVE_ANALYSIS.md](AUDIO_LIVE_ANALYSIS.md).

---

# 2026-10-20 (9. build): kapanış partisi — fluid.run, donmuş domain, T0

> **SONUÇ (canlı, 2026-10-20):**
> - **T0 EQUAL, iki fazda da (karar 1'in kapısı GEÇTİ).** Ölçüm `fluid.run`
>   sunucu `step_ms` medyanıyla yapıldı.
>   - Gaz: 5.86 → 6.20 ms (1.06×), aktif yoğunluk hücresi 7188 = 7188.
>   - Sıvı: 49.60 → 48.21 ms (0.97×), 80000 = 80000 parçacık; merkez ve hız
>     aynı.
>   - T6 (uzman domain tiplerinin sökümü) artık kullanıcı kararı.
> - Hızlı katman: substance, thermal_films, phase3_freeze, fire_steam ve
>   grain_readiness PASS.
> - **p9_floor:** donmuş domain'de tane adımı artık koşuyor; kum asılı
>   kalmıyor. 868 buz yüzü var, ama kum buzun içinden zemine iniyor
>   (min y 0.0099, COM 0.116 < buz üstü ~0.25). → Sonra listesi. Aday:
>   statik collider yapısı parmak iziyle önbelleğe alınıyor olabilir, buz
>   yüzleri o parmak izine girmiyor.
> - **p9_melt:** eriyor (200 tanenin 200'ü, 14.2 kJ), ama kütle kapısı
>   ölçüm artefaktıyla düştü: `fluid.run`'dan hemen sonraki ilk
>   `fluid.matter_models` okuması 0 kg verdi. → Sonra listesi (okuma, erime
>   değil).
> - seed_settle 2/5: bilinen (DEM titremesi, FLIP); Sonra listesinde.

**Derleme:** yalnız C++. Kapanış kapsamının (MATTER_ROADMAP) dört maddesi:

1. **10× yavaşlama: kodda değil.** Taze açılmış uygulamada grain_heat 120 adım
   13 s (ilk build'deki hız). 451 s, saatlerce açık kalan oturumdaki
   birikimdi. Duman koşucusu artık her testten sonra `T_` / `T: `
   adlı domain, kaynak, alan ve maddeleri siliyor; isteğe bağlı
   `--restart-app` uygulamayı yeniden açıyor (açık sahneyi kapatır).
   Oturum birikiminin kökü "Sonra" listesinde.
2. **`fluid.run(steps, dt, sample_every, domains, fields, sum_fields, max_fields)`:**
   N adım tek çağrıda. Örnekler, toplamlar ve en büyükler tek yanıtta dönüyor.
   Sunucu tarafında ölçülen `step_ms` (median / p90 / max / mean / all) de
   geliyor. `fluid.list_domains` aynı satır kurucuyu (`fluidDomainListJson`)
   kullanıyor. Python: `rt.fluid.run(steps, dt)`.
3. **Donmuş domain'de tane adımı:** `inspectMatterOwners` donmuş parselleri
   engel sayıp adımı tutuyordu ("does not accept pinned … carriers").
   Artık yalnız Solid bağlı statik bloklar tutuyor.
4. **P8 T0:** `rt_t0_matter_vs_specialized_perf.py` artık `fluid.run`
   `step_ms` ile karşılaştırıyor ve sonuç eşitliğini ölçüyor (sayılar %2,
   sıvı merkezi bir voksel, hız %10). Faz başına `PASS/FAIL t0_<faz>_equal`.

Sıra (hızlı katman, ~10 dakika):
1. `python scripts/test/rt_matter_smoke.py --quick`. Kollar: substance,
   thermal_films, phase3_freeze, fire_steam, grain_readiness, seed_settle,
   p9_melt, p9_floor.
   ★ `p9_floor` FAIL ve kum y ~0.45'te kaldıysa adım hâlâ tutuluyor:
   `mixed_status_last`'a bak.
2. `python scripts/test/rt_matter_smoke.py --only t0`: karar 1'in kapısı.
   Her iki faz EQUAL ise T6 (uzman domain tiplerinin sökümü) açılır; karar
   kullanıcının.
3. Plan sonu (bir kez): `python scripts/test/rt_matter_smoke.py --restart-app`.

# 2026-10-20 (8. build): P9 — buz çarpışması, Ice erimesi, cihaz buharı, gözenek ısısı

> **SONUÇ (canlı, 2026-10-20):**
> - **ice_melt PASS (4 kapı da):** 200 Ice tanesinin hepsi eridi
>   (`grains_end` 0, 202 erime olayı). Gizli ısı 14.6 kJ, tane en yüksek
>   273.55 K. Kütle 140.283 → 140.277 kg (bağıl 4.8e-5).
> - **ice_floor FAIL:** su tamamen dondu (`frozen_peak` 6000), ama
>   `ice_faces` 0 ve kum, doğduğu yükseklikte asılı kaldı (y 0.42–0.50;
>   3.5 s'de hiç düşmedi). ★ Bulgu: sıvı tamamen tutulan buz olunca tane
>   adımı hiç koşmuyor gibi. Buz yüzleri de bu yüzden raporlanmadı; çarpışma
>   yolu sınanmış sayılmaz.
> - **pore_heat koşmadı:** test 1200 s sınırını aştı. Kalıntı domain
>   sonraki testleri yavaşlattı, elle temizlendi.
> - Regresyon: gas_drag 6/6, contact_heat 7/7, grain_heat 8/8. ★ Ama
>   grain_heat 43 s'den 451 s'ye çıktı (temiz sahnede); ~10× yavaşlama, kökü
>   açık. Adaylar: kare başı buhar indirmesi (eşzamanlı fence) ve P6–P9 arası
>   diğer kare başı işler. `fluid.step_stats` ile aşama süreleri ölçülmeli.
> - sleep_transfer: dört kol PASS, strong A/B enerji kapısı yine sınırda
>   FAIL (3 koşunun 2'sinde).
> Sonraki parti:
> 1. Donmuş domain'de tane adımının neden durduğu.
> 2. grain_heat yavaşlamasının aşama ölçümü.
> 3. P9 testini kollara bölmek (`--arm`) ve kısaltmak.
> 4. sleep_transfer enerji toleransını gözden geçirmek.

**Derleme:** yalnız C++ (shader yok). 6. ve 7. build'in yazılmış kodu
(oturtma eşleşmesi, seed kesilme okuması, basınç iterasyonu IPC) de bu
derlemeye giriyor. Yeni dosya yok.

Yazılanlar:
1. **Taneler buza çarpıyor.** Donmuş sıvı parsellerinin hücreleri her karede
   dış yüzleriyle statik bir üçgen kümesine dönüşüyor (`appendGrainIceFaces`,
   aralık `held_ice`, hızı analitik sıfır) ve DEM'in mevcut üçgen temasına
   giriyor. Shader değişmedi. Okuma: `grain_diagnostics … ice_faces`.
2. **Gözenek suyu ısısını taşıyor.** Emilen su, hücresindeki sıvının
   sıcaklığıyla geliyor; tane sıcaklığı ısı kapasitesine göre karışıyor
   (`exchangeMatterGrainWater(…, thermal)`). Okuma: `pore_heat_mixed_j`.
3. **Cihazdaki buhar envanteri tane↔hava filmine ulaşıyor.** Ana bellek
   işaretçisi boşsa, buhar sütunu tane ısı geçişinden önce kare başına bir
   kez indiriliyor (hücre × 4 bayt). Okuma: `thermal_grain_air_humid`.
4. **Ice erimesi kodda vardı**; artık canlı testi var.

Sıra:
1. `python scripts/test/rt_matter_smoke.py --only p9_matter`: 8 kapı, 3 kol.
   - `ice_reaches_melt_point`, `latent_heat_paid`, `ice_grains_melt`,
     `melt_conserves_mass`. **Bozuksa:** `latent_heat_paid` PASS ama
     `ice_grains_melt` FAIL ise erime sürüyor ama 12 s yetmiyor; 4 mm tane
     ~12 s'de tam erir, kısmi erime de geçerli kanıttır (sayıya bak).
   - `held_ice_has_faces`, `sand_rests_on_ice`. **Bozuksa:** `frozen_peak`
     0 ise 180 K su donmadı ya da doğrudan Ice tanesi olarak doğdu (doğum
     kuralı); `ice_faces_peak` > 0 ama kum zeminde ise üçgen teması tek
     taraflı olabilir (yüz yönü).
   - `absorbed_water_brings_heat`, `grain_air_reads_vapour`. ★ Sessizce makul
     görünen: `humid_steps` 0, diğer her şey PASS. Bu, gaz şeridinin bu
     sahnede buhar envanteri tutmadığı anlamına gelebilir; envanterin
     cihazda olup olmadığına bakmadan "kuru hava" sonucu kabul etme.
2. `python scripts/test/rt_matter_smoke.py --only grain_heat contact_heat gas_drag sleep_transfer`:
   tane yolu regresyonları. Hepsi geçmeli.
3. (Park) `liquid_column`: P6d ölçümü, istenirse.

# 2026-10-20 (7. build): seed kesilme uyarısı + seed okuması + sıvı basınç iterasyonu IPC

**Derleme:** yalnız C++ (shader yok). 6. build'in oturtma eşleşme düzeltmesi
de bu derlemeye giriyor (henüz derlenmedi).

Yeni:
- **Seed kesilmesi artık sessiz değil.** Seed, istenen parçacık sayısını
  (bölgedeki katı olmayan hücreler × ppc; Fill Domain'de hedef seviye)
  bütçeyle karşılaştırıyor. Eksik kalırsa log'a uyarı düşüyor ve panelde
  "Last seed was cut by Max Particles: X of Y …, liquid ends at y=…" satırı
  çıkıyor. Okuma: `seed_mode`, `seed_min/max`, `seed_particles_per_cell`,
  `seed_fill_level`, `seed_ran`, `seed_truncated`,
  `seed_requested/emitted_particles`, `seed_top_y`, `seed_region_top_y`.
- **Sıvı basınç çözücüsü IPC'de:** `fluid.set_param pressure_iterations=`,
  `pressure_relative_residual=`. Okuma: `pressure_iterations`,
  `pressure_relative_residual`, `pressure_cg_iterations`,
  `pressure_cg_max_iterations`, `pressure_cg_final_relative_residual`,
  `pressure_cg_multigrid`. Panelde "Poisson Pressure Iterations"ın altında
  son adım satırı var; tavana vurunca "(hit the cap: liquid may compress)"
  yazıyor. Not: CPU PCG yolu son kalıntıyı raporlamıyor (0 kalır). Vulkan
  hızlı yolunda kalıntı, son kontrol anındaki değer.

Sıra:
1. `python scripts/test/rt_matter_smoke.py --only liquid_column`: 3 kapı.
   `truncation_reported`, `pressure_reported` ve `level_holds_at_400`.
   Asıl çıktı `docs/dev/liquid_column_live.json` içindeki 24 / 100 / 400
   satırları: `level_ratio`, `cg_iterations_mean`, `cg_residual_mean`.
   **Okuma:**
   - 24'te ~0.74 ve 400'de ~1.0: kök iterasyon tavanı. Karar: tavanı ızgara
     boyutuna bağla, ya da çok-ızgaralı ön koşulu aç.
   - 400'de de ~0.74: kök başka yerde (yoğunluk süzgecinin %13'lük ölü
     bölgesi, ΔU ∝ h).
   ★ Sessizce makul görünen: `cg_iterations_mean` tavandan küçük, ama seviye
   yine düşük. Bu, çözücünün "yakınsadım" dediği hâlde sütunu taşımadığı
   anlamına gelir; o zaman tolerans fazla gevşek.
2. `python scripts/test/rt_matter_smoke.py --only seed_settle`: su
   `settle_steps` artık 1 değil (6. build düzeltmesi).
3. Kullanıcı sahnesi: seed'i yeniden yap. Panel kesilme satırını göstermeli
   (100k bütçe).

# 2026-10-19 (6. build): oturtma eşleşme kusuru + FLIP ölçümü (konumla)

**Derleme:** yalnız C++ (`MatterSeedSettle.inl`, `ParticleSimulation.h`).
Oturtma artık eşleşen taşıyıcı yoksa yakınsadı saymıyor; id yoksa ya da id'ler
tekrarlıyorsa, sayı değişmediği sürece indeksle eşliyor. Kare probu da aynı.
Yeni sayaç: `measured_steps` (henüz okumaya bağlı değil).

**FLIP ölçümü (5. build, oturtma kapalı, dinlenik tank, 60 kare, ilk 10 atlandı):**

| FLIP | voksel | parçacık | p95 mm/kare | ortalama mm/kare | ortalama hız |
|---|---|---|---|---|---|
| 0.9 | 5 cm | 2304 | 3.42 | 1.81 | 0.124 m/s |
| 0.9 | 2.5 cm | 16128 | 4.37 | 1.51 | 0.095 m/s |
| 0.3 | 5 cm | 2304 | 0.59 | 0.32 | 0.020 m/s |
| 0.3 | 2.5 cm | 16128 | 0.80 | 0.35 | 0.021 m/s |

Okuma:
- Hareket konumda gerçek. FLIP 0.9 ile 0.3 arasında yaklaşık 5–6 kat fark var.
- ★ İnce çözünürlük gürültüyü KÜÇÜLTMÜYOR (mutlak mm/kare aynı, p95 biraz
  artıyor). "Prodüksiyonda ince ızgara gizler" varsayımı burada tutmadı.
  Yani gürültü h ile ölçeklenen bir aktarım hatası değil; adım başına
  h'den bağımsız bir hız ölçeğinde enerji giriyor.
- FLIP 0.3'te bile 0.02 m/s kalıyor (FLIP 0 probu: 0.004 m/s). Tabanın
  bir kısmı FLIP dışından geliyor; adaylar duvar/serbest yüzey sınırı.
- Dökülme canlılığı aşağıda.

**Dökülme canlılığı (5. build, baraj yıkılması):** 80×60×40 cm kapalı tank,
24×39×38 cm su sütunu, 2.5 cm voksel, 11968 parçacık, 2 s, oturtma kapalı.

| ölçü | FLIP 0.9 | FLIP 0.3 | fark |
|---|---|---|---|
| tepe ortalama hız | 1.19 m/s (0.28 s) | 1.10 m/s (0.30 s) | −8% |
| tepe en yüksek hız | 3.18 m/s | 3.36 m/s | +6% |
| balistik sıçrama v²/2g (kaba) | 0.52 m | 0.57 m | +11% |
| tepe yer değiştirme (en yüksek) | 76 mm/kare | 60 mm/kare | −21% |
| çarpmadan sonra kütle merkezi y en yüksek | 0.127 m | 0.106 m | −17% (duvara tırmanma) |
| ortalama hızın yarıya inmesi | 0.62 s | 0.60 s | aynı |
| 0.6 s sonrası sallanma (merkez x aralığı) | 0.244 m | 0.212 m | −13% |
| 1.5–2 s ortalama hız | 0.39 m/s | 0.22 m/s | −45% |
| 1.5–2 s p95 yer değiştirme | 13.6 mm/kare | 7.9 mm/kare | −42% |

Okuma: çarpmanın kendisi (tepe hızlar, sönümün ilk yarı ömrü) neredeyse
aynı; en hızlı sıçrayan parçacık 0.3'te biraz daha hızlı. FLIP 0.3'ün
kaybettiği şey, duvara tırmanma ve sallanma genliğinin %13–17'si. Kazandığı
şey: geç evrede ~%45 daha sakin, dinlenik tankta 5–6× daha sakin. Ölçülmeyen:
ince sıçrama damlacıklarının görünümü (render), bulut/köpük üretimi.
Sıçrama yüksekliği için doğrudan bir okuma (en yüksek parçacık y'si) yok;
v²/2g yalnız bir tavan tahmini.

**Hacim sıkışması (kullanıcı gözlemi, ölçüldü, 5. build):** 60×60 cm kapalı tank,
tank genişliğinde 20 cm su levhası. Bir kolda zeminde başlıyor, bir kolda 30 cm'den
düşüyor. Oturtma kapalı, τ = 0.5 s, ppc 8, 2.5 s.

| voksel | parçacık | sayıdan beklenen seviye | zeminde | düşen |
|---|---|---|---|---|
| 5 cm | 5760 | 0.250 m | 0.258 m | 0.260 m |
| 2.5 cm | 41472 | 0.225 m | 0.226 m | 0.225 m |
| 1.25 cm | 313344 | 0.2125 m | 0.181 m (−15%) | 0.180 m |
| 1 cm | 625128 | 0.217 m | 0.161 m (−26%) | 0.147 m (−32%) |

(Seviye = 2 × kütle merkezi y; tam genişlikte levha olduğu için.) 1 cm'de
çarpmada kütle merkezi 7 mm'ye iniyor, levha neredeyse ezilip geri açılıyor.
τ 0.05 s ile 5 / 2.5 cm'de fark yok.

Okuma: sıkışma çarpmadan bağımsız. Duran suda da var ve çözünürlükle büyüyor.
★ Baş aday: sıvı basınç çözücüsünün iterasyon tavanı
(`APICSolverParams::pressure_iterations = 24`, Jacobi ön koşullu CG).
Hidrostatik basınç 20–80 hücrelik bir sütunda 24 iterasyonda kurulamaz; sütun
yükünü taşıyamayan çözüm suyu sıkıştırır. Eski yoğunluk terimi h'den bağımsız
güçlü itmesiyle bunu maskeliyordu: 5. build'de onu düzeltince açığa çıktı.
İkinci katkı: yeni terimin gürültü süzgeci (ppc 8'de ~%13 fazla yoğunluğa
dokunmuyor) ve ΔU = h·e/τ'nun ince ızgarada zayıflaması.
Sınanamadı: sıvının `pressure_iterations` ve `pressure_relative_residual`
değerleri yalnız panelden ayarlanıyor. IPC'ye açık değiller, kullanılan
iterasyon ve son kalıntı da raporlanmıyor (kural 1 deliği).

Sıradaki parti:
1. Sıvı basınç iterasyonunu ve kalıntısını `fluid.set_param` /
   `fluid.list_domains`'e aç; kullanılan iterasyonu ve son kalıntıyı raporla.
2. 1 cm levhada iterasyon 24 / 100 / 400 ile sıkışmayı ölç.
3. Sonuca göre: iterasyon tavanını ızgara boyutuna bağla (ör. ≥ 2·max(nx,ny,nz))
   ya da Vulkan'da çok-ızgaralı ön koşulu aç; gerekirse süzgeç eşiğini
   hacim kaybına karşı daralt.
Not: kullanıcı sahnesi (5 m domain, 5.5 cm) tam 100k parçacık bütçesinde
duruyor. Seed tabandan yukarı katman katman dolduğu için, bütçe yetmezse üst
katmanlar hiç doğmuyor; bu ayrı bir "yükseklik eksik" kaynağı.

Sıra (6. build sonrası):
1. `python scripts/test/rt_matter_smoke.py --only seed_settle`: suyun
   `settle_steps` değeri artık 1 olmamalı; ya yakınsar ya düzlüğe oturur.
   Kum kapıları yine düşebilir (DEM dinlenme titreşimi, kök açık).

# 2026-10-19 (5. build): oturtma yer değiştirme ölçütü + kare başı hareket okuması + yoğunluk gevşemesi

> **SONUÇ (canlı, 2026-10-19):** madde 1/1, contact_heat 7/7, surface_water 6/6,
> FG1 8/8. Yeni yoğunluk terimi sıvı testlerini bozmadı; FG1 süresi 368 sn'den
> 70 sn'ye indi.
> **seed_settle 2/5:**
> - **Kum:** oturtma 5 adımda yakınsadı (p95 2.33 → 0.14 mm/adım, 76 ms). Ama
>   serbest koşuda oturtulmuş kol, oturtulmamıştan daha çok hareket ediyor:
>   p95 1.49 mm/kare (oturtulmamış 1.21), COM 2.5 mm kayıyor. Hızlar sıfırlıyken
>   yatak sürünüyor, serbest bırakılınca yeniden titriyor. ★ DEM yatağının
>   dinlenme tabanı da gerçek; kökü açık.
> - **Su:** ★ KUSUR. `settle_steps` 1, p95 0 ve "converged" geldi. Seed'den
>   hemen sonra parçacık id'leri henüz yok, eşleşen örnek sayısı 0 çıkıyor ve
>   p95 = 0 yakınsama sayılıyor. Düzeltme sonraki partide: örnek yoksa
>   yakınsamış sayma, id yoksa indeksle eşle.
> - **Su serbest koşu (konumla):** iki kolda da p95 4.35 mm/kare (~0.26 m/s).
>   Yani çalkantı yalnız hız sütununda değil, konumda da gerçek. Yoğunluk
>   payı düştü, kalan FLIP payı. FLIP ölçümü (0.9 / 0.3 × 5 / 2.5 cm) sırada.
> **sleep_transfer 4 kol PASS, A/B kapısı sınırda FAIL:** strong kolunda
> uyku açık ile kapalı arasındaki enerji farkı 4.4e-5 J, tolerans 4.1e-5 J.
> Önceki koşu geçmişti. Bu build DEM'e dokunmadı (yalnız okuma probu); kapı
> küçük enerjide dalgalı.

**Derleme:** C++ **ve shader paketi**. Yoğunluk terimi şu dosyalarda değişti:
`sim_fluid_cg_residual_init`, `sim_sparse_pressure.glsl` (tüm
`sim_sparse_pressure_*`), `sim_fluid_divergence_porous.glsl` ve bütün CG
push-constant yapıları (alan adı değişti, düzen aynı). CUDA çekirdeği de
güncellendi (OptiX donmuş, ama derlenmesi gerekiyor).

**Yoğunluk düzeltmesi yeniden yazıldı (kural 5, ad değişti):**
`density_correction` (kazanç) → `density_relaxation_s` (saniye, varsayılan 0.5,
0 kapalı). Madde alanı `solver_density_correction` →
`solver_density_relaxation_s`; eski anahtar gerekçeyle reddedilir. Yeni formül:
hücrenin sayımı akışkan 6 komşusuyla ortalanıyor; ortalamanın bir standart
hatasının (Poisson) üstündeki fazla, `max(τ, dt)` içinde 1/s diverjans olarak
gideriliyor. CPU, Vulkan, seyrek, gözenekli ve CUDA yollarında formül aynı.
Eski projelerde anahtar yok sayılır, 0.5 s varsayılanı kullanılır.

Oturtma için de yalnız C++ değişti. Emekli/yeniden adlandırılan alanlar (kural 5, anlamı
değişti): `fluid_settle_speed_m_s` → `fluid_settle_displacement_m`
(m/adım, varsayılan 0.0002); okumada `settle_speed_m_s/first/residual_speed`
→ `settle_displacement_m`, `settle_first_p95_m`, `settle_residual_p95_m`, yeni
`settle_plateau`. `fluid.settle` parametresi `speed_m_s` → `displacement_m`.
Yeni okuma: `live_motion_samples/mean_m/p95_m/max_m`. Bunlar son karede 4096
taşıyıcılık örneğin id ile eşlenmiş KONUM değişimidir.

**Tank tabanının kökü (canlı prob, derleme gerekmedi):** dinlenik su tankında
ortalama parsel hızı ~0.16–0.21 m/s.
- Yoğunluk düzeltmesi 0: 0.21 → 0.16. Tam sayı hücre sayımı, hızlı bir sabit
  sıçramaya dönüşüyor (1 fazla parçacık / ppc 4 × 1.2 ≈ 0.3 m/s). Bu terim
  ızgaradan ve dt'den bağımsız, yani boyutsal olarak da yanlış.
- dt 1/120: 0.14; CPU: 0.18. Bu bir g·dt artefaktı değil, iki arka uçta da var.
- **FLIP 0: 0.004 m/s.** Kök FLIP gürültüsü: G2P, parçacıkla ızgara arasındaki
  alt-hücre farkının %90'ını her adımda geri yüklüyor ve fark sönmeden
  birikiyor. Su `solver_flip_blend = 0.90`.
- Karar bekliyor (öneri sırasıyla): (a) suyun ve sıvı ön ayarlarının FLIP
  oranını düşürmek (APIC zaten açısal momentumu koruyor; 0–0.3); (b) FLIP
  oranını çözünürlüğe ve dt'ye bağlı yapmak; (c) yoğunluk düzeltmesini
  boyutlu bir gevşeme süresine çevirmek (ΔU = h·(fazla/hedef)/τ) ve tam sayı
  sayım yerine yumuşak (trilinear) yoğunluk kullanmak.

Sıra:

1. `python scripts/test/rt_matter_smoke.py --only seed_settle`: 5 kapı.
   Beklenen: `settle_ran`, `grains_*` PASS; `liquid_at_rest` ★ büyük olasılıkla
   FAIL (FLIP tabanı, yukarıda). `summary.water_settle.settle_plateau` true
   olmalı. Oturtma tabana ulaşınca erken durur ve 240 adımın tamamını ödemez.
   **Bozuksa:** `settle_ran` FAIL ve `settle_steps == max_steps` ise düzlük
   algılanmıyor; `grains_control_moves` FAIL ise kum seed'i zaten dinlenik ya
   da ölçüm duyarsız.
2. `python scripts/test/rt_matter_smoke.py --only sleep_transfer`: 5/5 kalmalı.
3. `python scripts/test/rt_matter_smoke.py --only substance surface_water fire_steam contact_heat`:
   yoğunluk terimini kullanan sıvı testleri. Hepsi geçmeli; madde testi eski
   anahtarın reddini de kontrol eder. **Bozuksa:** sıvı hacim kaybediyor ya da
   üst üste biniyor demektir. τ çok uzun olabilir; 0.5 s'yi 0.2 ile dene.
4. Ölçüm (FLIP kararı için, kod yok): `seed_settle_live.json` su kollarındaki
   `motion_p95_m`. Önceki ölçüm 0.16–0.21 m/s hızdı; yoğunluk payı (~0.05 m/s)
   artık düşmüş olmalı, kalan FLIP payı. Ardından FLIP 0.9 / 0.3 ve 5 / 2.5 cm
   ölçümü yapılacak.

# 2026-10-19 (4. build): seed oturtma (DEM + sıvı) + sleep_transfer yatağı

> **SONUÇ (canlı, 2026-10-19):** sleep_transfer **5/5 PASS**: tek katlı yatak
> oturtuldu ve projectile doğdu. Bu koşuda iki test aracı düzeltildi:
> `rt_grain_material` silinen kaynağı unutuyor, `save()` Windows'un geçici
> kilidinde yeniden deniyor. contact_heat 7/7, surface_water 6/6 ve FG1 8/8
> geçti, ama süreler arttı: 253→435 s ve 224→368 s.
> **seed_settle FAIL 2/5.** Oturtma 240 adımda yakınsamadı. Her adım dinlenmeden
> başlıyor ve sonunda aynı hızı veriyor: su ilk adımda da son adımda da 0.112
> m/s, kum 0.085→0.143 m/s. Bu seed enerjisi değil, çözücünün tek adımda
> yerçekimine verdiği tepkinin tabanı. Durma ölçütü yanlış seçildi. Açık/kapalı
> kollar 30 karede ayırt edilemiyor: kum tepe hızı 0.154 / 0.148, su ortalaması
> 0.215 / 0.222 m/s. ★ Duran bir tankta ortalama 0.2 m/s parsel hızı kendi
> başına şüpheli. Ya saklanan hız projeksiyon öncesi bir değer (ölçüm sorunu),
> ya da gerçek bir çalkantı tabanı var. Ayırt etmek için konum değişimi
> ölçülmeli. Bir sonraki parti şunları içeriyor:
> (1) Durma ölçütü hızdan adım başına en büyük yer değiştirmeye geçecek.
> (2) Canlı okumaya kare başına yer değiştirme eklenecek.
> (3) Tank tabanının kökü bulunacak.
> Şimdilik oturtma her seed'de 240 adımın tamamını ödüyor (~0.7–1 s/seed).

**Derleme:** yalnız C++ (shader yok). Yeni dosya `Fluid/MatterSeedSettle.inl`
(vcxproj'a eklendi). Yeni IPC `fluid.settle`; `fluid.list_domains` artık
`settle_*` ve `live_max/mean_speed_m_s` raporlar. Panel: Seed bölümünde
"Settle Seed at Rest", iki ayar, "Settle Live Now" ve son oturtma satırı.

★ Varsayılan AÇIK: her seed (Fill Domain ve Seed Box) ilk adımdan önce
dinlenmeye oturtulur. İlk karede bir duraklama beklenir (en fazla 240
gevşeme adımı; büyük tane sahnelerinde saniyeler sürebilir).

Sıra:

1. `python scripts/test/rt_matter_smoke.py --only seed_settle`: 5 kapı.
   `docs/dev/seed_settle_live.json` özetinde üç sayı var: oturtulmuş kumun
   tepe hızı, kontrol kolunun tepe hızı ve suyun ortalama hızı. **Bozuksa:**
   `grains_control_moves` FAIL ise kapalı kol da duruyor, yani test duyarsız
   (eşik mi yanlış, seed zaten mi iyi, ölç). `settle_ran` FAIL ise istek
   state'e ulaşmadı.
   ★ Sessizce makul görünen: `settle_converged` false olur ama kapılar geçer.
   Bu, adım sınırına takıldı demektir; `settle_residual_speed_m_s`'e bak.
2. `python scripts/test/rt_matter_smoke.py --only sleep_transfer`: yeni tek
   katlı 9×3 yatak, doğumdan sonra `fluid.settle` çağrılıyor. Beklenen: dört
   kolun PASS satırları ve "PASS revision-24 transfer". **Bozuksa:** "bed covers
   the projectile birth site" yatağın hâlâ yayıldığını, "bed did not settle"
   oturtmanın uyku zamanlayıcısını başlatmadığını gösterir.
3. `python scripts/test/rt_matter_smoke.py --only contact_heat fire_steam surface_water`:
   seed kullanan eski testler; oturtma onların başlangıcını değiştirdi. Hepsi
   geçmeli. Süre artışı oturtmanın maliyetidir.

Bilinen sınırlar: oturtma sırasında hareketli (animasyonlu) collider'lar
örneklenmez, gaz şeridi adım atmaz ve gözenek suyu alışverişi kapatılmadı.

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

# 2026-10-19: Substance katalog/panel ek kontrolü

**Son hızlı canlı kontrol:** Kullanıcı build'iyle Substance katalog/miras IPC testi
PASS; geçici profiller temizlendi. Mevcut 9 audio asset / 14 source ve akustik
ayarları korundu. Üç Vulkan akustik raporu ready/clear, her birinde 6 specular ve
2–3 diffuse yol. Bu, aşağıdaki görsel UI/proje ve birleşik domain fizik kabulünün
yerine geçmez. Rapor: `substance_catalogue_acceptance_20261019.json`.

> **Durum:** CANLI — kaynak teslimi; build/canlı kabul bekliyor. Aşağıdaki birleşik
> domain/termal kontrol listesi korunmuştur. Bu Substance eklemesi için C++ build'i gerekir.

1. **Hızlı bağımsız servis kontrolü:** Studio açıkken dış terminalden
   `python scripts/test/rt_test_substance_catalogue_ipc.py`. PASS: altı temel profil,
   temsilciyle birebir alanlar, inherited family, shared scope, atomik hata/revert.
   Katalog alanları yoksa exe eskidir; Generic metal / Steel eşliği bozuksa temel
   profil üretimi veya başlangıç sırası hatalıdır.
2. **Ortak UI:** obje, collider, domain ve flow source aynı aile seçicisini kullanmalı.
   Profile details altında General / Mechanical / Thermal / Acoustics / Advanced;
   read-only built-in'de sekmeler çalışmalı. Dar panelde alanlar kendi içinde kaymalı.
   Collider'da iki Substance formu görünürse ortak obje paneli devralması bozulmuştur.
3. **Paylaşım ve miras:** aynı proje profilini iki objeye ata; referans özeti iki
   bağlantıyı içermeli. Derive edip yalnız birini değiştir; diğerinin değeri korunmalı.
   Eski isimli projeyi kaydet/aç, atamalar ve fizik alanları aynı kalmalı.
4. **Sessizce makul görünen hata:** aile etiketi doğruyken gerçek fizik değerlerinin
   değişmesi veya ikinci collider malzemesinin otorite kalması. UI'ya bakmakla yetinme;
   substance.get / substance.get_object readback'lerini karşılaştır.

Ayrıntı: [SUBSTANCE_CATALOGUE_PANEL.md](SUBSTANCE_CATALOGUE_PANEL.md).

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
