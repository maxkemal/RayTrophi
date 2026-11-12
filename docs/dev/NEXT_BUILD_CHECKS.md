# NEXT BUILD CHECKS

## Ölçek partisi 2 — 2026-11-11 (AJAN DERLEDİ 19:29, canlı doğrulandı; kullanıcı derlemesi gerekmiyor)

Kullanıcı izniyle derlemeyi ajan aldı (ölçek planı süresince). Bu liste, kullanıcının kendi
açılışında gözle bakabileceği ve otomatik testin göremediği şeyler içindir.

**Bu derlemede (hepsi canlı koşuldu):**
- Domain bütçesi: varsayılan 4096 MiB. 1M kar 1 GiB bütçede bile bekletilmiyor (tahmin
  2048 → 384 B/parçacık).
- S1 dilim A (`MatterGpuStep.inl`): yayınlamada kalıcı ara tampon + swap, paralel NaN taraması,
  tüm-MPM'de sidecar sütun swap'ı; hazırlıkta tag başına madde, ıslak tepki kapalıyken tek
  yükleme.
  - ★ İlk derlemede ÇÖKTÜ: `thread_local` dizi OpenMP döngüsünde ada göre kullanılmıştı; her
    işçi iş parçacığı kendi boş kopyasını görüyor (null + 4·i). Döngüden önce işaretçiye
    bağlanarak düzeltildi; kural kodda yorum olarak duruyor.
- S5 blok P2G + sıralama (yeni kayıtlar 21/64 ve 5/64; cihazda paylaşımlı float atomik açık).
- `sim_sparse_mac_mark` karo kutusu; `sim_surface_centres` kutu sırası.

**Ölçülen:**
- 1M kar 2.05 → 1.76 s/kare; 1M kar+su 1.38 → 1.33 s.
- smoke MPM 30/30, multirate 7/7, yüzey paritesi geçti.
- A/B fiziği (`RT_P2G_BLOCK_OFF`): ≤ 0.3 mm.

**21:08 derlemesi (ajan):** hücre kümeli P2G denendi; daha yavaş çıktığı için söküldü. Exe,
19:29'daki blok P2G haline döndü. Smoke elastic + snow_support PASS.

**22:12 derlemesi (ajan):**
- `RtApi.cpp` `enqueue` artık ana döngüyü uyandırıyor (arka plandaki pencerede IPC 500 ms
  uykuyu bekliyordu).
- P2G permütasyon denemesi söküldü (exe değişmedi, yalnız `.spv`).
- Doğrulama: smoke 30/30, multirate 7/7, yüzey 8/8.

**11:48 derlemesi (ajan, 2026-11-12): kar yanında fırlayan su ÇÖZÜLDÜ.**
- Kök: gözenek sürüklemesi. Su yüz kütleleri yalnız sıvı adımında yenileniyor, parseller ise
  her MPM alt adımında ilerliyor. Sıçrama parselleri kütlesi ~0 kalmış yüzlere giriyor; sürükleme
  o yüzü kar hızına çekiyor; FLIP/delta bunu parsele tam kütlesiyle aktarıyor. Momentum
  korunmuyor, 50 m/s tavanına pompalanıyordu.
- Düzeltme: `sim_matter_liquid_drag.glsl` yalnız su kütlesi ≥ %5 · ρ_sıvı·h³ olan yüzlerde.
  Push 32 → 48 B, iki kayıt satırı güncellendi.
- Eleme sırası: sürükleme kapalı / yalnız su yazımı kapalı → yok; gözenekli, tepki, delta kapalı
  → var. Denenip sökülenler: φ ≥ %5 kapısı, MPM hız kelepçesi, sıvı adımında sürüklemeyi
  atlamak, MPM malzeme (FLIP taban) hızı.
- bal: en yüksek hız 9.1 m/s (önce 44–50), sıvı alt adımı 17–36 (önce 180'e kadar), sıçrama
  karesi 0.55–0.72 s (önce 1.6–2.4 s).
- Testler: smoke 30/30, multirate PASS, sıcak su + kar PASS.
  `mpm_porous` floats/near_surface ve `liquid_drag` converged FAIL, ama düzeltmesiz eski
  shader'la da aynı FAIL; önceden var olan açık konular.

**10:36 derlemesi (ajan, 2026-11-12): hareketli kollider dinlenik karı uyutmuyordu.**
- `MatterMpmSleep.cpp`: domain'in HERHANGİ bir yerinde hareketli kollider (solid_vel ≠ 0)
  dinlenik karı uyanık tutuyordu; voksel imzası her karede değiştiği için dinlenme sayacı da
  her karede sıfırlanıyordu.
- Artık yalnız parsel kutusunun yakını (3 hücre + 2 × kollider kare kayması) ve oraya varacak
  ayak izi sayılıyor.
- dem1: temastan önce 150 → **55 ms** (uykuda), kollider ~0.65 m kala uyanıyor, temas
  davranışı aynı.
- Smoke 30/30, `collider_phi` 12/12.
- ★ Sessiz başarısızlık: hızlı bir kollider uyuyan karın içine bir kare geç girerse kar o kare
  tepkisiz kalır. Bakılacak: hızlı kollider (> 1 hücre/kare) karın içine dalarken ilk temas
  karesi.

**00:13 derlemesi (ajan, 2026-11-12): hareketli kollider maliyeti.**
- `computeSolidFaceWeights` tek iş parçacığındaydı. Hareketli 2.5 m mesh-SDF küre her karede
  ~3M SDF örneği çekiyordu: 205 ms/kare, karla temastan bağımsız.
- Üç yüz döngüsü OpenMP ile paralelleştirildi (her yüz tek yazıcı, sonuç aynı): dem1'de
  **205 → 24 ms**.
- ★ Smoke testleri bu derlemede KOŞULMADI (açık sahne vardı). Sırada: `mpm_collider` ve
  `collider_phi` testleri.
- Not: dem1'deki tek seferlik seed projeyle kaydedilmiyor; proje yeniden açılınca domain boş.
- Sonraki kaldıraç (yazılmadı): yüzeyden uzak yüzde 16 yerine 1 örnek (SDF mesafesiyle) ya da
  GPU'ya taşıma.

**23:57 derlemesi (ajan): aşırı soğumanın iki kalan kökü kapandı + sıçrama maliyetinin kökü bulundu.**
- Bal sahnesinde geçici sondayla bulundu (sökülen `THERMAL-PROBE`):
  - (a) Kuru havada buharlaşmanın çiğ noktası ~0 K: buna **yaş termometre tabanı** eklendi
    (psikrometrik; kuru 293 K → 277.6 K, %50 → 286.2 K). CPU ve GPU.
  - (b) `conductThermalLiquid` hücre ΔE'sini her parsele TEKDÜZE ekliyordu: karışık hücrede
    kar parseli komşularının altına iniyordu. Artık yalnız yeni ortalamanın öbür yanındaki
    parsellerden, c_p|T − T_yeni| ağırlığıyla dağıtılıyor (enerji birebir, sınır aşılmıyor).
    Hücre içi düzeltme de aynı kuralla.
  - Sonuç: bal sahnesi 60 kare boyunca T_min = 263.1 K (karın kendisi). Donma hâlâ var
    (kare 60'ta 865 parsel). Su −10 °C karla karışıp donuyor; fiziksel olabilir, erime hızıyla
    birlikte bakılacak.
- **Sıçrama maliyeti (kare 22–48, 1–2.4 s):**
  - Birkaç SU parseli tabanda, kar yığınının olduğu köşede 20–50 m/s'ye fırlıyor
    (`SPEED-PROBE`, hâlâ kodda, yalnız > 10 m/s'de log yazar).
  - Sıvı CFL'si 17 → 180 adım, viskozite PCG kare GPU'sunun ~%65'i.
  - Termal kapalı da aynı; collider yapışması 0 da aynı; **kar kaynağı kapalıyken YOK**
    (vmax 5 m/s, 16 adım) → kök su–kar bağında. Sürükleme örtük, aday değil.
  - Adaylar: gözenekli projeksiyon / basınç tepkisi / aralık sonu saf-delta G2P. AÇIK.

**23:10 derlemesi (ajan): termal duraklama + Celsius.**
- Kök: sıcak su buharlaşmada tek adımda ~30 K'ye soğuyup donuyordu; donmuş parsel karma GPU
  adımını durduruyordu.
- Düzeltmeler:
  - Buharlaşma geriye Euler (CPU `GasSurfaceExchangeCpu.cpp` + GPU
    `sim_inventory_surface_exchange.comp`).
  - Donmuş parseller şerit dışında sabit (`kMatterGpuHeldModel`).
- Test `rt_test_matter_thermal_hot_water_snow_ipc.py`: 353 K su + kar, termal açık, en düşük
  T = 263 K, bekletilen kare yok. Smoke 30/30, multirate 7/7.
- Panel: mutlak Kelvin alanları "353.0 K (79.9 C)" gösteriyor (`UI/TemperatureLabel.h`).
- Bakılacaklar:
  1. Bal sahnesinde termal açık + özel su sıcaklığı: sim durmamalı, `thermal_min_kelvin`
     karın sıcaklığının altına inmemeli.
  2. Panelde sıcaklık alanları Celsius'u parantezde göstermeli (termal paneli, seed kaynağı,
     collider MSF, gaz, madde düzenleyici).
  - ★ Donmuş parsel yolu (gerçek donma) bu testte TETİKLENMEDİ; kar yanında su gerçekten
    donarsa sabit kalmalı ve sim devam etmeli.

**22:39 derlemesi (ajan): kar ölçülen sertleşmeyle boyutlanıyor.**
- Kar, tavan (×5) yerine kare başında ölçülen en sert parsel × 1.5 ile boyutlanıyor. Kare
  aşarsa tavanla yeniden koşuluyor.
- 1M kar 1.76 → 1.22 s/kare.
- Bakılacak: kar sahnesinde çarpma ve sıkışma davranışı öncekiyle aynı mı.
  ★ Sessiz başarısızlık: çarpma karesinde kar hafif "yumuşak" görünür (fazla sıkışma).
  Ölçüt: düşüş testinde J_p en küçük ≈ 0.54, ortalama ≈ 0.92 (tavanla koşuyla aynı).

0. Uygulamayı kendin aç, pencereyi arka plana al ve bir IPC testi koş: artık yavaşlamamalı.
   Bozuksa (ping ~500 ms): uyandırma olayı ulaşmıyor.

**Kullanıcının kendi sahnesinde bakması gerekenler (sırayla):**
1. Bal/kar sahnesini aç ve oynat: kar ve su davranışı öncekiyle aynı olmalı.
   Bozuksa: blok P2G karo dışı katkıyı ya da FLIP taban çizgisini yanlış topluyor. Uygulamayı
   `RT_P2G_BLOCK_OFF=1` ortam değişkeniyle aç; davranış düzeliyorsa kök blok yolu.
2. Sahnede kar, kum ya da elastik var ve "Rendered" (Vulkan RT) modunda yüzey düzgün
   görünüyor: yüzey merkez çekirdeği değişti, parite testi geçti.
   ★ Sessiz başarısızlık: yüzey biraz pürüzlü ama "makul" görünür; parite sayıları bunu yakalar,
   göz yakalamaz.
3. Eski bir projeyi aç: domain bütçesi kayıtlı değerini korumalı.

Ayrıntı: `docs/dev/OLCEK_MILYON_PARCACIK.md` §3.3–3.4.
