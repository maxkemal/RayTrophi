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
