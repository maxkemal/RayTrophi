# Latest checkpoint — 2026-10-14

User confirmed successful Studio compilation after the Sound Designer MSVC fixes.
New Designer listening/UI/external IPC results are pending. The cloud V2d shader/
visual checklist below keeps its separate acceptance gate; Studio compilation does
not establish that a shader rebuild or visual/physics test passed.

# 2026-10-14 (gece, 4): bulut hacimleri V2e — gölge yüzü, etek levhası (yalnızca shader)

> **Sonuç:** derlendi, CANLI — kullanıcı "çok daha doğal"; IPC karesinde
> levha ve kutu kenarı çizgileri yok, gölge yüzü gri + gümüş kenar.

Bulgular (kullanıcının ayarladığı kare, kamera güneşe bakıyor, IPC ekran
görüntüsü):
- gölge yüzü düz ve soluk, gök kadar açık → "saydam";
- kule tabanında kutu genişliğinde, düz tepeli KAHVERENGİ bir levha; kutu
  kenarlarında dik kesiliyor.

Düzeltmeler:
1. Gölge yüzünün ortam ışığı yarı yarıya yüzey NORMALİ yönündeki gökten
   (`skyAmb`). Güneşe sırtını dönen duvar artık güneş tarafının parlak
   ufkunu görmüyor (`cloud_rt.glsl`).
2. Etek: kare düşüş, kırık örtü (0.3..1), tam kalınlık yalnızca duvarda,
   kutu kenarından 2.5 km içinde sönüyor (`cloudVolumeLastEdge`,
   `cloud_common.glsl`).
3. Kutu içinde ama hacmin dışındaki katman örneğinin dikey AO tap'leri
   artık yalnızca katmanın kendi yoğunluğunu okuyor. Önceden 1.5 km
   yukarıdaki tap kuleye düşüyordu (σ·3 km ≈ 800): gök ışığı sönüyor,
   güverte zemin kahvesine boyanıyordu. Kulenin gökten kapattığı pay
   artık uzaklıkla azalan bir açıklık terimi.

## Kontrol listesi (sırayla)
1. **Derleme** (yalnızca shader).
2. **Levha (göz, aynı kare).** Kahverengi düz bant ve kutu kenarındaki dik
   çizgiler gitmeli; güverte kuleye doğru kırık kümülüslerle kabarmalı.
   - Bozuksa: bant sürüyor ama artık gri → AO tamam, şekil hâlâ levha:
     etekte `topFrac` satırını geçici kapat. Bant hâlâ kahverengi → kaynak
     etek değil, ışık; `volSdf < 1e5 && !column` dalı tutmuyor demektir.
3. **Gölge yüzü (göz).** Güneşe karşı bakan kulelerin gölge yüzü gökten
   belirgin koyu, gri-mavi olmalı; kenarlar parlak (gümüş kenar).
   - ★ Sinsi olan: uzak kuleler (≳30 km) yine soluk kalabilir. Bu hava
     perspektifi (`mie_density 1200`, `mie_anisotropy 0.99` güneşe doğru
     beyaz pus); hata değil, sahne ayarı. Yakın kuleye bak.
4. **Probe:** `.\scripts\ipc\Probe-CloudVolumes.ps1` → PASS (şekil
   değişmedi; yalnız ışık ve katman).
5. **Maliyet:** değişiklik yok sayılır (+1 LUT okuması).

---

# 2026-10-14 (gece, 3): bulut hacimleri V2d — gölge yüzü + çoklu hücre şekli

> **Sonuç (2026-10-14):** derlendi; kullanıcı "şekiller çok daha iyi". Probe
> PASS (madde 4; probe'un geçiş sayacı düzeltildi). Açık kalan: madde 2'nin
> gölge yüzü onayı ve madde 5 maliyet.

Önceki derlemede (V2c, kullanıcı, ışık yönü değiştirildi):
- saydamlık GÖLGE tarafında ve güneşle birlikte yer değiştiriyor → SDF
  atlaması değil, aydınlatma;
- şekil hâlâ tek düze bir kule: çoklu kümülonimbüs, parçalı çevre ve
  farklı karnabahar kabarmaları yok.

## Bu derlemede (yalnızca shader)
1. **Gölge yüzü** (`cloud_rt.glsl`, `cloud_common.glsl`).
   - Kök: ortam ışığının ufuk payı BAKIŞ azimutundan okunuyordu. Yoğun
     bulut birkaç metrede opaklaştığı için gölge yüzü yalnızca bu terimle
     çiziliyor ve tam arkasındaki gök rengini taşıyordu → "saydam".
   - Şimdi: zenit + 4 sabit azimut (bakıştan bağımsız). Hacim yüzeyinde
     SDF normali gök/zemin payını, iki sonda girinti koyuluğunu belirliyor;
     girintiler mavi değil gri (bulutlar arası yansıma).
2. **Çoklu hücre şekli** (`cloud_volume_gen.comp`).
   - Yan hat: ana hücrenin inflow arka kanadında 4 genç kule, dışa doğru
     alçalan merdiven, her biri kendi yavaş nabzıyla.
   - Besleyici kümülüs halkası: fırtınaya yakın daha sık ve uzun.
   - ~650 m Worley kabarmaları tüm yüzeyde; alt kabarcıklar artık duvara
     da iniyor, boyları geniş dağılıyor; shader kabarcık oyuğu 80 → 120 m.

---

## Kontrol listesi (sırayla)

1. **Derleme** (yalnızca shader).

2. **Gölge yüzü (göz).** Aynı kamera, güneş kuleyi arkadan/yandan
   aydınlatsın.
   - Görmen gereken: gölge yüzü arkadaki gökten belirgin şekilde ayrı,
     gri-mavi ve kabartmalı (tepe yüzleri açık, girintiler koyu, alt
     yüzler zemin renginde). Güneşi çevirince "saydamlık" gölgeyle
     birlikte gezmiyor.
   - Bozuksa:
     - hâlâ gök rengi ve düz: normal/açıklık dalı çalışmıyor olabilir
       (`T > 0.15 && volSdf > -400` koşulu). Geçici olarak `openness = 0.3`
       yazıp fark görülüyor mu bak;
     - gölge yüzü simsiyah: açıklık fazla kısıyor; `0.45` katsayısını
       `0.3`'e indir.
   - ★ Sinsi olan: ortam artık bakıştan bağımsız, bu yüzden **katman
     bulutlarının** gölgeli altları da biraz değişir (güneş tarafındaki
     parlak ufuk da ortalamada). Hafif ısınma/grileşme beklenen; mavi→turuncu
     bir kayma beklenmeyen.

3. **Şekil (göz).** Uzak kare.
   - Görmen gereken: ana kule + bir yanında alçalan 2–4 kule (merdiven),
     tabanda ve çevrede dağınık küçük kümülüsler (fırtınaya yakın olanlar
     daha büyük), yüzeyde farklı boyda kabarmalar, duvarlarda da.
   - Bozuksa:
     - yan kuleler kutu kenarında kesik: kutu 12.8 km; hücre yarıçapı
       büyükse yan hat sınırlanıyor (`halfBox - 1.9 rt`) — kesik değil
       eksik görünmeli;
     - besleyiciler kutu kenarında düz bir çizgide bitiyor: beklenen sınır
       (V3'te kutu dışı kabuk/katman devralır);
     - yüzey "köpük" gibi çok gürültülü: `kBillowAmpM` 170 → 110.

4. **Probe.** Zamanı durdur:
   ```
   .\scripts\ipc\Probe-CloudVolumes.ps1
   ```
   - Görmen gereken: PASS. Adım 3a profil farkı büyüyebilir; 3b düz taban
     korunmalı (taban kabarmalardan SONRA kesiliyor).
   - Bozuksa 3b: besleyici bulut tabanları ölçüm hattına denk geldi;
     ölçüm noktalarını yaz, bakarım.

5. **Maliyet.**
   - Render: hacim yüzeyindeki her görünür örnek +6 atlas okuması (yalnız
     birincil ışın). RT kare süresini V2c ile karşılaştır.
   - Üretim: Worley (27 hücre) + 9 besleyici + 4 yan kule voksel başına;
     yuvalar ~4 sn'de bir yeniden boyanıyor. `cloud_stats.volumes
     .repaints_last_update` ile kare sıçraması (takılma) var mı bak.

## Bilinen
- Taban henüz düz; sarkan loblar, scud ve içe akış V3'te. Besleyici halka
  bunun durağan hâli.
- Örs kutu kenarında kabukla dikişli olabilir.
- Bu sahnede yağış görünmüyor: `ground_fraction = 0` (sahne ayarı).


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
