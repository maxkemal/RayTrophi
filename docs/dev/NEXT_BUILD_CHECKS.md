# NEXT BUILD CHECKS

## KÖK BULUNDU: TLAS tanıtıcısı yeniden kullanılıyor + kayıptan sonra kilitlenme (2026-11-17 11:00, Claude)

**Durum:** CANLI KABUL (kullanıcı, 2026-11-17): derlendi, çökme yok, gölgeler doğru yerde, RT gölge + yansıma açıkken FPS maliyeti görülmedi. Madde 3–4 (Player uzun oyun, kayıpta kilitlenmeme) açık kalan gözlem.
**Ölçüm (10:48, defter):** hata `read_invalid 0x6770000`, offset 0 → **DESTROYED** 12544 B tampon,
oluşturan `rebuildRayFusionTLAS → createTLAS` (TLAS deposu). Bir önceki TLAS da AYNI adresteydi
(#2501 ve #2505 = 0x6770000). Sürücü yeni TLAS'a aynı handle değerini ve aynı VA'yı veriyor;
"descriptor'ı yeniden yazayım mı" kararını handle hash'iyle veren tüketiciler (RT gölge, yansıma)
silinmiş TLAS'ı izlemeye devam etti.
**Düzeltme:** `include/Backend/VulkanTlasGeneration.h` (yeni) — TLAS NESNESİ her oluşturulup
yok edildiğinde artan nesil. `MaterialPreviewRtShadow.cpp`, `MaterialPreviewReflection.cpp` ve RT
descriptor drenaj hash'i (`VulkanBackend.cpp`) bunu imzaya katar.
**Kilitlenme (11:00, cdb):** kayıptan sonra ana iş parçacığı `RasterViewportFrameRing::destroy` →
`vkWaitForFences(UINT64_MAX)` içinde NVIDIA sürücüsünde SONSUZA dek dönüyordu (CPU'ya düşemedi,
kapanamadı). `VulkanFailureReporting.h::waitFenceBounded` — kayıp cihazda beklemez, 100 ms
parçalarla en çok 10 s; 12 süresiz bekleme buna çevrildi (FrameRing 3, VulkanBackend 5, Clouds 2,
SimulationComputeVulkan 3).

1. Derle (Studio + Player). Demo sahnesini RayFusion'da aç, objeleri taşı, Solid↔RayFusion geç.
   Görmen gereken: çökme YOK. Bozuksa: SceneLog `[DeviceFault]` + `owner:` satırlarını getir —
   aynı `rebuildRayFusionTLAS` sahibi çıkarsa başka bir handle-hash tüketicisi kalmış demektir.
2. RT gölgeler ve RayFusion yansımaları obje taşındıktan sonra DOĞRU yerde mi (eski konumda gölge
   kalmamalı). Bu düzeltme o descriptor'ları daha sık yeniden yazıyor.
3. Player: balon demosu uzun oyna, patlat, odak değiştir. Çökme yok mu.
4. Kayıp olursa (başka sebeple): uygulama artık KİLİTLENMEMELİ — CPU'ya düşmeli ya da kapanmalı.
   Günlükte `fence unsignalled after 10 s` görürsen o yerin adı yazılı; bana getir.
5. ★ Sinsi olan: kare hızı. waitFenceBounded normal karede tek çağrı; FPS düştüyse söyle.

## Hata adresinin SAHİBİ — GPU tampon adres defteri (2026-11-17 10:40, Claude)

**Durum:** YENİ `Backend/VulkanBufferAddressLedger.cpp/.h` (Studio vcxproj + filters'a eklendi,
Player vcxproj yeniden üretildi, `--check` PASS); `VulkanBackend.cpp` createBuffer/destroyBuffer +
BLAS toplu silme yolu deftere yazar. /Zs PASS. Shader yok. Struct düzeni değişmedi.
**İLK ÖLÇÜM (10:37, Studio, demo sahnesi yüklenirken):** `[DeviceFault]` → SAYFA HATASI, zaman aşımı
DEĞİL: `read_invalid 0x2fced0000` (4 KB sayfa başı) + `ip_fault 0x20339a480`. Sıra: havuzlar raster'a
eklendi (gen=7, 38 mesh, 135201 yuva) → RayFusion AS yeniden (38 BLAS, 5 flat) → sonraki tek seferlik
gönderimde kayıp. Probe trace + bounce tabloları statik olarak tazeleniyor görünüyor (elenmedi, kanıt yok).

1. Derle, aynı şeyi yap (demo sahnesi RayFusion'da açılış / obje taşıma). Kayıpta SceneLog'da
   `address[1] owner:` satırları:
   - `DESTROYED buffer ... created by: <fonksiyon>` → serbest bırakılmış tamponu kullanan var; o
     fonksiyonun tamponu hangi tablo/descriptor'da yaşıyor, oradan düzeltilir. KÖK.
   - `LIVE buffer ... fault_offset=<büyük>` → canlı tamponun dışına taşan indeks.
   - `no live or recently destroyed buffer` → görüntü (image) ya da defter dışı ayırma.
2. Normal çalışmada fark olmamalı (her tampon oluşturmada birkaç µs yığın kaydı). Açılış/FPS aynı mı.
3. `VulkanDeviceFault_0.bin` NVIDIA dökümü x64/Release'de; sakla.

## Cihaz kaybı artık KÖKÜNÜ yazar — VK_EXT_device_fault (2026-11-17, Claude)

**Durum:** `Backend/VulkanBackend.cpp` + `include/Backend/VulkanFailureReporting.h` (4 .cpp yeniden
derlenir, struct düzeni DEĞİŞMEDİ), /Zs PASS (Player). Shader yok.
Neden: WAR düzeltmesinden sonra Player yine düştü (açılıştan ~7 s, patlama yok, Studio KAPALI) ve
Studio RayFusion'da obje taşırken düştü. Windows günlüğü her ikisinde nvlddmkm 153 çifti verir — bu
NVIDIA'da hem GPU sayfa hatasını hem zaman aşımını kapsar, yani "TDR" tek başına ayırt etmez. İki
günlükte de kayıp `[RayFusion] scene AS built` satırından hemen sonra, `endSingleTimeCommands`'da
GÖZLENİYOR (gözlenen yer, sebep değil).

1. Studio ya da Player'da bir sonraki cihaz kaybında SceneLog'da `[DeviceFault]` satırlarını ara.
   - `address[...] read_invalid/write_invalid 0x...` → GPU geçersiz adres okudu/yazdı (bayat tampon,
     yanlış instance tablosu, yok edilmiş AS). Adres + açıklamayı bana ver.
   - adres yok, açıklama/vendor "timeout/hang" benzeri → gerçek zaman aşımı (sonsuz döngü / çok uzun iş).
   - `vendor binary written to VulkanDeviceFault_N.bin` → NVIDIA çökme dökümü (x64/Release'de), sakla.
   - `[DeviceFault] no device was created with VK_EXT_device_fault` → uzantı açılmadı; bozuk demek.
2. ★ 10:30'da TAM OLARAK bu oldu (Studio, RayFusion'da obje taşıma): kaybı raster kare edinme yolu
   gördü, `[DeviceFault]` yazılmadı. Düzeltildi: `VulkanViewportBackend.cpp` (edinme + gönderim) ve
   `VulkanBackend.cpp` (submitTraceTonemapAsync) da sorguyu çağırır; /Zs PASS. Hâlâ satır yoksa
   dördüncü bir algılama yolu var — söyle.
   Gözlem (iki Studio çökmesi): kayıp, balon demosunun 5 havuz mesh'i (damla/konfeti, 135 bin örnek
   yuvası, BLAS'ları "5 flat") raster'a eklenip RayFusion AS yeniden kurulduktan HEMEN sonra.
3. Normal açılışta etkisi olmamalı: yalnız kayıpta sorgu yapar. Player/Studio açılışı aynı mı.
4. TDR sonrası uygulama kapanmıyorsa ÖLDÜRME — ajan `cdb -pv` ile yığını alacak.

## Aralıklı Player device lost — sim compute WAR yarışı düzeltildi (2026-11-17, Claude)

**Durum:** `Device/SimulationComputeVulkan.cpp` (artımlı, Studio+Player), /Zs PASS. Shader yok.
Senkronizasyon doğrulaması (Player, 3 dk) 3900× "kopya, hâlâ okunan tampona" (dispatch → kayıtlı
yükleme kopyası, arada yalnız compute→compute bariyeri) buldu; düzeltildi. Kalan kare içi
tehlikeler ve TDR hipotezi `BUG_VIEWPORT_DEVICE_LOST_ON_PROJECT_OPEN.md`'de.

1. Studio + Player derle. Player'ı NORMAL başlat (doğrulamasız), uzun oyna, odak değiştir, projeyi
   sürükle-bırakla yeniden aç. Çökme sıklığı düştü mü (tek gözlem kanıt değil; "her zaman olmuyor").
2. Hâlâ çöküyorsa: Studio RayFusion'da AÇIKKEN mi? O an Studio `viewport.device_recovery_status`
   sayacı artıyorsa TDR (iki süreç aynı GPU) — farklı düzeltme sınıfı.
3. Doğrulama koşusunu tekrarlamak için: `$env:RAYTROPHI_VK_VALIDATION="1";
   $env:VK_LAYER_ENABLES="VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT"` ile Player,
   SceneLog'da `vkCmdCopyBuffer` WAR satırı 0 olmalı.

## Player çökmesi — IPC ile YENİDEN ÜRETİLEMEDİ (2026-11-17 02:40–02:55, Claude)

Yeni exe (02:36, büzme + oda içi alan) ile kullanıcıda duvar dibindeki patlamada yine device lost; bu sefer log'da patlamada `topology_change` YOK (yalnız açılışta) → TLAS gizle/göster yolu elendi. Ajan Player'ı açıp ~60 patlama denedi, HİÇ çökme yok: arka/sol duvar dibi (40–60 cm, değen), sürpriz + konfeti + üçlü patlama (5252 damla, 17k kaynak düştü), yakalamalı senkron çizim, continuous asenkron çizim, 6 tur "arka planda boşta → çizimi aç + patlat". Kalan farklar: GERÇEK pencere odak olayı ve fare tıklaması yolu (SDL olay döngüsü). Kullanıcıdan: odaktayken tıklama mı, odağı geri getiren tıklama mı çöküyor; çöktüğü an log.

## Vulkan cihaz kaybı (odak dönünce) + balon alanı oda İÇİ (2026-11-17 gece, Claude)

**Durum:** `BalloonPaintDemo.cpp` değişti (artımlı), /Zs PASS. Shader yok.

**Cihaz kaybı — Player SceneLog:** art arda `createTLAS/topology_change` (biri 193 ms), sonra
`Device lost observed at endSingleTimeCommands/vkQueueSubmit`. Mekanizma hafızadaki dersin
aynısı: örnek SAYISI değişince `VulkanDevice::createTLAS` yok et/yeniden yarat dalına girer ve
uçuştaki RT işi (RT gölge, RayFusion) için BEKLEMEZ. Balon patlayınca mesh GİZLENİYORDU
(`setObjectVisible(false)`) → her patlama/doğuşta sayı değişiyordu. Düzeltme: patlayan balon
yerinde 1e-4'e BÜZÜLÜR, gizlenmez (yalnız hazırlıktan sonraki ilk gösterim sayıyı değiştirir).
Status `visual_visible` = mesh görünür VE balon şişik.
**AÇIK (motor geneli, dokunulmadı):** `createTLAS` yok etme dalı drenajsız. Sayıyı değiştiren
başka her şey (parçacık RT örnekleri, scatter, konfeti?) aynı riski taşır. Doğru düzeltme:
YALNIZ yok etme dalında koşullu drenaj (bkz. hafıza: koşulsuz kare drenajı perf'i öldürür).

**Balon alanı:** otomatik alan dış sınırı (tavan levhası z ±6,57) aldığı için balonlar yan
duvardan çıkıyordu (status z ±5,67). Artık oda İÇİ: levha gibi yüzeyler (bir eksende <0,3 m,
diğerlerinde >1 m) duvar sayılır, merkeze en yakın duvar sınırdır; açık taraf dikey duvarların
bittiği yer. Workshop'ta hesap: oda x −4..4, y 0,03..4,57, z −4..4 → alan ±3,1, y 0,83..3,57.

1. Player: uzun süre oyna, başka pencereye geç ve geri dön — cihaz kaybı yok. SceneLog'da
   patlamalarda `createTLAS/topology_change` ARTIK yok (yalnız açılışta). Hâlâ varsa ve konfeti/
   damlayla aynı anda geliyorsa: parçacık örnekleri sayıyı değiştiriyor (motor geneli açık).
2. `paint.wet_demo` status `spawn_min/max` ≈ [−3,1, 0,83, −3,1] / [3,1, 3,57, 3,1]; balonlar
   duvar içinde kalır.
3. Patlayan balon görünmez (büzülmüş), tıklanamaz (pick yalnız şişikleri tarar).

## Süzülen balonlar + odadan doğum alanı (2026-11-17, Claude) — aynı derlemeye girer

**Durum:** KAYNAK HAZIR, derlenmedi, /Zs PASS (`BalloonPaintDemo.cpp` iki hedef,
`GameDesignUI.cpp`). Header yok. Bu derlemede ayrıca: CPU `resolveOBBCollision` düzeltmesi
ve yaygın struct değişikliği (en üstteki uyarıya bak) — tam Rebuild önerilir.

Ne: balon doğum alanı artık boya yüzeylerinin sınırlarından (duvardan 0,9 m içeri, yerden
0,8 m, tavandan 1 m aşağı) — yükseltilmiş tavanla balonlar ~3,5 m'ye çıkar; ya da panelden
elle. Mesh balonlar süzülür: `float_amplitude` salınım (vars. 8 cm), `drift_speed` dolaşma
(vars. 0,12 m/s, büyük balon daha yavaş), duvar/tavan ve birbirinden kaçınma; dönüşüm 30 Hz.

1. Player'da balonlar yavaşça salınıp dolaşıyor, üst üste binmiyor, duvardan çıkmıyor; bazıları
   eskisinden yüksekte. Hiç kıpırdamıyorsa `drift_speed`/`float_amplitude` 0 ya da parçacık modu.
2. Tıklama hareket halindeki balonu yakalar (pick anlık konumu okur).
3. `python scripts/test/rt_balloon_motion_ipc.py` (Player) → 4/4; çıktıdaki `motion_ms_max`
   ÖLÇÜM: her 30 Hz'de 12 balonun dönüşümü — gizmo sürükleme yolu (obje dönüşümü + arka uçlara örnek dönüşümü, CPU köşe pişirmesi YOK, log YOK); CPU köşeleri yeniden doğuşta bir kez pişer. Demo artık TransformCommand kullanmıyor ("Redo: Transform" logu kalktı). ★ Sinsi olan: FPS
   hareket açılınca düşüyorsa sebep bu sayı — sessizce "oyun ağırlaştı" görünür.
4. Panel → Floating: Bob/Drift kaydırıcıları, "Balloon area from the room" kapalıyken alan elle.

## ★ OBB collider dokunmadığı damlayı öldürüyordu (2026-11-17, Claude) — ASIL KÖK

**Durum:** `sim_particle_collide.spv` DERLENDİ, iki yerde (bağlama/push AYNI; uygulamayı yeniden
başlatmak yeter). CPU referansı `ParticleSimulation.cpp` `resolveOBBCollision` DERLENMEDİ
(/Zs PASS). Kayıt yeri yeniden pişirme partisi düzeltildikten SONRA da sorun sürdü (kullanıcı).

**Ölçüm (Studio canlı, boya tüketmesi geçici kapalı, bir patlama):** 1800 olayın **1719'u
collider 9 = tavan `Cube_2` obb**, konumları y 0,9–2,1 (balonun etrafı), tavan 4,57–4,78'de.
Tavan aabb yapılınca: 1248 olay zeminde (`Cube_1`), duvarlar 258+, tavan 5 → kök obb yolu.
**Kök:** obb dalı temas OLMASA da konumu yerel çerçeveye çevirip geri yazıyordu; ötelenmiş
çerçevede (tavan y=4,67) gidiş-dönüş ~1e-7 fark bırakır, çağıran "değişti = temas" sayar,
Die yanıtı damlayı İLK adımda öldürür. Zemin `Cube_1` merkezde (öteleme 0) olduğu için hata
görünmüyordu. Belirtiler: "yarım küre patlama", duvar altı/zemin boyanmıyor.
**Düzeltme:** yalnız gerçek temasta geri yaz (şader + CPU). SDF yolları zaten erken dönüyor.

1. Player'ı YENİDEN BAŞLAT (yeni .spv): patlama küresel, zemin ve duvar altı boyanır.
   Bozuksa: `.spv` zaman damgası 01:28 değil → eski şader.
2. `python scripts/test/rt_collider_event_locality_ipc.py` (Studio) → 3/3: her olay kendi
   collider kutusuna ≤25 cm; zemine düşen olay ≥%10.
3. Tavanı (`Cube_2`) obb bırakabilirsin; aabb gerekmez.
★ Sinsi olan: bu hata merkezdeki/ötelenmemiş kutularda HİÇ görünmez; obb'li her yeni objede
(taşınmış bir oyuncak) damla sessizce havada ölür.

## ⚠ YAYGIN STRUCT DÜZENİ DEĞİŞTİ — `WetPaintStats` (+`geometry_rebakes`), `WetPaintGpu` (+üye), `rtapi::WetPaintStatusInfo` (+alan)

Artımlı derleme eski `.obj` bırakırsa erişim ihlali kod hatası gibi görünür: bu dosyaları
içeren TU'lar (9) yeniden derlenmeli; şüphede Player+Studio için Rebuild.

## Boya yüzeyi objeyi izler: taşı/ölçekle → yeniden pişir (2026-11-17, Claude)

**Kök (kullanıcı sahnesinde canlı ölçüldü):** yüzeyin texel dünya konumları `wet_add_surface`
anında BİR KEZ pişiriliyordu. Duvarlar 3,4 → 4,68 m uzatıldıktan sonra: duvara y=0,5/1,5/3,0'da
konan test lekesi tuttu (~862 texel), **y=4,3'te 0**; zemin `Cube_1`'e y=0,03/1/−1'de 0.
Görülen üç belirti tek kök: (a) duvarın üstü/zemin boyanmıyor, (b) patlama izi "yukarı yarım
küre" — Y'ye çarpan damla eski haritada Y'yi boyar, o texel artık 1,38×Y'de görünür, iz yukarı
kayar ve uzar, (c) akan boya texel uzayında olduğu için yine aşağı iner. Collider'lar doğru
çözülüyordu (`collider.list` resolved kutular uyuyor).
**Düzeltme:** geometri parmak izi (köşe sayısı, konum toplamı, sınırlar) saniyede 2 kez;
değişen yüzey yeniden pişer, atlas yeniden kurulur (o yüzeyin ıslak durumu sıfırlanır).
`paint.wet_status` → `geometry_rebakes`. **Derlemeden önceki geçici çözüm:** projeyi kaydet
ve yeniden aç (yükleme yüzeyleri güncel geometriyle yeniden pişirir).

1. Derleme sonrası workshop: duvarın en üstüne ve zemine balon boyası düşer; patlama izi
   balonun etrafında küresel. `geometry_rebakes` açılışta 0 olabilir (yükleme zaten taze pişirir).
2. `python scripts/test/rt_wet_paint_rebake_ipc.py` (Studio) → 3/3: duvarı 1,5 m kaldırır,
   ESKİ tepenin üstüne leke tutar, geri koyar.
3. Gizmo ile duvarı sürüklerken takılma: her 0,5 sn'de bir pişirme (512² yüzey ~25 ms
   tahmini, ölçülmedi). Rahatsız ederse sürükleme bitince pişirme (debounce) eklenir.
   ★ Sinsi olan: pişirme başarısızsa `status` "rebake_failed: …" der ve yüzey eski haritayla
   SESSİZCE devam eder.

## Ortak oda hedefi şovu (2026-11-17, Claude) — aynı derlemeye girer

**Durum:** KAYNAK HAZIR, derlenmedi, /Zs PASS (`BalloonPaintDemo.cpp` iki hedef,
`GameDesignUI.cpp`). Header/struct/shader yok.

Ne: `room_goal` (0 = kapalı, vars. **0,5**: boyanabilir texel'lerin yarısı "dolu oda").
Boyanma saniyede 2 kez ölçülür; %25/%50/%75'te 1/2/3 konfeti patlaması, %100'de konfeti
yağmuru + 4 patlama + kalan balonların art arda patladığı final; `finale_reset_seconds`
(vars. 10) sonra oda kendini temizler, yeni tur başlar. `celebrate_sound` isteğe bağlı.
Parlama/yanıp sönme YOK. Panel: Shared room goal bölümü + ilerleme çubuğu.

1. Game Design → Room goal 0,10 (kolay ulaşılsın) → Reconfigure → birkaç balon patlat:
   ilerleme çubuğu artar, %25/%50/%75'te havada konfeti.
2. %100: konfeti yağmuru + balonlar sırayla patlar + ~10 sn sonra oda temiz, çubuk 0.
   ★ Sinsi olan: çubuk hiç artmıyorsa kaplama ölçümü düşüyor (`coverage` false) — hata
   vermez, şov sessizce hiç gelmez. `paint.wet_demo` status `room_progress`'e bak.
3. `python scripts/test/rt_room_goal_show_ipc.py` (Studio ya da `--player`) → 6/6.

**Zemin (kullanıcı sahnesi, 2026-11-17 ölçüldü):** `PaintFloor` sahnede yok; zemin
`LeftWallBox_1` (LeftWallBox kopyası, hafif eğik: y −0,08..0,22). `LeftWallBox` ile AYNI
materyali (`Warm Ivory Trim`) paylaştığı için boya yüzeyi yapılamıyor (çekirdek reddeder).
Yol: Paint Mode → Binding Mode "Make Unique For Paint" → Make Unique Now, sonra Game Design'da
yüzey ekle; collider'ı aabb yerine **obb** yap (eğik zeminde aabb üstü 22 cm'e kadar havada
kalır, leke yarıçapı 12 cm → boya düşmez). Açık iş: "Make Unique For Paint" yalnız panelde,
IPC'de yok (CLAUDE.md §1 ihlali).

## Sürpriz balon (2026-11-17, Claude) — aynı derlemeye girer

**Durum:** KAYNAK HAZIR, derlenmedi, /Zs PASS (`BalloonPaintDemo.cpp` iki hedef,
`GameDesignUI.cpp`). Header/struct/shader yok. Sahneye yeni CPU parçacık sistemi
`BalloonConfetti` eklenir (demo hazırlanırken, projeyle kaydedilir).

Ne: `surprise_chance` (0..0,5, varsayılan **0,1** — mevcut projelerde de açık) ile her
yeniden doğuş zar atar. Sürpriz = ALTIN metal balon; patlayınca 1,5× damla, her damla rastgele
palet renginde (gökkuşağı boya), 220 dönen konfeti kartı (boya BIRAKMAZ: çarpışma olayı
kaydetmez), `surprise_sound` ya da yoksa patlama sesi 1,25× tiz. Panel: Surprise chance,
Surprise sound.

1. Studio Game Design → Surprise chance 0,5 → Reconfigure: balonların yarısı altın ve metalik.
   Altın görünmüyor ama status `surprise:true` diyorsa malzeme `metallic` yazılmıyor.
2. Altın balonu patlat: duvarda ÇOK renkli boya + havada süzülen renkli kartlar + daha tiz ses.
   ★ Sinsi olan: gökkuşağı damlalarının bir kısmı BEYAZ boyuyorsa profil yanlış sisteme
   gidiyor (renk c → `kSystems[c % 4]`).
3. Konfeti duvara/zemine boya bırakmamalı; bırakıyorsa `BalloonConfetti` olay kaydediyor.
4. `python scripts/test/rt_surprise_balloon_ipc.py` (Studio) → 6/6; `--player` → 4/4.
5. Sürpriz 0'a çek → hiç altın balon yok, normal balonların metalliği 0.

## Oda sıfırlama (elle / boşta / oturum sonu / F9) + balon boyu (2026-11-17, Claude)

**Durum:** KAYNAK HAZIR, derlenmedi, /Zs 7/7 PASS (Studio + Player). Header yok, struct
düzeni yok, shader yok. `tests/body_pose_game_test.cpp` beklentisi `room.reset` oldu.

Ne değişti:
- `paint.wet_demo reset`: boya temizlenir, patlamış balonlar geri gelir; havuzlar korunur
  (yeniden ısınma yok). Uçuştaki damlalar ≤2,8 sn içinde konar → boya 3 sn sonra bir kez
  daha temizlenir. Player **F9** (görevli) aynı çekirdeği çağırır. Oturum sonu artık
  `room.reset` eylemi yayar (boya + balonlar; `paint.reset` yalnız boya olarak kaldı).
- `idle_reset_seconds` (0 = kapalı, 10..3600): son OYUNCU patlatmasından bu kadar sonra
  oda kendiliğinden temizlenir. Otomatik (attract) patlamalar boyar ama etkinlik sayılmaz.
- `size_min/size_max` (0.3..2.0): her doğuşta boy; mesh = geometri × boy (dönüşümün kendi
  ölçeği atılır — kayıt/yükle ile katlanmaz), tıklama yarıçapı boyla, damla = taban × boy²,
  saçılma boyla, ses perdesi boy^-0.7. Damla başına LEKE boyu hâlâ tek global ayar (shader
  değişikliği gerektirir; ayrı iş).
- Panel: Balloon size aralığı, Idle reset, "Reset room" düğmesi.

Sıra:
1. Studio Game Design → Balloon size 0.6–1.4 → Reconfigure: balonlar gözle farklı boyda,
   büyükler daha pes patlar ve daha çok boyar. Hepsi aynıysa `size` status'ta bak.
2. **Player F9:** duvarlar 3 sn içinde tamamen temiz, bütün balonlar yerinde.
   ★ Sinsi olan: F9 sonrası birkaç leke "geri gelirse" ikinci temizlik (3 sn) çalışmıyor.
3. `python scripts/test/rt_balloon_reset_size_ipc.py` (Studio; `--player` ile Player) → 9/9.
4. Idle reset 30 sn ver, kaydet, Player'da aç, bir balon patlat, 30 sn dokunma → oda temiz.
5. Kaydet → aç → kaydet → aç: balon boyları aralıkta kalır, büyümez (katlanma yok).
6. **Canlı palet (aynı parti, derleme öncesi eklendi):** varsayılan artık 8 renk (kırmızı, turuncu, sarı, yeşil, turkuaz, mavi, mor, pembe), ikincil kanallar ~0 (eski kırmızının doğrusal 0,08 yeşili ekranda %31 = pastel). Game Design → Colours: renk kutuları, + ekle, sağ tık çıkar (2..8). Balon ve duvardaki boya AYNI renkte olmalı; farklıysa boya profil rengini değil başka bir şeyi okuyor. ★ Çok renk üst üste binince ıslak boya karışır — duvar kahverengiye dönüyorsa paleti daralt ya da karışım oranına bakılacak (ayrı iş). Eski `rt_workshop_balloons_ipc.py` renkleri artık status `palette`ından okur.

## Damla collider'ı Game Design'dan (SDF dahil) + balon başına patlama sesi (2026-11-16 gece 3, Claude)

**Durum:** KAYNAK HAZIR, derlenmedi, /Zs 8/8 PASS (Studio + Player). Header: yalnız
`BalloonPaintDemo.h` (+3 fonksiyon). Struct düzeni değişmedi. Shader yok.

Ne değişti:
- **Collider = mevcut atama.** Yeni bir collider sistemi YOK: Game Design'daki seçim, mevcut
  `ParticleColliderDesc`'i (kaynak obje + mod, objeyi izler) dört damla sistemine
  `<obje> PaintCollider` adıyla yazar. Parçacık sistemiyle birlikte kaydedilir; game_design'a
  ayrıca girmez. Modlar none / aabb / obb / mesh_sdf (32/64/128). IPC:
  `paint.wet_demo operation=set_collider`; her sonuç `colliders` listesini taşır (elle
  kurulmuş `BackWallBoxCollider` vb. dahil).
- **SDF pişerken demo hazırlanmaz** (havuzlar GPU-zorunlu): configure açık hata verir; proje
  yüklemesi pişmeyi bekler (`paint.wet_design` status `waiting for collider SDF: <ad>`,
  ~1 dk sonra vazgeçip söyler).
- **Ses:** yeni `audio.play_at {id, position, gain, pitch}` — bir kaynağın dosyasını verilen
  konumda BAĞIMSIZ tek-atışlık ses olarak çalar (gök gürültüsünün geçici ses yolu; en fazla
  16 tek-atış / 32 toplam, bütçe dolunca hata değil `played:false`). Demo config `pop_sound`
  (ses kaynağı id): her patlayan balon kendi konumunda bir kez, ±%8 perde farkıyla. Spatial
  kaynakta asset MONO olmalı (gök gürültüsüyle aynı kural).
- Panel: "Droplet colliders" bölümü (liste + seçili obje için şekil/SDF çözünürlüğü),
  Balloon demo'da "Pop sound" seçimi.

Sıra:
1. **Game Design → Droplet colliders** listesinde atölyenin 4 elle kurulmuş collider'ı görünür
   (`BackWallBox [aabb]`, `Floor... [plane]` …). Boşsa demo sistemleri yok/yanlış adda.
2. Bir oyuncak objesi seç, Shape = mesh_sdf → listede `[mesh_sdf, cooking]`, sonra `cooking`
   kalkar. Damlalar o objede boya bırakmaz (yüzey değilse) ama SEKER/ÖLÜR, içinden geçmez.
3. `python scripts/test/rt_balloon_collider_sound_ipc.py` (Studio; `--player` ile Player)
   → 4/4 (sessiz). Mono bir ses kaynağı varsa `--sound <id>` → 7/7.
4. **Kulakla:** üç balonu aynı anda patlat (`burst count 3` ya da hızlı tıklama) → üç ayrı
   patlama, her biri kendi yönünden; biri diğerini kesmez.
   ★ Sinsi olan: stereo asset → `pop_sound_error` "mono asset" der ama patlama sessizce
   sürer. Ses yoksa önce `paint.wet_demo` status `pop_sound_error`.
5. Kaydet → yeniden aç: SDF collider pişerken status `waiting for collider SDF`, sonra
   `applied: … balloon demo prepared`. Demo hiç hazırlanmıyorsa 1 dk sonra sebebi yazar.

**Ek (ilk canlı tur):** kullanıcının `sfxmint-balloon-pop-03.wav` dosyası STEREO 44.1 kHz (0,40 sn) → mono kopya `assets/audio/sfxmint-balloon-pop-03-mono.wav` üretildi (L/R tepe 26421/26418, karışım 26420: faz iptali yok). Studio `audio.status`: `device: not_initialized`, transport `paused` — `play_at` duraklatılmış zaman çizelgesinde `transport_paused` döndürüp ÇALMIYORDU (Studio'da düzenleme hep böyle). Düzeltildi: tek-atışlar oyun saatinde çalar, `holdTransport` onları tutmaz (`AudioService.cpp`, /Zs PASS, derlenmedi). "Audio device is not active yet" yazısı hata değil: aygıt ilk seste açılır.

**Ek 2:** beyaz top geri geldi: `PaintBall_08` materyalindeki eski boya dokusu proje DOSYASINDA kalmıştı (canlı temizlik kaydedilmemişti; derleme sonrası açılış geri getirdi). Studio'da yeniden temizlendi — **kullanıcı kaydetmeli.** Kalıcı önlem: demo artık temel renk dokusu taşıyan balon materyalini sebebiyle REDDEDER (`validateMeshVisuals`, /Zs PASS, derlenmedi); sessiz beyaz top yerine açık hata.

Bilinen maliyet (ölçülmedi): SDF her damla sisteminde ayrı pişer (aynı mesh 4 kez). GPU
pişirme hızlı olduğu için şimdilik kabul; karmaşık sahnede ölç.

## Game Design ayrı sekme, kaldırılan yüzey dokusunu bırakır, kaydırmalı listeler (2026-11-16 gece 2, Claude)

**Durum:** KAYNAK HAZIR, derlenmedi, /Zs 5/5 PASS. Header: `WetPaintGpu.h` (`removeSurface`
varsayılan argümanlı parametre; düzen aynı, 9 includer). `scene_ui.cpp` değişti.

Canlı ölçülen (Player + Studio, IPC): kaydedilen boya ayarları geri GELİYOR — 4 yüzeyin
`lifetime/roughness_channel/height_normal_channel/wetness` değerleri dosya = Studio canlı =
Player canlı; `pbr_texture_bytes` 15.7 MB her ikisinde. Panelde görünmemesinin nedeni:
ayarlar yalnız yüzey SEÇİLİYKEN açılıyordu ve Paint sekmesi seçimi kilitliyordu.
"Varsayılan materyalli küre": `PaintBall_08` (CPU mavi) ekranda beyazdı — bir ara boya
yüzeyi yapılıp kaldırılmış, beyaz boya dokusu `base_color` slotunda bağlı kalmıştı
(`material.get` rengi doğru raporluyordu: panel-yalan sınıfı). Studio'da kalıntı doku IPC ile
temizlendi (yalnız bu topta vardı); **kullanıcı projeyi kaydetmeli.**

1. **Sol şeritte "Game Design" sekmesi** (Paint Mode'un altında). Paint sekmesine gitmeden
   bir duvar seçince boya ayarları altında açılır; seçim kilitlenmez.
2. **Listeler en fazla 5 satır**, fazlası kutu içinde kayar; panel uzamaz.
3. **Bir topu yüzey yap → kaldır → top kendi renginde** (beyaz kalmaz),
   `material.textures` o topun materyalinde `generated/...` göstermez. Kalıyorsa
   `WetPaintGpu::removeSurface` scene'siz çağrılmış.
4. Player'da kaydedilmiş workshop: 12 top da renkli (beyaz yok).

## İlk canlı tur bulguları + kanal boyutu düzeltmesi (2026-11-16 gece, Claude)

**Durum:** `WetPaintGpu.cpp` + `GameDesignUI.cpp` değişti (artımlı, header yok), /Zs PASS.

Kullanıcı derledi: Player'da patlatma çalışmadı. Kök (canlı IPC ile okundu):
Studio'da yalnız `PaintLeft` boya yüzeyi yapılmış, kanallar (roughness + kabartma) açılmış,
"Prepare demo" `could not prepare wet paint GPU atlas` ile düşmüş (`paint.wet_status`:
`channel_resolution_mismatch: PaintLeft`). Proje bu haliyle kaydedildi → `game_design`
VAR ama `balloon_demo` YOK → Player yan dosyayı (doğru olarak) okumadı → balon yok.
Kanal hatasının kökü: yüzey bir boyutta eklenip kaldırılıp başka boyutta yeniden eklenince
taban doku eski boyutta kalıyor, set'in `resolution`'ı yeni istenene dönüyor, kanal dokuları
o yanlış boyutta yaratılıyor. Düzeltme: kanal dokuları taban dokunun GERÇEK boyutunda
yaratılır, yanlış boyutlu kalıntı kanal dokusu yeniden yaratılır; yüzeyin raporlanan
(ve kaydedilen) `resolution`'ı istenen değil gerçek boyuttur. Panel yeni yüzey için
varsayılan olarak mevcut yüzeylerin boyutunu önerir.

0. **Game Design paneli isim yazdırmıyor:** yüzeyler ve balon meshleri listede `x` ile çıkar, "Add selected" seçili objeyi ekler, yanındaki "Pick from scene..." açılır listesi sahnedeki objeleri (filtreli) gösterir. Yüzey olan obje balon listesinde, balon olan obje yüzey listesinde çıkmaz (çekirdek ikisini birden reddeder). Balon meshi verilince balon sayısı listenin boyudur (3..24). Yüzeyin boya ayarları, o yüzey sahnede SEÇİLİYKEN görünür.
1. **Studio, workshop açık:** `paint.wet_status` → `status` artık `channel_resolution_mismatch`
   değil; Game Design'da "Prepare demo" başarılı ("Prepared, stopped | flat_mesh").
   Hâlâ düşüyorsa hata metni panelin altında yazar.
2. Dört duvarı da (Floor/Back/Left/Right) yüzey yap, demoyu hazırla, KAYDET → Player'da aç →
   tıkla-patlat boyar. ★ Sinsi olan: demo hazırlanmadan kaydedilen proje Player'da hatasız
   açılır ama balon yoktur.
3. Sonra alttaki bölümün 3. maddesi (`rt_game_design_roundtrip_ipc.py --open`).

Ayrı açık (bu partide DOKUNULMADI): HDRI altında balonlarda köşeli koyu dilimler ve sağ
duvarda simsiyah elipsler. Ölçülen: RT gölge kapalı/açık ve RayFusion yansıma açık/kapalı
görüntüyü değiştirmiyor (yansıma açıkken 238k ışından yalnız 1.3k vuruş gölgelendi: "bounce
yalnız nokta/yönlü ışık"). Sahnede tek nokta ışık (yarıçap 0.01, sert gölge). Duvar
roughness 0.72 olduğu halde siyah elipsin içinde parlama var — gölgede çevre ışığı da
sönüyor olabilir. Kök bulunmadı.

## Oyun tasarımı projeyle kaydedilir + Studio "Game Design" bölümü (2026-11-16 gece, Claude)

**Durum:** KAYNAK HAZIR, C++ DERLENMEDİ. Shader yok. **İki yeni .cpp** (`Paint/GameDesign.cpp`
iki hedefte, `UI/GameDesignUI.cpp` yalnız Studio; vcxproj'lara eklendi, Player üretici
`--check` PASS). Header değişikliği: `BalloonPaintDemo.h` (+1 fonksiyon), `RtApiPaintWet.h`
(+1 struct, +1 fonksiyon) — yaygın struct düzeni değişmedi, artımlı derleme yeter.
Alttaki "tıkla-patlat" partisi de hâlâ derlenmedi; ikisi aynı derlemede gelir.

Ne değişti: proje artık `game_design` anahtarı taşır — ıslak boya yüzeyleri + ayarları,
splat kuralları, balon demosu (yüzey listesi olmadan). Kopya yok: kayıt canlı atlas/demodan
okunur. Yüklemede belge kuyruğa girer, **ortak sim saati ilk tikte** uygular (Studio ve
Player aynı yol). Player'da `.balloons.json` yalnız proje `game_design` taşımıyorsa okunur.
Demo `surfaces` OLMADAN yapılandırılınca sahnedeki yüzeyleri kendi ayarlarıyla boyar (kendi
"atölye görünümünü" yalnız `surfaces` verilirse uygular). Studio'da proje değişince ıslak
boya + demo artık sıfırlanır (önceden eski sahnenin yüzeyleri yeni projeye taşınıyordu).
Yeni IPC: `paint.wet_design` (Read) → `{design, pending, status}`; Python `rt.paint.wet_design()`.

Sıra (bağımsız ve hızlı önce):

1. **Studio açılıyor, Paint panelinin altında "Game Design" bölümü var.** Bir mesh seçili
   değilken "Select a mesh…" yazar. Yoksa: eski exe (zaman damgası) ya da
   `scene_ui_modifiers.cpp` derlenmemiş.
2. **Boş sahnede proje aç/yeni proje: hata yok, bölüm "Not prepared".** Önceki projeden
   yüzey listesi görünüyorsa `newProject` sıfırlaması çalışmıyor.
3. **Otomatik gidiş-dönüş (Studio, workshop açık):**
   `python scripts/test/rt_game_design_roundtrip_ipc.py --open` → **9/9 PASS**. Orijinal
   `workshop.rtp`'ye yazmaz, kopyayı temp'e kaydeder ve yolunu basar.
   - 1 FAIL: demo yüzeysiz yapılandırılamıyor → `validatedConfig`/`prepare` dalı.
   - 2 FAIL (lifetime 12 görünüyor): demo hâlâ kendi görünümünü basıyor → `builds_surfaces`.
   - 4/5 FAIL: `saveProjectImpl`'daki `game_design` bloğu çalışmadı.
   - 6–8 FAIL: `openProject` belgeyi kuyruğa koymadı ya da saat tikinde uygulanmadı
     (`status` hangi nesneyi bulamadığını yazar).
   - ★ **Sinsi olan 7:** design "neredeyse aynı" — örn. ayarlar varsayılana dönmüş ama yüzey
     sayısı doğru. Gözle bakan biri boyayı yine görür; test sayıları karşılaştırır.
4. **Player'da kopya (3. adımın bastığı yol) `project.open`:** `paint.wet_design` →
   `pending=false`, `status="applied: 4 paint surface(s)"`, `paint.wet_demo` prepared,
   12 mesh balon görünür; tıkla-patlat boya bırakır. `.balloons.json` kopyada YOK — yani demo
   tamamen projeden geldi. Demo prepared değilse: viewer yan dosyaya düştü ve bulamadı.
5. **Orijinal `workshop.rtp` (hâlâ yan dosyalı) Player'da eskisi gibi açılır** (yan dosya
   yedeği). Bozuksa `RuntimeSceneViewer.cpp` `sidecar_ok` dalı.
6. **Panelden düzenleme = IPC:** panelde bir yüzeyin "Open time"ını değiştir, sonra
   `paint.wet_get_settings` aynı değeri okur; panelde "Prepare demo" sonrası
   `paint.wet_demo` status'taki config panel formuyla aynı. Farklıysa panel durum tutuyor
   (CLAUDE.md §1 ihlali).

Karar bekleyen (kullanıcı): `workshop.rtp`'nin kendisi Studio'dan kaydedilip `game_design`
taşısın ve `.balloons.json` + `workshop.paint-profile.json` emekliye mi ayrılsın? Kaydedilen
ayarlar o zaman demonun eski sabit "görünümü" değil, panelde ayarlanan değerler olur.

## Balon demosu: tıkla-patlat, kanal maskesi, 12 sn kuruma, beyaz küre (2026-11-16 akşam, Claude)

**Durum:** kaynak hazır, shader derlendi (`paint_wet.spv` iki yerde), C++ DERLENMEDİ. Player
(ve Studio) artımlı ama GENİŞ: `WetPaintGpu.h` + `RtApiPaintWet.h` varsayılan değer değişti.
Struct düzeni, bağlama, push boyutu, yeni .cpp YOK.

**Kullanıcı canlı (aynı akşam, Player yeniden derlendi):** beyaz küre YOK, fareyle patlatma
İYİ. İki yeni bulgu ve düzeltmesi (shader yeniden derlendi; `BalloonPaintDemo.cpp` DERLENMEDİ):

- **Kabartma duvarda ters aydınlanıyordu ("boya yukarı kayıyor").** Kök: height→normal yeşil
  kanal işareti. Texel y aşağı büyür, ama raster/RT bitangent'i +v = görüntü-YUKARI
  (`material_preview_frag` `applyNormalMap` B'yi bunun için çevirir). Doğru detay normali
  `(-gh, +gv, 1)`; shader `-gv` yazıyordu. `paint_wet_channels.glsl` düzeltildi.
  ★ CPU Paint (`MeshPaintAdapter.cpp:947, 1019`) aynı `-dy` işaretini taşıyor: aynı hata
  büyük olasılıkla orada da var, DOKUNULMADI ve doğrulanmadı. Yeni `.spv` Player yeniden
  başlayınca yüklenir. Kontrol: duvardaki bir damlanın ÜST kenarı (ışık yukarıdaysa) aydınlık.
- **Boya hâlâ hızlı kuruyordu.** `lifetime_seconds` 12 tek başına yetmez: ıslaklık ayrıca
  `absorption × ıslaklık × 0,7 / sn` ile düşer (varsayılan 0,18), ve bir damlanın başlangıç
  ıslaklığı 1 değil. Demo artık hazırlanırken atölyenin akan profilini kendisi uygular
  (wetness .95, lifetime 12, diffusion .1, runoff .7, absorption .06, drip_head .8, load .95,
  deposit .9, pickup .2, radius 6) ve iki kanalı da AÇAR (`prepare`, ilk coverage'dan önce:
  kanal dokuları yüklemede bağlanır). Açık Player'a aynı değerler IPC ile canlı uygulandı.
  Sonuç: `apply_workshop_paint_profile.py` bu demo için gereksizleşti; `prepare_ms` /
  `atlas_ms` kanal hazırlığı kadar artar (ölçülmedi, 7,2 sn takılma adayı buraya taşındı).

Ne değişti:
- `paint.wet_demo operation=pop` (yeni işlem, yeni metot değil): `balloon` (id) ya da `screen`
  ([x, y] 0..1, sol üst). Player sol tık aynı çekirdek çağrıyı yapar (`PlayerMain.cpp`).
  Yeniden doğma zaten vardı (3,5 oyun saniyesi), dokunulmadı.
- Kanal maskesi (`paint_wet_channels.glsl`): roughness ve kabartma artık kaplamaya doğrusal
  değil `smoothstep(0.25, 0.85)` ile bağlı, yani lekenin ince kenarı parlamaz/kabarmaz.
  Varsayılan `normal_strength` 1.0 -> 0.5.
- Varsayılan `lifetime_seconds` 2.5 -> 12 (yalnız GPU ıslak boya; fırça 2.5 kalır).
- Isınma parçacıkları `size_scale` 0.001 (merkezdeki beyaz küre adayı).

Sıra, pastel atölye Player'da açık, otomatik demo kapalı:

1. **Tıkla** — bir balona sol tık: o balon patlar, boya sıçrar, ~3,5 sn sonra başka yerde yeni
   balon. Boşluğa tık: hiçbir şey. Patlamıyorsa önce F5 (debug kamera) açık mı bak; açıkken
   sol tuş kamerayı döndürür. Yanlış balon patlıyorsa pencere-görüntü eşlemesi kayık
   (pencere en-boyu render en-boyundan farklıysa beklenir, bakılmadı).
2. **Otomatik** — `python scripts/test/rt_player_balloon_pop_ipc.py` -> PASS (ret edilen 7 kötü
   istek, id ile, aynı id tekrar = -1, ekran noktasıyla, yeniden doğma).
3. **Beyaz küre** — taze açılışta, ilk patlamadan ÖNCE oda ortasına (0, 2, 0) bak: küre
   olmamalı. ★ Bu düzeltme DOĞRULANMAMIŞ bir teşhise dayanıyor (ısınmanın son parçacığı son
   adımdan sonra öldürülüyor, cihaz kopyası ilk gerçek doğuma kadar yaşıyor). Küre hâlâ
   duruyorsa teşhis yanlış: aday `PreallocatedBurstPool` emitter'ı ya da görsel sistem.
   Küre artık görünmüyor ama 1 mm'lik bir nokta olarak orada duruyor olabilir; asıl kök
   (öldürmenin cihaza gitmemesi) çözülmedi, yalnız görünmez yapıldı.
4. **Kuruma** — hiçbir profil uygulamadan patlat: boya ~12 sn ıslak kalmalı ve duvarda aksın.
   Hâlâ 2,5 sn'de kuruyorsa eski binary ya da sahneye bir profil 2,5 yazıyor.
5. **Kanal görünümü** — dört yüzeyde iki kanalı aç, patlat: parlak/kabarık alan renkli lekeden
   GENİŞ olmamalı; kabartma öncekinden yumuşak. Fazla zayıfsa `normal_strength` yükselt
   (0..8), eşikler shader'da sabit (0.25 / 0.85).
6. `rt_player_wet_paint_channels_ipc.py` yine PASS olmalı (roughness texel sayısı düşer; adım 2
   "> 0" bekler). 0 çıkarsa maske tek dab'i tamamen eliyor: dab kaplaması 0.25'i geçmiyor.

★ En sinsi: tık "çalışıyor" görünür ama yanlış balonu patlatır (komşu balon, ya da arkadaki).
Seçim merkezden 0,4 m yarıçaplı küreye göre ve en YAKIN olanı alır; balon mesh'i daha büyük ya
da küçükse kenar tıklamaları sessizce kaçar/taşar.

## GPU ıslak boya — roughness / height / normal kanalları (2026-11-16, Claude; Codex'in yarım partisi)

**Durum:** kaynak hazır, shader derlendi (`paint_wet.spv`, iki yerde), C++ DERLENMEDİ, canlı test YOK.
Derleme artımlı ama geniş (`WetPaintGpu.h`, `RtApiPaintWet.h` alan ekledi; `VulkanBackend.cpp`,
`MeshPaintAdapter.cpp` değişti). Studio + Player. Yaygın struct düzeni değişmedi.

Ne yapıldı: Codex çözüm shader'ını, atlas hazırlığını ve IPC/Python alanlarını yazmıştı; kanal
düzlemleri GPU'da üretiliyor ama **dokulara hiç kopyalanmıyordu**. Yayın adımı eklendi.
Kanallar varsayılan KAPALI; `paint.wet_set_settings roughness_channel / height_normal_channel` açar.

**Player ön ölçümü (2026-11-16 19:35, 18:32 tarihli Player = yayın adımından ÖNCEKİ binary,
yeni `.spv` ile, pastel atölye, PaintFloor 512):** `rt_player_wet_paint_channels_ipc.py` adım 1-3
PASS, adım 4 beklendiği gibi FAIL. Renk-only: kanal dispatch 0. Roughness: 532 kaplı texelde
1044 roughness texeli (262144 geçerli), normal 0, diğer üç yüzey 0. Height: mevcut lekeden 1144
normal texeli. Yeni dab: 2102 / 2138. `pbr_device_copies` 0 (renk kopyası 42, host bayt 0):
hesap doğru, dokuya ulaşmıyor. `last_tick_ms` 0,70 -> 2,08. Ek atlas 32 MiB, kanal dokusu 3 MiB.
Yani **Player'ı yeniden derle**, aynı betik adım 4-5'i de geçmeli.

**Player CANLI PASS 5/5 (2026-11-16 19:38 derlemesi, iki ardışık koşu):** kanal düzlemleri artık
dokulara ulaşıyor: `pbr_device_copies` +61/+62 (renk +42/+44), host bayt 0 (`device_copy`).
Reset sonrası kaplı/roughness/normal 0. `last_tick_ms` renk-only 0,85, kanallar açık 0,6-1,3
(tek koşu, gürültülü; maliyet karşılaştırması sayılmaz). Kanal atlası bir kez ayrılınca kanallar
kapansa da durur (32 MiB) ve her reset'te bir tam yenileme dispatch'i koşar (ikinci koşuda
`colour_only.pbr_dispatches_delta` = 1 bundan). AÇIK: gözle görünüm (parlaklık/kabarıklık,
boyasız alan aynı mı), kopya maliyeti, balon patlamasında dört yüzey, Studio (ikinci VkDevice
host yolu) hiç koşmadı.

**Player maliyet ölçümü (2026-11-16, `rt_player_wet_paint_channel_cost_ipc.py`, pastel atölye,
4 yüzey x 512, üçlü patlama 5400 damla, 60 Hz tempolu, capture kapalı, 2 tur dönüşümlü):**

| Yapı | boya adımı ms (ort, 2 tur) | en kötü kare aralığı ms | kopya/s renk + kanal |
|---|---|---|---|
| renk | 1,17 / 0,86 | 18,9 / 22,3 | 85-124 + 0 |
| +roughness | 1,18 / 0,98 | (7211) / 21,4 | 103-121 + 103-121 |
| +roughness+normal | 1,09 / 1,01 | 22,2 / 21,0 | 125 + 376-378 |

`frame_cost_ms` üç yapıda da 2,2 ms (turlar arası fark yapılar arası farktan büyük). Host bayt 0,
düşen kaynak 0, device-lost yok. Sonuç: bu ölçekte kanalların kare başı maliyeti ölçüm gürültüsünün
içinde (boya adımında en çok ~+0,1 ms), kare aralığı değişmiyor. SINIR: 60 Hz tempolu koşu kare
altı maliyeti gizler; kapasite (uncapped) ölçülmedi, `frame_cost_ms`'in yayın kopyalarını içerip
içermediği doğrulanmadı.
★ AÇIK: kanalın üç yüzeyde İLK açıldığı turda bir kez 7,2 s'lik sunum boşluğu görüldü (ikinci
turda yok, height/normal ilk açılışında da yok). Kök doğrulanmadı; aday: ilk kanal dokusu ataması
(katman yığını + UV pişirme + tam materyal yüklemesi). Oyun sırasında değil yüklemede açılmalı;
taze süreçte tekrar ölçülmeli.

Açık sahnede (Player, hiçbir şey yaratmaz, boyayı sıfırlar, ayarları geri koyar):
`python scripts/test/rt_player_wet_paint_channels_ipc.py` (`--target studio`, `--keep` var).

Sıra (bağımsız ve hızlı önce). Boş Studio sahnesi:

1. **Eski test bozulmadı mı** — `python scripts/test/rt_wet_paint_gpu_ipc.py` → PASS 8/8.
   Bozuksa: renk-only yol etkilenmiş (shader `_p0 = 0` dalı ya da `publish` yeniden yazımı).
   Diğer her şeyi maskeler, önce bu.
2. **Kanal testi** — `python scripts/test/rt_wet_paint_channels_ipc.py` → PASS. Adım adım:
   - `colour_only`: `pbr_dispatches = 0`, `pbr_buffer_bytes = 0`, roughness/normal texel 0.
     Değilse kanallar kapalıyken maliyet var.
   - `roughness`: `roughness_texels > 0` ve yüzeyin yarısından az; `normal_texels = 0`;
     kontrol yüzeyi 0. `roughness_texels` yüzeyin tamamına yakınsa orijinal roughness
     korunmuyor (skaler faktörün dokuya pişirilmesi yanlış).
   - `height_normal`: mevcut lekeden `normal_texels > 0`. 0 ise yükseklik tohumu
     (mod 8) çalışmıyor.
   - `new_dab`: `pbr_device_copies > 0` (Studio'da viewport ikinci VkDevice olduğundan
     `host_downloads` da artar, normal). İkisi de 0 ise kanal dokusu backend'de yerleşik
     değil ya da RGBA8 değil (`paint.wet_status.status` içindeki `channel_*` metnine bak).
   - `reset`: üç sayaç 0.
3. **Gözle** — `--keep` ile koş, Rendered/RT ve raster'da bak: leke çevresinden **daha parlak**
   (düşük roughness) ve kenarında **kabarıklık** görünmeli; boyasız zemin öncekiyle AYNI olmalı.
4. **Player** — pastel atölyede dört yüzeye `roughness_channel=true height_normal_channel=true`
   ver, tekli/üçlü patlama. `wet_status`: `publish_bytes_downloaded` 0 kalmalı (tek VkDevice),
   `pbr_device_copies` artmalı. Kare süresini renk-only ile karşılaştır.

★ **En sinsi başarısızlık — boyasız yüzey sessizce değişir.** Kanal açılınca materyalin skaler
roughness faktörü ve normal gücü üretilen dokuya pişirilir, materyalde 1'e çekilir. Pişirme
yanlışsa (roughness dokusunun kanal seçimi, yoğunluk çarpanı) BOYASIZ alan biraz daha parlak
ya da mat olur; test bunu yakalamaz çünkü "orijinal" diye pişirilmiş değeri sayar. Kanalı
açmadan ve açtıktan sonra boyasız bir bölgeye aynı ışıkta bak.

Bilinen sınırlar / bakılmadı:
- Her kanal ayrı bir `copyDeviceBufferToTexture` (her biri `waitIdle` + tek seferlik komut):
  iki kanal açıkken kare başına yüzey başına 1 yerine 4 kopya. Maliyet ölçülmedi;
  `pbr_device_copies` / `last_tick_ms` ile renk / +roughness / +height-normal ayrılabilir.
- Boya kuruduğu karede karo pasifleşiyorsa roughness `dry_roughness`'a tam varmadan kalabilir
  (son ıslaklık değeriyle). Kuru lekede roughness'e bak.
- Kanal kapatılınca / yüzey kaldırılınca üretilen dokular ve 1'e çekilmiş faktörler materyalde
  KALIR (geri alma yok). Proje kaydedilirse materyal bu haliyle kaydolur.
- Materyalde önceden CPU Paint ile oluşturulmuş bir kanal dokusu varsa (farklı çözünürlük)
  `channel_resolution_mismatch` ile reddedilir; sıkıştırılmış/R8 olarak yüklenmişse host yoluna düşer.
- Panel yok: ıslak boya yüzey ayarları yalnız script/IPC'den (kanallardan önce de böyleydi).
- Pastel atölye profili ve balon demosu kanalları AÇMIYOR.

## Pastel atölye — kullanıcı build + Player canlı kabul PASS (2026-11-16, Codex)

`rt_workshop_balloons_ipc.py` PASS: mesh poz/renk/gizleme/respawn, 3 negatif
config, tekli/üçlü 1800/5400 damla, dört yüzey boyası, taşma/host byte/device-loss0.
Görsel ek kontrolde 12 top birlikte gizlendi, gölgeleri de kayboldu; 21.600 damlada
bu senaryoda taşma0. Kaynak proje açık, boya görünür, auto demo kapalı, capture
ve display geri alındı. `PLAYER_WORKSHOP_ACCEPTANCE_2026_11_16.json`.

Asset roughness çarpımı .09→.18 düzeltildi; kaynak asset yeniden açıldı, C++/shader
build gerekmedi. Asset audit etkin scalar'ı test eder. Bu tur C++/shader değişmedi.
Kullanıcının isteğiyle wet12s/runoff.7 profili mevcut API'den uygulandı; 8.saniyede
üç duvar da ıslak. Kalıcı preset JSON + apply_workshop_paint_profile.py mevcut;
reopen/configure sonrası yeniden uygulama gerekir, startup'a bağlanmadı.

AÇIK: mevcut Paint height/normal/roughness kanallarını GPU wet output'a bağlamak
(A'nın aktif dosyaları; pano §4 devir isteği), maliyeti sonra ölçmek. Yeni port
bu kabulde PASS sayılmaz; Studio canlı yolu da ayrıca sınanmadı. Genel 64-source
sınırı/paint publication beklemeleri ve kamera/çocuk algısı hâlâ ayrı işler.

## Yeni kaynak paketi — görünmeyen ızgara yeniden kurulmaz (2026-11-16, Claude)

Tek satırlık kapı: `Backend/VulkanViewportBackend.cpp` (ızgara yeniden kurma koşulu
`grid_opacity > 0` ile kapılandı). Artımlı; Studio+Player. Derlenmedi.

**Belirti (kullanıcı):** Player'da kamera sahneye YAKINKEN hareket takılıyor, uzakta akıcı.
**Ölçüm (IPC orbit, 81 kare):** `overlay` aşaması CPU ortalama 1,4–2,2 ms ama p95 15,5 ms
(uzak) / 28,7 ms (yakın) ≈ bir tam GPU karesi; çağrı süresi p50 33 / p95 64 ms (yakın). Yani
karelerin ~%6'sı bir tam kare duruyor. **Kök:** editör ızgarası, kamera 10×-aralıklı kaba
hücreyi her geçtiğinde `drainInteractiveViewportInFlight()` + iki tampon yeniden yaratımıyla
kuruluyor; aralık kamera zemine yaklaştıkça küçüldüğü için yakında sık tetikleniyor. Player'da
ızgara opaklığı 0 — görünmeyen çizgiler için duruyordu. Önceki paketteki karo önbelleği ayrıca
doğrulandı: yakın/uzak orbit'te `shadow_atlas.frames_ran` 0/81, kamera maliyeti çizimde yok.

1. **Player, yakın kamera, F5 + sürükleme:** takılma gitmeli. IPC orbit'te `overlay` CPU p95
   ~2 ms'ye, çağrı p95 ≈ p50'ye inmeli. İnmediyse kök başka (aday: aynı penceredeki parçacık
   billboard'ları / RayFusion probe overlay).
2. **Studio:** ızgara görünürken davranış AYNI kalmalı (ızgara kamerayı izler, kaybolmaz).
   Izgara opaklığı 0 → sonra yeniden açınca ızgara doğru yerde yeniden kurulmalı.
3. ★ Studio'da ızgara açıkken aynı duraklama HÂLÂ var (bu paket yalnız görünmeyen ızgarayı
   kapsar); orada akıcı görünmesi karenin kısa olmasından. Kalıcı çözüm: ızgara tamponunu
   drain'siz emekliye ayırmak — yapılmadı.

AÇIKLANMADI: kullanıcının "yakında bazı ışıkların etkisi değişiyor" gözlemi. Ölçümde yakında
transmission GPU 4,5 → 13,4 ms, ana geçiş 8,8 → 11,8 ms (ekran kaplaması); ışık etkisinin
neden değiştiği aranmadı (aday: RayFusion probe ambient'i, ışık bütçesi).

## Yeni kaynak paketi — gölge önbelleği KARO başına + kameradan bağımsız yerel ışık (2026-11-16, Claude)

Önceki üç paket (ayıklama, statik önbellek, RT gölge beklemesi) kullanıcı testinde PASS
(gölgeler doğru; ışık/obje/kamera hareketi sorunsuz). Kalan şikâyet: **kamera hareket edince
atlas her kare yeniden çiziliyor** (+11 ms GPU, 34 → ~19 kare/s). Bu paket onu hedefler.
Tek dosya: `Viewport/MaterialPreviewShadow.cpp`; header/shader/IPC yok → artımlı, Studio+Player.
Derlenmedi, ölçülmedi.

Üç değişiklik:
- **Yerel ışık (nokta/spot/alan) uzak düzlemi** artık kameraya değil sahne sınır kutusunun
  ışığa en uzak köşesine göre (×1,05, [10, 2000] kelepçe). Görüş matrisi kamerayla değişmez.
- **Scatter olmayan mesh'ler gölgeye kameradan ayıklanmış listeden DEĞİL, tam global instance
  tamponundan çizilir.** Yan etki (düzeltme): kamera dışındaki nesneler artık gölge atıyor.
- **Önbellek karo başına:** render pass `LOAD`, yalnız imzası değişen görüş kendi karosunu
  `vkCmdClearAttachments` ile temizleyip yeniden çizer.
Scatter havuzu olan sahnede kamera hâlâ içeriğin parçasıdır (eski davranış, kazanç yok).

1. **Açılış (hızlı).** Gölgeler var, atlas karolarında çöp/eski gölge yok. Rastgele gölge
   lekeleri = karo temizlenmeden çizilmiş (LOAD + clear sırası).
2. **Ölçüm, kamera hareketi.** IPC `camera.orbit` döngüsü: `shadow_atlas.frames_ran` ≈ 0 ve
   GPU ~28 ms kalmalı (önce 80/81 kare, 39,4 ms). Directional ışık RT gölgeye devredilmemişse
   pan/dolly'de yalnız onun kaskadları çizilir (orbit'te o da sabit).
3. **★ En sinsi: uzak düzlem kırpması.** Yerel ışıktan ÇOK uzaktaki nesnelerin gölgesi
   kaybolmuş ya da ışığa yakın yüzeylerde gölge "akne"si / havada duran gölge (peter-panning)
   artmış mı? Uzak düzlem büyüdü, derinlik hassasiyeti değişti; sapma ve PCF sabitleri aynı.
   Büyük sahnede (>2000 birim) ışığa uzak köşe kelepçeye takılır — eski davranışla aynı sınır.
4. **★ Kamera dışı gölge.** Kameranın arkasındaki bir nesnenin görünen zemine gölgesi artık
   VAR olmalı (önceden yoktu). Yoksa direct-draw yolu boş çiziyor (`instanceCount`/`firstInstance`).
5. **Bayat gölge turu yeniden** (obje/ışık taşı, opaklık, gizle-göster, skin, balon): imza
   yeniden yazıldı, önceki PASS bu sürümü kapsamaz. Dışlama anahtarı aynı: `RT_SHADOW_ATLAS_CACHE=0`.
6. **Scatter'lı sahne** (bitki örtüsü): davranış ve maliyet eskisiyle aynı olmalı.

**Aynı derlemede — Player debug kamerası (`Core/PlayerMain.cpp`, yalnız Player):** F5 açar/kapar
(açılışta KAPALI, Scene Log'a "Debug camera on/off"). Açıkken sol sürükleme = orbit, sağ/orta
sürükleme = pan, tekerlek = yaklaş/uzaklaş; mevcut `rtapi::orbitCamera/panCamera/dollyCamera`
çekirdeğini çağırır (orbit/dolly zaten IPC'de). Kontrol: F5 basmadan fare hiçbir şey yapmamalı;
F5 sonrası üç hareket çalışmalı. Yön ters gelirse işaret tek satır. Kiosk tuş kapısı (Claude D)
geldiğinde F5 kapının arkasında kalmalı — şu an kapı yok, F5 herkese açık.

AÇIK (bu pakette aranmadı): kamera hareketinde kare CPU ≈ GPU görünüyordu; atlas artık
çizilmeyince bekleme sürüyorsa kökü ayrı — IPC temposu etkisini dışlamak için uygulama içi
kamera hareketi gerekir (Player debug kamera önerisi).

## Yeni kaynak paketi — RT gölge hazırlığındaki her-kare GPU beklemesi (2026-11-16, Claude)

Tek dosya: `Viewport/MaterialPreviewRtShadow.cpp` (header/shader/IPC yok → artımlı, Studio+Player).

**CANLI ÖLÇÜM (2026-11-16, Player derlendi; Studio derlemesi sürüyordu, Studio'da ÖLÇÜLMEDİ):**
duran kamera 173 kare/5 s: `shadow_atlas` CPU 24,4 → **3,3 ms**, kare CPU 27,8 → **11,2 ms**,
GPU 28,6 ms, **34,4 kare/s** (GPU sınırı 1/28,6 = 35). IPC `camera.orbit` ile 80 adım (40×+0,5°,
40×−0,5°): atlas 80/81 karede çizildi (önbellek kamera hareketinde doğru biçimde düşüyor),
durunca yeniden 0/104. Madde 2 PASS. ★ AÇIK: orbit sırasında aşama CPU'su 29,5 ms ve kare
CPU ≈ GPU (38,6 / 39,4) — kamera hareketinde başka bir bekleme hâlâ var, kökü aranmadı
(ölçüm IPC temposuyla sürüldü, 18,8 kare/s sürdürülebilir hız kanıtı değil). Madde 1, 3, 4
(görsel) ajan tarafından BAKILMADI; Player'da fare/ışık düzenleme yok, ışık testi Studio'da.

**Kök:** `prepareRtShadowFrame` RT gölge açıkken HER KARE koşulsuz
`drainInteractiveViewportInFlight()` çağırıyordu (ışık/hit/materyal tablolarını yeniden
yükleyip 8 descriptor'ı yeniden yazmak için). Yani CPU her kare önceki karenin GPU'dan
çıkmasını bekliyordu — `shadow_atlas` damgasında görünen ~24 ms buydu (atlas hiç çizilmezken
bile). Artık tablolar + descriptor girdileri imzalanıyor; imza aynıysa yükleme, yazma ve
bekleme atlanır, değiştiyse eskisi gibi drain + yeniden yazım.

1. **Açılış + RT gölge var mı (hızlı).** `viewport.rt_shadow` ready/prepared, güneş gölgesi
   ekranda. Yoksa: imza ilk karede geçerli sayılmış (olmamalı) ya da tablo yüklemesi atlanmış.
2. **Ölçüm, duran kamera, sürekli çizim, capture kapalı.** `shadow_atlas` CPU 24,4 ms →
   ~1 ms altı; kare CPU'su GPU'dan (27 ms) belirgin küçük; kare hızı ~32 → ~37/s (kare artık
   CPU+GPU değil, yalnız GPU ile sınırlı). CPU hâlâ ~24 ms ise imza her kare değişiyor
   (aday: `getRayFusionHitInstances` çıktısında dolgu baytları) — kazanç yok ama hata da yok.
3. **★ En sinsi: gizli yarışın açığa çıkması.** Bu drain, karenin geri kalanındaki başka
   her-kare yüklemeleri de farkında olmadan koruyor olabilirdi. Kamera/nesne/ışık hareket
   ederken titreme, bir kare geride kalan gölge/ışık, validation hatası ya da device-lost
   ara. Karşı kanıt: RT gölge KAPALIYKEN bu drain zaten hiç çalışmıyordu ve o yol sorunsuz.
   Görülürse: suç bu değişiklikte değil, o yüklemenin kendi drain'inin eksikliğinde — ama
   önce burayı geri alıp doğrula.
4. **Değişimde güncelleme.** Işık taşı / materyal opaklığı değiştir / nesne ekle-sil: RT
   gölge aynı karede izlemeli (imza değişir → tablolar yeniden yazılır).

## Yeni kaynak paketi — statik gölge atlası önbelleği (2026-11-16, Claude)

İki dosya, ikisi de `.cpp`: `Viewport/MaterialPreviewShadow.cpp` (imza + erken dönüş) ve
`Backend/VulkanViewportBackend.cpp` (4 satır: aşama damgası). Header/struct/shader/IPC alanı
değişmedi → artımlı derleme, Studio **ve** Player.

**CANLI ÖLÇÜM (2026-11-16, kullanıcı derledi, Player, `oyuncak.rtp`, duran kamera, sürekli
çizim, capture kapalı, 161 kare / 5 s):** `shadow_atlas.frames_ran` **0/161**, atlas GPU 0;
toplam GPU 37,3 → **27,1 ms**; ~32 kare/s (başlangıç 11,4). Kalan GPU: ana geçiş 11,6 ·
transmission 8,3 · RT gölge 5,6 · prepass 1,1. Madde 2 PASS. Madde 3 (bayat gölge) ajan
tarafından BAKILMADI. ★ Atlas hiç çizilmediği halde `shadow_atlas` CPU damgası hâlâ 24,4 ms
ve kare CPU'su (27,8) GPU'yu (27,1) izliyor: bu süre gölge kaydı DEĞİL; damga aralığındaki
`prepareRtShadowFrame` / yansıma hazırlığı / bulut geçişinden birinde GPU beklemesi var.
Kök bulunmadı — sıradaki inceleme adayı (CPU ile GPU'nun üst üste binmesini engelliyor olabilir).

Atlas artık yalnız imzası değişince yeniden çizilir. İmza her kare CPU'da yeniden hesaplanır:
gölge görüşleri (matris+karo), geometri/skin üretim sayaçları, her mesh'in tampon tutamaçları
ve sayıları, **bütün** instance transform+mask'leri, materyal kayıtları, frustum revizyonu ve
(GPU culling açıksa) kamera. Kameraya bağlı olduğu için kazanç **duran kamerada** gelir;
kamera dönerken atlas eskisi gibi her kare çizilir (9,3 ms).

1. **Açılış (hızlı).** Sahne açılır, gölgeler var. Tamamen gölgesiz = ilk kare de atlanmış
   (imza geçerlilik bayrağı hatası).
2. **Ölçüm, duran kamera.** Capture kapalı, sürekli çizim açık, `viewport.frame_timings`:
   `shadow_atlas.frames_ran` ≪ `frames` ve o aşamanın GPU süresi ~0 olmalı; toplam GPU
   37,3 → ~28 ms. `frames_ran == frames` ise imzada her kare değişen bir girdi var (aday:
   frustum revizyonu ya da scatter bütçesi) — önbellek hiç vurmuyor, hata değil ama kazanç yok.
3. **★ En sinsi: BAYAT gölge.** Kamera dururken (a) bir nesneyi taşı/animasyon oynat,
   (b) bir ışığı taşı, (c) bir materyalin opaklığını değiştir, (d) nesne gizle/göster.
   Her birinde gölge **aynı karede** izlemeli. Gölge eski yerinde kalıyorsa o değişiklik
   imzaya girmiyor. Bilinen kör noktalar (RayFusion AS kapısıyla ortak): skin'siz vertex
   tamponunun üretim sayacı artmadan yerinde yeniden yazılması (sculpt/sim deformu), zamanla
   canlanan materyal programı opaklığı, opaklık dokusunun İÇERİĞİ. Balon patlaması ve avatar
   (skin) hareketini özellikle dene.
4. **Dışlama anahtarı.** Şüphede uygulamayı `RT_SHADOW_ATLAS_CACHE=0` ortam değişkeniyle aç:
   gölge düzeliyorsa suç önbellekte, düzelmiyorsa başka yerde.
5. **Studio'da aynı sahne** — orbit sırasında davranış eskisiyle aynı, durunca atlas atlanır.

## Yeni kaynak paketi — gölge atlasında görüş başına ayıklama (2026-11-16, Claude)

Tek dosya: `Viewport/MaterialPreviewShadow.cpp` (header/struct/shader/IPC değişmedi →
artımlı derleme yeter; Studio **ve** Player aynı ortak yolu kullanır, ikisi de derlenmeli).
Önceden her mesh her gölge görüşüne çiziliyordu (nokta ışık = 6 tam sahne geçişi). Artık
mesh'in instance dünya kutularının birleşimi görüşün 5 kırpma düzlemine karşı sınanıyor.
Skinned, scatter havuzu/proxy, kutusu geçersiz veya >4096 instance'lı mesh'ler **her zaman
çizilir**.

**CANLI ÖLÇÜM (2026-11-16, kullanıcı derledi, Player, `oyuncak.rtp`, capture kapalı, 27 kare):**
gölge atlası GPU 33,1 → **9,3 ms**; toplam GPU ~60 → **37,3 ms**; sunulan 11,4 → **22,0 FPS**.
Kalan GPU: ana geçiş 12,1 · gölge atlası 9,3 · transmission 8,7 · RT gölge 5,7 · prepass 1,0.
`shadow_atlas` CPU 38,8 ms okunuyor ama bu aşama damgası `prepareRtShadowFrame` + yansıma
hazırlığı + bulut geçişini de kapsıyor ve kare CPU'su (41 ms) GPU karesini (37 ms) izliyor:
GPU beklemesi adayı, kayıt maliyeti kanıtı değil. Destek: aynı binary Studio'da tek tek
sürülen 10 karede aynı aşama CPU 24,3 → **7,3 ms** (GPU baskısı yokken). O Studio turunun
GPU süreleri (toplam 75,7 ms) Player aynı anda çizdiği için ÇEKİŞMELİ, kıyas değeri yok —
iki uygulama aynı GPU'yu paylaşırken ölçülen süre A/B'ye girmez. Madde 3 ölçümü PASS;
madde 1–2 (gölgelerin görsel doğruluğu) ajan tarafından BAKILMADI, kullanıcı gözü bekliyor.

Referans (aynı gün, `oyuncak.rtp`, 1680×945, 8 gölgeli ışık, eşleşmiş kamera):
gölge atlası GPU 33,1 ms, gölge kaydı CPU 24,3 ms, toplam GPU ~60 ms
(`PLAYER_HEAVY_SCENE_PROFILE_2026_11_16.md`).

1. **Açılış + görüntü (hızlı, bağımsız).** Player'da `oyuncak.rtp` aç. Gölgeler eskisi
   gibi görünmeli. Bozuksa: tamamen gölgesiz kare = düzlem işareti ters (her şey reddediliyor).
2. **★ En sinsi: tek tek eksik gölge.** Işığın görüş konisinin KENARINDAKİ büyük nesnelerin
   (zemin, duvar, uzun masa) gölgesi yerinde mi; kamerayı döndürünce gölge yanıp sönüyor mu?
   Eksikse: o mesh'in `worldBBox`'ı bayat (transform GPU'ya gidip kutu güncellenmemiş) —
   kare makul göründüğü için kimse hata diye bildirmez. Hareketli nesneyi (balon) ışık
   konisine sokup çıkararak da bak.
3. **Ölçüm.** Capture KAPALI, `viewport.frame_timings`: gölge atlası GPU ve gölge kaydı CPU
   33,1 / 24,3 ms'nin belirgin altına inmeli (nokta ışıklarda ~4–6×, spot'ta koni kadar).
   Hiç inmediyse: ışıklar directional/geniş spot ya da mesh'ler ayıklanamayan sınıfta
   (scatter/skinned) — o durumda sıradaki kaldıraç statik atlas önbelleği (aşağıda).
4. **Studio'da aynı sahne** — aynı sayılar; yalnız Player düştüyse eski Studio binary'si.

Bu partide YAPILMAYAN, sıradaki kaldıraç: **statik gölge atlası önbelleği** (ışık+caster
değişmediyse atlası yeniden çizme; sabit kameralı kiosk için 33 ms → ~0). Güvenli olması
için raster içeriğine (instance/vertex/materyal opaklığı/GPU-cull çıktısı) tek bir revizyon
sayacı gerekir; yazan yerler sayılmadan önbellek sessiz bayat gölge üretir. Ayrıca not:
GPU culling açıkken gölge geçişi KAMERA ayıklamalı instance listesini çiziyor (mevcut
davranış, bu partide dokunulmadı) — kamera dışındaki caster gölge atmıyor olabilir.

## Yeni kaynak paketi — C2 dış rig/IK ve mevcut domain vekilleri (2026-11-15, Codex)

Kullanıcı yönlendirmesi: mevcut rig→proxy→domain yolu yeniden kullanıldı; ayrı Jolt
iskeleti/dünyası eklenmedi. `anim.set_external_pose/external_poses/clear_external_pose`
script/IPC ortak çekirdekte; canonical flat skin + mevcut IK/joint limits. Kayıpta ve
düşük güvende domain proxy çözümü kesilir; bind/teleport vb. hareket geçmişini sıfırlar.
Yeni 3 ortak C++ ve Studio'ya 1 Python binding birimi; projeleri reload + artımlı build.
Mevcut yaygın struct/layout veya shader değişikliği yok; eski TAM build ihtiyacını kaldırmaz.
Sayaçlar CPU solve/skin, proxy sayıları; domain tarafında mevcut solver_stamps/perf.list.
Maliyet ölçümü doğruluk kabulü sonrasında; GPU realtime partikül proxy tüketimi/Jolt açık.
Sözleşme/komut/kabul: `PLAYER_EXTERNAL_BODY_RIG.md`; native test kaynağı çalıştırılmadı.
Bu paket henüz kullanıcı tarafından derlenmedi ve açık uygulamalarda C2 testi yapılmadı.

## Güncel kabul — C/B ilk paket: Studio/Player kullanıcı build + IPC PASS (2026-11-15)

İki uygulama kullanıcı tarafından derlendi. Player'da 36 s / 4 ve 10 çocuk tam
oturum PASS: 29/71 eylem, sıfır eylem hatası/olay kaybı; boya %16,43/%32,48.
Player 28 ve Studio 21 hedefli IPC kontrolü; Studio Python binding smoke 9 kontrol PASS.
Native bağımsız assert testi hâlâ çalıştırılmadı; canlı IPC aynı kapsanan çekirdek
davranışlarını gerçek uygulamada doğruladı. Kamera/rig/görsel final ve performans kabulü değil.
RPC/takvim gecikme sıçramaları kaydedildi; A optimizasyonu açık.
Kanıt: `PLAYER_BODY_GAME_ACCEPTANCE_2026_11_15.json`, sözleşme `PLAYER_BODY_GAME_RUNTIME.md`.
Player test odasında attract/pozlar boş/boya sıfır/demo takvimi kapalı; ekran ayarları
geri alındı. Studio boş bırakıldı. Bu kabul turunda C++/shader değişmedi, yeni build gerekmez.

## Önceki kaynak devri — Poz + kural/oturum çekirdeği (2026-11-15, Codex)

KAYNAK HAZIR, kullanıcı build/canlı kabul bekliyor. Yeni Input/Game ve API/IPC
birimleri iki projeye, Python yalnız Studio'ya kayıtlı; projeleri reload edin.
Bu paketten yaygın struct/layout değişikliği veya shader değişikliği yok; kendi
başına artımlı build yeterli, önceki TAM build gereksinimlerini iptal etmez.
`input.set_body_poses/body_poses/clear_body_poses`, `game.session/rules/post_event`
ortak çekirdeğe bağlı. Native test kaynağı `tests/body_pose_game_test.cpp`; assert
kapalı olmadan kullanıcı derleyip çalıştırmalı. Ardından hazır balon fixture'ında
`tools/perception_bridge/session_plan.json` ile 36 s / 4 ve 10 çocuk kabulü alın.
Tam komutlar ve sınırlar: `PLAYER_BODY_GAME_RUNTIME.md`.
Rig/IK, fiziksel kinematik gövdeler, Studio kural paneli, özel balon/final görselleri,
A GPU optimizasyonu ve D kiosk henüz tamamlanmadı. Build/uygulama açma/commit yok.

## Güncel devir — Balon boya demosu: Player build / IPC CANLI PASS (2026-11-15, Codex)

Kullanıcı Player'ı derledi; `rt_player_balloon_paint_ipc.py` kabulü tamamlandı.
Dört gerçek 1024² yüzeyin yüklemede hazırlanması, 12 balon, tekli/üçlü 1800/5400
damla, ömür sonunda bitiş, geçersiz count reddi, rastgele patlamalar/kapsama ve
GPU içi yayın PASS. Yükleme 959.71 ms (demo 352.04 / atlas 317.21 ms); kontrollü
olay giriş dt maks 10–11 ms. Aktif ek ölçüm ~168 fps; idle dahil uzun kayıtta
64–72 ms sıçrama da var. Sıçramasızlık veya <=2 ms boya hedefi kapatılmadı.

**Bu kabul turundan yeni derleme beklenmiyor.** Shader/ABI değişmedi. Studio'da
balon demosunun yeni canlı kabulü ayrıca yapılmadı. Alt bölümlerdeki eski
"Player kabulü sıradaki / yeniden derle" kayıtları tarihsel geliştirme günlüğüdür;
balon fixture'ı için güncel durum bu bölümdür.

Devir, sonraki işler ve tekrar komutu:
[PLAYER_BALLOON_PAINT_DEMO.md](PLAYER_BALLOON_PAINT_DEMO.md),
[PLAYER_GAME_PAINT_ROOM.md](PLAYER_GAME_PAINT_ROOM.md),
[ham kabul](PLAYER_BALLOON_PAINT_ACCEPTANCE_2026_11_15.json).
GPU istatistik/yayın beklemeleri, 64-kaynak/tile sınırı ve ayrı OBB/spread hataları açık.


## GPU ıslak boya (Boyama Odası adım 3 + 4'ün IPC yüzü) — 2026-11-15, Claude

**★ CANLI PASS (Studio, 8/8):** tek damla 4 yüzey (küp 1348 / küre 271 / silindir 428 / düzlem 84
texel); maliyet 16 aktif → 64 tick; akma küp ×1.075, düzlem ×1.0; partiküller cihazdan (`gpu_resident`,
117 cihaz damgası, 0 host olayı, zemin 2940 texel); sıfırlama 0; **EKRAN: material modda 253 kırmızı
piksel**. Yayın: RT render cihazına `device_copies` (GPU içi), material-preview viewport'una (ayrı
VkDevice) host yolu + materyal başı fırça senkronu. Karışım raporu: ıslak−kuru kırmızı +0.0036
(ürün kararı açık). Player kabulü (meshli fixture) sıradaki parti.

**Altıncı canlı deneme:** akma PASS (küp 1.056, düzlem 1.0). Ekran ölçüsü material modda 0 kırmızı
piksel; kullanıcı: backend değişince boya göründü → veri doğru, kare senkronu eksik. Kök: Studio'da
`g_materials_dirty` yalnız TAM sahne senkronunda (`g_needs_optix_sync` ile) servis ediliyor; host yolu
bayrağa güveniyordu. Düzeltme: `publish` bayrak + `hostPublishedMaterials()` döndürür; Studio ve Player yalnız o
materyaller için fırçanın vuruş başı senkronunu çağırır (`updateBackendMaterial(scene, id)`: tek slot +
kirli doku bölgesi, iki backend) — tam tablo yeniden kurulmaz (kullanıcı: maliyetli olmamalı). Karışım maddesi yalnız RAPOR: CPU modelinde tam örten ikinci boya
alttakini ~%1 alıyor (ıslak r 0.209 / kuru 0.206) — "mor" hedefi ürün kararı. **Artımlı derleme + restart.**

**Beşinci canlı deneme:** akma çalışıyor (ıslak ayarla küp 2104→2536 texel, 14 tickte) ama test
saniyeyle ölçüyordu: boştaki Studio ~3 kare/sn tikliyor, boya zamanı neredeyse durmuştu → test artık
TICK sayar. ★ Kullanıcı gözle: boya ne raster ne RT'de görünüyor. İki kök: (1) dört test nesnesi
AYNI materyali (`Default_Cube_Material`) paylaşıyordu — her yüzey dokusunu aynı slota bağladı, son
bağlanan (zemin) hepsinde görünüyordu; artık `paint.wet_add_surface` paylaşılan materyali açık hatayla
reddediyor, test her nesneye kendi materyalini veriyor. (2) Studio'da sim compute RT render cihazında,
material-preview viewport AYRI VkDevice (`render.volume_tables`) → host yolu tek yol, ama host yolu
`g_materials_dirty` kaldırmıyordu, doku hiç yeniden yüklenmiyordu; düzeltildi. Test yeni madde 8:
viewport görüntüsünde kırmızı piksel sayımı (bellekte). **Artımlı derleme + Studio restart.**

**Dördüncü canlı deneme (restart sonrası):** damga güçlendi (küp 576→1480 texel). Elle koşulan
maddeler PASS: partiküller cihazdan (`gpu_resident`, 119 cihaz damgası, 0 host olayı, zemin 2940
texel; bağlam kendiliğinden `simulation`), sıfırlama 0. Akma hâlâ 1.0 — kök gather aramasında: kuru
texel kendi ~0.3 px iziyle arıyordu, `round(p − 0.3) = p` → cephe hiç ilerlemiyordu. Düzeltme: sabit
adımlı yukarı akış araması (1…⌈brush_radius_px·0.6+2⌉, ≤12). Karışım: CPU ıslak fırçasının
`fresh_bias` / `underpaint_mix` (pickup) terimleri damgaya taşındı; yeni alan `pickup_rate`
(IPC/Python/ayarlar). Test iddiası düzeltildi: ıslak kırmızıya mavi, kuru kırmızıya maviden daha
çok kırmızı tutar (önceki "mor" beklentisi modelden değil benden geliyordu). Shader derlendi →
**C++ artımlı derleme (pickup_rate) + Studio yeniden başlat.**

**Üçüncü canlı deneme (11:50):** tek damla 4 yüzeyi boyuyor (küp 576 / küre 117 / silindir 184 /
düzlem 32 texel), boya EKRANDA görünüyor (gözle doğrulandı; soluk). Maliyet PASS: 16 aktif → 64 tick
(≤ 9×aktif), 1.84 s'de `idle`. `publish_mode: host_download` — ıslak boyanın compute bağlamı render
VkDevice'ından farklı (açık iş). Akma FAIL çıktı ama kök damganın ZAYIF birikimi: CPU tohumu (~%53
alfa) kullanılmış, CPU'nun eşlik eden `paintLayerAtUV` vuruşu atlanmıştı; akan boya < 0.5 alfa
taşıdığı için kapsama ölçüsü görmüyordu (ıslak bölge kayıyordu). Düzeltme yalnız shader'da (derlendi,
.spv iki yerde) → **Studio yeniden başlatılmalı**, C++ derlemesi gerekmez.

**İkinci canlı deneme: atlas cihazda (`compute_context: own`), damla tick'e ulaşıyor ama 0 texel.**
Kök: Vulkan `downloadBuffer`'ın anında yolu kayıtlı (gönderilmemiş) dispatch'leri göndermiyor;
istatistikler kernel'ler koşmadan okunuyordu, kendi bağlamımızda kimse senkronlamadığı için iş hiç
koşmadı. Düzeltme: okumadan önce `synchronize()` (yalnız `WetPaintGpu.cpp`, /Zs PASS) → **iki proje
artımlı derleme**. Test düzeltildi: primitif küre/silindir yarıçapı `size/2` (eski boyutlarla damla
onlara değmiyordu); küpün 6 yüzü UV'de aynı kareyi paylaşıyor, bake ilk yüzü (+Z) tutuyor → damla
artık (0, 0, 0.5)'te. Gözle: küpün HER yüzünde aynı yerde kırmızı leke (tek doku) beklenir.

**İlk canlı deneme (Studio derlendi): FAIL `no_vulkan_compute`.** Simülasyon compute bağlamı
yalnız GPU partikülü/domain istediğinde Vulkan; `SceneData::syncSimulationWorld` her
`getSimulationWorld()`'de onu geri CPU'ya çeviriyor, boş sahnede ıslak boya hiç koşmadı.
Düzeltme (yalnız `WetPaintGpu.*`, /Zs PASS): dünya bağlamı Vulkan'daysa o, değilse ıslak boyanın
aynı VkDevice'ta kendi bağlamı; `paint.wet_status.compute_context` = simulation | own. İki bağlam
arasında geçiş atlası yeniden kurar (boya silinir). Testte `--keep` bayrağı: nesneler ve lekeler
sahnede kalır (bayraksız betik her şeyi siler — PASS olsa da ekranda leke görünmez).
★ Düzeltme Player derlenirken yazıldı ve `WetPaintGpu.h` sınıf düzeni değişti: o derleme bitince
**iki projeyi bir kez daha artımlı derle** (karışık .obj kalmasın).
Studio fırçası hâlâ CPU'da — tasarım gereği (§5.3 son madde, ayrı parti).

CPU ıslak boya modeli (`noteWetDab` / `tickWetPaint`) çok yüzeyli bir GPU atlasına taşındı:
`shaders/paint_wet.comp` (tek shader, mod anahtarlı: damga / tick / çözümle / döndür /
tüket / kapsama / sıfırla), `Paint/WetPaintGpu.*` (bake, cihaz tamponları, kare başına
tick, yayın), ortak sim saatinden tikliyor (Studio + Player). Partikül çarpışma olayları
cihazda doğrudan okunuyor (indirme yok). Sonuç, yüzeyin materyal dokusuna cihaz içi kopyayla
gidiyor. IPC `paint.wet_*` (11 metot), Python `rt.paint.wet_*`, yetki, ajan tarifi, Player
izin listesi. Tasarım ve bilinçli farklar: `PLAYER_GAME_PAINT_ROOM.md` §5.3.
Shader derlendi (`paint_wet.spv` iki yerde). /Zs: yeni 4 dosya + değişen 9 dosya PASS (Studio
ve Player bayraklarıyla). IPC denetimi + Player proje üreticisi PASS.

**Derleme: iki projeyi reload (yeni .cpp'ler) → artımlı build, ama UZUN.**
`ParticleSimulation.h` yeni bir struct (`ParticleDeviceEventView`) ve iki metot bildirimi aldı;
**mevcut struct düzeni DEĞİŞMEDİ**, yani eski .obj tehlikesi yok, ama scene_data.h'yi içeren
her TU yeniden derlenir. `VulkanBackend.h` (+1 metot, `class Texture;`) ve
`RuntimeSceneViewer.h` (+1 sayaç, yalnız birkaç TU) de değişti.

Testler BOŞ Studio sahnesinde: `python scripts/test/rt_wet_paint_gpu_ipc.py`. Betik kendi
nesnelerini kurup siler. Bu sıra izole edilmiş maddeler; betik de aynı sırayla durur:

1. **Bake.** Dört primitif (küp, küre, silindir, 4×4 düzlem) yüzey olur; her birinde
   `valid_texels > 0`.
   Bozuksa: UV'ler texel'e düşmüyor (uv_set) ya da üçgen toplama boş döndü.
2. **Tek damla, dört yüzey.** (0.5, 0, 0)'daki tek damla dört yüzeyi birden boyar;
   `publish_mode` = `device_copy` beklenir.
   - `host_download` çıkarsa render backend'i compute ile AYNI VkDevice'ta değil demektir.
     Çalışır ama her kirli dikdörtgen hosttan geçer (`publish_bytes_downloaded`). Bildir.
   - ★ Sinsi durum: dört yüzeyde de `covered > 0` ama viewport'ta boya YOK. Kapsama atlastan
     okunuyor, ekran ise dokudan; ikisi ayrışabilir. **Gözle bak:** viewport'ta küpün +X
     yüzünde kırmızı bir leke görünmeli. Yoksa sebep şunlardan biri: doku cihazda
     yerleşik değil (`copyDeviceBufferToTexture` 0 dönüyor, kopya sayacı artmaz), format
     RGBA8 değil (sıkıştırılmış doku, -1 → host yolu) ya da materyal başka bir doku
     okuyor (bind olmadı).
3. **Maliyet ıslak alanı izler.** Tek küçük damlada `ticked_tiles` atlasın %25'inden az.
   Kuruyunca `status: idle` ve `active_tiles` 0 olmalı (boşta dispatch ve indirme yok).
   Bozuksa: döşeme bayrakları dönmüyor, ya da "ıslak" eşiği hiç altına inmiyor (lifetime).
4. **Akma.** Kuruduktan sonra küpün dikey yüzünde boyalı alan büyür (`cube_growth > 1.05`),
   yatay düzlemde ondan az büyür.
   Bozuksa: akış alanı boş (bake'te `computeTrianglePixelFlow` false), ya da gather
   taşımasının hedef doğrulaması hiç tutmuyor.
5. **Islakken karışım.** Aynı noktaya önce kırmızı, sonra mavi: ortalama renkte r ve b
   ikisi de > 0.2, g ikisinden küçük olmalı.
6. **Partiküller cihazdan.** `PaintRain` sistemi (auto, die, olay kaydı açık) düzleme
   yağıyor. Beklenen: `gpu_resident`, `device_event_dispatches > 0`,
   `host_events_applied == 0` ve düzlem boyanıyor.
   - ★ Sinsi durum: düzlem boyanıyor ama `host_events_applied > 0`. O zaman sistem CPU
     referansında koşuyordur: olaylar hosttan geçiyor, boya yine de doğru görünür.
     `device_event_dispatches` sayacına bak.
7. **Sıfırlama.** `paint.wet_reset` sonrası kapsama 0.

**Ölç ve rapora yaz (geçer/kalır değil):**
- Islakken `last_tick_ms`: tick + istatistik indirmesi, host duvar saati.
- `viewport.frame_timings` ile boyalı kare ve boşta kare karşılaştırması. Yayın, her kopyadan
  önce `waitIdle` yapıyor (`updateTexture2DRegion` ile aynı kural). Bu, kare boru hattını
  boyanın aktif olduğu karelerde durdurur. Pahalıysa sıradaki iş yayının senkron olmayan bir
  bariyere taşınması.

**Bilinen sınırlar (hata değil, notta yazılı):**
- Boya oturum durumudur: projeyle kaydedilmez.
- Aynı hedefe yapılan bir CPU boya işlemi (Studio fırçası, `paint.fill_layer`) kendi
  katmanlarından composite edip GPU boyasının üstüne yazar.
- Uç birikimi (bead) ve yükseklik kanalı GPU'da yok. CPU'da da yalnız `write_height_mask`
  açıkken çalışıyorlar.
- Akış bake'i yalnız yerçekimini kullanıyor; kuvvet alanları statik oda için yok.
- Bir döşemeye karede en fazla 64 kaynak düşer; fazlası `sources_dropped` sayacına yazılır.
- Yüzey varken partikül olaylarını ıslak boya tüketir (`splat.consume_particles`). Bu
  durumda `particle.collision_events` boş döner.
- Player: `paint.wet_*` izin listesinde ama Player'da primitif eklenemiyor. Player kabulü,
  meshli bir fixture ile ayrı bir test olacak (sıradaki parti).

## Player sahne sesi (ortak ses tick'i) — 2026-11-15, Claude

Ses sistemi Player'da derleniyordu ama hiç ilerletilmiyordu: tick'i (transport + sahne
senkronu + kamerada dinleyici + hava tepkileri + akustik + gök gürültüsü) Audio panelindeydi
(`AudioUI::updateListener`, Player derlemez). Gövde `audio::tickRuntime`
(`Scene/AudioRuntime.cpp`, yeni) oldu; panel yalnız timeline zamanını verip çağırır. Player
`advanceRuntime`'da aynı fonksiyonu oyun saatiyle çağırır (proje açılışından itibaren,
oynatılıyor). IPC: `runtime.simulation.audio_time_s / audio_ticks`; Player izin listesine
salt okunur `audio.status / list / source_runtime / mixer_get`. Ses cihazı ilk sesle tembel
açılır (`ensureDevice`). /Zs 5/5, include audit + Player proje üreticisi PASS.

**İki projeyi reload (yeni .cpp) → artımlı build** (Studio: AudioUI.cpp; header yalnız yeni
`Audio/AudioRuntime.h`, `Core/RuntimeSceneViewer.h`).

1. Studio: sesli bir sahnede timeline oynatınca ses eskisi gibi; dinleyici kamerayı izler.
   Bozuksa: `AudioUI::updateListener` artık yalnız `audio::tickRuntime` çağırıyor.
2. `python scripts/test/rt_player_audio_ipc.py` (Player açık; varsayılan EchoRoomsLab, 7 kaynak)
   → PASS: `audio_ticks` artar, transport oynar ve zamanı duvar saatini izler, kaynak pozları
   çözülür. Raporda `device` ve canlı `voices` bilgi amaçlı.
   - ★ Sinsi: transport ilerliyor, pozlar çözülüyor ama `device` "not_initialized" ve `voices`
     0 kalıyorsa kaynaklar oynatılmıyor demektir (kaynak ayarı ya da `ensureDevice` hatası):
     sayaçlar sağlıklı görünür, yalnız ses yoktur — KULAKLA dinle.
3. Regresyon: `rt_player_bootstrap_ipc.py` (metot listesi +4) ve diğer `rt_player_*`.

**01:14/01:17 CANLI:** ses tick'i Player'da çalışıyor — transport oynuyor ve zamanı duvar saatini
izliyor (3.05 s'de 3.03 s), 7 kaynağın pozu çözülüyor, `audio_ticks` 60→233. ★ Tam işaretlenen
sinsi durum çıktı: `device: not_initialized`, `voices: 0`. **Hata DEĞİL:** EchoRoomsLab'ın 7
kaynağı `driver: manual`, sahnede klip yok; ses cihazı yalnız klip varken ya da bir kaynak
çalınınca açılıyor. Studio'da AYNI sahne: timeline play → `not_initialized`; `audio.play` →
`ready`. Player'ın sesi gerçekten çıkarabildiğini kanıtlamak için Player IPC'ye
`audio.play / play_all / stop / stop_all` eklendi (oyun kurallarının eylemleri de bunlar) ve
test artık bir kaynağı Player'ın kendi IPC'siyle çalıp `device == ready` istiyor. /Zs PASS.
**Yalnız Player artımlı build** (RtPlayerIpcPolicy.h). Diğer 6 Player regresyonu 01:17'de PASS;
`rt_player_scene_ipc` / bootstrap yeni metot listesini beklediği için bu derlemeden sonra.
Son kontrol KULAKLA: test kaynak çalarken (1 s) slime sesi duyulmalı.
**01:23 Player: CANLI PASS** — Player IPC'siyle `audio.play` → cihaz `ready`, kaynak `playing`
(kazanç 1.0, uzamsal, yazılmış dünya konumu); `rt_player_scene_ipc` yeni metot listesiyle PASS.
Bootstrap Studio kapalı olduğu için koşulmadı. **Parti kapandı** (kulak kontrolü kullanıcıda).

## ★ TAM DERLEME — `ParticleSimulation.h` düzeni değişti (Boyama Odası adım 2) — 2026-11-15, Claude

**CANLI (00:33 Studio / 00:44 Player tam derleme):** GPU yolu PASS — `gpu_resident`, 2 collider
cihazda, 4214 olay (zemin 4127 / küre 87), konum hatası zemin 4.5e-10 m / küre 7.3e-8 m, normaller
≥ 0.99999987. **GPU ve CPU olay sayısı BİREBİR aynı (oran 1.0).** Bounce: 6000 canlının en alçağı
y = 0.02 (tam yarıçap), olay 0. Player regresyonu 8/8 PASS. Tek FAIL CPU referansındaydı ve
**benim olay normali tahminimdi, fizik değil**: CPU'nun swept küre testi partikülü yolunun
girdiği yere koyuyor, itme yönü radyal değil (0.11). Düzeltme kaynakta (yalnız
`ParticleSimulation.cpp`, /Zs PASS): CPU olay normali collider geometrisinden (düzlem, küre,
kapsül, AABB/OBB yüzü; SDF/konveks/BVH itme yönü). **00:52/00:55 artımlı build: TAM PASS**
(atlama seçeneği olmadan): CPU küre normali 0.11 → 0.99999993; GPU/CPU 4214 = 4214; Player
partikül regresyonu PASS. **Parti kapandı.** Test temizliği de düzeltildi (`collider.*` sistem yoksa
varsayılan sistem yaratıyor: önce collider'lar, sonra sistemler silinir).

**Artımlı derleme YETMEZ:** `ParticleSimulation.h` (scene_data.h içerir → neredeyse her TU)
yeni alan/üye aldı (`ParticlePhysicsSettings`, `ParticleResidentBuffers`,
`ParticleSimulationStats`, `ParticleSimulationSystem`). Eski .obj kalırsa erişim ihlali
kod hatası sanılır. **İki projeyi reload (yeni .cpp) → Rebuild (ya da Clean + Build).**
Önceki bulut partisi (canlı PASS) commit edilmedi; bu partiyle birlikte gidecek.

Ne yazıldı:
- **GPU partikül çarpışması:** yeni kernel `sim_particle_collide` (14 bağlama, 48 B push;
  `.spv` iki yerde, `compile_sim_shaders.bat`'a sim_particle_ballistic/spawn/collide eklendi —
  daha önce HİÇBİR betikte yoktu). Balistikten sonra; önceki konum `p − v·dt`. CPU
  `applyColliders` ile birebir: düzlem, küre, kapsül, AABB, OBB, pişmiş mesh SDF (+ sistem
  zemin düzlemi). Swept testler yalnız CPU'da. Konveks/mesh BVH → CPU (`gpu_status`
  `host_consumer_collider_mode_cpu_only`), pişmemiş SDF → `host_consumer_collider_sdf_not_cooked`.
- ★ **Düzelen sessiz hata:** device-resident yol sistemin zemin düzlemini (`setCollisionPlane`)
  hiç uygulamıyordu; artık uygular.
- **Çarpışma davranışı (sistem düzeyi):** `collision_response` bounce|die,
  `record_collision_events`, `collision_event_capacity`. Die: GPU'da yalnız cihazda öldürür;
  host yuvayı ömrü bitene dek tutar (erken yeniden kullanmaz), kinematik indirmede ölümler
  host'a gelir. CPU referansına da aynı davranış eklendi.
- **Olaylar:** 48 B kayıt (konum, normal, collider, profil, hız, yaş oranı), sınırlı tampon,
  tüketilene dek birikir, taşma sayılır. Kapasite büyümesinde KAYBOLMAZ.
- **4 katman:** `Api/RtApiParticleCollision.h` (RtApi.h'ye dokunmadan), IPC
  `particle.get_collision / set_collision / collision_events`, Python `rt.particle.*`, panel
  (Partikül fizik: On Collision, Record Contact Events, Event Capacity + son adım satırı),
  ProjectManager + SceneSerializer kaydı/yüklemesi.

/Zs 12/12 PASS; descriptor üreticisi, capability audit, Player include audit + proje
üreticisi PASS; kernel bağlama denetiminde partikül uyumsuzluğu yok.

Sıra (bağımsız ve hızlı önce):
1. Studio açılır, mevcut partikül sahnesi eskisi gibi oynar (Bounce varsayılan, olay yok).
   Bozuksa: `applyColliders` lambda dönüşümü (CPU yolu her çarpışmalı sistemi etkiler).
2. Panel: Partikül fiziğinde "On Collision", "Record Contact Events" görünür, değişir,
   proje kaydedip açınca korunur.
3. `python scripts/test/rt_particle_gpu_collision_ipc.py` (Studio açık, BOŞ sahne) → PASS:
   auto `gpu_resident` + `collision_stage_on_device` + 2 GPU collider; olaylar yüzeyde
   (zemin y = 0.02 ± 1e-3, normal yukarı; küre |p−c| = 0.32 ± 1e-3, normal radyal);
   GPU/CPU olay sayısı oranı 0.8–1.25; bounce'ta zeminin altında partikül yok.
   - auto `host_consumer_*` → uygunluk reddi (hangi collider modu, gerekçe yazıyor).
   - `gpu_resident` ama `collision_stage_on_device` false → kernel çağrılmadı
     (`deviceCollisionStageNeeded`) ya da `sim_particle_collide.spv` bulunamadı (log).
   - ★ Sinsi: GPU olayları var ama konumları yüzeyde DEĞİL (dy ≈ v·dt) → önceki konum
     yeniden kurulumu ya da paket satır/sütun düzeni (`packRows`) yanlış; görüntüde
     partiküller "yaklaşık" doğru yerde durur, kimse fark etmez.
   - GPU/CPU oranı < 0.8 → GPU'da tünelleme (swept test yok); hız/dt'yi raporla.
4. Player: partikül sahnesi regresyonu `rt_player_particles_ipc.py` PASS (çarpıştırıcısız
   sistem değişmedi), diğer `rt_player_*` regresyonları.
5. Bilinen açık (kural 1): sistemin zemin düzlemini (`setCollisionPlane`) IPC'den açan metot yok;
   GPU düzeltmesi bu yüzden yalnız panel/proje yüklemesiyle sınanabiliyor → Sonra listesi.

## Player bulut / yerel hava gönderimi — 2026-11-14, Claude

Studio Main'deki bulut bloğu (`setCloudTime` + `syncLocalWeather` + şimşek çarpma
çözücüsü + iki Vulkan adaptörüne `setCloudState`) `AppRuntime::pushCloudState`
(VulkanSceneSync) oldu; yayın durumu çağıranda (`CloudPushState`), hedef backend
kümesi değişince de yeniden yayınlar (Player'ın tembel/yeniden kurulan aygıtı).
Studio: bulut zamanı timeline karesinden, RT uyandırma Main'de kaldı. Player: bulut
zamanı OYUN SAATİNDEN, projenin kayıtlı bulut zamanından devam eder; yalnız bir bulut
katmanı açıkken ilerler (her yayın birikimi sıfırlar — bulutsuz sahne boşta kalabilmeli).
Player ayrıca `g_world_dirty / g_viewport_world_dirty`'yi Studio'nun aynı
`syncVulkanWorld` çağrısıyla servis eder. IPC `runtime.simulation.clouds_enabled /
cloud_time_s / cloud_pushes / world_syncs`. Önceki partinin bekleyen Player değişikliği
(ölçüm modu capture'sız çizer) de bu derlemede.

**CANLI PASS (23:51 Player / 23:57 Studio):** bulutlu sahnede `cloud_time_s` 2.01→6.13 (4 s), `cloud_pushes` 25→91, gökyüzü bulutsuzdan farkı 18.5, 4 s değişimi 5.4; bulutsuz sahnede yayın artmadı. Görüntü gözle: RayFusion bulut tabakası + ufuk, kenar rüzgârla kayıyor. Regresyon 6/6 PASS (bootstrap Studio kapalı olduğu için koşulmadı). İlk capture'sız ekran ölçümü (malzeme fixture'ı): 1680x945 323 FPS / 2.29 ms; tam ekran 1920x1080 251 FPS / 3.03 ms, **upload 1.28 ms = karenin %42'si**; ölçek 0.5 ile maliyet ~3.3× düşer; 60 Hz pacing 57.8. **Parti kapandı.** /Zs 6/6 PASS (Player 3, Studio 3), include audit PASS. **İki projeyi artımlı build**
(header: `Core/VulkanSceneSync.h`, `Core/RuntimeSceneViewer.h`; yaygın struct değişmedi).
Studio'yu fixture için ajan açtı (yalnız test sahnesi), kapatmak güvenli.

1. Studio: bulutlu bir sahnede timeline oynatınca bulutlar eskisi gibi akar; Rendered
   (RT) modda bulut zamanı değişince render uyanır. Bozuksa: Main'deki çağrı bloğu.
2. `python scripts/test/rt_player_clouds_ipc.py` (fixture hazır: `docs/dev/.captures/
   player-clouds/clouds.rtp`, "broken" preset, 60 m/s rüzgâr) → PASS:
   bulutsuz sahnede `cloud_pushes` artmaz; bulutlu sahnede `cloud_time_s` 4 s'de > 2 artar,
   `cloud_pushes` > 10 artar, gökyüzü bulutsuzdan farklı, 4 s'de değişir.
   - `cloud_pushes` artıyor ama gökyüzü bulutsuzla aynı → RayFusion bulut geçişi Player'ın
     raster karesinde çalışmıyor (yayın değil çizim sorunu).
   - ★ Sinsi: gökyüzü bulutlu ama 4 s'de DEĞİŞMİYOR → zaman yayılıyor ama shader eski
     zamanı okuyor; görüntü "doğru" görünür, yalnız animasyon ölüdür.
   - Bulutsuz sahnede `cloud_pushes` artıyorsa sahne hiç boşta kalmaz (sürekli çizim).
3. Regresyon: `rt_player_display_ipc.py` (artık capture açmaz), volume/scatter/particles/
   materials/sim_clock/scene/bootstrap PASS.

## Player FPS tanımı düzeltmesi + ölçüm modu — 2026-11-14, Claude

Kullanıcı (21:02 Player, iç oda `bedroom_new.rtp`, 727 mesh / 4.2 M üçgen) FPS'in
gerçek çizim yerine uyku/boşta durumunu saydığını yakaladı. Doğru: oturmuş sahnede
kare çizilmiyordu, "FPS" yalnız HUD yenilemesini, "kare süresi" sunum aralığını
(1011 ms) sayıyordu. Ayrıca `continuous` modu backend'i zorlamıyordu: birikmiş raster
karede `renderProgressive` kayıt yapmadan dönüyor (0 kare, cpu_record 0).

Düzeltme (yalnız Player): maliyet YALNIZ çizilen karelerden (`frame_cost_ms` =
render+upload, `capacity_fps`), backend ayrıntısı (frame / CPU record / submit /
slot wait; slot wait > 1 ms → "GPU-bound"), boşta pencerede son maliyet korunur
(`cost_age_s`); oranlar ayrı (`rendered_fps`, `presented_fps`). `continuous` artık
`forceRedraw` (resetAccumulation) ile gerçek kare çizdirir. HUD çıktı çözünürlüğünü de
gösterir (render 1440x757 = output 1920x1009 × 0.75; kamera değil, F2 ölçeği).
Player IPC'ye `viewport.frame_timings` / `reset_frame_timings`.

**23:26 Player CANLI:** FPS düzeltmesi tuttu. Basit sahne (12 üçgen, 1920x1009): sürekli modda 202 çizilen FPS, maliyet 4.54 ms (render 3.19 + upload 1.35 → upload karenin ~%30'u), GPU 0.85 ms. İç oda (727 mesh / 4.2 M üçgen, 1920x1009): 29.6 FPS, **GPU 24.5 ms = main pass 17.3 + gölge atlası 4.3 + depth prepass 1.2 + RT gölge 1.0**, CPU kaydı 3.2 ms. ★ DÜZELTME: aşağıdaki "~44 ms açıklanamayan" bir boşluk DEĞİLDİ — ölçümde açık olan `viewport.capture` senkron sunuma geçiriyor (`synchronous_present: True`), kare süresi GPU beklemesini içeriyordu. Ek düzeltme (yalnız Player build): sürekli mod odak dışında capture'sız çizer; `rt_player_display_ipc.py` capture açmaz.

★ Canlı ölçüm (21:02 build, kamera orbit ile gerçek kareler, 1440x757): CPU komut
kaydı medyan **3.25 ms** (ilk okumadaki 40 ms yükleme sonrası ısınmaydı), ama backend
karesi medyan **48 ms**; kayıt+submit+geri okuma+slot bekleme ≈ 4 ms → **~44 ms
alt kalemlerin dışında**. Sıradaki ölçüm `viewport.frame_timings` ile bu boşluğu bölmek.

/Zs 3/3 PASS. **Yalnız Player artımlı build.**
1. HUD: oturmuş sahnede "drawn 0 FPS: scene settled", maliyet "(last drawn)"; F4 ile
   drawn FPS > 0 ve maliyet satırları dolar.
2. `python scripts/test/rt_player_display_ipc.py --project E:/rt_k/bedroom_new.rtp`.
3. `viewport.frame_timings` (Player IPC) ile 44 ms'lik boşluğun aşaması.

## Player ekran yüzeyi + kare zamanlaması (FPS) — 2026-11-14, Claude

Player'a debug ve kiosk için ölçüm ve ekran denetimi:
- **Kare zamanlaması:** bir saniyelik pencere — sunulan FPS, saniyedeki render, kare
  süresi ort/maks, aşama dağılımı: `advance` (sim/runtime), `render` (GPU kare + CPU
  yüzeye geri okuma), `upload` (SDL doku yükleme), `present` (HUD + sunum). HUD'da ve
  `agent.get_state_summary.runtime.loop` ile `player.display` içinde.
- **Ekran:** kenarlıksız tam ekran (F11 / `--fullscreen` / IPC), pencere boyutu
  (`--width/--height`), render ölçeği 0.25..1 (render çözünürlüğü = çıktı × ölçek; SDL
  pencereye gerer), pacing 60 Hz ↔ sınırsız (F3 / `--pacing 0`), sürekli render (F4 /
  `--continuous`: oturmuş sahneyi de her iterasyonda çizer, ölçüm için; arka planda
  20 Hz uykuyu da kapatır, yoksa sonuç sessizce 20'ye takılır), HUD (F1 / `--no-hud`),
  Esc tam ekrandan çıkar.
- **IPC:** `player.display`, `player.set_display` (Render yetkisi; Python yok çünkü
  Player'da Python yok; Studio'da açık hata döner). Tüm anahtarlar uygulanmadan önce
  doğrulanır, yanlış tip `checkParams` ile dispatch öncesi reddedilir.
- Pacing artık Player'da (`FrameCycle::waitForNextIteration` Player'da çağrılmıyor;
  ortak AppFrame.h'ye dokunulmadı).

/Zs 13/13 PASS, descriptor üreticisi + capability audit + Player proje üreticisi PASS.
**İki projeyi artımlı build** (Studio: RtIpc/Security/descriptors + RuntimeProjectHost.h
içerenler; davranışı değişmez, `player.*` hata döner).

1. Player açılır; HUD'da FPS satırı ~1 s sonra dolar; F11 tam ekran ↔ pencere, Esc
   çıkış, F2 ölçek 1→0.75→0.5, F3 pacing, F4 sürekli. Bozuksa: tuşlar ImGui klavye
   odağındayken yutulur (`WantCaptureKeyboard`) — HUD'a tıklamadan dene.
2. `python scripts/test/rt_player_display_ipc.py` (Player açık; Studio açıksa onun
   açık hatası da sınanır) → PASS + ölçek/tam ekran tablosu.
   ★ Tabloyu oku: `render` sütunu GPU + geri okuma. Ölçek 0.5'te render süresi ~4×
   düşmüyorsa maliyet piksel değil sabit (sync/geri okuma bekleme) demektir — sunum
   yolunu (swapchain) önceliklendirmenin ölçüsü bu.
3. Regresyon: `rt_player_bootstrap_ipc.py` (metot listesi +2), volume/scatter/particles/
   materials/sim_clock/scene PASS.

## Player volume (VDB / gaz) + `scene.import_vdb` — 2026-11-14, Claude

İki iş tek partide:
1. **Kural-1 açığı kapandı:** VDB içe aktarma yalnız panelde, dosya diyaloğuna bağlıydı
   (hiçbir test sahneye VDB koyamıyordu). Gövde `rtapi::importVdbVolume`
   (`Api/RtApiVdbVolume.h/.cpp`, RtApi.h'ye dokunulmadı = tam derleme yok) oldu; panel
   onu çağırıyor. IPC `scene.import_vdb {path, orientation}`, Python `scene.import_vdb`,
   yetki `FilesRead|SceneWrite` + uzak yol denetimi, overlay tarifi; descriptor üreticisi
   ve `audit_ipc_capabilities.py` (Python aynası dahil) OK.
2. **Player volume:** `SceneUI::syncVDBVolumesToGPU` gövdesi `AppRuntime::publishVolumes`
   (VulkanSceneSync) oldu. Studio render + viewport, Player yalnız tek viewport aygıtı
   (Player'daki "editörde yok" saplaması artık ortak fonksiyonu çağırıyor; ortak API'nin
   `RtApiFluid`/`RtVolumeMaterialBindings` çağrıları da Player'da çalışır hale geldi).
   Ret kalktı; volume Hittable'ları düz-mesh denetiminden muaf; hazırlıkta ve
   `g_gas_volumes_dirty` servisinde aynı çağrılar. IPC `runtime.vdb_volumes / gas_volumes`.

**CANLI PASS (20:33 Player / 20:38 Studio):** `scene.import_vdb` Studio'da çalıştı (101 km bulut → 5 birim, GPU upload); Player `vdb_volumes=1`, volume'lu/volume'suz fark 944 piksel (bulut ölçekle küçük, ortalama fark 0.43 bu yüzden), görüntüde gölgeli bulut kümecikleri gözle doğru. Regresyon 6/6 PASS (bootstrap dahil). Madde 1 (panelden elle içe aktarma) kullanıcı gözü bekliyor. **Parti kapandı.** /Zs 12/12 PASS, include audit + üretici PASS. Ret örneği `rt_player_scene_ipc.py`:
`vdb_volumes` → `fracture_recipes`. **İki projeyi reload (yeni .cpp) → ikisini artımlı build.**
Header: yalnız yeni `RtApiVdbVolume.h`, `VulkanSceneSync.h`.

1. Studio panelinden VDB içe aktarma eskisi gibi (diyalog, toast, seçim, ölçek/hizalama).
   Bozuksa: `SceneUI::importVDBVolume` artık yalnız diyalog + API + seçim.
2. Fixture: `python scripts/test/rt_player_volume_ipc.py --make-fixture` (Studio açık, başlangıç
   sahnesi) → `scene.import_vdb` bilgi döner, `volume.rtp` kaydedilir.
3. Player: `python scripts/test/rt_player_volume_ipc.py` → PASS: `vdb_volumes=1`, aynı kamerayla
   volume'lu/volume'suz kare farkı > 500 piksel (çeyrek çözünürlük).
   - `vdb_volumes=0` → ProjectManager VDB'yi Player'da yüklemedi (dosya yolu / GPU upload).
   - 1 ama fark yok → tablo yayınlandı ama MaterialPreview volume dalı çizmiyor:
     ★ sinsi: `[MPVolume] volumeCount=0 bound=0 -> SKIPPED` logu "sahnede gaz yok" diye
     okunmuştu (2026-09-04); Player logunda bu satıra bak.
4. Regresyon: scatter / particles / materials / sim_clock / scene (`--player-only`) PASS.
5. Bilinen sınır: animasyonlu VDB dizisi (timeline `updateFromTimeline`) Player'da sürülmüyor.

## Player scatter / instance kabulü — 2026-11-14, Claude

Ölçüm: Vulkan aygıtında yükleyici scatter gruplarını sahneye açmıyor (native
InstanceGroup, `gpu_dirty`), raster build (`VulkanViewportBackend::buildRasterGeometry`)
grupları `InstanceManager`'dan doğrudan okuyor; Player'ın çağırdığı `buildRasterScene`
bunun içinde. Yani taşınacak gövde yoktu; engel yalnız iki ret satırıydı. Kaldırıldı;
çalışma sırasında scatter değişimi Studio'nun aynı `refreshRasterScatterInstances`
çağrısıyla servis ediliyor; IPC `runtime.scatter_groups / scatter_instances`.
Taban: eski Player fixture'ı `unsupported_in_player_scene: instances` ile reddediyor
(doğrulandı). `rt_player_scene_ipc.py` ret örneği `instances` → `vdb_volumes`.
**CANLI PASS (20:11 Player):** 1 grup / 150 instance, 622 sarı piksel, 174k görünür üçgen / 6 draw call (instancing); görüntü gözle doğru (gölgeli, ölçek değişken). Regresyon 4/4 PASS. **Parti kapandı.** /Zs 2/2 PASS.

1. `python scripts/test/rt_player_scatter_ipc.py` → PASS: 1 grup, ≥75 instance,
   görüntüde sarı çakıl > 40 piksel.
   - `scatter_instances` 0 → binary sidecar'daki instance'lar okunmadı
     (`deserializeBinaryInstances`).
   - instance var, sarı yok → raster build grupları atladı: ★ sinsi aday, kaynak mesh
     (`Pebble`, y=-20) çizilip kopyalar çizilmiyorsa görüntüde HİÇBİR şey yanlış görünmez.
2. Regresyon: particles / materials / sim_clock / scene (`--player-only`, yeni ret örneği) PASS.
3. Bilinen sınır: scatter rüzgârı (`InstanceManager::updateWind`) Studio'da timeline'a bağlı;
   Player'da sürülmüyor — sonraki animasyon/tick adımıyla.

## Player partikül kabulü — 2026-11-14, Claude

Player artık `particle_simulation` içeren sahneyi reddetmiyor. Partiküller ortak sim
saatinde adımlanır, Studio'nun aynı billboard yoluyla çizilir:
`SceneUI::uploadParticleBillboards` gövdesi `AppRuntime::uploadParticleBillboards`
(VulkanSceneSync) oldu; SceneUI yalnız çağırır. `UI/ParticleBillboardBuilder.cpp`
UI'a bağlı değildi, yalnız klasörü yüzünden Player dışındaydı → manifestle dahil
(dosya değişmedi, Codex'in dosyası). Partikül compute bağlamı Player aygıtında zaten
yayınlanıyor (`VulkanDevice::publishAsSimComputeContext`). /Zs 5/5 (builder Player
bayraklarıyla dahil), include audit + üretici --check PASS. Taban: eski Player fixture'ı
`unsupported_in_player_scene: particle_simulation` ile reddediyor (doğrulandı).

**CANLI PASS (19:51 Player / 19:55 Studio):** link temiz; 1183 canlı partikül, `particle_draws=1` (tek pulled çizim = GPU-resident yol, host kopyası yok), 426 magenta piksel; görüntü gözle doğru. Regresyon 3/3 PASS. Partiküller zeminden geçiyor (fixture'da çarpıştırıcı yok; adım 2'nin konusu). **Parti kapandı.** İlk not: **İki projeyi reload (Player'a yeni dosya) → ikisini artımlı build.** Header:
yalnız `Core/VulkanSceneSync.h`, `Core/RuntimeSceneViewer.h` (az TU).

1. **Player linki.** Builder ilk kez Player'da: LNK2019 çıkarsa builder'ın uzandığı
   bir sembol Studio-yalnız dosyada demektir (adını söyle, host/taşıma kararı).
2. Studio: partikül sahnesinde Solid/Material görünümde billboard'lar eskisi gibi;
   Debug görünümünde (display mode 1) billboard yok, nokta overlay var.
3. `python scripts/test/rt_player_particles_ipc.py` (Player açık) → PASS: `particles_alive > 200`,
   `particle_draws > 0`, görüntüde magenta piksel > 50.
   - alive 0 → emitter Player'da yüklenmedi ya da saat partikül sistemini adımlamıyor.
   - alive var, draws 0 → sistem `emitter_only` ya da `visible=false` kaydedilmiş.
   - draws var, magenta yok → billboard pass Player'ın MaterialPreview karesinde
     çizilmiyor. ★ Sinsi: GPU-resident sistemde "pulled" çizim aygıt eşleşmesine bağlı;
     eşleşmezse CPU quad yoluna düşer — çalışır, ama `particle.stats` host senkronu artar.
4. Regresyon: `rt_player_materials_ipc.py`, `rt_player_sim_clock_ipc.py`,
   `rt_player_scene_ipc.py --player-only` PASS.

## Player materyal tam desteği — 2026-11-14, Claude

**Canlı ölçüm (derlemeden ÖNCE, 19:28 Player):** çok materyalli + dokulu fixture
(`rt_player_materials_ipc.py --make-fixture`: dama dokulu zemin, kırmızı/yeşil/mavi
küp) Player'da 5 materyal, `default_material_used=false`; görüntü Studio'nunkiyle
gözle aynı (doku, renk, gölge). Statik materyal/doku yolu zaten tamam: Player
Studio'nun çift yönlü `Renderer::updateBackendMaterials/updateBackendMaterial`
çağrısını kullanıyor; tek aygıtta tek yükleme.

İki boşluk kapatıldı (yalnız `RuntimeSceneViewer.cpp` + `RtPlayerIpcPolicy.h`):
- Editör ızgarası Player'da çiziliyordu (proje `grid_opacity` 0.4 taşıyor) → Player
  yüklemede `grid_opacity=0`.
- Yükleme sonrası toplu materyal/doku/ışık kirliliği servis edilmiyordu →
  `advanceRuntime` Studio'nun aynı çağrılarıyla tüketiyor.
- Player IPC: `material.list/get/of_object/set` (oyun kuralı ve boya için; çekirdek yol).

**19:41 Player build ile CANLI PASS:** materyal testi (renkler, dama kontrastı, `grid_band_std=0.0`, `material.set` kırmızı→sarı görüntüde), sim saati ve EchoRoomsLab regresyonu PASS; görüntü gözle kontrol edildi. **Parti kapandı.** /Zs 3/3 PASS. **Yalnız Player artımlı build yeterli** (Studio'da davranış değişmedi;
politika başlığı Studio'da 4 IPC TU'yu yeniden derletir, gerekmez).

1. `python scripts/test/rt_player_materials_ipc.py` (Player açık) → PASS: renkler,
   dama kontrastı, `grid_band_std < 1`, `material.set` ile kırmızı küp sarıya döner.
   - `editor grid visible` → `grid_opacity` sonradan yeniden yükleniyor (ör. proje
     ayarının ikinci uygulanması) ya da backend opaklık 0'da da çiziyor.
   - `material.set did not reach the Player image` → per-slot yol viewport'a
     ulaşmıyor; ★ sinsi: test geçer görünür ama renk **bir sonraki tam yüklemede**
     gelir — bekleme döngüsü render_calls artışına bakıyor, görüntüye değil.
2. `python scripts/test/rt_player_sim_clock_ipc.py` ve `rt_player_scene_ipc.py --player-only`
   regresyonu PASS.

## Ortak simülasyon saati (Player adım 1) — 2026-11-14, Claude

**İlk canlı tur (19:11/19:14 exe'leri):** Studio maddeleri 1–3 PASS (fixture üretildi;
Studio'da ortak saatle kutu 3.0 → 0.25 m düşüp zeminde durdu, scrub kare 0/20/99 =
3.0/0.318/0.25 geri yükledi). Madde 5 PASS (EchoRoomsLab; sim'siz sahnede saat boşta).
**Madde 4 FAIL → düzeltildi, yalnız Player artımlı build bekliyor:** Player
`playing=true` veriyordu; `updateSimulationTimeline` serbest akışı yalnız
`live_mode && !playing` için koşar, timeline yolu playhead 0'da hiç adım atmaz
(184 tick, kutu y=3'te). `RuntimeSceneViewer.cpp` artık `playing=false`. /Zs PASS.
Bootstrap testinin sabit metot listesine iki yeni salt okunur metot eklendi.
**19:28 Player build ile madde 4 PASS:** kutu 2.96 → 0.25 m, 231 tick, 230 raster refit, 1 rebuild (yükleme), piksel imzası değişti. Madde 5 tekrar PASS. **Parti kapandı.**

### İlk partinin listesi

Simülasyon tick'i panelden (`scene_ui.cpp`) `Core/RuntimeSimulationClock`'a taşındı;
sim'in hareket ettirdiği gövdelerin raster refit'i Main'den `VulkanSceneSync`'e.
Player her döngüde aynı saati canlı modda sürüyor. Yeni .cpp iki projeye kayıtlı
(Player vcxproj üretici ile). /Zs 8/8 (Player 4, Studio 4) ve include audit PASS.
Önceki "ilk kare device-loss" maddesi 2026-11-14 akşam canlı PASS oldu.

**İki projeyi reload et (yeni dosya) → ikisini artımlı Release x64 Build/link.**
Header değişikliği yalnız `Core/RuntimeSceneViewer.h`, `Core/VulkanSceneSync.h` ve
`Api/RtPlayerIpcPolicy.h` (az TU); yaygın struct düzeni değişmedi.

Sıra (bağımsız ve hızlı önce):

1. **Studio açılır, boş sahnede Play/Pause çalışır.** Panel yalnız saati çağırıyor.
   Bozuksa: `scene_ui.cpp` çağrı bloğu (`AppRuntime::simulationClock().tick`).
2. **Studio'da var olan bir sim sahnesi (rigid/gaz) timeline'da eskisi gibi oynar,
   scrub geri yükler, sim parametresi değişince "Sim changed - cache reset" toast'ı
   çıkar ve playhead başa döner.** Bozuksa: `rewind_to_start` sonucu UI'da uygulanmıyor.
   ★ En sinsi olası hata: **Timeline Start/End değiştirmek simülasyonu sıfırlıyorsa**
   aralık-düzenleme koruması ilk tick'te kaçmıştır (saat ilk çağrıda aralığı yalnız
   öğrenir). Hata vermez, yalnız ağır sahne başa atlar.
3. **Fixture (Studio açık, BOŞ sahne):** `python scripts/test/rt_player_sim_clock_ipc.py --make-fixture`
   → `docs/dev/.captures/player-sim-clock/rigid_drop.rtp` yazılır, Studio sahnesi
   yeniden boşalır. Bozuksa: fixture sorunu, Player değil.
4. **Player kabulü:** `python scripts/test/rt_player_sim_clock_ipc.py` → PASS:
   kutu ≥1.5 m düşer, zeminde (y 0–0.6) durur, `driven_ticks` artar,
   `raster_refits + raster_rebuilds > 0`, piksel imzası değişir.
   - `box did not fall` + `driven_ticks` artıyor → solver adımlıyor ama rigid gövde
     kaydolmadı (ProjectManager rigid_bodies yüklemesi / syncSimulationWorld).
   - `driven_ticks` artmıyor → `advanceRuntime` çağrılmıyor ya da
     `anySimulationRuntimeEnabled()` false.
   - kutu düşüyor ama `motion never reached raster` / imza aynı → deform refit
     yolu: `setDeformRefitActive(true)` etkisiz, `markDirty(Geometry)` yolu servis edilmiyor.
   - ★ Sessizce makul: kutu düşüyor, imza değişiyor ama **her karede
     `raster_rebuilds` artıyor** → refit hep reddedilip tam raster kuruluyor; çalışır
     görünür, büyük sahnede kare süresi çöker. Beklenen: rebuild ~0–1, refit çok.
5. **EchoRoomsLab regresyonu:** `python scripts/test/rt_player_scene_ipc.py --player-only` PASS
   (sim'siz sahnede saat hiçbir şey yapmamalı; `simulation.clock_ticks` 0 kalır).
6. Player penceresi arka plandayken sim sürer (kiosk kararı); render yalnız ön planda
   ya da capture açıkken. Arka planda 20 Hz pompada dt 1/30'a kırpıldığı için sim
   gerçek zamandan yavaş akar — bilinen, kabul edilen sınır.

## Player ilk kare device-loss düzeltmesi — 2026-11-14

İlk canlı yükleme PASS (26 mesh/312 üçgen), ilk GPU görüntü FAIL: uzun NVIDIA
pipeline hazırlığından sonra device_lost=true, 1 submit/0 consume. Hazırlıkta geç
capability yayını ve boş materyal tablosu görüldü; RuntimeSceneViewer bunları
artık düzeltir/doğrular. Kesin GPU kök nedeni yeniden canlı kabul bekler.

**Yalnız Player'ı kapat → Player artımlı Build/link → aç.** Bu kabul için yeniden
Studio build istenmiyor. Sonra dış terminalde `python scripts/test/rt_player_scene_ipc.py
--player-only` veya ikisi de açıksa aynı komutun --player-only olmayan biçimi.
Yeni material_buffer_ready true, material_count>0, gerçek kare/draw ve reload kabulü
geçmeli. /Zs, 644/644 include ve Python syntax PASS. Ayrıntı ve açık cold-pipeline
UI borcu: PLAYER_STATIC_VIEWER_ACCEPTANCE.md başındaki canlı sonuç.


## Birleştirilmiş Player .rtp / Vulkan kabul paketi — kaynak hazır

Önceki AppFrame ve konsolsuz Release partileriyle birlikte statik scene viewer
bağlandı. Ara build checkpoint'leri yerine **iki projeyi reload edip ikisini bir
kez artımlı Release x64 Build/link et**. Yalnız kaynak/sözdizimi denetimi yapıldı;
yeni exe için link, shaderların runtime yüklenmesi ve görüntü henüz ölçülmedi.

İkisi açıkken `python scripts/test/rt_player_scene_ipc.py` dış terminalden çalışır.
Varsayılan fixture mevcut EchoRoomsLab.rtp; alternatif `--project "tam-yol.rtp"`.
Betik gerçek görüntü/GPU sayaçları, kamera, yeniden açma ve hata semantiğini ölçer;
Studio'ya yazmaz. Sonra resize, boşta tüketim, konsolsuz açılış ve temiz kapanış
kontrol edilir. Ayrıntı: [PLAYER_STATIC_VIEWER_ACCEPTANCE.md](PLAYER_STATIC_VIEWER_ACCEPTANCE.md).

Scope statik flat mesh/material/camera/light/environment MaterialPreview'dır.
Scatter/VDB/gas/su/particle/hair/fracture runtime yok; açık ret uygulanır. Tam ortak
sim/animasyon tick'i bu partiye dahil değildir. Yeni shader/ABI değişikliği yok.
Player 644/644 include temiz, generator/descriptor/capability audit PASS; yeni ve
bağlanan kaynakların /Zs kontrolleri PASS. Eski aşağıdaki notlar tarihsel; bu
birleştirilmiş kabul paketi build sırasını supersede eder.


## Ortak kare işlemleri — AppFrame (2026-11-13, kaynak hazır)

**İki proje reload + ikisi için artımlı Build/link.** Yeni `Core/AppFrame.cpp`
ikisinde de ClCompile; Main.cpp ve PlayerMain.cpp yalnız çağrı bağlantısı alır.
Önceki konsolsuz Release Player proje ayarı bu generator çıktısında korunur.
C++ layout/shader/IPC sözleşmesi değişmedi; Codex C++ build veya app launch yapmadı.

- Studio'nun 3 ImGui kare başlangıcı, 5 Render kapanışı, 4 draw/present yolu;
  olay pompası ve mevcut konumundaki IPC pompası ortak FrameCycle kullanır.
  Scene-loading sırasında IPC engeli, sunum renkleri/texture copy ve perf kapsamları
  yerinde kalır. Studio'ya Player'ın foreground/idle/pacing politikası uygulanmaz.
- Player bootstrap aynı ortak operasyonları kullanır. Değişmeyen HUD boşta
  çizilmez; odak dışı sunum yok; 60/20 Hz uyuyan pompa korunur.
- Bu yalnız ortak kare I/O katmanıdır: sim tick, geometri/kamera/malzeme/VDB
  senkronu ve Vulkan render halen Studio döngüsünde. `.rtp` Player'a açılmadı;
  Player discovery/izin listesi bu partide genişletilmedi.

Kullanıcı kabulü:
1. İki exe birlikte açıldığında dış terminalde
   `python scripts/test/rt_player_bootstrap_ipc.py` PASS olmalı.
2. Player sabit odaklı/odaksız/minimize halde tüketimi düşük kalmalı; geri dönüş,
   resize ve pencere kapatma çalışmalı. Yeni GUI subsystem ile konsol açılmamalı.
3. Studio sahne yükleme ekranı, normal viewport, çözünürlük değişimi ve yüklü
   sahneyle IPC cevapları korunmalı. Yükleme sırasında kuyruğa alınan çağrı,
   yükleme bittikten sonra çalışmalı; loader sahnesine arada yazmamalı.
4. Rendered/RayFusion mod geçişi ve normal kapanış mevcut davranışı korumalı.

Statik: generator --check PASS; iki XML'de AppFrame.cpp tam birer kez kayıtlı;
include audit 641/641 temiz, dört yasaklı kategori 0; include audit testleri 2/2;
diff whitespace kontrolü PASS. Bunlar yeni C++ link/canlı kabul yerine geçmez.


## Player konsol penceresi — 2026-11-13

Studio + Player birlikte açıkken dış IPC bootstrap kabul betiği PASS.
Kullanıcı idle düzeltmesini iki hedefte derlediğini ve Player boşta CPU/GPU
tüketiminin görünmediğini bildirdi. Scene render henüz yok; statik HUD görüntüsü
yeniden çizilmeden korunur.

Player generatoru Release/ReleaseDev için `SubSystem=Windows` üretir; Debug
Console kalır. SDL2main zaten link listesinde; SDL_main girişini WinMain üzerinden
çağırır. IPC named-pipe sunucusunun konsola ihtiyacı yoktur. **Player kapat, VS
projesini reload et, Release x64 için Build/link al.** Studio rebuild gerekmez.
Beklenen: konsol açılmadan Player/HUD başlar; tam dış IPC betiği yine geçer.
Generator/XML kontrolü PASS; yeni GUI subsystem link/canlı açılışı bekler.


## Player bootstrap boşta tüketim düzeltmesi — 2026-11-13

Kullanıcı Player 3e build PASS bildirdi. Açık Player üzerinde dış Python IPC
bootstrap kabul testi **PASS** (sandbox error 5 sonrası aynı prob escalation ile).
Bu test Studio'yu veya kapanışı doğrulamaz.

Yeni PlayerMain değişikliği için yalnız **Player artımlı build** gerekir; Studio
kaynakları değişmedi. Player kapatıldıktan sonra derle. Sabit HUD artık yalnız
pencere/girdi olayı veya IPC durum değişiminde çizilir; boşta NewFrame/clear/draw/
present çağrılmaz. Odak dışı/minimize sunum tamamen atlanır; IPC kuyruğu en çok
yaklaşık 50 ms aralıkla işlenir. Odakta olay patlamaları yaklaşık 60 FPS ile
sınırlı; boşta olay/IPC kontrolü yaklaşık 60 Hz, arka planda 20 Hz uyuyarak çalışır.
FPS yazısı sürekli çizim gerektirdiği ve bootstrap için anlamlı olmadığı için
"Scene: not loaded" durumuyla değiştirildi. Bu politika oyun simülasyonuna
uygulanmış değildir; bootstrap döngüsüne aittir.

Kabul: odakta fareyi sabit tutup CPU/GPU tüketimini karşılaştır; sonra odağı
başka pencereye geçir ve minimize et. Sürekli GPU sunumu durmalı, tüketim
belirgin azalmalı. Geri dönünce/resize sonrası HUD yenilenmeli. Arka planda da
`python scripts/test/rt_player_bootstrap_ipc.py` cevap vermeli (Studio hazırken).
Yeni performans davranışı henüz derlenmedi veya canlı ölçülmedi.


## Player Faz 3e — ayrı IPC ve bootstrap capability sınırı (kaynak hazır)

1. Visual Studio proje değişikliğini reload et. **Player Rebuild** yap: yeni
   `RT_PLAYER` hedef tanımı nedeniyle eski Player objeleri kullanılmamalı.
   Studio için değişen dört IPC kaynağını içeren **artımlı build** yeterli;
   C++ layout/shader ABI değişmedi. Codex C++ build veya uygulama başlatmadı.
2. Studio ve Player'ı birlikte aç. Ayrı terminalden:
   `python scripts/test/rt_player_bootstrap_ipc.py`
   Beklenen: Studio kendi kimliğini/geniş kataloğunu, Player bootstrap kimliğini
   ve dört keşif metodunu bildirir; mesh/scatter/fracture/project/script/batch
   Player'da `unsupported_in_player` hatası verir. Sayısal `agent.describe.method`
   parametresi hatayla reddedilir. Betik uygulama açmaz ve sahneyi değiştirmez.
3. Player'ı pencere kapatma düğmesiyle kapat. Task Manager'da Player işleminin
   bittiğini kontrol et (pencerenin kaybolması tek başına yeterli değil).
   Sonra `python scripts/test/rt_player_bootstrap_ipc.py --studio-only` çalıştır:
   Studio hâlâ cevap vermeli. Player'ı yeniden açıp tam betiği tekrar çalıştır.
4. Bağlı bir IPC istemcisi varken, ayrıca art arda `agent.discover` istekleri
   giderken Player'ı kapat; işlem kısa sürede bitmeli. İstemcide pipe kapanış
   hatası normaldir, Player'ın asılı kalması değildir. Bu yarış canlı doğrulanmadı.

Yerel named-pipe varsayılanları içindir. İki uygulamada birden isteğe bağlı TLS
etkinleştirilirse mevcut ortak port yapılandırması ayrıca ayrılmalıdır.
Canlı prob sandbox Windows error 5 verirse aynı dış Python probunu escalation
ile tekrar çalıştır; alternatif taşıma veya embedded script kullanma.

Statik sonuç: generator `--check` PASS; include audit 640/640 (dört yasaklı
kategori 0); mevcut include audit birim testleri 2/2; Python syntax/help PASS.
Bu sonuçlar C++ build/link, IPC cevabı veya temiz kapanış kanıtı değildir.


## Güncel kabul durumu — 2026-11-13

Kullanıcı tam Studio derlemesini sorunsuz tamamladığını bildirdi: Faz 1k'ye kadar
biriken değişikliklerde **Studio TAM build/link PASS (kullanıcı bildirimi)**.
Önceki Studio build bekleme notları Faz 1k'ye kadar bu bildirimle güncellendi.
Kısmi canlı smoke geçti; [rapor](PLAYER_LIVE_SMOKE_2026_11_13.md). Kullanıcı kalan
canlı testleri devraldı. Gerçek AOV resize/temiz kapanış ve sayısal kabul bekler.

## Player Faz 3d — SceneUI durum host'u + Embree host'u (Claude, 2026-11-13)

**Kullanıcı (2026-11-13 22:47): Player derlendi, linklendi, açıldı, HUD göründü
(madde 1 ve 3'ün görsel kısmı PASS). Ajan: madde 2 PASS — `dumpbin /dependents`
çıktısında yasaklı DLL yok. AÇIK: madde 3'te temiz kapanış, madde 4 (IPC cevabı ve
editör-bağımlı metotların hata döndürmesi).**

Studio'da yalnız iki None kaydı: **reload yeter, Studio derlemesi gerekmez.**
Studio kaynak koduna bu partide dokunulmadı. Sıralı kontrol:

1. **Player derleme + link (sağ tık → Build).** Beklenen: link GEÇER ya da tek haneli
   sayıda yeni sembol kalır. Log: `RayTrophiStudio\x64\Release\Player\RayTrophiPlayer.log`.
   LNK2005 (çift tanım) çıkarsa: bir Host dosyası ile gerçek implementasyon aynı anda
   Player'a girmiş demektir (adı ver).
2. (Link geçerse) **Bağımlılık kontrolü.** `dumpbin /dependents x64\Release\RayTrophiPlayer.exe`
   çıktısında `cudart`, `cufft`, `nvcuda`, `OpenImageDenoise`, `embree4`, `python311`
   **olmamalı**. Varsa ayrım sızdırıyor.
3. **Çalıştır.** Studio KAPALIYKEN `x64\Release\RayTrophiPlayer.exe`: siyah pencere,
   sol üstte HUD ("RayTrophi Player", "IPC: running", FPS). Pencere X'i ile temiz
   kapanır. Açılışta çöküyorsa `SceneLog.txt` son satırlarını ver.
   ★ Studio açıkken "IPC: stopped" görünür: pipe adı ortak, beklenen.
4. **IPC cevabı.** Player açıkken `Import-Module .\scripts\ipc\RtIpc.psm1 -Force` ve
   `Invoke-RtIpc agent.discover` (ya da herhangi bir okuma metodu) cevap verir.
   ★ Sinsi: `physics.fracture_object`, `mesh.*` operatörleri, `scatter.*` Player'da
   HATA dönmeli (ya da log'da "[Player] '...' is implemented in the Studio editor").
   Sessizce `ok` dönüyorsa bir stub başarı taklidi yapıyor demektir — bana bildir.

## Player Faz 3c — UI'sız authoring kodu Player'da + su/template host'ları (Claude, 2026-11-13)

**Kullanıcı (2026-11-13): Player derleme 0 hata, link 168 → 39.**

Studio'da yalnız iki None kaydı var: **reload yeter, Studio derlemesi gerekmez.**
Player projesi yeniden üretildi (803 kaynak; ~48 yeni dosya derlenecek).

1. **Player derleme (sağ tık → Build).** Derleme hatası beklenmez; yeni alınan
   MeshEdit/Paint/UV/Template/Import dosyalarından derleme hatası çıkarsa ilk
   20 satır yeter (yalnız 4'ü `/Zs` ile denetlendi).
2. **Player link.** 168'den belirgin düşmeli; MeshEdit/Paint/Template/su/seçim/
   import/Python grupları kapanmalı. Kalan beklenen: SceneUI'a bağlı çağrılar,
   AgentChatPanel, Embree 4 (+ yeni dosyaların getirdiği SceneUI sembolleri).
   Log: `RayTrophiStudio\x64\Release\Player\RayTrophiPlayer.log` — yolu vermen yeter.
   ★ Sinsi: sayı düşmek yerine ARTARSA başarısızlık değildir — yeni alınan
   servis dosyaları SceneUI'a uzanıyor demektir; liste EditorBridge kapsamını belirler.

## Player Faz 3b — ortak runtime state + codec + host'lar (Claude, 2026-11-13)

**Kullanıcı (2026-11-13): Studio build PASS, HDR/EXR sorunsuz; Player linki 364 → 168.**

Studio projesine 2 yeni ClCompile (`Core/RuntimeState.cpp`, `Utils/ImageCodecs.cpp`)
ve 3 None (Host) eklendi → **VS çözümünü reload et**. TAM rebuild GEREKMEZ: artımlı
build Main.cpp, scene_ui.cpp, scene_ui_modifiers.cpp, scene_ui_hierarchy.cpp,
ProjectManager.cpp (scene_ui_animgraph.hpp) ve iki yeni dosyayı derler. Shader/
struct düzeni/IPC değişmedi. Sıralı kontrol:

1. **Studio link.** LNK2005 (çift tanım) ya da LNK2001 (çözülmemiş) **yok**. Bu
   partide bir globalin tanımı Main/scene_ui'den `RuntimeState.cpp`'ye taşındı;
   LNK2001 `g_*`/`stbi_*`/`LoadEXR` → yeni dosya projeye girmemiş (reload);
   LNK2005 → aynı adın ikinci tanımı kalmış (adı bana ver).
2. **Studio açılış + proje aç.** Normal açılır, bir proje yüklenir, viewport
   çizilir. HDRI/EXR ortam dokusu yüklenir (tinyexr), PNG ekran görüntüsü/kaydet
   çalışır (stb_image_write). ★ Sinsi: EXR'li sahne açılıyor ama gökyüzü düz renk →
   tinyexr yapılandırma makrosu kaybolmuş olabilir; log'da EXR yükleme hatası ara.
3. **Animasyon grafiği.** Anim graph panelinde clip node'u sahnedeki clip'leri
   listeler (`g_uiClipsRef`). Liste boşsa panel pointer'ı kurmuyor demektir.
4. **Player link (sağ tık → Build).** Hata sayısı 364'ten belirgin düşmeli (~160
   civarı beklenir). Çıktıyı / `RayTrophiStudio\x64\Release\Player\RayTrophiPlayer.log`
   yolunu ajana ver; kalanlar MeshEdit/Paint/Template/SceneUI/su/Embree kararları.

## Player Faz 3a — ilk Player derleme/link denemesi (Claude, 2026-11-13)

Studio projesine yalnız iki None kaydı eklendi (PlayerMain.cpp, TextureHost.cpp);
Studio derlemesi etkilenmez ama VS çözümü **reload** ister (yeni proje). Player
"Build Solution"a dahil DEĞİL. Sıralı kontrol:

1. **Studio hâlâ tek başına derleniyor.** Build Solution → yalnız RayTrophi Studio
   derlenir. Player da derlenmeye başlıyorsa sln'de Build.0 satırı oluşmuş demektir.
2. **Player projesi yükleniyor.** Solution Explorer'da "RayTrophi Player" görünür,
   yüklenemedi uyarısı yok. Uyarı varsa: `python scripts/gen_player_vcxproj.py --check`.
3. **Player derleme (Release|x64, sağ tık → Build).** Derleme aşamasında hata
   beklenmez (`/Zs` geçen dosyalar dışındakiler denetlenmedi; derleme hatası çıkarsa
   ilk 20 satırı yeter). **Link aşamasında çok sayıda LNK2001/LNK2019 BEKLENİR** —
   başarısızlık değil, ölçüm: çıktı penceresinin tamamını (ya da
   `x64\Release\Player\*.log`) ajana ver; hatalar sembol/dosya bazında kümelenip
   sırayla ayrılacak.
   ★ Sinsi durum: link **temiz geçerse** ve exe CUDA/OIDN/Embree/Python DLL'i
   istemiyorsa harika; ama `dumpbin /dependents x64\Release\RayTrophiPlayer.exe`
   çıktısında `cudart`, `OpenImageDenoise`, `embree4`, `python311` görünürse dönüşüm
   bir bağımlılığı kaçırmış demektir.
4. (Yalnız link geçerse) Player'ı çalıştır: siyah pencere + sol üstte HUD
   ("RayTrophi Player", IPC: running, FPS). Studio açıkken IPC "stopped" görünür —
   pipe adı ortak, beklenen.

## Player Faz 2c — olay pompası + çözünürlük değişimi (Claude, 2026-11-13)

**Kullanıcı (2026-11-13): build PASS, viewport çözünürlük değişimi sorunsuz.**

Yalnız Main.cpp ve AppWindow.h/.cpp değişti; artımlı build yeter, reload gerekmez.
Shader/struct/IPC değişmedi. `/Zs` 3/3, include audit 548/548 PASS. Sıralı kontrol:

1. **IPC uyanması.** Uygulamayı arka plana al, `Invoke-RtIpc` ile herhangi bir metodu
   çağır: cevap hızlı gelmeli. 10× yavaşlık → pompa SDL_PushEvent olayını yutuyor
   (bkz. IPC'nin ana döngüyü uyandırmaması dersi).
2. **Kapatma.** Pencere X'i: kaydedilmemiş sahnede kaydet sorusu gelmeli. Soru
   gelmeden kapanıyorsa quit_requested → `ui.tryExit()` bağı kopmuştur.
3. **Kısayollar.** Ctrl+S, Delete, H; animasyon render'ında ESC/P çalışmalı.
4. **Odak.** Başka pencereye geçip geri dön: viewport donmuş kalmamalı
   (focus_gained yolu).
5. **Çözünürlük.** Render ayarlarında çözünürlüğü değiştirip Apply: kısa siyah kare,
   sonra yeni boyutta görüntü. ★ Sinsi durum: görüntü eski boyutta gerilmiş kalırsa
   doku yeniden kurulmamış ama hata da loglanmamış demektir; log'da
   "Failed to create SDL surfaces" varsa kurulum başarısız ve erken dönülmüştür.

## Player Faz 2b — ortak pencere/SDL yaşam döngüsü (Claude, 2026-11-13)

**Kullanıcı (2026-11-13): 3–5 PASS** (Türkçe giriş, docking/ayırma).

Yeni `Core/AppWindow.h/.cpp` proje/filter'a kayıtlı → **VS'de projeyi reload et**.
Faz 2a ile aynı artımlı build/link partisine girer (Main.cpp + iki yeni .cpp).
Header/struct düzeni değişmedi; shader derlemesi gerekmez. `/Zs` 3/3, include audit
548/548 PASS. **Kullanıcı bildirimi (2026-11-13): Faz 2a+2b build/link PASS,
Studio sorunsuz açıldı** (madde 1–2). Kalan 3–8 ayrıca doğrulanmadı.
Sıralı kontrol (bağımsız/hızlı önce):

1. **Link.** Beklenen: `window`/`renderer`/`surface`/`original_surface`/`raytrace_texture`
   için LNK2005 (çift tanım) veya LNK2001 (çözülmemiş) **yok**. Varsa: AppWindow.cpp
   projeye girmemiş (reload) ya da başka bir .cpp bu globalleri hâlâ tanımlıyor.
2. **Normal açılış.** Splash görünür, durum metinleri sırayla akar, ana pencere
   "RayTrophi Studio" başlığıyla maximize açılır. Pencere hiç görünmüyorsa veya
   splash takılı kalıyorsa: createMainWindow sonrası SDL_ShowWindow yolunu kontrol et.
3. **Yazı tipi ve Türkçe karakter.** Panellerde Segoe UI ve ş/ğ/ı doğru. Kare/soru
   işareti görüyorsan: ImGuiConfig.font_path geçmiyor → ImGui varsayılan fontuna düştü.
   ★ Sinsi olan bu: varsayılan font da okunaklıdır, yalnız Türkçe glifler bozulur.
4. **Docking + klavye.** Panel sürükle/yerleştir çalışır, Tab ile alan gezinmesi var.
5. **IME/metin girişi.** Bir ad alanına Türkçe harf yaz; karakter gelmiyorsa
   SDL_StartTextInput çağrısı kaybolmuştur.
6. **`--startup-diagnostics`** ile aç: StartupCrash/emergency logda
   `SDL_Init done … ImGui SDL init done` satırları eskisiyle aynı metin ve sırada.
7. **Temiz kapanış.** Pencereyi kapat: süreç takılmadan çıkar, `StartupCrash.log`
   silinir, SceneLog'da kapanışta hata yok. Kapanışta AV → terrain node önizleme
   dokusu renderer'dan sonra bırakılıyor demektir (sıra Main'de korunmalı).
8. **`--headless`/gizli açılış.** Pencere gösterilmeden IPC hazır olur ve kapanır.

## Player Faz 2a — Studio launch ayrımı

Main.cpp ve yeni Core/StudioLaunchOptions.cpp kullanıcı artımlı Studio build/link
partisine dahil. Yeni .cpp proje/filter kayıtlı; VS projeyi reload et.
`/Zs` 2/2, parser gövde eşitliği, XML tekil kayıtları ve include audit PASS.
Player 547/547 temiz; SDK sınırları 0/0/0/0. Shader derlemesi gerekmez.

Sonraki build'de normal açılış/pozisyonel proje ve --project, --script;
--render/--spp/--frame, --frames/--frames-dir ve --headless davranışını kontrol et.
Eksik argüman, bozuk/taşan integer ve bulunamayan proje hata metinleri korunmalı.
Bu parça yeni exe üzerinde canlı denenmedi; şu an açık exe Faz 1k build'idir.

## Player Faz 1k: Python include 16 → 0 — 2026-11-13 (Codex)

Sonraki kullanıcı TAM build/link partisine dahil; şimdi build talebi değildir.
Studio yeni on RtPython*Bindings.cpp dosyasını ortak servis/IPC modülleriyle
birlikte derler. Player yalnız ortak tarafı alır. İmzalar, IPC descriptor/schema,
struct ve shader alanları değişmedi; shader derleme/dağıtım ihtiyacı yok.

- `/Zs` **34/34 PASS**; **11 Player** include dökümü SDK'sız. **32** kaynak eşitliği,
  proje/filter kayıtları ve diff-check PASS. Include audit **547/547 temiz**,
  CUDA/OptiX/Embree/OIDN/Python = **0/0/0/0**. IPC capability/descriptor audit PASS.
- Kaynak sözleşmeleri **86/89**: timing boyutu, cache v18 ve eski params.h alan
  beklentileri parti öncesi API snapshot'ında da FAIL. Ayrı post-v2 JS denetimi
  dokunulmayan OptixWrapper launchOptixDisplayPost guard'ında FAIL. Ayrıntı ve
  test dosyaları [Python raporunda](PLAYER_PYTHON_BOUNDARY_AUDIT_2026_11_13.md).
- Kullanıcı kabulü: proje reload/tam Studio link, rt importu ve ses/UV/volume/
  lens-weather/object-substance/post/cutout/particle-label/weather/timing Python
  çağrıları; karşılık gelen IPC metotları; bozuk tür, eksik context/kamera,
  NaN/Inf reddi. Önceki partilerin kabul listesi korunur. Canlı/nümerik kabul yok.
- Player gerçek linki yapılmadı. Script/addon IPC yolları ve Python sahibi API/UI
  tick'leri runtime/capability sınırında ayrılmalı; include=0 DLL'siz exe kanıtı değil.

## Player Faz 1j: Embree include sınırı 0 — 2026-11-13 (Codex)

Sonraki kullanıcı TAM derleme/link partisine dahil; şimdi build talebi değildir.
Studio `HairEmbree.cpp`, Player `HairQueryHost.cpp` seçer; iki implementasyon aynı
hedefte linklenmez. EmbreeBVH.cpp SDK include'unu kendi alır. EmbreeBVH.h artık
SDK tanımı sağlamaz; rtc*/somut RTC kullanan kaynak kendi SDK include'unu almalı.

- `/Zs` 14/14 PASS: beş ortak kaynak iki dal, üç Studio kaynağı (Main dahil) ve
  Player Host. Altı Player dökümü SDK'sız. AssimpLoader.cpp aktif kayıt yok; /Zs yok.
  Taşınan dört CPU gövdesi/cleanup ve kalan ortak saç kodu kaynak eşitliği PASS.
- Include audit BOM ilk-satır hatası düzeltildi; iki regresyon testi PASS. Embree
  gerçek başlangıç 7 (önce 6 raporlanmıştı), son 0. Son sayılar **552 aday/536
  temiz**, CUDA/OptiX 0, Embree 0, OIDN 0, Python 16. Baseline sıfıra kilitlendi.
- Kabul: CPU Embree render/material remap, instance/particle bridge; saç lazy BVH,
  brush picking ve shadow occlusion; Vulkan groom animasyonu/GPU upload; proje
  yeniden açma, clear ve kapanış. Önceki partilerin kontrolleri korunur.
- Ana build/link ve canlı/sayısal kabul yok. Shader, struct alanı veya IPC değişmedi;
  shader derleme ihtiyacı yok. Kaynak seçimi ve ayrıntı
  [Embree sınır raporunda](PLAYER_EMBREE_BOUNDARY_AUDIT_2026_11_13.md).

## Player Faz 1i: CUDA/OptiX include 12 → 0 — 2026-11-13 (Codex)

Sonraki kullanıcı TAM derleme/link partisine dahildir; bu kayıt şimdi build talebi
değildir. **17 yeni adaptör** Studio projede kayıtlı: 9 Studio ClCompile, 8 Host
None. Projeyi reload et; iki alternatif aynı hedefte linklenmemeli. Tam seçim
[CUDA sınır raporunda](PLAYER_CUDA_BOUNDARY_AUDIT_2026_11_13.md).

- SDK'lı/CUDA'sız `/Zs` **35/35 PASS**; 17 Player include dökümünde CUDA/OptiX yok.
  31 taşınan kaynak eşitliği ve 81 alan/sizeof/alignof (iki dal) PASS. Include audit:
  **552 aday / 530 temiz**, CUDA/OptiX **0**, Embree 6, OIDN 0, Python 16.
- Erosion PTX derlendi, eski gövdeyle birebir aynı; kaynak + Release hash PASS.
  Stylize CUDA PTX kaynak-yolu sembol hash'i dışında aynı, iki ptxas PASS.
  Stylize native CudaCompile nesnesi sonraki kullanıcı derlemesinde yenilenmeli.
  Yeni shader/binding/push/struct alanı/IPC değişikliği yok.
- Kabul: Vulkan RT/raster; VDB sequence ve canlı gaz; Studio CPU/CUDA gaz; erosion
  Vulkan/CPU fallback; OptiX foliage/saç/malzeme; OIDN GPU-direct/resize/geçiş/kapanış;
  Stylize, proje yeniden açma ve kaynak temizliği. Önceki partilerin kabulü korunur.
- C++ proje build/link ve canlı/sayısal kabul yapılmadı. Player hedefi kurulunca
  gerçek SDK kütüphanesiz link testi gerekir; Renderer/loader/IPC editör ve offline
  sembol ayrımı henüz tamamlanmadı. Gas Host CPU fallback performansı ölçülmedi.

## Player kapsam kararı: eski Water/River dışarıda — 2026-11-13 (kullanıcı)

Mevcut su/nehir ve CUDA FFT ocean Player'a taşınmayacak; realtime su ayrı yazılacak.
Studio su kaynakları aynı kaldı; geçici CUDA/Host ayrımı geri alındı. Player aday
manifesti WaterManager/RiverManager/fft_ocean.cu'yu dışlar; include audit PASS:
547 aday / 514 temiz, CUDA/OptiX 12, Embree 6, OIDN 1, Python 16. Azalma kapsam
kararından gelir. Bu parti C++/shader değişikliği veya yeni derleme talebi içermez.

Player hedefi/link ayrımı hazırlanırken su tick/cleanup, proje yükleme, serializer
bind, render FFT bridge ve atmosfer/terrain su API çağrıları Studio'ya ayrılmalı
(tam liste `RAYTROPHI_PLAYER_PLAN.md` devir notunda). Pişmiş flat su geometrisi ve
malzemesi yüklenmeli; Water/River runtime nesnesi, CUDA FFT ve eski su authoring
API'si oluşmamalı. Atmosfer ve ayrı Fluid sim korunur. Player henüz linklenmedi;
manifest dışlaması tek başına çalışır Player kanıtı değildir.

## Player Faz 1h World vektör matematiği — 2026-11-13 (Codex; temizlik sürüyor)

`World.cpp` CUDA `vec3_utils.cuh` yerine aynı formüllü iki yerel host yardımcı kullanır.
SDK'lı ve CUDA'sız `/Zs` 2/2 PASS; Player include dökümünde CUDA SDK yok.
Yardımcı/include değişimini geri çeviren kaynak eşitliği kontrolü PASS.
Include audit PASS: 549 aday / 515 temiz, CUDA/OptiX 14 → 13; baseline güncel.
Shader/struct ABI ve IPC değişmedi; shader derlemesi gerektiren değişiklik yok.

Sonraki toplu kullanıcı derlemesinde `World.cpp` derleme/link kabulü; ardından
Nishita güneş yönü (elevation/azimuth), reset, proje yeniden açma ve CPU/Vulkan/OptiX
atmosfer görünümü kontrol edilmeli. Ana build/link ve canlı/sayısal test yapılmadı.
Bu kayıt şimdi derleme talebi değildir; önceki TAM derleme listesi korunur.

## Player Faz 1g atmosfer kaynak ayrımı — 2026-11-13 (Codex; temizlik sürüyor)

Studio: ortak `AtmosphereLUT.cpp` + yeni `AtmosphereLUTCuda.cpp` derlenir. Player:
ortak dosya + `AtmosphereLUTHost.cpp` derlenmeli; iki kaynak implementasyonu aynı
hedefe konmamalı. Host dosyası Studio projesinde yalnız `None` öğesidir.

SDK'lı ortak/CUDA ve `RT_NO_CUDA` ortak/Host `/Zs` 4/4 PASS; Player include dökümleri
CUDA SDK'sız. Hesaplama ve eski CUDA gövdeleri birebir kaynak karşılaştırması PASS.
Shader/struct ABI değişmedi, shader derlemesi gerektiren değişiklik yok.

Include denetimi: 549 aday / 514 temiz; CUDA/OptiX 15 → 14. Yeni Host dosyası aday
ve temiz sayısını birer artırır; ortak atmosfer dosyasının temizlenmesi de +1 temizdir.
Sonraki toplu kullanıcı kabulünde atmosfer parametre değişimi, Vulkan host LUT
yüklemesi, CPU gökyüzü ve OptiX gökyüzü, proje yeniden açma/kapanış kontrol edilmeli.
Ana build/link veya canlı/sayısal kabul yapılmadı; bu kayıt şimdi build talebi değil.

## Player Faz 1f ortak hacim başlığı — 2026-11-13 (Codex; temizlik sürüyor)

`GpuVDBVolume`/`GpuGasVolume` artık `GpuVolumeTypes.h` içinde; `params.h` bunları
oradan alır. Vulkan volume upload ve ortak volumetric yayıncı OptiX launch parametrelerini
çekmez. Alan tanımları, 149 alan offset'i, sizeof/alignof karşılaştırmaları PASS;
SDK'lı ve CUDA'sız başlık probları PASS. Üç tüketici `/Zs` 3/3 PASS.

Etkilenen üç PTX (raygen, miss_kernels, hitgroup_kernels) derlendi: NVCC/entry/target
kontrolü PASS; eski kaynak çıktılarıyla bayt bayt aynı. Kaynak/Release `ptx/` kopyaları hash PASS; Debug klasörü yok, oraya dağıtım yapılmadı; ilgisiz shader'lar derlenmedi. Shader ABI değişmedi.

Include denetimi: 548 aday / 512 temiz; CUDA/OptiX 17 → 15, diğer sayılar aynı.
Ana C++ build/link ve canlı kabul yapılmadı; sonraki toplu kullanıcı doğrulamasında
Vulkan RT ve viewport'ta VDB + gas hacmi, ardından OptiX hacmi görünümü kontrol edilmeli.
Bu kayıt şimdi ana uygulama derlemesi talebi değildir.

## Player Faz 1e include temizliği — 2026-11-13 (Codex; temizlik sürüyor)

`EmbreeBVH.h` artık `OptixWrapper.h` sağlamaz; export veri tipi için `OptixTypes.h`
alır. `OptixWrapper.h` OIDN SDK sağlamaz; tek gerçek kullanıcısı kendi include'unu alır.
InstanceManager/WaterManager kullanılmayan wrapper include'larını bıraktı.

- MSVC `/Zs` 6/6 aktif kaynak PASS: material API, ParticleRenderBridge, InstanceManager,
  WaterManager, OptixWrapper, EmbreeBVH. Eski AssimpLoader.cpp projede kayıtlı değil,
  derleme komutu bulunmadı; denetlenmedi. Tüm proje tekrar taranmadı.
- Include denetimi: 548 aday / 510 temiz; CUDA/OptiX 21 → 17, OIDN 12 → 1,
  Embree 6 ve Python 16 aynı. SDK include'u ortak başlıklara geri eklenmemeli.
- Shader/struct/IPC davranışı değişmedi. Sonraki kullanıcı derleme partisine dahil;
  bu not şimdi derleme talebi değil. Önceki OIDN/OptiX/CPU hair kontrolleri korunur;
  instance scatter ve su sahnesinin görünümü de o kabulde kontrol edilir.

## Player Faz 1d OIDN başlık ayrımı — 2026-11-12 (Codex; temizlik sürüyor)

Bu kayıt şimdi derleme talebi değildir; sonraki kullanıcı TAM derlemesine dahildir.
`Renderer` nesne yerleşimi değişti. Yeni `RendererOidn.cpp` Studio projesi/filters'a
kayıtlı; unresolved `createOidnState`/deleter/denoise sembolü görülürse proje reload ve
yeni kaynak kaydı kontrol edilmeli. Ortak başlığa OIDN SDK include'u geri eklenmemeli.

- `RendererOidn.cpp`, `Renderer.cpp`, `Main.cpp`: MSVC `/Zs` PASS; yeni modül kalite
  varsayılanlarını SDK enum değerleriyle static_assert eder. Taşınan metot gövdeleri
  durum erişim öneki hariç önceki kaynakla birebir aynı doğrulandı.
  Realtime-denoiser API, domain-creation API, PostSurface, VulkanBackend de PASS:
  toplam **7 / 7**. Bütün kaynaklar yeniden sözdizimi taramasından geçirilmedi.
- Include denetimi: 548 aday / 510 temiz, OIDN 71 → 12; diğer yasaklı kategoriler aynı.
- Sonraki canlı kabul: host OIDN + GPU OIDN, kalite değişimi, çözünürlük değişimi,
  OptiX ↔ Vulkan geçişi, GPU denoiser release ve uygulama kapanışı. Bu partide koşulmadı.
- ImGui/SDL/ses/Jolt ve gerektiği ortaya çıkan diğer runtime kütüphaneleri korunur;
  bu örnekler kapalı izin listesi değildir.

**Player kapsam düzeltmesi (2026-11-12, kullanıcı):** ImGui, SDL, mevcut ses sistemi
ve Jolt Player'da korunur; Studio panelleri/authoring araçları ayrı tutulur. Güncel
include denetimi 548 aday / 454 temiz, CUDA/OptiX 21, Embree 6, OIDN 71, Python 16 PASS.
Aşağıdaki 355 temiz / ImGui bağımlılığı notları eski ölçüte aittir. Bu karar yeni C++
değişikliği veya şimdi derleme gereksinimi oluşturmaz; temizlik sürüyor.

## Player Faz 1c API başlık ayrımı — 2026-11-12 (Codex; kullanıcı TAM derlemesi bekliyor)

Önceki 1a/1b partisiyle birlikte derlenmeli. `UIContext` artık `UIContext.h` içinde;
üye sırası/tipler/varsayılanlar kaynak karşılaştırmasıyla aynı doğrulandı.
`RtApiInternal.h → scene_ui.h` bağımlılığı kaldırıldı. Editör/renderer/selection/color
üyelerini kullanan kaynaklar ilgili başlığı açıkça içeriyor; yeni runtime davranışı yok.

- CUDA'sız tiplerde float4/int4/uint4 hizalaması 16, uchar4 4 bayt olarak düzeltildi.
  Yerleşim/aggregate probu gerçek CUDA SDK ve `RT_NO_CUDA` ile `/Zs` PASS.
- `RtApiInternal.h`'yi doğrudan kullanan **89 / 89 kaynak MSVC `/Zs` PASS**.
  `RtManualSimulationStep.h`, `RtApiSpline.cpp`, `PostSurface.cpp` eksik tanım
  bağımlılıkları düzeltildi, tekrar kontrol edildi. İlk binding denemelerindeki
  eksik Python include yolu denetim ortamında tamamlandı; son binding kontrolleri PASS.
  Bu kontrol obj/exe üretmez; link veya canlı davranış kabulü değildir.
- Include denetimi: 548 aday, 355 temiz; OIDN 113 → 71, diğer kategoriler aynı.
  ImGui hâlâ `scene_data.h → GeometryNodesV2.h → NodeCore.h` yoluyla geliyor.
- Derlemede eksik `SceneUI`/`Renderer`/`SceneSelection`/`ColorProcessor` tanımı görülürse
  kullanıldığı kaynak kendi başlığını almalı; `RtApiInternal.h`'ye editör include'u geri eklenmemeli.
- Canlı kabul: IPC/Python üzerinden sahne sorgusu ve bir geri alınabilir düzenleme;
  manuel fluid/rigid adım sonrası timeline'ın eski kareyi geri yüklememesi;
  seçme/undo-redo ve simülasyon API smoke kontrolleri. Önceki OptiX doku kontrolleri de geçerli.

## Player ayrımı Faz 1a — 2026-11-12 gece (Claude; YAZILDI, kullanıcı TAM derlemesi bekliyor)

Plan: `RAYTROPHI_PLAYER_PLAN.md`. Bu parti yalnız **include grafiğini** değiştirir; davranış,
struct düzeni, shader ABI, IPC aynı kalmalı. Yaygın başlıklar değiştiği için **tam derleme** olur.

**★ Listenin başında uyarı:** `World.h`, `Texture.h`, `Triangle.h`, `material_gpu.h`, `sbt_data.h`
değişti. Artımlı derleme hemen hemen her şeyi yeniden derler; "Build" yeterli, ama garip bir
erişim ihlali görürsen önce Rebuild.

**Ön denetim (ajan, derleme değil):** bütün `src/**/*.cpp` MSVC `/Zs` ile (yalnız sözdizimi,
`.obj` üretmez; komut satırları `CL.command.1.tlog`'dan). Sonuç: **157 / 773 birim denetlendi, 157 PASS, 0 FAIL.** Kapsam: değişen adları (`OptixWrapper`, `optix*`, `oidn`, `rtc*`, `cuda*`, doku yükleme üyeleri) metninde geçiren 80 birimin TAMAMI + alfabetik ilk 77 birim. Kalan 616 birim denetlenmedi (tam tarama ~1.5 saat sürüyordu); bunlar o adları doğrudan kullanmıyor ama bir başlık üzerinden dolaylı bağımlılık olabilir — tam derleme asıl kanıt.

**Faz 1b aynı partiye eklendi:** `ParallelBVHNode.h` / `ParallelBVHNode_optimized.h` artık
`OptixWrapper`'ı ileri bildirimle görüyor. `OptixWrapper.h`'yi kaybeden **238 birimin tamamı**
`/Zs` ile denetlendi: 236 PASS, 2 gerçek hata. `scene_ui_mesh_overlay.cpp` düzeltildi (PASS).

`VolumetricRenderer.cpp` düzeltildi (`params.h` eklendi, PASS). Dosyayı kilitleyen 12:45'ten kalma
17 asılı `cl.exe` kullanıcı onayıyla sonlandırıldı.

Sıra: hızlı ve bağımsız olan önce.

1. **Derleme biter mi.** Beklenen: 0 hata. İlk saniyelerde `GpuInteropTypes.h` hatası =
   benim hatam, dur ve bildir. `unresolved external Texture::cudaUploadFull/cudaUpdateFull/
   cudaUpdateRegion/cudaRelease` = `TextureCuda.cpp` projeye girmemiş (vcxproj'a ekledim; VS açıkken
   eklendiyse "Reload" ister). `C2027/C2079 OptixWrapper` veya `rtc*` tanımsız = bir dosya bu
   adları `Triangle.h`/`HairSystem.h` üzerinden görüyormuş; o dosyaya kendi include'u eklenir.
2. **`python scripts/audit_player_includes.py --check`** → `PASS`. FAIL = biri çekirdek başlığa
   yasaklı include ekledi (sayı tabandan yüksek).
3. **Uygulama açılır, Vulkan RT / RayFusion'da dokulu bir sahne doğru görünür.** Bozuksa: bu
   yoldaki tek değişiklik include sırası; doku yanlışsa `Texture.h` bölünmesine bak.
4. **OptiX backend'e geç, dokulu sahne.** Asıl dokunulan yol burası: CUDA doku yüklemesi artık
   `TextureCuda.cpp`'de. Görmen gereken: dokular eskisi gibi. Dokular **siyah/eksik** =
   `cudaUploadFull` çağrılmıyor ya da erken dönüyor. Boyama (mesh paint / terrain splat) sonrası
   OptiX'te fırça izi görünmeli (`cudaUpdateRegion`).
5. **OptiX'te doku silme/değiştirme, sonra proje kapatıp açma.** `cudaRelease` yolu. Çökme ya da
   "yanlış doku yanlış mesh'te" = serbest bırakma sırası bozulmuş (bildirim → senkron → yok etme).
6. **Saç (hair) olan sahne, CPU render.** `HairSystem.h` artık Embree'yi ileri bildirimle görüyor;
   CPU saç kesişimi eskisi gibi çalışmalı.

**★ En sinsi başarısızlık:** OptiX'te dokuların *yüklenmeyip düz renk* görünmesi. Vulkan birincil
yol olduğu için günlük kullanımda fark edilmez, hata da vermez; OptiX ancak TDR sonrası cankurtaran
olarak açıldığında ortaya çıkar. Madde 4'ü atlama.

**Bu partide YAPILMAYAN:** Player exe'si yok, `RT_NO_CUDA` hiçbir yerde tanımlı değil; uyum
başlığının CUDA'sız dalı hiç derlenmedi (Faz 3'te ilk kez derlenecek).


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

**15:11 derlemesi (ajan, 2026-11-12): yalnız parsel tek damla ("uydu" kümeleri).**
- Görsel çocuklar (parsel başına 8 küçük küre) yığındaki boşlukları doldurmak için var; havada
  tek kalan parselde ana noktanın çevresinde sıkı bir küme olarak görünüyordu. Artık çözücünün
  Sprey etiketini taşıyan parsel, yığındaki çocuklarla AYNI BOYDA tek küre çiziliyor.
  (14:40 sürümü ana parselin hacminde tek küre çiziyordu; küçük çocukların yanında iri top
  gibi durduğu için kullanıcı gözlemiyle değişti.) Köprü bunu point-sphere
  örneğinde NEGATİF `scale.y` ile işaretler (`fluidRenderProxyIsolated`).
- Yalnız çizim tarafı; çözücüye, adım süresine ve cache'e yük yok.
- ★ İlk deneme dizi komşuluğuna bakıyordu ve yığının yarısını da büyük küreye çevirdi:
  parseller doğum sırasını korur, karıştıktan sonra dizi komşusu uzayda komşu değildir.
- Canlı (10 648 su parseli, 84 sprey): yığın içi düzgün küçük küreler, havadakiler tek damla.
- ⚠ Kapsam dışı: raster "resident" GPU yolu (`fluid_sphere_proxy.vert`) etiketi göremiyor,
  o yol devredeyken eski kümeler kalır. Bu sahnede CPU impostor yolu çalışıyordu.
- Codex'in Field samples yolu ölçüldü (87k parsel): her adımda
  "transport unavailable or CFL budget exceeded", 0 örnek, adım başına 0.4–6 s CPU. Kapalı kalmalı.
- 1. Gözle: sıçrama sahnesi, Solid ve Rendered. Havadaki damlalar tek küre, yığın eskisi gibi.
- 2. Damla yığına geri düşerken küçük bir "pop" (tek küre → 8 çocuk) beklenir; sprey görsel
  olarak hacminin 1/çocuk sayısı kadarını gösterir (ince serpinti). Pop rahatsız
  ediyorsa etiket histerezisini (ayrılma ≤ 2, katılma ≥ 6 komşu) gevşet.
- 3. ★ Sinsi durum: mesh splat (icosphere vb.) kullanan bir domainde negatif ölçek geometriyi
  ters çevirirdi. İşaret yalnız procedural kürede konuyor; mesh splat sahnesinde spreyin
  normal göründüğünü bir kez kontrol et.

**13:43 derlemesi (ajan, 2026-11-12): DEM alt adımı ölçülen sönümden.**
- Kararlılık sınırı sönümü "bütçedeki bütün temaslar tek doğrultuda" sayıyordu (14 temasta
  toplam sönüm oranı 1.14, adımda 2.7× ceza). Shader artık tane başına gerçek sönüm oranını
  ölçüyor (`diagnostics[47]`), host geçen karenin en büyüğünü 1.5× payla kullanıyor; dört
  doğrusal temasın altına inmez. Kuvvet yasası DEĞİŞMEDİ, yalnız adım boyu.
- ★ Arayüz: tane shader revizyonu 43, tanı başlığı 48 kelime (ses telemetrisi bir kelime kaydı).
  Beş tane shader'ı derlendi, `.spv` iki yerde.
- 74k kum (R = 1 cm): 496 → 382 alt adım/kare, toplam sönüm oranı 1.14 → 0.51. Artık doğruluk
  sınırı (çarpışma başına 24 adım) belirliyor. Yığın oturuyor: en hızlı tane 2 s'de 0.02 m/s.
  IPC: `fluid.matter_models` → `cfl_terms.measured_damping_rate_s`.
- Testler: hızlı duman 16/16, grain_sleep 3/3, grain_inflow 6/6, tane sözleşmesi PASS.
- ⚠ İki FAIL, bu değişiklikle bağı DOĞRULANMADI (önceki exe ile koşulmadı):
  `grain_heat` 7/8 (`cools_at_physical_rate`; ısıl bir geçit) ve `sleep_transfer`
  (ilk adımda `sim.control_state` eşitlik assert'i, fizik koşmadan düşüyor).
- 1. Gözle: bilinen bir kum sahnesini oynat. Yığın eskisi gibi oturmalı. Titreyen ya da
  yavaşça "kaynayan" yığın görürsen ölçülen sönüm payı yetmiyor demektir:
  `DemContactTimeBounds.h`'de 1.5 çarpanını büyüt.
- 2. Hızlı çarpma (yüksekten dökülen kum, hızlı kollider): patlama OLMAMALI. Olursa çarpma
  karesi önceki karenin seyrek temas ölçümünü kullanmıştır; taban değeri (4) yükselt.
- 3. ★ Sinsi durum: yığın biraz daha "yumuşak" sönerse kimse hata demez. Aynı sahnede
  `cfl_terms.substeps` ile birlikte yığının son yüksekliğini eski kayıtla karşılaştır.

**13:24 derlemesi (kullanıcı, 2026-11-12): DEM tane nozulu (inflow).**
- Tane maddesi artık `liquid_emission=inflow` ile dökülüyor (`planGrainInflow`): akışa dik,
  titretilmiş altıgen katmanlar, katman yüzeyi geçince doğar. Debi = alan × hız, kare hızından
  bağımsız; tıkalı nozul borç biriktirmez.
- Canlı: `rt_test_grain_inflow_ipc.py` 6/6 (458 tane, beklenen 466; kare başına en çok 19,
  ortalama 14.8). Bağımsız planlayıcı testi 3/3: en yakın çift 2R × 1.05–1.07, sayı %1 içinde.
- Katı oranı nozul kesitine göre ~%27 (taneler duvardan bir yarıçap içeride doğar).
- 1. Gözle: panelden Sand kaynağını inflow yap, hız ver. Kesintisiz bir sütun dökülmeli.
  Kare başına ayrı öbekler görürsen katman fazı kareler arasında taşınmıyor demektir.
- 2. Nozulu zemine ya da yığına göm, birkaç saniye beklet, çek. Çekince topluca boşalma
  OLMAMALI. Olursa reddedilen doğumlar birikime geri yazılıyor demektir.
- 3. ★ Sinsi durum: kutu (object bounds) nozulu yalnız kutunun orta düzleminden döker;
  ince uzun bir kutuda hız ekseni yanlışsa debi makul görünür ama beklenenden düşüktür.
  `fluid_flux_target_m3_s` ile `total_emitted_particles` artışını karşılaştır.
- Ölçüm (değişiklik yok): 74k kum, R = 1 cm, 496 alt adım/kare, 348 ms/kare
  (`sim_matter_grain_step` 0.70 ms/çağrı). Sınırlayan kararlılık: 14 temas varsayımı ve
  toplam sönüm oranı 1.14 (aşırı sönümlü). Doğruluk sınırı tek başına 381 alt adım ister.
  Komşu listesi her alt adımda yeniden kuruluyor (22 ms/kare).

**12:06 derlemesi (ajan, 2026-11-12): kollider yüz ağırlıklarında tam kesin kısayol.**
- `computeSolidFaceWeights`: yüz merkezindeki SDF mesafesi Lipschitz payından uzaksa yüz
  tamamen içte (0) ya da tamamen dışta (255) sayılır, 16 örnek atlanır. Kesin bir sınır,
  yaklaşım değil: dem1'de 16 örnekle karşılaştırıldı, ~400k yüzün ~%83'ü kısayoldan,
  **0 uyuşmazlık**.
- dem1 (hareketli 2.5 m mesh-SDF küre): `sim.fluid.solid_face_weights` 24 → ~8 ms/kare.
  Adım: uyuyan karelerde 55 → ~40 ms.
- Tam GPU taşıması yapılmadı. Ağırlıkları dört host tüketicisi okuyor: yoğun basınç
  statiği yüklemesi, sparse kaynak paketleyici, CPU yolları ve gözenek geri yüklemesi.
- Testler: collider_phi 12/12, mpm_collider 8/8.
- 1. Gözle kontrol: kollider kara değerken temas biçimi öncekiyle aynı mı? Kar ya içine
  sızıyorsa ya da önünde boşluk kalıyorsa, Lipschitz payı kısa kalmış demektir.
  `sdf_lipschitz` 1.1 güvenlik çarpanını büyüt.
- 2. ★ Sinsi durum: ağırlıklar sessizce 0/255'e kayarsa sonuç makul görünür, ama kollider
  yüzeyi bir hücre kalınlaşır ya da incelir. İnce duvarlı mesh'te (kutu kabuğu) suyun duvardan
  geçmediğini kontrol et.

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
