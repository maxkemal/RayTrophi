# Karakter silme ve Player klip oynatma

> Durum: PLAYER KLİP / ÖLÇEK CANLI PASS — 2026-11-16. Karakter silme/undo kabulü ayrı ve bekliyor.

## Bulgu

`scene.delete` bir mesh düğümünü siler. Karakter context'i ve ona bağlı iskelet/
klipler için bir silme işlemi yoktu. Bu yüzden mesh'ten bağımsız karakter kaydı
Hierarchy ve animasyon listesinde kalabiliyordu. Bu, skin'in görünmemesinin
nedenini tek başına açıklamaz; o sorun henüz görsel olarak doğrulanmadı.

Player `advanceRuntime` simülasyonu yürütüyor, Renderer animasyon değerlendirmesini
çağırmıyordu. Yüklenen yürüyüş klibi ve skin verisi bu nedenle otomatik ilerlemiyordu.

## Ortak işlem

Hierarchy / Characters / Models altında **Delete character and meshes** düğmesi,
Python `rt.anim.delete_character(character)` ve IPC
`anim.delete_character {"character":"1"}` aynı `rtapi::deleteCharacter` hizmetidir.
Bu işlem Studio'da SceneWrite yetkisi ister; kiosk Player silme yetkisi genişletilmedi.

Tek undo komutu karakter context'ini, canonical flat mesh parçalarını, karakterin
kemik haritalarını, kliplerini, paylaşılmayan graph asset'ini, collider setlerini,
ilgili timeline/group referanslarını kaldırır. Diğer karakterlerin kemik indeksleri
yeniden numaralanmaz. Kaynak FBX değişmez; materyal/tekstür kütüphanesi silinmez.
Save-time mesh compaction sonrasında da mesh/context geri alımı için referanslar tutulur.
Kaynakta ayrı bir per-face geometri yolu oluşturulmadı.

Boş/bulunamayan karakter `character_not_found`; bağlı olmayan API, aktif final
render veya history yokluğu mevcut hata semantiğini izler. Dış beden pozu bağlantısı
varsa `character_external_pose_active_clear_first`: önce açıkça
`anim.clear_external_pose` çağrılır. Undo transient tracking'i sessizce kaybetmez.
`scene.delete` tek mesh silme anlamını korur.

## Player oynatma

Player kendi geçen zamanını ortak `Renderer::updateAnimationState` hizmetine verir.
Animasyon değerlendirmesi fizik tick'inden önce, flat skin güncellemesi raster
deform refit tüketiminden önce çalışır. Yazılmış graph/controller seçimi korunur;
ikinci klip oynatıcısı oluşturulmadı. Statik mesh'ler skin refit listesine eklenmez.
`agent.get_state_summary.runtime` içindeki `animation_playback`, `animation_ticks`,
`animation_updates`, `raster_refits` alanları kabulde okunur. Ortak viewport skin hizmetinin mevcut GPU/CPU fallback seçimi kullanılır; yeni
bir skin motoru veya düşük maliyet iddiası değildir. Maliyet ayrıca ölçülmelidir.

## Build ve kabul

Studio projesine bir yeni C++ birimi kaydedildi; Player projesi generator ile güncel.
İki hedef için proje reload + **artımlı build yeterli**. Özel `RuntimeSceneViewer`
sınıfına saat/sayaç üyeleri eklendi; shader/GPU ortak struct düzeni değişmedi.
Codex uygulama başlatmadı ve C++ build yapmadı; `/Zs` sözdizimi ve kaynak denetimleri
kullanıldı. Çalışan eski ikiliye yeni davranış uygulanmış sayılmaz.

Yeni build sonrası, açık uygulamalarda harici terminalden:

```powershell
python scripts/test/rt_character_lifecycle_ipc.py --delete 1 --keep 2 --output tools/perception_bridge/out/character-delete-acceptance
python scripts/test/rt_player_character_animation_ipc.py --character 1 --output tools/perception_bridge/out/player-animation-acceptance.json
```

İlk test iki karakterli Studio sahnesi ister; test sırasında sahneyi düzenlemeyin.
Yeni çıktı dizinine backup alır; geçersiz istek, silme, save/compaction, undo, redo,
reopen ve diğer karakterin kemik slotlarını doğrular. Sonunda backup'ı tekrar açar.
Kabul sonrası yalnız istenmeyen karakter ayrıca silinir. İkinci test yürüyen
karakterli Player sahnesinde eklem ilerlemesini ve raster refit'i ölçer; görünen
mesh deformasyonunun kalitesi ayrıca viewport'ta kontrol edilir. Bu canlı testler
bu kaynak paketinde henüz çalıştırılmadı.

## Link düzeltmesi

2026-11-15: Undo yolundaki iki özel SceneCommand görünürlük yardımcısına dış
sembol başvurusu kaldırıldı. İki hedefte kayıtlı ortak `rtapi::setObjectVisible`
kullanılıyor; flat görünürlük ve backend maskeleri aynı hizmetten güncelleniyor.
Artımlı build yeterli; derleme sırası bağımsız. Link kabulü kullanıcı build bekliyor.

## Player Rest varsayılanı — 2026-11-16

Kullanıcı build sonrası canlı IPC: animation_playback=true, animation_ticks=1232,
animation_updates=0, raster_refits=0; 79 ağırlıklı kemik ve bir klip yüklü. Ortak
`initializeRestPoseDefaults`, skinned karakteri Studio Rest görünümüne alıyordu;
Renderer bu görünümde klip değerlendirmesini atlıyor. Player `prepareScene` artık
hasAnimation karakterleri mevcut `setPoseView(..., "animated")` hizmetiyle başlatır.
Studio Rest varsayılanı ve klipsiz rig'ler korunur. IPC play komutu gerekmez.
Bu ek düzeltme için yalnız Player artımlı build ve projeyi yeniden açma gerekir;
yeni davranışın canlı kabulü henüz yapılmadı.

## Görünür skin / ölçek düzeltmesi — 2026-11-16

Canlı Player: graph pozu, baş yüksekliği yaklaşık 1,84 m; animation_updates=2518,
raster_refits=2517. Bu sayaçlar görünür mesh kabulü değildi. Statik SoA refit
`P_orig/N_orig` yüklüyordu; skinned FBX için bunlar ham bind koordinatlarıydı
(bu dosyada cm). Kemikler doğru ilerlerken raster mesh bind pozunda ve dev görünüyordu.
Player artık ortak Renderer ile kemikleri CPU skin uygulamadan değerlendirir,
statik refit'e animasyon mesh listesi eklemez; deform/instance servisinden sonra
Studio'nun da kullandığı `syncRasterSkinnedVertices` hizmetini çağırır. Bu hizmet
bind buffer + bone matrisi ile mevcut GPU skin ve gerektiğinde mevcut CPU fallback'i
kullanır. `animation_skin_sync_calls` yalnız hizmet çağrı sayısıdır, görünür sonuç
veya GPU dispatch başarısı kanıtı değildir. Test bu sayacı ve eklem ilerlemesini
kontrol eder; viewport'ta karakter boyutu ve yürüyüş ayrıca doğrulanmalıdır.
Shader değiştirilmedi. Player artımlı build; canlı kabul henüz bekliyor.

## Gün sonu canlı kabul — 2026-11-16

Kullanıcı Player'da doğru boyut ve görünür yürüyüşü doğruladı. Harici IPC yürüyüş
testi PASS: graph pozu, 61 yeni animasyon güncellemesi ve 61 ortak skin hizmeti
çağrısı. Gerçek kare testi PASS: kısa yakalama penceresinde 22 submit / 22 consume,
22 animasyon/skin çağrısı ve değişen görüntü imzası; GPU kaybı/simülasyon hatası yok.
Yakalama eski false ayarına döndürüldü, scene revision değişmedi. Balon demo aktif,
56 olay görülmüş; boya partikül havuzları mevcut GPU-resident yolu kullanıyor.
Bunlar performans/maliyet kabulü değildir; arka plan çizim kapısı ve geçici yakalama
pacing'i nedeniyle FPS karşılaştırması yapılmadı. Kayıt:
[PLAYER_CHARACTER_ACCEPTANCE_2026_11_16.json](PLAYER_CHARACTER_ACCEPTANCE_2026_11_16.json).

Mimari kural: Studio'nun mevcut çekirdek hizmetleri kullanılacak; CPU/OptiX/OIDN gibi
Player'da desteklenmeyen bağımlılıklar hedef yapılandırmasında ayrılacak. Player'a
özgü olmayan yeni iş mantığı ortak kalacak. Bu paket yeni animasyon/skin motoru
kurmaz; ortak Renderer, pose-view ve raster skin hizmetlerini host döngüsüne bağlar.

Sonraki gün: imported rig owned-IK devralma engeli, sonra dış poz/domain kabulü;
karakter silme/undo/reopen testi; Player IPC klip/geçiş kontrolü ve kapanış sürelerini
ayrı ölçme. Bu işler bugünkü Player klip PASS sonucuna dahil değildir. Açık sahne
silinmedi/kaydedilmedi, uygulamalar başlatılmadı/kapatılmadı, build/commit yapılmadı.
