# H1 — gerçek tane referans testi, 2026-10-05

## Hazır paket / build sınırı

GranularContact: normal Hooke yay/sönüm, geçmişli Coulomb kayma, sınırlı viskoz
rolling torku; sphere/sphere ve analitik plane aynı temas kanununu kullanır.
Ortak temas noktası çift kuvvet/tork dengesini korur. Katı küre I=2/5 mr².
Yerel saturation her tanede ayrı kalır; çift sürtünmesi zayıf yüzeyden gelir.
Bu ampirik karışım, kalibre edilmiş kapiler/çamur modeli değildir.

GranularReference: açık fiziksel id/radius/mass/position/velocity/angular_velocity,
hücre tabanlı 27-komşu aday araması, kimlik bazlı temas geçmişi, her mikro adımda
kuvvet/tork/gravity entegrasyonu. Stiffness/damping/travel dt sınırları vardır;
authored stiffness değiştirilmez. Ayrılan çift geçmişi atılır. Hatalı giriş ve
CPU iş bütçesi aşımı sonucu yayınlamaz. 1..64 sphere, en fazla 2 s, en fazla
1 milyon body-step. Bu sınır kontrollü referans içindir, üretim performansı değildir.

Shared servis: Python `rt.fluid.grain_reference(**kwargs)` ve dış IPC
`fluid.grain_reference`. Strict finite/type/unknown-key/identity doğrulaması,
Read capability, 7 parametreli generated discovery. Çıktı sampled fiziksel state,
kütle, lineer/açısal momentum, kinetik enerji ve temas sayaçlarıdır.
Geçici/saf diagnostic: scene/cache/emitter authoring veya timeline değiştirmez.

## Gerçek canlı test

Kullanıcı normal C++ build yapar; bu granül paketinde shader/ABI/cache değişmedi.
Başka ajanın fog paketinin shader/build gereksinimleri ayrı değerlendirilir.
Uygulama açıkken dış terminal (embedded script değil):

`python scripts/test/rt_h1_grain_reference_ipc.py`

Gate'ler: serbest uçuş g/position/no-spin; oblique iki-tane temas/spin,
linear/angular momentum ve dissipasyon; zeminde sekme; eğimde hareket/rolling;
aynı grupta kuru ve ıslak tane farklı sürtünme; 27 tane yığında dt 1e-4/5e-5
son konum farkı <=2 mm; overlap/r <.15; identity/mass/local S sabit;
API yanlış giriş reddi ve sim.control_state değişmemesi.
Kullanıcı build/canlı IPC: 21/21 PASS. Log aşağıda; production DEM kabulü değildir.

Log `docs/dev/matter_h1_grain_reference_live.json`; Python fizik çözmez,
gerçek uygulamanın yeni C++ çekirdeğini çağırır. Query CPU reference'dır,
Vulkan/MPM/emitter parity veya production DEM kabulü değildir.

İsteğe bağlı `--preview`: tüm sayısal gate'ler geçerse final yığın snapshot'ını
x+6 m'de yeni flat sphere primitive'lerle gösterir. Yeni ayrı test materyalleri
atanır; materyal 0 değiştirilmez. Mevcut sahne silinmez/kaydedilmez/resetlenmez.
Bu statik snapshot'tır, timeline playback değildir. Yeni objeler logda listelenir.

## Kontroller / açık iş

C++ regression kaynakları: matter_granular_contact_test.cpp ve
matter_granular_reference_test.cpp. Derlenmedi/çalıştırılmadı; canlı probe bu
kabulü kullanıcı binary'sinde ölçer. Yeni app binary üzerinde 21 canlı gate PASS. GPU/pore/wet/stage kaynak kontrolleri,
API discovery/capability audit, probe AST ve project XML PASS.

Aktif Matter solver routing değiştirilmedi. Referans fiziksel sphere state'i
MPM taşıyıcı/render child state'inden bağımsızdır. Üretim SoA/cache, flat
TriangleMesh collider adapter, Vulkan DEM ve production authoring/UI/persistence
henüz yoktur. Analitik plane sahne Triangle facadelerinden üretilmez.

Sıra: canlı reference kabul → fiziksel tane SoA/lifecycle + flat collider →
GPU ve ortak UI/Python/IPC/persistence → MPM–DEM dry/pore mass ve linear/angular
momentum korunumlu geçiş → yerel bağ graph/hasar/kopma (kar, çığ, toprak,
heyelan). Hiçbir grubu zorunlu homojen S/temperature/damage ile temsil etmeyiz.
Büyük normal dönüşlerinde frame-history transport, twisting/spinning ve kapiler
bağlar açık. G2 kuru yığın yerleşme/dt RED bu testten bağımsızdır.

Model ailesi: [LAMMPS granular contact](https://docs.lammps.org/pair_granular.html).
Buradaki rolling modeli SDS değildir; JKR/DMT veya bağ kopması uygulanmadı.


## Son canlı kabul — 2026-10-05

21/21 PASS; 27 tane, .4 s, dt 1e-4/5e-5 son konum max fark .284559 mm.
Pair linear momentum sıfır, angular momentum sapma ~1.54e-7 kg*m²/s;
çarpışma sonrası iki tanede spin ~9.18 rad/s. Aynı gruptaki dry/wet kayma
son vx 1.40447/1.95033 m/s (ilk hız 2 m/s), saturation 0/1 ayrı/sabit.
Pile overlap/r max .081526. Kütle ve kimlik sabit; scene control değişmedi.
Bu kısa transient kabulüdür; uzun yerleşme/repose veya MPM routing kabulü yok.

İlk koşu 20/21: sekme k=10 kN/m ile overlap/r .154674, dt-half .154653.
Bu ölçüm arşivlendi, gate gevşetilmedi. Sekme fixture k=20 kN/m, kt=5 kN/m
ile .109114; dt-half .109102. Diğer testlerde default k korunur. Solver kaynak
kodu değişmedi; yeni build gerekmez. Analitik yay sıkışması ile tutarlı sertlik
kalibrasyonu, otomatik Young/stiffness cap veya numerik tolerans değişikliği değil.

Loglar: matter_h1_grain_reference_live.json (PASS),
matter_h1_grain_reference_first_live.json (ilk RED),
matter_h1_bounce_stiffness_diagnostic.json (kontrollü k/dt karşılaştırması).
Boş sahneye 27 grain + yeni zemin/ayrı materyaller eklendi, kamera odaklandı;
proje kaydedilmedi. Önizleme static final snapshot, timeline DEM emitter'i değil.
Fog/volume shader kaynaklarına dokunulmadı.


## Göz testi — dış IPC tekrar

`python scripts/test/rt_h1_grain_visual_ipc.py --loops 3`

Mevcut 27 test grain'ini kullanır; C++ referansta .4–.61 m başlangıç yüksekliği,
1.2 s düşüş/dağılma klibi hesaplar, dış IPC batch transform ile gösterir.
Temas k=40 kN/m, kt=10 kN/m; bu ayrı görsel fixture'dır, kabul fixture'ı değişmez.
Her tekrarın başı/sonu 1 s bekler; tamamlanınca ilk poz bırakılır. İstemci render
hızını garanti etmez; duvar saati replay hızı numerik sim hızının ölçümü değildir.
Timeline Play ile bağlı değildir; Python fizik çözmez, C++ sampled state replay
eder. Scene/cache/sim driver resetlenmez, proje kaydedilmez.
`matter_h1_grain_visual_live.json` input, trajectory ve replay durumunu saklar.
