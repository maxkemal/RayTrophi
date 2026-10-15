# UVW canlı sonuç ve proje açılışında timeline reset

## Yeni kullanıcı binary'sinde ölçüm

İlk snapshot playhead=0/runtime particles=0 ve son üretilmiş SDF stats'ını
gösterdi; stale stats kabul edilmedi. 61. kare istendiğinde ilk query runtime
henüz 6,666 parçacıktaydı. İlerleme tamamlanıp 25,416 parçacığa ulaştıktan sonra
karşılaştırma yapıldı. UI playhead hedefe geçmiş olması runtime'ın aynı anda
tamamlandığını tek başına kanıtlamaz.

Aynı durmuş fizik state'inde position_smoothing=.9/0/.9/0; position_hash ve
velocity_hash dört ölçümde aynı. Capture kapalı süre medyanları:

| Render merkez filtresi | UVW | SDF producer profiler | Render sync parent |
|---|---:|---:|---:|
| .9 | 16.23 ms | 16.01 ms | 54.77 ms |
| 0 | 13.22 ms | 10.01 ms | 49.34 ms |

Önceki binary, aynı nominal sahne/kare/25,416 parçacık/default koşulda UVW
46.5–50.8 ms ve render sync 76.7–82.9 ms ölçmüştü. Dolayısıyla UVW maliyeti
belirgin azalmış görünüyor. Binary'ler arası particle hash birebir aynı değil;
yaklaşık karşılaştırmadır, kesin kontrollü hızlanma yüzdesi değildir. Yeni
filtreden gelen yaklaşık 6 ms SDF maliyeti kazancın bir kısmını tüketiyor.
Restore, physics, fog ve volume publish ayrıca maliyet taşır; kullanıcı
toplam playback'te aynı oranda değişim görmeyebilir. Parent/child süreler
üst üste eklenmez.

Görüntü A/B'de filtre silueti biraz toparlıyor/daraltıyor, fakat pütürlü sütun
karakterini belirgin biçimde çözmüyor. Kullanıcının “kalite farkı algılamadım”
gözlemi destekleniyor; kalite kabulü açık. Bu turn'de başka bir reconstruction
formülü veya default filtresi eklenmedi. Raw=.0 yolu hâlâ kullanılabilir.

Ham kayıt/görüntü:
[uvw_filtered_live_frame61_2026_10_10.json](uvw_filtered_live_frame61_2026_10_10.json),
filter .9 (temporary capture removed),
raw .0 (temporary capture removed).
İki karşılaştırma sonunda position_smoothing=.9 ve capture=false geri
yüklendi; sahne 61. karede durmuş bırakıldı.

## Timeline kaynak düzeltmesi

ProjectManager model timeline ve render settings karelerini zaten 0'a
alıyordu. Ancak TimelineWidget'ın private current_frame'i eski projeden
kalabiliyor; draw sırasında scene.timeline.current_frame'e geri yazılıyor.
Main'de yalnız async-finalize yolunu düzeltmek de doğrudan/senkron project
load yollarını kapsamaz. Saved UI layout restore resetten sonra yapılırsa
aynı eski UI state'i tekrar getirir.

Yeni ProjectTimelineReset modülü: newProject tamamlanınca ve başarılı
openProject sonunda loader-safe atomic reset debt'i yayımlanır. SceneUI draw
ana thread'inde, pending UI layout restore'dan SONRA, timeline paneli ve
processAnimations/simulation güncellemesinden ÖNCE tüketilir. Widget stopped
frame 0, model frame 0, context/global render current frame 0 ve playing=false
birlikte ayarlanır. Loader thread widget'ı değiştirmez.

Widget'a project rest-frame kapısı eklendi. Saved authored start_frame>0
olsa bile ilk durmuş proje frame 0'da tutulur; toolbar'ın unconditional range
clamp'i otomatik ileri sarmaz. Play veya nonzero frame seçildiğinde normal
range clamp geri gelir. Authored frame range ve keyframe'ler silinmez.
Yeni UI capability değil; bütün existing native/new/open/script/IPC project
servisleri ProjectManager'ın aynı reset debt'ini kullanır.

Source/math/XML reset kontrolü PASS; yeni timeline modülü için C++ build
yapılmadı. Kullanıcı derlemesinden sonra paused/playing eski kare 63/123 ile
UI Open/New, recent/template/sync path ve IPC open test edilir. İlk sim
güncellemesinde widget/model/settings frame 0 ve paused olmalı; cache eski
kareye kadar üretilmemeli. Authored range start>0 ile de 0 korunmalı; ilk Play
ve explicit seek normal çalışmalı. Source kontrolü native kabul yerine geçmez.
