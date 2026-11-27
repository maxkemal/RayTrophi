# Player paketleme — tek ortak çıktı

Kullanıcı kararı, 2026-11-18: Codex ve Claude aynı paketleme betiğini ve aynı
aktif dağıtım klasörünü kullanır. Ayrı tarihli/özellik adlı Player paketleri
paralel dağıtım hedefi değildir. Ajanlar paketlemeden önce
`AJAN_KOORDINASYON.md` §1'e paketleme satırı ekler, bitince kaldırır.

```powershell
./scripts/package_player.ps1
# İstenirse aynı yükün zip'i:
./scripts/package_player.ps1 -Zip
```

Aktif çıktı **`dist/RayTrophiPlayer/`**, arşiv dosyası
`dist/RayTrophiPlayer.zip`. `-OutDir` yalnız açıkça istenen tanı/deneme çıktısı
içindir. Önceki `Player-depth-2026-11-18` dağıtımı `.archive` altına taşındı;
güncel çalıştırma yolu ana klasördeki `RayTrophiPlayer.exe` dosyasıdır.

Varsayılan yük mevcut Release exe ve bağımlılıkları, shaderlar, Gizli Oyuncaklar
sahnesi (`hidden-pictures/hidden_pictures.rtp`), sahne sesleri, taşınabilir
Python, algı köprüsü, MediaPipe/OpenCV ve model içerir. Birden fazla oyun
paketlendiğinde Player açılışta oyun kütüphanesini gösterir. Tek oyunlu paket
doğrudan sahneyi açar. `--project` ile verilen proje doğrudan yüklenir.
Blender, GLB ve glTF kaynak dosyaları dağıtım dışındadır.
Kişisel MP4/kalibrasyon verisi pakete alınmaz.
`-NoSceneAudio` ve `-NoPerception` yalnız özellikle istenen eksiltilmiş paket
seçenekleridir; normal teslimde kullanılmaz.

Betik C++ build çalıştırmaz. Kullanıcının derlediği `x64/Release` çıktısını alır.
Yük önce ayrı geçici klasörde hazırlanır; model/native import kontrolü ve
sahne JSON'undaki tüm ses başvurularının varlık/SHA-256 kontrolleri geçmeden
aktif pakete kopyalanmaz. `package_manifest.json` içerik seçeneklerini ve ses
dosyalarının hash listesini taşır. Eksik ya da sahne dışına çıkan ses yolu hata
verir; sessizce eksik sesli paket başarılı sayılmaz.

Aktif klasördeki `ipc_tokens.json`, `imgui.ini` ve loglar korunur. Zip yalnız
hazırlanan dağıtım yükünden üretilir; bu yerel çalışma dosyaları zip'e girmez.
Varsayılan yenileme aynı klasöre yayın yapar; elle dosya/DLL/model toplamayın.
Paketleme başarısız olursa kalan `.staging-*` klasörü aktif dağıtım değildir.

2026-11-18 birleştirmesi: iki sahnede toplam **10 ses başvurusu** kaynakla hash
eşitliğiyle doğrulandı; algı runtime'ı da ana pakete dahil edildi. Ses çalma için
bu tur Player başlatılmadı; dosya/bütünlük kabulü yapıldı. Önceki MP4/depth canlı
kabulü `PLAYER_DEPTH_PACKAGE_ACCEPTANCE_2026_11_18.md` içindedir.


## Gizli Renkler tek sahne dağıtımı — 2026-11-20

Varsayılan `-Scenes` yalnız pastel-paint-workshop/workshop.rtp; başlık Gizli
Renkler. Başka sahneler kaynak depoda kalır. Yayında doğrulanmış üretilmiş
`dist/RayTrophiPlayer/scenes` ağacı staging sahne ağacıyla değiştirilir; listeden
kalkan sahneler recursive kütüphane taramasında geri görünmez. Mevcut yerel
ipc_tokens.json, imgui.ini, player_operator.json ve loglar korunur. ZIP yalnız
temiz staging yüküdür: taşınabilir roots=[scenes], locale=tr, appearance=camera
başlangıç tercihleri vardır; cihaz kimliği/kalibrasyon/video/profil/token yoktur.

`<stem>.distribution.json`, paket kopyasının game_design/name/description
alanlarına uygulanır; kaynak .rtp'ye yazılmaz. Bu snapshot açık Player'dan
salt okunur `paint.wet_design` ile alındı: 11 balon, 3 gizli obje, walls_floor
ve 3m kamera boşluğu. Varsayılan body_controls=false; yeni lokalize etiket
Kamera / iskelet kontrolü açılınca body girişi de çalışır, fare yedek kalır.
Eski .balloons.json paket kopyası aynı güncel demo config'iyle eşitlenir.

`<stem>.preview.png` kütüphane kartı ve `<stem>.TEST.md` test notu sahneyle
kopyalanır. İlk sahnenin notu ayrıca kökte KAMERA_TESTI.md olur. Yeni kart
imagegen ile üretildi; kasıtlı ürün artwork'üdür, geçici doğrulama görüntüsü değildir.

Çalıştırılan `./scripts/package_player.ps1 -Zip` sonucu: tek sahne, 11 DLL,
396.8 MiB aktif klasör; 191557490 byte ZIP (182.7 MiB). ZIP SHA-256:
`256a4b08ad40ecc91ef01f5c7f76d9d9106d533f8b8ae4eaa9556b028abb3fba`.
4915 yük dosyasında ZIP/aktif klasör hash eşitliği PASS; exe kaynak Release
ile, .bin/.shared ve kart kaynakla eşit; 7 ses başvurusu mevcut ve hash PASS.
EN/TR kataloglar, model/runtime, tek .rtp ve eski sahnelerin yokluğu PASS.
Yerel token/imgui/operator tercih dosyalarının korunumu hash PASS.

Taşınabilir Python native import PASS: Python3.10.6, MediaPipe0.10.35,
OpenCV4.14.0.94, numpy2.2.6. Paket kopyasındaki gerçek pose modeli CPU'da
oluşturulup kapatıldı, PASS; kamera açılmadı. Model SHA-256:
`59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a`.
Fare/gizli obje/F9 kullanıcı tarafından çoklu testte sorunsuz bulundu;
ikili iskelet çekirdeğinin sentetik kabulü de önceki turda PASS. Fiziksel
kamerayla uçtan uca oyun kabulü yapılmadı; bu paket kamera saha testi içindir.
Panelin pozitif köşe koordinatları ile salonun negatif koordinatlarının
hizalanması gerektiği test notunda açıkça belirtilir; gerçek RGB-D sensör
entegresi doğrulanmış diye sunulmaz. C++ build/app launch yapılmadı.
## Shader envanteri

`scripts/player_shader_manifest.json` RayFusion, balon boyası, derinlik kartı
ve sahne efektlerinin doğrulanmış dosyalarını listeler. Paket, zorunlu bir
shader Release dizininde eksikse hazırlığa başlamadan hata verir. Yalnızca
Studio mesh düzenleme, seçili obje konturu ve hair viewport shader'ları
paketten çıkarılır. Henüz sınıflandırılmamış shader'lar ve diğer Release
shader dosyaları taşınmaya devam eder.

Salt okunur envanteri görmek için `./scripts/package_player.ps1 -ShaderInventoryOnly`
çalıştır. Çıktı her SPIR-V dosyasının grubunu ve binary başlığındaki sürümü verir.
Bu komut paket oluşturmaz. SPIR-V sürümü cihazın Vulkan feature ve limit
uyumluluğunu tek başına göstermez.

Manifestte sınıflandırılan 12 zorunlu ve 28 koşullu shader Vulkan 1.2 hedefinde
SPIR-V 1.5 olarak doğrulandı. Yeni Player kaynağı loader'a göre 1.2 veya 1.3
instance ister ve RayFusion özelliklerine sahip adaptörleri seçer. Canlı 1.2
kabulü beklenirken paket dokümanındaki Vulkan 1.3 gereksinimi geçerlidir.
Manifest dışında korunan shader'lar da bulunur.

## Vulkan 1.2 saha kabulü

Testi alıcı kendi makinesinde yapar. Paketi yazılabilir bir klasöre çıkarır,
Player'ı çalıştırır, workshop görüntüsünü, balon patlamasını ve yüzeydeki boyayı
kontrol eder. Kurulu kamera varsa derinlik kartını da dener. Sonucu ve
`SceneLog.txt` dosyasını yeniden başlatmadan önce gönderir. O koşuda
`StartupCrashStack.txt` oluştuysa onu da ekler. Paket README'sindeki EXE
SHA-256 değeri gönderilen sürümü tanımlar.

Tanıda instance API sürümü ile fiziksel cihaz API sürümüne ayrı bakılır.
`[Player GPU]` satırları 15 binding'lik çarpışma ve 17 binding'lik boya
gereksiniminin storage descriptor limitlerine sığıp sığmadığını gösterir.
Ayrıca HDR/LDR, derinlik ve Transmission görüntülerinin kullanım sorgularını,
HDR blending ve Transmission atomik desteğini kaydeder. Sorgu başarısı,
çalışma anındaki bellek tahsisi için ayrı bir kontroldür.
`[Player compute]` satırları spawn, ballistic, collide ve wet paint
shader'larının kurulum aşamasını, SPIR-V sürümünü ve Vulkan sonuç kodunu verir.
`First scene frame uploaded` kaydı GPU kare sayaçlarını, draw sayısını ve
sıfır dışı örnek sayısını verir. Görsel kabulü alıcının sahne gözlemi tamamlar.
