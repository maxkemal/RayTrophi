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

Varsayılan yük: mevcut Release exe ve bağımlılıkları, shaderlar, iki sahne ve
yan dosyaları, **sahnelerin ses klasörleri**, taşınabilir Python, algı köprüsü,
MediaPipe/OpenCV ve model. Kişisel MP4/kalibrasyon verisi pakete alınmaz.
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
