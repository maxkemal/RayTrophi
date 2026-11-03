# UV Editor canlı test ve tools düzeltmesi

Tarih: 2026-11-03. Kullanıcı kendi build'ini aldı; açık sahnede `Default_Cube` vardı.
Testler uygulamayı açmadan, dış süreçte `scripts/test/rt_ipc.py` named-pipe istemcisiyle
çalıştırıldı. Sandbox error 5 sonrası probe gerekli escalation ile başarıyla çalıştı.

## Çalışan binary üzerinde doğrulananlar

- İlk `rt_test_uv_authoring_ipc.py` probe'u geçti: ada izolasyonu, sayfalama,
  texture değişimi tercihi, stale revision, yanlış ada kimliği ve undo/redo.
- Ek geçici küpte non-uniform scale sonrası uniform normalize oranı korudu;
  normalize undo eski UV'leri aynen geri getirdi.
- xatlas capability true. Pack ve auto unwrap ayrı ayrı başarıyla tek atlas üretti;
  UV'ler finite ve 0–1 sınırında, vertex/triangle sayıları korundu.
- `rt_test_uv_python.py`, dış IPC `script.run_file` çağrısıyla geçti. Bu script
  pipe istemcisi kullanmaz: yalnız rt.uv sorguları, no-op transform ve hata
  semantiğini test eder. Stale write RuntimeError, NaN JSON ValueError olarak reddedildi.
- Kullanıcının `Default_Cube` UV revision'ı test öncesi/sonrası aynı kaldı.
  Testler yalnız benzersiz geçici küpler üzerinde geometri değiştirdi; bunlar
  `scene.delete` ile silindi. SceneHistory/projenin modified durumu testten etkilenir.
- Mevcut UV paneli computer-use ile görsel incelendi; sabit genişlik, sayısal
  alan etiketleri ve boyutlandırma eksikliği doğrulandı. Repo'ya screenshot kaydedilmedi.

## Canlıda bulunan hata ve kaynak düzeltmesi

Silinen geçici küpler `uv.targets` listesinde kalıyordu. Motor undo için onları
world.objects içinde pending-delete olarak tutuyor; UV servisi bu durumu
filtrelemiyordu. `UvService.cpp` hedef keşfi/çözümlemesi ve shared-geometry kontrolü
artık `isEditorPendingDeleteObjectName` kullanıyor. Mevcut derlenmiş binary'de hata
yeniden üretildi; düzeltilmiş davranışın canlı kabulü sonraki kullanıcı build'ini bekler.
Probe artık silinen hedefin listeden çıktığını ve uv.info'nun onu reddettiğini de denetler.

## Island Tools kaynak değişikliği

- Tools/canvas arasında sürüklenebilir ayırıcı; geniş pencerede yatay, dar pencerede
  dikey boyutlandırma. Fonta göre alt sınırlar ve canvas alanı korunur.
- Çift tıkla varsayılan boyuta dönüş; Tools kutusuyla alanı gizleme/açma.
- Ortak PropertyLayout satırlarıyla etiket/sayı hizası, katlanabilir Transform/Atlas,
  tam genişlikte işlem butonları, pivot alanı ve transform alanlarını sıfırlama.
- Toolbar dar alanda satıra geçer. Frame gerçek canvas boyutuna göre hesaplanır;
  hedef açılırken ilk görünüm otomatik çerçevelenir.
- Splitter parent cursor'ı tükettiğinden her iki child, önceden kaydedilmiş
  başlangıç konumuna açıkça yerleştirilir; UI_LAYOUT_GUARDRAILS uygulanır.

Önceki dış private çağrı hatası: UvEditorUI, SceneUI::dockToBottom'a doğrudan erişiyordu.
Kullanıcının `scene_ui.h` üzerinde yaptığı public erişim düzeltmesine tekrar dokunulmadı.
Artık SceneUI kendi üye fonksiyonu içinden docking callback'i veriyor;
UvEditorUI.cpp ortak scene_ui.h'ı hiç include etmiyor. Dar UvEditorUI.h sözleşmesi
güncellendi. Geniş ortak header'ı değiştirmekten doğan ek yeniden derleme ihtiyacı yaratılmadı.

## Son kontrol durumu

`python -B scripts/test/check_uv_authoring_contracts.py`: 5/5 geçti.
`git diff --check`: ilgili değişikliklerde whitespace hatası yok.
Build, shader compilation ve uygulama yeniden başlatma çalıştırılmadı.

Yeni build sonrası gerekli kabul: splitter sürükleme/çift tık, dar pencere ve Tools
gizleme/açma, dock/float açılışı, sayısal alanlar/pivot ve genişletilmiş IPC probe'u.
Yeni arayüz kodu henüz derlenmediği için bu görsel davranışlar canlıda doğrulanmış sayılmaz.
