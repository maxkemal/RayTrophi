# Audio arka plan üretim ve analiz işleri

## 2026-10-15 devam: Arrange, kaynak waveform ve tarif geçmişi

Kullanıcı test scriptlerini yapısal çalışmaların sonuna erteledi. Bu revizyonda yeni
test scripti yazılmadı veya canlı probe çalıştırılmadı. Aşağıdaki önceki build/probe
kabulü bu yeni kaynak değişikliklerinin kabulü değildir; yeni kullanıcı derlemesi
ve toplu kapanış doğrulaması bekler.

- `arrange_preview(..., background=true)` immutable assets/clips/tracks/mixer
  snapshot'ından bounded worker üzerinde decode ve mix yapar. UI bu yolu kullanır.
  Varsayılan `false` mevcut synchronous script/IPC sözleşmesini korur. Sonuç
  `arrange_status.job` ve `rendering` üzerinden izlenir; servis transport tick'i
  tamamlanmış güncel sonucu native voice'a uygular. Status salt okunur kalır.
- Stop, Stop All, kurgu/track/mixer ve seçim/aralık değişikliğinin mevcut clear
  yolları pending sonucu iptal eder. Worker decode sınırlarında ve mix döngüsünde
  iptali kontrol eder; native playback nesnelerine erişmez. Apply ayrıca mix
  snapshot'ını ve dosya boyut/mtime değerlerini karşılaştırır. Eski sonuç çalınmaz.
  Decode çağrısının içi kesilemez; sonuç yayınlama hakkı hemen iptal edilir.
- `design_waveform(..., background=true)` ayrı bounded worker üzerinden dosyayı
  çözer ve range envelope üretir. UI her aralık için güncel sonucu bekler; önceki
  aralığın waveform'unu yeni koordinatlarda göstermez. Yanıt queued/running/failed
  veya waveform alanlarıyla `state=ready` döner. Aynı parametrelerle tekrar çağrı
  sonucu alır; yeni parametre veya asset/dosya metadata'sı eski isteği değiştirir.
  Tek çözümlenmiş kaynak paylaşılan salt okunur PCM cache'inde tutulur; pan/zoom
  yeniden decode gerektirmez. Scene clear/restore işi ve cache'i temizler.
- `design_undo(background=true)`, `design_redo(background=true)` ve
  `design_reuse(id=..., background=true)` mevcut render worker'ını kullanır.
  Job cevabı `id` ve hedef `recipe` içerir; mevcut design_job_status/cancel/apply
  ile yönetilir. Undo/redo stack'i yalnız başarılı, hâlâ güncel apply sırasında
  değişir. İptal, failed veya stale sonuç geçmişi tüketmez. UI bu yolu kullanır.
- Tüm seçenekler UI, `rt.audio` ve `audio.*` IPC'de aynı service/core üzerinden
  doğrulanır. `background` boolean olmalıdır; synchronous kullanım korunur.

Designer render/analysis, source waveform ve Arrange için ayrı birer lazy worker
vardır; her biri tek aktif ve tek değiştirilebilir pending istek taşır. PCM ve
snapshot görev tarafından sahiplenilir; scene/UI/native nesneleri worker'a geçmez.
Bu bir realtime DSP callback sistemi değildir. Apply sırasında native voice
değişimi ve Arrange peak/RMS özeti hâlâ servis iş parçacığındadır. Dosyaya save/import,
kesintisiz crossfade/DSP ve ölçülmüş toplam performans bütçeleri açık kalır.

Son toplu kabul listesine eklenen kontroller: Arrange queued→applied ve Stop/scene
değişiminde iptal; pan/zoom ve dosya değişiminde waveform sahipliği; async undo/redo
iptalinde history korunumu; reuse ve synchronous/worker PCM eşitliği. Scriptler bu
kapanış aşamasında genişletilecek. Codex derleme veya uygulama açılışı yapmadı.

2026-10-15: kullanıcı Studio derlemesini tamamladı; yeni dış IPC probu geçti.
Bu belge kesintisiz realtime DSP veya bütün audio yol haritasının
tamamlandığı anlamına gelmez.

Derleme sonrası `rt_test_audio_jobs_ipc.py` açık Studio üzerinde dış süreçten
başarıyla çalıştı: en yeni iş sahipliği, explicit apply, synchronous/worker PCM ve
analiz eşitliği, iptal, atomik doğrulama ve sahne izolasyonu geçti. Probe workspace'i
geri yükledi; dosya/asset oluşturmadı. Manuel UI, dinleme ve performans kabulü açık.

## Ortak çekirdek ve kullanım

Designer Play, Refresh, kontrol bırakma sonrası otomatik preview ve değişmiş tarifi
Save asset ile kaydetme artık üretimi worker'a verir. Analyze görünümü de STFT'yi
worker'da hesaplar. UI, Python ve IPC aynı AudioSoundDesign/AudioDesignJobs yolunu
kullanır; render sonucunun uygulanması ve native playback yalnız servis iş parçacığında
yapılır. Kayıt ve asset import mevcut canonical save işlemini kullanmaya devam eder.

| İşlem | Sözleşme |
|---|---|
| `design_job_start` | `kind=render` için `recipe`; `kind=analysis` için isteğe bağlı `options`. Job ID, revision ve queued/running durumu döner. |
| `design_job_status` | Parametresiz, salt okunur. En son işin id/kind/revision/state alanları; hatada error. |
| `design_job_cancel` | Parametresiz. Pending ve tamamlanmış sonuçları atar; aktif işin sonuç yayınlama hakkını hemen iptal eder. |
| `design_job_apply` | Parametresiz. Tamamlanmış en son sonucu bir kez alır. Render için design, analiz için analysis döner; eski bağlam için stale. Henüz bitmemiş işi beklemez. |

Python'da `rt.audio.<işlem>(...)`, dış süreçte `audio.<işlem>` kullanılır. Start
`SceneWrite|FilesRead`, status `Read`, cancel/apply `SceneWrite` yetkisindedir.
İşler transient'tir; asset, klip, fizik telemetry veya proje verisi oluşturmaz.
Render apply mevcut Designer tarif geçmişine tek başarılı tarif değişimi ekler.

```python
# Dış terminal; uygulamanın embedded script alanında çalıştırma.
from rt_ipc import RtIpc
import time

c = RtIpc()
job = c.call("audio.design_job_start", kind="render", recipe={
    "mode": "synth", "preset": "metal", "duration_seconds": 2,
})
while True:
    status = c.call("audio.design_job_status")
    if status["id"] != job["id"]:
        raise RuntimeError("İşin yerini daha yeni bir istek aldı")
    if status["state"] not in ("queued", "running"):
        break
    time.sleep(0.02)
if status["state"] == "completed":
    result = c.call("audio.design_job_apply")
    assert result["state"] == "applied", result
else:
    raise RuntimeError(status)
c.close()
```

## Sahiplik, doğrulama ve yaşam döngüsü

- Servis başına ihtiyaç anında açılan tek worker, bir aktif ve bir değiştirilebilir
  pending iş vardır. Sınırsız iş/PCM cache'i veya detached thread oluşturulmaz.
  Render ve analiz aynı bounded kuyruğu paylaşır; yeni istek öncekinin yerini alır.
- Tarif, asset listesi ve analiz PCM'i kopyalanır. Worker scene/UI/native ses
  nesnelerine erişmez. Recipe ve analiz seçeneklerinin yapısal doğrulaması yeni işin
  sahipliği alınmadan yapılır; yanlış kind, alan/tip/aralık ve işlemle uyuşmayan
  recipe/options mevcut işi değiştirmeden reddedilir. Decode/DSP hataları failed olur.
- Apply Designer revision ve asset listesini yeniden karşılaştırır. Render girdisi
  dosyaların boyut/mtime değerleri başlatma, worker ve apply sınırlarında denetlenir;
  bu metadata kontrolü kriptografik dosya bütünlüğü garantisi değildir.
- Yeni start eski sonucu yayınlayamaz. Synchronous render/undo/redo/reuse, Stop,
  Stop All, preview stop, scene clear/restore ve servis yıkımı bekleyen işi iptal eder.
  UI kendi başlattığı işin ID'sini ve tarif taslağını kontrol ederek sonucu uygular.
  Script tarafından başlatılan üretimi kendiliğinden uygulamaz.
- completed sonucu almak job status'u consumed yapar; apply sonucu ayrıca
  applied/stale döner. Sonucu almak ikinci kez apply ile PCM'i yeniden uygulamaz.
- İptal DSP döngülerinde ve STFT pencerelerinde kontrol edilir. Decode ve
  Signalsmith exact çağrısı bölünmeyen, mevcut süre/bellek sınırlarına bağlı fazlardır;
  iptal bunların içinden zorla çıkmaz. Servis yıkımı worker'ın çıkışını join ile bekler.
- Önceki audition render boyunca devam edebilir; başarılı apply native voice'u
  durdurup PCM'i değiştirir. Crossfade ve kesintisiz realtime parametre işleme henüz yok.

Mevcut synchronous design_render/design_analyze otomasyon uyumluluğu için korunur;
aynı hazırlama/analiz çekirdeğini kullanır. Source waveform decode/cache, Arrange
miks üretimi, undo/redo/reuse ve dosyaya save/import hâlâ çağıran iş parçacığındadır.
Bu yolların arka plana taşınması ve ölçülmüş performans bütçeleri sonraki alt aşamadır.

## Kabul

Statik denetimde 80 benzersiz dinamik audio işlemi, binding parametre kayıtları,
IPC capability/security mirror, descriptor güncelliği, VS kaynak/header kaydı/XML,
Windows branch delimiter, dosya boyutu, Python syntax, UTF-8 ve diff whitespace
kontrolleri geçti. Bunlar C++ derlemesi veya worker'ın canlı çalışmasının kanıtı değildir.

Codex derleme veya uygulama açılışı yapmadı. Bu audio değişikliği için Studio C++
derlemesi gerekir; yeni audio shader yoktur. Başka alanlardaki shader değişiklikleri
bu belgenin kapsamında değildir.

Derleme sonrası açık Studio'ya dış terminalden:

```powershell
python scripts/test/rt_test_audio_jobs_ipc.py
```

Probe dosya veya sahne asset'i oluşturmaz ve ses çalmaz; UI'nin otomatik işlerini
engellemek için Designer'ı geçici kapatır, workspace'i sonunda geri yükler. Önceki
Designer tarifi varsa geri üretir; başlangıçta boşsa test tarifi transient olarak kalır.
Tarif geçmişini/önceki audition cursor'unu geri kurmaz. Kontroller: yeni iş sahipliği,
explicit apply, synchronous/worker PCM hash ve STFT eşitliği, iptal, hatalı isteklerde
iş durumunun korunumu ve sahne verisinin değişmemesi.

Manuel: uzun/reverb'lü tarif üretirken viewport gezin; art arda kontrol değiştir,
Refresh/Play/Stop ve Stop All dene; eski sonuç son tarifi değiştirmemeli. Analyze'da
FFT/kanal değiştir; eski analiz yeni etiket altında görünmemeli. Save asset sırasında
üretim asenkron olmalı, asset/timeline işlemleri yalnız sonuç hazırken gerçekleşmeli.
Scene değişimi, undo/redo ve uygulama kapanışında worker/native voice ömrünü doğrula.

## 2026-10-15 mevcut kurgu canlı denetimi

Bu yeni kod derlenmeden önce açık boş sahnede mevcut binary dış IPC ile denetlendi.
Track probu CRUD/sıra, gain/pan/mute/solo/routing, export PCM, atomik hatalar ve silme
politikalarında geçti. Ayrı dar tur batch hata atomikliği, move/duplicate/align/update,
gain/pan curve, Arrange loop/ilerleyen cursor/sıfırdan büyük PCM enerjisi ve klip
değişimiyle eski audition'ın durmasını doğruladı. Sonunda assets/clips/tracks sıfırdı.
Bu sonuç yeni worker kodunu, manuel undo/redo'yu, proje round-trip'i veya işitsel
kaliteyi doğrulamaz; M4 kapanışı bu ek kabulleri bekler.
