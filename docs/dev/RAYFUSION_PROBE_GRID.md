# RayFusion — probe penceresi çalışma zamanı değeri oldu

> **Durum:** AKTİF — YAZILDI, DERLENMEDİ (2026-09-08). Kaynak denetimleri geçti
> (`audit_rayfusion_probe_grid.py` + `audit_ipc_capabilities.py`). Görsel kabul
> ve GPU ölçümü yapılmadı. Bu dilim bir kalite iddiası değil, **ölçüm zemini**.

## Neden bu, ve neden şimdi

Probe alanı bugüne kadar derleme sabitiydi:
`4×2×4` hücre, `spacing 3`, en küçük hücre `(-2,-1,-2)`. Yani dünya orijinine
çivili `12×6×12`'lik bir kutu. Ölçülmüş sonucu ([NEXT_BUILD_CHECKS](NEXT_BUILD_CHECKS.md)
2026-09-08 kaydı):

- Oda `x ∈ [0,5]`, `z ∈ [0,6]`; alan `x,z ∈ [-6,6)`, `y ∈ [-3,3)`.
- `y` katmanları −1.5 ve +1.5 → 32 probe'un pratikte **16'sı** iş görüyor.
- Bu 16'nın **6'sı binanın dışında**, açık havada. `hit_fraction = 0.105`.

Açık duran her görsel soru — kıvrımlardaki mavi tabaka, bounce'ın görünmemesi,
`alpha_tested = 0` — bu handikapla ölçüldü ve hiçbiri kapsamdan ayrıştırılamadı.
[RAYFUSION_PROBE_SAMPLING.md](RAYFUSION_PROBE_SAMPLING.md) bunu açıkça yazıyor:
"kök neden görüntüyle ayrıştırılmadı".

Ayrıştıramamanın sebebi teşhis eksikliği değildi: **değişkeni oynatmak yeniden
derleme istiyordu.** Kural 1'in tam da engellemek için var olduğu durum bu —
panelden bile erişilemeyen, yalnız derleyiciden erişilebilen bir parametre.

## Ne yapıldı

`rayfusion.set_probe_grid` / `rt.rayfusion.set_probe_grid()` ve panelde
**Probe window** düzenleyicisi. Beş dokunuş tam:

| Katman | Dosya |
|---|---|
| Değer tipi | `RayFusion/ProbeField.h` — `GridRequest`, `kMaxProbeSlots` |
| Çekirdek API | `Api/RtApiRayFusion.h` + `RtApiRayFusion.cpp` |
| Backend | `IBackend.h` sanal + `MaterialPreviewProbeField.cpp` sahibi |
| IPC | `RtIpcRayFusion.cpp` |
| Python | `RtPythonRayFusion.cpp` |
| Yetki | `RtIpcSecurity.cpp` → `Render` (+ `audit_ipc_capabilities.py` aynası) |
| Ajan tarifi | `ipc_descriptor_overlay.json` → üretici çalıştırıldı |
| Panel | `UI/rayfusion_status_panel.hpp` |

**Varsayılan değişmedi.** `4×2×4`, `spacing 3`, `(-2,-1,-2)`: hiç kimse
`set_probe_grid` çağırmazsa alan bit bit eskisiyle aynı. Bu dilim hızdan bir şey
götürmez; bir kadran açar.

### Sözleşme

- **Kısmi düzenleme.** Göndermediğin alan neyse o kalır. Pencereyi taşımak için
  şeklini yeniden yazmak zorunda kalmak, dokunmak istemediğin bir sayacı sessizce
  sıfırlamanın yoludur.
- **Yerleşim ya `minimum` (hücre) ya `center` (dünya birimi).** İkisi birden
  hata — kimsenin hatırlamayacağı bir öncelik kuralı değil. `center`, **nihai**
  spacing ve counts ile çözülür; bu yüzden çağıran kendi çeviremez (aynı çağrıda
  spacing değişiyorsa eski spacing ile çevirirdi).
- **Açık yerleşim `follow_camera`'yı KAPATIR.** Aksi halde takip bir sonraki
  karede üzerine yazardı ve kabul edilmiş istek hiçbir şey yapmamış görünürdü.
- **Şekil değişimi (counts/spacing) her ölçümü düşürür.** Hücreler artık farklı
  dünya hacimleri; eski paketleri taşımak, başka bir yer için ölçülmüş değeri
  geçerli sanmak olurdu.
- **Yalnız yerleşim değişimi kaydırır** (`scroll`): pencerede kalan dünya
  hücreleri değerini korur, yalnız yeni açılanlar yeniden planlanır.
- **Fail-closed.** Reddedilen istekte hiçbir alan yazılmaz; `error` hangi alanın
  yanlış olduğunu söyler. Denetim script'i her `return false;`'un ilk yazmadan
  ÖNCE olduğunu doğruluyor.
- Rapor edilen pencere **UYGULANMIŞ** olandır, istenen değil — ve viewport bir
  kare üretene kadar hâlâ eskisidir.

### Maliyet — ölçülmedi, ama sınırları belli

- **Kare başına ışın maliyeti ızgarayla büyümez.** Parti başına probe sayısı
  `min(maxProbes, maxRays/raysPerProbe)` ile bütçeden gelir; slot sayısı buna
  girmez. Büyük pencere **yakınsama süresi** öder (valid → total daha çok kare).
- **Tampon bir kez tavana göre ayrılır** (`kMaxProbeSlots = 1024` → 2 MiB).
  Izgara değişiminde yeniden ayırma yok: uçuştaki bir karenin okuduğu tamponu
  serbest bırakmak ve descriptor'ı geride bırakmak bu deponun bilinen tuzağı.
- **Yükleme AKTİF pencereye göre boyutlanır.** 32 slotluk ızgara, 1024'lük
  ayırma yüzünden 2 MiB kopya ödemez.

### ⚠ Bu dilimin ödemediği borç

Her yayın partisinden önce `drainInteractiveViewportInFlight()` çağrılıyor
(`m_rasterFrameRing->waitAll(true)` — tam ring drenajı) ve ardından aktif
pencerenin tamamı yükleniyor. Bugün görünmez: alan iki partide dolup uyuyor.

**Kamera takibi sürekli açıkken bu görünür hale gelir** — hareket boyunca her
kare yayın olur, her yayın kare halkasının paralelliğini sıfırlar. Yani
"bütçeli otomatik kapsam" açılmadan önce yükleme yolu değişmeli: yalnız değişen
slotları, drenajsız, kare başına bir bölgeye yazan bir yol. Bu dilim onu
**yapmadı**; ızgarayı büyütmek bugün elle ve ölçüm amaçlı yapılıyor.

## Sıradaki iş bu dilim değil, ÖLÇÜM

Bu kadran şunun için açıldı: aynı sahnede pencereyi odanın üstüne oturtup
`hit_fraction`'ın 0.105'ten nereye gittiğine bakmak. O tek ölçüm, otomatik
yerleşimin (yoğunluk seçimi, duvar içindeki probe'ları taşıma) gerekip
gerekmediğini ve hedefini söyler. Şu an o iş **sezgiye göre** yazılacaktı.

Sıralı kabul listesi: [NEXT_BUILD_CHECKS.md](NEXT_BUILD_CHECKS.md).

## İlgili

- [RAYFUSION_RENDERER.md](RAYFUSION_RENDERER.md) — faz tablosu ve dikiş sırası
- [RAYFUSION_PROBE_SAMPLING.md](RAYFUSION_PROBE_SAMPLING.md) — 8 komşu tüketicisi
- [RAYFUSION_SPECULAR_VISIBILITY.md](RAYFUSION_SPECULAR_VISIBILITY.md) — speküler sky görünürlüğü
- [RAYFUSION_PROBE_OVERLAY.md](RAYFUSION_PROBE_OVERLAY.md) — işaret çizimi ve kamera takibi
