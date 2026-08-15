# Sıradaki derlemede kontrol listesi

> **Durum:** CANLI — Her is partisinde bastan yazilir; yalnizca en son parti gecerlidir.

**Parti: Template Hub — Faz 2B.2b-i (dürüst reddetme) + varlık lisans kaydı.**

Tek konu: *preflight'ın verdiği söz tutuyor mu, ve hiçbir başarısızlık yolu
aktif sahneyi siliyor mu.*

Sıralama ölçütü: **bağımsız ve hızlı görülen önce**, diğerlerinin sonucunu
maskeleyen sonra.

---

## Neden bu değişiklik yazıldı

`prepare()` tek bir şey vaat eder: **ready ⇒ open() kabul eder.** Bu vaat
yalnız teamülle ayaktaydı, çünkü desteklenen preset listesi **iki yerde** ayrı
ayrı duruyordu:

| Yer | Rol |
|---|---|
| `TemplateLoader::preflightRecipe` | `ready` der |
| `TemplateRecipeStager::stage` | commit eder |

Birine preset eklenip diğerine eklenmezse **hata alınmaz**: `prepare` `ready`
döner, `open` `recipe_commit_not_available` ile reddeder. Faz 4'te altı preset
eklenirken bu tuzağa basılması an meselesiydi. Liste artık **tek kanonik
tabloda** (`kPresetBuilders`) ve `stage()` dispatch'i de aynı tablodan sürülüyor
— ayrışma yapısal olarak imkânsız.

İkinci kök: `preflightProject`, `openProject`'in **gerçek** kabul şartından
gevşekti. `openProject` yalnız `is_v3 && has_geometry && has_binary` yolunu
yükler (`is_v3` = **tam** `"3.0"` eşitliği); geri kalan her şey legacy dalına
düşüp `false` döner — ama bunu `newProject()` sahneyi **çoktan sildikten sonra**
yapar ([ProjectManager.cpp:1854](../RayTrophiStudio/source/src/Core/ProjectManager.cpp#L1854),
[2470](../RayTrophiStudio/source/src/Core/ProjectManager.cpp#L2470)). Preflight
ise `rfind("3",0)` ile `"3.1"`i geçiriyor ve `has_geometry=false`'u hiç
sorgulamıyordu ⇒ **`ready` dediği bir proje sahneyi silip başarısız oluyordu.**

---

## Sıra

### 1. Derleme (bağımsız, en hızlı)

Dokunulan dosyalar: `Template/TemplateRecipeStage.h`,
`Template/TemplateRecipeStage.cpp`, `Template/TemplateLoader.cpp`.

**Yeni .cpp YOK ⇒ vcxproj değişmedi. Yeni IPC metodu YOK ⇒ capability
değişmedi.** `audit_ipc_capabilities.py` çalıştırmaya gerek yok.

- **Görmen gereken:** temiz derleme.
- **Bozuksa ne demek:** muhtemelen include eksiği — `TemplateRecipeStage.cpp`
  `<iterator>` (std::size), `TemplateLoader.cpp` `<array>` + `<cstring>`
  (memcmp) ekledi.

### 2. Sağlıklı yol bozulmamış olmalı (maskeleyeni ÖNCE ele)

Uygulamayı aç. Açılış **General Scene** template'i üzerinden geliyor.

- **Görmen gereken:** eskisiyle **birebir aynı** açılış — Default_Cube seçili,
  key light, kamera küpü çerçeveliyor, Solid/Rendered geçişi çalışıyor.
- **Bozuksa ne demek:** preset dispatch'inin tabloya taşınması sırasında bir
  build fonksiyonu yanlış bağlandı. `empty` ve `general_scene` davranışı
  değişmemeliydi; bu parti **hiçbir sahne içeriğini** değiştirmedi.

### 3. Preset paritesi (yeni assertion)

```powershell
.\scripts\ipc\Start-RayTrophi.ps1
```
sonra gömülü test: `scripts/test/rt_test_templates.py`

- **Görmen gereken:** `[rt.templates] PASS - N discovered, M valid, 2 recipe parity`
  ve canlı named-pipe suite'in tamamı **0 FAIL**.
- **Bozuksa ne demek:** `prepare() ready dedi ama open() reddetti` mesajı
  çıkarsa tek kanonik liste bağlanmamış demektir — `preflightRecipe` hâlâ kendi
  kopyasını kullanıyordur.
- **`recipe parity` sayısı 2'den küçükse:** registry recipe template'lerini
  keşfetmiyor; asıl arıza 3. maddede değil registry'de.

### 4. Reddetme mutasyonsuz mu (asıl garanti)

Sahneye elle birkaç nesne ekle, sonra kaydetmeden bir template açmayı dene:

```powershell
Invoke-RtIpc templates.open @{ template_id = 'raytrophi.start.empty' }
```

- **Görmen gereken:** `unsaved_changes` kodu ile **ret**, ve sahnen **olduğu
  gibi** duruyor — tek bir nesne bile kaybolmamış.
- **Bozuksa ne demek:** `reject` politikası mutasyon sınırının yanlış tarafında;
  bu, partinin tek gerçek garantisinin çökmesi demektir.

### 5. ★ EN SİNSİ: project preflight'ı bu partide HİÇ KOŞMUYOR

★★★ Bu partinin en riskli kısmı **çalıştığını hiçbir testin göstermemesi.**
Depoda **tek bir project-backed template yok** (`empty` ve `general_scene`
ikisi de recipe). Yani sıkılaştırılmış `preflightProject` kodu **hiç
çağrılmıyor**; 1–4 arasındaki her madde geçse bile o kod yanlış olabilir ve
kimse fark etmez. Faz 2B.2b-ii'ye gelindiğinde kod "eski ve güvenilir"
görüneceği için bir daha da sorgulanmaz.

Bunu kapatmak için elle bir tuzak paket kur — `assets/templates/_probe_project/`:

```json
manifest.json  →  "scene": { "type": "project", "path": "scene.rtp" }
scene.rtp      →  { "format_version": "2.0", "has_geometry": true }
```

`templates.refresh` sonra `templates.prepare` çağır.

- **Görmen gereken:** `ready=false`, kod **`unsupported_project_version`**, ve
  hata metninde `found: 2.0`. Sahne değişmemiş olmalı.
- **Sonra `format_version`'ı `"3.0"` yap, `.bin` koyma:** kod
  **`missing_project_binary`** olmalı.
- **Sonra boş olmayan ama çöp bir `.bin` koy:** kod
  **`invalid_project_binary`** olmalı (magic `RTP3`–`RTP8` tutmuyor).
- **`has_geometry`'yi `false` yap:** kod **`unsupported_project_layout`**.
- **Bozuksa ne demek:** herhangi biri `ready=true` dönerse tam olarak
  düzeltmeye çalıştığımız arıza duruyordur: preflight, commit'i sahneyi
  silerek başarısız olacak bir projeye "hazır" diyor.
- **İş bitince probe paketini sil** — registry'de kalırsa Template Hub'da
  görünür.

### 6. Lisans kaydı (kod değil, doğrulaması hızlı)

- `RayTrophiStudio/assets/THIRD_PARTY_ASSETS.md` okunabilir ve doğru mu.
- 11 vegetation descriptor'ı damgalandı; Asset Browser'ı aç, bir ağacın
  bilgisinde `license` alanı **"SALE FORBIDDEN"** ibaresini gösteriyor mu.
- **Bozuksa ne demek:** `AssetRegistry` damgayı geri yazarken eziyorsa
  ([AssetRegistry.cpp:866](../RayTrophiStudio/source/src/Scene/AssetRegistry.cpp#L866))
  kayıt kalıcı değildir ve her tarama lisansı siler.
- ★ Damgalanamayan 3 model var (`Ox_eye_daisy`, `Quercus_Rubra`,
  `Red Currant`) — `.asset.json`'ları yok, sessizce `unknown` kalıyorlar.
  Asset Browser'dan descriptor ürettirip
  `python scripts/stamp_vegetation_licenses.py x64/Release/assets --write`
  ikinci kez koşturulabilir.

---

## Bu partide DEĞİŞMEYENLER

- Sahne içeriği, template geometrisi, UI state uygulaması — hiçbiri.
- Project template **hâlâ açılmıyor** (2B.2b-ii ertelendi). Bu parti onu
  açmayı değil, **reddin dürüst olmasını** sağlıyor.
- Vulkan/OptiX/render yolları — hiç dokunulmadı.
