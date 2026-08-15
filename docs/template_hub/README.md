# Template Hub Faz 0 Sözleşmesi

Bu klasör, `TEMPLATE_HUB_UX_ROADMAP.md` için uygulanabilir Faz 0 sözleşmelerini taşır. Buradaki JSON örnekleri runtime template paketleri değil; registry ve loader geliştirilirken kullanılacak kanonik sözleşme örnekleridir.

## Dil politikası

- Kalıcı manifest, JSON Schema, guidance ve makine tarafından okunan bütün JSON değerleri İngilizce yazılır.
- JSON alan adları sonradan yerelleştirilmez veya dile göre değiştirilmez.
- Kullanıcıya gösterilen metinler ilk sürümde İngilizce fallback taşıyabilir; ileride çeviri gerekiyorsa manifest formatını değiştirmeyen ayrı localization kaynakları kullanılır.
- Dokümantasyon Türkçe olabilir; runtime JSON sözleşmesi olamaz.

## Kaynak dosya boyutu politikası

- 2000 satırı aşan mevcut kaynak dosyalarına Template Hub özellik mantığı eklenmez.
- Registry, validator, UI-state adapter, loader ve Hub görünümü ayrı, odaklı yeni `.h/.cpp` modüllerinde geliştirilir.
- Büyük mevcut dosyalar yalnız zorunlu minimum include/call/registration bağlantısını alabilir.
- Özellikle `scene_ui.cpp` ve `ProjectManager.cpp` yeni implementasyonun tutulacağı dosyalar değildir.

## Script ve IPC sözleşmesi

- Template Registry ve Loader yalnız UI özelliği olarak tasarlanmaz.
- Kanonik servis; UI, scripting API ve IPC tarafından ortak kullanılır.
- UI, Python binding veya IPC dispatcher içinde template keşfi/yükleme iş mantığı kopyalanmaz.
- Faz 1 registry ile birlikte salt okunur listeleme, metadata sorgulama ve doğrulama için script/IPC veri modeli belirlenir.
- Faz 2 loader ile birlikte kimlikle template açma, yükleme durumu ve hata sonuçları script/IPC'ye eklenir.
- Etkileşimli unsaved-changes penceresi script/IPC içinde gösterilmez. Çağrı açık bir conflict policy taşır veya `unsaved_changes` gibi makine tarafından işlenebilir sonuç döndürür.
- UI, script ve IPC aynı template kimliklerini, enum değerlerini, doğrulama kurallarını ve hata kodlarını kullanır.
- Script ve IPC testleri teslim edilmeden ilgili template fazı tamamlanmış sayılmaz.

## Dosyalar

- `template_manifest.schema.json`: manifest v1 JSON Schema
- `examples/*.json`: ilk altı built-in `Start` template'inin sözleşme örnekleri

## Kimlik ve keşif kuralları

- Template kimliği kalıcı, küçük harfli ve noktayla ad alanına ayrılmıştır: `raytrophi.start.character_paint`.
- Built-in template'ler ileride uygulama asset kökü altındaki tek bir template registry klasöründen keşfedilir.
- Registry klasör sırasına güvenmez; önce `sort_order`, sonra `display_name`, son olarak `id` ile deterministik sıralar.
- Aynı `id` iki kez bulunursa built-in kayıt sessizce ezilmez; registry bunu hata olarak raporlar.
- Manifest veya preview okunamaması sahne yüklemeyi başlatmadan teşhis edilir.
- `schema_version` major uyumsuzluğu template'i devre dışı bırakır; bilinmeyen minor alanlar ileri uyumluluk için görmezden gelinebilir.

## UI state allow-list

Template yalnız aşağıdaki üretim durumlarını önerebilir:

| Alan | Sözleşme |
|---|---|
| `properties_context` | `scene`, `render`, `terrain`, `water`, `volumetric`, `simulation`, `world`, `modeling`, `hair`, `system`, `paint`, `scatter`, `stylize`, `sculpt` |
| `bottom_editor` | `none`, `dope_sheet`, `graph_editor`, `console`, `terrain`, `anim_graph`, `geometry`, `material`, `assets` |
| `contextual_dock` | `none`, `paint`, `hair`, `sculpt`, `terrain` |
| `contextual_dock_width` | 50–400 piksel arası başlangıç önerisi |
| `viewport_shading` | `solid`, `material_preview`, `rendered`, `matcap` |
| `frame_target` | Açılışta çerçevelenecek opsiyonel sahne nesnesi adı |
| `show_timeline` | Timeline görünürlüğü için açık boolean tercih |

Semantik isimler bilinçli olarak kullanılır. Manifestler `active_properties_tab: 8` veya `shading_mode: 2` gibi C++ uygulama ayrıntıları taşımaz. Runtime adapter semantik değeri o sürümün gerçek UI state'ine eşler.

## Template'in değiştiremeyeceği kullanıcı tercihleri

- Tema, font ve genel UI ölçeği
- Erişilebilirlik tercihleri
- Kişisel kısayollar
- Kişisel asset library yolları
- Docking'in global açık/kapalı tercihi
- Telemetri, ağ veya güvenlik ayarları
- Son kullanıcıya ait render/export klasörleri

## Yükleme sırası sözleşmesi

İlerideki loader şu sırayı korur:

1. Manifesti ve uyumluluğu doğrula.
2. Zorunlu assetlerin varlığını doğrula.
3. Mevcut unsaved-changes akışını tamamla.
4. Scene/recipe'yi mevcut Project Manager/serializer yolu üzerinden geçici yükleme bağlamında hazırla.
5. Scene başarıyla hazırlandıktan sonra UI allow-list'ini uygula.
6. Opsiyonel guidance içeriğini göster.
7. Başarısızlıkta yarım scene veya yarım UI state bırakma.

## İlk template içerik sözleşmeleri

### Empty

- Sahne içeriği: boş dünya; otomatik cube/light/camera oluşturulması bu template için kapatılır.
- UI: Scene, alt editör kapalı, sağ dock kapalı, Solid.
- Amaç: deneyimli kullanıcıya gerçekten temiz başlangıç.

### General Scene

- Sahne içeriği: kanonik flat mesh yolunda varsayılan cube, ışık ve kamera.
- UI: Scene, timeline görünür, sağ dock kapalı, Solid.
- Amaç: mevcut Blender-style default scene davranışının template karşılığı.

### Product Lookdev

- Sahne içeriği: nötr stüdyo zemini, ürün hedefi, kamera ve hafif stüdyo ışık rig'i.
- UI: Scene, Material editor açık, Material Preview.
- Amaç: materyal ve ışık çalışmasına temiz başlangıç.

### Portrait & Groom

- Sahne içeriği: doğrulanmış scalp/karakter hedefi, temiz groom başlangıcı, portre kamera/ışık rig'i.
- UI: Hair, Hair dock açık, Rendered veya desteklenmiyorsa güvenli fallback.
- Amaç: kullanıcıyı doğrudan grooming bağlamına yerleştirmek.

### Character Paint

- Sahne içeriği: boyanabilir mesh, doğrulanmış material slot/texture set ve temiz layer başlangıcı.
- UI: Paint, Paint dock açık, Material Preview.
- Amaç: hedef ve kanal kurulumu aramadan boyamaya başlamak.

### Terrain Environment

- Sahne içeriği: güvenli başlangıç çözünürlüğünde terrain, sun/sky ve küçük başlangıç graph'ı.
- UI: Terrain, Terrain editor açık, sağ terrain dock yalnız geçerli bir bağlamsal araç varsa açık.
- Amaç: viewport ile graph arasında hazır procedural çevre başlangıcı.

## Faz 1'e geçiş kapısı

Registry koduna başlanmadan önce:

- Şema ve altı örnek manifest ürün kararı olarak onaylanmış olmalı.
- Runtime template kök klasörü kesinleştirilmeli.
- Scene kaynağı için `.rtp` paket mi yoksa kontrollü recipe mi kullanılacağı template bazında kararlaştırılmalı.
- UI semantik değerlerinin `SceneUI` durumlarına tek bir adapter üzerinden eşleneceği kabul edilmeli.

## Implementation status

Phase 1 and the first built-in runtime package were confirmed on 2026-08-14:

- Canonical service: `Template/TemplateRegistry.h/.cpp`
- Embedded scripting: `rt.templates.refresh/list/get/validate`
- IPC: `templates.refresh/list/get/validate`
- IPC capability: all Phase 1 methods require `Read`
- Offline contract test: `scripts/test_template_manifest_contract.py`
- Embedded smoke test: `scripts/test/rt_test_templates.py`
- Named-pipe coverage: `scripts/ipc_test_client.py`
- Public API notes: `docs/TEMPLATE_REGISTRY_API.md`

No scene/project loading and no Template Hub UI were added in Phase 1.

The first package is `RayTrophiStudio/assets/templates/empty`. It contains an English-only manifest, controlled Empty recipe, lightweight SVG preview, and first-open guidance. The project copies built-in template packages to the application output under `assets/templates` after a user build.

Live validation discovered one valid package and confirmed `templates.get`, `templates.validate`, and `templates.prepare` with a `ready` load plan. The complete named-pipe suite passed with 167 PASS / 0 FAIL.

## Phase 2 split

Static inspection found that the current `ProjectManager::openProject` clears the active scene before geometry, materials and assets have fully loaded. A late failure therefore cannot restore the previous project. Template loading must not wrap that destructive path and claim transactional safety.

- **Phase 2A — confirmed:** headless `TemplateLoader::prepare` load plan; project/recipe JSON preflight; required `.rtp.bin` and optional `.aux.json` checks; explicit `reject|discard` conflict policy; script and IPC parity; no mutation. Embedded script validation passed and the live named-pipe suite completed with 164 PASS / 0 FAIL on 2026-08-14.
- **Phase 2B.1 — confirmed:** `TemplateSession` commits the fully preflighted Empty recipe through a non-fallible new-project boundary. `TemplateUiStateAdapter` applies the manifest's properties context and bottom-editor/timeline intent. Embedded Python and IPC both call `rtapi::openTemplate` and return the same structured result. Reject conflicts preserve the active scene; project-backed and fallible recipe commits remain disabled. Embedded validation passed and the live named-pipe suite completed with 172 PASS / 0 FAIL on 2026-08-14.
- **Deferred backend note:** an initialized Vulkan RT scene with no geometry keeps its sample counter at zero and shows avoidable runtime cost until geometry is added. Empty template behavior remains correct and stable; investigate this as a final-pass Vulkan empty-scene optimization rather than blocking Template Hub work.
- **Phase 2B.2a — implemented, user validation pending:** the General Scene recipe stages its flat `TriangleMesh`, camera, and key light entirely before clearing the active project. Only a ready stage crosses the commit boundary. The legacy facade-based default-scene creator is not used.
- Application startup without a project and File → New Project now route to the built-in General Scene template service. The staged flat mesh receives an immediate CPU BVH rebuild, eager hierarchy cache registration, and manifest frame-target selection.
- **Phase 2B.2b — pending:** staged project state or equivalent rollback-safe commit for project-backed templates and remaining production recipes.
