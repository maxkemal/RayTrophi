# UV authoring: mevcut durum, SculptForge incelemesi ve uygulama planı

Tarih: 2026-11-03. Durum: kaynak incelemesi ve önerilen tasarım; uygulama tamamlanmadı.
İlk kaynak dilimi artık [UV_EDITOR_FIRST_SLICE.md](UV_EDITOR_FIRST_SLICE.md) içinde
izleniyor. Aşağıdaki mevcut-durum envanteri uygulama öncesi audit snapshot'ıdır;
tam roadmap hâlâ tamamlanmadı, ilk dilimin build/canlı kabulü bekleniyor.
Build çalıştırılmadı, uygulama açılmadı. Mevcut çalışma ağacındaki değişiklikler korunmuştur.
Buradaki yeni modül ve API isimleri öneridir; mevcut özellik olarak okunmamalıdır.

## 1. Sonuç

RayTrophi'deki mevcut UV workflow, materyal örnekleme ayarları ve sınırlı mesh UV
dönüşümleridir. Tam bir UV authoring editörü değildir. xatlas için CMake ve VS2022
entegrasyon kabuğu vardır; varsayılan `external/xatlas` dizini bu checkout'ta yoktur.
`source` altında xatlas çağrısı, unwrap servisi veya seam tabanlı UV sistemi bulunamadı.
Harici bir build değişkeniyle başka konumdan sağlanan xatlas veya kullanıcının mevcut
çalışan binary'si bu statik incelemenin kapsamı dışındadır.

Öneri: MIT xatlas tabanlı otomatik unwrap ve mevcut UV'leri yeniden paketleme;
RayTrophi'ye ait flat UV topoloji/işlem çekirdeği; 3D viewport ile eşleşen ayrı alt
UV editörü. Manuel seam'lere bağlı conformal unwrap ikinci algoritmik aşamadır.
Yalnız küçük materyal önizlemesini büyütmek bu ihtiyacı çözmez.

## 2. Yerel kaynak kanıtları ve eksikler

Yollar `RayTrophiStudio/` köküne göredir.

| Kaynak | Mevcut durum / eksik |
|---|---|
| `CMakeLists.txt:17` ve `RayTrophiStudio.vcxproj:155` | xatlas kaynak/header varlığına bağlı tanım, include ve derleme girdileri var. Çağıran uygulama kodu yok. Backend kullanılabilirliği API/UI'da açıkça raporlanmalı. |
| `source/src/UI/scene_ui_materials.cpp:941` | Önizleme en fazla 300 üçgenlik örnek. Topoloji, tüm adalar, overlap veya seçim için otorite olamaz. |
| Aynı dosya, `collectFlatUvMeshes`, `flatUvVisit` | Flat attribute düzenleme mevcut; hedef hâlâ nesne adı + materyal. Çok materyalli sibling mesh'ler ayrı ele alınıyor. Yeni servis doğrudan sahnedeki flat mesh kimliklerini çözmeli. |
| Aynı dosya, `flatUvBounds` | Sınır okuması mutable ziyaretçiyi kullanıyor; seçili ek seti `uv` kanalına da kopyalıyor. Salt okunur sorgunun yazma yan etkisi ayrılmalı. |
| Aynı dosya, `Normalize UVs` | U ve V ayrı ölçekleniyor; en-boy oranı ve texel density değişebilir. Uniform fit ve explicit non-uniform normalize ayrı işlemler olmalı. |
| Aynı dosya, `applyUvEditToSelectedSet` | Flip/normalize gibi işlemler UI lambda'larında. Bu yolda history kaydı görünmüyor; ortak transaction/undo servisine taşınmalı. |
| Aynı dosya, `refreshGeometryAfterUvChange` | UV değişiminden sonra backend geometri yeniden kuruluyor. Sürüklemenin her adımında tam rebuild yerine preview/commit ve uygun attribute invalidation gerekli. |
| `source/src/UI/scene_ui_modifiers.cpp:299` | View projection, base_mesh_cache/Triangle facade yolu ve UVProjectionCommand kullanıyor. Flat-only yeni çekirdeğe taşınmalı. |
| `source/src/Scene/Triangle.cpp:183–299` | Flat UV'ler vertex attribute'u `uv`, `uv1`...; applyUVSet seçilen seti `uv` içine kopyalıyor. Set 0 ve aktif render verisi aynı yerde; özgün set 0'ı koruma sözleşmesi gerekli. |
| `source/include/DNA/GeometryDetail.h` | Flat indeksler ve vertex attribute'ları var. Seam/pin/UV-corner kimliği için açık kalıcı veri sözleşmesi gerekli. |
| `source/src/Api/RtApi.cpp:799,856`, `RtPython.cpp:575` | `rt.mesh.uvs/set_uvs` mevcut; vertex buffer okuma/yazma, tam UV authoring API'si değil. Setter sayım/render kilidi kontrol ediyor; finite kontrolü ve UV transaction görünmüyor. |
| `source/src/MeshEdit/FlatMeshGeometryCommand.cpp` | Snapshot/undo için örnek var; apply ilk aynı adlı TriangleMesh'te dönüyor. Çok materyalli nesne için doğrudan yeniden kullanılması yeterli değil. |

Mevcut authoring eksikleri: seam işaretleme/temizleme, UV köşesi/kenarı/yüzü/adası
seçimi, pin, cut/stitch, unwrap/pack ayrımı, ada transformu, density, stretch/overlap
analizi, checker ve 2D–3D seçim eşleşmesi. Scriptte ham UV buffer desteği vardır;
eşdeğer seam/unwrap/pack IPC authoring yüzeyi incelenen kaynaklarda bulunmadı.

## 3. SculptForge: kod, yöntem ve lisans ayrımı

İncelenen `main` hareketli bir referanstır; uygulama öncesi exact commit sabitlenmeli.
Dosya başlıkları ve lisans bildirimleri esas alınmıştır; README'nin özellik listesi
yer yer kaynak kodunun gerisindedir. Burada performans iddiaları benchmark yapılmış
RayTrophi sonuçları olarak kullanılmaz.

| Sistem | Lisans / kaynak | RayTrophi için değer |
|---|---|---|
| xatlas / thekla_atlas bildirimleri | MIT; [upstream header](https://github.com/jpcy/xatlas/blob/master/source/xatlas/xatlas.h), [SculptForge notices](https://github.com/islandmed/SculptForge/blob/main/THIRD_PARTY_NOTICES.md) | Otomatik chart/unwrap, mevcut UV mesh paketleme, margin, rotation, progress/cancel. Bağımlılığı upstream'ten sabit SHA ile almak uygun. |
| Vendored xatlas düzeltmeleri | [SOURCE.txt](https://github.com/islandmed/SculptForge/blob/main/third_party/xatlas/SOURCE.txt) MIT upstream `f700c7790aaa030e794b52ba7791a05c085faf0c` ve iki yerel düzeltmeyi belgeliyor; dosya MIT başlığı taşıyor | İlk yüz normaliyle de planar bölge kontrolü; sınır vertices'i olmayan chart için bounding-box güvenliği. Yeniden üretilebilir regression vakaları hazırlanmalı; patch provenance kaydı korunmalı. |
| `AutoUv.cpp` | GPL-3.0-or-later | xatlas wrapper, aşamalı progress/cancel, büyük chart alanını sınırlama, köşe başına çıktı. xatlas MIT diye bu wrapper MIT olmaz. |
| [Uv.h / Uv.cpp](https://github.com/islandmed/SculptForge/blob/main/src/core/Uv.h) | GPL-3.0-or-later | Corner-twin, UV vertex birleşimi, island CSR, border, seam/pin; iyi bir veri sorumluluğu referansı. Kopyalanacak MIT modül değil. |
| [UvUnwrap.cpp](https://github.com/islandmed/SculptForge/blob/main/src/core/UvUnwrap.cpp) | GPL-3.0-or-later | Seam boyunca parçalama; kapalı/handle yüzeyde ek kesikler; LSCM, Tutte başlangıcı, ARAP refinement, pinned solve. Yayınlanmış yöntemler bağımsız uygulanabilir. |
| [UvPack.cpp](https://github.com/islandmed/SculptForge/blob/main/src/core/UvPack.cpp) | GPL-3.0-or-later | Density eşitleme, minimum bounding box, skyline ve xatlas shape packing, UDIM. Wrapper/skyline kodu GPL; xatlas AddUvMesh/PackCharts API'si MIT. |
| [PanelUvEditor.cpp](https://github.com/islandmed/SculptForge/blob/main/src/app/PanelUvEditor.cpp) | GPL-3.0-or-later | GPU UV görünümü, 3D seam aracı, box selection, G/R/S, checker/stretch/density, background unwrap. İş akışı referansı; panel kodu taşınmaz. |
| [UvCommands.cpp](https://github.com/islandmed/SculptForge/blob/main/src/core/UvCommands.cpp) | GPL-3.0-or-later | uv_unwrap/pack/seam/select/transform/edit/density/trim/info ve tek undo adımı; UI ve otomasyon için ortak komut fikri. |
| [TextureTransfer.h](https://github.com/islandmed/SculptForge/blob/main/src/core/TextureTransfer.h) | GPL-3.0-or-later | Yeni UV texel'inden aynı üçgenin barycentric eski UV'sine geçiş; katman transferi ve gutter dilation. Boyayı koruma için bağımsız geliştirme referansı. |

Lisans kararı: RayTrophi kök LICENSE.txt MIT bildiriyor. MIT dağıtım yönünü koruyan
planda SculptForge'un GPL UV editörü/solver/wrapper kodu içeri alınmaz. GPL kodunu
birleşik ürüne katmak lisans/dağıtım kararını değiştirir; yalnız attribution eklemek
yeterli kabul edilmemeli. Algoritma fikri ile kaynak kodunun lisansı farklı konulardır.
İncelenen UV sisteminde doğrulanmış esas MIT aday xatlas'tır; tüm SculptForge UV
sistemini MIT diye sınıflandırmak yanlış olur. Yerel MIT LICENSE başlığındaki
`[year] [fullname]` placeholder'ları da ayrıca düzeltilmesi gereken depo bilgisidir.

## 4. Flat veri modeli ve güvenli commit

Önerilen otorite yine TriangleMesh / DNA'dır; ayrı Triangle koleksiyonu kurulmaz.

1. Her hedef: mesh kimliği + geometry revision + UV-set kimliği. Nesne adı tek başına
   çok materyalli sibling veya instance verisinin kimliği olamaz.
2. DNA içinde flat per-corner UV authoring verisi ve UV indeksleme sözleşmesi;
   seam kenar bitleri ve pin köşe bitleri. Corner sayısı `indices.size()`.
   Bu domain, mevcut tüm attribute'ları vertex-sized sayan altyapıya gelişigüzel
   `add_attribute` ile eklenmez; clone/serialize/resize/topology-edit desteği tasarlanır.
3. Geometrik bağlantı ile UV bağlantısı ayrı tutulur. Aynı konumdaki kopuk kabuklar
   otomatik kaynak yapılmaz. İçe aktarımda seam nedeniyle ayrılmış vertex'ler için
   kaynak topoloji eşlemesi veya kontrollü adjacency çözümü gerekir.
4. Render uyumluluğunda corner UV'leri flat vertex kanallarına derleyen remap/split
   katmanı; bütün setler, normal, materialID, skin/custom attribute'ları korunur.
   xatlas `xref` yalnız UV ataması için değil kaynak attribute eşlemesi için de kullanılır.
   Face/corner sırası korunmalı veya açık remap ile selection/paint verisine iletilmeli.
5. Set 0 kalıcı özgün veridir; aktif render UV cache'i veya set binding'i ayrı
   tanımlanır. `uv1` seçimi `uv0`'ı kaybettirmemeli. Materyal başına farklı set
   kullanımı bütün backend'lerde aynı sözleşmeye bağlanır.
6. UV-only değişimler tangent/paint/seam/UV analiz cache'lerini invalid eder;
   topoloji split'i ayrıca BVH/render/edit cache günceller. Normal map görünümü
   tangentin doğru UV setinden üretilmesiyle doğrulanır.
7. İşlem snapshot üzerinde çalışır. Başarıda ana thread'de tek atomik commit ve undo;
   hata/iptalde hiçbir kısmi scene değişikliği yoktur. Job bittiğinde mesh/set/topoloji
   ve UV revision'ı değişmişse eski sonuç uygulanmaz.
8. UV değişimi boyanmış texture'ı etkiler. Koruma modu tüm ilgili kanallar/katmanlar
   için barycentric rebake ve gutter uygular; desteklenmeyen kanal sessizce atılmaz.
   İlk sürümde transfer tamam değilse bu işlem açıkça reddedilir veya kullanıcı
   açıkça görünümün değişmesini seçer. İptal ve undo texture'ı da kapsar.

## 5. Panel ve sahne iş akışı

Mevcut alt editör sistemi içinde `UV Editor` sekmesi; dock/float desteği. Sol bağlam
rayı ve sağ araç dock'u korunur. `TEMPLATE_HUB_UX_ROADMAP.md`, panel standardı ve
layout guardrails geçerlidir. Properties'de UV set/örnekleme dönüşümü ve editörü açan
kısayol kalır. Mesh dönüşümüyle materyal sampling dönüşümü açıkça ayrılır.

- Üst şerit: hedef nesne/mesh, UV set, vertex/edge/face/island modu ve seçim kapsamı.
- Orta: zoom/pan/frame ile tam UV canvas, checker/texture arka planı; 0–1 sınırı,
  seçili adalar, seam/pin, stretch ve overlap katmanları. Büyük mesh GPU çizimi ve
  görünür alan culling kullanır; 300 üçgen örneği seçim kaynağı yapılmaz.
- Sağ araç dock'u: Seam, Unwrap, Pack, Transform, Analyze grupları. Margin px,
  texture resolution, rotation ve texel density kesin birimlerle gösterilir.
- 3D viewport: kenar seçip Mark/Clear Seam, path seçimi, sharp/material sınırı
  türetme, seçili yüz/ada eşleşmesi. Gizli yüz ve x-ray seçim kuralları açık olmalı.
- İşlem: hedefi seç → seam ekle/sil → unwrap preview → atlas pack → ada düzenle
  → kalite incele → commit. Her drag tek undo; Escape başlangıca döndürür.

Seam silmek yalnız işareti kaldırır; mevcut UV koordinatlarını otomatik birleştirmez.
Stitch/weld ayrıca ada hizalama ve UV köşe birleştirme işlemidir. Seam eklemek de
3D geometriyi yırtmaz. “Seamless” görünüm ayrıca gutter, filtreleme, tangent ve
texture sürekliliğine bağlıdır; seam işaretini kaldırmak bunu tek başına sağlamaz.

## 6. Ortak servis, script ve IPC

Önerilen modüller: `UV/UvTopology`, `UvService`, `UvXatlasBackend`, `UvPacking`,
`UvAnalysis`, `UvTransaction`, `UvPersistence`, `UI/UvEditorUI`,
`Api/RtApiUv`, `RtUvBindings`, `RtIpcUv`. Mevcut 2000+ satırlık dosyalara yalnız
include/kayıt/çağrı bağlantısı eklenir. UI'da business logic bulunmaz.

Önerilen script `rt.uv.*`, IPC `uv.*`; ikisi aynı UvService'i çağırır:

| İşlem | Ortak anlam |
|---|---|
| capabilities/info/get_layout | Backend, setler, revision, ada/seam/pin ve kalite; salt okunur |
| set_create/copy/delete/activate | Set yaşam döngüsü, kalıcı kimlik ve aktif render seçimi |
| select, seam_mark/clear, pin/unpin | Açık mesh/corner/edge kimlikleri; UI seçimine örtük bağımlılık yok |
| unwrap | auto / seam-conformal / projection; yöntem desteği açık raporlanır |
| pack | Mevcut adaları yeniden yerleştirir; unwrap yapmaz |
| transform, cut, stitch, density | Seçili UV elemanlarına transaction ile düzenleme |
| analyze, export_layout | Overlap/degenerate/flip/stretch/density; atlas çizimini SVG/PNG dışa aktar |
| job_status/cancel, preview_commit/discard | Uzun iş, iptal, revision doğrulama ve atomik commit |

Hatalar: invalid target/set/index, non-finite, degenerate/unsupported topology,
backend unavailable, render locked, stale revision, cancelled, unsupported texture
preservation. NaN/Inf ve indeksler iş başlamadan kontrol edilir. IPC verisi büyük
mesh'te sayfalanır/özetlenir; named pipe'a sınırsız layout payload basılmaz.
Bu isimler mevcut IPC yöntemleri değildir; uygulamayla birlikte schema/descriptors,
Python binding ve dokümantasyon teslim edilir.

## 7. Uygulama sırası ve kabul

| Aşama | Teslim / kabul kapısı |
|---|---|
| 0 — veri ve lisans | Upstream xatlas SHA/LICENSE/SOURCE kaydı; backend capability; flat UV domain, set-0 koruması, multi-mesh kimliği ve save/load sözleşmesi. |
| 1 — ilk kullanılabilir dilim | Ortak servis + MIT xatlas auto unwrap + mevcut ada pack + transaction/undo + script/IPC + ayrı 2D panel. Tek 0–1 atlas; unsupported durumlar açık hata. |
| 2 — sahnede seam authoring | 3D edge picking, seam mark/clear/path, otomatik sınır seçimi; seam'i gerçekten gözeten unwrap, cut/stitch ve 2D–3D seçim. |
| 3 — hassas düzenleme | Bağımsız LSCM ve pin, gerekirse ARAP; relax/straighten/align, density, kalite katmanları ve texture koruma. |
| 4 — ileri üretim | UDIM, multi-object atlas, trim sheet, locked islands ve gelişmiş GPU picking. Render/export/paint UDIM desteği doğrulanmadan yalnız koordinat üretmek tam UDIM desteği sayılmaz. |

xatlas ChartOptions'ta doğrudan manuel seam-edge listesi veya pin kısıtı yoktur.
Normal seam ağırlığı/UV hint'i bu garantinin yerine geçmez. Seam'e göre ön-parçalama
ve chart üretiminin coincident vertex'leri yeniden bağlamaması doğrulanmalı;
gerekirse parçalar ayrı girdiler olur. Kapalı yüzeyleri disk parçalara ayırma ve
pin'li conformal solve bağımsız çekirdek işidir. Pack-only akışı AddUvMesh / PackCharts
ile, ada biçimini ve seçim dışı adaları koruma testleriyle yapılır.

### Kullanıcının build sonrası çalıştıracağı kabul listesi

1. xatlas mevcut/yok iki yapılandırmada doğru capability ve açıklayıcı hata.
2. Küp, açık plane, kapalı sphere, torus, disconnected ve non-manifold mesh:
   seam ekle/sil → unwrap → pack; finite UV ve geçerli indeksler.
3. Önceden UV'den bölünmüş import, aynı konumda ayrı kabuk, çok materyalli sibling,
   instance ve negatif/non-uniform transform: yanlış kaynak veya eksik hedef olmaması.
4. Set 0/1 değiştir → düzenle → geri dön → save/reload → undo/redo: her set korunmalı.
5. Move/rotate/scale drag tek undo, Escape iptal; job sürerken hedef silme, set
   değiştirme ve yeniden unwrap: stale sonuç scene'e yazılmamalı.
6. UI, script ve IPC aynı işlemde aynı UV/seam/ada sonucu ve hata semantiği üretmeli.
7. Checker, tangent normal ve boyalı çok kanallı materyal: UV değişiminden sonra
   korunmuş appearance; gutter/mipmap sınırları ve undo texture restore kontrolü.
8. Büyük mesh'te topology kurma süresi, drag frame zamanı, pack süresi, peak RAM ve
   history maliyeti ölçülmeli; tüm yüzlere facade allocation olmamalı.
9. Canlı IPC yalnız kullanıcı uygulamayı açtıktan sonra ayrı süreçte
   `scripts/test/rt_ipc.py` ile; embedded script içinden named-pipe probe çalıştırılmaz.

Bu turda yalnız statik kaynak incelemesi yapıldı; yukarıdaki kabul maddeleri
çalıştırılmış testler veya tamamlanmış özellikler değildir.
