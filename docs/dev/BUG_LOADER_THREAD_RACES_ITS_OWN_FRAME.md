# Yükleyici ipliği, kendisini başlatan KAREYLE yarışıyor

> **Durum:** ARŞİV — kök neden bulundu, düzeltme **kullanıcı tarafından CANLI
> DOĞRULANDI** (2026-09-09): raporlanan senaryo artık çökmüyor.
>
> ★ Kapanan şey **erişim ihlali**. Bu notun ürettiği iki iş devam ediyor ve
> ikisi de başka yerde yaşıyor: (a) aynı pencerenin device-lost hatasını da
> açıklayıp açıklamadığı — o sınama
> [BUG_VIEWPORT_DEVICE_LOST_ON_PROJECT_OPEN.md](BUG_VIEWPORT_DEVICE_LOST_ON_PROJECT_OPEN.md)
> içinde ve kalkanın kapatılmasını gerektiriyor, (b) panelin kare başına
> üçgen taraması (aşağıda "Bakılmamış kalan"). **Çökmenin gitmesi bunların
> hiçbirini ölçmedi.**

## Belirti (kullanıcı raporu)

RayFusion modunda, **yoğun foliage test sahnesi açıkken**, transmission verisi
olan **başka bir proje** açılınca erişim ihlali. Yalnızca bu iki proje ve
**yalnızca bu sırada**.

```
RayTrophiStudio.exe!DrawVolumePerformancePanel(UIContext & ctx) Satır 193
RayTrophiStudio.exe!SceneUI::drawRenderInspectorContent(UIContext & ctx) Satır 1699
RayTrophiStudio.exe!SceneUI::drawRenderSettingsPanel(UIContext & ctx, float) Satır 2847
RayTrophiStudio.exe!SceneUI::drawPanels(UIContext &) Satır 4210
RayTrophiStudio.exe!SceneUI::draw(UIContext & ctx) Satır 3833
RayTrophiStudio.exe!SDL_main(int argc, char * * argv) Satır 4080
```

193. satır:

```cpp
if (tris) for (const auto& tri : *tris) if (tri) {
```

## Kök neden

**Yükleyici ipliği, onu doğuran karenin ortasında başlar; karenin geri kalanı
sökülmekte olan sahneyi okumaya devam eder.**

Zincir:

1. `SceneUI::draw` en başta `drawMainMenuBar(ctx)` çağırır (`scene_ui.cpp:3823`).
2. `drawMainMenuBar`, **Template Hub'ı da çizer** (`scene_ui_menu.hpp:1304`) —
   File menüsünün "Open Project" maddesiyle birlikte.
3. Hub'daki bir "recent project" tıklaması `performOpenProject`'e gider
   (`scene_ui.cpp:8629`); o da `std::thread loader_thread(...)` başlatır,
   `detach()` eder ve **derhal geri döner**.
4. Kare devam eder: `drawPanels` → `drawRenderSettingsPanel` →
   `drawRenderInspectorContent` → `DrawVolumePerformancePanel`.
5. Panel `InstanceManager::getGroups()`'u ve **her scatter kaynağının üçgen
   vektörünü** gezer.
6. Aynı anda yükleyici ipliği `openProject` → `newProject`
   (`ProjectManager.cpp:1091`) → `InstanceManager::clearAll()` → `groups.clear()`
   çalıştırır. Bu, her `InstanceGroup`'u ve içindeki her `ScatterSource`'un
   `triangles` vektörünü **yok eder**.
7. Panelin elindeki `tris` işaretçisi ve yineleyicileri serbest bırakılmış
   belleği gösterir → **erişim ihlali**.

### ★★★★ Main.cpp'deki kapı BİR TUR GEÇ

`Main.cpp` içinde "EARLY SCENE LOADING GUARD" var (~3674) ve doğru şeyi yapıyor:
`ui.scene_loading` doğruyken yalnızca yükleme modalini çizip `continue` ediyor,
`ui.draw`'a hiç varmıyor. Ama bu kapı **döngünün başında** duruyor — yükleyici
ise `ui.draw`'un **içinde** doğuyor. Yani:

| kapı | ne zaman bakar | yükleyici ne zaman doğar |
|---|---|---|
| Main.cpp:3674 | tur N'in **başı** | tur N'in **ortası** (`ui.draw` içinde) |

Arada kalan yarım tur korumasızdır: `drawPanels`, overlay'ler, `processAnimations`,
`handleSceneInteraction`, **ve render/present bloğunun tamamı**.

### Neden yalnızca O İKİ proje, o sırada

Bu bir yarış; sıraya bağlılığı projelerin *içeriğinden* değil, iki pencerenin
çakışmasından gelir:

- **Önceki sahne ağır foliage olmalı** ki panelin taraması yeterince UZUN sürsün.
  Panel her karede her kaynağın her üçgeni için `nodeName + "#" + materialID`
  şeklinde bir `std::string` kurup `unordered_set`'e atıyor — foliage kütüphanesi
  kaynakları yüz binlerce facade üçgeni taşır.
- **Yeni proje ayrıştırması yeterince KISA olmalı** ki `clearAll()` panel hâlâ
  gezerken düşsün (`clearAll`, JSON parse + `.bin` slurp'ünden sonra gelir).

Hafif bir sahneden açarsan panel döngüsü mikrosaniyelerde biter ve yarışı
kaçırırsın; bu yüzden "sadece bu iki proje" gibi görünüyor.

## ★★★ Muhtemelen AÇIK device-lost hatasının da aynı kökü

[BUG_VIEWPORT_DEVICE_LOST_ON_PROJECT_OPEN.md](BUG_VIEWPORT_DEVICE_LOST_ON_PROJECT_OPEN.md)
"Hâlâ bakılmamış yüzey" listesinde şu madde **yoktu**: yüklemeyi *başlatan*
karenin kendisi. O notta 4. sırada elenen şey "yükleme sırasında kare basılıyor
(progress callback)" idi — ve doğru elendi: callback kare basmıyor. Ama
**yüklemenin başladığı kare** render bloğuna girip raster iş kaydediyor ve
present ediyor, tam da yükleyici ipliği `newProject` → `resetForProjectReload`
ile viewport kaynaklarını yıkarken.

Bu, o notun A/B tablosundaki tek kalan değişkenle birebir örtüşür:

> *"Senaryo 1 ile 2 arasındaki fark yüklenen şey değil, önceden yüklenmiş olan
> şey"* — çünkü ağır bir sahnede o son kare uzun sürer (ölçülen 26–467 ms) ve
> sökmeyle çakışma penceresi o kadar açık kalır.

Ve kullanıcının kendi ifadesi zaten bunu söylüyordu:

> *"bir işlem bitmeden belleği geçersizleştiriliyor gibi"*

**Bu bir iddia değil, sınanabilir bir tahmindir — ve HENÜZ SINANMADI.**
Sınama: `enterSolidViewportForSceneLoad` kalkanı KAPATILIP senaryo 2
tekrarlanır; bu düzeltmeyle "device lost while **submitting**" satırı artık
basılmamalıdır, çünkü o kare artık submit edilmiyor. Tarif device-lost notunda
yaşıyor (bu dosya kapandı, o dosya açık).

★ Erişim ihlalinin düzelmiş olması bu tahmini **doğrulamaz**: iki arıza aynı
pencereden geçiyor olabilir de olmayabilir de, ve çökmenin gitmesi yalnızca
CPU tarafındaki okuyucuyu ölçtü.

## Düzeltme

Üç kapı, dıştan içe:

| # | yer | ne yapar |
|---|---|---|
| 1 | `scene_ui.cpp`, `drawMainMenuBar`'dan hemen sonra | Yükleme başladıysa `SceneUI::draw`'un geri kalanını atlar. Menü çizimi öncesinde açık ImGui Begin/Push YOK, yani düz `return` güvenli. |
| 2 | `Main.cpp`, `ui.draw`'dan hemen sonra | Aynı bayrağa bakar; `ImGui::EndFrame()` + `continue` ile render/submit/present bloğunu atlar. `Render()` yerine `EndFrame()` çağrılır ki sonraki turdaki `NewFrame` dengeli olsun. |
| 3 | `scene_ui_volume_performance.cpp`, foliage döngüsü | Savunma amaçlı: yükleme aktifse döngüyü hiç kurmaz. Okuyan taraf olduğu için buradaki çökme suçu hep başka yerde arattırır. |

★ 1. ve 2. kapı `scene_loading` (SceneUI üyesi) **ve** `g_scene_loading_in_progress`
(global) — ikisine birden bakar. Import Model yolu
(`scene_ui_menu.hpp:619`) yalnızca birincisini kaldırıyor, `performOpenProject`
ikisini birden.

### Yan düzeltme: panel flat SoA kaynaklara KÖRDÜ

Aynı döngüde ikinci bir arıza vardı ve bu deponun en pahalı sınıfından:
**panel yalan söylüyordu.** Kayıt sayısı yalnızca `source.triangles` (facade)
üzerinden hesaplanıyordu. Oysa `InstanceManager::deserializeFast` bir proje
dosyasından yüklenen kaynağı `flat_meshes` ile doldurur ve `triangles`'ı **boş
bırakır** — yani dosyadan açılmış her sahnede foliage VRAM tahmini olduğundan
az çıkıyordu, hatasız ve ipuçsuz. Artık `flat_meshes` boş değilse kayıt sayısı
mesh sayısıdır (çok materyalli import'ta her materyal kendi `TriangleMesh`'i
olur ve hepsi aynı `nodeName`'i paylaşır — facade tarafında sayılan
`nodeName#materialID` ile aynı büyüklük).

## Bakılmamış kalan (bilerek)

- **Panelin kare başına maliyeti.** Facade kaynaklarda döngü hâlâ üçgen başına
  bir `std::string` kuruyor, her karede. Bir önbellek eklenmedi: yanlış bir
  geçersizleştirme anahtarı bu paneli tekrar yalancı yapardı, ve bu panel
  "Performance" bölümü açıkken çalışıyor — yani ölçüm yapılırken. Önce ölçülsün.
- **Import Model yolu `g_scene_loading_in_progress`'i kaldırmıyor.** Bu
  düzeltme ona bağlı değil (1. ve 2. kapı `scene_loading`'i de görüyor), ama
  bayrağın diğer tüketicileri (OptiX param güncellemesi, `Renderer.cpp:8633`,
  `drainMainThreadQueue`) import sırasında korumasız kalıyor.
