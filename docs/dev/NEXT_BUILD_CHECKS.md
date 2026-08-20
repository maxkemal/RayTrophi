# Sıradaki build — sıralı kontrol listesi

> **Durum:** AKTİF — 2026-08-20 dördüncü partisi (sim node §B: World kapsamı).
> Her partide üzerine yazılır.
>
> ★★ **Bu parti DERLENMEDİ** — derleme kullanıcıda (CLAUDE.md kural 2). Aşağıki
> §0 bu partinin ilk derlemede sırayla kontrol edilmesi gereken maddeleri
> listeliyor. §A önceki partiden devralınan açık arızalar, §B bir sonraki
> oturumun devraldığı iş. Kapanmış işi açık göstermek bu depoda iki kez
> pahalıya patladı, o yüzden burada yalnızca **gerçekten açık/kontrol
> edilmemiş** olanlar var.

## §0 — BU PARTİ: ilk derlemede sırayla kontrol et

Sıra: bağımsız ve hızlı görülen önce, diğerlerinin sonucunu maskeleyen sonra.

### 0.1 Derleme (bağımsız, en hızlı)

`RtApi.h/.cpp`, `RtIpc.cpp`, `RtPython.cpp`, `RtIpcSecurity.cpp` (dokunulmadı,
sadece namespace tekrar kullanıldı), `SimulationNodes.h/.cpp`,
`RtApiSimNodes.cpp`, `scene_ui_simnodes.cpp` değişti — yeni `.cpp` dosyası
**yok**, `.vcxproj`'a dokunmaya gerek olmamalı. **Ne görmen gerek:** temiz
derleme. **Bozuksa ne demek:** muhtemelen `SimCommand::Scope`'a `World`
eklenmesi bir `switch` ifadesini eksik bıraktı (derleyici uyarır) veya
`WorldThermalNode`/`DomainParamNodeBase` dynamic_cast zincirinde bir imza
uyuşmazlığı var.

### 0.2 Descriptor/audit betikleri (bağımsız, hızlı, KOŞULDU ve GEÇTİ)

```powershell
python scripts\gen_ipc_descriptors.py --check
python scripts\audit_ipc_capabilities.py
```

★ Bu partide zaten koşuldu ve geçti (315 metot, %100 belgeli, `world.`
namespace'i mirror'da zaten vardı). Yeniden koşmak yalnızca derlemeden SONRA
kaynağın değişmediğini doğrular.

### 0.3 IPC yolundan `rt_test_sim_graph.py` (uygulama açık olmalı)

```powershell
.\scripts\ipc\Start-RayTrophi.ps1
Import-Module .\scripts\ipc\RtIpc.psm1 -Force
Invoke-RtIpc script.run_file @{ path = 'scripts/test/rt_test_sim_graph.py' }
type x64\Release\scripts\test\_sim_graph_result.txt
```

**Ne görmen gerek:** `RESULT: ALL PASSED`, özellikle yeni "World scope: World
ThermalNode overrides the ambient thermal state" bölümü — ★★★ diferansiyel
test satırı (`graph-applied ambient_kelvin matches the direct API reading
exactly`) bu partinin asıl iddiasını doğruluyor: graph yolu, doğrudan
`rt.world.set_thermal` yolunun ürettiği SAYININ AYNISINI üretiyor mu.
**Bozuksa ne demek:** `simGraphApply`'nin `is_world` dalı ya hiç
tetiklenmiyor (scope string'i "world" olarak gelmiyor — `simGraphEvaluate`'in
switch'ini kontrol et) ya da `readWorldThermalParameter`/
`writeWorldThermalParameter` anahtar eşlemesi `WorldThermalNode::fields`
listesiyle uyuşmuyor.

### 0.4 ★ En sinsi olası başarısızlık: sessizce makul görünen sonuç

`world.get_thermal`/`set_thermal` doğrudan çağrıldığında (node'suz) doğru
çalışabilir ama `sim.world_thermal` node'u üzerinden hiçbir şey YAZMAZ ve
`sim_graph.apply` yine de `applied: 0, failed: [], refused: []` ile "başarılı"
görünebilir — çünkü hiçbir alan tiklenmemişse bu **doğru** davranış (opt-in),
ama field hiç tiklenemiyor da olabilir (UI'da `WorldThermalNode` dynamic_cast
dalı tetiklenmiyorsa sağ panel boş kalır ve kimse "node bir işe yaramıyor"
dışında bir belirti görmez — bu depoda bu TAM OLARAK daha önce olmuş arıza).
**Kontrol:** panelde World scope'a geç, World Thermal node'u ekle, sağ panelde
tik kutuları GÖRÜNMELİ (bkz. `scene_ui_simnodes.cpp`'deki yeni
`WorldThermalNode` dalı).

## Bu partide kapanan (kayıt — aksiyon yok)

| iş | doğrulama |
|---|---|
| Çözücü analitik doğrulaması | 6 vaka + 1b, **ALL PASSED** — bkz. [PHYSICS_VALIDATION.md](PHYSICS_VALIDATION.md) |
| `scene.get_world_transform` (+ `simulated` bayrağı) | serbest düşüş 4.84370 m okunuyor |
| `fluid.seed` sıfır örtüşme reddi + türetilmiş varsayılan bölge | IPC 3 ve 4 yeşil, panel "Seed Fluid Now" kullanıcı tarafından test edildi |
| Türkçe locale ondalık virgül (ret mesajı) | `region (-0.400 50.000 -0.400)` — nokta |
| **Sim kontrol sözleşmesi** (`physics.step` playhead + `sim.control_state`) | IPC 1 / 1b yeşil; **normal Play panelden doğrulandı** |
| IPC test kanalı | `rt_ipc.py` + süit koşuyor — bkz. [IPC_TEST_CHANNEL.md](IPC_TEST_CHANNEL.md) |
| Sıralama alanları ajana ulaşıyor | `physics.step` → `verify_with: [sim.control_state]`, `next: [scene.get_world_transform]` |
| Yetki aynası | `sim.control_state` = `Read`, audit yeşil, 313 metot %100 belgeli |
| **Sim node §B step 4: World kapsamı** | `WorldThermalState` script yüzeyine kavuştu (`world.get_thermal`/`set_thermal`) + `sim.world_thermal` node; descriptor/audit betikleri koşuldu ve GEÇTİ; **derlenmedi, IPC test yolu koşulmadı** — bkz. §0 |

---

## §A — AÇIK ARIZALAR (ölçülü, kök bulunmadı)

Sıra: bağımsız ve hızlı görülen önce.

### A1. ★★★★ İkinci domain birincinin parçacıklarını SİLİYOR

```powershell
python scripts\test\rt_test_physics_ipc.py     # 5. vaka
```

**Ölçüldü:** `fluid.get` 22932 → **0**, `list_domains` da 22932 → 0. İki okuyucu
hemfikir olduğu için bu **okuyucu arızası değil**, gerçek kayıp.

★★★ Yalnızca **IPC yolunda** görünüyor; script içinden aynı dizi 6760 → 6760
koruyor. Yani kare döngüsünün yeni domain'i görünce yaptığı bir şey.

**Neden önce bu:** bu motorun etrafında kurulduğu her coupling senaryosu
("Fuel yanar, Smoke'u besler") aynı anda iki domain istiyor.

### A2. ★★ Timeline ile sürülen düşüş DURUYOR

```
frame  6 -> 49.678     frame 24 -> 49.436
frame 12 -> 49.436     frame 48 -> 49.436
```

Bir koşuda frame 24'te `y=50.0, simulated=False` döndü — yani sadece durmuyor,
**arada rest'e sıfırlanıyor**. `physics.step` yolundan bağımsız; timeline'ın
kendi yakalama/cache yolunda (`syncRigidToFrame` → `advanceRigidTimelineToFrame`,
UI tick başına `kMaxStepsPerTick = 8` ile sınırlı).

★ Yalnızca son kareyi okuyan bir çağıran **havada durmuş** bir gövde görür —
makul görünen, tamamen yanlış bir gözlem.

### A3. ★★★ Silinmiş adı tekrar kullanmak HAYALET nesne üretiyor

`scene.delete` yalnızca **pending-delete** işaretliyor; fiziksel kaldırmayı kare
döngüsü yapıyor, ve script ana thread'i tuttuğu için o döngü hiç dönmüyor. Aynı
adı hemen geri eklemek bir cesetle çakışıyor: `add_primitive` başarı dönüyor,
sonraki `set_transform` **"object not found"** diyor.

Ayrıntı: [BUG_DELETED_NAME_REUSE_GHOST.md](BUG_DELETED_NAME_REUSE_GHOST.md).

★★ Her iki test rig'i de koşu-benzersiz ad kullanarak bunu **dolaşıyor,
çözmüyor** — yeşil bir koşu ad tekrarı hakkında hiçbir şey söylemiyor.

### A4. ★★★ `viewport.render_frames` kareyi YAYIMLAMIYOR

Ölçüldü ve belgelendi, düzeltilmedi. Ajan tarafında "render ettim ama
göremiyorum" olarak görünür.

### A5. Ergonomi: script sürerken araya girmek pratikte zor

Adımlar 1/240 s'lik paketler hâlinde çok hızlı akıyor. Kontrol scrub anında
**derhal** kullanıcıya geçiyor (IPC 1b bunu ölçüyor), ama kullanıcının o
pencereyi bulması ayrı bir iş. Arıza değil, eksik.

---

## §B — SIRADAKİ İŞ: sim node

Fizik referansı **artık hazır**, ve sim node'un bitti tanımı buna dayanıyor:

1. [SIMULATION_NODE_CONCEPTUAL_MODEL.md](SIMULATION_NODE_CONCEPTUAL_MODEL.md)
   okunmadan başlanmaz — *graph BEYAN EDİLMİŞ NİYETTİR*.
2. Fizik vakalarını **node graph'ı olarak yeniden ifade et**.
3. Diferansiyel test: *"graph, doğrudan yolun ürettiği sayının aynısını
   üretiyor mu?"* — ★★ enstrüman özneyle aynı arıza kipini paylaşmamalı, o
   yüzden referans dosyası node'dan GEÇMEZ.
4. ✅ **World kapsamı — BU PARTİDE YAPILDI, DERLENMEDİ.** Bkz.
   [SIMULATION_NODE_OBJECT_MODEL.md](SIMULATION_NODE_OBJECT_MODEL.md) §8 adım
   4 ve §0 yukarıda. `rt.world.get_thermal`/`set_thermal` + `sim.world_thermal`
   node, diferansiyel testiyle birlikte.
5. **Sıradaki: Object kapsamı göçü** — MSF node'ları (`sim.substance`,
   `sim.pyrolysis`, `sim.phase_change`, `sim.surface_inspect`) zaten object
   kapsamında yaşıyor (bkz. §8 step 1-3, `simulation_object_graphs`); asıl kalan
   iş global/eski bir graph'tan göç edecek bir şey kalmadıysa bu adımı
   **ölçmeden** ✅ işaretleme — önce `object_sim_graphs` dışında hâlâ MSF
   yazan bir yol olup olmadığını doğrula.
6. **Ölçüm ilişkileri** — domain ile kesişen collider/force raporu, `rt.attr.*`.

★★★ **Node testini IPC kanalından koştur.** Script içinden koşarsan aynı
körlüğü miras alır: kare döngüsünün yaptığı hiçbir şeyi göremez.
`rt_test_sim_graph.py` `script.run_file` üzerinden zaten IPC kanalından koşuyor
(bkz. §0.3) — yeni testler eklerken bu dosyaya ekle, ayrı bir script açma.

---

## Nasıl koşulur

```powershell
# 1) uygulama açık, pencere görünür olmalı
.\scripts\ipc\Start-RayTrophi.ps1

# 2) çekirdek mantık (hızlı, döngüye kör)
Import-Module .\scripts\ipc\RtIpc.psm1 -Force
Invoke-RtIpc script.run_file @{ path = 'scripts/test/rt_test_physics_validation.py' }
type x64\Release\scripts\test\_physics_validation_result.txt

# 3) uygulamayı gerçekten sürer (döngü arızalarını görür)
python scripts\test\rt_test_physics_ipc.py

# 4) descriptor denetimleri
python scripts\audit_ipc_capabilities.py
python scripts\verify_descriptor_claims.py --live
```

★★★ **IPC süitinin 0. vakası kare döngüsünün döndüğünü ÖLÇER** ve dönmüyorsa
koşuyu durdurur. Bu doğru davranıştır: dönmeyen bir döngüde o dosya sessizce
in-process rig'in yavaş bir kopyasına dönüşür ve yeşili göründüğünün **tam
tersini** anlatır.

★ Test scriptleri **iki yere** kopyalanır (`scripts/` + `x64/Release/scripts/`)
— ama `rt_test_physics_ipc.py` ve `rt_ipc.py` **kopyalanmaz**: onlar
uygulamanın yüklediği scriptler değil, uygulamayı dışarıdan süren istemciler.
