# Sıradaki build — sıralı kontrol listesi

> **Durum:** AKTİF — 2026-08-18 partisi (kapsamlı sim graph'ları + A cinsi:
> §8 adım 1–2–3).
> Her partide üzerine yazılır.

Sıra ölçütü: bağımsız ve hızlı görülen önce, başkasının sonucunu maskeleyen sonra.

## ✅ DOĞRULANDI — 2. tur (2026-08-18 10:19 binary)

Ölçülen her madde geçti, **hiç FAIL ve hiç NOT VERIFIED yok**:

| Madde | Sonuç |
|---|---|
| 0. `unknown owner is refused` (1. turun bug'ı) | ✅ düzeldi |
| 1 + 1b opt-in + 1c emitter (`sim_graph`) | ✅ ALL PASSED |
| 2 object kapsamı (`substance`) | ✅ ALL PASSED |
| 3 kuplaj raporu (`couplings`) | ✅ ALL PASSED |
| 4 render + cache | ✅ ikisi de ALL PASSED |
| 5 `rt.editor` kapsam seçimi | ✅ ALL PASSED |
| 7 öksüz grafik (`rt_probe_orphan_graph.py`, YENİ) | ✅ ALL PASSED |
| 8 granül regresyonu | ✅ PASS (soft ever=True, stiff ever=False) |

**Gözle kalan tek şey: 6. madde (Nodes paneli).**

★ 9. madde listeden DÜŞTÜ: o arıza 2026-08-16'da çözülmüştü, buraya yanlışlıkla
taşınmıştı. Ayrıntı ve ders aşağıda.

★ 7. madde artık scriptli: `scripts/test/rt_probe_orphan_graph.py`. Domain
silme grafiği düşürüyor; obje silme grafiği bırakıyor ama `owner_missing=true`
diyor — ikisi de ölçüldü.
`audit_ipc_capabilities.py`: **297 metot / 30 namespace**, hepsi sınıflandırılmış
(yeni: `sim_graph.list` Read, `sim_graph.create` / `.delete` SceneWrite,
`editor.set_sim_graph_scope` SceneWrite).

**Yeni dosya yok.** Değişen: `NodeSystem/SimulationNodes.h`, `scene_data.h`,
`scene_ui.h`, `Api/RtApi.h`, `RtApiSimNodes.cpp`, `RtApiUiState.cpp`,
`RtApiFluid.cpp`, `RtIpc.cpp`, `RtPython.cpp`, `RtIpcSecurity.cpp`,
`scene_ui_simnodes.cpp`, `Scene/SimulationNodes.cpp`, ve **sekiz** test
scripti + `rt_testlog.py`.

**Adım 3'te eklenen node'lar:** `sim.solver`, `sim.domain_settings`,
`sim.emitter`. Yeni komut türü: `set_emitter`.

★★★ **Bu partinin önceki durumu commit'te:** `09e025e` — bozulursa dönülecek
bilinen-iyi hâl orası. O commit'in tabanı ölçüldü: 1, 4, 5 YEŞİL.

---

## ÖNCEKİ TABANIN ÖLÇÜLEN HÂLİ (09e025e, bu partiden ÖNCE)

Göçten sonra bir şey düşerse sebebi belirsiz kalmasın diye ölçüldü:

| Kontrol | Sonuç |
|---|---|
| `rt_test_editor_state.py` | **ALL PASSED** (`a refused call changed nothing` dahil) |
| N0–N7 beşlisi (graph / couplings / substance / render / cache) | **hepsi ALL PASSED** |
| `rt_test_granular_soft_stability.py` | **PASS** — soft `ever=True`, stiff `ever=False` |
| Nodes sekmesi (gözle) | ölçülmedi; mekanik yarısı editor_state'in exclusivity turuyla kapsandı |
| Çakışık kutu siyah bandı | **ölçülmedi** — hâlâ AÇIK, bu partiyle ilgisiz |

★ Eski kontrol listesi 4. madde için "below_load ever=True final=False" diyordu;
bu **iki koşunun karışımıydı.** Testin gerçek sözleşmesi: soft `ever=True`,
stiff `ever=False`. Yukarısı düzeltilmiş hâli.

---

## 0. ★★★ İLK TURDA YAKALANAN GERÇEK BUG — düzeltildi, derlenmeyi bekliyor

`rt.sim_graph.nodes(scope, "olmayan_domain")` **başarıyla boş liste döndürüyordu.**

`simGraphNodes` bulunamayan grafikte sessizce `{}` dönüyordu; çağıran taraf
bunu "bu grafikte node yok" diye okuyordu. Yani yanlış yazılmış bir sahip adı
**hatasız** geçiyor, ve kullanıcı boş bir tuvale baktığını sanıyordu.

★★★ Deponun klasik şekli: **varsayılan bir ölçüm değildir.** Boş liste
"grafik bulunamadı" ile "grafikte node yok"u ayırt edilemez yapıyordu — ve
ikisinden biri sessiz bir yazım hatası.

**Düzeltme:** imza `Result simGraphNodes(scope, owner, out_nodes)` oldu; IPC
`__error` döndürüyor, Python `raise` ediyor.

- **Görmen gereken:** `rt_test_sim_graph.py` → `RESULT: ALL PASSED`
  (`unknown owner is refused` artık geçmeli).
- **Bozuksa:** hâlâ FAIL veriyorsa Python binding'i `requireResult` çağırmıyordur.

★ Diğer beş test bu düzeltmeden etkilenmemeli; etkilenirse `nodes()` çağrısı
olan yerler `raise` almaya başlamış demektir — o zaman gerçekten olmayan bir
grafiği sorguluyorlardır ve sebep testte değil kurulumdadır.

## 1. `rt.sim_graph` kapsam sözleşmesi — önce bu, 2 ve 3'ün aleti bu

```powershell
Import-Module .\scripts\ipc\RtIpc.psm1 -Force
Invoke-RtIpc script.run_file @{ path = 'scripts/test/rt_setup_sim_graph_scene.py' }
Invoke-RtIpc script.run_file @{ path = 'scripts/test/rt_test_sim_graph.py' }
Get-Content .\x64\Release\scripts\test\_sim_graph_result.txt
```

- **Görmen gereken:** `RESULT: ALL PASSED`, ve içinde bu dört yeni satır:
  `unknown owner is refused`, `creating a graph for a nonexistent entity is
  refused`, `retargeting the owner node is refused`, `the refused retarget
  changed nothing`.
- **Bozuksa:** "unknown owner is refused" düşerse `findGraph` bulunamayan
  grafiği yaratıyor demektir — yani yanlış yazılmış bir domain adı ikinci, boş
  bir grafik üretir ve **sonraki her düzenlemeyi kabul eder.**

★★★ **En sinsi başarısızlık burada:** `the refused retarget changed nothing`.
Sahip node'unun hedefini değiştirmeyi reddedip **yine de yazmak**, grafiği bir
varlığın adı altında tutup başka bir varlığı sürer hâle getirir — ve o andan
sonra **her okuma yanlış olanla tutarlı çıkar.** Hata satırını görürsün, yan
etkiyi görmezsin.

## 1b. ★★★ OPT-IN sözleşmesi — bu partinin EN PAHALI kontrolü

1. maddedeki `rt_test_sim_graph.py` aynı koşuda bunu da ölçüyor.

- **Görmen gereken:** `an UNTOUCHED field was not written`, `only the touched
  field was written`, `a field switched off writes nothing`, `an integer
  parameter round-trips exactly`.
- **Bozuksa:** `an UNTOUCHED field was not written` düşerse Solver/Domain
  Settings node'ları **her apply'da bütün alanlarını yazıyor** demektir —
  kullanıcının hiç dokunmadığı authored değerler kimsenin seçmediği
  varsayılanlarla eziliyor, ve override katmanı sonra o uydurmaları sadakatle
  "geri yüklüyor".

★★★ **Bu partinin en sinsi başarısızlığı budur** ve hiç kimse bug diye
raporlamaz: sahne makul görünür, sayılar makul görünür, sadece kullanıcının
ayarı sessizce gitmiştir. `in_use=false` ile `value=0` **aynı şey değildir**.

★★ `an integer parameter round-trips exactly` düşerse `writeParameter`'daki
yuvarlama kesmeye dönmüştür: yazılan 8 geri 7 okunur, ve `clear_overrides`
**yanlış authored değeri** geri yükler.

## 1c. Emitter — bağlamayı DEĞİŞTİRMEMELİ

- **Görmen gereken:** `the emitter's other authored fields survived` (domain,
  density, source_mode aynı kalır) ve `authored emitter radius restored exactly`.
- **Bozuksa:** o satır düşerse yazma yolu oku-değiştir-yaz değil, taze kurma
  olmuştur — tek değer için ~25 authored alan sıfırlanır. Belirti: "debiyi
  değiştirdim, emitter yer değiştirdi ve maddesi gitti".
- ✅ **1. turda ÖLÇÜLDÜ ve geçti** (`rt_setup_sim_graph_scene.py` artık bir
  flow source kuruyor). Dördü de OK: komut yayıldı, radius değişti, diğer
  authored alanlar sağ kaldı, geri yükleme birebir.

## 2. Object kapsamı — ikinci kapsam gerçekten ayrı mı

```powershell
Invoke-RtIpc script.run_file @{ path = 'scripts/test/rt_setup_sim_substance_scene.py' }
Invoke-RtIpc script.run_file @{ path = 'scripts/test/rt_test_sim_substance.py' }
Get-Content .\x64\Release\scripts\test\_sim_substance_result.txt
```

- **Görmen gereken:** `RESULT: ALL PASSED` + `the object graph is listed under
  the object scope`.
- **Bozuksa:** o satır düşerse üç depolama tek haritaya çökmüş demektir; aynı
  adı taşıyan bir domain ile bir obje **birbirinin grafiğini sessizce ezer.**

★ Bu test collider adıyla sahiplik kuruyor. `ownerExists` collider'ı kabul
etmezse `create` reddeder ve test daha ilk adımda durur — belirti "graph not
found", sebep isim çözümü.

## 3. Kuplaj raporu bütün kapsamları görüyor mu

```powershell
Invoke-RtIpc script.run_file @{ path = 'scripts/test/rt_test_sim_couplings.py' }
Get-Content .\x64\Release\scripts\test\_sim_couplings_result.txt
```

- **Görmen gereken:** `RESULT: ALL PASSED` + `each declaration names the graph
  it came from`.
- **Bozuksa:** `declared` boş gelirse rapor tek grafiğe kapanmıştır. ★★ Kuplaj
  iki domain'i birleştirir, yani **hiçbir grafiğe tek başına ait değildir**;
  raporu kapsamla sınırlamak tam olarak göstermesi gereken şeyi gizler.

## 4. Kalan iki N-testi (regresyon)

```powershell
Invoke-RtIpc script.run_file @{ path = 'scripts/test/rt_setup_sim_render_scene.py' }
Invoke-RtIpc script.run_file @{ path = 'scripts/test/rt_test_sim_render.py' }
Invoke-RtIpc script.run_file @{ path = 'scripts/test/rt_test_sim_cache.py' }
```

- **Görmen gereken:** ikisi de `RESULT: ALL PASSED`.
- **Bozuksa:** render testi **iki** grafik kuruyor (gas + fluid). Biri düşerse
  ikinci `fresh_graph` çağrısı ilkini eziyordur — yani depolama isimle değil,
  tek slotla anahtarlı.

## 5. `rt.editor` kapsam seçimi

```powershell
Invoke-RtIpc script.run_file @{ path = 'scripts/test/rt_test_editor_state.py' }
Get-Content .\x64\Release\scripts\test\_editor_state_result.txt
```

- **Görmen gereken:** `RESULT: ALL PASSED` + `world scope drops the owner name`
  + `a refused scope change moved nothing`.
- **Bozuksa:** `selecting a scope with no graph is allowed` düşerse
  `setSimGraphScope` grafiğin varlığını şart koşuyordur — o zaman **boş durum
  ekrana hiç gelmez ve grafik yaratmanın UI yolu kapanır.**

★ Bu partide 1. maddedeki aynı tuzağın ikinci kopyası: reddedilen bir seçim
değişikliğinin tuvali yine de oynatması.

## 6. Nodes paneli — gözle (otomatik testi olmayan tek kısım)

- **Görmen gereken:** üstte `Scope [object|domain|world]` ve yanında sahip
  seçici. Bir domain seçince ya grafik gelir ya da **"No graph for this domain
  yet." + `Create graph`** düğmesi. Grafiği yarattığında tuvalde **tek bir
  node** olur: sahip node'u, adı sabit yazılı (düzenlenemez).
- **Bozuksa:** sahip node'unda isim seçici çıkıyorsa `isOwnerNode` panele
  bağlanmamıştır — kullanıcı düzenlemeyi dener, API reddeder, ve kontrol
  **bozuk görünür.**

★ `Clear Graph`'a bastıktan sonra tuval **boş kalmamalı**: sahip node'u geri
gelir. Boş kalıyorsa `simGraphClear` yeniden tohumlamayı atlıyordur ve o
tuvale yazılan her node **hiçbir şeyi adlandırmaz.**

## 7. Öksüz grafik göstergesi (yeni, ve sessiz bayatlığın tek engeli)

```powershell
# bir domain yarat, grafiğini yarat, sonra domain'i sil
Invoke-RtIpc sim_graph.list @{}
```

- **Görmen gereken:** domain silindikten sonra o grafik **listede olmamalı**
  (`removeFluidDomain` düşürüyor). Bir obje silindiğinde ise grafik listede
  kalır ama `owner_missing: true` der.
- **Bozuksa:** `owner_missing` hiç `true` olmuyorsa `ownerExists` her şeye evet
  diyordur — o zaman öksüz grafik **sağlıklı görünür**, çizmeye ve düzenleme
  kabul etmeye devam eder, hiçbir şeyi sürmez.

★★ Bu, fracture UI state'inin sahne değişimini sağ kalmasıyla **aynı şekil.**

## 8. Granül regresyonu (bu partiyle ilgisiz, izole sahne gerekir)

```powershell
python .\scripts\test\rt_test_granular_soft_stability.py
```

- **Görmen gereken:** `PASS`, soft `below_load ever=True`, stiff `ever=False`.
- **Bozuksa:** bu parti granül koduna dokunmadı; düşerse sebep başka yerdedir.

★ Sahnede başka etkin domain varsa test **kendini durdurur** (`fluid.step` her
etkin domain'i ilerletir). Önce `fluid.remove_domain @{ domain = '<ad>' }` —
parametre adı `domain`, `name` değil.

## 9. Çakışık kutu siyah bandı — ✅ ÇÖZÜLDÜ (2026-08-16), kontrol listesinde DEĞİL

Bu madde önceki listelerden **yanlışlıkla taşındı**. Arıza 2026-08-16'da
çözüldü: `docs/dev/VOLUME_BOX_REENTRY_POSTMORTEM.md` (REFERANS). Kök, gaz
marşının kutu çıkışında ışını sabit `tFar + 0.002` mesafeyle ilerletip
`skipGasVolumes`'u `false` bırakmasıydı; çare **maske**, mesafe değil.

★★★ **Ders, bu partide İKİNCİ kez yaşandı.** Aynı listede `flow_source.delete`
borcu da kapanmışken açık yazılıydı ve düzeltildi — sonra bir madde aşağıda
aynı hata tekrarlandı. **Bir kontrol listesini devralırken maddeleri kopyalama,
her birinin hâlâ açık olduğunu DOĞRULA.** Kapanmış bir işi açık göstermek
ajanı ve insanı çözülmüş bir şeye tekrar yollar.

★ Kuralın kendisi hâlâ bağlayıcı ve ölçüm hâlâ anlamlı — ama **regresyon**
olarak, "doğrulanmamış arıza" olarak değil: `density_samples / volume_rays`
sağlıklı sahnede de ~0,9 olmalı. Postmortem'in kendi kuralı: *sağlıklı durumda
DA ölç.*

## Bu partide alınan kararlar

- **Ayrı `ScopedGraph` tipi YOK.** `scope` + `owner` + `ownerNodeId` doğrudan
  `SimulationNodeGraph`'ın alanları. Gerekçe ve tam sözleşme tablosu:
  `SIMULATION_NODE_OBJECT_MODEL.md` §8.
- **Grafikler sahnede** (`scene_data.h`), statikte değil — yani sahne değişimini
  sağ kalmazlar. Eskiden kalırlardı.
- **Override katmanı kapsamlı DEĞİL.** Aynı anahtarı yazan iki grafik tek yazılı
  değeri paylaşır; kapsamlı bir geri alma, ikinci temizleyen grafiği **hiçbir
  şeyi geri koyamaz** hâle getirirdi.
- **Kuplaj raporu bütün kapsamları tarar** (§4.3: genel görünüm bir rapordur,
  dördüncü bir düzenleme yüzeyi değil).
- **Object kapsamı iki ismi de kabul eder** (obje ve collider) — MSF alanının
  source object'e, maddenin collider'a anahtarlı olması yüzünden.

## Bu partide bilerek YAPILMAYAN

- **Serileştirme.** Grafikler kaydedilmiyor; eskiden de kaydedilmiyordu (statikti).
  Ama artık sahnede oldukları için proje açılışında **temizleniyorlar** — davranış
  değişti, ve doğru yönde.
- **Obje silme/yeniden adlandırmada grafik düşürme.** Hooklanmadı; `owner_missing`
  ile **ölçülüyor**. Hepsini hooklamış gibi yapmaktansa görünür kılmak seçildi.
- **§8 adım 4–6:** World kapsamı, Object kapsamına MSF göçü, ölçülen
  etkileşimler.
- **`granular_enabled` kadranı** — restart semantiği ölçülmediği için bilerek
  açılmadı. Ölçüp açmak ayrı bir iş.
- **String domain parametreleri** (`backend`, `boundary`, `render_mode`,
  `coord_space`) — override katmanı için ikinci bir metin tablosu gerekiyor;
  bu partide sayısal yüzey genişletildi, metin yüzeyi genişletilmedi.
- Inspector'da `Create…` **bilerek yapılmadı**: node hiçbir varlık yaratmaz.

## Açık borçlar (değişmedi)

- `script.console` IPC metodu · Foam'un script yüzeyi · `WorldThermalState`'in
  script yüzeyi (World kapsamı bu yüzden boş) · `splat_material` slotu.
- ★ **`flow_source.delete` borcu KAPALI** — `flow_source.remove` zaten var;
  eski not yanlıştı, düzeltildi (bkz. object model §7).
- D.4'ün kalan yarısı: Geometry / Material / Terrain / Animation çizimlerinin
  `Nodes` penceresine teker teker taşınması.
