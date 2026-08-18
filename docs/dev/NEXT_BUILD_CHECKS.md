# Sıradaki build — sıralı kontrol listesi

> **Durum:** AKTİF — 2026-08-17 ikinci partisi (D.4 Nodes sekmesi + `rt.editor`).
> Her partide üzerine yazılır.

Sıra ölçütü: bağımsız ve hızlı görülen önce, başkasının sonucunu maskeleyen sonra.

Bu parti **derlenmedi** — kod yazıldı, derleme kullanıcıda.
`audit_ipc_capabilities.py`: **293 metot / 30 namespace**, hepsi sınıflandırılmış.

**Yeni dosyalar** (ikisi de `.vcxproj` + `.filters`'a eklendi):
`source/src/UI/scene_ui_simnodes.cpp`, `source/src/Api/RtApiUiState.cpp`.

---

## 1. `rt.editor` — önce bu, çünkü 2'nin ölçüm aleti bu

```powershell
Import-Module .\scripts\ipc\RtIpc.psm1 -Force
Invoke-RtIpc script.run_file @{ path = 'scripts/test/rt_test_editor_state.py' }
Get-Content .\x64\Release\scripts\test\_editor_state_result.txt
```

- **Görmen gereken:** `RESULT: ALL PASSED`.
- **Bozuksa:** "only one editor open after…" satırı düşerse alt panellerin
  karşılıklı dışlaması bozulmuş demektir; ekranda iki panel üst üste görünür.

★ **En sinsi başarısızlık burada:** `a refused call changed nothing`. İlk
yazımda `setBottomEditor` **önce hepsini kapatıyor**, sonra ismi tanımadığını
fark ediyordu — yani reddedilen çağrı ekranı boşaltıyordu. Yazarken düzeltildi
(isim artık hiçbir şeye dokunmadan doğrulanıyor); test bu davranışı kilitliyor.
Düşerse belirti "hata verdi **ve** ekran boşaldı" olur: kullanıcı hatayı görür,
yan etkiyi görmez.

## 2. Nodes sekmesi — gözle (otomatik testi olmayan tek kısım)

Alt şeritte artık **tek `Nodes` sekmesi** ve yanında alan seçici var. Eski dört
sekme (Graph/terrain, AnimGraph, Geometry, Material) **kaldırıldı**.

- **Görmen gereken:** şerit `Dope Sheet | Graph Editor | Console | Python |
  Nodes [seçici] | Assets`. Seçiciden Geometry seçince eski Geometry Graph
  penceresi açılıyor (çizimi henüz taşınmadı — bu plana uygun).
- **Bozuksa:** bir alan seçilince hiçbir şey açılmıyorsa `closeOtherBottomPanels`
  indeksleri kaymıştır (Nodes = 7).

★ Dördünün **çizim gövdesine dokunulmadı**, yalnızca giriş noktaları katlandı.
Bir panel bozulduysa sebep sekme yönlendirmesidir, panelin kendisi değil.

## 3. Simülasyon node paneli — kullanıcı raporunun cevabı

Sağda **Properties** kenar çubuğu var: node açıklaması, pin rehberi, ve gerçek
sahne isimlerinden seçen alanlar.

- **Görmen gereken:** boş sahnede `Nodes` açıp Domain node'u eklediğinde sağda
  "No simulation domains in this scene" uyarısı ve **Domain node'un domain
  YARATMADIĞINI** söyleyen açıklama. Bir domain oluşturduktan sonra aynı yerde
  isim seçici dolu gelmeli.
- **Bozuksa:** seçici boş kalıyorsa `rtapi::listFluidDomains` sahneye bağlı değil
  demektir (`rtapi is not bound` mesajı beklenir).

★ **En sinsi başarısızlık:** seçicide isim **var** ama seçmek node'u
değiştirmiyor. Sebep `simGraphSetNodeText`'in o anahtarı tanımaması olur ve
belirti "tıkladım, bir şey olmadı" — hata satırı toolbar'ın altında kırmızı
görünür, oraya bak.

## 4. Granül regresyonu (önceki partiden, izole sahne gerekir)

```powershell
python .\scripts\test\rt_test_granular_soft_stability.py     # PASS bekleniyor
```

- **Görmen gereken:** `PASS`, `below_load ever=True final=False`.
- **Bozuksa:** `detached > 0` veya `peak overburden` 16 kPa'nın çok altındaysa
  `FluidParticles::compact()` sıkıştırma yolu bozulmuştur.

## 5. Node katmanı N0–N7 (önceki parti, regresyon kontrolü)

```powershell
Invoke-RtIpc script.run_file @{ path = 'scripts/test/rt_setup_sim_graph_scene.py' }
Invoke-RtIpc script.run_file @{ path = 'scripts/test/rt_test_sim_graph.py' }
Invoke-RtIpc script.run_file @{ path = 'scripts/test/rt_test_sim_couplings.py' }
Invoke-RtIpc script.run_file @{ path = 'scripts/test/rt_setup_sim_substance_scene.py' }
Invoke-RtIpc script.run_file @{ path = 'scripts/test/rt_test_sim_substance.py' }
Invoke-RtIpc script.run_file @{ path = 'scripts/test/rt_setup_sim_render_scene.py' }
Invoke-RtIpc script.run_file @{ path = 'scripts/test/rt_test_sim_render.py' }
Invoke-RtIpc script.run_file @{ path = 'scripts/test/rt_test_sim_cache.py' }
```

Beşi de `RESULT: ALL PASSED` kalmalı. Bu partide node sınıflarına yalnızca
`metadata.description` eklendi; davranış değişmedi, yani bir düşüş olursa
`description` eklemesi bir ctor'u bozmuş demektir.

★ `PASSED SO FAR ... NOT VERIFIED` bir BAŞARI DEĞİLDİR.

## 6. Çakışık kutu siyah bandı (HÂLÂ DOĞRULANMADI)

Sahne: bir fluid domain + bir gas domain, **aynı sınırlarla**.

- **Görmen gereken:** `density_samples / volume_rays` **0,9 civarı**.
- **Bozuksa:** oran **0,02 ve altına** düşer, `volume_rays` birkaç kat artar.

★ **En sinsi:** bant kaybolur ama maliyet yüksek kalır. **Gözle değil, oranla.**

## 7. Karanlık izoyüzey / kayıtlı graph (önceki partiden, sırayla en son)

`DataType`'a beş değer sona eklendi; eski bir proje açıp materyal/geometri
graph'larının doğru pin tiplerini gösterdiğini gör. Belirti "node yanlış tipte
veri okuyor" olur, yükleme hatası değil.

---

## Bu partide alınan karar: `rt.ui` istisnası yeniden yazıldı

CLAUDE.md güncellendi. Ayrım artık "UI mi değil mi" değil, **çizim çağrısı mı
değer mi**:

- `rt.ui` (çizim) süreç içinde kalır — frame'in draw context'i dışında anlamsız.
- `rt.editor` (**yeni**) hangi editörün açık olduğunu değer olarak verir ve
  IPC'den geçer.

Gerekçe: bu deponun en pahalı hata sınıfı **panelin yalan söylemesi**, ve
yalnızca çekirdeği okuyabilen bir ajan o ayrışmaya **yapısal olarak kördür**.

★★ Widget sürüşü ("X yazan düğmeye bas") bilerek AÇILMADI: etiketleri yük
taşıyan hale getirir ve UI'yi tekrar otorite yapar.

## Bu partide düzeltilen arıza

**`EditorState` ilk halinde ilk açık paneli döndürüyordu.** İki panel birden
açık olsaydı okuyucu birini raporlayıp diğerini **gizlerdi** — yani en çok
sorulacak arızayı göremeyecek bir okuyucu. `open_editors` listesi eklendi;
`bottom_editor` artık o listenin ilk elemanı.

## Açık borçlar (değişmedi)

- `script.console` IPC metodu · Foam'un script yüzeyi · `WorldThermalState`'in
  script yüzeyi · `flow_source.delete` · `splat_material` slotu.
- Granül `below_load` göstergesi anlık — yapışkan gösterge panel sayısının
  anlamını değiştireceği için **kullanıcı kararına bırakıldı**.
- D.4'ün kalan yarısı: Geometry / Material / Terrain / Animation çizimlerinin
  `Nodes` penceresine **teker teker** taşınması.
