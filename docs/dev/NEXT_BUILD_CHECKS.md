# Sıradaki build kontrolleri

> **Durum:** CANLI — her iş partisinde üzerine yazılır. Son güncelleme: 2026-08-21.

**Parti 3 + 4 birlikte derlenecek.** Parti 3 = Faz 0 çekirdeği
(`mesh_resolution`) — ilk derlemesi canlı doğrulandı, sonrasında
`terrain.create(mesh_resolution=)` eklendi. Parti 4 = Solid raster indeksli
çizim (aşağıda, §P4).

**Parti 3 — Faz 0 çekirdeği: `mesh_resolution`.** Vertex ızgarası artık alan
(analiz) ızgarasından bağımsız. Ölçülen kazanç: alan 4096 + mesh 1024 ⇒ kare
döngüsü tıkanması **7 142 ms → ~398 ms (~18×)**, analiz çözünürlüğü kaybı yok.

★ **Bilerek KAPSAM DIŞI:** `paint_resolution` (splat/satmap kadranı). Maske
grafiği o çözünürlükte **değerlendirilmeden** o kadranı açmak, yalnızca yukarı
örnekleme satın alır ve panelin yalan söylemesi demektir — yol haritası bunu
açıkça reddediyor. Ayrı parti.

## Değişenler

| Dosya | İş |
|---|---|
| `TerrainSystem.h` | `mesh_resolution` + `meshGridWidth/Height()` + `meshMatchesField()` |
| `TerrainManager.cpp` | Mesh alandan **örneklenerek** kuruluyor (yükseklik box-filter, normal alandan); analiz alanları vertex ızgarasına **yeniden örnekleniyor**; dirty-sector yolu ayrık meshte tam yola düşüyor; serileştirme |
| `RtApi.h` / `RtApiTerrain.cpp` | `TerrainInfo.mesh_resolution/mesh_width/mesh_height`, `setTerrainMeshResolution` |
| `RtIpc.cpp` / `RtPython.cpp` | `terrain.set_mesh_resolution`, `rt.terrain.set_mesh_resolution`, `mesh_grid` alanı |
| `scene_ui_terrain.hpp` | Panelde "Mesh Resolution" kadranı + üçgen sayısı + **ne kaybedildiği** |
| `RtIpcMethodDescriptors.cpp` | ÜRETİLDİ |
| `scripts/test/rt_test_terrain_mesh_resolution.py` **(YENİ)** | + `x64/Release/` kopyası |

Yeşil: `audit_ipc_capabilities.py` OK · `gen_ipc_descriptors.py` 335/321 ·
`verify_descriptor_claims.py` OK.

---

## Parti 3'ün canlı sonuçları (ilk derleme)

| Kontrol | Sonuç |
|---|---|
| 1. Kimlik kapısı (`mesh_resolution: 0`, `mesh_grid` = alan) | ✅ |
| Alan ayrık meshte kıpırdamıyor | ✅ 4096² sabit, `sample_height` 103.227 → 103.227 |
| Alandan büyük mesh reddediliyor | ✅ kırpılmıyor, mesajla reddediliyor |
| 0 geri yüklüyor | ✅ |
| 3. Kazanç | ✅ **6 323 ms → 33 ms**; solid raster 5 587 → 422 ms, embree 2 821 → 155 ms |
| 2. Analiz maskeleri | ⚠ **DOĞRULANMADI** — vertex attribute'ları IPC'den okunmuyor |

⚠ **Bu derlemeye eklenen:** `terrain.create(mesh_resolution=)`. İlk ölçüm
gösterdi ki sonradan küçültmek pahalı yapıyı bir kez yine de kurduruyor.

---

## Sıralı kontroller

### 1. Eski sahneler DEĞİŞMEDİ mi (kimlik kapısı)

Kaydedilmiş bir terrain projesi aç, ya da `mesh_resolution` dokunmadan yeni bir
terrain üret.

- **Görmen gereken:** `terrain.get` → `mesh_resolution: 0`, `mesh_grid` alan
  ızgarasıyla **aynı**; görüntü öncekiyle aynı.
- **Bozuksa ne demek:** varsayılan 0 olmaktan çıkmış demektir; her eski sahnenin
  geometrisi sessizce değişir. Bu bozuksa aşağıdaki her ölçüm kirlidir.

### 2. ★★★ Analiz maskeleri hayatta mı (EN SİNSİ MADDE)

`scripts/test/rt_test_terrain_mesh_resolution.py` çalıştır, sonra **foliage'lı**
bir terrain'de mesh'i düşürüp bitki dağılımına bak.

- **Görmen gereken:** bitkiler aynı yerlerde; sadece biraz daha kaba.
- **Bozuksa ne demek:** bitkiler **kaybolduysa** veya tamamen düzgün dağıldıysa,
  analiz alanları vertex aynasına ulaşmıyordur. Eski kod boyut testini
  `vertexCount`'a göre yapıyordu ve ayrık meshte her maskeyi **sessizce
  siliyordu** — hatasız, logsuz. Yeniden örnekleme kodu buna karşı yazıldı;
  bu madde onun testidir.

### 3. Kazanç gerçek mi

```
Invoke-RtIpc perf.reset
Invoke-RtIpc terrain.create @{ name='F4k'; resolution=4096; height_scale=120.0 }
Measure-Command { Invoke-RtIpc scene.list_objects }      # ~7 s bekleniyor
Invoke-RtIpc terrain.set_mesh_resolution @{ name='F4k'; mesh_resolution=1024 }
Invoke-RtIpc perf.reset
Invoke-RtIpc terrain.create @{ name='F4kB'; resolution=4096; height_scale=120.0 }
Invoke-RtIpc terrain.set_mesh_resolution @{ name='F4kB'; mesh_resolution=1024 }
Measure-Command { Invoke-RtIpc scene.list_objects }
Invoke-RtIpc perf.list
```

- **Görmen gereken:** `accel.vulkan_solid.raster_geometry` ~6 400 ms'den
  ~320 ms'ye düşmeli; `terrain.mesh_fill` de düşmeli ama o zaten küçüktü.
- **Bozuksa ne demek:** hızlandırma yapısı düşmediyse mesh gerçekten
  küçülmemiştir — `terrain.get` ile `mesh_grid`'i doğrula.

### 4. Görsel: gölgeleme yaşıyor mu, siluet düşüyor mu

Aynı dağ sahnesini mesh 1024 ve mesh 256 ile render et.

- **Görmen gereken:** gölgeleme detayı büyük ölçüde **duruyor** (normaller
  alandan örnekleniyor); **siluet** gözle görülür kabalaşıyor.
- ★ Siluet farkı **beklenen ve dürüstçe ilan edilmiş** bir kayıp, bug değil.
  Alan çözünürlüğünde normal map bu partide **yok** — vertex arası detay hâlâ
  kayıp. Gölgeleme de yassılaştıysa `sample_normal` alan yerine mesh ızgarasını
  okuyordur.

### 5. Sculpt / dirty-sector yolu

Ayrık mesh'li (`mesh_resolution` düşük) bir terrain'i fırçayla sculpt et.

- **Görmen gereken:** doğru yerde deformasyon.
- **Bozuksa ne demek:** yanlış yerde tümsek çıkıyorsa `updateDirtySectors`
  koruması devreye girmiyordur — o yol alan koordinatlarıyla vertex indeksliyor
  ve ayrık meshte **başka vertex'leri** oynatır. Belirtisi bir sculpt hatası
  gibi görünür, çözünürlük hatası gibi değil.

### 6. Kaydet/yükle turu

`mesh_resolution = 512` ayarla, kaydet, yükle.

- **Görmen gereken:** 512 geri geliyor.
- **Bozuksa ne demek:** sidecar alanı yazılmıyor; sahne her açılışta tam
  çözünürlüklü mesh'e dönerek yavaşlar ve kimse sebebini bağlamaz.

---

## §P4 — Solid raster indeksli çizim (YAZILDI, derlenmedi)

`VulkanViewportBackend::buildRasterGeometry`'nin flat `TriangleMesh` dalı,
mesh'in **zaten sahip olduğu** indeks tamponunu atıp geometriyi köşe başına
açıyordu. 4096² terrain'de ölçülen maliyet **+7.4 GB** — mesh'in kendisinin
(1.31 GB) 5.6 katı ve bu partilerin en büyük tek tahsisatı.

Artık kaynaklı vertex + indeks yükleniyor. Çizim yolu `indexCount > 0` iken
zaten `vkCmdDrawIndexed` seçiyordu, yani shader/pipeline tarafı değişmedi.

Beklenen: 100.6 M köşe → **16.8 M vertex (6×)**; ~3.6 GB CPU staging → ~600 MB.

| Değişen | İş |
|---|---|
| `RasterTriGroup` | `indices` alanı |
| flat dal | skin'siz mesh'te kaynaklı+indeks; **skinli mesh de-index KALIYOR** |
| yükleme bloğu | indeks tamponu oluştur+yükle, `indexCount` ata |
| `updateRasterMeshFromMeshSoA` | kaynaklı düzeni tanıyor |

★ Facade `Triangle` dalına **dokunulmadı** ve dokunulmamalı: o gerçekten
kaynaksız bir soup. Temizlik fırsatçı — flat SoA yolu düzeldi, facade yolu
kendi göçünde ölecek.

### P4.1 — Solid modda her şey görünüyor mu

Terrain + normal mesh + import edilmiş model + scatter'lı bir sahne, Solid mod.

- **Görmen gereken:** hepsi eskisi gibi.
- **★ En sinsi hâli: bir nesnenin YARISI çizilir.** Bir grup hem kaynaklı hem
  açılmış vertex alırsa indeksli çizim gerisini atlar — hata yok, eksik üçgen.
  Buna karşı `groupCanWeld` koruması kondu; eksik geometri görürsen ilk oraya bak.

### P4.2 — Bellek gerçekten düştü mü

4096² terrain üret, `perf.list`.

- **Görmen gereken:** `accel.vulkan_solid.raster_geometry` RSS deltası
  **+7.4 GB → ~+1.5 GB**.
- **Bozuksa ne demek:** düşmediyse flat dal kaynaklı yola hiç girmiyordur
  (`meshHasSkinning` yanlış true, ya da `groupCanWeld` false).

### P4.3 — ★ Sculpt hızı (sessiz regresyon riski)

Ayrık olmayan bir terrain'i fırçayla sculpt et.

- **Görmen gereken:** eskisi gibi akıcı.
- **Bozuksa ne demek:** takılıyorsa `updateRasterMeshFromMeshSoA` `false`
  dönüyor ve her fırça darbesi tam `buildRasterGeometry` tetikliyordur. Bu bir
  **yavaşlama** olarak görünür, hata olarak değil — kimse bunu indeksleme
  değişikliğine bağlamaz.

### P4.4 — Skinli karakter

Animasyonlu bir karakteri Solid modda oynat.

- **Görmen gereken:** deformasyon çalışıyor. Skinli yol bilerek de-index kaldı
  (`syncRasterSkinnedVertices` `vertexCount == indices.size()` kapısına bakıyor).

### P4.5 — Seçim konturu

Bir flat mesh seç. Kontur çizimi de `indexCount`'a bakıyor; kontur kaybolduysa orası.
