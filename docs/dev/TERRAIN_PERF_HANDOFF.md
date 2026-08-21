# Terrain performans işi — devir notu

> **Durum:** AKTİF — Faz 0 çekirdeği canlı doğrulandı; sıradaki iş aşağıda
> önceliklendirildi. 2026-08-21.

Bu not, terrain üretim maliyeti işini devralan ajan içindir. Tasarım kararları
[TERRAIN_SATMAP_COLORIZER_ROADMAP.md](TERRAIN_SATMAP_COLORIZER_ROADMAP.md)'de
(özellikle **Faz 0** ve **Ek A**); burada yalnızca **durum, ölçülmüş gerçekler,
tuzaklar ve sıradaki iş** var.

---

## 1. Nereden başladı, nerede bitti

Şikâyet: "4k terrain ilk üretimde uzun sürüyor." Ölçüldüğünde ortaya çıkan:

| | önce | sonra |
|---|---|---|
| 4096² terrain, kare döngüsü tıkanması | **6 323 ms** | **33 ms** (mesh 1024) |
| `accel.vulkan_solid.raster_geometry` | 5 587 ms, +7.4 GB | 422 ms |
| `accel.cpu.embree_build` | 2 821 ms, +6.6 GB | 155 ms |
| `terrain.mesh_fill` | 386 ms | (alan hâlâ 4096²) |

★★★ **Mesh doldurma hiçbir zaman darboğaz değildi** — toplam maliyetin ~%5'i.
Süre, çağrı döndükten *sonra* kare döngüsünde tembel kurulan hızlandırma
yapılarında geçiyordu. İlk üç saat bunu bulmaya gitti çünkü **ölçüm aleti oraya
bakmıyordu.**

---

## 2. Ölçüm aleti: `rt.perf` — önce bunu öğren

`RTPERF_SCOPE("ad")` ([PerfProfile.h](../../RayTrophiStudio/source/include/PerfProfile.h))
bir fazın süresini **ve çalışma kümesi deltasını** kayıt defterine yazar.
`perf.list` / `perf.get` / `perf.reset` / `perf.set_logging` ve `rt.perf.*`.

Mevcut bölümler:

```
terrain.graph.evaluate | .height | .aux_outputs | .finalize_mesh
terrain.mesh_fill | .create | .update | .publish_fields
terrain.splat_resize
scene.sync_transformed_vertices
accel.cpu.embree_build
accel.vulkan_rt.rebuild | .update_geometry | .raster_geometry
accel.vulkan_solid.raster_geometry
Renderer::rebuildBVH(...) | Renderer::rebuildBackendGeometry(GPU)
```

### ★★ Kare döngüsü tıkanmasını ölçme tekniği

`perf.list` IPC kuyruğuna **girmez** (bilerek), `scene.list_objects` girer.
İkisinin duvar saati farkı doğrudan **kare döngüsünün meşgul olduğu süredir**:

```powershell
Invoke-RtIpc perf.reset
Invoke-RtIpc terrain.create @{ name='T'; resolution=4096; height_scale=120.0 }
Measure-Command { Invoke-RtIpc scene.list_objects }   # tıkanma
Invoke-RtIpc perf.list | Sort-Object last_ms -Descending
```

Bu teknik olmadan tıkanma **görünmez**: uygulama donmuş gibi durur, hiçbir
sayaç bir şey söylemez.

### Okuma kuralları

- **Eksik bölüm 0 değil YOK.** `perf.get` `found:false` döner. Sıfır "ölçtüm,
  bedavaymış" diye okunur.
- **Boş `accel.*` bölümü "bedava" değil, "bu viewport modunda çalışmadı".**
  Yapılar tembel kurulur; Solid ve Rendered ayrı ayrı ölçülmeli.
- **Kayıtlı toplam duvar saatinden büyük olabilir** — Embree worker thread'de
  çakışık koşar. Hata değil.
- **`count` alanına bak.** Çift mesh doldurma yalnızca sayaç sayesinde görüldü;
  "son süre" raporlayan bir profiler onu yapısal olarak gösteremezdi.

---

## 3. Yapılan ve DOĞRULANAN

| İş | Durum |
|---|---|
| `rt.perf` ölçüm katmanı (4 dokunuş + descriptor) | ✅ canlı |
| `MeshProfileTimer.h` **silindi** | ✅ Scene Log'a yazan + makrosu kapalı ölü alet |
| `[MESHMEM]` geçici bloğu **silindi** | ✅ Triangle facade ölçüyordu |
| Mesh doldurmada geçici tamponların kaldırılması (~940 MB) | ✅ 4k'da +1.31 GB, staging yok |
| İndeks kopya döngüsü (100 M yineleme) kaldırıldı | ✅ |
| Çift mesh doldurma (`createTerrain`) | ✅ `count` 2 → 1 |
| Tembel hızlandırma yapısı ölçümü | ✅ 6.7 s'lik delik kapandı |
| **`mesh_resolution`** — vertex ızgarası alandan bağımsız | ✅ 6 323 → 33 ms |
| Alan ayrık meshte kıpırdamıyor | ✅ `sample_height` 103.227 → 103.227 |
| Alandan büyük mesh **reddediliyor** (kırpılmıyor) | ✅ |
| Görsel regresyon (dağ preset'i render) | ✅ |

**Reddedilen:** `P_orig`/`N_orig` sökme (402 MB). Depoda **46 dosyada 296 kez**
okunuyor — israf değil, yayılmış bir sözleşme. Tasarruf gerçekti, gerekçe yanlış.

**Derlenmemiş (son değişiklik):** `terrain.create(mesh_resolution=)`. İlk ölçüm
gösterdi ki sonradan küçültmek pahalı yapıyı bir kez yine de kurduruyor.

---

## 4. ⚠ DOĞRULANMAYAN — ilk iş bu olmalı

### 4.1 ★★★ Analiz maskeleri ayrık meshte hayatta mı

Kod yazıldı ve gerekçeli, **ama ölçülmedi.**

Vertex aynası (`updateTerrainMesh`, publish_fields) boyut testini eskiden
`vertexCount`'a karşı yapıyordu. Bu yalnızca iki ızgara aynı olduğu için
doğruydu; ayrıldıkları anda **her analiz alanı testi düşer ve
`remove_custom_attribute` ile silinirdi** — foliage/scatter maskesiz kalır,
hata yok, log yok, görünür sebep yok. Test artık **alan hücre sayısına** karşı
ve alanlar vertex ızgarasına bilinear yeniden örnekleniyor.

**Neden doğrulanamadı:** mesh vertex attribute'ları IPC'den okunamıyor.
`attr.list`/`attr.stats` simülasyon/MSF alanları için; mesh attribute'u yok.

**İki yol:**
1. Foliage'lı bir sahnede mesh'i düşürüp bitki dağılımına **gözle** bak —
   bitkiler kayboluyorsa veya düzgün dağılıyorsa aynalama kopmuştur.
2. **Daha iyi ve kural 1'in gerektirdiği:** `terrain.field_stats` benzeri bir
   metot aç — bir `analysisFields` girdisinin hem alan hem **mesh vertex**
   tarafındaki min/max/mean'ini döndürsün. O zaman bu sınıf hata bir daha
   gözle aranmaz. ★ Bu, yol haritasının Faz 5'indeki `terrain.export_field`
   ile aynı ihtiyaç.

### 4.2 Sculpt / dirty-sector yolu

`updateDirtySectors` vertex dizisini **alan koordinatlarıyla** indeksliyor.
Ayrık mesh'te yanlış vertex'leri oynatırdı; koruma eklendi (ayrık mesh'te tam
yola düşer) **ama denenmedi.** Belirtisi bir **sculpt hatası** gibi görünür,
çözünürlük hatası gibi değil.

### 4.3 Kaydet/yükle turu

`mesh_resolution` sidecar'a yazılıyor, okuma varsayılanı 0. Denenmedi.
Bozuksa sahne her açılışta tam mesh'e döner ve kimse sebebini bağlamaz.

---

## 5. ★★★ Solid mod belleği: EVET, ve sebebi belli

Sorulan soru buydu. Ölçüm: 4096² terrain (33.5 M üçgen) için
`accel.vulkan_solid.raster_geometry` **+7.4 GB** — bu partide ölçülen **en
büyük tek bellek tüketicisi**, mesh'in kendisinin (1.31 GB) 5.6 katı.

Sebep [VulkanViewportBackend.cpp:4885-4940](../../RayTrophiStudio/source/src/Backend/VulkanViewportBackend.cpp#L4885)'te:
flat `TriangleMesh` dalı **indeks tamponunu atıyor ve mesh'i köşe başına
açıyor** (de-index). Her üçgen köşesi için ayrı ayrı push ediliyor:

| Dizi | Köşe başına |
|---|---|
| `grp.positions` | 12 B |
| `grp.normals` | 12 B |
| `grp.uvs` | 8 B |
| `grp.matIds` | 4 B |
| **toplam** | **36 B/köşe = 108 B/üçgen** |

33.5 M üçgen × 108 B = **3.6 GB CPU staging**, üstüne GPU yüklemesinin kopyası
⇒ ölçülen ~7.4 GB.

★★ **Oysa flat mesh'in indeksleri VAR** (`geom->indices`) ve vertex'leri
kaynaklı. 4096² terrain: 16.8 M kaynaklı vertex, 100.6 M açılmış köşe —
**6× fark**. İndeksli çizime geçmek 3.6 GB'ı ~600 MB'a indirir.

★ Dikkat: üstteki facade-`Triangle` dalı de-index'i **gerektiriyor** olabilir
(kaynaksız soup). Bu iş yalnızca flat `TriangleMesh` dalını hedeflemeli —
depo kuralı zaten "yeni kod flat SoA'yı esas alır" diyor.

★ `matIds` köşe başına `uint32` tutuluyor; terrain'de hepsi aynı değer.
İndeksli çizim bunu da 6× küçültür.

**Faz 0 bunu 16× küçültür ama ORTADAN KALDIRMAZ.** İkisi bağımsız: biri üçgen
sayısını düşürür, diğeri üçgen başına maliyeti.

### ✅ ÇÖZÜLDÜ (parti 4, YAZILDI-derlenmedi)

Flat dal artık **kaynaklı vertex + indeks** yüklüyor. Çizim yolu `indexCount > 0`
iken zaten `vkCmdDrawIndexed` seçiyordu — yani altyapı hazırdı, yalnızca üretici
indeksleri atıyordu. Beklenen: 100.6 M köşe → 16.8 M vertex.

Kapsam bilerek dar tutuldu:

- **Skinli mesh'ler de-index KALDI.** `syncRasterSkinnedVertices` ve
  `patchRasterMeshTriangles` `rmb.vertexCount == indices.size()` kapısına
  bakıyor; onları kaynaklamak per-frame skinning güncellemesini **sessizce
  kapatırdı**, gürültülü biçimde bozmazdı.
- **Karışık düzen koruması** (`groupCanWeld`): bir grup hem kaynaklı hem açılmış
  vertex alırsa indeksli çizim gerisini atlar — eksik üçgen, hata yok.
- **`updateRasterMeshFromMeshSoA` kaynaklı düzeni tanıyor.** Tanımasaydı her
  fırça darbesi tam `buildRasterGeometry` tetiklerdi: bir **yavaşlama**, hata
  değil.
- **Facade `Triangle` dalına dokunulmadı** — o gerçekten kaynaksız bir soup.

Kontroller `NEXT_BUILD_CHECKS.md` §P4'te.

---

## 6. Sıradaki iş — öncelik sırası

1. **⚠ Analiz maskesi doğrulaması (§4.1).** Faz 0'ın en riskli parçası ve tek
   doğrulanmamış iddiası. Ölçüm metodu yoksa **önce ölçüm metodunu aç**.
2. ~~Solid raster indeksli çizim~~ ✅ **yazıldı** (§5). Derlenip §P4
   kontrollerinden geçmesi gerekiyor; özellikle **P4.3 sculpt hızı** sessiz
   regresyon riski taşıyor.
3. **`accel.vulkan_solid.raster_geometry` Rendered modunda da koşuyor** —
   5.9 s, ekranda gösterilmeyen bir temsil için; toplam tıkanmayı 7.1 s'den
   9.9 s'ye çıkarıyor. ★ "Gereksiz" demiyorum, **ölçmedim**: moda geri dönüş
   veya seçim/overlay için tutuluyor olabilir. Ama 7.4 GB'lık bir eager
   build'in yazılı gerekçesi olmalı ve yok. **Önce gerekçeyi ara, sonra sök.**
4. **`scene.sync_transformed_vertices` Solid'de 53 ms, Rendered'da 502 ms.**
   Ana thread'de, async BVH gönderilmeden önce. 10× fark açıklanmadı.
5. **Faz 0'ın kalanı:** alan çözünürlüğünde normal map (şu an normaller alandan
   *örnekleniyor* — bedava kısmı alındı, ama vertex ARASI detay ve siluet
   gerçekten kayıp, panel bunu yazıyor), scatter'ın vertex aynası yerine UV'den
   örneklemesi.
6. **`paint_resolution`** — ★★★ maske grafiği o çözünürlükte **değerlendirilmeden**
   bu kadranı açma. Yalnızca yukarı örnekleme satın alır ve panelin yalanı olur;
   yol haritasının en çok uyardığı şey bu. Kadran, değerlendirme yoluyla
   **birlikte** gelmeli.
7. **Heightmap içe aktarma** dosya çözünürlüğünden kurtarılmalı; `createTerrainFromHeightmap`
   hâlâ **8-bit** yüklüyor (256 seviye ⇒ teraslama) oysa node yolu 16-bit.
8. Sonra SatMap'in kendisi (Faz 1-5) + arazi tipi preset kütüphanesi (Faz 2b).

---

## 7. Devralan ajana notlar

- **Build KULLANICININ.** `msbuild` çalıştırma. Kodu yaz, kontrol listesi bırak.
- **Uygulama açıkken IPC'den kendin test edebilirsin** — bu partinin bütün
  ölçümleri öyle alındı. `render.start` görüntü üretir ve görüntü okunabilir,
  yani **görsel doğrulama da otomatik.**
- Test scriptleri **iki yere**: `scripts/test/` ve `x64/Release/scripts/test/`.
- ★ **Script içinden async graph evaluate'i BEKLEME.** Graph worker'ın finalize
  adımı ana thread'de koşar — script o thread'i tutuyorsa döngü asla tamamlanmış
  görmez. Bu tuzağa bir kez düşüldü; daha kötüsü, döngü sessizce düşseydi bütün
  yükseklikler 0 olur ve "alan değişmedi" iddiası **0 == 0 karşılaştırıp yeşil
  geçerdi.** IPC'den sür (her çağrı ayrı istek) ya da düz araziyi tespit edip
  iddiayı DOĞRULANMADI say.
- Dispatch'e dokunduysan: `gen_ipc_descriptors.py` + `audit_ipc_capabilities.py`
  + `verify_descriptor_claims.py`. Üçü de şu an yeşil (335 metot, 321 belgeli).
