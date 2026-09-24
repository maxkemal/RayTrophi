# Domain taşınınca emitterler geride kalıyordu

> **Durum:** AKTİF — 2026-09-24'te yazıldı, RUNTIME DOĞRULANMADI (derleme kullanıcıda).

## Belirti (kullanıcının tarifi)

> "gaz sisteminde domain taşınsa da emitterler taşınmıyor; garip şekilde cache
> oluşursa domain taşınsa da cache silinmediğinden eski konumuna zıplıyor"

## ★★★★★ İki belirti, TEK sebep — ve yanlış hikâye daha inandırıcı

Cache **siliniyordu**. Hem gizmo yolu hem panel yolu `clearSimFrameCache()`
çağırıyor, ve `computeSimConfigSignature()` zaten `bounds_min/max`'i hash'liyor.

Olan şuydu: domain taşınıyor, cache temizleniyor, **emitterler eski dünya
konumlarında kalıyor**, ve yeniden simüle edilen duman yine eski yerde
beliriyor. Dışarıdan bu "cache silinmemiş, eski konumuna zıpladı" gibi görünür
— ve cache zaten şüpheli bir şey olduğu için o hikâye daha inandırıcıdır.

> **Bayat olan cache değil, EMITTERLERDİ.**

★★★ Ders: bir belirti tanıdık bir suçluyu işaret ediyorsa, o suçlunun gerçekten
o işi yapıp yapmadığını **ölç**. Burada tek bir `flow_source.list` okuması
hikâyeyi baştan çürütürdü.

## Neden hiçbir test yakalamadı

**Domain taşıma IPC'de hiç yoktu.** Yalnızca gizmo ve panelden yapılabiliyordu.
CLAUDE.md kural 1'in tarifi tam olarak bu: panelden erişilen bir yetenek **test
edilemez** sayılır. Önce ölçü aleti (`sim.move_domain`) eklendi, sonra arıza
kapatıldı.

## Sözleşme

`ParticleSimulationSystem::translateGridDomain(index, delta)` — kutuyu taşır ve
ona bağlı olanları taşır. `carryGridDomainAnchors(index, delta)` ise yalnızca
bağlıları taşır; gizmo ve panel bunu kullanır çünkü kutuyu kendileri yazıyor
(tek sürükleme hem taşıyıp hem yeniden boyutlandırabilir).

| Ne | Taşınır mı | Gerekçe |
|---|---|---|
| `bounds_min/max` | ✔ | domain'in kendisi |
| parent'sız flow source `position` | ✔ | domain'e bağlı |
| o kaynakların **keyframe** pozisyonları | ✔ | yoksa düzeltme **play'e basana kadar** yaşar |
| `parent_prev_position` | ✔ (sentinel değilse) | bayat kalırsa bir karelik devasa miras hız üretir |
| **parent'lı** kaynak | ✘ | zaten sahibi var; tek konumun iki otoritesi olamaz |

★★ Parent'lı kaynağın taşınmaması bir eksiklik değil **karar**: kuvvet alanının
domain'e bağlanmama gerekçesiyle aynı (bkz. `project_sim_node_object_model_scopes`).

## Dokunulan yerler

- `ParticleSimulation.h/.cpp` — `translateGridDomain`, `carryGridDomainAnchors`
- `scene_ui_gizmos.cpp` — sürüklemenin **taşıma bileşeni** kutu yazılmadan önce
  yakalanıyor, senkrondan **önce** taşınıyor (yoksa ilk yayımlanan kareye eski
  konum pişer)
- `scene_ui_simulation_domains.cpp` — panel de aynı devri yapıyor
- `RtApi.h` / `RtApiFluid.cpp` / `RtIpc.cpp` / `RtPython.cpp` — `sim.move_domain`
- `RtIpcSecurity.cpp` — **`sim.` namespace'i eklendi.** Yoksa `authorize()`
  fail-closed davranır ve metot hiçbir teşhis vermeden reddedilirdi.
- `scripts/ipc/Probe-DomainMoveCarriesSources.ps1`

★ Gaz ve sıvı **tek gövde**: ikisi de `SimulationGridDomainDesc` kullanıyor.
Ayrı `gas.move_domain` / `fluid.move_domain` iki gövde olurdu, ve bu depoda iki
gövde her zaman ayrışır.

## Açık

- Runtime doğrulanmadı.
- Domain **döndürme/ölçekleme** bağlıları taşımıyor — yalnızca öteleme ele
  alındı. Ölçekleme için kaynakların domain-lokal mi kalması gerektiği ayrı bir
  karar; şimdilik kasten dokunulmadı.

---

# İkinci tur: taşıma artık bake'i öldürmüyor

> 2026-09-24, aynı gün. Emitterler taşındıktan **sonra** ortaya çıkan şikâyet.

## Belirti

> "cache varsa turuncu domain gizmo mavi gizmoyu gizliyor; önce cache gizmosu
> taşınmalı, sonra mavi gizmoya ulaşılabiliyor"

Ve devamında kullanıcının kendi teşhisi, ki doğru olan oydu:

> "belki de turuncu gizmoya hiç gerek yok, taşıma da cache'i resetlememeli"

## ★★★★★ Cache'in `bounds`'u bilgi değil, yetkili kopyanın BAYAT İKİZİ

`CachedGridDomain::meta` bütün `SimulationGridDomainState`'i saklıyor — içinde
`bounds_min/max` ve `grid.origin` de var. Ama bir domain'in **nerede olduğunu**
authored descriptor bilir; cache onu yalnızca bir kez daha yazmıştır.

`restoreSimFrame` / `restoreSimFrameFromDisk` bu kopyayı **olduğu gibi** geri
kuruyordu. Yani taşınmış bir domain, cache'e geri her dönüşte baked konumuna
zıplıyordu: kutu, hacim ve duman eski yerde, tıklanabilir descriptor yeni yerde.
Kullanıcının "turuncu gizmo mavi gizmoyu gizliyor" diye tarif ettiği şey buydu —
iki ayrı gizmo değil, **aynı domain'in iki farklı konum otoritesi.**

> İki otoritesi olan bir konum, otoritesi olmayan bir konumdur.

## Karar: silmek yerine YENİDEN OTURTMAK

Eski davranış taşımada `clearSimFrameCache()` çağırıyordu — **bütün sistemlerin,
bütün domainlerin, bütün kareleri.** Bir kutuyu 2 metre kaydırmanın bedeli
saatlerce süren bir bake'ti.

Oysa bir ötelemede cache'in içinde geçersizleşen **hiçbir hücre yok**:

| Cache içeriği | Uzay | Ötelemede |
|---|---|---|
| density / temperature / fuel / interaction | **indeks** (hücre başına) | değişmez |
| `bounds_min/max`, `grid.origin` | dünya | canlı desc'ten yeniden yazılır |
| sıvı `particles.position` | dünya | kaydırılır |
| sıvı `particles.uvw`, `uvw_b` | dünya | kaydırılır (yoksa doku kayar) |
| `foam.position` | dünya | kaydırılır |
| `particles.velocity` | **yön** | dokunulmaz |

`rebaseRestoredGridDomainStates()` bunu `setGridDomainStates()` içinde yapıyor —
★★★ RAM cache'inin de disk bake'inin de restore'da geçtiği **tek boğaz nokta**.
İki yol, tek düzeltme; biri diğerinden ayrışamaz.

★★ Kapı bir **mesafe** testi değil, **saf-öteleme** testi: extent ve voxel_size
aynı kalmalı. Yeniden boyutlandırma hücre düzenini gerçekten değiştirir ve o
bake gerçekten ölüdür — o durum eskisi gibi imza yoluyla düşürülür.

## İmza tarafı: hash "taşındı ama eşdeğer" diyemez

`computeSimConfigSignature()` authored bounds'u hash'liyor, yani taşıma imzayı
değiştiriyor ve kare döngüsünün auto-invalidate'i cache'i **bir tik sonra**
düşürürdü. Hash bunu ifade edemez.

Çözüm imzanın anlamını bozmak değil: taşıyan taraf `acceptSimConfigAsBaked()`
diyerek yeni kurulumu "bake'in ait olduğu kurulum" ilan ediyor. İmza hâlâ mutlak
konumu hash'liyor — ki diğer her düzenleme için doğrusu bu.

★ Bu çağrı tehlikelidir ve öyle belgelendi: cache'i gerçekten uzlaştırmadan
çağırmak, bayat bir bake'i sonsuza kadar hiçbir belirti vermeden oynatır.

## Bilinen sınır — kapı DEĞİL, kayıt

Taşınan bir bake, yalnızca domain'in içeriği dünyaya bağlı bir şeyle
etkileşmediyse **fiziksel olarak** da doğrudur. Sahnede collider, force field,
zemin veya ikinci bir domain varsa replay eski etkileşimi yeni yere taşır.

Bunu kapı yapmadım: kullanıcı kutuyu taşıdığında dumanın gelmesini bekler, ve
her ihtimale karşı bake silmek tam olarak sökülen davranıştı. Ama sonuç **yanlış
görünmez, sadece eskidir** — bu deponun en pahalı hata sınıfı. O yüzden burada
yazılı.

## Yapılmayan: domain'e özgü invalidasyon

Kullanıcı haklı olarak "reset domaine özgü olmalı, şu an bütün sistem cache'i
siliniyor" dedi. Yapılmadı, çünkü `sim_frame_cache_` her karede domain başına
**yoğun bir vektör** tutuyor ve giriş başına geçerlilik bayrağı yok; tek domain
düşürmek "yarısı geçerli kare" diye yeni bir durum icat etmek demek.

★ Ve taşıma artık reset istemediği için bunun **en acil gerekçesi ortadan
kalktı**. Çözünürlük/voxel/yangın parametresi gibi gerçekten bayatlatan
düzenlemeler için hâlâ değerli — ayrı iş olarak açık.
