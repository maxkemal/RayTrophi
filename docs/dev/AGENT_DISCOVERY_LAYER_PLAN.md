# Agent Discovery Layer — Kendi Kendini Anlatan API

> **Durum:** AKTİF — İkinci parti yazıldı (2026-08-19), DERLENMEDİ.
> Kabul ölçütleri §9'da güncel durumla işaretli; kontrol listesi
> [NEXT_BUILD_CHECKS.md](NEXT_BUILD_CHECKS.md).
> **Bağımlılıklar:** Mevcut IPC altyapısı (Faz 4c); viewport ölçüm katmanı
> [AGENT_VIEWPORT_MEASUREMENT_PLAN.md](AGENT_VIEWPORT_MEASUREMENT_PLAN.md)
> ile birlikte çalışır. Ajan pipeline kararları
> [AGENT_PIPELINE_ARCHITECTURE.md](AGENT_PIPELINE_ARCHITECTURE.md) bu
> katmanın gerçek kullanımda sınanmasından sonra netleşir.
>
> Son güncelleme: 2026-08-19

---

## 0. Neden

Bugün (2026-08-18) bir AI ajanı RayTrophi'ye IPC üzerinden bağlandığında
~214 method var. Hiçbir endpoint "ne yapabilirim, hangi method'lar var,
parametreleri ne" sorusunu cevaplamıyor. Ajan ya 500 sayfalık SDK dokümanını
okumalı ya da method isimlerini ezberlemeli.

Üç problem:

1. **Keşfedilemezlik:** ajan bağlandığında yetenekleri öğrenemez.
2. **Elle senkronizasyon:** her yeni özellik IPC + script'e aynalınır
   ([AGENTS.md](../../AGENTS.md) kuralı), ama discovery katmanı elle
   güncellenmeli — kırılgan, senkrondan çıkar.
3. **Ajan koordinasyonu:** çok ajanlı sistemlerde (MCP, AutoGen, CrewAI)
   yönetici/kontrolcü/çalışan rolleri tanımsız.

---

## 1. Mimari kararlar

### 1.1 ★ Oto-kayıt (dispatch ↔ discovery senkronizasyonu)

Mevcut dispatch: RtIpc*.cpp dosyalarında `if (method == "...")` chain.
Mevcut denetim: `scripts/audit_ipc_capabilities.py` bu pattern'i parse eder.

**Karar (2026-08-18):** Her IPC method kendi descriptor'ını **tanımlama
noktasında** bildirir. Static initializer pattern ile global `MethodRegistry`'ye
kaydolur. Discovery handler'ları bu registry'den okur.

> ### ★★★ REVİZE (2026-08-19): descriptor tablosu ÜRETİLİYOR
>
> Elle kayıt denendi ve **ölçüldü**: 300 metot kaydedildi, 299'unda parametre
> yoktu, 297'sinde özet yerine `"<metot adı> operation"` yazıyordu — ve
> `agent.discover` yine `coverage ≈ 1.0` diyordu. 300 satırlık elle bakım
> *doğru* kalmıyor, yalnızca *mevcut* kalıyor.
>
> Bugünkü model ikiye ayırıyor:
>
> | Yarı | Kaynak | Neden |
> |---|---|---|
> | Parametreler, tipler, zorunluluk, varsayılanlar, güvenlik yetkisi | `scripts/gen_ipc_descriptors.py` dispatch kodunu **okur** | Makine doğrulayabilir → asla ayrışamaz |
> | Özet, notlar, birimler, tag, ilişkili metotlar | `scripts/ipc_descriptor_overlay.json` (elle) | Makinenin bilemeyeceği tek şey: metodun NE İÇİN olduğu |
>
> `RtIpcMethodDescriptors.cpp` artık **üretilmiş bir dosyadır, elle
> düzenlenmez.** Yeni bir metot eklerken sırayla: dispatch'i yaz →
> `python scripts/gen_ipc_descriptors.py` → overlay'e bir satır özet ekle →
> tekrar üret. Overlay'de karşılığı olmayan metot `documented = false` ile
> çıkar ve `agent.discover` bunu **documented_coverage** olarak raporlar; yani
> eksik belge gizlenmez, ölçülür.
>
> Üretici üç gizli parametre kalıbını da çözer, çünkü ilk geçiş bunlara kördü:
> alan makroları (`RT_GAS_JSON(fire_enabled, bool)`), tüm isteği bir yardımcıya
> devreden handler'lar (`applyFlowSourceJson`) ve yerel lambda setter'lar
> (`flt("speed", info.speed)`). Bunlar olmadan `flow_source.create` sıfır,
> `particle.add_emitter` üç parametreli görünüyordu — gerçekte 25'er tane var.

```cpp
// RtIpcMethodRegistry.h
struct MethodParam {
    const char* name;
    const char* type;        // "string"|"float"|"int"|"bool"|"vec3"|"matrix"
    bool required;
    const char* description;
    const char* default_value; // nullptr if no default
    const char* enum_values;   // nullptr or "a|b|c"
};

struct MethodDescriptor {
    const char* name;        // "fluid.create_domain"
    const char* domain;      // "fluid"
    const char* summary;
    const char* notes;       // nullable
    const char* access;      // "read"|"write"|"render"|"admin"
    const char* capability;  // "Read" | "SceneWrite" | "Render" | ... (2026-08-19)
    bool undoable;
    const char* return_type;
    const char* tags;        // "simulation|fluid|create|domain"
    const char* related;     // "fluid.get|fluid.seed"
    const MethodParam* params;
    int param_count;
    bool documented;         // false = schema is real, prose was never written
};

struct MethodRegistration {
    MethodRegistration(const MethodDescriptor& desc);
};
```

**Kullanım (2026-08-18 tasarımı — artık üretici yazıyor, ama emitilen kod
aynı şekle sahip):**

```cpp
static const MethodParam fluid_create_params[] = {
    {"name",       "string", true,  "Domain name",      nullptr, nullptr},
    {"domain_min", "vec3",   true,  "AABB minimum",     nullptr, nullptr},
    {"domain_max", "vec3",   true,  "AABB maximum",     nullptr, nullptr},
    {"voxel_size", "float",  true,  "Grid resolution",  nullptr, nullptr},
    {"type",       "string", true,  "fluid or gas",     nullptr, "fluid|gas"},
};
static const MethodRegistration reg_fluid_create({
    "fluid.create", "fluid",
    "Create a new APIC liquid or gas grid domain",
    "After creating, seed with fluid.seed for liquid...",
    "write", false, "FluidDomainInfo",
    "simulation|fluid|create|domain|liquid|gas",
    "fluid.get|fluid.update|fluid.seed",
    fluid_create_params, 5
});
if (method == "fluid.create") { /* mevcut lambda aynen */ }
```

**Denetim (2026-08-19'da yazıldı):** `audit_ipc_capabilities.py` artık ikinci
geçiş olarak `gen_ipc_descriptors.py --check` çalıştırır; üretilmiş tablo
dispatch ile ayrışmışsa **FAIL eder**. Yani "IPC'de var ama şeması yok" durumu
CI'da yakalanır — bir ajan için o durum "bu yetenek yok" demektir.

### 1.2 ★ Ajan rolleri (informational)

| Rol | Açıklama | Tipik method'lar |
|-----|----------|------------------|
| **manager** (yönetici) | Hedefi parçalar, kaliteyi doğrular, yeniden dener | `agent.discover`, `agent.search_capabilities`, `agent.get_state_summary`, `viewport.probe` |
| **controller** (kontrolcü) | Çok adımlı workflow planlar, hata kurtarır | `agent.describe`, `agent.list_methods`, `batch`, `agent.get_state_summary` |
| **worker** (çalışan) | Tek IPC çağrısı yapar, ham sonuç raporlar | `scene.*`, `material.*`, `fluid.*`, ... |

★ Roller **informational** — RayTrophi zorlamaz. Security token'ların
capability bitmask'i (Read/SceneWrite/Render/...) zaten erişimi kontrol eder.
Roller bunun üzerine **semantic** bir katman ekler: bir MCP host'u veya
orchestrator framework'ü bu bilgiyi okuyarak ajan dağılımı yapar.

### 1.3 Kurallar — çiğnenmeyecek

- `agent.*` method'ları **tamamı read-only**, security capability = `Read`.
  ★ **İstisna açıkça tanımlandı (2026-08-19):** `agent.chat_send` panele yazar
  ve `sender` alanını çağıran belirler, yani salt-okunur bir token'la
  kullanıcının paneline "System" imzalı mesaj düşürülebilirdi. Kendi
  yetkisini aldı: `AgentChat` (1u << 8). Kural delinmedi, sınırı çizildi.
- Hiçbir agent method sahne durumunu değiştirmez.
- Registry statik veridir, çalışma zamanı tahsis yok.
- Mevcut dispatch chain'i bozulmaz — `agent.*` dispatch template dispatch'in
  yanına, `dispatchMethod()` başına eklenir.
- [IPC_SECURITY_PERFORMANCE.md](IPC_SECURITY_PERFORMANCE.md) veri modeli
  sınırı korunur: yalnızca isim, id ve değer geçer.

---

## 2. agent.* IPC method'ları

### 2.1 `agent.discover`

Params: `{}` — İlk çağrı. Uygulama kimliği + domain listesi + rol bilgisi.

Dönüş: `app`, `version`, `discovery_version` (breaking change sayacı),
`protocol`, `description`, `domains[]` (her biri `{name, summary,
method_count}`), `agent_methods[]`, `roles`, `total_methods`,
`registered_methods`, `coverage` (1.0 altı = discovery eksik).

### 2.2 `agent.list_methods`

Params: `{ "domain": "fluid" }` (isteğe bağlı; verilmezse tüm method'lar).
Her method: `name`, `summary`, `access`, `params_hint`.

### 2.3 `agent.describe`

Params: `{ "method": "fluid.create" }` — Tam şema: parametreler (tip,
required, default, enum), dönüş tipi, undoable, ilişkili method'lar, notlar,
tag'ler. Registry'den otomatik okunur.

### 2.4 `agent.search_capabilities`

Params: `{ "query": "make a wooden object burn" }` — Keyword/tag-based
fuzzy matching. İki kaynak:

1. **Method tag'leri:** Query tokenize edilir, her token registry'deki
   tag listesine eşleştirilir, eşleşen tag sayısına göre sıralanır.
2. **Workflow reçeteleri:** Önceden tanımlı multi-step senaryolar
   (combustion, pour_liquid, fracture, animate, terrain, groom, scatter,
   material_author, lighting, render_sequence, vb.) keyword'lere göre
   eşleştirilir.

Dönüş: `relevant_workflows[]` (adımlar + key method'lar),
`all_related_domains[]`.

### 2.5 `agent.get_examples`

Params: `{ "method": "fluid.create" }` veya `{ "workflow": "combustion" }`.
Çalışan JSON-RPC çağrı örnekleri döndürür.

### 2.6 `agent.get_state_summary`

Params: `{ "domains": ["scene","fluid","lights"] }` (isteğe bağlı).
Sahnenin anlık kompakt özeti: obje sayısı/listesi, ışıklar, fluid domain'ler,
kamera, timeline, render durumu. AGENT_VIEWPORT_MEASUREMENT_PLAN
tamamlandığında `viewport.probe` verisi de dahil edilir.

### 2.7 `agent.roles`

Params: `{}` veya `{ "role": "manager" }`. Ajan hiyerarşisini tanımlar:
her rolün açıklaması, önerilen method'ları, karar noktaları, delegasyon
zinciri. Bkz. bölüm 1.2.

---

## 3. Dosyalar

| Dosya | Tür | Satır | Açıklama |
|-------|-----|-------|----------|
| `source/src/Api/RtIpcMethodRegistry.h` | NEW | ~100 | Descriptor struct + Registry singleton |
| `source/src/Api/RtIpcMethodRegistry.cpp` | NEW | ~250 | Registry impl + keyword search engine |
| `source/src/Api/RtIpcMethodDescriptors.cpp` | NEW | ~1800 | Mevcut ~214 method'un registration'ları |
| `source/src/Api/RtIpcWorkflowRecipes.h` | NEW | ~40 | Workflow recipe struct |
| `source/src/Api/RtIpcWorkflowRecipes.cpp` | NEW | ~400 | ~15 built-in workflow reçetesi |
| `source/src/Api/RtIpcAgentDiscovery.h` | NEW | ~20 | Agent dispatch header |
| `source/src/Api/RtIpcAgentDiscovery.cpp` | NEW | ~800 | 7 agent.* method handler |
| `source/src/Api/RtIpc.cpp` | MODIFY | +6 | Agent dispatch entegrasyon |
| `source/src/Api/RtIpcSecurity.cpp` | MODIFY | +1 | `agent.*` → Read capability |
| `scripts/audit_ipc_capabilities.py` | MODIFY | +3 | agent.* heuristic |

---

## 4. Ajan bağlanma akışı

```
Ajan                              RayTrophi Studio
  │                                      │
  ├─── connect (pipe/TLS) ─────────────►│
  │                                      │
  ├─── agent.discover ─────────────────►│  "Ben kimim, ne yapabilirim?"
  │◄──── identity + domains + roles ────┤
  │                                      │
  ├─── agent.search_capabilities ──────►│  "Tahta objeyi yakmak istiyorum"
  │    ("make wooden object burn")      │
  │◄──── workflow + key methods ────────┤
  │                                      │
  ├─── agent.describe("fluid.create") ─►│  "Parametreleri ne?"
  │◄──── full schema + examples ────────┤
  │                                      │
  ├─── fluid.create({...}) ────────────►│  Eylem
  │◄──── result ────────────────────────┤
  │                                      │
  ├─── agent.get_state_summary ────────►│  "Sahne ne durumda?"
  │◄──── compact snapshot ──────────────┤
```

## 5. Ajan hiyerarşisi akışı

```
┌──────────────────────────────────────────────┐
│  MANAGER (Yönetici)                          │
│  "Bir orman yangını sahnesi oluştur"         │
│  agent.discover → yetenekleri öğren          │
│  agent.search_capabilities("forest fire")    │
│  agent.get_state_summary → mevcut durum      │
│  KARAR: terrain + ağaçlar + ateş gerekli     │
│                                              │
│  ┌── delegate ─────────────────────────────┐ │
│  │  CONTROLLER (Kontrolcü)                 │ │
│  │  "Terrain oluştur ve ağaçla"            │ │
│  │  agent.describe → method şemaları       │ │
│  │  PLAN: terrain → erosion → scatter      │ │
│  │  ┌── delegate ─────────────────────┐    │ │
│  │  │  WORKER (Çalışan)               │    │ │
│  │  │  terrain.create / scatter.fill  │    │ │
│  │  └─────────────────────────────────┘    │ │
│  │  agent.get_state_summary → doğrula      │ │
│  └─────────────────────────────────────────┘ │
│                                              │
│  ┌── delegate ─────────────────────────────┐ │
│  │  CONTROLLER: "Ateş simülasyonu kur"     │ │
│  │  gas domain + fire + MSF substance      │ │
│  │  viewport.probe → ateş görünüyor mü?    │ │
│  └─────────────────────────────────────────┘ │
│                                              │
│  SON: get_state_summary + viewport.probe     │
│  → tüm bileşenler OK, kullanıcıya bildir    │
└──────────────────────────────────────────────┘
```

## 6. Yeni method ekleme kontrol listesi

Mevcut kural: her yeni yetenek IPC + script + UI'dan geçer. Bu plan bir
adım daha ekler ama **ek dosya düzenleme gerektirmez**:

| Adım | Dosya | Otomatik mı? |
|------|-------|--------------|
| 1. `rtapi::` fonksiyon | RtApi.h + RtApi*.cpp | Elle |
| 2. Python binding | RtPython*.cpp | Elle |
| 3. IPC dispatch | RtIpc*.cpp | Elle |
| 4. Security capability | RtIpcSecurity.cpp | Elle |
| 5. Descriptor parametreleri | `python scripts/gen_ipc_descriptors.py` | ✅ dispatch'ten okunur |
| 6. Descriptor özeti | `scripts/ipc_descriptor_overlay.json` — tek satır | Elle (yoksa `documented=false`) |
| 7. Audit | `audit_ipc_capabilities.py` ikisini de kontrol eder | ✅ |

★ 5. adımı unutursan audit FAIL eder. 6. adımı unutursan hata almazsın ama
`documented_coverage` düşer — eksik **görünür**, gizlenmez. Aradaki fark
kasıtlı: şema yokluğu bir arıza, prose yokluğu bir borçtur.

## 7. Viewport measurement entegrasyonu

[AGENT_VIEWPORT_MEASUREMENT_PLAN.md](AGENT_VIEWPORT_MEASUREMENT_PLAN.md)
tamamlandığında `agent.get_state_summary` viewport verilerini otomatik içerir:

- `viewport.status` → backend, samples, rendering_active
- `viewport.probe` → mean_luminance, black_fraction, nan_fraction

Bu, yönetici ajanın render kalitesini ölçmesini sağlar:
- `mean_luminance == 0` → sahne karanlık, ışık eksik
- `black_fraction > 0.3` → büyük karanlık alanlar
- `nan_fraction > 0` → shader hatası

## 8. Sonrası

Bu bittiğinde sıradaki ana hat:

1. [AGENT_VIEWPORT_MEASUREMENT_PLAN.md](AGENT_VIEWPORT_MEASUREMENT_PLAN.md) —
   ajanın gördüğünü ölçebilmesi (bu katmanla birlikte tam döngüyü kapatır).
2. [AGENT_PIPELINE_ARCHITECTURE.md](AGENT_PIPELINE_ARCHITECTURE.md) —
   çoklu uygulama örneği, kimlik tahsisi, iş dağıtımı. Bu katmanın gerçek
   kullanımda sınanmasından sonra netleşir.

## 9. Kabul ölçütü

★ `coverage` ölçütü değişti. Registry artık dispatch'ten ÜRETİLDİĞİ için
"kayıtlı/dispatch edilen" oranı tanım gereği 1.0'dır ve bir şey ölçmez; audit
script'i ikisinin ayrışmasını zaten FAIL eder. Ölçülmeye değer olan
`documented_coverage`'dır. `agent.discover` artık hiçbir elle tutulan toplam
sayı **bildirmez**.

1. `agent.discover` → `registered_methods`, `documented_methods`,
   `documented_coverage`; uydurulmuş bir payda yok. (2026-08-19: 307 / 307 / 1.0)
2. `agent.search_capabilities("burn wood")` → combustion workflow reçetesi.
3. `agent.describe("fluid.create")` → tam parametre şeması registry'den.
4. `agent.get_state_summary` → sahne yüklüyken doğru snapshot.
5. `agent.roles` → 3 rol tanımı.
6. `audit_ipc_capabilities.py` → `agent.*` sınıflandırılmış, PASS.
7. Yeni test method ekleme → registration + dispatch → `agent.list_methods`
   otomatik listelemesi.
