# Ajan Çalışma Zamanı — Faz 4 öncesi inceleme

> **Durum:** REFERANS — 2026-08-19 kod incelemesi.
> ★ **Aynı gün ele alındı:** B1-B14'ün tamamı için kod yazıldı, **derlenmedi.**
> Her bulgunun altında `→ YAPILDI` satırı ne değiştiğini söyler. Doğrulama
> listesi: [NEXT_BUILD_CHECKS.md](NEXT_BUILD_CHECKS.md). Bu belge artık bir
> yapılacaklar listesi değil, **neyin neden böyle olduğunun kaydıdır.**
> İncelenen katmanlar: `RayTrophiAgent/` (Python runtime), `RtIpcMethodRegistry`,
> `RtIpcMethodDescriptors`, `RtIpcWorkflowRecipes`, `RtIpcAgentDiscovery`,
> `scene_ui_agent_chat`. Plan belgesi:
> [AGENT_DISCOVERY_LAYER_PLAN.md](AGENT_DISCOVERY_LAYER_PLAN.md).

---

## 0. Özet hüküm

**İnceleme anındaki durum (sabah):** iskelet doğru kurulmuştu ve planın mimari
kararları sağlamdı, ama elde olan şey bir *keşif katmanı* değil, bir **keşif
katmanı iskeletiydi**: 300 metot kayıtlı, kayıtların **299'unda tek bir
parametre tanımı yok**, 297'sinde özet yerine `"<metot adı> operation"` yazıyor.
`agent.discover` buna rağmen `coverage ≈ 1.0` bildiriyordu.

★ Bu, bu deponun en pahalı hata sınıfının yeni bir örneğiydi: **ölçü aleti
doluluk raporluyor, ölçtüğü şey boş.** `coverage` kaydın *varlığını* sayıyordu,
*içeriğini* değil.

Bu yüzden Faz 4'ün ilk maddesi ("LLM bağlan, ateş simülasyonu yap") o haliyle
koşulsaydı ajan `agent.describe("fluid.create_domain")` çağırıp `params: {}`
alacak, sistem prompt'undaki "parametre uydurma" kuralıyla çelişkiye düşecek ve
**uyduracaktı**. Sonra suç LLM'e yazılacaktı.

**Aynı gün yapılan (akşam):** descriptor tablosu artık dispatch kodundan
üretiliyor (307 metot, 307 özet, 238'i parametreli), `coverage` yerine ölçülmüş
`documented_coverage` raporlanıyor, yerel pipe dört örnekli, `agent.chat_send`
kendi yetkisini aldı, panel süreç durumunu tahmin etmiyor ve aktivite akışı
çekirdeğin audit log'undan besleniyor. **Hiçbiri derlenmedi** — doğrulama
[NEXT_BUILD_CHECKS.md](NEXT_BUILD_CHECKS.md).

---

## 1. Ne kuruldu (ölçülmüş envanter)

| Katman | Durum | Ölçüm |
|---|---|---|
| `MethodRegistry` (statik kayıt, tembel indeks, keyword arama) | Kuruldu | 153 satır, kilitli singleton |
| `RtIpcMethodDescriptors.cpp` | Kuruldu, **içi boş** | 300 kayıt / yalnızca 1 tanesinde parametre |
| Workflow reçeteleri | **Sağlam** | 28 metot adının **28'i de gerçekten dispatch ediliyor** |
| `agent.*` dispatch (9 metot) | Kuruldu | discover, describe, list_methods, search, get_examples, get_state_summary, roles, chat_send, chat_poll |
| Güvenlik sınıflandırması | Kuruldu | `agent.` → `Read`; `audit_ipc_capabilities.py` PASS (307 metot, 30 önek) |
| `.vcxproj` kaydı | Tam | 5 yeni dosyanın hepsi ekli |
| UI paneli | Kuruldu | Menü + çizim + süreç başlat/durdur |
| Python runtime | Kuruldu | provider soyutlaması, 3 meta-tool, tool döngüsü |

Dispatch ↔ registry farkı ölçüldü: **307 dispatch, 300 kayıt, fark tam olarak 7
`agent.*` metodu** — yani ajanın kendi araçları registry'de yok, `agent.describe`
onları tarif edemiyor. Bunun dışında sızıntı yok; bu iyi bir sonuç.

`agent.list_methods` tüm liste ~27 KB (~7K token) — tek çağrıda LLM'e sığar.
Gerçek özetler yazıldığında ~80 KB'ye çıkar; o noktada "önce ara, sonra tarif
et" tasarımı zorunlu hale gelir. Yani meta-tool kurgusu doğru boyutlandırılmış.

---

## 2. Planın kabul ölçütleri karşısında (plan §9)

| # | Ölçüt | İnceleme anı | Akşam |
|---|---|---|---|
| 1 | `coverage: 1.0` | ⚠ Sabit paydadan geliyordu (297), sonuç 1.01 | ✅ Ölçüt değişti: sabit payda kaldırıldı, `documented_coverage` = 307/307 |
| 2 | `search_capabilities("burn wood")` → combustion reçetesi | ✅ | ✅ (+ gerçek özetler artık aramayı da besliyor) |
| 3 | `describe(...)` → tam parametre şeması | ❌ 300 kayıttan 299'u boş | ✅ Dispatch'ten üretiliyor; 238 metot parametre taşıyor, kalanı gerçekten parametresiz |
| 4 | `get_state_summary` → doğru snapshot | ⚠ Ölçülemeyen probe 0.0 basıyordu | ✅ Ölçüm yoksa sayı yok; domain, timeline ve render de eklendi |
| 5 | `agent.roles` → 3 rol | ✅ | ✅ |
| 6 | audit PASS | ✅ ama drift geçişi yok | ✅ `--check` ile descriptor bayatlığı da FAIL ediyor |
| 7 | Yeni metot → otomatik listelenme | ✅ | ✅ (+ üretici adımı §6 kontrol listesinde) |

★ Bu tabloyu **derlemeden sonra** tekrar doğrula: sağ sütun yazılmış kodun
iddiasıdır, ölçülmüş sonucu değil.

---

## 3. Bulgular — etkiye göre sıralı

### B1 ★★★ Descriptor'lar boş; `coverage` bunu gizliyor

→ **YAPILDI.** `RtIpcMethodDescriptors.cpp` artık `scripts/gen_ipc_descriptors.py` tarafından dispatch kodundan üretiliyor; parametreler/tipler/varsayılanlar/yetki makineden, özet ve notlar `scripts/ipc_descriptor_overlay.json`'dan geliyor. 307 metot, 307 özet, 238'i parametre taşıyor. `coverage` yerine `documented_coverage` raporlanıyor ve sabit payda tamamen kaldırıldı.

299/300 kayıtta `params = nullptr, 0`; 297 kayıtta özet
`"scene.set_transform operation"`; `notes` yok, `related` boş, `tags` yalnızca
domain adı.

Üç sonucu var:

1. `agent.describe` işe yaramaz → sistem prompt'undaki anti-halüsinasyon kuralı
   uygulanamaz hale gelir.
2. `MethodRegistry::search` name+summary+tags üzerinde puanlıyor; summary metot
   adının kopyası, tags domain adı → **arama fiilen metot adı eşleşmesine
   çöküyor**. Bugünkü arama kalitesinin tamamı 15 reçeteden geliyor.
3. `coverage` metriği kaydın varlığını sayar → doluluk yalanı.

**Öneri:** `coverage`'ı ikiye ayır — `registered_coverage` (kayıt/dispatch,
gerçek dispatch sayısından hesaplanmalı, sabitten değil) ve
`documented_coverage` (en az bir parametresi veya kendi özeti olan kayıt oranı).
Bugün ikincisi 0.003 çıkar ve tam olarak bunu göstermesi gerekir.

### B2 ★★ `access` alanı 30 salt-okunur metotta `"write"` diyor

→ **YAPILDI.** `access` ve yeni `capability` alanı, audit script'inin `requiredCapabilities` aynasından türetiliyor — elle yazılmıyor, dolayısıyla tekrar ayrışamaz.

`scene.list_objects`, `scene.get_transform`, `fluid.list_domains`,
`gas.get_settings`, `timeline.get_frame`, `project.path`, `editor.get_state`…
hepsi `"write"` etiketli. Güvenlik katmanı bunları `Read` sayıyor — yani
**descriptor ile `RtIpcSecurity` birbirini yalanlıyor.** `agent.roles` çıktısında
manager rolü `can_write: false` ilan ediliyor; bu ikisini birlikte okuyan bir
ajan sahneyi *okumayı* reddeder.

Bu alan elle düzeltilmemeli: `requiredCapabilities()` ile aynı kaynaktan
türetilmeli, yoksa tekrar ayrışır.

### B3 ★★★ Yerel pipe **tek örnekli** — ajan QA altyapısının yuvasını işgal ediyor

→ **YAPILDI.** `RtIpcTransportLocal.cpp` dört pipe örneği açıyor, her birine bir worker thread. Ajan bağlıyken PowerShell/pytest de bağlanabiliyor; çoklu ajan artık transport açısından mümkün.

`RtIpcTransportLocal.cpp:138` → `CreateNamedPipeW(..., nMaxInstances = 1, ...)`,
tek iş parçacıklı `serverLoop`. Python ajanı bağlıyken:

- `scripts/ipc/RtIpc.psm1`, `scripts/test/*.py`, elle sürdüğün her şey
  **bağlanamaz**. Bu deponun ★★★ birinci kuralı IPC'nin QA altyapısı olduğunu
  söylüyor; ajan runtime'ı onu kilitliyor.
- Plan §5'teki manager → controller → worker hiyerarşisi **bugünkü transport ile
  fiziksel olarak imkânsız**: aynı anda tek bağlantı var.

Kanıt `RayTrophiAgent/agent.log`: 00:48:35 / 00:48:44 / 00:48:54'te üç ayrı ajan
süreci başlamış, üçü de "Connected" demiş. `agent_startup.log`'da bir önceki
oturum `Write failed with error 233` (ERROR_PIPE_NOT_CONNECTED) ve
`Read failed with error 109` (ERROR_BROKEN_PIPE) ile düşmüş — tek yuvanın el
değiştirmesinin imzası.

**Karar gerekiyor** (Faz 4'ten önce, çünkü hepsini etkiliyor): ya
`nMaxInstances` çoğaltılıp bağlantı başına iş parçacığı, ya ajan için ayrı bir
pipe adı, ya da ajan da sıradan bir istemci sayılıp çoklu bağlantı desteklenir.

### B4 ★★ Panelin 3 saniyelik "Offline" penceresi süreç çoğaltıyor

→ **YAPILDI.** Durum üç hâlli (Connected / Busy / Disconnected), Start düğmesi `WaitForSingleObject` ile canlılık doğrulanmadan çıkmıyor, giriş kutusu her durumda açık — mesaj kuyruğa girip ilk poll'da alınıyor. Ajan tarafında poll artık ayrı bağlantıda, yani model turu kalp atışını durdurmuyor.

`drawInputArea()` son poll'dan 3 sn geçtiyse "Offline: Agent disconnected."
yazıp **giriş kutusunu gizliyor ve "Start AI Agent" düğmesini gösteriyor** —
`agent_process_handle` dolu olsa bile. Ajan bir LLM turunu işlerken
(`handle_user_message` bloklar, 16 tool iterasyonuna kadar) poll atmaz → panel
her uzun görevde offline'a düşer → kullanıcı Start'a basar → **ikinci ajan**.
Log bunun üç kez olduğunu gösteriyor.

En az üç şey gerekiyor: (a) Start düğmesini `agent_process_handle` ile kapıla,
(b) `WaitForSingleObject(...,0)` ile sürecin gerçekten yaşadığını doğrula,
(c) ajan tarafında poll'u LLM turundan ayrı bir iş parçacığına al ya da
"çalışıyorum" kalp atışı gönder. En sağlamı (c): **durumu ajan bildirmeli,
panel tahmin etmemeli.**

### B5 ★★ İstemci okuma tamponu 512000 bayt, `ERROR_MORE_DATA` döngüsü yok

→ **YAPILDI.** `ipc_client.py` mesajı `ERROR_MORE_DATA` bitene kadar okuyor, ayrıca cevap `id`'si isteğinkiyle eşleşmezse bağlantıyı düşürüyor: kayma sessizce sürmek yerine anında hata veriyor.

`core/ipc_client.py` tek `ReadFile` yapıyor. Sunucu sınırı 16 MB
(`kMaxMessageBytes`). Mesaj modunda tampon yetmezse `ReadFile` FALSE +
`ERROR_MORE_DATA` döner; kod kırpılmış tamponu parse etmeye çalışır, artık
boruda kalır → **bundan sonraki her cevap bir öncekinin kuyruğuyla eşleşir.**
Sessiz, makul görünen desync. Bugün 27 KB'lik `list_methods` ile patlamaz;
büyük sahnede `scene.list_objects` veya `nodes.graphs` ile patlar. Düzeltme
küçük: `ERROR_MORE_DATA` iken okumaya devam eden bir döngü.

### B6 ★★ `get_state_summary` viewport'ta varsayılanı ölçüm gibi sunuyor

→ **YAPILDI.** Probe ölçülemediğinde sayı basılmıyor; yerine ne yapılması gerektiğini söyleyen bir metin dönüyor. `include_probe` parametresi eklendi. Aynı disiplin domain'lere de uygulandı: `particle_count` yalnızca `live_state` true iken sayı, değilse "not measured".

Capture kapalıyken `probeViewportFrame` `available=false` ve tüm alanlar 0.0
döner; JSON yine de `probe_mean_luminance: 0.0`, `probe_black_fraction: 0.0`
basar. LLM bunu "sahne kapkaranlık ama siyah oran 0" diye okur — çelişkili ve
yanlış. ★ Varsayılan bir ölçüm değildir: `probe_available` false ise sayısal
alanlar **hiç yazılmasın**, yerine `"probe": "unavailable — enable viewport
capture"` gitsin.

Ayrıca her `get_state_summary` tam kare taraması yapıyor ve ajan bunu döngüde
çağıracak. Bir ölçüm/saniye sınırı ya da açık bir `include_probe` parametresi
mantıklı.

### B7 ★ `agent.chat_send` `Read` yetkisiyle **yazıyor**

→ **YAPILDI.** Yeni `AgentChat` yetkisi (1u << 8). Salt-okunur token artık panele mesaj basamıyor.

Plan §1.3: "`agent.*` metotlarının tamamı read-only". `chat_send` panele mesaj
basıyor ve `sender` alanını **çağıran belirliyor** — uzaktan salt-okunur bir
token panele "System" imzalı mesaj düşürebilir. Ya `chat_send` kendi
capability'sine taşınmalı, ya da `sender` bağlantı kimliğinden türetilmeli.
Kural ya uygulanır ya değiştirilir; sessizce delinmesi en kötüsü.

### B8 ★★ Python tarafındaki "registry doğrulaması" doğrulama yapmıyor

→ **YAPILDI.** Bootstrap `agent.list_methods` ile 307 tam adı önbelleğe alıyor; doğrulama tam ad üzerinden, noktasız metotlar (`undo`, `batch`, `request_render`) dahil. Bilinmeyen ad reddedilirken `did_you_mean` listesi dönüyor.

`CapabilityRegistry.is_method_allowed()` yalnızca **domain önekine** bakıyor:
`scene.uydurma_metot` geçer, ajan hatayı ancak motordan alır (bu tolere edilir).
Asıl sorun ters yönde: noktasız metotlar — `undo`, `redo`, `batch`, `version`,
`request_render`, `reset_accumulation` — `len(parts) >= 2` şartına takılıp
**hiçbir zaman çağrılamaz**. Ajan yaptığı işi geri alamaz ve render tetikleyemez.
Üstelik reddin mesajı "registry'de yok" diyor — yani araç **yanlış teşhis**
öğretiyor.

Çözüm hazır: bootstrap'ta `agent.discover` yanında `agent.list_methods` çağır,
300 adı sete al, tam ad doğrula. ~10 satır.

### B9 ★ `agent.*` metotları kendi registry'sinde yok

→ **YAPILDI.** Üretici dispatch'i taradığı için dokuz `agent.*` metodu da kendiliğinden kayıtlı.

7 discovery metodu kayıtlı değil (`chat_send`/`chat_poll` kayıtlı). Ajan kendi
araçlarını `describe` edemiyor ve `list_methods` onları listelemiyor.

### B10 ★★ Gemini sağlayıcısı araç protokolünü düzleştiriyor

→ **YAPILDI.** Geçmiş artık `types.Content` olarak kuruluyor: asistanın `function_call` parçaları ve araç sonuçları `function_response` olarak gidiyor, sistem prompt'u `system_instruction`'a taşındı, metin çıkarımı parçalar üzerinden yapıldığı için yalnızca-araç yanıtı artık patlamıyor.

`gemini_provider.py`: bütün geçmiş `text` parçasına çevriliyor; `system` rolü
`user` yapılıyor (`system_instruction` değil); asistanın `functionCall` parçası
geçmişe hiç yazılmıyor, araç sonucu düz `user` metni olarak gidiyor. Yani
Gemini yolunda **model kendi çağrısını görmüyor**, sadece cevabını görüyor —
çok adımlı görevlerde döngüye girip aynı çağrıyı tekrarlama davranışının kaynağı
tam olarak budur. Ayrıca yanıt yalnızca fonksiyon çağrısı içerdiğinde
`response.text` erişimi güvenli değil; `try` bloğu bunu "provider error"a
çevirip turu düşürür.

Faz 4'ün 1. maddesinde "OpenAI/Gemini/Local" yan yana yazılmış; bu üçü bugün
**aynı olgunlukta değil.** İlk testi OpenAI (veya OpenAI-uyumlu local) yolu ile
yap, Gemini'yi ayrı bir iş kalemi say.

### B11 ★ Panelin "To" (hedef) açılır listesi yalan söylüyor

→ **YAPILDI.** Liste kaldırıldı; tek hedef var, çünkü tek ajan var. Çoklu ajan gerçekten geldiğinde geri gelir.

`Coordinator / Observer / Proposer / Actor / Publisher` seçilebiliyor, `target`
alanı `chat_poll` ile ajana gidiyor ve **`main.py` onu tamamen yok sayıyor** —
her mesaj tek orkestratöre düşüyor. ★ Panelin yalan söylemesi bu deponun en
pahalı hata sınıfı. Ya hedef gerçekten yönlendirilsin, ya liste çoklu ajan
gelene kadar tek girişe insin.

### B12 ★★ "Show Core Activity (IPC)" kutucuğu ölü

→ **YAPILDI, ve kaynağı değişti.** Akış artık ajanın kendi anlatımından değil, **çekirdeğin audit log'undan** besleniyor (`rtipc_audit::recent`, sequence imleçli). Ajanın yapmadığı çağrılar da görünüyor; ajan kilitlense bile panel doğru kalıyor. `chat_send` ayrıca `type` parametresi aldı (reply|activity|thought|error).

`AgentMessageType::AgentActivity` mesajını **üreten hiçbir yol yok**; her
`chat_send` `AgentReply` olarak basılıyor ve `chat_send` şemasında bir `type`
parametresi yok. Yani kutucuk hiçbir zaman hiçbir şeyi filtrelemiyor. Bu
doğrudan yarınki 3. maddenin konusu — bkz. §4.

### B13 ★ Dayanıklılık boşlukları (çoğu tek satırlık)

→ **YAPILDI.** Geçmiş `MAX_HISTORY_MESSAGES` ile kırpılıyor (öksüz tool mesajı bırakmadan), panel 600 mesajda tavanlanıp kaçını attığını yazıyor, widget kimlikleri mesaj id'sinden geliyor, yıkıcı ajan sürecini sonlandırıyor, poll hatası yalnızca bağlantı gerçekten düştüyse yeniden bağlanıyor.

- `conversation_history` sınırsız büyüyor; uzun oturum sessizce bağlam limitine
  çarpar.
- İstemcide okuma zaman aşımı yok; motor meşgulken (render) süreç asılı kalır,
  panel offline'a düşer (→ B4).
- Sunucu tarafı dispatch zaman aşımı 30 sn; ana iş parçacığı meşgulken poll bile
  30 sn bekleyebilir.
- `poll_res` içindeki **her** hata tam bağlantı yeniden kurulumu tetikliyor;
  geçici bir dispatch hatası bağlantıyı gereksiz yere düşürüyor.
- `AgentChatPanel::messages` sınırsız; `ImGuiListClipper` yok. Uzun ajan
  oturumunda panel kare süresini yiyecek. Kopyala düğmesinin ID'si mesajın
  **adresinden** üretiliyor — vektör yeniden tahsis ettiğinde ID kayar.
- Yıkıcıda `stopAgentProcess()` yok, `agent_process_handle` kapatılmıyor.
  Uygulama kapanınca Python süreci ayakta kalıyor; yeniden bağlanma 10 sn'de
  başarısız olunca kendi kendine çıkıyor (log bunu doğruluyor).

### B14 ★ Audit henüz registry drift'ini görmüyor

→ **YAPILDI.** `audit_ipc_capabilities.py` ikinci geçişte `gen_ipc_descriptors.py --check` çalıştırıyor ve tablo bayatsa FAIL ediyor.

Plan §1.1'in "ikinci geçişi" (dispatch edilen ama kayıtlı olmayan metot uyarısı)
yazılmadı. Yazılsaydı B9'u kendiliğinden yakalardı. Bir adım ötesi daha değerli:
**"kayıtlı ama parametresi belgelenmemiş"** uyarısı — B1'i CI'da görünür kılan
tek şey budur.

---

## 4. Yarınki hedefler doğru mu?

### Doğru olan yönler

- **Keşif katmanı fikri doğru ve doğru yerde.** 27 KB'lik tam liste tek çağrıya
  sığıyor; "önce ara, sonra tarif et, sonra çalıştır" üçlüsü doğru kurgu.
- **Reçeteler bu katmanın en değerli parçası** ve tek doğrulanmış parçası: 28
  metot adının 28'i gerçek. Bir LLM için "hangi 6 çağrı, hangi sırada" bilgisi
  metot listesinden kat kat değerli. Reçete sayısını artırmak, descriptor
  doldurmaktan sonraki en yüksek getirili iş.
- **Rollerin zorlayıcı olmaması** doğru karar; erişimi zaten capability bitmask'i
  belirliyor.
- Süreç yönetiminin `cmd.exe` yerine doğrudan `python -E` ile yapılması ve
  gerekçesinin koda yazılmış olması doğru.

### Sıraya itiraz

**Yarınki 1. madde (LLM bağla, ilk görev) ilk sırada olmamalı.** B1 duruyorken
ilk test şunu ölçer: "GPT-4o RayTrophi'nin parametre adlarını tahmin edebiliyor
mu?" Bu bir yetenek testi değil, bir şans testi — ve sonucu hangi katmanın bozuk
olduğunu **ayırt edemez**. Önce en az bir zincirin (öneri: combustion reçetesinin
tamamı — `fluid` + `gas` + `flow_source` + `timeline`) descriptor'ları gerçek
parametrelerle doldurulmalı. O zaman ilk test "reçeteyi izleyebiliyor mu?"
sorusunu ölçer ve cevabı tek katmana işaret eder.

**2. madde (`get_scene_context`) bir kodlama işi değil, bir bağlama işi.** C++
tarafında `agent.get_state_summary` zaten var ve çalışıyor. Yapılacak tek şey
`META_TOOLS_SCHEMA`'ya dördüncü bir giriş eklemek ve `ToolExecutor.execute`
içinde ona yönlendirmek — ~20 satır. Yeni bir C++ yeteneği yazma; **var olanı
sistem prompt'unun vaat ettiği isimle bağla** (bugün prompt "reserved for future"
diyor; motor hazır, prompt geride).

**3. madde (aktivite akışı) doğru hedef, ama yanlış uçtan çekiliyor.** Ajanın
kendi çağrılarını `chat_send` ile raporlaması, ★ üretici≠tüketici ayrışmasına
davetiye: ajan raporlamayı unuttuğu/kilitlendiği anda panel sessizce eksik
gösterir ve kimse fark etmez. Daha sağlamı: **aktiviteyi çekirdek yazsın** —
`processJsonMessage` zaten her çağrıyı audit'e kaydediyor (`rtipc_audit::record`:
metot + süre + izin + sonuç). Panel bu akıştan beslenirse "Show Core Activity"
adı gerçeğe döner: gösterdiği şey **core'un yaptığı iş** olur, ajanın anlattığı
iş değil. Ajanın kendi düşüncesini yazması ayrı bir mesaj tipi olarak kalabilir.
Her hâlükârda `agent.chat_send` bir `type` parametresi almalı
(`reply|activity|thought|error`), yoksa kutucuk ölü kalır (B12).

**4. madde (hata toleransı) doğru ve şu an test edilebilir durumda** — hata zarfı
istemciye düzgün ulaşıyor (`{"id":N,"error":"..."}` → `res.get("result", res)`
tüm zarfı LLM'e veriyor). Tek uyarı: bugün ajanın göreceği ilk hata büyük
olasılıkla **motorun değil, B8'deki sahte doğrulamanın** hatası olacak ve
"registry'de yok" diyeceği için ajan yanlış yöne düzelme yapacak. Yani bu test
B8 düzeltilmeden anlamlı değil.

### Listede eksik olan iki hedef

1. **Transport kararı (B3).** Tek örnekli pipe hem senin QA scriptlerini hem de
   planın kendi çoklu-ajan hiyerarşisini dışarıda bırakıyor. Bu bir Faz 4 maddesi
   değil, Faz 4'ün ön koşulu.
2. **Ajanın işini doğrulaması.** `viewport.probe` + `render.start` + görüntü okuma
   zinciri zaten var; ajanın "yaptım" demeden önce **ölçmesi** sistem prompt'unda
   yazıyor ama araç setinde yok. Bu bağlanmadan ajanın raporu, panelin
   yalanından farksızdır.

---

## 5. Önerilen sıra (2026-08-19'da uygulandı)

★ Aşağıdaki sıra **yazıldı**; kalan tek iş derleyip 8. maddeyi koşmak. Güncel
doğrulama listesi [NEXT_BUILD_CHECKS.md](NEXT_BUILD_CHECKS.md) dosyasında.

| # | İş | Neden bu sırada | Bozuksa ne demek |
|---|---|---|---|
| 1 | B8: Python registry'yi `agent.list_methods` ile tam ada doğrula; noktasız metotlara izin ver | 10 satır, hiçbir şeye bağımlı değil, sonraki her testin gürültüsünü kaldırır | Ajanın gördüğü hataların kaynağı belirsiz kalır |
| 2 | B2 + B9 + `coverage` payda düzeltmesi | Saf veri düzeltmesi, derleme dışında risk yok | `agent.roles` ile `access` çelişkisi ajanı sahneyi okumaktan alıkoyar |
| 3 | B1: combustion zincirinin descriptor'larını gerçek parametrelerle doldur | İlk gerçek LLM testinin ön koşulu | `describe` boş dönerse test halüsinasyon ölçer |
| 4 | B4: Start düğmesini kapıla + süreç canlılığını `WaitForSingleObject` ile doğrula | Tek başına görülebilir; çoklu süreç sonraki her ölçümü kirletir | Log'daki üç eşzamanlı ajan tekrarlanır |
| 5 | B3 transport kararı | Diğerlerinden büyük; ama 1-4 onsuz da ölçülebilir | Ajan çalışırken elle IPC ile doğrulama yapamazsın |
| 6 | `get_scene_context` → `agent.get_state_summary` bağlanması + B6 | 3. maddeden sonra anlamlı | Ajan "sahne karanlık" diye yanlış teşhis kurar |
| 7 | Aktivite akışı (audit → panel) + `chat_send` `type` parametresi | Görsel; 1-6'nın sonucunu maskeleyebilir | Kutucuk ölü kalır |
| 8 | İlk uçtan uca LLM görevi ("ateş simülasyonu kur") | Hepsinin sonucunu maskeler, en sona | — |
| 9 | B5 (`ERROR_MORE_DATA` döngüsü) — büyük sahneye geçmeden önce | Bugün belirti vermez | Sessiz cevap kayması |

### ★ En sinsi başarısızlık

**`agent.discover`'ın `coverage: 1.0` demesi.** Sayı bir sabitten geliyor,
kayıtların içi boş ve hiçbir test bundan şikâyet etmiyor. Bunu kimse bug diye
raporlamaz — çünkü rapor "her şey yolunda" diyor.

İkinci sıradaki B5'in cevap kayması: ajan doğru soruyu sorar, bir öncekinin
cevabını alır, ve ikisi de geçerli JSON'dur.
