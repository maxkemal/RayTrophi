# Sıradaki build — sıralı kontrol listesi

> **Durum:** AKTİF — 2026-08-19 partisi (ajan keşif katmanı + çalışma zamanı).
> Her partide üzerine yazılır.

Sıra ölçütü: bağımsız ve hızlı görülen önce, başkasının sonucunu maskeleyen sonra.

★ Önceki parti (2026-08-18, kapsamlı sim graph'ları) **2. turda tamamen
DOĞRULANDI**; açık kalan tek şey 6. maddeydi (Nodes panelinin kapsam seçicisini
gözle görme). O madde hâlâ açık ve bu partiden bağımsız — bkz.
[SIMULATION_NODE_OBJECT_MODEL.md](SIMULATION_NODE_OBJECT_MODEL.md).

Bu partide değişenler: descriptor'lar artık **üretiliyor**, yerel pipe **çok
örnekli**, `agent.chat_send` kendi yetkisini aldı, panel süreç durumunu tahmin
etmiyor, Python çalışma zamanı iki bağlantı kullanıyor.
Ayrıntı: [AGENT_RUNTIME_PHASE4_REVIEW.md](AGENT_RUNTIME_PHASE4_REVIEW.md).

---

## 0. Derlemeden önce (build gerektirmez, 10 saniye)

```powershell
python scripts/audit_ipc_capabilities.py
```

**Görmen gereken:** `307 dispatched methods, 30 namespace prefixes` →
`descriptors up to date - 307 methods, 307 documented, 238 carry parameters` →
`OK - every dispatched method is classified, no dead prefixes, descriptors current.`

**Bozuksa ne demek:** `STALE` diyorsa descriptor tablosu dispatch ile
ayrışmış; `python scripts/gen_ipc_descriptors.py` çalıştır ve **çıkan diff'e
bak** — beklemediğin bir metot/parametre değişmişse asıl haber odur.

## 1. Derleme

Yeni dosyalar `.vcxproj`'a eklendi: `RtPythonAgent.cpp` / `.h`.
Değişen dosyalar: `RtIpcTransportLocal.cpp` (çok örnekli pipe),
`RtIpcAgentDiscovery.cpp`, `RtIpcMethodRegistry.h/.cpp`,
`RtIpcMethodDescriptors.cpp` (üretilmiş), `RtIpcWorkflowRecipes.h/.cpp`,
`RtIpcSecurity.h/.cpp` (yeni `AgentChat` yetkisi), `RtApi.h/.cpp`
(`boundContext()`), `RtPython.cpp` (`rt.agent`, `rt.viewport.render_frames`),
`scene_ui_agent_chat.*`.

## 2. İki eşzamanlı IPC bağlantısı — ★ bu partinin mimari maddesi

Uygulamayı aç, sonra **aynı anda iki** istemci bağla:

```powershell
Import-Module .\scripts\ipc\RtIpc.psm1 -Force
Invoke-RtIpc agent.discover @{}          # 1. pencere, açık bırak
Invoke-RtIpc scene.list_objects @{}      # 2. pencere
```

**Görmen gereken:** ikisi de cevap veriyor.
**Bozuksa ne demek:** pipe hâlâ tek örnekli ya da worker thread'lerden biri
kalkmamış; `startLocal` kısmi başarıyla dönmüş olabilir.
★ En sinsi hali: ikinci istemci **asılı kalır** (hata vermez) — bekleyip
zaman aşımına düşerse bu bir FAIL, "yavaş" değil.

## 3. Script katmanı — `rt.agent` var mı

Uygulama içi konsoldan ya da `script.run_file` ile:

```python
import rt; print(rt.agent.discover()["registered_methods"])
```

**Görmen gereken:** 307 (ya da o gün dispatch edilen sayı).
**Bozuksa ne demek:** `rt.agent` yoksa `registerAgentBindings` çağrısı
düşmüştür; `AttributeError` alırsan bağlama derlenmemiş demektir.

## 4. Ajan katmanı testi

```
scripts/test/rt_test_agent_layer.py     (ve x64/Release/scripts/test/ kopyası)
```

**Görmen gereken:** `_agent_layer_result.txt` içinde `ALL PASSED`.

Tek tek neyi kanıtladığı:
- `describe(fluid.create_domain)` gerçek parametreleri **dispatch'ten** biliyor,
- `scene.list_objects` artık `read` (önce `write` diyordu),
- `agent.chat_send` yetkisi `AgentChat` (Read değil),
- reçetelerin adlandırdığı her metot gerçekten dispatch ediliyor,
- **capture kapalıyken probe sayı değil "unavailable" diyor.**

★ **Bu dosyadaki en kritik tek satır:** "probe with capture off says
unavailable". FAIL ederse durum summary yine 0.0 basıyordur ve bir ajan
"sahne karanlık" diye yanlış teşhis kurar — hiçbir çökme olmadan.

★★ `NOT VERIFIED` çıkan madde **geçmiş değildir**: `render_frames` sonrası
kare yakalanamadıysa probe'un dolu hali ölçülememiş demektir.

## 5. Panel — süreç durumu üç hâlli mi

Agent Chat panelini aç (`.env` yoksa mock sağlayıcı yeter).

1. **Start AI Agent** → durum satırı: `Starting` → birkaç saniye içinde
   `Connected - agent is polling.`
2. "sahneyi anlat" yaz → `core:` satırları akmaya başlamalı (Show Core
   Activity açıkken) ve ardından yeşil bir cevap.
3. Ajan çalışırken **Start düğmesi görünmemeli** (yerinde Stop olmalı).

**Bozuksa ne demek:** Start hâlâ görünüyorsa `isAgentProcessAlive()` yanlış
cevap veriyor → ikinci bir python süreci doğurabilirsin (bu partiden önce
`agent.log`'da üç süreç yan yana görünüyordu).
★ Sinsi hali: durum `Busy` de takılı kalır — poll ulaşmıyordur ama süreç
yaşıyordur; bu, iki bağlantılı çalışma zamanının **çalışmadığı** anlamına gelir.

## 6. Aktivite akışı çekirdekten mi geliyor

Ajan çalışırken PowerShell'den elle bir çağrı yap:

```powershell
Invoke-RtIpc scene.add_primitive @{ type = 'cube'; name = 'ActivityProbe' }
```

**Görmen gereken:** panelde `core: scene.add_primitive -> success` satırı —
ajan o çağrıyı yapmamış olmasına rağmen.
**Bozuksa ne demek:** akış hâlâ ajanın kendi raporundan besleniyor; o zaman
ajan kilitlendiğinde panel sessizce eksik gösterir.

## 7. Süreç yaşam döngüsü

- Görev Yöneticisi'nden `python.exe`'yi öldür → panel `Disconnected` demeli ve
  **Start düğmesi geri gelmeli**.
- Studio'yu kapat → `python.exe` **kalmamalı** (yıkıcı `TerminateProcess`
  çağırıyor).

**Bozuksa ne demek:** kalan süreç bir sonraki Studio oturumuna bağlanır ve
kimsenin başlatmadığı bir ajan sahneyi sürer.

## 8. Gerçek model turu (anahtar gerektirir) — en sona

`RayTrophiAgent/.env`: `LLM_PROVIDER=openai`, `OPENAI_API_KEY=...`

Panelden: **"bana küçük bir ateş simülasyonu kur"**

**Görmen gereken sıra:** `search_capabilities` → combustion reçetesi →
`describe` çağrıları → `fluid.create_domain` / `gas.set_settings` /
`flow_source.create` → `get_scene_context` ile doğrulama → düz bir özet.

**Bozuksa ne demek:**
- Model parametre uyduruyorsa `describe` boş dönmüştür (0. maddeye dön).
- Aynı çağrıyı tekrar tekrar yapıyorsa sağlayıcı geçmişi kaybediyordur —
  Gemini yolunda bu tam olarak eski hataydı; OpenAI yolunda görürsen
  `raw_message` geçmişe eklenmiyordur.
- "Yaptım" deyip sahnede bir şey yoksa: `execute_rpc` sonucundaki hata
  yutulmuştur — `agent.log` içindeki `tool ...` satırlarına bak.

## 9. Büyük yanıt — sessiz kayma testi (bugün belirti vermez, sonra vurur)

```powershell
Invoke-RtIpc agent.list_methods @{}      # ~30 KB
```
Ardından hemen:
```powershell
Invoke-RtIpc timeline.get_frame @{}
```

**Görmen gereken:** ikinci çağrı **bir sayı** döndürür.
**Bozuksa ne demek:** ikinci cevap birincinin kuyruğuysa istemci
`ERROR_MORE_DATA` döngüsünü yapmıyordur. Python istemcisi artık yapıyor; bu
madde PowerShell modülünün de yaptığını doğrular.
★ Bu arızanın hiçbir hata mesajı yoktur: iki cevap da geçerli JSON'dur.

---

## Bu partinin en sinsi başarısızlığı

**4. maddedeki probe kontrolü ile 6. maddedeki aktivite akışı.** İkisi de
"çalışıyor gibi" görünerek geçebilir: probe sayı döndürürse rapor dolu görünür
(ama ölçüm yoktur), aktivite akışı ajanın kendi anlatımından beslenirse panel
dolu görünür (ama çekirdeği göstermez). İkisinde de ekranda gördüğün şey
doğrulanmış değil, **makul** olur.
