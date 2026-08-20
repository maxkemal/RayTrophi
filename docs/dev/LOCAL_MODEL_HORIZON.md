# Yerel model ufku — 8B bir model uzun görevde nerede dağılıyor

> **Durum:** AKTİF — 2026-08-19'da ölçüldü, iki kök bulundu ve düzeltildi.
> Ölçüm koşumu tekrarlanabilir; sayılar her koşuda güncellenmeli.

## Neden bu ölçüm

Qwen3:8b'nin **araç çağırabildiği** daha önce ölçülmüştü: iki turlu bir döngüde
ara → sonucu oku → doğru metodu çağır dizisini tamamlıyor. Ama küçük modeller
tek çağrıda değil, **ufukta** dağılır. Asıl soru şuydu: 7 adımlık bir görevde ne
oluyor?

Ölçüm ajanın **gerçek** yığınını kullanır (`Orchestrator` + `ToolExecutor` +
`LocalLLMProvider` + gerçek sistem promptu), araya yalnızca bir kayıt cihazı
konur. Görev: küp ekle → taşı → ışık ekle → gökyüzünü nishita yap → malzeme
rengini kırmızı yap → kareyi yakala ve ölç → raporla.

## Koşu 1 — taban çizgisi

```
7 arac cagrisi, 71 saniye, 7 adimdan 2'si tamam
uydurulan metotlar: scene.create_object, scene.set_position  (ikisinden de kendi cikti)
```

Son cevap:

> "... Next, let's proceed with step 3 ... `{"method": "lights.add", ...}` ...
> **Would you like to continue with the next steps?**"

★★★ Sonraki çağrıyı **yapmak yerine metin olarak yazdı** ve izin istedi.
`MAX_TOOL_ITERATIONS=64` idi ve hiç yaklaşılmadı — model 7'de bıraktı. Küçük
model için **her tamamlanan adım bir tur sınırı gibi okunuyor** ve araç-çağırma
kipinden sohbet kipine düşüyor.

★ İyi haber: uydurduğu iki metottan da kendi kendine çıktı. `execute_rpc`'deki
isim doğrulaması + `did_you_mean` ağı tuttu, hiç tekrar yok.

## Koşu 2 — ve ikinci, daha kötü kök

Sistem promptuna "görevi bitir, izin isteme" kuralı ve döngüye sınırlı bir
**devam dürtmesi** eklendikten sonra sonuç *daha kötü* göründü:

```
1 arac cagrisi, 28 saniye, final answer: "Done."
```

Model bir çağrı yaptı, sonra **boş** bir yanıt döndürdü, ve döngü onu `"Done."`
diye raporladı. Ham yanıta bakınca sebep çıktı:

★★★ **Qwen3 düşüncesini ayrı bir `reasoning` alanına yazıyor; `content` boş
kalıyor.** Sağlayıcı yalnızca `content`'e bakıyordu. Yani model düşündü, hiçbir
şey söylemedi, ve **orkestratör bunu tamamlanma saydı.**

Bu, bu deponun en tanıdık hata sınıfı: **yokluk, ölçülmüş sıfır sanıldı.**
`ViewportProbeInfo.available`'ın, `live_state`'in ve `documented_coverage`'ın
çözdüğü şeyin aynısı, bu sefer ajan döngüsünde.

## Düzeltmeler

| Kök | Düzeltme |
|---|---|
| Adım sonunda izin isteyip duruyor | `Orchestrator._continuation_nudge()` — nesirde araç çağrısı ya da izin sorusu görünce sınırlı (3) itiş |
| Reasoning modelinin cevabı kayboluyor | `openai_provider` `reasoning` alanını da okur; **cevap yerine geçmez**, ayrı taşınır |
| Boş yanıt "Done." oluyordu | Boş yanıt artık ne olduğunu söyler: "model cevap üretmedi, iş yarım olabilir" |
| `task_success = True` koşulsuz atanıyordu | Metrik kaldırıldı. Görevin başarısı bu döngünün **ölçebileceği** bir şey değil; yerine `ended_without_tool_call`, `stopped_early`, `empty_answer`, `nudges` |

★★ Son satır bilerek: ölçemediğini ölçmüş gibi raporlayan bir metrik, hiç
metrik olmamasından kötüdür.

## Koşu 3 — düzeltmelerden sonra

```
9 arac cagrisi, 79 saniye, uydurulan metot: HIC
metrics: nudges=1, stopped_early=True, empty_answer=False,
         recipe_used=True, verification_performed=True
```

| | koşu 1 | koşu 2 | koşu 3 |
|---|---|---|---|
| araç çağrısı | 7 | 1 | **9** |
| uydurulan metot | 2 | 0 | **0** |
| boş yanıtı "Done." sandı | — | **evet** | hayır |
| dürtme işe yaradı | — | — | **1 kez** |

★ Davranış niteliksel olarak da düzeldi: koşu 3'te model **çağırmadan önce
`describe_capability` kullandı** ve iki parametre hatasından da şemayı okuyarak
çıktı. İstenen davranış tam olarak buydu.

★★ Ama hâlâ 7 adımdan ~3'ünde duruyor. Dürtme sınırı (3) yetmedi; kalan
davranış "adımı anlat, sonra dur". Bu, promptla tamamen çözülecek bir şey
değil — ufuk modelin kendisinde.

### ★★★ Ve koşu 3 bir MOTOR arızası ortaya çıkardı

Ajan `AgentBox`'ı ekledi (**başarı**), taşıdı (**başarı**), ama koşu sonunda
sahnede yoktu. Ajanın hatası sanıldı; değildi — silinmiş bir adı tekrar
kullanmak "yarı var" bir nesne üretiyor. Ayrıntı ve yeniden üretim:
[BUG_DELETED_NAME_REUSE_GHOST.md](BUG_DELETED_NAME_REUSE_GHOST.md).

★ Bunu yakalayan şey ajan değil, **koşunun öncesi/sonrası sahneyi
kaydetmesiydi.** Yalnızca çağrı sonuçlarına bakan bir ölçüm "9 çağrının 9'u
başarılı" derdi. Ajan testi altyapısının kendisi de bu kurala tabi:
*dönen değer bir ölçüm değildir.*

## Kalan bilinen zayıflıklar

- Model Türkçe isteği İngilizceye çevirip arıyor ve **"küp"ü "cup" diye
  çevirdi**. Kullanıcı Qwen3'ün Türkçesinin zayıf olduğunu doğruladı; yerel
  modelle İngilizce sürmek şu an doğru seçim.
- Tek turda **paralel araç çağrısı** üretiyor (3 tanesini birden uydurmuştu).
  Orkestratör hepsini sırayla işliyor, ama paralel uydurma demek tek turda üç
  yanlış demek.
