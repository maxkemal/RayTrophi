# State machine hiç geçiş yapmıyor, node seçilince animasyon duruyor

> **Durum:** REFERANS — 2026-09-13, üç kök de **CANLI DOĞRULANDI** (aşağıdaki ölçüm).

## Belirti (kullanıcıdan)

İki animasyon klibi var. AnimGraph'ta bir State Machine node'u var.

- Hangi klip **Push** edilirse o oynuyor. (Yani klip yolu sağlam.)
- State machine **seçilince** "garip şekilde" animasyon **duruyor**.
- State machine iki klip arasında **hiç geçiş yapmıyor**.

"Seçilince" kelimesi teşhisin tamamını taşıyor: bir node'u seçmek hiçbir şeyi
değiştirmemeli. Değiştiriyorsa, seçim başına çalışan bir kod yolu vardır.

## Kök neden 1 — çalışma-zamanı durumu ASSET'ten geri yazılıyordu

`drawNodePropertiesPanel()` seçili node'u her karede canlı runtime node'una
aynalıyor:

```cpp
assetNode->onSave(nodeJson);
runtimeNode->onLoad(nodeJson);   // syncRuntimeNodeFromAsset()
```

`StateMachineNode::onSave` `currentState`'i de yazıyordu. Asset grafiği hiç
**değerlendirilmiyor**, yani onun `currentStateName`'i sabit. Sonuç:

1. Kare N: geçiş tamamlanır, runtime `currentStateName = "Move"`.
2. Kare N+1: ayna `"Idle"`'ı geri yazar, `isTransitioning = false`.
3. Koşul hâlâ sağlanıyor → aynı geçiş yeniden tetiklenir.

Makine **kalıcı olarak geçiş hâlinde** kalır. Panelde "hiç geçiş yapmıyor"
diye görünür, ekranda "donmuş" diye.

**Düzeltme:** `currentState` artık serialize **edilmiyor**. Canlı durum runtime
node'a aittir; `computePose()` geçersiz/boş bir state'i zaten default state'e
düşürüyor — bu proje açılışında da doğru davranış. Inspector canlı durumu
runtime'dan **okuyor** (tıpkı klip `currentTime`'ı gibi), asla geri yazmıyor.

## Kök neden 2 — aynanın pin kimliklerini SİLMESİ

`StateMachineNode::onLoad` state başına bir input pin kuruyor:

```cpp
inputs.clear();
...
addInput(state.name + " Pose", ...);   // pin.id == 0
```

`addInput` id atamaz; id'yi graph verir. `AnimationNodeGraph::loadFromJson`
bunu **biliyor** ve `onLoad` sonrası kaydedilmiş pin id'lerini yeniden
uyguluyor — yorumu bile var. Ama ayna ikinci çağrı yeriydi ve o düzeltmeyi
**devralmadı.**

Sonuç: runtime graph'ın link'leri artık var olmayan pin id'lerine bakıyor,
`getInputPose()` hiçbir link bulamıyor, her state **boş poz** döndürüyor.
Hata yok, log yok — karakter duruyor. Üstelik bir sonraki **Push**'a kadar
kalıcı: pin'ler seçim kalksa da geri gelmiyor.

**Düzeltme:** `syncRuntimeNodeFromAsset` artık `onLoad` sonrası pin id'lerini
asset node'dan aynalıyor (asset ile runtime birbirinin klonu, id uzayı ortak).

> **Ders:** bir düzeltme bir çağrı yerinde doğrulandıysa, **ikinci çağrı yeri
> aranmadıkça** düzeltilmiş sayılmaz. Bu depoda aynı sınıf daha önce de vardı
> (`bugfix_removed_branch_claim_verified_at_one_call_site`).

## Kök neden 3 — blend `wasUpdated`'ı düşürüyordu

`Renderer::updateAnimationWithGraph` kemik matrislerini **yalnızca**
`pose.wasUpdated` iken yazıyor. `BlendNode::blendPoses` sıfırdan bir `PoseData`
kuruyordu ve `wasUpdated` varsayılanı `false`.

State machine geçiş sırasında **tam olarak** `blendPoses`'in çıktısını
döndürüyor. Yani geçiş boyunca poz renderer'a hiç ulaşmıyordu: karakter
`blend_time` süresince son pozunda kalıyordu.

Tek başına bu bile "geçişte donuyor" üretir; 1 ve 2 ile birlikte **kalıcı**
donma üretir.

**Düzeltme:** `blendPoses`, `AdditiveBlendNode` ve `LayeredBlendNode` artık
`wasUpdated`'ı girdilerinden OR'luyor.

## Yol boyunca bulunan iki ölü/yanlış yol (söküldü)

- **Blend node'ları kemik offset'ini uyguluyordu.** `localMat * offsetMatrix`
  — ama poz zincirinin o aşaması **lokal uzayda**; `FinalPoseNode` hiyerarşiyi
  çözüp offset'i zaten uyguluyor. İki kez uygulanacaktı. Fiilen hiç tetiklenmedi
  çünkü aramanın anahtarı olan `boneNames` **hiçbir zaman doldurulmuyordu**.
- **`PoseData::boneNames` hiçbir üretici tarafından yazılmıyordu.** Tüketen üç
  yer vardı. En pahalısı `LayeredBlendNode`: `affectedBones` maskesi boş ada
  bakıyor, hiçbir kemik eşleşmiyor, katman **sessizce hiçbir şey yapmıyordu**.
  `AnimClipNode` artık adları yazıyor.

## Ölçü aleti (bu hata sınıfı neden bu kadar yaşadı)

State machine'in canlı durumu **yalnızca panelde** görünüyordu. Script'ten
bakan bir ajan için şu üç durum ayırt edilemezdi:

- default state'e çakılmış makine,
- hiç geçiş yazılmamış makine,
- state girişleri kopmuş makine.

Üçü de "karakter tek klip oynuyor ya da duruyor" diye görünür.

Eklenenler:

| Metot | Ne verir |
|---|---|
| `anim.state_machines` | canlı state, hedef state, geçiş ilerlemesi, state başına `pose_connected`, geçiş koşulları, son olay kaydı |
| `anim.force_state` | makineyi bir state'e atlatır (paneldeki state tıklamasının script ikizi) |

`pose_connected` bu tablodaki en değerli alan: **boş poz üreten bir state
hiçbir katmanda hata değildir.**

Probe: `scripts/ipc/Probe-AnimStateMachine.ps1` (ve `x64/Release` kopyası).

## Canlı doğrulama (derleme sonrası, aynı gün)

`anim.state_machines` karakter `1` üzerinde:

```
current_state = "d"   transitioning = false   target_state = ""
states: w (default, pose_connected=TRUE)  d (current, pose_connected=TRUE)
```

Ve altı ardışık örnekte klip zamanı ilerliyor (`normalized_time`
0,873 → 0,894 → 0,914 → 0,934 → 0,954 → 0,975, `playing=true`), state
sabit `d`. Yani:

- **Kök 2 kapandı** — `pose_connected` ikisinde de true, pin kimlikleri ayakta.
- **Kök 1 kapandı** — state `d`'de DURUYOR; eski davranışta her kare geri
  yazılıp aynı geçiş yeniden tetikleniyordu.
- **Kök 3 kapandı** — geçiş boyunca ve sonrasında poz renderer'a ulaşıyor.

## ★ Düzeltmeden SONRA gelen bildirim ve neden hata değildi

Kullanıcı "geçiş oluyor ama tek seferde, sonra sürekli 2. klip oynuyor" dedi.
Ölçüm bunun **yazılmış graph** olduğunu tek çağrıda gösterdi:

```
transitions: [ w -> d   condition=none  has_exit_time=FALSE  blend_time=5.0 ]
```

`d -> w` **yok**. Makine doğru davranıyordu. İki yan bulgu da aynı satırda
görünür oldu: `has_exit_time=false` + `condition=none` geçişi **ilk karede**
tetikliyor (yani `w` klibi hiç oynamıyor), ve 5 saniyelik `blend_time`
kullanıcının "tek seferlik geçiş" diye tarif ettiği cross-fade'in ta kendisi.

> **Bu, ölçü aletinin karşılığını ödediği yer.** Aynı belirti ("2. klip
> takılı kaldı") hem düzeltilen üç kökün geri gelmesi, hem de eksik bir geçiş
> anlamına gelebilirdi. Panel ikisini de aynı gösterir; `anim.state_machines`
> ayırdı.

## Sonradan bulunan dördüncü kök (küçük, aynı fonksiyonda)

`Transition::evaluate()` bir **Trigger parametresini TÜKETİYOR**, ve eski kod
koşulu exit time'dan ÖNCE soruyordu. Exit time sağlanmamış bir karede trigger
sessizce yenip atılıyordu.

Belirtisi "trigger çalışmıyor" değil, **"trigger bazen çalışıyor"** — klibin
tetiklendiği anda nerede olduğuna bağlı. Düzeltildi: exit time önce bakılıyor,
bekleyen trigger gerçekten işleyebilecek bir kareye kadar yaşıyor.

Bu kullanıcının grafiğinde tetiklenmiyordu (koşul `none`), yani **canlı
doğrulanmadı.**

## Açık kalan: state machine yazarlığı HÂLÂ panel-only

`anim.state_machines` okur, `anim.force_state` sürer — ama bir state ya da
geçiş **eklemek** yalnızca panelden yapılabiliyor. Sonucu somut: bu hatanın
regresyon testi yazılamıyor, çünkü script sıfırdan bir state machine kuramıyor;
probe böyle bir sahne yoksa `exit 2` ile SKIP ediyor.

Zor kısmı tasarım kararı: yazarlık **asset** grafiğine gitmeli (runtime klonu
bir sonraki Push'ta üzerine yazılır ve projeye kaydedilmez), asset grafiği ise
`g_animGraphUI.graphs` içinde UI tarafında yaşıyor.
