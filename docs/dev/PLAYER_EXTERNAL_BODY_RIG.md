# Boyama Odası C2 — dış beden pozu → rig → mevcut domain vekilleri

> **Durum: KAYNAK HAZIR / KULLANICI BUILD VE CANLI KABUL BEKLİYOR — 2026-11-15.**
> C/B ilk paketinin canlı kabulü ayrı ve geçerlidir. Bu C2 paketinde C++ build,
> uygulama başlatma ve canlı test yapılmadı; `/Zs`, kaynak/descriptor/proje denetimleri
> kullanıldı. Native assert testi kaynak olarak verildi, çalıştırılmadı.

## Kullanıcı yönü ve tek fizik yolu

Kullanıcı mevcut rig/domain etkileşim sisteminin yeniden kullanılmasını istedi.
Bu yüzden ayrı Jolt dünyası, ikinci iskelet/proxy sistemi ve sahte mesh oluşturulmadı.
Jolt/partikül çekirdeği ve shader'lar değiştirilmedi.

```text
input.set_body_poses (mevcut, kamera dış süreçte)
  → ExternalBodyRigMath: mevcut matchIKPose / solveIKPose / constrainJointPose
  → mevcut rig joint globals + canonical TriangleMesh/DNA skinning
  → mevcut KinematicColliderRegistry (authored proxy setleri)
  → mevcut domain snapshot / voxel damgalama / fluid-gas-granular tüketicileri
```

Kinematik sistemin önceki kabulü ve sınırları:
[KINEMATIC_COLLIDER_SOURCES.md](KINEMATIC_COLLIDER_SOURCES.md).
Örnekleme/voxel damgalama mevcut sistemde CPU'dadır; solid/hız alanlarını mevcut
CPU/Vulkan domain yolları tüketir. C2, ağır çözücüyü CPU'ya geçiren yeni bir fallback
eklemez; mevcut domainin hangi backend'de çalıştığı kendi tanılama alanlarından
ayrıca doğrulanmalıdır. Realtime/GPU balon partiküllerinin bu proxy'leri tüketmesi
mevcut K4 açığıdır; bu paket onu tamamlamaz. Jolt rigid cisim teması da bu pakette yoktur.

## Ortak script / IPC yüzeyi

Üç işlem de aynı `rtapi` ve Input çekirdeğini kullanır:

| IPC | Python |
|---|---|
| `anim.set_external_pose(character, person_id, config)` | `rt.anim.set_external_pose(character, person_id, config)` |
| `anim.external_poses()` | `rt.anim.external_poses()` |
| `anim.clear_external_pose(character)` | `rt.anim.clear_external_pose(character)` |

Status Read; set/clear SceneWrite. Yeni Player izinleri mevcut rig listeleme ve
`physics.collider.proxy_set.*` / `physics.collider.proxy.set/delete` hizmetlerini
de açar. `perf.list` salt okunur CPU ölçümlerini Player'dan alabilir. Python
birimi yalnız Studio'da, üç yeni ortak C++ birimi iki hedefte kayıtlıdır.

Ön koşullar: sahnede mevcut, metre ölçekli rig; geçerli node hierarchy, IK kontrolleri
ve joint-limit kuralları. Skinned mesh varsa geçerli bone slot/offset eşleri gerekir.
Mevcut IK'nin rigid-pose/chain kısıtları korunur; rastgele ölçekli FBX'i otomatik
normalize eden bir retarget sistemi değildir. Rig edit/pose modu açıkken bind reddedilir.
En çok 10 rig bağlanır; kişi ID'si negatif olmayan int32. Bağlantı kişi gelmeden kurulabilir.

Örnek (kemik ve kontrol adları **kendi rig'inin gerçek adlarıyla değiştirilmeli**):

```json
{
  "character": "Avatar",
  "person_id": 1,
  "config": {
    "anchor_bone": "Avatar_Pelvis",
    "anchor_joint": "pelvis",
    "controls": {
      "LeftHandIK": {"joint": "l_wrist", "pole": "l_elbow"},
      "RightHandIK": {"joint": "r_wrist", "pole": "r_elbow"},
      "LeftFootIK": {"joint": "l_ankle", "pole": "l_knee"},
      "RightFootIK": {"joint": "r_ankle", "pole": "r_knee"}
    },
    "min_confidence": 0.5,
    "blend": 1.0,
    "teleport_metres": 1.0
  }
}
```

Anchor bone zorunlu; anchor joint varsayılan pelvis. Controls, 1..16 mevcut IK
kontrolünü kanonik 15 eklemden birine bağlar; pole isteğe bağlıdır. Min confidence
0,1..1 (0,5), blend 0..1 (1), teleport mesafesi 0,1..10 m (1). Tür/alan/ad/eşik
hataları eski binding'i değiştirmeden reddedilir; bool kişi ID'si/sayısal ayar değildir.

Anchor rig kökünü dünya uzayında taşır; uzuvlar mevcut IK ile çözülür, kemik uzunlukları
ve joint limitler korunur. Root translation kilidi, takip edilen oda çevirisini iptal
etmez. Gövde yaw, el/ayak yönelimi, otomatik kemik adı eşleme ve vücut oranı ölçekleme
çıkarılmaz. Aynı rig'in önceki klibiyle animasyon blend'i değildir; blend rest→takip içindir.

## Vekiller, kayıp ve yaşam döngüsü

Vekil setlerini bir kez mevcut editör/IPC/script hizmetiyle hazırlayın. Örneğin
`physics.collider.proxy_set.create(name, target_character)` ve `auto_fit(set_id, ...)`;
Python karşılığı `rt.collider.proxy_set.create/auto_fit`. **Her kare auto-fit yapmayın.**
Authored set aynı karaktere bağlı kalır; C2 her kare ikinci bir set üretmez ve registry'yi
yeniden yazmaz. Var olan örnekleyici aynı çözülen rig globals'ı kullanır.

Mevcut `input` 0,5 s kayıp toleransı korunur. Sonrasında görünür rig bind pozuna döner;
o karakterin domain proxy'leri unresolved olur, eski poz hayalet collider olarak kalmaz.
Düşük güvenli hedef/pole kontrolünün kökü ve alt kemiklerinin proxy'leri maskelenir;
diğer güvenilir uzuvlar kullanılabilir. Tekrarlanan kişi değil, bağlı sabit ID izlenir;
kameranın yeni ID atadığı kişi için açık rebind gerekir.

Yeni bind, clear, tracking kaybı/geri dönüş, rig placement değişimi ve eşikten büyük
anchor sıçramasında ilgili proxy setlerinin **mevcut** hareket geçmişi sıfırlanır.
Böylece ilk örnekten sahte hız/itki çıkarılmaz. Salt okunur proxy sample zaten registry
kopyasını kullanır; teşhis çağrısı çözücünün hız geçmişini tüketmez.

Rig revision değişirse `rig_changed_rebind_required`; edit/pose sonradan açılırsa
`rig_edit_conflict` görülür, domain proxy'leri çözülmez. Clear bind geometrisini geri
yazar ve dış bağlantıyı kaldırır. Authored setler silinmez; clear sonrasında normal rig
örneklemesine dönerler. İstenmiyorsa mevcut seti ayrıca disable/delete edin.
Yeni/açılan projede transient dış bağlantılar silinir; binding projeye kaydedilmez.

Input alımı, ortak canlı saat ve Renderer pose hook'u aynı cache'i kullanır. Player
mevcut flat deform/refit yoluyla görünür mesh'i yeniler. Per-face Triangle koleksiyonları
geometri kaynağı değildir. Normal rig/klip/rest kullanıcıları için eski yol korunur.

## Maliyet: şimdi sayaç, doğruluk sonrası karşılaştırma

`anim.external_poses.bindings[]` alanları:

- `state`, `collider_available/error`, `unreliable_bones`, `last_error`, `limit_hits`;
- `solves`, `skin_updates`, `motion_resets`;
- `solve_ms`, `solve_max_ms`, `skin_ms`: **CPU host** süreleri;
- `proxy_sets`, `proxies`: mevcut authored set/sayıları, resolved hücre sayısı değildir.

Değişmeyen paket/placement/kişi varlığında IK tekrar çözülmez. Düşük güven maskesi
paketle cache'lenir; her proxy kemiği için bütün kişi pozları kopyalanmaz. Meshless,
görünmez rig per-vertex skin masrafı taşımaz; yoğun görünür avatarın CPU flat skin
maliyeti ayrıca ölçülmelidir. Bu sayaç/status çağrıları GPU readback eklemez.

Domain için mevcut `physics.collider.proxy_set.solver_stamps` gerçek step/hücre/hız
kayıtlarını verir. Mevcut `perf.list` içindeki `sim.fluid.voxelize_colliders` ve
`sim.fluid.solid_face_weights` CPU bölümleri de alınabilir. Sayaç yoksa süre **ölçülmedi**
sayılır; sıfır maliyet denmez. Native domain, render, GPU süresi ve kamera→ekran
gecikmesi solve_ms değildir. C2 için canlı maliyet henüz ölçülmedi.

Karşılaştırma sırası: aynı domain/çözünürlük/aktif hücreler ve içerikle 0→1→4→10 kişi,
meshless rig ve görünür avatar ayrı; CPU solve/skin, damgalama hücresi/süresi, backend
durumu ve ekran kare süresi birlikte. Önce doğru hedef/ayağın domaini etkilediği,
kayıpta proxy'nin kalktığı ve domain backend'inin değişmediği doğrulanır.

## Build ve kabul

Mevcut ortak struct düzeni, shader binding/push ve Jolt düzeni değişmedi. Yeni kaynaklar
nedeniyle Studio/Player projelerini reload edip artımlı build alın. Yeni API modülleri
ve proxy sampler kaynakları yeniden derlenir; önceki TAM build gereksinimlerini kaldırmaz.
Bu paket için Codex C++ build/uygulama başlatma yapmadı.

Uygulamasız:

```powershell
python scripts/test/check_external_body_rig_contracts.py
python scripts/audit_ipc_capabilities.py
python scripts/audit_player_includes.py --check
python scripts/gen_player_vcxproj.py --check
```

`tests/external_body_rig_math_test.cpp`: IK hedefi, kemik uzunluğu, çevrilmiş placement,
root kilidiyle takip, düşük güven/bozuk mapping ve registry teşhis kopyasının hız
geçmişini tüketmemesi için native assert kaynağıdır; `/Zs` kontrolü çalıştırma değildir.

Kullanıcı build'i sonrası mevcut rig + authored proxy set + canlı domain test sahnesinde,
config dosyasına örnekteki **config nesnesini** gerçek adlarla yazıp ayrı terminalden:

```powershell
python tools/perception_bridge/external_rig_probe.py --character Avatar --config tools/perception_bridge/out/avatar-mapping.json --proxy-set 1 --seconds 6 --require-stamps
```

Studio için `--pipe \\.\pipe\RayTrophiStudio` ekleyin. Aktif kamera/başka kişi pozu veya
o karakterde mevcut dış binding varken probe başlamaz. Rig oluşturmaz, sahne açmaz;
domain mevcut canlı modda olmalı. Probe sahte ID 417 gönderir; hareket, bozuk mapping'de
atomiklik, kayıpta unresolved ve authored setin korunmasını denetler. Domain sayacı
ilerlemeden eski bir stamp'i PASS saymaz. `--require-stamps` yoksa domain kabulü ayrı
NOT_RUN kalabilir. Temizlikte pozlar ve kendi binding'i kaldırılır, authored set korunur.
Domain simülasyonunun ilerlemesi geri sarılmaz; test sahnesinde çalıştırın.

Rapor `tools/perception_bridge/out/external-rig.json` (ignored): CPU solve/skin snapshot'ları,
pose RPC, gerçek solver stamp'leri ve varsa profiler count/total farkından CPU ortalaması.
Profiler max_ms birikimlidir, yalnız test aralığının maksimumu olarak yorumlanmaz.
Bu teslimde C2 canlı kabulü veya sayısal maliyet başarısı iddia edilmez.
