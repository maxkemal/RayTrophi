# Boyama Odası — poz, kural ve oturum çekirdeği

> **Sonraki paket C2 kaynak hazır:** dış rig/IK ve mevcut domain proxy yolu,
> ayrı Jolt eklenmeden bağlandı; henüz kullanıcı build/canlı kabulü yok.
> Sözleşme ve maliyet sayaçları: [PLAYER_EXTERNAL_BODY_RIG.md](PLAYER_EXTERNAL_BODY_RIG.md).

> **Durum: KULLANICI STUDIO/PLAYER BUILD + İLK PAKET CANLI IPC PASS — 2026-11-15.**
> Player'da 4/10 çocukla tam oturum, 28 hedefli IPC kontrolü; Studio'da 21 IPC
> kontrolü ve 9 Python binding smoke kontrolü geçti. Native bağımsız assert testi
> çalıştırılmadı; test kaynağı ve yeni C++ birimleri önceden `/Zs` ile denetlendi.
> Rig/IK sürüşü, fiziksel kinematik gövdeler, Studio kural paneli ve özel balon/final
> görselleri bu ilk pakette tamamlanmış değildir. A ve D işleri sırada bekliyor.

## Kullanıcı build'i sonrası canlı kabul

Kanıt: [PLAYER_BODY_GAME_ACCEPTANCE_2026_11_15.json](PLAYER_BODY_GAME_ACCEPTANCE_2026_11_15.json).
Her koşu 36 s, 1080 poz; gerçek `session_plan.json`, ayrı süreç ve Player named-pipe.

| Çocuk | Gönderim | Boya en yüksek | Başarılı eylem | Eylem/olay kaybı |
|---|---:|---:|---:|---|
| 4 | 29,93 Hz | %16,4324 | 29 | 0 / 0 |
| 10 | 29,93 Hz | %32,4847 | 71 | 0 / 0 |

İki koşuda da attract→playing→finale→resetting→attract tam sırası, reach/arms-up/stomp,
bölge giriş/çıkış, boya artışı ve pozitif ekran FPS'i PASS. 4 çocuk: 24 ayak boya
eylemi + 4 balon + 1 reset; 10 çocuk: 60 ayak boya + 10 balon + 1 reset.
Arms-up olayları kişi başına bir kez üretildi. Sonrasında pozlar, boya ve demo
takvimi temizlendi; Player test odasında attract, sıfır kişi/kapsama; ekran ayarları
eski değerine döndü. Studio sahnesi boş, oturum yapılandırılmamış bırakıldı.
Device-loss, düşen boya kaynağı ve host boya indirmesi/baytı 0.

Hedefli IPC kontrolleri: bozuk poz/kural/olay reddi ve atomiklik, boolean ID/zaman,
taşan ID/koordinat, zaman tekrarı, clear ile yeni zaman başlangıcı, jest kenarı,
gönderen zaman sıçramasında kimliğin korunması, yerel 0,5 s zaman aşımı, ayrı
hedeflerde bağımsız iki kişilik grup ve cooldown, gerçek PaintFloor flat AABB
temasından grup eylemi, kısa timer döngüsü ve kapsamayla erken final PASS.
Kapsama finali %0,5580 oranında yaklaşık 0,528 oyun saniyesinde görüldü; süre 30 s idi.
Studio Python binding smoke harici `script.run_file` ile tek sonlu dosya çağrısıdır:
içinde IPC, bekleme veya kare döngüsü yoktur. Canlı akış testleri gömülü Python'da yapılmadı.

**Performans kapısı kapanmadı:** poz RPC p95 4/10 çocuk için 29,12 / 18,00 ms,
maksimum 182,21 / 189,72 ms. Takvim gecikmesi p95 549,95 / 45,37 ms, maksimum
818,78 / 208,46 ms. Ortalama 30 Hz'e yakınlık düşük jitter garantisi değildir.
Ekran FPS medyanı 56,32 / 56,72; 4 çocuk başlangıç örneğinde FPS 0 görüldü.
Kamera/sensör→ekran gecikmesi ve görsel içerik doğruluğu bu kabulün kapsamı değildir.

Tekrar betiği: `tools/perception_bridge/ipc_contract_probe.py --output <rapor.json>`;
Studio için `--pipe \\.\pipe\RayTrophiStudio --body-only`. Tam Player probu hazır
balon fixture'ı/atlas ve attract oturumu gerektirir. Ham koşular ve kapanış ölçümleri
`tools/perception_bridge/out/live-build-2026-11-15/` altında ignored olarak tutulur.
Bu kabul turunda ürün C++/shader değişmedi; yeni build gerekmiyor.

## Ortak çalışma yolu

`Input/BodyPoseRuntime` → `Game::postEvent` → `Game::Session` / `Game::Rules`
→ `Game::tickRuntime` eylem adaptörü → mevcut boya/balon/ses servisleri.
IPC ve Python aynı `rtapi` girişlerini kullanır. İş mantığı binding'lerde kopyalanmaz.
Player ve Studio'nun ortak `RuntimeSimulationClock` canlı tick'ine bağlandı;
Studio'da live mod gerekir. Bake/final render tick'i oyun akışını çalıştırmaz.

Yeni kaynaklar iki projeye kayıtlıdır; Python birimleri yalnız Studio'dadır.
Yeni namespace'ler `input.` / `game.`; `input.body_poses` ve `game.rules` Read,
diğer yeni endpoint'ler SceneWrite ister. `game.session` komut endpoint'i olduğundan
status işlemi de SceneWrite ister. Bu, ileride D'nin dar operatör rolü tasarlanırken
metot/işlem düzeyinde ayrıca ele alınmalıdır. Shader/binding/push/struct ABI değişmedi.

## Poz sözleşmesi

IPC `input.set_body_poses(t, persons)` = Python `rt.input.set_body_poses(t, persons)`.
`input.body_poses` / `rt.input.body_poses()` okur;
`input.clear_body_poses` / `rt.input.clear_body_poses()` temizler.

```json
{"t": 0.0, "persons": [{"id": 1, "confidence": 1.0, "joints": {
  "head": [0, 1.2, 1.5, 1]
}}]}
```

Yukarıdaki eklem bölümü **kısaltılmıştır**; gerçek pakette 15 eklemin tamamı gerekir:
head/neck/pelvis ve l_/r_ shoulder/elbow/wrist/hip/knee/ankle.
Dünya uzayı, metre, Y yukarı. Sonlu koordinatlar ±10000 m; güven [0,1].
ID benzersiz ve negatif olmayan int32; paket ve 0,5 s kayıp toleransı dahil tutulan
toplam kişi sınırı 10. Gönderen zamanı sonlu, negatif olmayan ve kesin artan olmalıdır.
Yeni gönderen saati için önce clear gerekir; bozuk pakette pozlar/zaman ilerlemez.

Kişi kaybı gönderen zamanıyla değil **alıcı steady clock** ile hesaplanır. İskelet
mesajda yoksa son poz 0,5 s kalır, sonra düşer. Çok sayıda yeni ID eski kimlikler
henüz düşmeden gelirse `body_capacity_exceeded` döner. Clear kişileri ve gönderen
zamanını temizler, süreç sayaçlarını ve önceden post edilmiş oyun olaylarını silmez.
Yeni proje/açılış yolu oyun oturumunu, pozları ve bekleyen olayları temizler.

Jestler güven >=0,5 koşuluyla çekirdekte üretilir:

| Olay | Koşul |
|---|---|
| `gesture.arms_up` | İki bilek başın 10 cm üstüne çıkar; 2 cm histerezis, yalnız yükselen kenar |
| `gesture.reach` | En az bir bilek pelvisten oda -Z yönünde 45 cm uzanır; 30 cm histerezis |
| `gesture.stomp` | Ayak aşağı hızı >=1,2 m/s ve Y<=8 cm; kişi başına 0,3 s tekrar aralığı |

Stomp hızı 1..200 ms gönderen örnek aralığında hesaplanır; büyük boşlukta hız
uydurulmaz. Bunlar ilk oda eşikleridir; vücut yönüne göre otomatik retarget değildir.
Okuma yanıtı joints yanında bilek/ayak/pelvis analitik kürelerini de verir.

## Kurallar ve oturum

IPC `game.session(operation="status", config={})` = Python
`rt.game.session(operation="status", config={})`. İşlemler:
configure/start/stop/reset/status. Config yalnız configure için alınır.
`game.rules` / `rt.game.rules()` normalize kuralları ve fires/dropped sayaçlarını okur.
`game.post_event(event=...)` / `rt.game.post_event(event)` aynı sınırlı kuyruğa yazar.

```json
{
  "operation": "configure",
  "config": {
    "duration_seconds": 26,
    "finale_seconds": 3,
    "reset_seconds": 2,
    "coverage_goal": 1.0,
    "auto_start": false,
    "zones": [{"name": "play_area", "min": [-3,-0.2,-3], "max": [3,3,3]}],
    "contact_targets": [],
    "rules": [{
      "id": "foot_paint", "event": "gesture.stomp", "state": "playing",
      "cooldown_seconds": 0.05,
      "action": {"type": "paint.dab", "radius": 0.3, "amount": 1.0}
    }]
  }
}
```

Configure atomiktir ve oturumu attract'a alır; çalışma durumu projeye kaydedilmez.
Süreler 0,25..3600 s (varsayılan 180/8/3), kapsama hedefi 0,001..1 (0,65).
Kurallar <=128, bölgeler <=32, adlandırılmış flat mesh temas hedefleri <=64.
Yanlış alan/tür/eşik ve bilinmeyen eylem reddedilir. Bölge AABB min<max olmalı.

Durumlar **attract → playing → finale → resetting → attract**. Start yalnız
yapılandırılmış attract'ta geçerlidir. Auto_start açıkken kişi varsa oyun başlar.
Süre veya kapsama hedefi finali başlatır. Reset girişinde sonraki canlı tick'te
ortak boya sıfırlama eylemi uygulanır. Stop/reset resetting'e geçer.
Final bu pakette bir **durumdur**; balık/çiçek/konfeti görselleri henüz bağlanmadı.
Uzun karede oyun dt'si en çok 0,25 s ilerler; değişken canlı dt kullanılır.

Kural: benzersiz id, kanonik event ve action zorunlu; state/target isteğe bağlıdır.
`min_value` varsayılan 0, `cooldown_seconds` 0, `min_persons` 1,
`group_window_seconds` 0,5. Min persons 1..10; süreler/değer eşiği 0..3600.
Cooldown tek kişide hedef+kişi, grup kuralında hedef başınadır. Grup yalnız aynı
hedefe pencere içinde gelen **farklı kişi** olaylarını sayar; aynı kişi/joint sayılmaz.
Zone enter/exit pelvis üzerinden hesaplanır; kayıp kimlik çıkışı 0,5 s toleransı izler.
`contact_targets` yalnız adı verilen canonical `TriangleMesh` dünya AABB'leriyle
eklem kürelerini 10 Hz karşılaştırır. Temas sürdükçe olay gelir; cooldown kullanın.
Bu yol gerçek üçgen yüzey/OBB teması, swept collision veya Jolt rigid body değildir.
Per-face Triangle facade geometri kaynağı olarak kullanılmaz.

Eylemler:

| type | Alanlar / hizmet |
|---|---|
| `paint.dab` | radius 0,001..5 m (0,15), amount 0..1 (0,8); isteğe bağlı color[3], position[3]. Varsayılan konum olaydan, renk ID mod 6'dan; mevcut WetPaintGpu kuyruğu |
| `paint.reset` | Mevcut atlas reset hizmeti |
| `balloon.burst` | count 1..3 (1); hazırlanmış mevcut balon demosu. Rastgele balon patlatır; hedefe özgü pop değildir |
| `audio.play` | Mevcut ses kaynağının id'si; ortak AudioService |

Olaylar ortak kanonik adlardır: gesture.stomp/arms_up/reach, contact, zone.enter/exit,
balloon.pop, coverage.threshold, timer, session.state. Coverage yalnız oturum hedefini
aşarken, timer oyun süresi biterken; session.state target yeni durum adıyla üretilir.
Olay kuyruğu 4096 ile sınırlı; taşma mevcut dropped_events sayacında görülür.
Kare başına 256 kural eylemi sınırı vardır, kural dropped sayaçları taşmayı gösterir.
Durum reset eylemi bu bütçeye eklenebilir. Eylem hataları actions_failed/last_error'da
görünür, sessiz başarıya çevrilmez. Kapsama ölçümü canlıda 0,5 s'de bir GPU indirgemesi
yapar; A'nın asenkron okuma optimizasyonu bu teslimde uygulanmadı.

## Doğrulama ve kullanıcı build kontrolü

Uygulama açmadan:

```powershell
python scripts/test/check_body_game_contracts.py
python scripts/audit_ipc_capabilities.py
python scripts/audit_player_includes.py --check
python scripts/gen_player_vcxproj.py --check
python -m unittest discover -s tools/perception_bridge/tests -v
python tools/perception_bridge/fake_children.py --dry-run --children 4 --seconds 36 --session-plan tools/perception_bridge/session_plan.json --output tools/perception_bridge/out/integrated-plan
```

Yeni C++ çekirdek/API/IPC/Python birimleri ve native test kaynağı `/Zs` kontrolünden
geçirildi; nesne/link çıktısı üretilmedi. Native `tests/body_pose_game_test.cpp`:
bozuk pakette atomiklik, zaman tekrarı, arms-up kenarı, stomp, yerel zaman aşımı,
grup kişi/hedef ayrımı, bozuk kuralda atomiklik, tam oturum ve hedef finali assert'leri
içerir. **Native assert'ler çalıştırılmadı**, kullanıcı bağımsız test build'inde yürütmeli.

Kullanıcı sırası:

1. İki projeyi reload edip Studio + Player build alın; yeni modül kayıtları nedeniyle
   proje yenilemesi gerekir. Ortak yaygın struct/header layout değişmedi.
2. Native testi, BodyPoseRuntime.cpp + GameRules.cpp + GameSession.cpp ile C++17,
   exception açık, assert açık (`/UNDEBUG`) olarak derleyip çalıştırın.
3. Güncel Player'da balon fixture'ını açın; hazır atlas/balonlar gereklidir.
4. Ayrı terminalden aşağıdaki 4 ve 10 çocuk kabulünü çalıştırın; uygulama gömülü
   script workspace'i bu IPC testine uygun değildir.

```powershell
python tools/perception_bridge/fake_children.py --children 4 --seconds 36 --session-plan tools/perception_bridge/session_plan.json --output tools/perception_bridge/out/live4
python tools/perception_bridge/fake_children.py --children 10 --seconds 36 --session-plan tools/perception_bridge/session_plan.json --output tools/perception_bridge/out/live10
```

Plan rastgele otomatik demo takvimini durdurur, boyayı sıfırlar, 26+3+2 s oturum kurar;
stomp boyar, arms-up balon patlatır. Tam durum sırası, hareket/bölge olayları, sıfır
eylem hatası, pozitif boya artışı ve ekran FPS'i JSON raporunda denetlenir.
Bu bir kamera/rig/görsel final kabulü değildir. Kamera köprüsünün kalan sınırları
[algı köprüsü belgesinde](PLAYER_PERCEPTION_BRIDGE.md).

Devam: rig dış poz/IK ve fiziksel vekil entegrasyonu → hedefli/özel balonlar ve
görsel final/Studio kural paneli → A GPU beklemeleri/partikül düzeltmeleri → D kiosk.
