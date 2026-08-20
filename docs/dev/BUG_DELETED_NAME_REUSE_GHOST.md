# AÇIK ARIZA: silinmiş bir adı tekrar kullanmak "yarı var" bir nesne üretiyor

> **Durum:** AKTİF — 2026-08-19'da ölçüldü ve belirleyici olarak yeniden
> üretildi. Kök neden bulunmadı, düzeltme yapılmadı. Kod değişikliği gerektirir.

## Nasıl bulundu

Yerel model ufuk testinde ([LOCAL_MODEL_HORIZON.md](LOCAL_MODEL_HORIZON.md))
ajan `AgentBox` ekledi, motor **başarı** döndürdü, sonra `set_transform`
**başarı** döndürdü — ama koşu bitince sahnede öyle bir nesne yoktu. Ajanın
hatası sanıldı; değildi.

★ Bunu bulan şey ajanın kendisi değil, **koşunun öncesi/sonrası sahne listesini
kaydediyor olması**. Sadece çağrı sonuçlarına bakan bir test "7 çağrının 7'si
başarılı" derdi.

## Yeniden üretim (belirleyici)

```python
c.call('scene.add_primitive', {'type':'cube','name':'GhostBox'})  # -> 'GhostBox'
'GhostBox' in c.call('scene.list_objects')                        # -> True
c.call('scene.delete', {'name':'GhostBox'})                       # -> True
'GhostBox' in c.call('scene.list_objects')                        # -> False
c.call('scene.add_primitive', {'type':'cube','name':'GhostBox'})  # -> 'GhostBox'  (BASARI)
'GhostBox' in c.call('scene.list_objects')                        # -> False       (YOK)
```

Aynı adı **silmeden** tekrar kullanmak sorunsuz (`AgentBox2`, `T_*` adlarının
hepsi eklendi). Sorun **sil → aynı adla ekle** dizisine özgü.

## Ölçülen davranış — iki okuyucu birbirini yalanlıyor

`GhostBox` ikinci eklemeden sonra:

| çağrı | sonuç |
|---|---|
| `scene.list_objects` | listede **yok** |
| `scene.object_exists` | **False** |
| `scene.get_transform` | **başarı**, kimlik matrisi |
| `scene.set_transform` | **başarı** (`true`) |
| `material.of_object` | **başarı**, `["Default_Cube_Material"]` — başka nesnenin malzemesi |

★★★ Karşılaştırma kontrolü: **hiç var olmamış** bir ad doğru davranıyor —
`scene.set_transform {name:'NoSuchObject_xyz'}` → `object not found`. Yani
mesele "isim çözümlenemiyor" değil; **silinmiş ad bir kalıntıya çözümleniyor.**

Hayalet kendi durumunu tutuyor: `set_transform(999,999,999)` yazıp geri okuyunca
`[999,999,999]` dönüyor. Yani nesne **var**, ama sayıcı/arayıcı onu görmüyor.

## Şiddet

★ İyi haber: hayalete yazmak **gerçek bir nesneyi oynatmıyor**. Test edildi —
`Default_Cube` ve `T_nosize` transformları değişmedi.

★★★ Kötü haber: bir ajan (veya script) için bu **sessiz kayıp iş**. Nesneyi
kurar, taşır, malzemesini ayarlar, her adımda `true` alır, sonra render'da
hiçbir şey görmez ve nedenini gösteren tek bir hata mesajı yoktur.

## Muhtemel kök — nerede aranmalı

Bu deponun bilinen sınıfı: **"Yok" ≠ "silinmiş"**, ve
`CLAUDE.md`'deki flat SoA notu:

> Sahnede nesne sayan/arayan kod yalnızca facade tararsa flat mesh'leri "yok"
> sayar ve bunun hiçbir belirtisi olmaz. `rtapi::listObjects()` ikisini de ele
> alıyor, örnek al.

`object_exists` ile `get_transform`'un **farklı arama yolları** kullanıyor
olması en olası açıklama: silme, listeleyicinin gördüğü kaptan çıkarıyor ama
transform'un baktığı kayıttan çıkarmıyor; ikinci ekleme de yalnızca ikinciye
yazıyor. Akraba vaka: `bugfix_rigid_body_rebuild_leaks_old_jolt_body`
("yeniden kur" SİLME değildir).

## Sıradaki adım

1. `rtapi::objectExists()` ile `rtapi::getObjectTransform()`'un arama yollarını
   yan yana koy — hangisi facade, hangisi flat SoA, hangisi ayrı bir isim
   haritası.
2. `scene.delete`'in **hangi** kaplardan çıkardığını say.
3. ★ Düzeltmeden sonra kontrol: silme sonrası `object_exists` ve
   `get_transform` **aynı** cevabı vermeli — ikisinin ayrışabildiği her nokta
   bu arızanın tekrar doğabileceği yerdir.
