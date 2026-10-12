# 2026-10-12 parti 11: hücre içi yapı (kuleler + girdap) · basınç → hava · fırtına gizmosu · yukarı yönlü yıldırım

Parti 10 derlendi; kullanıcı elle konan hücrenin doğru çalıştığını gördü.
Parti 9 ve 10'un geri kalan maddeleri ölçülmedi; listenin sonunda kısaca
duruyorlar. Parti 7–11 commit edilmedi.

Tasarım: `docs/dev/ATMOSPHERE_STORMS.md` §3.1, §5, §7.

## Özet
- **Hücre haritası 256×512.** Üst yarı eski alanlar. Alt yarı yeni:
  - RG: girdap kayması
  - B: yerel tepe
  - A: updraft
  Ayrı binding yok, iki yarı arası filtre sızıntısı yarım texel kıskacıyla
  önleniyor. `CloudCellGPU` 48 B oldu (id, yaş, dönüş, kule).
- **Kuleler:** Hücre başına 5 updraft. Her biri kendi ~15 dk döngüsünde
  doğuyor, yükseliyor ve çöküyor. Kuleler arasında gövde katmanın yarısı
  kadar; örtü orada %70.
- **Girdap:** Şekil okuması hücre merkezi etrafında döndürülüyor
  (spiral bantlar).
- **Updraft:** Kule çekirdeğinde erozyon yarıya iniyor, kabarmalar daha uzun.
- **Yeni alanlar:**
  - `storms.turrets` (0..1, varsayılan 0.8)
  - `storms.rotation` (−1..1, varsayılan 0.4)
  - `lightning.upward_fraction` (0..1, varsayılan 0.1)
- **Basınç:** `deriveWeather`, deniz seviyesi basınç sapmasını okuyor.
  Alçak basınç örtüyü, konveksiyonu, hücreleri ve geniş yağmuru artırıyor.
  Yüksek basınç örtüyü siliyor, tepeleri kesiyor ve fırtınayı bastırıyor.
  `world.get_weather.derived.synoptic_lift` alanı eklendi.
- **Gizmo:** Yeni `StormGizmoUI.cpp`.
  - Hücreler kolon olarak çiziliyor, renk evreyi gösteriyor.
  - Elle konan hücreler ve alan merkezi XZ'de sürüklenebiliyor.
  - Yazma `rtapi::setCloudsJson` üzerinden yapılıyor.
- **Yukarı yönlü yıldırım:**
  - Kanal yerden başlıyor, çatallar buluta doğru açılıyor, örümcek ağı
    bulutun içinde.
  - Dallar dönüş vuruşunda sönmüyor.
  - `cloud_stats.lightning[].kind` = `upward`.
- Yeni `.cpp` dosyası vcxproj ve filters'a eklendi.

---

## Kontrol listesi (sırayla)

1. **Derleme + shader'lar.**
   - Görmen gereken: hata yok. Log'da `cloud_cells pipeline unavailable` YOK.
   - Bozuksa: `StormGizmoUI.cpp` link hatası verirse vcxproj satırı eksik
     demektir.

2. **Basınç (IPC, görüntüden önce).**
   - Yap: `world.set_climate {surface_pressure_pa: 98300}`, ardından
     `world.get_weather`. Sonra `103800` ile tekrarla.
   - Görmen gereken:
     - Alçakta `synoptic_lift` ≈ +1; örtü, `storm_cells` ve
       `stratiform_rain` artar.
     - Yüksekte ≈ −1; örtü ×0.3, kalınlık ≤ 1500 m, `storm_cells` 0'a yakın.
     - 101325'te değerler eskisiyle aynı.
   - Bozuksa: değerler değişmiyorsa `rederiveClouds` çağrılmıyor demektir;
     panelde "Derive from climate" kapalı olabilir.
   - ★ Sinsi olan: elle kurulmuş (türetilmemiş) bir katman basınçtan
     ETKİLENMEZ. Bu beklenen davranış, bug değil.

3. **Hücre listesi yeni alanlarla.**
   - Bak: `world.cloud_stats.storms.cells`.
   - Görmen gereken: `age_s` artıyor. Elle konan hücrede `manual_index`
     0, 1, …; otomatik hücrelerde −1.

4. **Gizmo.** Fırtına açık, elle bir hücre ekle.
   - Görmen gereken:
     - Yerden bulut tabanına halkalı kolonlar; renk evreye göre (büyüyen
       mavi, olgun turuncu, dağılan gri, kalıcı sarı).
     - Alan karesi ve "Storm domain" tutamağı.
     - Hücre tutamağına tıklayınca XZ okları çıkar; sürükleyince hücre ve
       bulut birlikte kayar, panel değerleri güncellenir.
     - Boş yere tıklayınca seçim kalkar.
   - Bozuksa:
     - Tutamak nesne seçimini de tetikliyorsa `hit` yayılmıyor.
     - Bırakınca hücre geri zıplıyorsa sürüklenme düzeltmesi yanlış
       (ömrü > 0 olan hücre).
   - ★ Sinsi olan: kolonlar sahne ölçeğinde çok büyük (km). Kamera yakınken
     ekran dışında kalabilirler. Uzaklaşıp bak.

5. **Hücre içi yapı (asıl istek).** Elle kalıcı hücre (ömür 0), RT.
   - Görmen gereken:
     - Tek düze blok yerine farklı yüksekliklerde 3–5 kule; aralarında daha
       alçak gövde.
     - Kule tepeleri sert karnabahar; spiral bant izi.
     - Timeline oynarken kuleler dakikalar içinde yükselip çöker.
   - Karşılaştır: `storms.turrets 0` eski düz kolonu, `rotation 0` girdapsız
     hali vermeli.
   - Bozuksa:
     - Hücre kenarında yatay bir dikiş varsa iki yarı birbirine sızıyor
       (uv kıskacı, `cloudStormMapAt`).
     - Kulelerin yerine delikler varsa `flow.z` 0 okunuyor; topFrac 0.25'e
       düşüyor.
   - ★ Sinsi olan: katman ince ise (stratokümülüs, ~1–2 km) kuleler yine
     alçak kalır, çünkü kule katmanı aşamaz. Kalın katmanla (türetilmiş
     kararsız iklim) bak.

6. **Yukarı yönlü yıldırım.** `upward_fraction 1`, ground strike 1, gece.
   - Görmen gereken:
     - Kanal yerden yukarı büyür; çatallar yukarı ve buluta doğru açılır;
       bulut içinde yatay ağ en son çıkar.
     - Bağlandıktan sonra dallar parlak kalır.
     - `cloud_stats.lightning[i].kind = "upward"`.
     - `segments[0].a` yerde (y düşük), `appear` 0.
   - Bozuksa: dallar hâlâ aşağı açılıyorsa start/end takası yapılmıyor.

7. **Kare süresi (kule yükü).** Dünkü sahne, hücreler açık.
   - Görmen gereken: parti 10'a göre fark küçük. Hücre içindeki örneklerde
     ek bir doku okuması var; dışarıda yok.
   - Bozuksa: belirgin artış varsa `cell.a > 0` kapısı çok geniş demektir;
     dilation, majorant segmenti kadar.

## Önceki partilerden ölçülmeyenler (kısa)
- **Parti 10:**
  - Yağış yalnızca hücre altında mı (`stratiform_rain 0`)?
  - Şimşek yalnızca olgun hücreden mi çıkıyor?
  - Oynatırken kare süresi (bekleme kaldırma).
- **Parti 9:**
  - Kanal ağı `segments`.
  - Bulut içi ışık dalları izliyor mu?
  - Play → durdur sonrası bloklaşma.

## Bilinen
- `scripts/audit_ipc_capabilities.py`, `audio.*` metotlarında FAIL veriyor.
  Bunlar paralel ajanın ses işi, bu partiyle ilgisiz. Tarif tablosu
  yeniden üretildi.
