# Import/export — kapanmamış borçlar

> **Durum:** AKTİF — Faz 3 (Assimp'in sökülmesi) 2026-09-06'da bitti ve
> doğrulandı. Bu not, o plandan **bilerek dışarıda bırakılan** işleri tek yerde
> toplar. Kapanan planın kendisi: [FAZ3_DEVIR_NOTU.md](FAZ3_DEVIR_NOTU.md).

Her madde **ölçülmüş** veya **gözlenmiş**tir; hiçbiri "şunu da yapsak iyi olur"
değil. Sıralama önem derecesine göre.

---

## 1. ★★★ Rule 1 borcu: `scene.import_model` hiçbir şey döndürmüyor

**Durum:** ölçüm var, yüzey yok.

Okuyucular `ImportStats`'ı zaten dolduruyor — mesh/üçgen/materyal/görüntü
sayıları, kemik ve klip sayısı, instance sayısı, ve **faz faz süre** (ayrıştırma
/ materyal+doku / geometri / animasyon). Bunların hiçbiri script tarafına
açılmıyor: `scene.import_model` çağrılır, bir şey olur, geriye hiçbir sayı
dönmez.

**Neden borç:** CLAUDE.md kural 1'in tam ihlali. Bu projede manuel test seyrek,
IPC katmanı **QA altyapısı**. İçe aktarma maliyeti ve sonucu script'ten
okunamadığı sürece import gerilemeleri **ancak gözle** yakalanabilir — ve
"1.100 mesh mi 977 mesh mi geldi" gözle yakalanmaz.

**Yapılacak:** beş dokunuş + `python scripts/gen_ipc_descriptors.py`.
★ Faz 3'te `RtApiImport`/`RtIpcImport`/`RtPythonImport` dosyaları **silindi**
(içlerinde yalnızca okuyucu seçenekleri vardı, onlar da Assimp ile öldü). Bu iş
o dosyaları geri getirecek; doğal yeri orası.

---

## 2. ★★ Paylaşılan mesh'ler: 4.39M kopya üçgen

**Durum:** ölçüldü, düzeltilmedi.

Bir Unreal GLB'sinde **yalnızca 7 mesh** birden fazla düğüm tarafından
paylaşılıyor, ama tekrar sayıları yüksek — biri **315 düğüm**. Okuyucu dosyanın
düğüm grafiğinde ne yazıyorsa onu kuruyor (sayım doğru, çift kayıt **yok**), ama
fazladan tutulan geometri **4.393.271 üçgen**: RAM'e, VRAM'e ve BLAS sayısına
yazılıyor.

**Yapılacak:** "aynı mesh'i gösteren N düğüm" desenini tanıyıp
`EXT_mesh_gpu_instancing` varmış gibi `InstanceGroup`'a çevirmek.
★ İki şart: motor mesh başına tek `transform` tuttuğu için bu **gerçek
instancing** ister (sahne objesi kopyası değil), ve bir **eşik** gerekir —
2 düğümlük bir mesh için InstanceGroup kurmak kazançtan çok karmaşıklık getirir.
Seçimin/outliner'ın gördüğünü değiştirdiği için **açık bir import seçeneği**
olmalı, sessiz bir optimizasyon değil.

---

## 3. ★★ `KHR_texture_transform` okunmuyor

**Durum:** bilinen eksik.

Blender'da Mapping node kullanan her varlık bu uzantıyla gelir; offset/scale/
rotation **sessizce** yok sayılıyor.

★ Kopyala-yapıştır bir düzeltme burada işe yaramaz, çünkü üç konvansiyon
çakışıyor: motorun `applyMaterialUVTransform`'u **(0.5, 0.5) merkezli**,
glTF'inki **(0,0) çıpalı**, üstüne import'taki V çevirmesi var. Üçünü birden
çözen gerçek bir dönüşüm gerekiyor.

---

## 4. ★★ Yazıcının skin sözleşmesi standart dışı

**Durum:** okuyucu telafi ediyor, yazıcı bozuk.

Kendi export ettiğimiz skinli glTF, standart skin semantiğini sağlamıyor.
Okuyucu bunu `generator == "RayTrophi Studio"` **sniff'i** ile telafi ediyor.

★ Bu telafi kalıcı hale gelirse dosyalarımız **Blender'da yanlış açılır** ve biz
bunu asla göremeyiz — kendi turumuz simetrik olduğu için hatayı ölçemez. Doğru
sıra: yazıcıyı düzelt, sonra sniff'i sök.

---

## 5. ★ OBJ: Assimp'ten bilerek farklı davranan tek yer

**Durum:** kasıtlı, belgelenmiş.

Assimp `AI_MATKEY_ROUGHNESS_FACTOR` okur; MTL'de karşılığı yalnızca `Pr`. Sadece
`Ns` yazan klasik bir MTL Assimp'te **roughness 0.0** ile, yani **ayna** olarak
gelirdi. Doğrudan okuyucu `Pr` yoksa `Ns`'den türetiyor (`sqrt(2/(Ns+2))`).

Borç değil, **karar**. Buraya yazılma sebebi: eski dosyaları yan yana koyan biri
farkı görüp bug sanmasın.

---

## 6. ★ Kozmetik: `assimpVertexIndices` alan adı

`Triangle.h`'de bir alan hâlâ `assimpVertexIndices` adını taşıyor (5 dosya, 10
kullanım). Anlamı değişmedi — orijinal vertex indeksleri — sadece adı artık
yanlış bir kaynağa işaret ediyor. Mekanik bir yeniden adlandırma; derleyici
kaçırdığını söyler.

## 7. ★ `TerrainSatMapNodes.cpp` iki yerde

`src/Physics/` (1802 satır) ve `src/Scene/` (178 satır). vcxproj yalnızca
Physics olanı derliyor, ama **CMake recursive glob kullandığı için ikisini de**
derler. Hangisinin ölü olduğuna bakılmadı; ölüyse sökülmeli (kural 5).

---

## Faz 3'ten devralınan, hâlâ geçerli referans

Yeni bir okuyucu/yazıcı yazan herkes önce
[FAZ3_DEVIR_NOTU.md](FAZ3_DEVIR_NOTU.md) §3'ü okusun: bu alanda **zaten bir kez
ödenmiş** tuzakların listesi (nodeName gruptur-kimlik değil, dalı kaldırmak
yola giden yolları kapatmaz, simetrik hatayı round-trip ölçemez, kaybedilen
optimizasyon hiçbir şey olarak raporlanır, "geçerli" ≠ "güvenle okunur"...).

★ Bir de bu partinin kendi dersi: **bir bağımlılığı sökerken tarama aracını işe
göre kalibre et.** Faz 3'ün son partisinde derleme üç turda oturdu ve üçü de kod
hatası değil tarama zayıflığıydı — (1) uzantı listesinde `.hpp` yoktu, (2) desen
yerine isim listesi kullanıldı, (3) satır bazlı arandı ve çok satırlı çağrı
kaçtı. Doğrusu: yorumları söküp **dosyanın tamamında desenle** ara.
