# Camda specular yansıma lobu yoktu — çünkü ışıklar TLAS'ta değil

> **Durum:** REFERANS — kök neden bulundu, Vulkan tarafı düzeltildi (2026-09-06).
> OptiX'te NEE boşluğu **hâlâ açık**; bu notun sonundaki "Kalan" bölümüne bakın.

**Belirti (kullanıcı, 2026-09-06):** "Vulkan RT'de cam malzemede specular bir
yansıma lobu gözleyemedim."

Belirtinin okunuşu önemli: bu "lob zayıf" değil, **lob yok** demekti. Ve bunu
üreten şey tek bir hata değil, **ayrı ayrı doğru olan iki kararın kesişimi**.

---

## Kök neden: iki doğru karar, bir kör nokta

**1. Cam lobu NEE'yi atlar — ve bu doğrudur.**
`closesthit.rchit`'te transmission lobu seçildiğinde `scatterGlass()` çağrılır ve
fonksiyon **direct lighting bloğuna girmeden `return` eder**. Gerekçe sağlam:
Fresnel yansıt/kır kararı bir *specular seçim*dir, bir BRDF değerlendirmesi
değil. OptiX kopyası aynısını `*is_specular = true` ile yapar ve oradaki yorum
bunun neden böyle olduğunu da yazmış: NEE her kırılmada koşunca kırılan yolda
parlak specular fireflies üretiyordu.

**2. Sahne ışıkları ANALİTİKTİR — ve bu da doğrudur.**
`lights.l[]` bir uniform dizidir; nokta/yön/alan/spot ışığın **TLAS'ta geometrisi
yoktur**. Işıklar yalnızca `sample_light_direction_gl()` ile *açıkça örneklenerek*
görülebilir.

Tek tek bakınca ikisi de savunulabilir. Birlikte şunu üretirler:

> Cam yüzeyin ayna lobu, NEE'yi atladığı için ışığı **örnekleyemez**; ışığın
> geometrisi olmadığı için de yansıyan ışın onu **çarpamaz**. Yol ne kadar uzun
> olursa olsun, kaç örnek verilirse verilsin sonuç aynı: **sıfır.**

Bir cam küre yalnızca *ortamı* yansıtabiliyordu — Physical Sky'ın güneş diski,
emissive bir mesh, arkasındaki geometri. Lambayı asla. Karanlık dünyada duran bir
cam küre, sahnede beş lamba olsa bile **tamamen siyahtı**.

★ **Genel ders:** bir lobun NEE'yi atlaması, ışıkların *izlenebilir geometri*
olduğu bir renderer'da bilgi kaybı değildir — ışın onları zaten bulur. Işıklar
analitikse aynı karar **yeteneği tamamen siler**. "NEE'yi atla" kararı, ışık
modelinden bağımsız okunamaz.

---

## Bunun neden fark edilmesi zor

- **Ortam ışığı yalanı kapatıyor.** Physical Sky açıkken camda bir parlaklık
  *vardır* (güneş diski gerçekten çevrede duruyor). Yani hata yalnızca lamba ile
  aydınlatılan sahnelerde görünür — ve o zaman da "cam koyu çıktı" diye okunur.
- **Metal aynı sahnede çalışıyor.** Metal, generic NEE bloğundan geçer.
  Yan yana duran metal ve cam küreden yalnızca metalin parlaması, insanı
  "camın Fresnel'i zayıf" diye ayarlamaya iter — ayarlanacak bir şey yoktur.
- **Su bu duvara ÇOKTAN çarpmıştı.** `addWaterV3DirectLighting`'in başındaki
  yorum aynen şunu söylüyor: *"generic material NEE block is intentionally
  bypassed by the water fast path, so water needs its own dielectric GGX
  estimator or scene lights only appear through secondary rays."* Yani teşhis
  depoda zaten yazılıydı — **bir yüzey tipi için**. Cam aynı sınıfta olduğu
  halde payını almadı.

---

## Düzeltme

Su'nun estimator'ı **paylaşılan bir servise** dönüştürüldü ve cam onun tüketicisi
oldu (yeni bir kopya değil):

| Dosya | Değişiklik |
|---|---|
| `bsdf_scatter.glsl` | `addWaterV3DirectLighting` → `addDielectricDirectLighting`. Foam'a özel `foamCoverage` parametresi genel `diffuseAlbedo` + `diffuseWeight` çiftine dönüştü (su hâlâ foam'ı bununla veriyor), ve `originPush` eklendi — level-set çağıranı kendi bant çıkış mesafesini kullanabilsin diye |
| `closesthit.rchit` | Üçgen cam lobunda, `scatterGlass`'tan **önce**, ön yüzde estimator çağrısı |
| `volume_closesthit.rchit` | Sıvı izoyüzeyine bağlı transmissive materyalde aynı çağrı (aynı boşluk oradaydı) |
| `material_scatter.cuh` (OptiX) | ★ Ayrı bir hata: `ggx_glass_effective_normal` bir **yön** döndürüp mikrofaset **normali** olarak kullanılıyordu → pürüzlü cam iki kez yansıtılıyordu. Vulkan ikizinin (`ggxSampleHemisphere`) başındaki uyarı tam bunu anlatıyor; düzeltme o kopyaya hiç uğramamış. Artık yarı vektörü döndürüyor |

### Ağırlık neden 1, `transmission` değil

Cam dalına **zaten `transmission` olasılığıyla** giriliyor ve bu dalda `1/p`
telafisi uygulanmıyor (`scatterGlass` yansıma lobunda `attenuation *= 1.0`).
Dolayısıyla estimator, cam lobunun tüm payıdır. `transmission = 1` — yani asıl
şikâyet edilen durum — için taban dalı hiç koşmaz, çift sayım yoktur.

★ `transmission < 1`'de taban dalı kendi specular'ını `1/(1-t)` ile tam ağırlıkla
üretmeye devam eder; yani yarı saydam bir materyalde toplam specular
`spec_taban + t·spec_cam` olur. Bu **bu depodaki mevcut karışım sözleşmesidir**
(OptiX'te de `albedo *= 1/(1-eff_transmission)` ile aynısı yapılır), bu partide
değiştirilmedi — ama doğru olduğu için değil, **ayrı bir iş olduğu için.**

---

## Ölçü aleti

`scripts/probe_glass_specular_lobe.py` (ve `x64/Release/scripts/` kopyası).

Siyah dünya (solid mod, arka plan 0,0,0) + **tek** nokta ışık: ortam olmadığı
için karedeki her siyah olmayan piksel bir doğrudan-ışık cevabıdır, ve
`render.probe` bunu parlak noktanın ekranda nereye düştüğünü bilmeden ölçer.

★ Kritik parça **kontrol küresi**: aynı küre önce metal olarak ölçülür. "Cam
siyah" ile "lamba kapalı / kamera boşluğa bakıyor / viewport hiç converge
etmedi" **birebir aynı ölçümü** verir, ve otomatik koşuda ikinci aile daha
olasıdır. Karanlık bir özneyi ancak parlak bir kontrol anlamlı kılar.

★ Aletin yakaladığı sinsi başarısızlık ise tersidir: cam **yanıyor ama fazla
parlak** (estimator iki kez ekleniyor). Bu yüzden script cam/metal oranını
raporlar — 1.5 IOR'lu bir dielektrik normal geliş açısında ~%4 yansıtır, yani
camın aynadan parlak çıkması "siyah değil" testini geçse bile yanlıştır.

---

## Kalan

- **OptiX'te NEE boşluğu duruyor.** `ray_color.cuh` içindeki NEE `!is_specular`
  ile kapılı ve transmission lobu `is_specular = true` yazıyor. Vulkan birincil
  olduğu için (kural 6) bu partide yalnızca mikrofaset normali düzeltildi;
  estimator portu ayrı bir iş. Sonucu: **aynı sahne iki backend'de farklı** —
  lambalı bir sahnede cam Vulkan'da parlar, OptiX'te parlamaz.
- **Kısmi transmission'daki karışım sözleşmesi** (yukarıda ★): taban katmanı
  `1/(1-t)` ile tam ağırlık alıyor. Denetlenmedi.
