# Boyayla beliren 3D tavşan

2026-11-20. Kaynak ve deneme sahnesi hazır; C++ build ve yeni özelliğin canlı
kabulü kullanıcı build'ini bekliyor. Açık eski Studio'da yalnız tavşanın opak
geometrisi görsel olarak kontrol edildi. İlk salon ve önceki mural denemesi
üzerine yazılmadı.

## Deneme

`RayTrophiStudio/assets/scenes/paint-reveal-rabbit/paint-reveal-rabbit.rtp`

Duvara yakın, 3.136 üçgenli bir tavşan; tek malzeme, birbirine binmeyen 14 UV
adası, 512² wet-paint yüzeyi. Gövde/baş/kulaklar kabartmalı; göz, burun ve
yanaklar da geometri. Tek mesh-sdf collider damlacık sistemlerine kayıtlıdır.
Collider, objenin görünmez kısmında da boya teması üretir; görünmezlik fizik
çarpışmasını kapatmaz. Her kare yeni geometri üretilmez.

Başlangıç opacity veri dokusu sıfırdır. Yeni binary, kayıttaki `reveal_opacity`
ayarını yükler ve boya geldikçe yüzeyin yalnız boyalı bölümünü görünür yapar.
Wet paint oturumluktur; temizleme ve yeniden açma objeyi tekrar gizler.

Sahne mevcut eski binary ile `--prepare-only` kullanılarak hazırlanmıştır.
**Eski binary'den yeniden kaydetmeyin:** yeni ayarı tanımadığı için düşürür.
Build sonrası dosyayı diskten yeniden açın. Bu sahne henüz Player paketine
eklenmedi; paketlenen eski exe yeni davranışı desteklemez.

## Ortak işlem

Studio Game Design > boya yüzeyi > Material channels:
`Reveal object with paint`.

Python: `rt.paint.wet_set_settings(object, reveal_opacity=True)`.
IPC: `paint.wet_set_settings`, `{"object":"3_PaintRevealRabbit","reveal_opacity":true}`.
Get settings ve wet status aynı bool'u döndürür. Hepsi mevcut
`rtapi::setWetPaintSurfaceSettings` -> `WetPaintGpu` hizmetini kullanır.
GameDesign JSON ayarı kaydeder; eski projelerde varsayılan false'tur.

Bool dışındaki türler ve bilinmeyen anahtarlar mutasyondan önce reddedilir.
Python kwargs da bilinmeyen anahtarları ve bool olmayan reveal değerini reddeder.
Eksik hedef veya yüzey mevcut wet-paint hata sözleşmesini kullanır.

Bu mod **opak geometri** içindir ve opacity haritasını sahiplenir. Görünürlük,
paint coverage değeridir; özgün kesit/foliage maskesini çarpmaz. Modu kapatmak
opak görünürlüğe döner, paint reset ise mod açıkken tekrar görünmez yapar.
Malzemenin opacity scalar değeri ayrıca çarpılır; tavşanda 1'dir.

## GPU ve maliyet

17 binding ve 64-byte push constant korunur. Surface flag bit 3 (8) reveal'dır.
Push `_p1` bit0 mevcut force-refresh, bit1 isteğe bağlı beşinci output plane
olduğunu belirtir. Yeni shader eski host'un yalnız 0/1 gönderdiği `_p1` ile
fazladan plane okumaz/yazmaz. Kaynak plane4/5 UV metriği olarak aynıdır.

İlk reveal yüzeyi eklenince mevcut PBR atlasına bir RGBA8 opacity output plane
eklenir: atlasın toplam texel sayısı × 4 byte. Ayrıca reveal yüzeylerinin opacity
dokusu ve host kaynak snapshot'ı vardır. Reveal olmayan sahnelerde eski atlas
boyutları kalır. Reveal kapatılınca mevcut atlas sonraki rebuild'e kadar tutulur.
Opacity de mevcut dirty-rectangle publish yoluyla kopyalanır; Player'ın aynı
device yolu D2D, Studio'nun ikinci device yolu mevcut host fallback'idir.

`paint_wet.comp` derlendi, `spirv-val --target-env vulkan1.2` geçti ve SPV hem
source/shaders hem x64/Release/shaders içine kondu. Mevcut bulk compile satırı
paint_wet'i zaten içerir. Başka shader derlenmedi.

## Build ve kabul

1. Studio ve Player C++ hedeflerini güncel header'larla derleyin (Codex çalıştırmadı).
2. Yukarıdaki projeyi yeniden açın; Material görünümünde duvar boş olmalı.
3. Balon boyası tavşana değince yalnız o bölgede renkli hacim belirmeli.
4. Clear paint sonrası hacim ve gölgesi gizlenmeli. Reveal kapatılınca opak
   tavşan, tekrar açılınca yalnız mevcut boya görünmeli.
5. Kaydet/aç: bool korunmalı, oturum boyası temiz başlamalı. Player'da da aynı.

Harici otomatik kabul (oturum boyasını temizler, görüntüleri RAM'de tutar):

```powershell
python scripts/test/rt_paint_reveal_ipc.py
python scripts/test/rt_paint_reveal_ipc.py --player
```

Test: yanlış tür/bilinmeyen anahtar atomik reddi; temiz tavşan görüntüsünün
gizlenmiş mesh ile eşitliği; gerçek wet dab'in geometriyi açması; reset ve
enable/disable görüntü farkları. Canlı reveal testi henüz ÇALIŞTIRILMADI.
Geometri/UV statik kontrolleri ve shader doğrulaması yapıldı; C++ derleme sonucu
ve gerçek Player alpha/gölge/collider kabulü bu testin yerini tutmaz.

Sahneyi yeniden hazırlama: `python scripts/scenes/paint_reveal_rabbit.py`.
Build öncesi yalnız hazırlama: `--prepare-only`; yalnız OBJ/data mask: `--assets-only`.
IPC istemcisi ayrı terminalde çalıştırılır, embedded script çalışma alanında değil.

## Kullanici kabulu

2026-11-20: kullanici yeni Player ile denedi ve gizli 3D tavsanin boyayla
belirmesinin calistigini, sonucu basarili buldugunu bildirdi. Bu gozle kabul;
otomatik IPC/golge/yeniden yukleme testinin calistirildigi anlamina gelmez.
