# Realtime gaz/VDB — sonsuz uzak düzlemden geçersiz ışın

> **Durum:** ARŞİV — kullanıcı düzeltmenin çalıştığını, hızlı ve doğru göründüğünü, UI parametreleriyle kontrol edildiğini doğruladı (2026-09-06). Hacmin gölge düşürmesi ayrı açık iş.

**Kabul güncellemesi:** aşağıdaki derlenmedi/görsel kabul bekliyor ifadeleri
ilk teslimin kaydıdır. Sonraki kullanıcı testi gaz/VDB görünürlüğünü ve UI
kontrolünü doğruladı. Bütün kamera/örtüşme regresyon maddelerinin ayrı ayrı
koşulduğu bildirilmedi; bu kapsamda ek kabul iddiası yok.

**Gölge incelemesi:** mevcut `selfShadow()` yalnız örneklenen hacmin içini
örnekler. Raster yüzeylerin `evaluatePreviewShadow()` yolu mesh shadow atlasını
okur; bu atlasın üreticisi `m_rasterMeshes` çizer, gaz/VDB extinction alanı
üretmez. Volume shader da opaque shadow atlasını okumaz. Hacim→yüzey ve
yüzey→hacim gölgeleri, görünürlüğü düzelten bu ışın değişikliğinden ayrı iştir.

## Açık uygulamada ölçülen

Kullanıcı gaz ve VDB'nin Rendered modunda doğru, Realtime'da görünmez olduğunu;
SDF yüzeyin çalıştığını bildirdi. IPC ile mevcut sahne okundu:

| Ölçüm | Render | Viewport |
|---|---:|---:|
| `render.volume_tables` instance_count | 2 | 2 |
| buffer_allocated | true | true |
| dense_gas_mirror_buffers | 0 | 1 |
| sim_device_is_this_backends | true | false |

`gas.list_domains`: görünür, etkin, canlı `Fireball Gas`, volume yolu,
59×106×59 hücre. Logda VDB için `volume_type=2`, gaz için
`source_type=5 volume_type=4 liveDenseGas=1` ve `[MPVolume] ... RECORDED` var.
Yakalanan Realtime karede hacim görünmüyor. Geçici viewport capture kapatıldı;
sahne, kamera ve shading seçimi değiştirilmedi.

Kaynak ve runtime `material_preview_volume.spv` dosyaları 11:44:07'de üretilmiş,
çalışan uygulama 11:45:16'da açılmıştı. Bu ölçüm sırasında eski shader/değişikliğin
derlenmemiş olması açıklaması geçerli değildi. **Bu partinin yeni shader'ları
ise henüz derlenmedi.**

Önceki [tablo yayımlama arızası](REALTIME_VOLUME_TABLE_NEVER_PUBLISHED.md)
bu ölçümde tekrar etmiyor. Tablo sayısı ve draw kaydı, geçerli kamera ışını veya
görünür piksel kanıtı değildir.

## Sayısal olarak yeniden üretilen hata

Raster projeksiyon `near=0.01`, `far=1000000` kullanıyor. Float32'de
`far / (near - far)` tam `-1` oluyor. Shader `inverse(viewProj)` ile `z=1`
uzak düzlemini geri açıp `farH.xyz / farH.w` hesaplıyordu.

IPC'den okunan kamera:

- position = (10.2010746, 4.34994745, 14.39588165)
- target = (2.33573723, 1.82005453, -0.82628345), up = (0, 1, 0)
- FOV = 40°, görüntü = 1680×945

Aynı kaynak projeksiyon/view formüllerinin NumPy float32 hesabında merkezde
`farH = (-0.4541626, -0.14608765, -0.87890625, 0)` elde edildi.
Bölme sonlu bir ışın vermiyor. İnversiyon yuvarlaması küçük negatif `w`
üretirse ışın ters dönebilir. GPU'nun inversiyon ara sonuçları ayrıca
ölçülmedi; nihai görsel doğrulama build sonrasına açıktır.

SDF shader'ı da aynı kırılgan uzak düzlem bölmesini taşıyordu. Kullanıcının
SDF'yi başka görünümde görmesi bu hesabın bütün kameralarda güvenli olduğunu
göstermez; iki çağıran ortak yardımcıya alındı.

## Değişiklik ve sınır

`material_preview_ray.glsl`, NDC z=0 ve z=0.5 sonlu noktalarından ışın kurar.
Başlangıç near plane üzerindedir; perspektif, ortografik ve hacim içindeki
kamera aynı işlemi kullanır. Hem `material_preview_volume.frag` hem
`material_preview_sdf_surface.frag` bu yardımcıyı çağırır.

Volume shader NDC'yi fullscreen üçgenden alır. Depth snapshot olmayan
çağrıda 1×1 fallback doku boyutuyla piksel koordinatını bölüp yanlış yöne
bakmaz. Opaque derinlik kesmesi de aynı ışın başlangıcına göre hesaplanır.

Yeni API/IPC parametresi, C++ dosyası, hacim verisi veya malzeme değişikliği
yok. Mevcut UI/script/IPC shading işlemleri aynı düzeltilmiş shader'ı kullanır.
Bu parti RT ile bütün hacim ışıklandırma özelliklerinin eşitliği iddiası değildir.

## Kontroller

Derleme yapmadan 3 kamera × 5 ekran noktası × 2 projeksiyon = 30 float32
ışın, bağımsız kamera-bazı yönüyle karşılaştırıldı. Hepsi sonlu; en büyük yön
vektörü hatası 0.0000933. Eski hesap 15 perspektif örneğinin tamamında
`farH.w <= 0` verdi. İki shader'ın ortak yardımcıyı çağırdığı ve eski bölmenin
kaldığı bir çağıran olmadığı statik kontrol edildi. Bunlar GLSL derlemesi
veya GPU görsel kabulü değildir.

Sıralı manuel kabul: [NEXT_BUILD_CHECKS.md](NEXT_BUILD_CHECKS.md).
