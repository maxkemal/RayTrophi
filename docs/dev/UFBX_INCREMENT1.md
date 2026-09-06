# Faz 3 — ufbx ilk artış

> **Durum:** AKTİF — kullanıcı derledi; Twin Towers FBX ufbx import logu doğrulandı. Görsel kabul bekliyor. 2026-09-05.

Bu artış statik FBX okuyucusunu kurar. Skinning/animasyon, OBJ okuyucusu ve
Assimp'in kaldırılması sonraki ayrı kabul partileridir. İlk kullanıcı ölçümü:
Twin Towers FBX, 776 mesh / 187.404 üçgen / 13 materyal, ufbx toplam 52,092 ms
(parse 31,200; materyal 0,308; geometri 20,298 ms). Prefetch 0/0 olduğundan doku
aktarımının kabulü açık. Önceki Assimp 303 ms kaydı yalnızca dosya okumasıdır;
eş kapsamlı toplam hız karşılaştırması henüz yok.

## Kullanım ve ortak sözleşme

- **File → FBX Reader → ufbx (static models)**, sonra normal model importu.
- Python: `rt.scene.set_fbx_reader("ufbx")`, `rt.scene.get_fbx_reader()`,
  ardından mevcut `rt.scene.import_model(path)`.
- IPC: `scene.set_fbx_reader {"reader":"ufbx"}`,
  `scene.get_fbx_reader {}`, ardından `scene.import_model {"path":"..."}`.
- Ayar oturumluk; uygulama yeniden açıldığında `assimp`. Yeni import başında
  okunur; mevcut nesneleri dönüştürmez. Geçerli değerler tam olarak `assimp`, `ufbx`.
- Geçersiz değer mevcut ayarı korur; çekirdek başarısız Result, Python ValueError,
  IPC `invalid_parameter` üretir. Getter `Read`, setter `SceneWrite` ister.
  Importun mevcut `FilesRead | SceneWrite` yetkisi değişmedi.

UI, rtapi, Python ve IPC aynı `ImportSettings` servisini kullanır.
Ana sahne `loadSceneModel → loadModel`, bitki ve klip yükleme doğrudan `loadModel`
çağırır. glTF/GLB daima cgltf; OBJ halen Assimp. FBX seçilen okuyucuya gider.
Bitki geometri cache'i FBX için okuyucu kimliğini de denetler; ayar değişince
önceki okuyucudan kalan geometri yeni seçimmiş gibi sunulmaz.
Okuyucu hatasında başka okuyucuya geçilmez. Ana sahne adaptörü hatayı fırlatır;
`ProjectManager` başarısız importu proje listesine eklemez. Ayrıntı logdadır;
`scene.import_model` halen genel import başarısızlığı döndürür.

## Geometri ve hiyerarşi

- `ImportedModel` sözleşmesi, doğrudan `TriangleMesh`/DNA SoA; köşe başına vertex.
  Mevcut mesh tüketicilerinin zorunlu `indices` alanına yalnızca ardışık adresler
  yazılır; kaynak verinin otoritesi bu alan veya facade koleksiyonları değildir.
- ufbx poligon üçgenlemesi kaynak mesh başına bir kez; UV seam ve split normal
  korunur. Materyal bölümleri aynı nodeName ve Transform'u paylaşır.
- Düğüm adları bütün hiyerarşide benzersizleştirilir. Düğüm materyalleri okunur;
  aynı kaynak mesh'in farklı düğümlerde farklı materyali olabilir.
- Hedef sağ elli Y-up, metre. Geometric transform ve özel scale inheritance
  ufbx helper node'larıyla taşınır; mesh `geometry_to_world` kullanır.
- Normal yoksa ufbx üretir. Bütün UV setleri saklanır; FBX V yönü çevrilmez.
  Materyalin adlandırılmış UV seti birincil `uv`ye kopyalanır. Eksik set ve
  slotlar arası farklı set kullanımında uyarı basılır.
- İlk vertex renk katmanının RGB'si `Cd` içinde saklanır; otomatik materyal
  çarpanı değildir. Vertex alpha ve ek renk katmanları bu artışta taşınmaz.
- Facade'lar yalnızca mevcut tüketicilerin adaptörüdür: sahneye temsilci,
  bitki kütüphanesinin mevcut tüketicisine köşe verisinden tam facade kümesi.

## Assimp FBX envanteri ve performans

| Eski okuyucudaki işlem/alan | Yeni yol |
|---|---|
| Triangulate, smooth normals, global scale | ufbx triangulate, generate_missing_normals, hedef eksen/birim |
| JoinIdenticalVertices | Flat köşe verisi üretiminde weld yapılmıyor |
| Paralel doku prefetch ve cache | Canlı materyal slotları; en fazla 8 çalışan; GPU upload/cache yazımı seri |
| Dosya ve gömülü doku | İkisi de; cache anahtarı görüntü kimliği + TextureType |
| Materyal başına UVWSRC, ek UV setleri | Adlandırılmış FBX UV seti → mesh üzerindeki kanal; bütün setler saklanır |
| Diffuse, roughness, metallic, specular, normal, emission, opacity, transmission | ufbx PBR haritaları ve klasik FBX karşılıkları; GPU snapshot ortak yardımcıdan |
| Clearcoat, IOR | Sabit değerler taşınır |
| Tangent üretimi | Yapılmaz; mevcut backend türetir |
| Paylaşılan kaynak mesh | Üçgenleme tekrar kullanılabilir; düğümler ayrı SoA sahibi |

Gömülü görüntü kimliğinde geçersiz file_index ortak cache anahtarı yapılmaz.
Dosya bulunamaz/çözülemezse adıyla uyarı verilir. Layered/procedural texture,
texture UV transform, bump/displacement/AO haritaları uyarılı eksiktir. Classic
specular renk haritası, transparency renk haritası ve anisotropy ayrıca kabul
edilmemiştir. Kamera/ışıklar bu artışta uyarıyla atlanır.

**Skin/blend/cache deformer veya animasyon eğrisi taşıyan FBX açık hatayla
reddedilir**, sessizce statik karakter üretilmez. Bu kontrol materyal kaydından
önce yapılır. `loadSkinning=false` bu sınırı aşmaz; sonraki artış beklenir.

## Açık işler ve doğrulama sınırı

§7, §8 ve §9(a) kullanıcı tarafından doğrulandı. §9(b) sayım hatası değildir:
7 glTF mesh'i tekrar kullanılıyor, fazladan yaklaşık 4.39M üçgen kopyası var.
Eşik ve açık import seçeneğiyle InstanceGroup dönüşümü ayrı artış olarak duruyor.

Bu partide `AssetRegistry` FBX/OBJ metadata taraması hâlâ Assimp kullanır.
AssimpLoader, Texture aiTexture kurucusu ve kütüphane bağlantısı henüz sökülmedi.
Yeni `.cpp` dosyaları vcxproj'a eklendi; CMake Import klasörünü artık tarar.
ufbx.c tek UfbxVendor.cpp çeviri biriminde C++ olarak derlenir; ayrıca projeye
ufbx.c eklenmemeli. Yeni modüllerde MSVC exception unwinding açıktır.

Statik kontrol: IPC yetki/descriptor audit başarılı; proje XML kayıtları ve
include yolları denetlendi. **Kullanıcı build aldı ve ufbx import logunu paylaştı;
görsel doğrulama henüz ayrıca raporlanmadı.** Sıralı
kabul adımları [NEXT_BUILD_CHECKS.md](NEXT_BUILD_CHECKS.md)'de.
