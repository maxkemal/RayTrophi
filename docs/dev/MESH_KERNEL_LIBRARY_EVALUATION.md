# Mesh Kernel Library Evaluation

Bevel, inset, robust edge selection ve corner join işlemleri mevcut deneysel
facade yolunda production kalitesine ulaşmadı. Bu nedenle bevel UI'dan
çıkarıldı; mevcut yol yalnızca araştırma/prototip olarak tutuluyor.

Değerlendirme ölçütleri:

- half-edge veya eşdeğer topology ownership;
- multi-edge / multi-face bevel ve köşe join kalitesi;
- winding, self-intersection ve degenerate cleanup;
- flat `TriangleMesh` / DNA SoA publisher adaptörü;
- deterministic CPU fallback ve gerektiğinde GPU preview;
- undo snapshot, cancellation ve büyük mesh davranışı;
- lisans ve statik/dinamik dağıtım koşulları;
- Python/IPC/UI'nin aynı operation service'i kullanabilmesi.

Kütüphane seçilse bile sahne geometrisinin sahibi flat SoA kalacak. Kütüphane
yalnızca topology operation kernel olacak; sonuç doğrulama, publish, undo ve
renderer refresh `FlatMeshEditService` üzerinden yapılacak.

İlk teknik aday sınıflandırması:

- **Manifold**: özellikle manifold/topologically robust boolean vertical slice için
  ilk aday. Girdi/çıktı face provenance (`originalID` / `faceID`) aktarımı da
  değerlendirilecek.
- **CGAL Polygon Mesh Processing**: repair, orientation, self-intersection,
  corefinement ve mesh processing kapsamı güçlü; ancak bileşen lisansı GPL veya
  ticari lisans kararını erkenden gerektiriyor.
- **libigl**: MPL2 lisanslı, hafif yardımcı algoritmalar için uygun; tek başına
  DCC seviyesinde topology ownership ve bevel kernel olarak varsayılmayacak.
- **OpenMesh / PMP**: half-edge ve polygon processing tabanı olarak incelenecek;
  production bevel kalitesi ayrıca doğrulanmadan seçilmeyecek.

Bu liste seçim değildir; önce küçük bir adapter spike ve kabul testleri ile
ölçülecek.

Bevel yeniden açılmadan önce minimum kabul testi:

1. tek edge, bağlı iki face;
2. aynı yüzde çoklu edge;
3. komşu yüzlerde çoklu edge;
4. köşe ve boundary yakınındaki seçimler;
5. convex/concave ve ters winding mesh;
6. undo/redo ile tam vertex/index/attribute snapshot eşitliği.
