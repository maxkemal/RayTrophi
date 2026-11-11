# SORM araştırma paketi v0.3 — 11.11.2026

Okuma sırası:

1. [Hipotez ve bilimsel sınırlar](01_HIPOTEZ.md)
2. [Matematiksel adaylar ve açık kararlar](02_MATEMATIK_MODELLER.md)
3. [Deney protokolü ve kabul kapıları](03_DENEY_YOL_HARITASI.md)
4. [Kaynakta doğrulanan mimari ve uygulama planı](04_RAYTROPHI_ENTEGRASYON.md)
5. [Yöntem önerileri ve kodlama öncesi kararlar](05_METOD_KARARLARI.md)
6. [Ortak ilerleme ve sonuç günlüğü](06_ARASTIRMA_GUNLUGU.md)
7. [Kesin M0 v1 sayısal ve deney sözleşmesi](07_M0_SOZLESMESI.md)

**Güncel durum — 11.11.2026:** M0 matematik/protokol tanımı `07` belgesinde
tamamlandı. M0 için oradaki kesin seçimler önceki açık adayların yerini alır.
K0'ın yalnız M0 tanım kapısı kapandı; uygulama/API şeması ve K1 sayısal/görsel
kabulü henüz yapılmadı. M1 ve öğrenme protokolü hâlâ açık, kodlama başlamadı.

**Tüm ajanlar için zorunlu:** Her anlamlı ilerleme ve olumlu/olumsuz her sonuç,
ilgili araştırma belgesine ve 06 günlüğüne mutlaka işlenir. Kayıt yapılmadan iş
tamamlanmış sayılmaz. [Ajan çalışma kuralları](AGENTS.md) ve ortak
[koordinasyon panosu](../AJAN_KOORDINASYON.md) birlikte kullanılır. Başka ajanlar
dosya sahipliğini bildirerek çalışmayı devralabilir; mevcut kodlama sınırı korunur.

v0.3, yerel kaynak incelemesini ve RayTrophi’nin erken görsel laboratuvar rolünü işler. Bu revizyon yalnız dokümantasyondur; deney çalıştırıldığı, bilimsel başarı elde
edildiği veya uygulama kodunun hazır olduğu anlamına gelmez. Kodlama henüz başlamamıştır.

Temel ilke: başlangıçta parçacıklara bilgi kimliği, sembol uzmanlığı veya hedef grup
atanmaz. Aynı yerel yasaların, farklı etkileşim geçmişleriyle işlevsel farklılaşma
üretip üretmediği sınanır.

İlk uygulama kapsamı K0/K1: CPU M0, 1.000 birim, güvenilir ham kayıt, ortak
Python/IPC servisi, render gerektirmeyen batch yolu ve CPU M0 ile birlikte
sahnede canlı etkinlik/komşuluk gözlemi (K1-G). M1'in kesin denklemleri,
parametreleri ve işlevsel doğrulama eşikleri açık kararlardır. M0 parametre ve
sayısal eşikleri `07` belgesinde sabittir. M2/M3, GPU compute ve gelişmiş görsel
analiz sonraki aşamalardır.
Viewport GPU çizimi CPU hesaplanan durumu gösterebilir; GPU çözücü gerekmez. Telemetri ve API yüzeyleri ilk teslimin parçasıdır.

Önce model/protokol sözleşmesi kesinleştirilir, ardından ayrıca yetkilendirilen
kodlama yapılır. Mevcut Matter/DEM/MPM/FLIP sistemleri yeniden tasarlanmaz.
