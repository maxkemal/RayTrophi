# SORM — İlerleme, karar ve sonuç günlüğü

Bu dosya bütün ajanların ortak, kronolojik araştırma kaydıdır. Tamamlanmış
kayıtlar silinmez; düzeltmeler önceki kayda atıfla eklenir. Ham büyük veriler burada
tutulmaz; kalıcı çıktı konumu ve yeniden üretim bilgisi bağlanır.

## Kayıt şablonu

- Kayıt kimliği / tarih / ajan:
- Aşama ve durum: öneri / karar / kaynak hazır / çalıştırıldı / başarısız /
  iptal / engelli / doğrulandı; hangi kapsamın doğrulandığını belirt.
- Soru ve yapılan ilerleme:
- Model/protokol/çıktı sürümü, kaynak revizyonu ve dirty bilgisi:
- Parametreler, tohumlar, ortam ve tam komut veya manifest/çıktı bağlantısı:
- Beklenen kabul ölçütü ve gözlenen sayısal/görsel sonuç:
- Çalıştırılmayan kontroller, sınırlılıklar ve başarısızlık nedeni:
- Yorum ve karar: gözlemden ayrı yaz.
- Güncellenen ilgili belgeler:
- Sonraki adım, açık karar ve devralacak ajan için gerekli bilgi:

Uygulanmayan alanlar nedenleriyle “uygulanmaz” yazılır. “Doğrulandı” durumu bütün
hipotezin doğrulandığı anlamına gelmez; örneğin yalnız metin bağlantıları veya
tek bir sayısal kabul doğrulanmış olabilir.

## SORM-DOC-001 — 11.11.2026 — Codex

- **Durum:** Dokümantasyon ve yerel kaynak incelemesi; kod/deney yok.
- **İlerleme:** v0.1 araştırma paketi okundu; v0.2'de matematiksel belirsizlikler,
  M1 donmuş-iz kontrolü, yanlış pozitifler, CPU M0 ve dosya bazlı entegrasyon
  planı işlendi. v0.3'te kullanıcı yönlendirmesiyle erken K1-G görsel laboratuvar
  kapsamı eklendi. Bu kayıt önceki sohbet çalışmalarının geriye dönük özetidir.
- **Kanıt/kapsam:** Yerel compute, API/IPC, headless ve billboard kaynakları
  incelendi. Markdown sürüm/bağlantı/metin kontrolleri yapıldı. Sayısal sonuç yok.
- **Sınırlılık:** Build, shader, canlı uygulama, GPU/CPU eşdeğerliği ve bilimsel
  deney çalıştırılmadı. Matematik parametreleri henüz tamamlanmadı.
- **Belgeler:** `README.md`, `01_HIPOTEZ.md`, `02_MATEMATIK_MODELLER.md`,
  `03_DENEY_YOL_HARITASI.md`, `04_RAYTROPHI_ENTEGRASYON.md`.
- **Sonraki adım:** M0 sayısal sözleşmesi ve tek M1 adayını kesinleştir.

## SORM-DOC-002 — 11.11.2026 — Codex

- **Durum:** Yöntem önerileri ve ajan çalışma kuralları belgelendi; kod yok.
- **İlerleme:** Altı ayrıntılandırma adımı, düğüm/kenar izi seçimi, matematiksel
  açık sorular, metrik adayları ve C++ çekirdek/Python deney ayrımı kaydedildi.
- **Karar:** Her ajan her anlamlı ilerlemeyi ve olumlu/olumsuz her sonucu ilgili
  belgelere ve bu günlüğe işler. Kullanıcının açık çalışma şartıdır.
- **Sürüm/parametre/komut:** Dokümantasyon ekidir; sayısal model veya deney
  manifesti oluşturulmadı. Deney parametresi ve çalıştırma komutu uygulanmaz.
- **Sınırlılık:** Önerilen M1 ve metrikler nihai seçim değildir. Bilimsel başarı
  veya uygulama teslimi iddiası yoktur.
- **Belge kabulü:** Sekiz Markdown dosyasının UTF-8, satır sonu boşlukları ve
  yerel Markdown bağlantıları statik kontrolden geçti; sayısal test yapılmadı.
- **Belgeler:** `05_METOD_KARARLARI.md`, bu günlük, `AGENTS.md`, `README.md`,
  `04_RAYTROPHI_ENTEGRASYON.md`; ortak koordinasyon panosunda çalışma kuralı.
- **Sonraki adım:** Geliştirme/pilot öncesi 05 belgesindeki karar kapılarını
  kapat; kodlamayı ayrıca yetkilendir. Çalışma devralan ajan önce paketi ve
  koordinasyon panosunu okuyup dosya sahipliğini bildirsin.

## SORM-K0-001 — 11.11.2026 — Codex

- **Durum:** M0 v1 matematik/protokol tanımı tamamlandı; kod ve deney yok.
- **Yetki:** Kullanıcı “Önce M0 sözleşmesini kesinleştir” seçimini yaptı.
- **Sürüm:** `sorm.m0.v1`, `sorm.baseline.v1`; kaynak revizyonu/derleyici ve
  çalıştırma komutu bu dokümantasyon işi için uygulanmaz, mevcut ağaçta başka
  değişiklikler de vardır. Gerçek manifest/API şeması K1'de hazırlanacak.
- **İlerleme:** c=1/6, θ=σ=0, periyodik 10³ kafes; eş bütçeli iki 8-birim giriş
  alanı; 2 darbe + 20 boşluk, 100 sunum; Heun Δs=0,01 ve iki inceltme; 0,1 ham
  kayıt; 61-örnek RMS uzaklığı ve kesin sayısal kabul eşikleri belgelendi.
- **Analitik inceleme:** Sürekli modelin sınırlılığı ve sıfır-giriş maksimum
  gevşeme sınırı gerekçelendirildi. Homojen giriş için kapalı çözüm kabul fikstürü
  tanımlandı. Bu gerekçeler ayrık çözücü kabulü değildir.
- **Beklenen sonuç:** M0 öğrenme iddiası üretmez; hızlı durum eşitlenince problar
  önceki referansla aynı olmalıdır. A/B geometrik ayrışması öğrenme değildir.
- **Statik kabul:** Dokuz Markdown dosyasının metin/yerel bağlantı kontrolleri
  ve kafes/giriş/takvim aritmetiği geçti: 3000 kenar, 8+8 giriş birimi, 984 dış
  birim, 2220 süre, 61 prob örneği. İlk basit bağlantı tarayıcısı kod bloğundaki
  toplam gösterimini bağlantı sandı; kod blokları dışlanarak kontrol düzeltildi
  ve yeniden geçti. Bu kontroller çözücü veya deney çalıştırmaz.
- **Sınırlılık:** C++/shader/build/CPU sayısal koşu/canlı görsel/IPC kabulü
  yapılmadı. Kabul eşikleri önceden seçilmiş değerlerdir, deneysel PASS değildir.
  M1 denklemi, plastisite ve işlevsel eşikler açık kalır.
- **Belgeler:** `07_M0_SOZLESMESI.md`; README, AGENTS, 02–05 ve bu günlükte
  güncel kanonik referans ve tamamlanma durumu işlendi.
- **Sonraki adım:** K1 uygulaması ayrıca yetkilendirildiğinde aynı sözleşmenin
  C++ çekirdek/API/manifest ve K1-G görünümünü oluştur; sayı değiştirmek gerekirse
  gerekçesi ve yeni sürümüyle önce kaydet. Diğer ajanlar 07'yi okuyarak devralır.
