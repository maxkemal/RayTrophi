# Revizyon 23 uyandırma incelemesi

Kullanıcı gözlemi: yığın ~90 karede görünürde oturuyor; üstte yuvarlanan ince
katmanın enerjisi, alttaki sıkışmış yığını hareket ettirmeye yetmeyebilir.
Bu test turunda shader değişmedi; canlı ölçümler revision 23'e ait.

## Kaynakta doğrulanan kapılar

`sim_matter_grain.glsl` tane uyandırır; bir hash hücresindeki bütün taneleri tek
hücre flag'iyle uyandırmaz. Ancak temas zincirinde dolaylı wake yayılması mümkün:

1. `g_fast && linked` → `wakeNeighbour(j)`. Kaynak tanenin hızı/açısal yüzey hızı
   uyku eşiğini aşması yeterli. Hedefin kütlesi, aktarılabilen itki veya destek
   kapasitesi bu kararda yok.
2. `contact()` ve `bridge()` göreli hızdan `g_dynamic_contact` oluşturur. Contact
   kapısı sönüm, Coulomb limiti, rolling/twist tork hesabından önce çalışır.
   Tam spin kullanılır; bu, gerçekten aktarılabilen torkla aynı büyüklük değildir.
3. `balanced = can_sleep && !g_dynamic_contact && grainSleepBalanced(...)`.
   Ardından `!balanced && !g_fast` bütün erişilebilir komşulara wake bırakır.
   Dolayısıyla gerçek net kuvvet/tork dengesi geçse bile sırf göreli hareket,
   "dengesiz destek" yayılması yolunu çalıştırabilir. Bu iki neden ayrılmalı.

## Sonraki düzeltmenin fizik sözleşmesi

- Gerçek kuvvet/tork dengesizliği ile hareketli temas veto'su ayrı durum olmalı.
  Kuvvet dengesizliği yayılması, yalnız gerçek denge testi başarısızsa çalışmalı.
- Enerji/itki kararı sönüm, Coulomb sürtünmesi, rolling/twist ve köprü kuvveti
  sonucundan türemeli. Statik normal ön yükün tamamı "yeni aktarım" sayılamaz.
- Hedefin kütlesi ve ataleti yanında destek yönleri, temas ön yükü ve kayma/dönme
  rezervi dikkate alınmalı. Tek büyük enerji sabiti yüzeydeki gevşek taneyi
  yanlış kilitleyebilir; tek anlık enerji testi biriken yavaş yükü kaçırabilir.
- Yalnız komşuları sabit varsayan yerel sertlik yeterli değil: zemine bağlı
  sıkışmış yığın ile birlikte hareket edebilen serbest kümeyi ayırmak gerekir.
- Düşük enerjili teğet hareket, kuvvetleri dengeli sıkışmış hedefte yalnız gerekli
  denetimi tetikleyebilmeli; koşulsuz rest sıfırlama ve ikinci halka wake gerekmemeli.
- Güçlü darbe, destek kaybı, yavaş yük artışı ve köprü değişimi fiziksel tepkiyi
  uyandırmalı. Mevcut <=20 ms / son-substep denetim ve sayaç doğruluğu korunmalı.
- `contact()` yay ilerlemesinde `pc.step_contact.x` (tek substep dt) kullanır;
  uyuyan hedef ise aradaki adımları atlayıp periyodik audit yapar. Göreli hareket
  sıfırsa bu fark etkisizdir. Hareketli komşunun düşük aktarımı daha uzun süre
  tolere edilecekse, audit aralığındaki gerçek tangential/rolling ilerleme ve
  biriken yavaş yük de hesaba katılmalı; sadece audit sayacını düzeltmek yetmez.
- Başka tanenin in-place tangential/rolling history'si okunmamalı: mevcut tek
  bankalı history sahip tarafından güncelleniyor. Destek sertifikası kullanılacaksa
  okunan durum ayrı ve kararlı olmalı; aktarılan küçük darbeler de birikmeli.

## Kabul örnekleri

1. Oturmuş yığının üstüne düşük enerjili birkaç tane: alttaki sıkışmış çekirdeğin
   uyku oranı korunmalı; açık/kapalı makro deplasman, enerji ve açı paritesi geçmeli.
2. Aynı yığın, güçlü darbe: temas alanı ve gerekli çevresi uyanmalı, hareket ve
   momentum bilançosu kapalı referansla eşleşmeli.
3. Yavaş yük rampası + destek kaldırma: düşük anlık hız kalıcı kilit yaratmamalı.
4. Static/sliding, wet/bridge, motion ve dört kollu repose kapıları korunmalı.
5. 1.3M: sleeping oranı, gerçek GPU kernel ve duvar süresi birlikte ölçülmeli.
   Sadece daha fazla sleeping veya daha az wake başarı sayılmaz.

Bu sözleşme uygulanmış bir enerji modeli değildir; ölçüm sonrası kaynak düzeltmesi
ve kullanıcı build'i için doğrulanan sorun/korunacak koşullar kaydıdır.
