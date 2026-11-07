# Panel tasarım pilotu — Surface Core

Tarih: 2026-11-03  
Durum: UYGULANDI — kullanıcı build ve görsel doğrulaması bekleniyor.

Güncel kapsam bu ilk pilotu geçti: Surface Core yerine yedi bölümlü materyal
inspector uygulanmıştır. Güncel build/görsel kabul listesi ve animasyon sınırlamaları
için [Materyal inspector standardı](MATERIAL_INSPECTOR_STANDARD.md) esas alınmalıdır.
Aşağıdaki ilk-pilot adımları önceki dilimin kaydıdır.

## Kapsam

`PropertyLayout.h` sunum katmanında ortak özellik satırı ve hafif grup başlığı sunar.
Ölçüler font ve mevcut ImGui spacing değerlerinden türetilir. Dar alanda etiket
kontrolün üstüne geçer; normal alanda kontroller ortak başlangıçta hizalanır.

`MaterialSurfaceBasicsUI.h`, mevcut Surface Core bölümünün ilk beş kontrolünü taşır:
Base Color, Metallic, Roughness, IOR ve Transmission. Modern modda keyframe alanı
kontrolün sağında sabittir. Kısa Metal/Rough/Trans etiketleri açık adlara dönüştürüldü.
Sayısal alanlarda standart slider ve Ctrl+click ile hassas giriş vardır.
Retro LCD kaldırıldı. Mevcut DrawSmartFloat çağrıları da ortak modern satırı kullanır;
bu nedenle doğrulama artık yalnız pilotu değil bu helper'ın diğer çağıranlarını da kapsar.

Surface Core başlığı tema renklerini kullanan, çevre kutusu ve renkli sol şeridi
olmayan opt-in grup ile çizilir. Diğer section çağıranları değiştirilmedi.
Bu pilot başlığı global section süsleme tercihlerinden bağımsızdır; yayılım öncesi
bu tercihlerin yeni stile nasıl uygulanacağı değerlendirilmelidir.
Surface Core'un Dispersion ve sonrasındaki kontrolleri henüz satır standardına taşınmadı.

Mevcut materyal mutasyonları, changed bildirimi ve keyframe callback'leri korunur.
Yeni authoring işlemi, script/IPC metodu, kalıcı tercih veya veri modeli eklenmedi.
Birleşik domain, solver, Substance ve IPC dosyalarına bu dilimde dokunulmadı.
Yeni modüller header-only olduğundan ayrı linker/build kaynak kaydı gerekmez.

## Kullanıcı build ve canlı kontrol listesi

1. Normal proje build'ini çalıştır. Codex build veya shader derlemesi çalıştırmadı.
2. Bir mesh seç, Principled material editöründe Surface Core'u aç.
3. Modern slider modunda beş temel alanın hizasını ve sağ keyframe alanını kontrol et.
4. Dock'u yaklaşık 240/320/440 logical px genişliklerde dene. Dar alanda etiket üstte
   olmalı; renk alanı, slider ve keyframe düğmesi birbirini örtmemeli.
5. Desteklenen UI ölçeği/DPI değerlerinde, normal ve custom temada kontrol et.
6. Beş değeri değiştir; render/material güncellemesinin önceki gibi geldiğini doğrula.
   Sayısal alana Ctrl+click ile gir; normal slider davranışı korunmalı.
7. Beş alanın keyframe'ini ekle/kaldır; frame değiştir ve doğru alanların
   animasyonlandığını doğrula. Seçili nesne/material değişince yanlış hedefe yazılmamalı.
8. Renk popup'ını aç/kapat; seçim, klavye odağı ve viewport mouse capture'ını kontrol et.
9. Theme ayarlarında LCD Widget Theme/Slider Style/Retro palet seçeneklerinin
   kaldırıldığını doğrula. World ve Camera'da DrawSmartFloat kullanan birkaç alanı
   dar/geniş panel, sayısal giriş ve keyframe ekleme/kaldırmayla kontrol et.
10. Surface Core'u daralt/aç; alttaki Dispersion, Interior, Bubble, emission ve
    Subsurface gruplarının hâlâ erişilebilir ve çalışır olduğunu kontrol et.
11. Aynı açık panel düzeninde viewport alanını karşılaştır; yeni kalıcı alan ayrılmamalı.

## Sonraki dilim

Pilotun görsel kabulünden sonra, aynı bölümdeki ileri kontrolleri küçük gruplara
ayırıp ortak satırlara taşımak. Diğer disiplinlere yayılım bundan sonra yapılır.
Sahne içi overlay kontrolü bu pilotun kapsamına dahil değildir.

## İkinci dilim ve bildirilen hata

Retro slider implementasyonu, LCDTheme ve global stil state'i kaldırıldı.
Ortak slider ve tema renkli, klavyeyle odaklanabilen keyframe düğmesi
`PropertyLayout.h` içinde tutulur. `scene_ui.cpp` yalnız ince çağrı köprüsü içerir.
Material pilotu aynı helper'ı kullanır; ikinci bir slider implementasyonu yoktur.

İlk çıkarımda `transmission` değişkeni yeni modüle taşınmış, fakat Surface Core'un
devamındaki stone/resin/glass koşulları eski yerel değişkene bağlı kalmıştı.
Bildirilen C2065 için temel alanlar çizildikten sonra materyalin güncel transmission
değeri yerel olarak okunur. Transmission'ı 0.5'in iki tarafında değiştirerek ilgili
interior gruplarının aynı frame'de doğru geçiş yaptığını da doğrula.
