# Mesh Paint Tablet Support Plan

## Amaç

Bu not, RayTrophi icin mesh paint ile baslayacak tablet/kalem destegi calismasinin teknik yol haritasini toplar. Hedef once dusuk riskli bir `mesh paint` entegrasyonu yapmak, sonra ayni input omurgasini `sculpt` icin yeniden kullanmaktir.

## Repo Baglami

- Uygulama `SDL2 + ImGui` kullaniyor.
- Ana event dongusu `raytrac_sdl2/source/src/Core/Main.cpp` icinde `SDL_PollEvent()` ve `ImGui_ImplSDL2_ProcessEvent()` ile akiyor.
- Mesh paint stroke akisi `raytrac_sdl2/source/src/UI/scene_ui_modifiers.cpp` icinde su an dogrudan `ImGui::IsMouseDown()` ve `ImGui::IsMouseClicked()` uzerinden kontrol ediliyor.
- Sculpt stroke akisi de benzer sekilde `raytrac_sdl2/source/src/UI/scene_ui_mesh_overlay.cpp` icinde sol fare durumuna bagli.
- `PaintStrokeContext` su anda sadece `layer_index` ve `dt` tasiyor; pressure, tilt, eraser, pointer type gibi tablet verileri icin alan yok.

## Kritik Bulgular

### 1. SDL2 tarafinda hazir pen API beklememek lazim

Yapilan kontrolun sonucu:

- Repo, `imgui_impl_sdl2` backend kullaniyor.
- SDL3 dokumantasyonunda acik bir `CategoryPen` API yuzeyi var.
- SDL2 dokumantasyonunda buna denk dusen yerlesik bir pen/stylus kategori yuzeyi gorunmuyor.

Bu, SDL2 uzerinden hic pen verisi alinmaz anlamina gelmez. Ancak sunu ifade eder:

- `SDL2 + ImGui` akisi tek basina pressure/tilt/eraser icin yeterli kabul edilmemeli.
- En olasi senaryo, SDL2 tarafinda kalemin bir bolumunun `mouse` gibi gorunmesi ve ileri seviye tablet verilerinin Windows native katmanindan alinmasidir.

### 2. Bu projede asil sorun brush algoritmasi degil, input omurgasi

Tablet destegi icin ilk degistirilmesi gereken yer `MeshPaintAdapter.cpp` degil. Esas kontrol noktasi su katmanlarda:

- `Main.cpp`: native event toplama
- `scene_ui_modifiers.cpp`: mesh paint stroke baslatma/bitirme
- `scene_ui_mesh_overlay.cpp`: sculpt stroke baslatma/bitirme
- `IPaintSurfaceAdapter.h`: stroke context veri modeli

## SDL2 Icindeki Olasiliklar

### Olasilik A: SDL2 zaten fare emulasyonu veriyordur

Bu durumda:

- kalem hareketi `mouse motion` gibi gelir,
- uc basisi `left mouse` gibi gorunur,
- ama pressure/tilt/eraser buyuk olasilikla kaybolur.

Bu senaryo minimum uyumluluk saglar ama gercek tablet destegi degildir.

### Olasilik B: SDL2 uzerinden platform-native pencere bilgisi alip Windows mesajlarini dinlemek gerekir

Bu proje Windows odakli calistigi icin en gercekci yol budur.

Muhtemel yaklasim:

- SDL penceresinden `HWND` elde etmek
- pencere prosedurune `WM_POINTER` tabanli bir katman eklemek veya subclass etmek
- pressure, tilt, rotation, eraser bilgisini native taraftan almak
- bu veriyi uygulama icinde ortak bir `TabletInputState` yapisina yazmak

Bu yol, `SDL2 + ImGui` yapisini bozmadan tablet destegi eklemenin en dusuk riskli yoludur.

## Onerilen Mimari

## Faz 1: Ortak input state

Yeni bir ortak veri modeli eklenmeli.

Ornek alanlar:

- `active`
- `pointer_type`
- `pointer_id`
- `screen_pos`
- `pressure`
- `tilt_x`
- `tilt_y`
- `rotation`
- `eraser`
- `buttons`

Bu state her frame guncellenmeli ve UI/brush kodu dogrudan `ImGui::IsMouseDown()` yerine bu state ile calismaya baslamali.

Not:

- Fare fallback'i korunmali.
- Tablet verisi yoksa sistem bugunku davranisa geri donmeli.

## Faz 2: Stroke context genisletme

`PaintStrokeContext` genisletilmeli.

Eklenmesi mantikli alanlar:

- `input_pressure`
- `input_tilt_x`
- `input_tilt_y`
- `input_rotation`
- `input_is_eraser`
- `input_source`

Bu alanlar ilk surumde tamamen kullanilmasa bile context icinde yer almasi ileri asamalari kolaylastirir.

## Faz 3: Mesh paint MVP

Ilk uygulama sadece `mesh paint` uzerinde yapilmali.

Ilk baglanacak eksenler:

- `pressure -> strength`
- `pressure -> radius`

Bu mapping dogrudan kullanici ayarini degistirerek degil, her dab icin `effective brush` ureterek yapilmali.

Yani:

- kullanici panelindeki temel brush ayari korunur
- frame/dab bazinda gecici bir efektif brush hesaplanir
- undo, layer, clone, wet paint akisi minimum etkilenir

## Faz 4: Preview senkronu

Brush preview ile gercek dab ayni effective radius/strength mantigini gormeli.

Aksi halde:

- ekranda gorulen cap ile vurulan cap farkli olur
- kullanici tablet destegini bozuk hisseder

## Faz 5: Sculpt aktarimi

Mesh paint stabil olduktan sonra ayni input omurgasi sculpt tarafina tasinmali.

Ama sculpt icin ilk surum de yine sinirli olmali:

- once `pressure -> strength`
- sonra gerekirse `pressure -> radius`
- daha sonra araca ozel ince ayarlar

Ozellikle `Grab`, `Clay`, `Layer`, `Smooth` gibi araclar pressure degisimine daha hassas davranir.

## Uygulama Sirasi

1. `Main.cpp` tarafinda native tablet verisi icin giris noktasi ac.
2. Ortak `TabletInputState` ekle.
3. `scene_ui_modifiers.cpp` icinde mesh paint stroke kontrolunu bu state ile calistir.
4. `PaintStrokeContext` alanlarini genislet.
5. Mesh paint icin effective brush mantigi ekle.
6. Preview'i effective brush ile uyumlu hale getir.
7. Ancak bundan sonra sculpt tarafina gec.

## Neden Once Mesh Paint?

Mesh paint daha kolay baslangic noktasi cunku:

- geometriyi degistirmiyor
- CPU/GPU mesh sync riski sculpt kadar yuksek degil
- pressure etkisi kullaniciya hizli geri bildirim veriyor
- undo ve stroke mantigi daha izole test edilebilir

Sculpt ise ek olarak su zorluklari getirir:

- geometri mutasyonu
- hit fallback davranislari
- mirror etkileri
- grab gibi plane tabanli hareketler
- CPU/GPU sync ve viewport guncelleme maliyeti

## Riskler

### 1. ImGui SDL backend tek basina yetmeyebilir

`ImGui_ImplSDL2_ProcessEvent()` mevcut haliyle klasik mouse/keyboard akisini iyi tasir, ama tabletin gelismis eksenlerini uygulamaya sokmak icin yeterli olmayabilir.

### 2. SDL2 surumune guvenip yanlis varsayim kurmamak gerek

Repodaki backend dosyalari SDL2 ile derleniyor, fakat bu tek basina stylus pressure destegi oldugu anlamina gelmez. Bu konu native test ile dogrulanmali.

### 3. Native pencere hook'u dikkat ister

`HWND` uzerinden mesaj dinleme veya subclass etme yapilacaksa:

- orijinal pencere proseduru korunmali
- ImGui ve SDL event akisi bozulmamali
- focus/capture davranislarina dikkat edilmeli

## Ilk Dogrulama Listesi

Kod yazmadan once su uc sey kucuk bir spike ile test edilmeli:

1. SDL2 event akisinda kalem sadece mouse gibi mi gorunuyor?
2. SDL penceresinden `HWND` guvenli sekilde elde edilebiliyor mu?
3. `WM_POINTER` uzerinden pressure degeri duzgun okunuyor mu?

Bu uc sorunun cevabi netlesmeden tam entegrasyona girilmemeli.

## Ilk MVP Kapsami

Ilk hedef asagidaki kadar dar tutulmali:

- sadece Windows
- sadece mesh paint
- sadece pressure
- pressure sadece strength ve opsiyonel radius etkilesin
- mouse fallback bozulmasin

Tilt, eraser, rotation ve brush-a-ozel davranislar ikinci fazda acilabilir.

## Sonuc

Bu projede tablet destegi mumkun ve mantikli ilk hedef `mesh paint`. Ancak SDL2 kullaniyoruz diye stylus pressure destegi otomatik var varsayimi yapilmamali. Daha guvenli varsayim sudur:

- SDL2 mevcut haliyle temel pointer/mouse uyumlulugu saglayabilir
- gercek tablet verisi icin Windows native pointer katmani gerekecektir
- en dusuk riskli uygulama, once ortak input state acip sonra mesh paint'e pressure tabanli effective brush eklemektir

Bir sonraki adimda bu notu uygulama gorev listesine donusturmek mantikli olur.