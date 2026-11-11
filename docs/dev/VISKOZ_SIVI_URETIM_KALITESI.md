# Viskoz sıvı — üretim kalitesi değerlendirmesi ve ölçüm planı

> **Durum:** AKTIF — 2026-11-09: madde 1–2 CANLI, Huppert 5/5 PASS (2 mm: 1.06/1.10, ızgara yakınsıyor); sıradaki: collider alt-kare, yüzey gerilimi, Ribe kıvrılması.

## Neden

Kullanıcı: "görsel olarak doğru görünmesi fizik olarak doğru olduğu anlamına gelmiyor".
Önce gerçek boyut ve yoğunlukta, ANALİTİK referansı olan bir kurulumla ölçüyoruz.

## Ölçüm 1 — Huppert (1982) viskoz yerçekimi akıntısı (2026-11-09)

Düz zemine bırakılan viskoz kütlenin yarıçapı r(t) = 0.894 (ρ g V³ t / 3μ)^(1/8).
Kurulum: Honey (ρ 1420, ν 0.007 → μ 9.94 Pa·s), V = 48 ml (4×3×4 cm kutu), h = 2 mm,
24×4×24 cm kapalı alan, 8 ppc (56k parsel). r = (p95−p05)/2/0.9 iki eksen ortalaması.
Probe: scratchpad `probe_huppert.py` (betiğe dönüşecek).

| t (s) | Huppert | 16 GS adımı (varsayılan) | 64 GS adımı |
|---|---|---|---|
| 1 | 4.63 cm | 6.05 (1.31×) | 5.36 (1.16×) |
| 4 | 5.50 | 7.90 (1.44×) | 6.72 (1.22×) |
| 10 | 6.17 | 9.29 (1.51×) | 7.77 (1.26×) |
| üs (1→10 s) | 0.125 | 0.186 | 0.161 |

r ∝ μ^(−1/8) → etkin viskozite 16 adımda gerçeğin ~1/9 (1 s) – 1/25'i (10 s), 64 adımda
~1/3 – 1/6. Adım sayısı sonucu değiştiriyor = **çözücü yakınsamıyor**. Süre ~58 ms/kare
(288k hücre).

Kök (kod): `sim_fluid_viscosity.glsl` sabit sayıda kırmızı-siyah Gauss-Seidel adımı,
artık (residual) kontrolü yok; a = ν dt / h² burada kare başına ~70 → GS'nin düşük
frekanslı (kütle sarkması) modları yüzlerce adım ister. Kendi yorumu: "az adım viskoziteyi
eksik uygular, asla patlamaz" — tam olarak sessizce makul görünen sonuç.

## Üretim kalitesi için eksikler (öncelik sırası)

1. **Viskozite çözücüsü yakınsamalı:** sabit GS yerine artık toleranslı PCG / MG
   (basınç için MGPCG zaten var). Kabul: Huppert yarıçapı ±%5, üs 0.125 ± 0.01.
2. **Tam gerilme formu:** şimdi bileşen başına ayrık ν∇²u. Üretim çözücüleri (Batty &
   Bridson 2008; Larionov ve ark. 2017) ∇·(μ(∇u+∇uᵀ)) ve serbest yüzeyde doğru çekme
   koşulu kullanır; bal ipinin kıvrılması (rope coiling) ve bükülme bunsuz çıkmaz.
   Kabul: Ribe (2004) kıvrılma frekansı.
3. **Collider teması:** ikili voksel + kare başına bir kez; yapışma bandı ~1 voksel/kare'ye
   kadar tutuyor. Alt-kare collider hareketi + φ yüz teması (MPM'deki gibi).
4. **Yüzey gerilimi:** sıvı çözücüde yok (yalnız tane/madde alanında). Damla kopması,
   ince iplik ve ıslatma için gerekli. Kabul: Rayleigh–Plateau / Laplace basıncı.
5. **Çözünürlük ölçeği:** üretim bal çekimi h 0.5–1 mm; sparse yol var, ölçülmeli.

## Sıradaki ölçümler (referanslı)

- Landau–Levich–Derjaguin film kalınlığı (plaka çekme, viskoz–yerçekimi rejimi).
- Stokes düşüşü (küre, Re ≪ 1): v = 2Δρ g R² / 9μ.
- Ribe ip kıvrılması (madde 2'den sonra).

## Madde 1 — yazıldı (2026-11-09, derlenmedi)

`sim_fluid_viscosity_pcg.glsl` (+ seyrek iki ikiz), host `solveViscosityPcg`
(`SparseViscosityGpu.cpp`). Şerit uzayı: lane = bileşen·M + yerel; x hızın kendisi (MAC
deposu). Modlar: 0 başlat (r = b − Ax, d, p = z), 1 q = Ap, 2 x += αp / r −= αq, 3 p = z + βp,
4 tek grup indirgeme → skalerler (α, β cihazda). Tolerans 1e-4 (önkoşullu norm), 500
iterasyon. Kabul testi `scripts/test/rt_test_viscous_huppert_ipc.py` (±%10, üs 0.09–0.16).
CPU referansı (`applyViscosity`) hâlâ sabit GS: yalnız GPU yolu düzeldi.

## Canlı ölçüm (2026-11-09, exe 10:50 + PCG .spv)

**Ölçüm aracı düzeltmesi:** parseller hacim taşır; Huppert profilinin x izdüşümünde %95
nicelik 0.7525 R'dedir (düzgün diskte 0.8055). İlk tabloda 0.9 kullanıldı → oranlar ~%16
düşük okundu. Aşağıdakiler 0.7525 ile (aynı 2 mm, 19×3.6×19 cm alan).

| Koşu (h 2 mm, bal) | r/Huppert 1 s | 4 s |
|---|---|---|
| PCG (yakınsıyor: ≤72 iter, unconverged 0) | 1.27 | 1.34 |
| + FLIP karışımı 0 (saf APIC) | 1.27 | 1.35 |
| + tüm sönümlemeler kapalı | 1.27 | 1.35 |
| **+ yoğunluk düzeltmesi kapalı** (`solver_density_relaxation_s` 0) | **1.21** | **1.13** |

Izgara yakınsama (eski 0.9 tahminciyle, göreli): 3 mm 1.26 → 2 mm 1.12 → 1.5 mm 1.13 (4 s):
2 mm'den sonra çözünürlük sınırlamıyor. ν×10 / ν/10 ile oranlar ~aynı (~1.3): sapma
viskoziteden BAĞIMSIZ bir mekanizma. Duvar hayalet ağırlığı (2a, yarım hücre) düzeltildi;
etkisi ölçülemeyecek kadar küçük (1.121 → 1.118) ama doğru.

**Kök:** yoğunluk düzeltmesi (parselleri hücre başına 8'e çeken konum projeksiyonu) ince
filmde kütleyi yanal yayıyor; r⁸ ∝ t/μ olduğundan 4 s'deki 1.34/1.13 farkı etkin
viskozitede ~4×'e karşılık gelir. Kapalıyken geç zaman üssü yasadan DÜŞÜK (oran 1.21 →
1.13 iniyor): 2 hücrelik filmde kayma fazla dirençli olabilir, ya da başlangıç bloğunun
sanal zaman kökeni. Sıradaki iş: düzeltmenin serbest yüzey hücrelerinde (kısmi dolu) kütle
taşımasını engellemek; kabul: Huppert ±%10 yoğunluk düzeltmesi AÇIKken.

Maliyet: 288k hücre / 56k parsel, 96 kare ~10 s (PCG ≤72 iter).

### ★ DÜZELTME (2026-11-09, temiz τ taraması): yoğunluk düzeltmesi kök DEĞİL

Önceki "kök: yoğunluk düzeltmesi" yanlıştı. Boş sahnede (önceki tarama kullanıcı sahne
değiştirirken koştu, geçersiz):

| τ | r/Huppert 1 s | 2 s | 4 s | p95 y (4 s) |
|---|---|---|---|---|
| 0.05 (bal) | 1.27 | 1.30 | 1.34 | 3.3 mm |
| 0.2 | 1.26 | 1.30 | 1.33 | 3.7 mm |
| 1.0 | 1.23 | 1.27 | 1.28 | 3.4 mm |
| 0 (kapalı) | 1.21 | 1.20 | 1.13 | **1.9 mm** |

Kapalıyken birikinti yarı yarıya inceliyor = HACİM KAYBI (parseller sıkışıyor); küçük
yarıçap kaybolan hacimden. Düzeltme hacmi koruyor, dokunulmamalı. ★ Ders: bir ablasyonda
"daha iyi" görünen oranı, korunması gereken ikinci büyüklükle (hacim) birlikte oku.

Kalan aday (en güçlü): **birinci derece serbest yüzey.** Hücre, parsel varsa tam sıvı
sayılıyor; 2–3 hücrelik filmde kısmi dolu üst hücre filmi h/2'ye kadar kalın gösterir, akı
∝ H³ → ~2× akı → r ×2^(1/8) ≈ 1.09, üstüne yüzeydeki yarım-havada yüzler. Üretim çözümü
madde 2 ile aynı yer: Batty & Bridson (2008) varyasyonel viskozite — hücre/yüz SIVI HACİM
KESİRLERİ ve tam gerilme formu. Kabul: Huppert ±%10, hacim korunumu ±%3 (p95 y ile
birlikte raporlanır).

## Madde 2 — varyasyonel viskozite yazıldı (2026-11-09, C++ derlemesi bekliyor)

`sim_fluid_viscosity_pcg.glsl` `variational()`: Batty & Bridson (2008) enerjisi, aynı PCG
çerçevesinde (bileşenler tek şerit uzayında, çapraz terimler kenarlardan). Hacim kesri =
parsel sayısı / ppc (yalnız havaya komşu sıvı hücrelerde), yüz = 2 hücre ortalaması, kenar =
4 hücre. Bilinmeyen = sıvı yüzler; katı yüz = katı hızı, hava yüzü = mevcut değer;
wall_slip katıya değen kenar terimlerini ölçekler. Push 72 B (ppc). Kontrol: rijit dönmede
iki kenar terimi de sıfır. Duvarın yarım hücre ofseti (2a) bu formda yok; ölçülen etkisi
ihmal edilebilirdi.

## Canlı (exe 11:51 tam Rebuild, 2026-11-09): Huppert 5/5 PASS

Varyasyonel ilk hali ters gitti (oran 1.19/1.37, üs 0.23; gövde kalınlaştı ama ince bir
etek kaçtı): sıvıya kenar komşusu HAVA yüzleri "bilinen" sayılıp eski hızlarıyla
bağlanıyordu. Düzeltme: bu yüzler kAirMass = 0.01 kütle ağırlığıyla bilinmeyen (gerilmesiz
uzatma; Batty'de de böyle). İkinci ölçüm hatası: tohum hücreye oturunca 4×3×4 cm kutu 2 mm'de
56.4 ml oluyor (48 değil); r ~ V^(3/8) → oranlar %6 fazla okunuyordu, test artık hacmi
parsel/ppc·h³'ten alıyor.

| h (19 cm alan, ekilen hacimle) | 1 s | 4 s | p95 y 4 s |
|---|---|---|---|
| 3 mm | 1.10 | 1.14 | 4.0 mm |
| 2 mm | 1.06 | 1.10 | 4.2 mm |
| 1.5 mm | 1.04 | 1.07 | 5.1 mm |

Test alanı (24 cm, 2 mm): 1.061 / 1.095, üs 0.148, PCG ≤120 iter, 96 kare 13.5 s.
Önceki (bileşen Laplacian, aynı hacim düzeltmesiyle) ≈ 1.20/1.27. Kalan %4–10 ızgarayla
monoton azalıyor = çözünürlük hatası. Sıvı duvar 7/7.

## Kullanıcı sahnesi: "bal çok akışkan" = bal hiç viskoz değildi (2026-11-09)

Su varsayılanlı alana Bal tohum katmanı: alan Su fiziğini koşuyordu (ν 0, duvar kayması 1,
PCG iterasyonu 0). Madde başına ν alanı yalnız malzeme bağlamalarından kuruluyordu.
Düzeltme `adoptParcelLiquidSubstance` + `appendParcelSubstanceViscosities`
(FluidDomainSubstance.cpp); rapor alanı `liquid_physics_substance`. Test kolu
`layer_runs_honey`. Karışımda duvar kayması hâlâ alan başına (açık).
