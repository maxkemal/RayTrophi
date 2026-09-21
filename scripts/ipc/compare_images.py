"""Iki goruntu arasindaki AYDINLATMA acigini olcer.

    python scripts/ipc/compare_images.py raster.jpg rt.png [tekrar.jpg]

Ucuncu dosya verilirse TEKRAR KOLU olarak kullanilir: ilk goruntuyle arasindaki
fark olcumun taban gurultusudur, ve bundan kucuk her sinyal OKUNAMAZ. 2026-09-10
olcumunde taban gurultusu 0,00012, sinyal 0,04474 idi (369x).

★★★ Neden bu olcutler. "Duz aydinlatma" ve "sert golge" izlenimdir; asagidaki
    ikisi onlari sayiya cevirir:

    YAKIN/UZAK orani  - sabit bir ambient, sik kanopi altindaki pikselle acik
      zemindeki pikseli AYNI gokyuzu isigiyla aydinlatir, o yuzden mesafeyle
      artan kapanmayi da goremez. RT'de 2,02, raster'da 1,18 olctu.

    ARA TON payi      - ikili (0/1) bir golge maskesi zeminde iki tepe birakir
      ve arasi bostur. Penumbra tam olarak o araligi doldurur. RT 0,335,
      raster 0,018.

⚠ Kanopideki yerel std bir HACIM olcutu DEGILDIR: raster'da daha yuksek cikar
  ama sebebi alfa-test aliasing'i ve sert golge benekleridir. Ayrica JPEG ile
  PNG karsilastiriliyorsa sikistirma gurultusu de oraya biner. Yorumlamayin.
"""
import sys
import numpy as np
from PIL import Image


def load(path, size=None):
    im = Image.open(path).convert('RGB')
    if size and im.size != size:
        im = im.resize(size, Image.LANCZOS)
    return np.asarray(im).astype(np.float64) / 255.0


def luma(a):
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def band(L, y0, y1):
    h = L.shape[0]
    return L[int(h * y0):int(h * y1)]


def midtone_fraction(patch):
    lo, hi = np.percentile(patch, 5), np.percentile(patch, 95)
    if hi - lo < 1e-6:
        return 0.0
    n = (patch - lo) / (hi - lo)
    return float(((n > 0.25) & (n < 0.75)).mean())


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 1
    a_path, b_path = argv[1], argv[2]
    size = Image.open(a_path).size
    A, B = load(a_path), load(b_path, size)
    La, Lb = luma(A), luma(B)
    print(f'A = {a_path}   B = {b_path}   {size[0]}x{size[1]}')

    noise = None
    if len(argv) > 3:
        Lc = luma(load(argv[3], size))
        noise = float(np.abs(La - Lc).mean())
        signal = float(np.abs(La - Lb).mean())
        print()
        print(f'taban gurultusu (tekrar kolu) = {noise:.5f}   <-- bundan kucugu OKUNAMAZ')
        print(f'sinyal            |A - B|     = {signal:.5f}')
        print(f'sinyal / gurultu              = {signal / max(noise, 1e-9):.1f}x')

    rows = []

    def add(name, x, y):
        rows.append((name, f'{x:.4f}', f'{y:.4f}', f'{y - x:+.4f}'))

    add('tam kare ort. luma', La.mean(), Lb.mean())
    add('sahne (alt %55) ort.', band(La, 0.45, 1.0).mean(), band(Lb, 0.45, 1.0).mean())
    add('sahne std (kontrast)', band(La, 0.45, 1.0).std(), band(Lb, 0.45, 1.0).std())
    near_a, near_b = band(La, 0.85, 1.0).mean(), band(Lb, 0.85, 1.0).mean()
    far_a, far_b = band(La, 0.42, 0.50).mean(), band(Lb, 0.42, 0.50).mean()
    add('yakin serit', near_a, near_b)
    add('uzak serit', far_a, far_b)
    add('YAKIN/UZAK orani  *', near_a / max(far_a, 1e-9), near_b / max(far_b, 1e-9))
    add('zemin ARA TON payi *', midtone_fraction(band(La, 0.78, 1.0)),
        midtone_fraction(band(Lb, 0.78, 1.0)))

    w = max(len(r[0]) for r in rows)
    print()
    print(f"{'olcut'.ljust(w)} | {'A':>9} | {'B':>9} | {'fark':>9}")
    print('-' * (w + 37))
    for n, x, y, d in rows:
        print(f'{n.ljust(w)} | {x:>9} | {y:>9} | {d:>9}')
    print('* kararı veren iki ölçüt bunlar; ustteki dosya notuna bakin.')

    print()
    print('dikey profil |A - B| (kare 10 dilim) -- fark NEREDE:')
    diff = np.abs(La - Lb)
    h = diff.shape[0]
    for i in range(10):
        s = diff[int(h * i / 10):int(h * (i + 1) / 10)].mean()
        print(f'  %{i * 10:>3}-{(i + 1) * 10:>3}  {s:.5f} ' + '#' * int(s * 4000))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
