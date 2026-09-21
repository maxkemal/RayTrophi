"""Raster derinlik on gecisi: tek govde mi, ve olcu aleti UYGULANANI mi gosteriyor?

Iki ayri ariza sinifini pinler.

1. ★★★ KOPYALANMIS DONGU. Bu dosyada ayni sinifin kuyrugu bir kez kopyalanmis
   ve kopya bir cagriyi dusurmustu -- realtime viewport'ta GPU culling HIC
   acilmadi ve gorsel belirtisi yoktu (bkz.
   docs/dev/RASTER_GPU_CULLING_NEVER_ENABLED.md). On gecis ayni mesh'lerden
   ayni kapilarla gecmek zorunda, o yuzden govde PAYLASILIR: dis dongu
   (rasterPass) + depthOnlyPass bayragi. Ikinci bir mesh dongusu acilirsa
   kapilar zamanla ayrisir ve on gecis ile asil gecis FARKLI geometri cizer --
   sonuc: derinlik testi tutmaz, nesneler kaybolur ya da z-fighting olur.

2. ★★★ SAYIM CIFTLENMESI. Telemetri blogu on geciste kosarsa her sey iki kez
   sayilir. Sonuc "makul ama yanlis" bir sayidir; kimse bug diye raporlamaz.

Ayrica: kolun raporladigi sey ISTEK, telemetrideki UYGULANAN. Ikisi ayri
alanlarda yasamali (bkz. bugfix_zero_aperture_sentinel dersi: kontrol durumu
deger kumesinde kodlanmaz).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "RayTrophiStudio" / "source"

fails = []


def read(rel):
    p = SRC / rel
    if not p.exists():
        fails.append(f"dosya yok: {rel}")
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


def strip_comments(text):
    """// yorumlarini at -- denetim ACIKLAMAYA takilmamali."""
    return "\n".join(re.sub(r"//.*$", "", line) for line in text.splitlines())


vp = read("src/Backend/VulkanViewportBackend.cpp")
vp_code = strip_comments(vp)

# --- 1) tek govde: mesh dongusu YALNIZ BIR KEZ acilmali -------------------
# ★ Olcut, "m_rasterMeshes uzerinde kac dongu var" DEGIL. Secim anahat gecisi
#   de ayni kap uzerinde doner ama kendi kamerasi, kendi pipeline layout'u ve
#   kendi kapilari vardir (scatter'i tamamen eler) -- o mesru bicimde ayridir.
#   Yasaklanan sey, MATERYAL ONIZLEME cizimini yapan ikinci bir dongudur.
loop_starts = [m.start() for m in re.finditer(
    r"for\s*\(\s*const\s+auto&\s*\[\s*meshKey\s*,\s*rmb\s*\]"
    r"\s*:\s*m_rasterMeshes\s*\)", vp_code)]
if not loop_starts:
    fails.append("m_rasterMeshes cizim dongusu bulunamadi; yeniden adlandirilmis olabilir.")
else:
    bounds = loop_starts + [len(vp_code)]
    mp_loops = sum(1 for i in range(len(loop_starts))
                   if "materialPreviewPipeline" in vp_code[bounds[i]:bounds[i + 1]])
    if mp_loops == 0:
        fails.append("hicbir mesh dongusu materyal onizleme pipeline'ini baglamiyor.")
    elif mp_loops > 1:
        fails.append(
            f"materyal onizleme cizimini yapan {mp_loops} ayri mesh dongusu var. On "
            "gecis AYNI govdeyi kullanmali (dis rasterPass dongusu), kendi kopyasini "
            "DEGIL -- kopyalanan kapilar zamanla ayrisir ve on gecis ile asil gecis "
            "FARKLI geometri cizer; sonuc z-fighting ya da kaybolan nesnedir.")

# Ve on gecis pipeline'i TEK bir yerde baglanmali.
if vp_code.count("materialPreviewDepthPrepassPipeline") > 0:
    binds = len(re.findall(r"vkCmdBindPipeline[^;]*materialPreviewDepthPrepassPipeline",
                           vp_code, re.S))
    if binds > 1:
        fails.append(f"on gecis pipeline'i {binds} ayri yerde baglaniyor; tek bind "
                     "noktasi olmali.")

if "rasterPass" not in vp_code or "depthOnlyPass" not in vp_code:
    fails.append("dis gecis dongusu (rasterPass / depthOnlyPass) yok; on gecis "
                 "ayri bir dongude kosuyor olabilir.")

# --- 2) telemetri on geciste sayilmamali ----------------------------------
if not re.search(r"if\s*\(\s*depthOnlyPass\s*\)\s*\{", vp_code):
    fails.append("telemetri blogu depthOnlyPass ile korunmuyor; sayimlar CIFTLENIR "
                 "ve sonuc makul gorunur.")

# --- 3) pipeline secimi govde icinde ---------------------------------------
if "materialPreviewDepthPrepassPipeline" not in vp_code:
    fails.append("on gecis pipeline'i baglanmiyor.")
else:
    if not re.search(r"depthOnlyPass\s*\n?\s*\?\s*m_interactiveViewport\.materialPreviewDepthPrepassPipeline",
                     vp_code):
        fails.append("pipeline secimi depthOnlyPass ucluyle yapilmiyor; iki ayri "
                     "bind noktasi olusmus olabilir.")

# --- 4) pipeline durumu: derinlik yazar, renk yazmaz -----------------------
m = re.search(r"dpCBA\.colorWriteMask\s*=\s*([^;]+);", vp_code)
if not m:
    fails.append("on gecis renk yazim maskesi ayarlanmiyor.")
elif m.group(1).strip() != "0":
    fails.append(f"on gecis colorWriteMask '{m.group(1).strip()}' -- 0 olmali, yoksa "
                 "gecis HDR hedefine yaziyor ve asil gecisin ustune biniyor.")
if not re.search(r"dpDS\.depthWriteEnable\s*=\s*VK_TRUE", vp_code):
    fails.append("on gecis derinlik YAZMIYOR; o zaman bir sey yapmiyor demektir.")
if not re.search(r"dpCBA\.blendEnable\s*=\s*VK_FALSE", vp_code):
    fails.append("on geciste harmanlama kapatilmamis.")

# --- 5) ISTEK ile UYGULANAN ayri alanlarda ---------------------------------
if "m_rasterGeometryStats.depth_prepass" not in vp_code:
    fails.append("telemetriye depth_prepass yazilmiyor; kol acikken pipeline "
                 "kurulamamissa disaridan anlasilmaz.")
else:
    blk = re.search(r"const bool depthPrepassActive\s*=(.*?);", vp_code, re.S)
    if not blk:
        fails.append("depthPrepassActive hesabi bulunamadi.")
    elif "materialPreviewDepthPrepassPipeline" not in blk.group(1):
        fails.append("depthPrepassActive pipeline varligini KONTROL ETMIYOR -- kol "
                     "acikken telemetri true der ama gecis kosmaz. Olcu aleti "
                     "ISTENENI raporluyor demektir.")

# --- 5b) ★★★ TELEMETRI ZINCIRI: gpu_culling nereden geciyorsa oradan gecmeli
#
# Bu alan backend struct'indan IPC'ye kadar BES ayri kabi geciyor ve her
# sicrama elle yaziliyor. Bir sicrama atlanirsa alan sessizce hep false kalir
# -- "varsayilan bir olcum degildir" dersinin tam ornegi. Ilk yazimda bu
# denetim yokken IKI sicrama (RasterGeometryStats->RasterFrameTelemetry koprusu
# ve Python telemetri sozlugu) gercekten atlanmisti.
#
# Olcut gpu_culling'den TURETILIR, elle listelenmez: o alan bu zincirin
# tamamindan geciyor, yani ileride bir kap eklenirse kontrol kendiliginden
# genisler.
CHAIN_FILES = [
    "include/Backend/VulkanBackend.h",        # RasterGeometryStats (uretici)
    "include/Viewport/RasterFrameTelemetry.h",# viewport telemetri kabi
    "include/Api/RtApi.h",                    # API kabi
    "src/Backend/VulkanViewportBackend.cpp",  # uretici yazimi + koprü
    "src/Api/RtApiViewport.cpp",              # API'ye kopya
    "src/Api/RtIpc.cpp",                      # IPC yayini
    "src/Api/RtPython.cpp",                   # Python yayini
]
for rel in CHAIN_FILES:
    body = strip_comments(read(rel))
    # Yalnizca KOD gecisleri: alan erisimi ya da sozluk/JSON anahtari.
    uses_gc = re.search(r'(?:\.|")gpu_culling', body)
    uses_dp = re.search(r'(?:\.|")depth_prepass', body)
    if uses_gc and not uses_dp:
        fails.append(
            f"{rel}: gpu_culling bu kaptan geciyor ama depth_prepass GECMIYOR. "
            "Atlanan bir sicrama alani sessizce false birakir -- hata vermez, "
            "olcum 'on gecis hic kosmadi' der.")

# ★ Dosya granulerligi TEK BASINA yetmez: bir dosyada alan baska bir satirda
#   geciyorsa eksik sicrama gizlenir (koprü tam boyle kacti). O yuzden ayrica
#   ATAMA HEDEFI bazinda kontrol: `X.gpu_culling =` varsa `X.depth_prepass =`
#   de olmali. Bu, kabin adini bilmeden zinciri pinler.
for rel in CHAIN_FILES:
    body = strip_comments(read(rel))
    for lhs in sorted(set(re.findall(r"(\w+)\.gpu_culling\s*=", body))):
        if not re.search(re.escape(lhs) + r"\.depth_prepass\s*=", body):
            fails.append(
                f"{rel}: '{lhs}.gpu_culling' yaziliyor ama '{lhs}.depth_prepass' "
                "YAZILMIYOR -- bu kaba alan hic ulasmiyor ve sessizce false kaliyor.")

# --- 6) bes dokunus --------------------------------------------------------
touches = {
    "cekirdek API (RtApi.h)": ("include/Api/RtApi.h", "setRasterDepthPrepass"),
    "API govdesi": ("src/Api/RtApiViewport.cpp", "setRasterDepthPrepass"),
    "IPC dispatch": ("src/Api/RtIpc.cpp", "viewport.set_raster_depth_prepass"),
    "Python": ("src/Api/RtPython.cpp", "set_raster_depth_prepass"),
    "telemetri yayini": ("src/Api/RtIpc.cpp", "depth_prepass"),
}
for name, (rel, needle) in touches.items():
    if needle not in read(rel):
        fails.append(f"{name}: '{needle}' {rel} icinde yok.")

# --- 7) kol kapatilabilir olmali -------------------------------------------
if "m_rasterDepthPrepassAllowed" not in read("include/Backend/VulkanBackend.h"):
    fails.append("kapatma kolu yok. Kapatilamayan bir duzeltme, kendisini "
                 "yargilayacak olcumu de oldurur.")

if fails:
    print("FAIL - derinlik on gecisi denetimi")
    for f in fails:
        print("  - " + f)
    sys.exit(1)

print("PASS - derinlik on gecisi tek govdede, olcum UYGULANANI gosteriyor")
print("  mesh dongusu TEK; on gecis dis rasterPass dongusuyle ayni govdeyi kullaniyor")
print("  telemetri depthOnlyPass ile korunuyor (cift sayim yok)")
print("  pipeline: derinlik YAZAR, renk yazmaz, harmanlama kapali")
print("  depthPrepassActive pipeline varligini kontrol ediyor (istek != uygulanan)")
print("  bes dokunus tam, kol kapatilabilir")
