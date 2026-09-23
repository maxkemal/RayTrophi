"""Raster LOD ucgen butcesi mesh basina PAYLASTIRILIYOR mu?

Arizanin sekli (docs/dev/RASTER_MICROTRIANGLE_WALL.md):
raster_cull.comp'un finalize()'i MESH BASINA kosar. `allowed` degerini
g.scatterTriangleTarget'in TAMAMINA karsi cozerse, N cull mesh'in her biri
butun butceyi kendisinin sanir ve efektif tavan N KATI olur. Olculen sahnede
N=35 idi: geri besleme orani her kare 1,5'e yapisti, mesafe esigi clamp'e
kacti ve 1033 instance'in yalnizca 28'i proxy'ye dustu -- yani butce HIC
uygulanmadi. Gorsel belirti yok: sahne dogru cizilir, yalnizca yavastir.

Bu denetim iki tarafi da pinler: shader payi okumali, CPU payi doldurmali.
Struct duzeni AYRICA audit_shader_struct_layout.py tarafindan korunuyor.
"""
import re
import sys
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from rt_repo_root import repo_root as _repo_root
ROOT = _repo_root()
SRC = ROOT / "RayTrophiStudio" / "source"

fails = []


def read(rel):
    p = SRC / rel
    if not p.exists():
        fails.append(f"dosya yok: {rel}")
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


def strip_comments(text):
    """// yorumlarini at. Denetim, ACIKLAMANIN kendisine takilmamali --
    kok neden anlatan yorum g.scatterTriangleTarget'tan soz ediyor."""
    return "\n".join(re.sub(r"//.*$", "", line) for line in text.splitlines())


# --- 1) shader: pay okunuyor, tam hedef okunmuyor -------------------------
comp = read("shaders/raster_cull.comp")
comp_code = strip_comments(comp)

if "p.targetShare" not in comp_code:
    fails.append("raster_cull.comp `p.targetShare` okumuyor -- butce hala "
                 "mesh basina CARPILIYOR olabilir.")

m = re.search(r"uint\s+allowed\s*=(.*?);", comp_code, re.S)
if not m:
    fails.append("raster_cull.comp icinde `uint allowed = ...` bulunamadi; "
                 "LOD cozumu yeniden adlandirilmis olabilir.")
elif "g.scatterTriangleTarget" in m.group(1):
    fails.append("raster_cull.comp `allowed` degerini DOGRUDAN "
                 "g.scatterTriangleTarget'tan cozuyor. finalize() mesh basina "
                 "kosar; bu, her mesh'e butun butceyi vermek demektir.")

# Pay Q16 kesir: 65536 olcegi iki tarafta da gorunmeli.
if "65536" not in comp_code:
    fails.append("raster_cull.comp'ta Q16 olcegi (65536) yok; pay kesri "
                 "yorumlanmiyor olabilir.")

# --- 2) CPU: pay dolduruluyor ve normalize ediliyor -----------------------
cpu = read("src/Backend/VulkanBackend_Raster.cpp")
cpu_code = strip_comments(cpu)

fn = re.search(r"void\s+VulkanBackendAdapter::rebuildRasterCullBindings\(\)"
               r"\s*\{(.*?)\n\}", cpu_code, re.S)
if not fn:
    fails.append("rebuildRasterCullBindings() bulunamadi.")
else:
    body = fn.group(1)
    if "targetShare" not in body:
        fails.append("rebuildRasterCullBindings() `targetShare` doldurmuyor; "
                     "pay sifir kalir ve shader'da allowed=1 olur -- yani HER "
                     "SEY proxy'ye duser. Bu sessiz ve terstir.")
    if "kFlagLodSplit" not in body:
        fails.append("rebuildRasterCullBindings() payi kFlagLodSplit'e gore "
                     "ayirmiyor; butceye tabi olmayan mesh'ler de pay alir.")
    if not re.search(r"totalDemand|toplamTalep", body):
        fails.append("rebuildRasterCullBindings() paylari bir TOPLAMA gore "
                     "normalize etmiyor; oransal pay yok demektir.")
    # ★ En sinsi hâli: split mesh'e 0 pay birakmak. Bir kacis kolu olmali.
    if not re.search(r"totalDemand\s*<=\s*0", body):
        fails.append("totalDemand sifir oldugunda kacis kolu yok. Pay 0 "
                     "kalirsa shader allowed=max(1u,0)=1 cozer ve sahne "
                     "TAMAMEN proxy cizilir -- hatasiz, ve 'LOD calisiyor' "
                     "gibi gorunur.")

# --- 3) telemetride hedef HALA raporlaniyor mu ----------------------------
# Hedef raporlanmali ki A/B yapilabilsin; ama artik UYGULANAN da olculmeli.
vp = read("src/Backend/VulkanViewportBackend.cpp")
if "scatter_triangle_target" not in vp:
    fails.append("scatter_triangle_target telemetriden dusmus; butcenin "
                 "uygulanip uygulanmadigi disaridan olculemez.")

if fails:
    print("FAIL - raster LOD butce denetimi")
    for f in fails:
        print("  - " + f)
    sys.exit(1)

print("PASS - LOD ucgen hedefi mesh basina PAYLASTIRILIYOR")
print("  shader p.targetShare okuyor, `allowed` artik tam hedeften cozulmuyor")
print("  CPU payi kFlagLodSplit mesh'leri arasinda talebe orantili dagitiyor")
print("  totalDemand==0 icin kacis kolu var (0 pay = her sey proxy tuzagi)")
