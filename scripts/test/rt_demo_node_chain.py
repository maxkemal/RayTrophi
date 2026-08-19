# Hızlı node zinciri demosu — panelde gezinmek için hazır sahne.
#
# Tek çağrıda kurar:
#   - bir fluid domain (tohumlanmış, adımlanmış) + bir gas domain
#   - bir flow source (emitter)
#   - bir collider objesi (madde/MSF için)
#   - ÜÇ kapsamlı graph: domain (fluid), domain (gas), object (collider)
#
# Çalıştır:
#   Invoke-RtIpc script.run_file @{ path = 'scripts/test/rt_demo_node_chain.py' }
#
# ★ Bu bir TEST DEĞİL — hiçbir şey iddia etmez, sadece sahne kurar ve panelde
# neye bakman gerektiğini yazar. Test dosyaları PASS/FAIL der; bu dosya demez.
# İkisini karıştırmak, "kurulum başarılı" satırını geçen bir test sanmaya yol
# açar.
import os
import sys

import rt

sys.path.insert(0, os.path.join("scripts", "test"))
import rt_testlog  # noqa: E402

rt_testlog.start("demo_node_chain")
log = rt_testlog.log

FLUID = "DemoFluid"
GAS = "DemoGas"
EMITTER = "DemoEmitter"
COLLIDER = "DemoCrate"

# ── Temizlik ────────────────────────────────────────────────────────────────
# ★ Var olanı silmek, üstüne yazmaktan iyidir: yarı kalmış bir önceki demo
# sahnesi, panelde "neden bu node burada?" sorusunu doğurur.
for name in (FLUID, GAS):
    if any(d["name"] == name for d in rt.fluid.list_domains()):
        rt.fluid.remove_domain(name)
if any(s["name"] == EMITTER for s in rt.flow_source.list()):
    rt.flow_source.remove(EMITTER)

# ── Sahne ───────────────────────────────────────────────────────────────────
log("== sahne kuruluyor ==")

# ★ Kutular BİLEREK çakışık DEĞİL. Çakışık hacim kutuları bu depoda aylarca
# süren bir siyah bant + maliyet patlamasına mal oldu (2026-08-16'da çözüldü,
# VOLUME_BOX_REENTRY_POSTMORTEM.md). Demo sahnesi o durumu üretmemeli.
rt.fluid.create_domain(FLUID, domain_min=(-1, 0, -1), domain_max=(1, 2, 1),
                       voxel_size=0.1)
rt.fluid.seed(FLUID, seed_min=(-0.5, 0.9, -0.5), seed_max=(0.5, 1.7, 0.5),
              particles_per_cell=4)
rt.fluid.set_param(FLUID, backend="gpu", preset="water", boundary="closed")
for _ in range(8):
    rt.fluid.step(0.0166)
log("   %s: %d partikül" % (FLUID, rt.fluid.get(FLUID)["particle_count"]))

rt.fluid.create_domain(GAS, type="gas",
                       domain_min=(1.5, 0, -1.5), domain_max=(4.5, 3, 1.5),
                       voxel_size=0.1)
log("   %s kuruldu (fluid kutusuyla çakışmıyor)" % GAS)

rt.flow_source.create(EMITTER, FLUID,
                      source_mode="point", position=(0.0, 1.4, 0.0),
                      radius=0.25, density=1.0,
                      fluid_particles_per_second=800.0)
log("   %s kuruldu" % EMITTER)

crate = None
try:
    if any(c["name"] == COLLIDER for c in rt.collider.list()):
        rt.collider.remove(COLLIDER)
    # ★ add_primitive GERÇEK adı döndürür. İstediğin adı varsaymak, sahne o adı
    # zaten kullanıyorsa sessizce başka bir objeyi hedeflemene yol açar.
    crate = rt.scene.add_primitive("cube", COLLIDER, 1.0)
    rt.scene.set_transform(crate, translation=(0.0, 0.5, 0.0))
    rt.collider.create(crate, source_object=crate)
    log("   %s collider olarak kuruldu" % crate)
except Exception as exc:                                       # noqa: BLE001
    # ★ Sessizce atlamak yerine SÖYLE. Object kapsamı graph'ı olmadan demo
    # eksiktir, ve eksik olduğunu bilmeden panele bakmak zaman kaybıdır.
    log("   UYARI: collider kurulamadı (%s) -- object kapsamı graph'ı atlanacak" % exc)

# ── Graph 1: fluid domain — kimlik → çözücü → ayarlar ───────────────────────
log("== graph: domain/%s ==" % FLUID)
owner = rt_testlog.fresh_graph(rt, "domain", FLUID)

solver = rt.sim_graph.add_node("domain", FLUID, "sim.solver")
rt.sim_graph.connect("domain", FLUID, owner, solver)
# ★ Alanlar OPT-IN: sadece dokunduğun yazılır. Aşağıda ikisine dokunuyoruz,
# geri kalanı authored değerinde kalır ve panelde tiksiz görünür.
rt.sim_graph.set_node_value("domain", FLUID, solver, "kinematic_viscosity", 0.4)
rt.sim_graph.set_node_value("domain", FLUID, solver, "viscosity_sweeps", 12)

settings = rt.sim_graph.add_node("domain", FLUID, "sim.domain_settings")
rt.sim_graph.connect("domain", FLUID, solver, settings)
rt.sim_graph.set_node_value("domain", FLUID, settings, "pore_amount", 0.3)

emit_node = rt.sim_graph.add_node("domain", FLUID, "sim.emitter")
# ★ Zincire BAĞLI. Bağlanmamış bir Emitter hiçbir şey yaymaz: tuvalde yalıtılmış
# duran ama yine de çözücüyü yapılandıran bir node, graph'ı okunamaz yapardı.
rt.sim_graph.connect("domain", FLUID, settings, emit_node)
rt.sim_graph.set_node("domain", FLUID, emit_node, "emitter", EMITTER)
rt.sim_graph.set_node_value("domain", FLUID, emit_node, "radius", 0.45)
rt.sim_graph.set_node_value("domain", FLUID, emit_node, "temperature", 340.0)

insp = rt.sim_graph.add_node("domain", FLUID, "sim.field_inspect")
rt.sim_graph.connect("domain", FLUID, emit_node, insp)
rt.sim_graph.set_node("domain", FLUID, insp, "channel", "temperature")
log("   node sayısı: %d" % len(rt.sim_graph.nodes("domain", FLUID)))

# ── Graph 2: gas domain — görünüm ───────────────────────────────────────────
log("== graph: domain/%s ==" % GAS)
gas_owner = rt_testlog.fresh_graph(rt, "domain", GAS)
volume = rt.sim_graph.add_node("domain", GAS, "sim.material_volume")
rt.sim_graph.connect("domain", GAS, gas_owner, volume)
rt.sim_graph.set_node("domain", GAS, volume, "preset", "smoke")
log("   node sayısı: %d" % len(rt.sim_graph.nodes("domain", GAS)))

# ── Graph 3: object — madde ─────────────────────────────────────────────────
if crate:
    log("== graph: object/%s ==" % crate)
    obj_owner = rt_testlog.fresh_graph(rt, "object", crate)
    sub = rt.sim_graph.add_node("object", crate, "sim.substance")
    rt.sim_graph.connect("object", crate, obj_owner, sub)
    rt.sim_graph.set_node("object", crate, sub, "substance", "Wood (Oak)")
    pyro = rt.sim_graph.add_node("object", crate, "sim.pyrolysis")
    rt.sim_graph.connect("object", crate, sub, pyro)
    log("   node sayısı: %d" % len(rt.sim_graph.nodes("object", crate)))

# ── Ne göreceksin ───────────────────────────────────────────────────────────
log("")
log("== kurulan graph'lar ==")
for g in rt.sim_graph.list():
    log("   %-7s %-14s nodes=%d owner_node=%s owner_missing=%s" % (
        g["scope"], g["owner"], g["nodes"], g["owner_node"], g["owner_missing"]))

log("")
log("== PANELDE BAK ==")
log("  Alt şerit -> Nodes -> alan seçici 'simulation'")
log("  1. Scope çubuğu: [domain] ve yanında sahip seçici.")
log("     %s ile %s arasında geçiş yap; TUVAL DEĞİŞMELİ." % (FLUID, GAS))
log("  2. Scope'u [object] yap -> sahip seçici sahne objelerini listeler.")
log("  3. Scope'u [world] yap -> 'No graph for this world yet.' + Create graph.")
log("     Dünyanın script yüzeyi yok, o yüzden sahip node'u da YOK -- boş graph")
log("     dürüst olan.")
log("  4. Sahip node'unu seç: sağda isim SABİT yazılı, seçici YOK.")
log("     (Yeniden hedeflenemez; kapsamı değiştirmek için Scope çubuğu.)")
log("  5. Solver node'unu seç: tik kutulu kadranlar.")
log("     kinematic_viscosity ve viscosity_sweeps TİKLİ, gerisi TİKSİZ.")
log("     ** Tiksiz = bu graph o parametre hakkinda fikri yok, YAZMAZ.")
log("        'value=0' ile ayni sey DEGIL.")
log("  6. voxel_size'i tikle -> 'requires a restart' uyarisi cikar.")
log("     Apply, 'allow restart' kapaliyken onu REDDEDER ve raporlar.")
log("  7. Emitter node'u: flow source secici + opt-in kadranlar.")
log("     domain bagini DEGISTIRMEZ -- o baglama kaynagin kendisinde.")
log("  8. Toolbar 'Clear Graph' -> tuval BOS KALMAMALI, sahip node'u geri gelir.")
log("")
log("  Apply denemek istersen: once Evaluate (hicbir sey uygulamaz, niyeti")
log("  gosterir), sonra Apply. Geri almak icin 'Clear Overrides'.")
log("")
log("SAHNE HAZIR")
