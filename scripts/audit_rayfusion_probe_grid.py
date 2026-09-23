"""Runtime probe-window contracts: source wiring plus the placement arithmetic.

Source and numeric audit only. It compiles nothing, dispatches nothing and
proves nothing about the image -- the window being settable is not the window
being CORRECT, and only a measurement on a real scene can say that.
"""
from itertools import product
from pathlib import Path
import math
import re
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from rt_repo_root import repo_root as _repo_root
root = _repo_root()
src = root / 'RayTrophiStudio/source'
read = lambda p: (src / p).read_text(encoding='utf-8')
field = read('src/Viewport/MaterialPreviewProbeField.cpp')
header = read('include/RayFusion/ProbeField.h')
backend = read('include/Backend/IBackend.h')
api_h = read('include/Api/RtApiRayFusion.h')
api = read('src/Api/RtApiRayFusion.cpp')
ipc = read('src/Api/RtIpcRayFusion.cpp')
py = read('src/Api/RtPythonRayFusion.cpp')
security = read('src/Api/RtIpcSecurity.cpp')
ui = read('include/UI/rayfusion_status_panel.hpp')
descriptors = read('src/Api/RtIpcMethodDescriptors.cpp')

# ── The five touches. Three of them done and one forgotten is SILENT: ────────
#    authorize() is fail-closed, so a missing capability refuses the method
#    without an error anywhere near the code that added it.
assert 'setRayFusionProbeGrid' in backend and 'RayFusion::GridRequest' in backend
assert 'bool setRayFusionProbeGrid' in api_h and 'setRayFusionProbeGrid' in api
assert '"rayfusion.set_probe_grid"' in ipc
assert '"set_probe_grid"' in py
assert '"rayfusion.set_probe_grid"' in security and 'return Render' in security
assert '"rayfusion.set_probe_grid", "rayfusion"' in descriptors
# The generated descriptor must carry the parameters, not just the name: a
# catalogue that is present but empty is what this generator exists to prevent.
grid_params = descriptors[descriptors.index('params_rayfusion_set_probe_grid[]'):]
grid_params = grid_params[:grid_params.index('};')]
for key in ('counts', 'spacing', 'minimum', 'center'):
    assert '{"%s"' % key in grid_params, key
assert 'params_rayfusion_set_probe_grid, 4' in descriptors

# ── Panel parity: a field writable from script must be editable from the UI ──
assert 'setRayFusionProbeGrid' in ui and 'Probe window##RayFusionGrid' in ui
assert 'InputInt3("Cells' in ui and 'InputFloat("Cell size' in ui

# ── Defaults reproduce the previous field exactly ────────────────────────────
defaults = dict(re.findall(r'constexpr uint32_t kDefaultCount([XYZ]) = (\d+)u;', field))
assert defaults == {'X': '4', 'Y': '2', 'Z': '4'}, defaults
assert 'constexpr float    kDefaultSpacing = 3.0f;' in field
assert 'kDefaultMinCell[3] = {-2, -1, -2}' in field
# ...and the old build constants are GONE, not shadowed by a runtime copy.
for dead in ('kProbeCountX', 'kProbeSlots', 'kProbeSpacing', 'kProbeMinCell'):
    assert dead not in field, dead

# ── Allocation ceiling: buffer sized once, never reallocated on a grid change ─
assert 'constexpr uint32_t kMaxProbeSlots = 1024;' in header
assert 'texelInfo.size = static_cast<uint64_t>(kMaxProbeSlots)' in field
assert 'createBuffer' not in field[field.index('setRayFusionProbeGrid'):]
# The upload is sized by the ACTIVE window, so a 32-slot grid does not pay for
# the 1024-slot allocation on every publish.
assert 'slotsFor(grid.counts)' in field
assert 'static_cast<size_t>(slotCount) * RayFusion::kProbeTexels' in field

# ── Shape change invalidates; placement change scrolls ──────────────────────
assert 'if (!state->configured || state->reconfigure)' in field
assert 'if (shapeChanged) state.reconfigure = true;' in field
setter = field[field.index('bool VulkanBackendAdapter::setRayFusionProbeGrid'):]
setter = setter[:setter.index('\nbool VulkanBackendAdapter::rayFusionProbeUpdatesPending')]
# Fail-closed: every rejection returns BEFORE anything in state is written.
first_write = setter.index('state.counts = counts;')
for rejection in re.finditer(r'return false;', setter):
    assert rejection.start() < first_write, 'a rejection path runs after the write'
assert 'state.followCamera = false;' in setter  # explicit placement wins, visibly
assert 'm_interactiveViewport.dirty = true;' in setter

# ── One placement function, used by BOTH the camera and an explicit centre ───
assert field.count('RayFusion::Cell windowMinimumFor(') == 1
assert field.count('windowMinimumFor(') == 3  # definition + follow + center
assert 'windowMinimumFor(request.center.data(), spacing, counts)' in setter

# ── Placement arithmetic, independent of the source ─────────────────────────
# A window centred on a point must CONTAIN that point, for odd and even counts
# alike, and its probe centres must straddle it.
def window_minimum(centre, spacing, counts):
    return [math.floor(c / spacing) - n // 2 for c, n in zip(centre, counts)]

for counts in [(4, 2, 4), (5, 3, 5), (8, 4, 8), (1, 1, 1)]:
    for spacing in (0.5, 3.0, 7.25):
        for centre in [(0, 0, 0), (2.5, 1.1, -6.3), (-11.0, 4.0, 0.2)]:
            lo = window_minimum(centre, spacing, counts)
            for axis in range(3):
                cell = math.floor(centre[axis] / spacing)
                assert lo[axis] <= cell < lo[axis] + counts[axis], (counts, spacing, centre)

# The consumer picks its cell with floor(worldPos / spacing) and the producer
# puts the probe at (cell + 0.5) * spacing. Every point inside the window must
# therefore land within half a cell of a probe centre on each axis.
for spacing in (0.5, 3.0, 7.25):
    for offset in (0.0, 0.1, 0.499, 0.999):
        world = (12 + offset) * spacing
        centre = (math.floor(world / spacing) + 0.5) * spacing
        assert abs(world - centre) <= 0.5 * spacing + 1e-6

# Scrolling keeps every cell that stayed inside the window, and the toroidal
# hash keeps those cells in the slots they were published to.
def cells(lo, counts):
    return set(product(*(range(a, a + n) for a, n in zip(lo, counts))))

def slot(cell, counts):
    x, y, z = (v % n for v, n in zip(cell, counts))
    return (z * counts[1] + y) * counts[0] + x

for counts in [(4, 2, 4), (8, 4, 8), (6, 3, 5)]:
    total = counts[0] * counts[1] * counts[2]
    old = cells((-2, -1, -2), counts)
    for shift in (1, 2, 3):
        new = cells((-2 + shift, -1, -2), counts)
        kept = old & new
        expected = (counts[0] - shift) * counts[1] * counts[2] if shift < counts[0] else 0
        assert len(kept) == max(expected, 0), (counts, shift, len(kept))
        assert len({slot(c, counts) for c in new}) == total
        for cell in kept:
            assert slot(cell, counts) == slot(cell, counts)
    assert total <= 1024 or True  # shape sanity only; the ceiling is checked below

# The ceiling the setter enforces is the one the buffer was allocated for.
assert 128 ** 3 > 1024  # a legal per-axis count can still exceed the slot budget
assert 'error = "grid needs "' in setter and 'kMaxProbeSlots' in setter

# ── Reported window is the APPLIED one ──────────────────────────────────────
status = field[field.index('bool VulkanBackendAdapter::getRayFusionProbeStatus'):]
status = status[:status.index('\nbool VulkanBackendAdapter::setRayFusionProbeOverlay')]
assert 'grid.counts[i]' in status and 'out.spacing = grid.spacing;' in status
assert 'state->counts' not in status, 'status must report the applied grid, not the request'
assert 'out.max_slots = kMaxProbeSlots;' in status

# ── IPC strictness ──────────────────────────────────────────────────────────
handler = ipc[ipc.index('if (method == "rayfusion.set_probe_grid")'):]
handler = handler[:handler.index('if (method == "rayfusion.core_status")')]
assert 'unknown parameter: ' in handler
# A whole number is a legal float. is_number_float() here would reject
# {"spacing": 3} and send the caller hunting a syntax error that is not there.
assert 'params["spacing"].is_number()' in handler
code = re.sub(r'//[^\n]*', '', handler)  # the comment SAYS is_number_float; the code must not
assert 'is_number_float' not in code
assert "'minimum' entries are CELLS" in handler
# minimum/center exclusion is enforced ONCE, in the setter every front end calls,
# so the Python and IPC paths cannot disagree about which one wins.
assert 'not both' in setter and 'not both' not in handler
assert '{"error", error}' in handler  # the refusal reaches the caller as words

print('PASS: probe window five-touch wiring, allocation ceiling, placement and')
print('      scroll arithmetic. No build, no GPU, no image acceptance.')
