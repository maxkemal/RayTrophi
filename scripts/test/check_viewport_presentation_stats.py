"""Non-build audit: frame FPS must come from presentation, never sample timing."""

from pathlib import Path
import re
import xml.etree.ElementTree as ET


root = Path(__file__).resolve().parents[2]
source = root / "RayTrophiStudio/source"
main = (source / "src/Core/Main.cpp").read_text(encoding="utf-8")
stats = (source / "src/Viewport/ViewportPresentationStats.cpp").read_text(encoding="utf-8")
scatter = (source / "shaders/bsdf_scatter.glsl").read_text(encoding="utf-8")

# Main can clear frame metrics on mode changes, but cannot publish sample time
# into them. There must be exactly one publisher after actual SDL presentation.
for field in ("avg_total_frame_time_ms", "avg_total_frame_fps"):
    assignments = re.findall(r"render_settings\." + field + r"\s*=\s*([^;]+);", main)
    assert assignments and all(value.strip() == "0.0f" for value in assignments)
assert main.count("Viewport::notePresentationStats(") == 1
present = main.rindex("SDL_RenderPresent(renderer);")
publish = main.index("Viewport::notePresentationStats(")
assert present < publish < main.index("rtapi::noteDisplayPathTiming(dpt);", present)
assert "Viewport::resetPresentationStats();" in main
assert re.search(r"\btotal_frame_fps\b", main) is None
assert "settings.avg_total_frame_fps = 1000.0f / settings.avg_total_frame_time_ms;" in stats
assert "lastPresentation = Clock::time_point{};" in stats

# Both changes must actually be consumed by the user's normal project build.
project = ET.parse(root / "RayTrophiStudio/RayTrophiStudio.vcxproj")
registered = {item.get("Include") for item in project.iter() if item.get("Include")}
assert r"source\src\Viewport\ViewportPresentationStats.cpp" in registered
assert r"source\include\Viewport\ViewportPresentationStats.h" in registered
assert "gas_surface_guiding" not in scatter
assert "guideGasDiffuseRay" not in scatter
assert r"source\shaders\gas_surface_guiding.glsl" not in registered
print("PASS: presentation owns frame metrics; sample publishers removed; guiding rolled back.")
