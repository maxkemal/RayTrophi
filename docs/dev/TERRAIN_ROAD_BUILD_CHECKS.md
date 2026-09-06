# Terrain Curve / Road Carve Build Checks

> **Durum:** AKTIF - Faz 3.6 / road Faz 1 sinirinda bekleyen dogrulama paketi.
> Uygulama ve statik baglanti hazir; derleme ve canli kontroller asagida.
>
> ★ 2026-08-31: alan sayisi **besten altiya** cikti - `infrastructure.ditch`
> eklendi ve `Road Fields Output` icin **zorunlu** bir pindir. Bu partinin tam
> kontrol listesi [NEXT_BUILD_CHECKS.md](NEXT_BUILD_CHECKS.md) icindedir.

1. Build RayTrophi Studio once. Confirm the new `TerrainCurve*` and
   `TerrainRoadCarve*` and `TerrainRoadFieldsOutput*` translation units compile.
2. Open or create one `SplineObject` crossing a terrain-sized XZ area.
3. Confirm Terrain Graph exposes `Curve Input`, `Curve to Mask`, and
   `Road Carve`; select the spline in `Curve Input`.
4. While the application IPC endpoint is available, run:

   ```powershell
   python scripts/test/rt_test_terrain_road_carve.py --spline "<name>"
   ```

5. Expected: `Road Fields Output` atomically publishes six `infrastructure.*`
   fields from one Road Carve revision; all values are finite;
   core, shoulder and ditch are non-empty; exclusion covers at least core; and
   the one-metre fixture offset produces positive Fill. The temporary terrain is
   removed in `finally`; the authored spline is never modified.
6. Preview Road Core, Shoulder, Cut, Fill, Ditch and Foliage Exclusion. Pulling
   different outputs must leave `Snapshot revision` unchanged. Editing the
   spline or road profile and evaluating once must increment it once.
7. A ditch that measures zero everywhere means the cross-section never reached
   the rasterizer - the road is a flat linear trench again, which is exactly
   what a flow solver reads as a river bed, and nothing about the height field
   would say so.
