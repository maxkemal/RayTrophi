# Third-Party Asset Licensing

RayTrophi Studio itself is MIT licensed. Some runtime assets under `assets/` are
**not** MIT and carry their own terms. This file records the provenance and the
restrictions that travel with those files.

Keep this file accurate. A wrong or missing license note is worse than no asset.

---

## Vegetation — e-on software / Bentley Systems

**Affected paths**

```txt
assets/vegetation/trees/
assets/vegetation/bushes/
assets/vegetation/flowers/
assets/vegetation/grass/
```

**Provenance**

Plant models were downloaded from **PlantCatalog**, opened in **PlantFactory**,
exported to FBX, and converted to glTF/GLB. Format conversion does not change
their derivation status: the EULA covers derivatives "in full or in part".

- Source: <https://www.bentley.com/software/e-on-software-free-downloads/>
- Governing agreement: <https://www.bentley.com/wp-content/uploads/eula-for-e-on-products.pdf>
- Copyright: Bentley Systems

**Governing clause (E-ON SOFTWARE END USER LICENSE AGREEMENT)**

> **1.5** "Software" means VUE, PlantFactory, and PlantCatalog provided for free
> download.
>
> **1.9** "User Made Assets" means any procedural or static model files,
> textures, and meshes (and any part of the foregoing) that are created using the
> Software.
>
> **2.2 User Made Assets.** Such license granted in section 2.1 does not permit
> selling any Assets that are either in full or in part derivatives from
> PlantCatalog assets, materials, and/or texture maps. The license granted in
> Section 2.1 permits, however, sharing or selling User-Made Assets created in
> VUE or PlantFactory (either files in generic 3D format, proprietary VUE, or
> PlantFactory formats or images).

**What this permits and forbids**

| Action | Status |
|---|---|
| Use in renders, projects, client work | Permitted |
| Share / redistribute the model files | Permitted — clause 2.2 names "sharing" |
| **Sell** these assets, or anything containing them | **Forbidden** — PlantCatalog derivative |

The prohibition is on *selling*, not on inclusion. Because these files are
PlantCatalog derivatives, the restriction is permanent and survives remodeling,
retexturing and format conversion.

The EULA imposes no attribution requirement. This file exists for provenance
tracking, not because attribution is compelled.

> Note: the download page uses looser wording ("may not resell ... on
> marketplaces"). The EULA text above is the binding one and has no marketplace
> qualifier. Follow the EULA.

**Consequences for distribution**

1. These assets must not ship in any paid RayTrophi build, tier, or bundle.
2. Built-in templates must not list them under `assets.required`. A template
   that requires them cannot be shipped in a paid build without breaking.
   Reference them as `assets.optional` with a procedural fallback.
3. If RayTrophi ever gains a paid distribution, these files must be removed from
   the package first, and any template depending on them must still open.

**Current status:** not distributed. `x64/` is git-ignored, so these files exist
only in local build outputs and are not carried by the repository.

---

## Volumes — mixed provenance, per-file unverified

```txt
assets/volume/vdb/Cumulonimbus_Field_*/
```

Recalled provenance: partly **EmberGen** (JangaFX) output, partly **free
CGTrader** downloads. Used for development and testing only.

These two sources have opposite licensing shapes and cannot be covered by one
statement:

| Source | Shape |
|---|---|
| EmberGen output | JangaFX grants the user rights to generated content; effectively RayTrophi's own asset |
| CGTrader free download | Licensed **per asset** by the individual uploader; terms vary and some are editorial-only |

There is no blanket "CGTrader free assets may be redistributed" rule. Each file
needs its own product page checked before it can be distributed.

`sourceTool` is `"unknown"` on every descriptor, so today the two groups are not
distinguishable from metadata alone.

**Status: development and testing only. Not distributed.** They live under
git-ignored `x64/` and must not enter an installer or a built-in template's
`assets.required` until each file's origin is identified and recorded here.

To clear them for distribution: set `sourceTool` to `EmberGen` on the generated
ones (which then need no further permission), and for each CGTrader file record
the product URL and its stated license below.

---

## Everything else

`assets/scenes/default/default.glb`, `assets/templates/**`, `assets/matcaps/**`
are RayTrophi's own content and fall under the project's MIT license.

Built-in templates are procedural by design and carry no binary asset payload;
see `docs/dev/TEMPLATE_HUB_UX_ROADMAP.md`.

---

## Adding a new third-party asset

1. Add a section here: path, source URL, license document URL, the governing
   clause quoted verbatim, and what it forbids.
2. Set `license` and `source` in the asset's `.asset.json`. `AssetRegistry`
   preserves both fields across rewrites, so the value is durable.
3. If the terms restrict selling or redistribution, state the consequence for
   templates and for the installer, not just the abstract rule.
