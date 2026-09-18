# PixelFoundry IDE — Roadmap

> Audience: maintainers and prospective contributors. This is a living document, updated as the project evolves.
> 中文版（Chinese）: [ROADMAP_CN.md](ROADMAP_CN.md)。

## 1. Vision

Turn the full "AI generation → pixelization → polish → game assets" flow into a **pixel-first, all-in-one tool**:

- Input: text prompt / your own reference image / image-to-video
- Output: pixel animations (GIF/APNG/PNG sequences), sprites, **map tiles & tile maps**, game-engine-ready assets

## 2. Guiding Principles

1. **Pixel-first**: every operation (scaling / keying / clustering / seamless tiling) must keep pixels crisp and sharp — "Perfect Pixel" is this project's brand promise.
2. **Demo offline**: the Mock API must always run the full pipeline, so new users can try it with zero setup.
3. **Provider-agnostic**: the API adapter layer isolates providers; adding one means configuration, not code changes.
4. **Quality over quantity**: every feature ships at "pixel-grade" completion — no half-finished pile-ups.
5. **Sustainable maintenance**: tests, CI, i18n, and docs evolve together with features.

## 3. Current State (v0.3.0)

| Module | Status |
|--------|--------|
| Solo one-click pipeline (text→image→video→pixelize→key→export) | ✅ Working (reference i2i, loop closing, background stability, multi-provider adapters, size auto-fallback) |
| IDE step workspace (6 steps, timeline, per-step params panel) | ✅ Working |
| Pixel editor (4 tools + selection/layers, color families, color wheel, import/export) | ✅ Working |
| Sprite workflow (grid sheet → crop → key → export, IDE sync) | ✅ Working |
| Standalone pixel board (resolution settings, two-way sync, video first-frame) | ✅ Working |
| **Tilemap mode (5th mode, v0.2–v0.3 main line)** | ✅ Working (see below) |
| zh/en i18n + UI scaling + DSH-style icons | ✅ Working |
| CI (GitHub Actions, Py3.11/3.13 × Win/Linux), **527** tests | ✅ Running |
| Windows packaging (PyInstaller onedir) + GitHub Release | ✅ v0.3.0 |

### Tilemap mode (shipped)

- **Terrain ecosystems**: one prompt → 2×2×3 sheet (base terrain + 3 features); automatic grid-frame
  stripping, text detection (constant-stroke-width signature) and 9-cell median texture to kill one-off
  text, offset-quilt splicing for **pixel-exact wrap equality**, then **aligned composition** per mask so
  adjacent tiles are pixel-identical along shared edges; band depth / layered outline / bevel / AO measured
  from the AI art; irregular inward noise on non-interior edges; **block-noise percolation** across
  terrain — and across *different tile packs* — removes hard boundaries.
- **Buildings**: wall **16-tile family** (straight / corner / inner+outer corner / tee / cross / end /
  isolated) with fixed cross-section geometry, transparent exterior for overlay compositing, procedural
  1px outline, top-face + front shading, AI-derived doors and pillars, plus a **live auto-stitching wall
  layer** in the map view.
- **Props**: variant-grid generation → background key chosen from the prompt (pure white, or pure black
  for light subjects such as snow) with range deletion + border flood fill → alpha hardened to 0/255 →
  trimmed and bottom-aligned; placement is scalable (25–400 %).
- **Tile packs & export**: complete tileset folder + zip (47-tile atlas 8×6, every single tile,
  textures/pieces/props, all metadata, README); import from folder / zip / `.tilepack`.
- **Map preview**: usable **without generating first**; mix several packs (terrain / buildings / props),
  grid toggle, Ctrl+left-drag pan, wheel zoom, brush / eraser / rotate / scale; **Perlin-noise big world**
  (up to 400×400, optional scattered buildings, mask-cached tile composition).
- **Top-down 2.5D**: a top-face layer (47-tile family) plus a cliff layer (16-tile family derived from the terrain art);
  height can be painted live in the preview, and long cliffs stay continuous tile-to-tile.
- **Reference image (i2i)**: tileset generation accepts a reference image to keep palette/style consistent.
- **Layout compatibility**: 47-tile (8×6 blob convention), 16-tile wall family, and **dual grid** for both
  generation and preview.
- 64 algorithmic invariants locked down by tests (shared-edge pixel equality, no text/frame residue,
  47-class coverage, transparent exteriors, zero background residue, deterministic output).

Known tech debt: large onedir package (~250 MB), some workflow logs not yet i18n'd, GUI details not fully
covered by tests, no Tiled `.tmx/.tsx` import/export yet, map layering limited to terrain + overlay + wall.

## 4. Phase Plan

### Phase A: Map Tile Generation + Tile-Map Editor (main body done in v0.2–v0.3 ✅)

**A1 Tile-set generation** ✅ Done
- Text-to-image for a 2×2×3 terrain-ecosystem sheet (base terrain + 3 features), a 2×2 building sheet and
  prop variant grids, with built-in prompts for equal cells, solid background, **seamless tiling**,
  consistent style, and no text/frames/grid lines;
- Frame stripping (expected-position ± tolerance darkest-run search + residual sweep), text detection and
  patching, offset-quilt splicing (wrap-equal), 9-cell median texture; aligned composition derives the whole
  tile family (band / outline / bevel / AO measured from the AI art);
- Export a **complete tileset folder + zip** (47-tile atlas 8×6, every single tile, metadata), compatible
  with common Tiled layouts.
- DoD met: adjacent tiles are pixel-identical along shared edges; long runs and walls show no per-tile seam.

**A2 Tile-map editor** ✅ Done (5th mode)
- Tile-grid canvas with multi-pack mixing (terrain / buildings / props), brush / eraser / rotate / scale,
  grid toggle, Ctrl+left-drag pan;
- **Live auto-stitching wall layer** (16-tile piece chosen from the four neighbours), building overlay layer,
  transparent-exterior compositing;
- **Perlin-noise big world** preview (procedural continents / rivers / mountains, optional scattered
  buildings) and **dual-grid** rendering;
- Export PNG preview + map JSON (overlay name/rotation/scale, wall layer, dual flag).
- TODO: Tiled `.tmx/.tsx` import/export, flood fill / rectangle / multi-layer stack.

**A3 Pipeline integration** ✅ Mostly done
- Tile packs (folder/zip) load into any preview and mix freely; one pack format for terrain, buildings and
  props;
- TODO: push tiles/maps back into IDE and pixel mode for polishing.

**A4 Large-map performance** 🚧 Partial
- Done: tile composition cached by (terrain art, mask) — a 96×64 map renders in about a second; cross-pack
  blending and the wall layer are vectorized;
- TODO: viewport rendering, async export of very large maps, lazy tileset loading.

### Phase B: Solo Performance & Generation Quality

**B1 Performance**
- **Result caching**: hash prompts/images; repeated generations reuse results (saves tokens and time);
- **Parallel pixelization**: numpy vectorization + multi-frame thread/process pools;
- **Fewer tokens**: compress LLM templates (already max_tokens 800), keep min/max side for first-frame images, downsize image requests when possible;
- Async large-image preview/export with finer progress granularity.

**B2 Quality**
- **Multi-candidate generation**: `n=2–4` outputs scored objectively (sharpness / grid purity / subject completeness), best auto-picked, user can pick from the UI;
- **Prompt template upgrade**: few-shot examples + parameterized style presets (pixel style / palette / outline strength);
- **Video consistency**: per-frame color-histogram matching after sampling, better loop start/end detection;
- **Keying upgrade**: optional outline pass, edge anti-aliasing, feathering;
- **Seed control**: reproducible generation.

**B3 Quality regression baseline**
- A sample image set + objective scoring script; every change runs the baseline to prevent quality regressions (optional CI job).

### Phase C: Pixel Editor & Sprite Refinements

**C1 Pixel editor**
- More tools: line / rectangle / ellipse / symmetry / magic wand / pixel-font text;
- **Real layer stack** (replacing the single floating layer) + layers panel;
- In-editor animation preview (play frame sequences inline), onion-skin strength control;
- Customizable shortcuts, pattern brushes.

**C2 Sprite**
- **In-page per-frame editing** (click a cell to edit directly, no IDE sync first);
- Crop improvements: content bounding box + configurable padding, border-removal params UI;
- Multi-action sheets (one sheet, multiple action rows), collision-box annotation;
- Multi-candidate generation picker.

### Phase D: Continuous Polish (throughout)

- **D1 Engineering**: PyInstaller onefile + icon + version self-check; optional auto-update; CI packaging job + codecov;
- **D2 UX**: empty-state onboarding, shortcut help, simplified settings, friendlier error messages, log levels & search;
- **D3 i18n**: 100% English coverage, more language packs (e.g. Japanese), font adaptation;
- **D4 Docs**: user manual (zh/en), sample gallery, API adapter docs, FAQ;
- **D5 Community**: issue templates, Contributing guide, first external contribution.

## 5. Milestones

| Milestone | Scope | Status |
|-----------|-------|--------|
| **M1 (v0.2)** | A1 tile sets + A2 tile-map MVP + packaging | ✅ Released |
| **M2 (v0.3)** | 47/16/dual-grid algorithm work, building 16-tile family, prop pipeline, tile-pack import/export, boundary percolation blending, Perlin big world, preview-without-generating, screenshot gallery | ✅ Released |
| **M3** | Tiled `.tmx/.tsx` import/export, map layer stack + rectangle/fill tools, A3 polish round-trip, A4 viewport/async export | Next release |
| **M4** | B1 result caching/parallelism + B2 multi-candidate & prompt templates + C2 inline sprite editing | After M3 |
| **M5** | Stabilization, 100 % i18n, D1 auto-update, community, **v1.0** | Stable release |

## 6. Quick Wins (highest ROI first)

1. Packaging (onefile / drop pycache / size) — benefits every release;
2. Solo result caching — the most direct time/money saver for users;
3. Tile-set generation (A1) — fully reuses the sprite pipeline: low dev cost, big feature win;
4. In-editor animation preview + onion-skin strength — daily editor UX;
5. Multi-candidate generation — biggest perceived quality gain.

## 7. How to Track

- Create GitHub **Milestones (M1–M4)** and **Labels**: `tiles` / `map-editor` / `solo-quality` / `performance` / `pixel-editor` / `sprite` / `i18n` / `packaging` / `docs` / `good-first-issue`;
- Every PR links to an Issue; milestones are decomposed from this document;
- Quality bar: any change must pass `pytest` (currently 527 tests) without regressions.

---

*Last updated: 2026-09-17 (updated alongside v0.3.0: tilemap mode main body shipped)*
