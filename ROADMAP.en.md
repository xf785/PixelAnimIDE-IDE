# PixelAnimIDE Roadmap

> Audience: maintainers and prospective contributors. This is a living document, updated as the project evolves.
> 中文版见 [ROADMAP.md](ROADMAP.md)（Chinese version: [ROADMAP.md](ROADMAP.md)）。

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

## 3. Current State (v0.1.0, released)

| Module | Status |
|--------|--------|
| Solo one-click pipeline (text→image→video→pixelize→key→export) | ✅ Working (reference i2i, loop closing, background stability, multi-provider adapters, size auto-fallback) |
| IDE step workspace (6 steps, timeline, per-step params panel) | ✅ Working |
| Pixel editor (4 tools + selection/layers, color families, color wheel, import/export) | ✅ Working |
| Sprite workflow (grid sheet → crop → key → export, IDE sync) | ✅ Working |
| Standalone pixel board (resolution settings, two-way sync, video first-frame) | ✅ Working |
| zh/en i18n + UI scaling + DSH-style icons | ✅ Working |
| CI (GitHub Actions, Py3.11/3.13 × Win/Linux) | ✅ Running |
| Windows packaging (PyInstaller) + GitHub Release | ✅ v0.1.0 |

Known tech debt: large onedir package (~260 MB), some workflow logs not yet i18n'd, `__pycache__` accidentally bundled, GUI details not fully covered by tests.

## 4. Phase Plan

### Phase A: Map Tile Generation + Tile-Map Editor (new feature line)

**A1 Tile-set generation** (reuses the sprite pipeline)
- Text-to-image for an i×j tile set (terrain / objects / decorations) with a strong built-in prompt: equal cells, **seamlessly tileable**, solid background, consistent style, no text or border lines;
- Algorithmic crop + one-click keying + **seamlessness pass** (edge sampling blend so all four edges tile);
- Export **PNG tileset + JSON index** (Tiled `.tsx`-compatible).
- DoD: a default 16-tile terrain sheet tiles seamlessly with no seams.

**A2 Tile-map editor** (a 5th mode)
- Canvas as a tile grid with a right-side tile palette (reusing family-palette interactions: left-click select, right-click batch replace);
- Brush / rectangle / fill / eraser, tile flip & rotate, **multiple layers** (ground / objects / decoration), collision markers;
- Reuse pixel-editor capabilities for grid / zoom / undo / pan;
- Export **PNG + JSON** (plus Tiled `.tmx` import/export).
- DoD: paint a 32×32 three-layer map and open the export in Tiled.

**A3 Pipeline integration**
- Tiles / maps sync into IDE and pixel mode for polish; sprite animations can be placed in a map preview.

**A4 Large-map performance**
- Lazy-load tilesets >4096, viewport-rendered map editor, async large-map export.

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

| Milestone | Scope | Target |
|-----------|-------|--------|
| **M1 (v0.2)** | A1 tile-set + A2 tile-map editor MVP + B1 caching/parallel + C1 line/ellipse/symmetry + onefile packaging | next major release |
| **M2** | B2 multi-candidate & template upgrade + C2 in-page sprite editing + D4 docs/sample gallery | after M1 |
| **M3** | A3/A4 pipeline integration & large-map perf + C1 layer stack + D1 auto-update | after M2 |
| **M4** | stability polish, 100% i18n, community ops, **v1.0** | stable release |

## 6. Quick Wins (highest ROI first)

1. Packaging (onefile / drop pycache / size) — benefits every release;
2. Solo result caching — the most direct time/money saver for users;
3. Tile-set generation (A1) — fully reuses the sprite pipeline: low dev cost, big feature win;
4. In-editor animation preview + onion-skin strength — daily editor UX;
5. Multi-candidate generation — biggest perceived quality gain.

## 7. How to Track

- Create GitHub **Milestones (M1–M4)** and **Labels**: `tiles` / `map-editor` / `solo-quality` / `performance` / `pixel-editor` / `sprite` / `i18n` / `packaging` / `docs` / `good-first-issue`;
- Every PR links to an Issue; milestones are decomposed from this document;
- Quality bar: any change must pass `pytest` (currently 333 tests) without regressions.

---

*Last updated: 2026-08-26 (created alongside v0.1.0 release)*
