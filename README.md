# PixelFoundry IDE

> **PixelFoundry — Pixel Game Asset Foundry** · 像素铸造 IDE
> 中文文档：[README_CN.md](README_CN.md) · UI 语言可在 设置 → 常规 → 语言 切换

[![CI](https://github.com/xf785/PixelFoundry-IDE/actions/workflows/ci.yml/badge.svg)](https://github.com/xf785/PixelFoundry-IDE/actions/workflows/ci.yml)
[![Release](https://img.shields.io/badge/release-v1.0.0-blue.svg)](https://github.com/xf785/PixelFoundry-IDE/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)]()

**One-click pixel animation IDE** — a free, open-source desktop tool that turns **text or your own image into crisp, game-ready pixel assets**:

> text → image generation → video animation → **strict pixelization** → background removal → GIF / APNG / PNG sequences / sprite sheets

Built for indie game devs, pixel artists, and AI tinkerers who want *real* pixel art — sharp edges and exact colors — not blurry downscaled images.

<p align="center">
  <img src="docs/screenshots/showcase-trees.png" alt="Prop assets from a single run: 8 tree types (32x32, pure white background)" width="430"/>
  <img src="docs/screenshots/showcase-npcs.png" alt="Character assets from a single run: 8 NPCs (32x32, pure white background)" width="430"/>
</p>
<p align="center"><em>Prop and character assets generated in a single run: 8 tree types + 8 NPCs, true 32×32 pixel art on a pure white
background — key them out and they are game-ready (raw files and metadata live in <code>docs/screenshots/tree/</code> and
<code>docs/screenshots/npc/</code>).</em></p>

<p align="center">
  <img src="docs/screenshots/01-solo-pipeline.png" alt="PixelFoundry — Solo one-click pipeline (UI shown in Chinese)" width="880"/>
</p>
<p align="center"><em>Main window (Solo one-click pipeline): parameters on the left, live preview with the three prompts in the
middle, results and log on the right.</em></p>

## ✨ Highlights

- 🚀 **One-click Solo pipeline** — text → animation → pixel art, fully automated. Works offline with deterministic mock APIs, no keys required.
- 🎯 **Perfect-pixel engine** — frame-0 grid detection (FFT + purity/boundary search) + exact per-cell sampling on every frame, so colors stay precise and edges stay hard.
- 🎬 **IDE step workspace** — 6 independently runnable steps, step-aware parameter panel, frame timeline, live keying preview.
- 🧩 **Sprite sheets** — one img2img call produces a whole i×j grid sheet, auto-cropped, loop-closed and keyed.
- 🖌️ **Krita-style pixel editor** — color-family palette with right-click whole-family replace, right-click color wheel, selection & floating layers, onion skin, palette lock.
- 🖼️ **Standalone pixel board** — 4th mode with resolution settings, two-way IDE sync, and video-first-frame handoff (NEAREST upscale, never blurry); Krita-style resizable docks plus a **tile/prop pack browser** that browses **every folder level** inside a pack (atlases, individual tiles, textures, raw sheets), with category switcher, search, thumbnails, place/replace on canvas and auto-restore |
- 🌐 **Bilingual + scalable UI** — Chinese / English, UI scale 0.8×–1.5×, self-drawn DSH-style icons.
- 🔌 **Provider-agnostic** — one-click presets for DeepSeek, Kimi, Zhipu, SiliconFlow, Ark, DashScope, Hunyuan, Ollama, gpt.ge, Kling… plus proxy & SSL options.
- 📦 **Open source** — MIT license, CI on GitHub Actions (Win + Linux), Windows releases via PyInstaller.

## 🚀 Quick Start

```bash
# 1. Create a virtual environment and install dependencies (Windows)
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# 2. Launch the GUI
python main.py

# 3. No API keys? Run the full pipeline with deterministic mock APIs
python main.py --demo
```

> Demo mode writes `demo_output/` containing a GIF, PNG frame sequence, metadata and project files.

## 🌐 UI Language

The UI supports **Chinese / English** — switch in **Settings → General → Language** (restart to apply). Untranslated strings fall back to Chinese; the translation table lives in `ui/i18n.py` (extend it to cover more).

## 🕹️ The Four Modes

### Solo Mode — one-click pipeline

1. In **Settings**, configure the three APIs (LLM / Image / Video), or tick **"Use mock API (no key)"** for offline use.
2. Type a description on the **Parameters** tab (e.g. *"an orange kitten holding a sword, chibi, side view"*), pick an action & output options; optionally upload a **reference image** card for **image-to-image (i2i)**.
3. Hit **Start generating**: prompts → first frame (i2i when a reference exists) → animation → pixelization → background removal → export.
4. The right-side preview supports **playback speed** (0.5x/1x/1.5x/2x/3x).
5. **Sync to IDE** imports the first frame + final frame sequence into the IDE workspace for fine editing.

### IDE Mode — step-by-step control

The top-left **2×2 icon switch** (Solo ✦ / IDE ⊞ / Sprites ▤ / Pixel ▦) opens **IDE** (left rail expands). The Solo pipeline is split into **6 independently runnable steps** — text → image → animation → pixelize → background removal → export:

- The **right parameter panel follows the selected step** and is **collapsed by default** (one chevron to expand); the bottom log box can be collapsed/expanded.
- **No full pipeline needed**: import your own image as **reference / first frame**, then jump straight to the Animation step (image as first frame) or the Image step (i2i).
- **Prompts** tab for hand-editing; **Preview** plays the animation (cursor-focus wheel zoom, crisp NEAREST sampling, speed control); **Edit** edits pixels.
- Bottom **timeline**: click to select, drag to reorder, insert/duplicate/delete/append frames.
- **Pixel editor**: canvas fills the panel; controls live in a right icon column (left-click = tool, right-click = second-level options) and a collapsible bottom color-family bar. Four background modes (checker/white/black/green), grid toggle, Ctrl+wheel cursor-focus zoom, Ctrl+left-drag pan, **right-drag region fill**.
- **Selection & layers**: rect / lasso / Ctrl+click multi-select → **Ctrl+C copy → Ctrl+V paste as a semi-transparent floating layer** → **Ctrl+right-drag move (any tool)** → **Ctrl+M merge** (Esc cancels).
- **Color families + color wheel**: colors auto-cluster into families (White / Red / Light-red…); right-click a family to replace it wholesale **preserving the inner gradient**; right-click-and-hold on the canvas opens a **Krita-style color wheel** (hue ring + S/V square + recent colors).
- **Save** writes frames + prompts + params (`frames/` + `ide_project.json`); **Open** restores.

### Sprite Mode — grid sprite sheets (text-only, no video)

1. Enter a description, optional action loop, **frame count**, **grid i×j** (e.g. 4×4 = 16), cell size and max colors.
2. **Generate sprite sheet**: text → **base image** → one img2img call producing the **whole i×j grid sheet** → algorithmic **crop into frames** (row-major, auto-inset removes AI cell border lines) → **loop close** (last = first) → one-click **keying** → export GIF / PNG sequence / keyed sheet + metadata.
3. Live preview tabs: base image / sprite sheet / frame sequence.
4. **Sync to IDE** imports the base image (as first frame) + cropped frames for per-frame pixel polish.

### Pixel Mode — standalone pixel board

A dedicated pixel canvas (reusing the full editor):

- **New canvas**: preset (16–512) or custom W×H + background (transparent/white/black).
- **Sync from IDE** pulls the current IDE frame for fine pixel editing; **Sync to IDE** sends the canvas as **first frame + i2i reference**.
- **Use as video first frame** hands it to Solo for image-to-video — if below the API minimum size it is **NEAREST-upscaled** (hard edges, no blur) via `video_image_min_side` (default 256) paired with `video_image_max_side` (default 512).
- **Export PNG**.
- **Krita-style three-column layout**: left *Assets* docker | canvas | right *Canvas* docker — **every
  splitter handle is draggable**, each dock collapses into a slim vertical tab (instant bigger canvas),
  docker headers fold, and widths/collapsed states are remembered. The common actions (import /
  sync from IDE / sync to IDE / use as first frame / export PNG) live in the **top toolbar**, so they
  no longer eat sidebar width.
- **Tile / prop packs**: import `.tilepack`, exported zips, exported folders — or a **plain image
  folder without a manifest**. **Every folder level inside a pack is browsable**: expand `atlas/`
  (47-tile sheets), `tiles/terrain_1/` (each individual tile), `textures/`, `source/` (raw AI sheets),
  `pieces/`, `props/` and clicking any level filters the thumbnail grid. Combined with the category
  switcher (all / terrain / buildings / props / sheets / atlases / tiles / textures) plus search, then
  **Place on canvas** (centred, canvas size kept) or **Replace canvas**. Loaded packs are remembered.

<p align="center">
  <img src="docs/screenshots/02-pixel-canvas-pack-browser.png" alt="Standalone pixel canvas: in-pack folder browsing plus the colour-family palette" width="880"/>
</p>
<p align="center"><em>Standalone pixel canvas: the left docker browses <b>every folder level inside a pack</b>
(all packs → terrain → atlas / tiles / textures / source / pieces / props; clicking a level filters the thumbnails),
the right docker holds canvas settings, canvas info and export, and the colour-family palette floats on top.</em></p>

<p align="center">
  <img src="docs/screenshots/03-pixel-canvas-props.png" alt="Prop pack: the trees under props/ previewed one by one and placed on the canvas" width="880"/>
</p>
<p align="center"><em>Prop pack: every tree under <code>props/</code> previews individually — one click places it on the canvas
(the 8 trees at the top of this README came from exactly this pack).</em></p>

### Tilemap Mode — text-to-tileset with seamless stitching

The fifth mode: **one text prompt → a complete, game-ready tileset + playable map preview**.

- **Reference image (i2i)**: feed your own reference image into the tileset generation to lock palette/style across the whole set.
- **Top-down 2.5D**: two planned layers — the existing 47-tile terrain (top faces) plus a **cliff 16-tile family**
  derived from the same art (no extra generation); the cliff face is drawn on the *lower* tile just south of a plateau,
  so plateaus read as standing on the ground. Paint height (raise/lower) live in the preview.
- **Terrain ecosystems (地块生态)**: the AI paints one 2×2×3 sheet (base terrain + 3 features, e.g. lake / mud / rocks); algorithms strip grid frames, detect text marks, splice the texture into a **wrap-equal seamless** tile and derive the whole tile family procedurally.
- **Aligned composition**: band depth, layered outline, bevel and ambient shadow are measured from the AI art once, then composed per mask — so adjacent tiles are **pixel-identical along their shared edges** and long walls/runs never show per-tile seams.
- **Edge noise + boundary percolation (交界融合)**: irregular inward noise on non-interior edges, plus **block-noise percolation** where two terrains (even two *different* tile packs) meet — the two sides interlock instead of showing a hard line.
- **Buildings (建筑类)**: the wall **16-tile family** (straight / corner / inner+outer corner / T / cross / end / isolated) with fixed cross-section geometry, transparent exterior for overlay compositing, procedural 1px outline, top face + front shading, plus AI-derived **door** and **pillar** pieces.
- **Props (素材/道具)**: trees, flowers, rocks… generated as a variant grid, background removed automatically (pure-white or pure-black key chosen from the prompt: light subjects such as snow get a black key), alpha hardened to 0/255, bottom-aligned so props "stand" on the ground; placement is **scalable** (25–400 %).
- **Tile packs**: export a **complete tileset folder + zip** (47-tile atlas 8×6, every single tile, all metadata, prop/piece PNGs,
  **plus the raw AI source sheets** under `source/`); import folders, zips or `.tilepack` into any preview.
- **Map preview**: open it **without generating anything**, load several packs, paint terrain / buildings / props together, toggle the grid, **Ctrl+left-drag to pan**, wheel to zoom, and generate a **Perlin-noise big world** (up to 400×400) with optional scattered buildings.
- **Dual-grid** and **16-tile** families are supported for both generation and preview.

<p align="center">
  <img src="docs/screenshots/04-tilemap-ecosystem-generate.png" alt="Tileset parameters and the generated terrain ecosystem sheet (base terrain + 3 features, grid frames stripped)" width="880"/>
</p>
<p align="center"><em>1 · Terrain ecosystem: fill in the terrain plus three features (lake / tall grass / rocky outcrop) → a 2×2×3 sheet is generated with the grid frames already stripped; an optional reference image keeps the palette consistent.</em></p>

<p align="center">
  <img src="docs/screenshots/07-map-preview-terrains.png" alt="Map preview painting four terrains with percolated boundaries" width="880"/>
</p>
<p align="center"><em>2 · Map preview: four terrains painted side by side — lake, flowers and rocks blend into the base terrain with percolation noise instead of a hard line.</em></p>

<p align="center">
  <img src="docs/screenshots/06-map-preview-dialog.png" alt="Map preview brushes: terrain, 16-tile auto walls, rotation and scale" width="880"/>
</p>
<p align="center"><em>3 · Brushes: terrain paint, <b>auto-stitched 16-tile walls</b> (the rectangle is one drag, corners chosen automatically), rotation, 25–400 % scaling and the grid toggle; the footer loads/saves tile packs and generates a Perlin-noise big world.</em></p>

<p align="center">
  <img src="docs/screenshots/05-tile-editor-dialog.png" alt="Tile editor: pick one of the 3x3 sheet cells and repaint it" width="880"/>
</p>
<p align="center"><em>4 · Tile editor: repaint any cell of the AI sheet (the centre cell drives the seamless terrain texture) with palette + pixel tools.</em></p>

## 📋 Feature Table

| Module | Description |
|--------|-------------|
| API config management | Multiple configs per API type: CRUD, default, connectivity test, JSON import/export |
| Encrypted keys | `cryptography` Fernet, no plaintext on disk |
| LLM / Image / Video APIs | OpenAI-compatible httpx wrapper, unified timeout/retry/error handling, configurable endpoints, video polling task model; **fully custom mode** (JSON request-body templates with `$prompt/$model/$image/$size/$frames`… placeholders, custom response field paths, request method, extra headers, multi-line editors) — connect any non-OpenAI-compatible service |
| Solo workflow | Full auto pipeline + progress/log/cancel; local fallback prompts when the LLM fails |
| Perfect pixelization | Frame-0 grid detection (FFT + purity/boundary candidate search), exact per-cell sampling on all frames, frequency-based palette; non-pixel art auto-skipped; graceful fallback |
| Pixel-style image gen | Pixel keywords force a preset pixel resolution (long edge max(pixel_size, 256)) |
| i2i reference image | Solo & IDE text-to-image support a reference card (Doubao/Jimeng style); `image_field` / `image_mode` (data URI vs multipart, gpt.ge auto); unsupported sizes auto-retry at 512/768/1024/1536 |
| IDE reference / first frame | Import your own image and jump straight to Animation (as first frame) or Image (i2i) |
| LLM auto-tuning | LLM suggests `frame_count`/`fps` from the action; applied only when the user hasn't customized |
| Loop close | Extraction keeps first+last frames with **content-aware middle sampling** (greedy farthest-point — skips static holds, keeps the most distinct poses); consecutive identical frames auto-deduped; last frame forced = first; `last_frame` passes the first frame as the last to the API |
| Playback speed | 0.5x–3x, calibrated to actual video duration |
| Silent video | Audio stripped with ffmpeg `-c copy` (no re-encode) |
| Background stability | Animation prompts force a stable pure-white background; `last_frame` stabilizes the endpoints |
| Subject integrity | Prompts force the subject fully visible, centered, with clear margins — never cropped or touching edges |
| Forced solid background | Prompt + adaptive background normalization (pale subject → black, else white); precise mask keying |
| Background removal | Color key + tolerance + shrink (removes white fringe) + feather; **tiered tolerance modes** (inspired by FrameRonin): `contiguous` (only border-connected bg removed — protects interior white pixels), `hybrid` (big tolerance on connected bg, small tolerance inside the subject), `adaptive` (large disconnected regions get a tolerance bonus); IDE **live keying preview** dialog applied to all frames |
| Export | GIF (transparent), **APNG**, PNG sequence, **sprite sheet + FrameRonin-style index JSON** (per-frame x/y/w/h + timestamps, spacing/orientation/auto-square layout), JSON metadata, project files |
| Dual-resolution export | Pixel-art output exports both the native grid resolution and the user preset, sharing one palette (identical colors) |
| IDE workspace | Step nav / center preview+edit+prompts / right step-aware params (collapsible) / bottom timeline + collapsible log |
| Timeline | Frame thumbnails, click-select, drag-reorder, insert/duplicate/delete/append blank frame |
| Pixel editor | Pencil/Eraser/Eyedropper/Fill/**Line/Rectangle/Ellipse**/Select, undo/redo (Ctrl+Z / Ctrl+Shift+Z), integer zoom + grid + checkerboard, cursor-focus wheel zoom, Ctrl+left pan, **local import/export images**, background modes, scrollable right icon column + collapsible palette bar |
| Shape tools | Line / Rectangle / Ellipse with a live drag preview that commits on release (one undo per shape); right-click a rectangle/ellipse for **stroke / fill**, right-click a line for brush thickness; default keys B/E/I/G/L/U/O/M |
| Symmetry drawing | Left-click cycles off → horizontal → vertical → four-quadrant mirror, right-click picks the axis; the axis is shown in orange and every stroke is mirrored (great for characters and icons) |
| Wrap-around drawing | Strokes that cross a canvas edge continue on the opposite side — **seamless tiles** (grass, brick, floor) line up in one pass |
| Canvas transform | Flip H/V, rotate 90° either way, **crop to selection**, canvas resize (content anchored centre/corner, new area transparent), integer nearest-neighbour content scaling (2×/4×…, never blurry) — all undoable |
| Export | Right-hand *Export* docker: **1×/2×/4×/8× nearest-neighbour upscale** to PNG, or one-click **copy to clipboard** for pasting into an engine or a document |
| Palette interop | The family dialog can **export / import GIMP `.gpl` palettes** (plain `R G B` text works too); an imported palette is locked automatically so drawing snaps to it — round-trips with Aseprite / Krita / GIMP |
| Tool icons | Self-drawn 16px flat-line icons (DSH style, recolorable); left-click = tool, right-click = second-level options (brush size / shape style / selection mode / fill / background) |
| Brush size | 1–8 px square brushes for pencil/eraser |
| Selection & floating layer | Rect / lasso / Ctrl+click multi-select with a **screen-space blue dashed border** (1px cosmetic pen, crisp at any zoom); Ctrl+C → Ctrl+V semi-transparent floating layer → Ctrl+right-drag move (any tool, auto-lifts a selection) → Ctrl+M merge (alpha, undoable) |
| Region fill | Right-drag a rectangle and release to fill it with the current color (undoable); quick right-click/hold still opens the color wheel |
| Color-family palette | Colors clustered by RGB distance (White / Red / Light-red…); top 6 families + "…" dialog (compact swatch grid, names on hover); right-click replaces a whole family preserving the inner gradient |
| Right-click color wheel | Krita-style: hue ring + saturation/value square + recent-color bar (last 10), live preview, release commits, Esc cancels (snaps when palette locked) |
| Preview zoom | 0.2x–8x zoom/fit with cursor-focus wheel zoom, percentage shown; NEAREST sampling keeps pixels crisp |
| Preview speed | Playback speed in Solo **and** IDE preview (0.5x/1x/1.5x/2x/3x), applies live |
| Onion skin | Semi-transparent overlay of adjacent frames while editing |
| Palette lock | Drawing/filling snaps to the nearest locked color; one-click extract palette from the current frame |
| Project persistence | IDE workspace save/load (`frames/` + `ide_project.json`) |
| UI scale | Settings → General → UI scale (0.8×/1.0×/1.25×/1.5×) — fonts **and** all fixed UI sizes scale together |
| Custom shortcuts | Settings → **Shortcuts** with **two-level navigation in the left category list** (click to open the form, click the item again to expand the Solo/IDE/Sprites/Pixel submenu right below it, ▸/▾ arrow, current mode highlighted; click again to collapse while the form stays): two-level dropdown (category → action), press-to-record, conflict warning, reset per-action / all; each mode has its own key set (IDE: play/pause, fit, timeline insert/duplicate/delete; Solo/Sprites: play/pause, fit; Pixel editor: undo/redo/copy/paste/merge, selection, zoom, tools) — applies immediately |
| Dark mode switch | Settings → General: **iOS-style toggle switch** (pill track + white knob, click to toggle, 150ms eased slide, #007AFF on / #D0D0D0 off, dark-theme variant) replaces the theme dropdown |
| Solo → IDE sync | First frame + final frame sequence imported into the IDE workspace in one click |
| Sprite generation | Text-only grid sheet with **Auto / Manual toggle in the left rail** (slide left = auto, right = manual; manual: 7 steps, each step can be rerun or continued): **high-res base image (1024×1024) used as-is for i2i** → one-call i×j sheet (strong built-in prompt: uniform cells, first/last pose identical, character never mutates) → crop (row-major, auto `cell_inset` removes AI black borders) → **Perfect-Pixel dual resolution** (native grid size + user size, NEAREST upscale, shared palette) → loop close → keying; exports **both** resolutions in three formats: PNG sequence / **algorithmically re-composited grid sheet** (+ index JSON) / GIF |
| Sprite → IDE sync | Base image (as first frame) + cropped frames imported for pixel polish, then IDE export |
| Standalone pixel board | 4th mode: resolution settings, IDE sync both ways, video-first-frame handoff, PNG export; **Krita-style three-column docker layout** (collapsible side docks, draggable splitters, remembered widths) and a **tile/prop pack browser** that browses every folder inside a pack |
| Krita-style shell | **Menu bar** (File / Edit / View / Workspace / Help: new canvas, open/save project, undo/redo, copy/paste/merge selection, theme, UI scale, fullscreen, reset layout, about) + a **contextual toolbar** whose actions follow the active workspace (each page exposes `toolbar_actions()`) + a status bar showing UI scale / theme |
| Dockers | Every parameter/resource column is a collapsible docker (title + icon + fold chevron); **collapsing frees space** — a folded docker keeps only its title bar and the freed height is handed to the other expanded dockers in the same column (stacked fill), restored to its previous height when re-opened; panels are drag-resizable and a whole dock collapses into a 22px vertical tab |
| i18n | Chinese/English UI (Settings → General → Language); `ui/i18n.py` translation table |
| Token savings | First-frame images ≤ `video_image_max_side` (512) before upload; LLM `max_tokens` 800; image size prefers the API default; minimal GET polling |

## 🔌 Provider Adaptation

Provider differences are configured in **Settings** — no code changes:

- **Presets**: one-click fill Base URL / model / adapter params (DeepSeek, Kimi, Zhipu, SiliconFlow, Ark, DashScope, Hunyuan, Ollama, gpt.ge, Kling…). With an API key set, **Query models** lists available models.
- **LLM / Image** (OpenAI-compatible): provider root URL; if the path differs (`404 Invalid URL`), set the **endpoint path** or a full URL override. **i2i upload mode**: data URI by default; gpt.ge requires **multipart file upload** (`image_mode=multipart`, auto-enabled for `api.gpt.ge`). **Size fallback**: rejected sizes auto-retry at 512/768/1024/1536.
- **Video**: `generic` (OpenAI-compatible polling), **Doubao Seedance (Ark)**, **gpt.ge V-API** presets; request-body templates with `$model/$prompt/$image/$frames/$fps/$duration` placeholders; `submit_url`/`poll_url` support `{base}`/`{id}`; polling fields configurable; **`last_frame`** sends the first frame as the last for first/last consistency.
- **Proxy / SSL**: per-API advanced options — proxy URL for blocked networks, `verify_ssl` toggle; network errors auto-retry with troubleshooting hints.

## 📁 Project Layout

> **Tech stack**: Python 3.10+ / **PySide6 (Qt 6 Widgets)** desktop GUI with QSS themes (`ui/styles/dark.qss`, `light.qss`),
> Pillow + NumPy for image processing and plain `requests` API clients. No web engine anywhere: the menu bar, toolbar,
> docker panels and the pixel canvas are native Qt widgets / hand-painted (icons are QPainter vector drawings that recolour with the theme).

```
PixelFoundry/
├── main.py                 # entry (GUI + --demo)
├── requirements.txt
├── config/                 # global + API config (encrypted keys)
├── core/
│   ├── api/                # BaseAPI / LLM / Image / Video / Mock / factory
│   ├── workflow/           # Solo / IDE / sprite workflows
│   ├── processing/         # pixelizer / background / frame_utils / prompt_utils
│   ├── editing/            # pixel canvas model (draw / shapes / symmetry / wrap / transforms / undo / selection / paste)
│   └── storage/            # keyring (encrypted), project files
├── ui/                     # main window, pages, widgets, QSS + DSH-style icons
│   ├── i18n.py             # zh/en translation table (tr())
│   ├── pages/              # solo / ide / sprite / pixel
│   └── widgets/            # image_viewer / pixel_editor / color_wheel / timeline / api_config_widget / reference_box
├── assets/prompts.json     # preset action prompt library (extensible)
├── docs/screenshots/       # UI screenshots used in the docs
└── tests/                  # unit + e2e + GUI smoke tests
```

## ✅ Testing

```bash
.\.venv\Scripts\python.exe -m pytest -v
```

Covers: pixelization, background removal, frame utils (real mp4 extraction), keyring, config management, API clients (httpx MockTransport), mock clients, Solo e2e, IDE workflow, canvas editing, i18n, GUI smoke.

## 🗺️ Roadmap

Phase 1–4 shipped (MVP / IDE / sprite workflow / standalone pixel board / i18n / CI / release **v0.1.0**). Up next:

- **Map tiles & tile-map editor** (tileset generation + a 5th tile-map mode)
- **Solo performance & quality** (caching, multi-candidate generation, quality baselines)
- **Pixel editor & sprite refinements** (more tools, real layer stack, in-page sprite editing)
- **Continuous polish** (packaging, auto-update, docs, community)

See [**ROADMAP.md**](ROADMAP.md) (English) / [**ROADMAP_CN.md**](ROADMAP_CN.md) (中文) for milestones M1–M4.

## 📄 Notes & License

- Runtime config & keys live in the user data dir (Windows: `%APPDATA%\PixelFoundry\`).
- Video providers differ; adapt via config (endpoints, polling, status fields) — see `core/api/video_api.py`.
- User assets and outputs are stored locally by default.
- **Open-source / commercial compliance**: nav icons come from DeepSeek Harness (MIT License, Copyright (c) 2026 DeepSeek) — keep the attribution, see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). This project is not affiliated with DeepSeek.
- Licensed under the [MIT License](LICENSE). Copyright (c) 2026 StrFaith.
