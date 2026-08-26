# PixelAnimIDE

A pixel animation IDE — an all-in-one desktop tool: **text → image generation → video animation → strict pixelization → background removal → sprite-frame export**.

- **Tech stack**: Python 3.10+ · PySide6 · Pillow · numpy · httpx · cryptography · imageio-ffmpeg
- **Status**: Phase 1–4 completed (MVP / IDE / sprite workflow / standalone pixel board / i18n / CI / release) — see [Roadmap](#roadmap)

## Quick Start

```bash
# 1. Create a virtual environment and install dependencies (Windows)
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# 2. Launch the GUI
python main.py

# 3. No API keys needed: full pipeline with deterministic mock APIs (demo mode)
python main.py --demo
```

> Demo mode writes `demo_output/` with GIF, PNG frames, metadata and project files.

## UI Language

The UI supports **Chinese / English**. Switch in **Settings → General → Language** (takes effect after restart). Untranslated strings fall back to Chinese; the translation table lives in `ui/i18n.py` — add entries there to extend coverage.

## Solo Mode (one-click)

1. Open **Settings** and configure the three APIs (LLM / Image / Video), or tick **"Use mock API (no key)"** for offline use.
2. In **Solo** mode, enter a description on the **Parameters** tab (e.g. "an orange kitten holding a sword, chibi, side view"), pick an action and output options; optionally upload a reference image card for **image-to-image (i2i)**.
3. Click **Start generating** — the pipeline runs: prompts → first frame (i2i when a reference exists) → animation → pixelization → background removal → export.
4. The right-side preview shows the result with a **playback speed** dropdown (0.5x/1x/1.5x/2x/3x).
5. Click **Sync to IDE** to import the **first frame** and **final frame sequence** into the IDE workspace for fine editing.

## IDE Mode (step-by-step)

The **Solo | IDE | Sprites | Pixel** 2×2 icon switch in the top-left opens **IDE** (the left rail expands). IDE splits the Solo pipeline into 6 independently runnable steps:

1. Step nav: text → image → video animation → pixelize → background removal → export; each step can be re-run; the **right parameter panel follows the selected step** and is **collapsed by default** (one chevron to expand); the **bottom log box can be collapsed/expanded**.
2. **No full pipeline needed**: import your own image as **reference / first frame**, then go straight to the **Animation** step (image used as first frame) or the **Image** step (i2i).
3. The **Prompts** tab lets you hand-edit generated prompts; **Preview** plays the animation (zoom with −/＋/**Fit** and the mouse wheel, cursor-focus, crisp NEAREST sampling, playback speed); **Edit** edits pixels.
4. Bottom **timeline**: click to select, drag to reorder, insert/duplicate/delete/append frames.
5. **Pixel editor**: canvas fills the panel; all controls live in the right icon column (self-drawn DSH-style icons, left-click picks a tool, right-click opens second-level options: brush size / selection mode / fill mode / background) and a collapsible bottom color-family bar. Four background modes (checker/white/black/green), pixel grid toggle, Ctrl+wheel cursor-focus zoom, Ctrl+left-drag pan, right-drag region fill. **Selection tool**: rect/lasso/Ctrl+click multi-select → **Ctrl+C copy → Ctrl+V paste as semi-transparent floating layer** → **Ctrl+right-drag move (any tool)** → **Ctrl+M merge**.
6. **Color-family palette + right-click color wheel**: colors are clustered into families (White / Red / Light-red…); top 6 families + "…" dialog (hover for names); right-click replaces a whole family preserving the inner gradient; **right-click-and-hold on the canvas** opens a Krita-style color wheel (hue ring + saturation/value square + recent colors).
7. **Save** writes frames + prompts + params (`frames/` + `ide_project.json`), **Open** restores.

## Sprite Mode

**Sprites** generates a grid sprite sheet with text-to-image only (no video):

1. Enter a description, optional action loop, **frame count**, **grid i×j** (e.g. 4×4=16), cell size and max colors.
2. Click **Generate sprite sheet** — the pipeline runs: text→**base image** → one img2img call producing the **whole i×j grid sheet** → algorithmic **crop into frames** (row-major, auto-inset removes AI cell border lines) → **loop close** (last = first) → **one-click keying** → export GIF / PNG sequence / keyed sheet + metadata.
3. Live preview tabs: base image / sprite sheet / frame sequence.
4. **Sync to IDE** imports the base image (as first frame) + cropped frames into the IDE workspace.

## Pixel Mode (standalone pixel board)

The 4th mode **Pixel** is a standalone pixel canvas (reuses the full editor):

1. **New canvas**: preset (16–512) or custom W×H + background (transparent/white/black).
2. **Sync from IDE** pulls the IDE current frame/first frame for fine pixel editing.
3. **Sync to IDE** uses the canvas image as **first frame + i2i reference**.
4. **Use as video first frame** sends it to Solo for image-to-video — if the resolution is below the API minimum it is **NEAREST-upscaled** (crisp pixels, no blur); otherwise sent as-is. Controlled by `video_image_min_side` (default 256, long edge) paired with `video_image_max_side` (default 512).
5. **Export PNG**.

## Features

| Module | Description |
|--------|-------------|
| API config management | Multiple configs per API type: CRUD, default, connectivity test, JSON import/export |
| Encrypted keys | `cryptography` Fernet, no plaintext on disk |
| LLM / Image / Video APIs | OpenAI-compatible httpx wrapper with unified timeout/retry/error handling; configurable endpoints; video polling task model |
| Solo workflow | Full auto pipeline + progress/log/cancel, local fallback prompts when LLM fails |
| Perfect pixelization | Frame-0 grid detection (FFT + purity/boundary candidate search), exact per-cell sampling for all frames, frequency-based palette; non-pixel art auto-skipped |
| Pixel-style image gen | Pixel keywords force preset pixel resolution (long edge max(pixel_size, 256)) |
| i2i reference image | Solo & IDE text-to-image support a reference image (Doubao/Jimeng-style small card); `image_field` and `image_mode` (data URI vs multipart, gpt.ge auto) configurable; unsupported sizes auto-retry at larger standard sizes |
| IDE reference/first frame | Import your own image and go straight to Animation (as first frame) or Image (i2i) |
| LLM auto-tuning | LLM suggests frame_count/fps from the action; applied only when the user hasn't customized |
| Loop close | Frame extraction keeps first+last with even middle sampling; last frame forced = first; `last_frame` passes the first frame as last to the API |
| Playback speed | 0.5x–3x, calibrated to actual video duration |
| Silent video | Audio track stripped with ffmpeg `-c copy` (no re-encode) |
| Background stability | Animation prompts force "background stays pure white, unchanged per frame"; `last_frame` stabilizes the endpoints |
| Subject integrity | Prompts force "subject fully visible, centered, clear margin, never cropped/touching edges" |
| Forced solid background | Prompt + adaptive background normalization (pale subject → black, else white); precise mask keying |
| Background removal | Color key + tolerance + **shrink (removes white fringe)** + feather; IDE **live keying preview dialog** with tolerance/shrink/feather |
| Export | GIF (transparent), **APNG**, PNG sequence, **sprite sheet**, JSON metadata, project files |
| Dual-resolution export | Pixel-art output exports both native grid resolution and the user-preset resolution, sharing one palette |
| IDE workspace | Step nav / center preview+edit+prompts / right params (step-aware, collapsible) / bottom timeline + log |
| Pixel editor | Pencil/Eraser/Eyedropper/Fill/Select, undo/redo, integer zoom + grid, cursor-focus wheel zoom, Ctrl+left pan, **local import/export images**, background modes, right icon column + collapsible palette bar |
| Color-family palette | Colors clustered by RGB distance into families; right-click replaces a whole family preserving the gradient |
| Right-click color wheel | Krita-style: hue ring + S/V square + recent colors, live preview, release commits |
| Selection & layers | Rect/lasso/Ctrl+click select; Ctrl+C copy → Ctrl+V semi-transparent floating layer → Ctrl+right-drag move (any tool, auto-lifts a selection) → Ctrl+M merge (alpha, undoable) |
| Sprite workflow | Text-only grid sprite sheet: base image → one-call i×j sheet → crop (auto-inset removes black borders) → loop close → keying → export |
| Standalone pixel board | 4th mode with resolution settings, IDE sync both ways, video-first-frame handoff, PNG export |
| i18n | Chinese/English UI (Settings → General → Language); `ui/i18n.py` translation table |
| UI scale | Settings → General → UI scale (0.8×–1.5×): fonts **and** all fixed UI sizes scale together |

**Token savings**:
- First-frame images sent to the video API are scaled to ≤ `video_image_max_side` (default 512) before upload;
- Prompt-generation LLM calls use `max_tokens` 800;
- Image size prefers the API default size, configurable smaller;
- Polling uses minimal GETs.

## Provider Adaptation

Provider differences are configured in **Settings**, no code changes:

- **Presets**: one-click fill Base URL / model / adapter params (DeepSeek, Kimi, Zhipu, SiliconFlow, Ark, DashScope, Hunyuan, Ollama, gpt.ge, Kling…); with an API key set, "Query models" lists available models.
- **LLM / Image** (OpenAI-compatible): fill the provider root URL; if the path differs (404 Invalid URL), set the **endpoint path** or a full URL override. **i2i upload mode**: data URI by default; gpt.ge requires **multipart file upload** (`image_mode=multipart`, auto-enabled for `api.gpt.ge`). **Size fallback**: rejected small sizes auto-retry at larger standard sizes (512/768/1024/1536).
- **Video**: `generic` (OpenAI-compatible polling), **Doubao Seedance (Ark)**, **gpt.ge V-API** presets; request-body templates with `$model/$prompt/$image/$frames/$fps/$duration` placeholders; `submit_url`/`poll_url` support `{base}`/`{id}`; polling fields configurable; **`last_frame`** sends the first frame as the last frame for first/last consistency.
- **Proxy / SSL**: advanced options per API — proxy URL for blocked networks; `verify_ssl` toggle; network errors auto-retry with hints.

## Directory

```
PixelAnimIDE/
├── main.py                 # entry (GUI + --demo)
├── requirements.txt
├── config/                 # global + API config (encrypted keys)
├── core/
│   ├── api/                # BaseAPI / LLM / Image / Video / Mock / factory
│   ├── workflow/           # Solo / IDE / sprite workflows
│   ├── processing/         # pixelizer / background / frame_utils / prompt_utils
│   ├── editing/            # pixel canvas model (draw/undo/selection/paste)
│   └── storage/            # keyring, project files
├── ui/                     # main window, pages, widgets, QSS + DSH-style icons
│   ├── i18n.py             # zh/en translation table (tr())
│   ├── dialogs/            # settings, background key preview
│   ├── pages/              # solo / ide / sprite / pixel
│   └── widgets/            # image_viewer / pixel_editor / color_wheel / timeline / api_config_widget / reference_box
├── assets/prompts.json     # preset action prompt library (extensible)
└── tests/                  # unit + e2e + GUI smoke tests
```

## Testing

```bash
.\.venv\Scripts\python.exe -m pytest -v
```

Covers: pixelization, background removal, frame utils (real mp4 extraction), keyring, config management, API clients (httpx MockTransport), mock clients, Solo e2e, IDE workflow, canvas editing, i18n, GUI smoke.

## Roadmap

- [x] Phase 1 (MVP): Solo pipeline + pixelization + white keying + GIF/PNG export
- [x] Phase 2: IDE workspace (step nav / preview+edit / params / timeline), pixel canvas editing, project save/load
- [x] Phase 3: APNG/sprite export, onion skin, palette lock, background params, Krita-style palette & color wheel, selection/layers, sprite workflow, standalone pixel board, i18n, UI scale
- [x] Phase 4: PyInstaller packaging + GitHub Actions CI + GitHub Release (v0.1.0)
- [ ] Phase 5+ (future): map tile generation & tile-map editing, Solo performance/quality, pixel editor & sprite refinements, continuous polish — see [**ROADMAP.en.md**](ROADMAP.en.md) / [**ROADMAP.md**](ROADMAP.md) (milestones M1–M4)

## Notes

- Runtime config & keys live in the user data dir (Windows: `%APPDATA%\PixelAnimIDE\`).
- Video providers differ; adapt via config (endpoints, polling, status fields), see `core/api/video_api.py`.
- User assets and outputs are stored locally.
- **Open-source/commercial compliance**: nav icons come from DeepSeek Harness (MIT License, Copyright (c) 2026 DeepSeek) — keep the attribution, see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). This project is not affiliated with DeepSeek.
