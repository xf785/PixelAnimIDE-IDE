"""把一组同尺寸像素素材拼成 i×j 展示图（纯白背景，NEAREST 放大保持硬边）。

用于 README 顶部的素材总览，例如某次生成的道具包：

    python tools/make_showcase_sheet.py docs/screenshots/tree/tree_*.png \
        --out docs/screenshots/showcase-trees.png --cols 4 --rows 2 --cell 160

默认 4×2、每格 160px（32px 素材即 5 倍最近邻放大）、格间距 20px、外边距 20px，
输出一律为不透明白底 PNG（贴图带 alpha，按掩码合成到白底上）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

BG = (255, 255, 255, 255)


def build_sheet(paths: list, cols: int, rows: int, cell: int, gap: int, margin: int) -> Image.Image:
    need = cols * rows
    if len(paths) != need:
        raise SystemExit(f"需要 {cols}×{rows} = {need} 张图片，实际给了 {len(paths)} 张")
    width = margin * 2 + cols * cell + (cols - 1) * gap
    height = margin * 2 + rows * cell + (rows - 1) * gap
    sheet = Image.new("RGBA", (width, height), BG)
    for index, path in enumerate(paths):
        with Image.open(path) as raw:
            sprite = raw.convert("RGBA").resize((cell, cell), Image.Resampling.NEAREST)
        col, row = index % cols, index // cols
        x = margin + col * (cell + gap)
        y = margin + row * (cell + gap)
        sheet.alpha_composite(sprite, (x, y))
    return sheet


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="按行列顺序排列的素材图片")
    ap.add_argument("--out", required=True, help="输出 PNG 路径")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cell", type=int, default=160, help="每格边长（NEAREST 放大后的像素数）")
    ap.add_argument("--gap", type=int, default=20, help="格间距")
    ap.add_argument("--margin", type=int, default=20, help="外边距")
    args = ap.parse_args()

    paths = [Path(p) for p in args.images]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        print("ERROR: missing input files: " + ", ".join(missing), file=sys.stderr)
        return 1

    sheet = build_sheet(paths, args.cols, args.rows, args.cell, args.gap, args.margin)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.convert("RGB").save(out, "PNG", optimize=True)
    print(f"wrote {out}  {sheet.width}x{sheet.height}  {out.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
