"""把 PyInstaller 产物打成发布 zip（ZIP 条目统一用正斜杠，跨平台解压友好）。

用法：
    python tools/package_release.py                 # 输出到 release/
    python tools/package_release.py --out dist_zip  # 自定输出目录
    python tools/package_release.py --tag v1.0.0    # 用指定 tag 命名（默认读 APP_VERSION）

产物：<out>/PixelFoundry-<tag>-win64.zip，内含顶层目录 PixelFoundry/（解压即用）。
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import APP_NAME, APP_VERSION  # noqa: E402

DIST = ROOT / "dist" / APP_NAME


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="release", help="输出目录（默认 release/）")
    ap.add_argument("--tag", default=f"v{APP_VERSION}", help="版本标签（默认 v{APP_VERSION}）")
    args = ap.parse_args()

    if not DIST.is_dir():
        print(f"找不到构建产物：{DIST}（先跑 pyinstaller PixelFoundry.spec）", file=sys.stderr)
        return 1

    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{APP_NAME}-{args.tag}-win64.zip"
    if zip_path.exists():
        zip_path.unlink()

    files = sorted(p for p in DIST.rglob("*") if p.is_file())
    print(f"打包 {len(files)} 个文件 -> {zip_path.relative_to(ROOT)}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            zf.write(path, path.relative_to(DIST.parent).as_posix())

    size_mb = zip_path.stat().st_size / 1e6
    print(f"完成：{size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
