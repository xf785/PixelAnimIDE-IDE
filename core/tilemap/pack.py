"""瓦片包（Tile Pack）：把一次生成的瓦片集存成单文件，之后可再次加载叠加预览。

一个包 = 一份 zip：
- `manifest.json`：格式/版本/类别/瓦片尺寸/地形清单/建筑参数/实测艺术参数（art_meta）；
- `terrain/<id>.png`：该地形的**特征纹理**（`BaseTileSet.center`）；
- `terrain/<id>_base.png`：该地形的**另一方地形纹理**（`base_texture`，地块类用于条带）；
- `walls/*.png`：建筑类的墙纹理 / 顶面 / 门 / 立柱 + 整族拼件（可选，供叠加预览）。

之所以存纹理而不是只存图集：预览要按掩码**重新构图**（`compose_art_tile` /
`compose_wall_piece`），需要纹理 + 实测参数（条带宽度、描边分层、转角半径、边缘噪声）。
地块包与建筑包可以同时加载到一个预览里，从而看到「地块 + 建筑」的综合效果。
"""
from __future__ import annotations

import io
import json
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

from .tiles import EDGE_NAMES, BaseTileSet
from .walls import WallArt

logger = logging.getLogger("PixelAnimIDE.tilemap.pack")

PACK_FORMAT = "pixel-anim-tilepack"
PACK_VERSION = 1
PACK_SUFFIX = ".tilepack"


@dataclass
class TilePack:
    """一个可保存/加载的瓦片包。"""

    name: str
    category: str                                   # ground / building / classic
    tile_size: int
    terrains: Dict[int, BaseTileSet] = field(default_factory=dict)
    terrain_names: Dict[int, str] = field(default_factory=dict)
    base_terrain: Optional[int] = None
    wall_art: Optional[WallArt] = None
    pieces: Dict[str, Image.Image] = field(default_factory=dict)
    atlas_mode: str = "47"
    meta: Dict = field(default_factory=dict)

    @property
    def has_terrain(self) -> bool:
        return bool(self.terrains)

    @property
    def has_walls(self) -> bool:
        return bool(self.pieces) or self.wall_art is not None


# --------------------------------------------------------------------------- #
# 写盘
# --------------------------------------------------------------------------- #
def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _pack_wall_art(art: Optional[WallArt]) -> Optional[dict]:
    if art is None:
        return None
    return {
        "size": art.size,
        "margin": art.margin,
        "outline": list(art.outline),
        "corner_radius": art.corner_radius,
        "top_h": art.top_h,
        "front_h": art.front_h,
        "shade": art.shade,
        "lighten": art.lighten,
        "edge_noise": art.edge_noise,
        "art_meta": art.art_meta,
        "files": {
            "texture": "walls/texture.png",
            "top": "walls/top.png" if art.top is not None else None,
            "door": "walls/door.png" if art.door is not None else None,
            "pillar": "walls/pillar.png" if art.pillar is not None else None,
        },
    }


def save_tilepack(path, pack: TilePack) -> Path:
    """把瓦片包写成一个 zip 文件（返回实际路径）。"""
    path = Path(path)
    if path.suffix != PACK_SUFFIX:
        path = path.with_suffix(PACK_SUFFIX)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _manifest_dict(pack) or {
        "format": PACK_FORMAT,
        "version": PACK_VERSION,
        "name": pack.name,
        "category": pack.category,
        "tile_size": int(pack.tile_size),
        "atlas_mode": pack.atlas_mode,
        "base_terrain": pack.base_terrain,
        "meta": pack.meta,
        "terrains": [
            {
                "id": int(tid),
                "name": pack.terrain_names.get(tid, f"terrain{tid}"),
                "band": int(tset.band),
                "radius": int(tset.radius),
                "line_color": list(tset.line_color),
                "line_width": int(tset.line_width),
                "art_meta": tset.art_meta,
                "center": f"terrain/{int(tid)}.png",
                "base_texture": f"terrain/{int(tid)}_base.png",
            }
            for tid, tset in pack.terrains.items()
        ],
        "walls": _pack_wall_art(pack.wall_art),
        "pieces": sorted(pack.pieces.keys()),
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for tid, tset in pack.terrains.items():
            zf.writestr(f"textures/terrain_{int(tid)}.png", _png_bytes(tset.center))
            base = tset.base_texture if tset.base_texture is not None else tset.center
            zf.writestr(f"textures/terrain_{int(tid)}_base.png", _png_bytes(base))
        art = pack.wall_art
        if art is not None:
            zf.writestr("walls/texture.png", _png_bytes(art.texture))
            for key in ("top", "door", "pillar"):
                img = getattr(art, key)
                if img is not None:
                    zf.writestr(f"walls/{key}.png", _png_bytes(img))
        sub = "props" if pack.category == "prop" else "pieces"
        for name, piece in pack.pieces.items():
            zf.writestr(f"{sub}/{name}.png", _png_bytes(piece))
    logger.info("瓦片包已保存: %s（地形 %d、拼件 %d）", path, len(pack.terrains), len(pack.pieces))
    return path


# --------------------------------------------------------------------------- #
# 读盘
# --------------------------------------------------------------------------- #
def load_tilepack(path) -> TilePack:
    """读取瓦片包（旧的 `.tilepack` zip / 导出目录 / 导出 zip 都能读）。"""
    return load_tileset(path)


def pack_from_session(session, name: str = "") -> TilePack:
    """从 `TilemapSession` 生成瓦片包（地块类存各地形纹理，建筑类存墙纹理+拼件）。"""
    params = getattr(session, "params", None)
    category = getattr(params, "category", "classic")
    tile_size = int(getattr(params, "tile_size", 32) or 32)
    pack = TilePack(
        name=name or (getattr(params, "description", "") or "tilepack"),
        category=category,
        tile_size=tile_size,
        atlas_mode=str(getattr(params, "atlas_mode", "47")),
        meta={"wall_thickness": getattr(params, "wall_thickness", None),
              "edge_noise": getattr(params, "edge_noise", None)},
    )
    if category == "ground" and getattr(session, "terrain_sets", None):
        features = list(getattr(params, "features", {}) or {})
        eco = getattr(session, "ecosystem", None)
        base_pos = getattr(eco, "base_pos", None) if eco is not None else None
        for tid, tset in session.terrain_sets.items():
            pack.terrains[int(tid)] = tset
            if int(tid) == 1:
                pack.terrain_names[1] = str(getattr(params, "description", "") or "base")
            else:
                idx = int(tid) - 2
                pack.terrain_names[int(tid)] = features[idx] if idx < len(features) else f"feature{tid}"
        pack.base_terrain = 1
        pack.meta["base_block_pos"] = base_pos
    elif category == "classic" and getattr(session, "processed", None) is not None:
        processed = session.processed
        pack.terrains[1] = processed
        pack.terrain_names[1] = str(getattr(params, "description", "") or "terrain")
        pack.base_terrain = 1
    if getattr(session, "pieces", None):
        pack.pieces = dict(session.pieces.get("pieces", {}))
        pack.wall_art = session.pieces.get("art")
    return pack

# --------------------------------------------------------------------------- #
# 完整瓦片集目录导出 / 导入（含 47 图集、逐张瓦片、全部元信息；可打包成 zip）
# --------------------------------------------------------------------------- #
def _mask_list_for(tset: BaseTileSet) -> Dict[str, object]:
    from .autotile import canonical_masks

    return {
        "tile_size": int(tset.size),
        "band": int(tset.band),
        "radius": int(tset.radius),
        "line_color": list(tset.line_color),
        "line_width": int(tset.line_width),
        "art_meta": tset.art_meta,
        "reachable_masks": canonical_masks(),
    }


def export_tileset_dir(
    out_dir,
    pack: TilePack,
    *,
    make_zip: bool = True,
    atlas_modes: Sequence[str] = ("47", "blob47"),
) -> Dict[str, Path]:
    """导出**完整瓦片集目录**（不是单一后缀文件）：

    ```
    <out_dir>/
      manifest.json                  # 与瓦片包同一 schema（可再次导入）
      README.txt                     # 用法说明
      textures/terrain_<id>.png      # 地形特征纹理（供再次导入/再构图）
      textures/terrain_<id>_base.png # 另一方地形纹理（地块类的条带）
      atlas/terrain_<id>_47.png      # 47-tile 图集（8×6）+ 同名 .json（掩码→槽位）
      atlas/terrain_<id>_blob47.png  # FrameRonin 3×24 布局图集 + .json
      tiles/terrain_<id>/tile_<n>_mask_<m>.png   # 逐张单独瓦片 + index.json
      atlas/walls_16.png + .json     # 建筑：16-tile 族图集
      pieces/<名字>.png              # 建筑拼件 / 素材
      props/<名字>.png               # 素材（道具），附 props.json
      map/*.json                     # 项目信息
    ```
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "textures").mkdir(exist_ok=True)
    (out / "atlas").mkdir(exist_ok=True)
    (out / "tiles").mkdir(exist_ok=True)
    (out / "map").mkdir(exist_ok=True)
    manifest = _manifest_dict(pack)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    from .autotile import build_47_sheet_art
    from .blob47 import build_blob47_atlas

    index: Dict[str, object] = {"format": "pixel-anim-tileset-folder", "tile_size": pack.tile_size,
                                "category": pack.category, "terrains": {}, "atlas_modes": list(atlas_modes)}
    for tid, tset in pack.terrains.items():
        name = pack.terrain_names.get(tid, f"terrain{tid}")
        (out / "textures" / f"terrain_{int(tid)}.png").write_bytes(_png_bytes(tset.center))
        base = tset.base_texture if tset.base_texture is not None else tset.center
        (out / "textures" / f"terrain_{int(tid)}_base.png").write_bytes(_png_bytes(base))
        entry: Dict[str, object] = {"name": name, **_mask_list_for(tset), "atlas": {}, "tiles": {}}
        for mode in atlas_modes:
            if mode == "blob47":
                sheet, meta = build_blob47_atlas(tset)
            else:
                sheet, meta = build_47_sheet_art(tset)
            stem = f"terrain_{int(tid)}_{mode}"
            sheet.save(out / "atlas" / f"{stem}.png")
            (out / "atlas" / f"{stem}.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            entry["atlas"][mode] = f"atlas/{stem}.png"
            if mode == "47":
                slot_to_mask = {int(v): int(k) for k, v in meta["mask_to_index"].items()}
                tdir = out / "tiles" / f"terrain_{int(tid)}"
                tdir.mkdir(parents=True, exist_ok=True)
                tiles: Dict[str, str] = {}
                for slot in range(int(meta["tile_count"])):
                    tile = sheet.crop(((slot % meta["sheet_cols"]) * tset.size,
                                       (slot // meta["sheet_cols"]) * tset.size,
                                       (slot % meta["sheet_cols"] + 1) * tset.size,
                                       (slot // meta["sheet_cols"] + 1) * tset.size))
                    mask = slot_to_mask.get(slot, -1)
                    fname = f"tile_{slot:02d}_mask_{mask:03d}.png"
                    tile.save(tdir / fname)
                    tiles[str(slot)] = f"tiles/terrain_{int(tid)}/{fname}"
                entry["tiles"] = tiles
                (tdir / "index.json").write_text(
                    json.dumps({"mask_to_index": meta["mask_to_index"], "index_to_mask": meta["index_to_mask"],
                                "files": tiles}, ensure_ascii=False, indent=2), encoding="utf-8")
        index["terrains"][str(int(tid))] = entry

    if pack.wall_art is not None:
        artifact = pack.wall_art
        (out / "atlas" / "walls_16.png").parent.mkdir(parents=True, exist_ok=True)
        (out / "textures").mkdir(exist_ok=True)
        (out / "textures" / "wall.png").write_bytes(_png_bytes(artifact.texture))
        for key in ("top", "door", "pillar"):
            img = getattr(artifact, key)
            if img is not None:
                (out / "textures" / f"wall_{key}.png").write_bytes(_png_bytes(img))
        from .walls import build_wall_atlas

        sheet, meta = build_wall_atlas(artifact)
        sheet.save(out / "atlas" / "walls_16.png")
        (out / "atlas" / "walls_16.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        index["walls"] = {"atlas": "atlas/walls_16.png", "slots": meta.get("slots"),
                          "mask8_to_slot": meta.get("mask8_to_slot")}
    if pack.pieces:
        pdir = out / ("props" if pack.category == "prop" else "pieces")
        pdir.mkdir(parents=True, exist_ok=True)
        for name, piece in pack.pieces.items():
            piece.save(pdir / f"{name}.png")
        (pdir / "index.json").write_text(
            json.dumps({n: f"{pdir.name}/{n}.png" for n in sorted(pack.pieces)},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        index["pieces_dir"] = pdir.name
    (out / "map" / "info.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.txt").write_text(
        "PixelAnimIDE 瓦片集导出\n"
        "=======================\n"
        "manifest.json  —— 可再次导入（预览里「添加瓦片包」选本文件夹或其 zip）\n"
        "textures/      —— 地形纹理（特征纹理 + 另一方地形纹理），重新构图用\n"
        "atlas/         —— 47-tile 图集（8×6）、FrameRonin 3×24 布局图集、建筑 16-tile 图集（附 .json 索引）\n"
        "tiles/         —— 逐张单独瓦片（文件名含槽位与掩码）\n"
        "pieces/ props/ —— 建筑拼件 / 素材（透明 PNG）\n"
        "map/info.json  —— 全部元信息（掩码→槽位、拼件清单、实测艺术参数）\n",
        encoding="utf-8",
    )
    paths = {"dir": out}
    if make_zip:
        zpath = out.with_suffix(".zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(out.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(out))
        paths["zip"] = zpath
    logger.info("瓦片集已导出: %s（%d 个地形）", out, len(pack.terrains))
    return paths


def _manifest_dict(pack: TilePack) -> dict:
    return {
        "format": PACK_FORMAT,
        "version": PACK_VERSION,
        "name": pack.name,
        "category": pack.category,
        "tile_size": int(pack.tile_size),
        "atlas_mode": pack.atlas_mode,
        "base_terrain": pack.base_terrain,
        "meta": pack.meta,
        "terrains": [
            {
                "id": int(tid),
                "name": pack.terrain_names.get(tid, f"terrain{tid}"),
                "band": int(tset.band),
                "radius": int(tset.radius),
                "line_color": list(tset.line_color),
                "line_width": int(tset.line_width),
                "art_meta": tset.art_meta,
                "center": f"textures/terrain_{int(tid)}.png",
                "base_texture": f"textures/terrain_{int(tid)}_base.png",
            }
            for tid, tset in pack.terrains.items()
        ],
        "walls": (_pack_wall_art(pack.wall_art) or None),
        "pieces": sorted(pack.pieces.keys()),
        "prop_kind": "props" if pack.category == "prop" else "pieces",
    }


def load_tileset(path) -> TilePack:
    """导入瓦片集：支持**导出文件夹**、其 **zip**、以及旧的 `.tilepack`。"""
    p = Path(path)
    if p.is_dir():
        man = p / "manifest.json"
        if not man.exists():
            raise ValueError(f"文件夹里没有 manifest.json: {p}")
        return _load_from_reader(json.loads(man.read_text(encoding="utf-8")),
                                 lambda rel: (p / rel).read_bytes(), p.name)
    with zipfile.ZipFile(p, "r") as zf:
        names = set(zf.namelist())
        if "manifest.json" not in names:
            raise ValueError(f"压缩包里没有 manifest.json: {p}")
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        return _load_from_reader(manifest, zf.read, p.stem)


def _load_from_reader(manifest: dict, read, default_name: str) -> TilePack:
    if manifest.get("format") != PACK_FORMAT:
        raise ValueError("不是瓦片集/瓦片包（manifest.format 不匹配）")
    terrains: Dict[int, BaseTileSet] = {}
    names: Dict[int, str] = {}
    for item in manifest.get("terrains", []):
        center = Image.open(io.BytesIO(read(item["center"]))).convert("RGBA")
        base_item = item.get("base_texture") or item["center"]
        base = Image.open(io.BytesIO(read(base_item))).convert("RGBA")
        tid = int(item["id"])
        terrains[tid] = BaseTileSet(
            size=center.size[0], center=center,
            edges={n: center for n in EDGE_NAMES},
            corners={n: center for n in ("tl", "tr", "bl", "br")},
            line_color=tuple(item.get("line_color", (0, 0, 0))),
            line_width=int(item.get("line_width", 1)),
            band=int(item.get("band", 0)), radius=int(item.get("radius", 0)),
            base_texture=base, art_meta=dict(item.get("art_meta", {})),
        )
        names[tid] = str(item.get("name", f"terrain{tid}"))
    wall_art = None
    walls = manifest.get("walls")
    if walls:
        files = walls.get("files", {})

        def _img(key: str) -> Optional[Image.Image]:
            rel = files.get(key)
            if not rel:
                return None
            try:
                return Image.open(io.BytesIO(read(rel))).convert("RGBA")
            except Exception:  # noqa: BLE001
                return None

        wall_art = WallArt(
            size=int(walls.get("size", manifest.get("tile_size", 32))),
            texture=_img("texture") or Image.new("RGBA", (32, 32), (150, 140, 128, 255)),
            top=_img("top"), door=_img("door"), pillar=_img("pillar"),
            margin=int(walls.get("margin", 8)),
            outline=tuple(walls.get("outline", (34, 30, 32))),
            corner_radius=int(walls.get("corner_radius", 0)),
            top_h=int(walls.get("top_h", 3)), front_h=int(walls.get("front_h", 3)),
            shade=float(walls.get("shade", 0.8)), lighten=float(walls.get("lighten", 1.12)),
            edge_noise=int(walls.get("edge_noise", 0)),
            art_meta=dict(walls.get("art_meta", {})),
        )
    sub = manifest.get("prop_kind", "pieces")
    pieces: Dict[str, Image.Image] = {}
    for name in manifest.get("pieces", []):
        for rel in (f"{sub}/{name}.png", f"pieces/{name}.png", f"props/{name}.png"):
            try:
                pieces[name] = Image.open(io.BytesIO(read(rel))).convert("RGBA")
                break
            except Exception:  # noqa: BLE001
                continue
    return TilePack(
        name=str(manifest.get("name", default_name)), category=str(manifest.get("category", "ground")),
        tile_size=int(manifest.get("tile_size", 32)), terrains=terrains, terrain_names=names,
        base_terrain=(int(manifest["base_terrain"]) if manifest.get("base_terrain") is not None else None),
        wall_art=wall_art, pieces=pieces, atlas_mode=str(manifest.get("atlas_mode", "47")),
        meta=dict(manifest.get("meta", {})),
    )
