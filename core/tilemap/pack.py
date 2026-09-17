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
from typing import Dict, List, Optional, Tuple

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
    manifest = {
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
            zf.writestr(f"terrain/{int(tid)}.png", _png_bytes(tset.center))
            base = tset.base_texture if tset.base_texture is not None else tset.center
            zf.writestr(f"terrain/{int(tid)}_base.png", _png_bytes(base))
        art = pack.wall_art
        if art is not None:
            zf.writestr("walls/texture.png", _png_bytes(art.texture))
            for key in ("top", "door", "pillar"):
                img = getattr(art, key)
                if img is not None:
                    zf.writestr(f"walls/{key}.png", _png_bytes(img))
        for name, piece in pack.pieces.items():
            zf.writestr(f"pieces/{name}.png", _png_bytes(piece))
    logger.info("瓦片包已保存: %s（地形 %d、拼件 %d）", path, len(pack.terrains), len(pack.pieces))
    return path


# --------------------------------------------------------------------------- #
# 读盘
# --------------------------------------------------------------------------- #
def load_tilepack(path) -> TilePack:
    """读取瓦片包（zip）。"""
    path = Path(path)
    with zipfile.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        if manifest.get("format") != PACK_FORMAT:
            raise ValueError(f"不是瓦片包文件: {path}")
        terrains: Dict[int, BaseTileSet] = {}
        names: Dict[int, str] = {}
        for item in manifest.get("terrains", []):
            center = Image.open(io.BytesIO(zf.read(item["center"]))).convert("RGBA")
            base = Image.open(io.BytesIO(zf.read(item["base_texture"]))).convert("RGBA")
            tid = int(item["id"])
            terrains[tid] = BaseTileSet(
                size=center.size[0],
                center=center,
                edges={n: center for n in EDGE_NAMES},
                corners={n: center for n in ("tl", "tr", "bl", "br")},
                line_color=tuple(item.get("line_color", (0, 0, 0))),
                line_width=int(item.get("line_width", 1)),
                band=int(item.get("band", 0)),
                radius=int(item.get("radius", 0)),
                base_texture=base,
                art_meta=dict(item.get("art_meta", {})),
            )
            names[tid] = str(item.get("name", f"terrain{tid}"))
        art_meta = manifest.get("walls")
        wall_art = None
        if art_meta:
            files = art_meta.get("files", {})

            def _img(key: str) -> Optional[Image.Image]:
                rel = files.get(key)
                if not rel or rel not in zf.namelist():
                    return None
                return Image.open(io.BytesIO(zf.read(rel))).convert("RGBA")

            wall_art = WallArt(
                size=int(art_meta.get("size", manifest.get("tile_size", 32))),
                texture=_img("texture") or Image.new("RGBA", (32, 32), (150, 140, 128, 255)),
                top=_img("top"), door=_img("door"), pillar=_img("pillar"),
                margin=int(art_meta.get("margin", 8)),
                outline=tuple(art_meta.get("outline", (34, 30, 32))),
                corner_radius=int(art_meta.get("corner_radius", 0)),
                top_h=int(art_meta.get("top_h", 3)),
                front_h=int(art_meta.get("front_h", 3)),
                shade=float(art_meta.get("shade", 0.8)),
                lighten=float(art_meta.get("lighten", 1.12)),
                edge_noise=int(art_meta.get("edge_noise", 0)),
                art_meta=dict(art_meta.get("art_meta", {})),
            )
        pieces: Dict[str, Image.Image] = {}
        for name in manifest.get("pieces", []):
            rel = f"pieces/{name}.png"
            if rel in zf.namelist():
                pieces[name] = Image.open(io.BytesIO(zf.read(rel))).convert("RGBA")
    pack = TilePack(
        name=str(manifest.get("name", path.stem)),
        category=str(manifest.get("category", "ground")),
        tile_size=int(manifest.get("tile_size", 32)),
        terrains=terrains,
        terrain_names=names,
        base_terrain=(int(manifest["base_terrain"]) if manifest.get("base_terrain") is not None else None),
        wall_art=wall_art,
        pieces=pieces,
        atlas_mode=str(manifest.get("atlas_mode", "47")),
        meta=dict(manifest.get("meta", {})),
    )
    logger.info("瓦片包已加载: %s（类别 %s、地形 %d、拼件 %d）", path.name, pack.category,
                len(pack.terrains), len(pack.pieces))
    return pack


# --------------------------------------------------------------------------- #
# 从会话建包
# --------------------------------------------------------------------------- #
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
