"""core.tilemap —— 瓦片地图模式核心（第 5 种模式）。

模块：
- prompts.py   瓦片集严格提示词（文生 3×3 基础九宫格）
- tiles.py     3×3 自适应裁切、规格归一、九宫格数据结构
- seamless.py  无缝化算法（纹理全向无缝 / 墙面轴向无缝 / 转角推导）
- autotile.py  47-tile 瓦片集生成（8 邻域位掩码 + 四分块构图）、双网格导出
- map.py       瓦片地图数据模型与渲染（铺设、自动衔接、预览合成）
"""
from .tiles import BaseTileSet, crop_base_3x3, normalize_tileset
from .seamless import (
    make_edge_seamless,
    make_texture_seamless,
    process_base_set,
)
from .autotile import (
    BIT,
    build_47_sheet,
    compose_tile,
    derive_corners,
    mask_from_neighbors,
)
from .map import TileMapModel

__all__ = [
    "BaseTileSet",
    "crop_base_3x3",
    "normalize_tileset",
    "make_texture_seamless",
    "make_edge_seamless",
    "process_base_set",
    "BIT",
    "compose_tile",
    "build_47_sheet",
    "derive_corners",
    "mask_from_neighbors",
    "TileMapModel",
]
