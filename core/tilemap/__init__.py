"""core.tilemap —— 瓦片地图模式核心（第 5 种模式）。

模块：
- prompts.py   瓦片集严格提示词（3×3 基础 / 2×2 地块生态图 / 2×2 建筑图）
- tiles.py     3×3 自适应裁切、2×2 块图裁切、九宫格/生态/建筑数据结构
- seamless.py  无缝化算法（偏移错位缝合：纹理全向 / 单轴；地块组预处理）
- autotile.py  47-tile 生成（程序化构图 + 九宫格艺术片构图）、双网格、多地形掩码
- buildings.py 建筑类：抠白底、墙体无缝、透明拼件（直段/端头/转角/立柱）
- map.py       瓦片地图数据模型（多地形 + 建筑 overlay）与渲染、序列化
"""
from .tiles import (
    BASE_POS_CHOICES,
    BLOCK_POSITIONS,
    BaseTileSet,
    BuildingSheet,
    EcosystemSheet,
    block_base_score,
    building_from_blocks,
    crop_base_3x3,
    crop_blocks,
    detect_base_block,
    ecosystem_from_blocks,
    normalize_tileset,
)
from .seamless import (
    make_axis_seamless,
    make_edge_seamless,
    make_texture_seamless,
    prepare_terrain_set,
    process_base_set,
)
from .autotile import (
    BIT,
    build_47_sheet,
    build_47_sheet_art,
    build_dual_pieces_sheet_art,
    canonical_mask,
    canonical_masks,
    compose_art_tile,
    compose_tile,
    derive_corners,
    mask_for_terrain,
    mask_from_neighbors,
    nearest_mask,
)
from .buildings import PIECE_NAMES, opaque_ratio, process_building_sheet, rotate_piece
from .map import TileMapModel
from .prompts import (
    build_building_prompts,
    build_ecosystem_prompts,
    build_tileset_prompts,
)

__all__ = [
    "BaseTileSet",
    "EcosystemSheet",
    "BuildingSheet",
    "crop_base_3x3",
    "crop_blocks",
    "ecosystem_from_blocks",
    "building_from_blocks",
    "normalize_tileset",
    "BASE_POS_CHOICES",
    "BLOCK_POSITIONS",
    "block_base_score",
    "detect_base_block",
    "make_texture_seamless",
    "make_axis_seamless",
    "make_edge_seamless",
    "prepare_terrain_set",
    "process_base_set",
    "BIT",
    "canonical_mask",
    "canonical_masks",
    "nearest_mask",
    "compose_tile",
    "compose_art_tile",
    "build_47_sheet",
    "build_47_sheet_art",
    "build_dual_pieces_sheet_art",
    "derive_corners",
    "mask_for_terrain",
    "mask_from_neighbors",
    "PIECE_NAMES",
    "opaque_ratio",
    "process_building_sheet",
    "rotate_piece",
    "TileMapModel",
    "build_tileset_prompts",
    "build_ecosystem_prompts",
    "build_building_prompts",
]
