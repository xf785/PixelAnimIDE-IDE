"""瓦片包（保存/加载/叠加预览）与柏林噪声大地图测试。"""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from core.tilemap import BIT, TileMapModel, compose_art_tile
from core.tilemap.bigmap import generate_perlin_map, scatter_walls
from core.tilemap.pack import TilePack, load_tilepack, pack_from_session, save_tilepack
from core.tilemap.seamless import align_terrain_set, median_tile_texture
from core.tilemap.tiles import BaseTileSet, crop_blocks, ecosystem_from_blocks
from core.tilemap.walls import build_piece_set, wall_art_from_sheet

S = 32


def _cell(col, size=96):
    a = np.zeros((size, size, 4), np.uint8)
    a[..., :3] = col
    a[..., 3] = 255
    return Image.fromarray(a, "RGBA")


def _eco_art(tile_size=S):
    """从合成底图走真实管线得到一套地形艺术（含 art_meta）。"""
    size = 96 * 6
    img = Image.new("RGBA", (size, size), (236, 240, 246, 255))
    px = img.load()
    for y in range(size):
        for x in range(size):
            blob = min(x, size - 1 - x, y, size - 1 - y) < 96 * 2
            px[x, y] = ((38, 104, 172, 255) if blob and x < 96 * 3 else (236, 240, 246, 255))
    blocks, _ = crop_blocks(img)
    eco = ecosystem_from_blocks(blocks, tile_size=tile_size, feature_names=["f1"])
    raw = eco.terrain_sets()
    base_tex = median_tile_texture(raw[1].all(), tile_size)
    ground = np.median(np.asarray(base_tex.convert("RGB")).reshape(-1, 3), axis=0)
    return {tid: align_terrain_set(ts, base_texture=base_tex, tile_size=tile_size,
                                   ground_rgb=ground, plain=(tid == 1))
            for tid, ts in raw.items()}


class _FakeSession:
    def __init__(self, terrain_sets, category="ground", pieces=None, processed=None):
        from core.workflow.tilemap_workflow import TilemapParams

        self.params = TilemapParams(description="雪原", category=category,
                                    features={"水塘": "pond"}, tile_size=S)
        self.terrain_sets = terrain_sets
        self.pieces = pieces
        self.processed = processed
        self.ecosystem = type("E", (), {"base_pos": "tr"})()


# --------------------------------------------------------------------------- #
# 1) 瓦片包：保存 / 加载 / 往返一致
# --------------------------------------------------------------------------- #
def test_tilepack_roundtrip_terrain(tmp_path):
    arts = _eco_art()
    pack = TilePack(name="雪原包", category="ground", tile_size=S,
                    terrains=arts, terrain_names={1: "雪原", 2: "水塘"}, base_terrain=1)
    path = save_tilepack(tmp_path / "snow", pack)
    assert path.exists() and path.suffix == ".tilepack"
    loaded = load_tilepack(path)
    assert loaded.name == "雪原包" and loaded.category == "ground"
    assert set(loaded.terrains) == set(arts)
    assert loaded.base_terrain == 1
    assert loaded.terrain_names[2] == "水塘"
    # 纹理与实测参数都要还原（否则预览构图会变样）
    for tid, tset in arts.items():
        got = loaded.terrains[tid]
        assert (np.asarray(got.center) == np.asarray(tset.center)).all()
        assert (np.asarray(got.base_texture) == np.asarray(tset.base_texture)).all()
        assert got.band == tset.band and got.line_width == tset.line_width
        assert got.art_meta.get("outline") == tset.art_meta.get("outline")
        assert got.art_meta.get("edge_noise_px") == tset.art_meta.get("edge_noise_px")
    # 加载后构图结果与原艺术完全一致
    mask = BIT["L"] | BIT["R"] | BIT["T"] | BIT["TL"] | BIT["TR"]
    assert (np.asarray(compose_art_tile(loaded.terrains[2], mask))
            == np.asarray(compose_art_tile(arts[2], mask))).all()


def test_tilepack_roundtrip_building(tmp_path):
    def sheet():
        wall = BaseTileSet(size=S, center=_cell((150, 140, 128), S), edges={}, corners={})
        top = BaseTileSet(size=S, center=_cell((186, 176, 160), S), edges={}, corners={})
        door = Image.new("RGBA", (S, S), (255, 255, 255, 255))
        ImageDraw.Draw(door).rectangle((11, 8, 20, 27), fill=(90, 70, 50, 255))
        pillar = Image.new("RGBA", (S, S), (255, 255, 255, 255))
        ImageDraw.Draw(pillar).rectangle((10, 5, 21, 26), fill=(140, 130, 118, 255))
        from core.tilemap.tiles import BuildingSheet

        return BuildingSheet(
            wall=wall, top=top,
            opening=BaseTileSet(size=S, center=door, edges={}, corners={}),
            pillar=BaseTileSet(size=S, center=pillar, edges={}, corners={}),
        )

    art = wall_art_from_sheet(sheet(), tile_size=S)
    pieces = build_piece_set(art)
    pack = TilePack(name="石墙包", category="building", tile_size=S,
                    wall_art=art, pieces=pieces)
    path = save_tilepack(tmp_path / "wall", pack)
    loaded = load_tilepack(path)
    assert loaded.category == "building"
    assert set(loaded.pieces) == set(pieces)
    assert loaded.wall_art is not None
    assert loaded.wall_art.margin == art.margin
    assert loaded.wall_art.outline == art.outline
    for name, piece in pieces.items():
        assert (np.asarray(loaded.pieces[name]) == np.asarray(piece)).all(), name


def test_pack_from_session_and_reject_bad_file(tmp_path):
    arts = _eco_art()
    pack = pack_from_session(_FakeSession(arts), name="会话包")
    assert pack.name == "会话包" and pack.category == "ground"
    assert pack.terrain_names[2] == "水塘" and pack.base_terrain == 1
    path = save_tilepack(tmp_path / "s", pack)
    assert load_tilepack(path).tile_size == S
    bad = tmp_path / "bad.tilepack"
    bad.write_bytes(b"not a zip")
    with pytest.raises(Exception):
        load_tilepack(bad)


# --------------------------------------------------------------------------- #
# 2) 柏林噪声大地图
# --------------------------------------------------------------------------- #
def test_perlin_big_map_is_deterministic_and_uses_all_terrains():
    arts = _eco_art()
    model = TileMapModel(64, 48, tile_size=S)
    for tid, tset in arts.items():
        model.set_terrain(tid, tset)
    model.base_terrain = 1
    kinds = generate_perlin_map(model, seed="雪原", water_id=2, plain_id=1, mountain_id=2)
    grid = np.asarray(model.grid)
    assert grid.shape == (48, 64)
    assert set(np.unique(grid).tolist()) <= {1, 2}
    assert (grid == 1).any() and (grid == 2).any(), "水与平原都应出现"
    assert set(np.unique(kinds).tolist()) <= {0, 1, 2}
    # 同种子可复现；换种子结果不同
    m2 = TileMapModel(64, 48, tile_size=S)
    for tid, tset in arts.items():
        m2.set_terrain(tid, tset)
    m2.base_terrain = 1
    k2 = generate_perlin_map(m2, seed="雪原", water_id=2, plain_id=1, mountain_id=2)
    assert (k2 == kinds).all()
    m3 = TileMapModel(64, 48, tile_size=S)
    for tid, tset in arts.items():
        m3.set_terrain(tid, tset)
    m3.base_terrain = 1
    assert not (generate_perlin_map(m3, seed="别的种子", water_id=2, plain_id=1, mountain_id=2) == kinds).all()


def test_perlin_big_map_renders_with_tile_cache():
    """大地图也能秒级出图（合成结果按掩码缓存）。"""
    arts = _eco_art()
    model = TileMapModel(96, 64, tile_size=S)
    for tid, tset in arts.items():
        model.set_terrain(tid, tset)
    model.base_terrain = 1
    generate_perlin_map(model, seed="cache", water_id=2, plain_id=1, mountain_id=1)
    img = model.render()
    assert img.size == (96 * S, 64 * S)
    assert (np.asarray(img)[..., 3] == 255).all()


def test_scatter_walls_places_piece_family():
    art = wall_art_from_sheet(
        type("S", (), {})() if False else __import__("core.tilemap.tiles", fromlist=["BuildingSheet"]).BuildingSheet(
            wall=BaseTileSet(size=S, center=_cell((150, 140, 128), S), edges={}, corners={}),
            top=BaseTileSet(size=S, center=_cell((186, 176, 160), S), edges={}, corners={}),
            opening=BaseTileSet(size=S, center=Image.new("RGBA", (S, S), (255, 255, 255, 255)), edges={}, corners={}),
            pillar=BaseTileSet(size=S, center=Image.new("RGBA", (S, S), (255, 255, 255, 255)), edges={}, corners={}),
        ),
        tile_size=S,
    )
    pieces = build_piece_set(art)
    model = TileMapModel(48, 32, tile_size=S)
    model.set_terrain(1, _eco_art()[1])
    model.base_terrain = 1
    kinds = np.ones((32, 48), np.int32)
    placed = scatter_walls(model, pieces, kinds=kinds, every=11)
    assert placed > 0
    assert model.overlay, "应写入 overlay 层"
    names = {item[2] for item in model.overlay.values()}
    assert names, names
    # 用到的件必须来自 16 族（含门洞造成的端头/转角）
    from core.tilemap.walls import W16_SLOTS

    assert names <= set(W16_SLOTS)
