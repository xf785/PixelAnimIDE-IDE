"""无缝化算法（确定性、无 AI 依赖）。

1. 全向纹理无缝：镜像相位平均（水平镜像均值 + 垂直镜像均值再平均）——
   数学上保证 left==right、top==bottom 逐像素相等（周期平铺零接缝），
   两条镜像中线的残影互相稀释；随后重量化回原调色板恢复像素硬边与颜色数。
2. 墙面/边界瓦片轴向无缝：把边界朝向旋转到顶部，仅沿边界方向做镜像平均
   （边界线带不动），再把边界线带统一为同一颜色（四张边瓦片共享线色），
   保证长墙无限重复、四个朝向风格一致。
3. 转角瓦片：不直接修 AI 的角瓦片，而是由「无缝中心纹理 + 统一线色」按
   位掩码构图推导（见 autotile.derive_corners），构造上保证与墙面/中心
   零接缝；AI 角瓦片仅可选地以低权重混合内部细节。
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from core.processing.pixelizer import extract_dominant_palette, map_to_palette
from .tiles import EDGE_NAMES, BaseTileSet

logger = logging.getLogger("PixelAnimIDE.tilemap.seamless")

RGB = Tuple[int, int, int]


def _requantize_rgba(img: Image.Image, source: Image.Image, extra: Optional[RGB] = None, max_colors: int = 31) -> Image.Image:
    """重量化到 source 主色（+ 可选附加色），恢复像素画硬边与帧间一致性。"""
    palette = list(extract_dominant_palette(source.convert("RGB"), max_colors))
    if extra is not None:
        lc = (int(extra[0]), int(extra[1]), int(extra[2]))
        if lc not in palette:
            palette.append(lc)
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    q = map_to_palette(rgba.convert("RGB"), palette).convert("RGBA")
    q.putalpha(alpha)
    return q


def make_texture_seamless(img: Image.Image, band: Optional[int] = None, max_colors: int = 32) -> Image.Image:
    """把纹理修正为全向无缝瓦片（确定性；输入输出同尺寸 RGBA）。

    偏移错位缝合（Offset Quilt）：把纹理平移半格，仅对中央十字接缝的
    窄带（默认每侧 tile//8，≥2px）做交叉淡化，再平移回来——接缝变成
    新的左右/上下边缘，两侧像素完全相等（周期平铺零接缝）；纹理主体
    细节原样保留，不再出现全图镜像平均导致的「渐变+色块」退化。
    """
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba).astype(np.float32)
    s = arr.shape[0]
    b = max(2, int(band or max(2, s // 8)))
    b = min(b, s // 2 - 1)

    shifted = np.roll(np.roll(arr, s // 2, axis=0), s // 2, axis=1)
    out = shifted.copy()
    c = s // 2

    def _blend(axis: int) -> None:
        """沿 axis=1（垂直缝）或 axis=0（水平缝）交叉淡化中央带。

        接缝两侧第 i 列（0 起）：i=0 严格 50/50（保证新边缘逐像素相等），
        i≥1 时以 t=i/(2b) 线性回到原纹理（带宽外的像素完全不动）。
        垂直缝先处理（源=shifted），水平缝随后处理（源=已融合的 out），
        这样第二次融合不会破坏第一次的列相等性。
        """
        src = shifted if axis == 1 else out
        for i in range(b):
            t = 0.0 if i == 0 else i / (2.0 * b)  # i=0 为接缝对，严格 50/50
            if axis == 1:
                a_idx = (c - 1 - i) % s
                b_idx = (c + i) % s
                a = src[:, a_idx].copy()
                bb = src[:, b_idx].copy()
                out[:, a_idx] = (0.5 + t) * a + (0.5 - t) * bb
                out[:, b_idx] = (0.5 - t) * a + (0.5 + t) * bb
            else:
                a_idx = (c - 1 - i) % s
                b_idx = (c + i) % s
                a = src[a_idx].copy()
                bb = src[b_idx].copy()
                out[a_idx] = (0.5 + t) * a + (0.5 - t) * bb
                out[b_idx] = (0.5 - t) * a + (0.5 + t) * bb

    _blend(1)
    _blend(0)

    # 平移回来：接缝带成为新边缘（左右/上下逐像素相等）
    out = np.roll(np.roll(out, -s // 2, axis=0), -s // 2, axis=1)
    rgb = np.clip(out[..., :3], 0, 255).astype(np.uint8)
    alpha = np.where(out[..., 3] >= 128, 255, 0).astype(np.uint8)
    merged = Image.fromarray(rgb, "RGB")
    q = _requantize_rgba(merged, rgba, max_colors=max_colors)
    q.putalpha(Image.fromarray(alpha, "L"))
    return q


def make_edge_seamless(img_top: Image.Image, line_width: int = 2) -> Tuple[Image.Image, RGB]:
    """把「边界朝上」的墙面瓦片修正为沿边界方向无缝。

    边界线带（顶部 line_width 行）保持原样并统一为带内中位数颜色；
    其余部分沿水平轴镜像平均（左右边缘相等）。返回 (瓦片, 线带颜色)。
    """
    s = img_top.size[0]
    lw = max(1, min(int(line_width), max(1, s // 4)))
    rgba = img_top.convert("RGBA")
    arr = np.asarray(rgba).astype(np.float32)
    band = arr[:lw, :, :3].reshape(-1, 3)
    band_color = tuple(int(c) for c in np.median(band, axis=0))
    out = 0.5 * (arr + arr[:, ::-1, :])  # 左右无缝
    out[:lw, :, :3] = np.array(band_color, dtype=np.float32)
    out[:lw, :, 3] = 255.0
    rgb = np.clip(out[..., :3], 0, 255).astype(np.uint8)
    alpha = np.where(out[..., 3] >= 128, 255, 0).astype(np.uint8)
    merged = Image.fromarray(rgb, "RGB")
    q = _requantize_rgba(merged, rgba, extra=band_color, max_colors=31)
    q.putalpha(Image.fromarray(alpha, "L"))
    return q, band_color


# 旋转：把四张边瓦片统一朝向（边界朝上）处理，再转回
_EDGE_ROT = {
    "top": (None, None),
    "right": (Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_270),
    "bottom": (Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_180),
    "left": (Image.Transpose.ROTATE_270, Image.Transpose.ROTATE_90),
}


def make_axis_seamless(img: Image.Image, axis: str = "h", band: Optional[int] = None) -> Image.Image:
    """仅沿单轴做偏移错位缝合（保持另一轴内容完全不变）。

    用于地块/建筑边瓦片：top/bottom 沿水平重复、left/right 沿垂直重复，
    同时保留 AI 绘制的过渡艺术（不做线带统一化，避免破坏纹理）。
    """
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba).astype(np.float32)
    s = arr.shape[0]
    b = max(2, int(band or max(2, s // 8)))
    b = min(b, s // 2 - 1)
    c = s // 2
    if axis == "h":
        shifted = np.roll(arr, c, axis=1)
        out = shifted.copy()
        for i in range(b):
            t = 0.0 if i == 0 else i / (2.0 * b)
            a = shifted[:, (c - 1 - i) % s].copy()
            bb = shifted[:, (c + i) % s].copy()
            out[:, (c - 1 - i) % s] = (0.5 + t) * a + (0.5 - t) * bb
            out[:, (c + i) % s] = (0.5 - t) * a + (0.5 + t) * bb
        out = np.roll(out, -c, axis=1)
    else:
        shifted = np.roll(arr, c, axis=0)
        out = shifted.copy()
        for i in range(b):
            t = 0.0 if i == 0 else i / (2.0 * b)
            a = shifted[(c - 1 - i) % s].copy()
            bb = shifted[(c + i) % s].copy()
            out[(c - 1 - i) % s] = (0.5 + t) * a + (0.5 - t) * bb
            out[(c + i) % s] = (0.5 - t) * a + (0.5 + t) * bb
        out = np.roll(out, -c, axis=0)
    rgb = np.clip(out[..., :3], 0, 255).astype(np.uint8)
    alpha = np.where(out[..., 3] >= 128, 255, 0).astype(np.uint8)
    q = _requantize_rgba(Image.fromarray(rgb, "RGB"), rgba, max_colors=48)
    q.putalpha(Image.fromarray(alpha, "L"))
    return q


def _union_palette(tiles, cap: int = 64):
    """多张瓦片主色的并集（保持帧间/瓦片间颜色一致，按频率截断）。"""
    out = []
    for t in tiles:
        for c in extract_dominant_palette(t.convert("RGB"), 16):
            if c not in out:
                out.append(c)
        if len(out) >= cap:
            break
    return out[:cap]


def prepare_terrain_set(base: BaseTileSet, max_colors: int = 64, band: Optional[int] = None) -> BaseTileSet:
    """地块/特征瓦片组预处理（保留 AI 艺术，仅做必要无缝化）。

    - 中心：全向偏移错位缝合（纹理细节保留）；
    - 四边：沿边界方向单轴缝合（过渡艺术不动、不做线带统一）；
    - 四角：原样保留（构图时直接取四分之一块）；
    - 全部重量化到九宫格并集调色板，保证颜色家族一致。
    """
    center = make_texture_seamless(base.center, band=band, max_colors=max_colors)
    edges: Dict[str, Image.Image] = {}
    for name in EDGE_NAMES:
        rot_in, rot_out = _EDGE_ROT[name]
        oriented = base.edges[name].transpose(rot_in) if rot_in else base.edges[name]
        processed = make_axis_seamless(oriented, axis="h", band=band)
        edges[name] = processed.transpose(rot_out) if rot_out else processed
    palette = _union_palette([center] + list(edges.values()) + list(base.corners.values()), cap=max_colors)
    quant = lambda im: _requantize_rgba_palette(im, palette)
    return BaseTileSet(
        size=base.size,
        center=quant(center),
        edges={n: quant(t) for n, t in edges.items()},
        corners={n: quant(t) for n, t in base.corners.items()},
        line_color=base.line_color,
        line_width=base.line_width,
    )


def _requantize_rgba_palette(img: Image.Image, palette) -> Image.Image:
    """按给定调色板量化（保留 alpha）。"""
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    q = map_to_palette(rgba.convert("RGB"), palette).convert("RGBA")
    q.putalpha(alpha)
    return q


# --------------------------------------------------------------------------- #
# 纹理提取与「对齐式」地形艺术（第 8 轮：构造性无缝 47 拼接）
# --------------------------------------------------------------------------- #
def _inset_crop(img: Image.Image, frac: float) -> Image.Image:
    """去掉四周 frac 比例的外框（AI 的格线/描边/边缘脏像素都在外侧）。"""
    w, h = img.size
    dx = max(1, int(round(w * float(frac))))
    dy = max(1, int(round(h * float(frac))))
    if w - 2 * dx < 8 or h - 2 * dy < 8:
        return img
    return img.crop((dx, dy, w - dx, h - dy))


def make_tile_texture(
    tile: Image.Image,
    target: int,
    inset_frac: float = 0.10,
    max_colors: int = 64,
) -> Image.Image:
    """把一格 AI 像素画变成「可网格对齐平铺」的无缝纹理（target×target）。

    1. 去掉外侧 inset_frac（甩掉格线框与边缘噪声）；
    2. 偏移错位缝合 → 左右/上下边缘逐像素相等（周期平铺零接缝）；
    3. 块众数/最近邻缩放到瓦片尺寸（保持像素硬边）；
    4. **缩放后再缝合一次**：块众数降采样按块取值，会破坏步骤 2 的边缘相等性，
       而「相邻瓦片共享边逐像素相等」这条不变量正建立在最终尺寸的边缘相等上。
    """
    from .tiles import resize_tile

    core = _inset_crop(tile.convert("RGBA"), inset_frac)
    seamless = make_texture_seamless(core, max_colors=max_colors)
    return make_texture_seamless(resize_tile(seamless, int(target)), max_colors=max_colors)


def median_tile_texture(
    tiles,
    target: int,
    inset_frac: float = 0.10,
    max_colors: int = 64,
) -> Image.Image:
    """九格逐像素中位数 → 纹理（抹掉只出现在个别格里的文字/贴花/水印）。

    基础地形块的 9 格理应是同一种纹理；任何只画在某一格里的东西（文字、签名、
    装饰）在 9 格中位数里都会被「投票」掉，这是对「底图上写了字」最稳的兜底。
    """
    from .tiles import resize_tile

    crops = [np.asarray(_inset_crop(t.convert("RGBA"), inset_frac), dtype=np.float32) for t in tiles]
    h = min(c.shape[0] for c in crops)
    w = min(c.shape[1] for c in crops)
    stack = np.stack([c[:h, :w] for c in crops], axis=0)
    med = np.median(stack, axis=0).astype(np.uint8)
    tex = make_texture_seamless(Image.fromarray(med, "RGBA"), max_colors=max_colors)
    return make_texture_seamless(resize_tile(tex, int(target)), max_colors=max_colors)


def _first_feature_depth(profile: np.ndarray, target: np.ndarray, tol: float) -> Optional[int]:
    """从外侧向内扫描，返回第一个「接近特征色」的像素下标（无则 None）。"""
    d = np.linalg.norm(profile - target[None, :], axis=1)
    idx = np.nonzero(d <= tol)[0]
    return int(idx[0]) if len(idx) else None


def measure_terrain_art(
    base: BaseTileSet,
    ground_rgb: Optional[np.ndarray] = None,
    default_band_frac: float = 0.25,
    default_rim_frac: float = 0.03,
) -> Dict:
    """从 AI 九宫格实测「基础地形条带深度」「描边宽度/颜色」（比例，与尺寸无关）。

    特征块的结构是「中心格=纯特征、边格外侧条带=基础地形」，因此：
    - 条带深度 = 边格上从外向内第一个「像特征」的像素位置（四侧三格取中位数）；
    - 描边 = 条带内明显暗于两端的像素（宽度取中位数、颜色取中位数）。
    全部用比例表达，因此 AI 画在多少像素上都成立；测量失败时回退默认值。
    """
    tiles = base.all()
    order = ["tl", "top", "tr", "left", "center", "right", "bl", "bottom", "br"]
    by_name = dict(zip(order, tiles))
    centre = np.asarray(by_name["center"].convert("RGB"), dtype=np.float32)
    s = centre.shape[0]
    feat_rgb = np.median(centre.reshape(-1, 3), axis=0)
    if ground_rgb is None:
        ground_rgb = np.median(
            np.concatenate([
                np.asarray(by_name[n].convert("RGB"), dtype=np.float32).reshape(-1, 3)
                for n in ("top", "bottom", "left", "right")
            ], axis=0),
            axis=0,
        )
    tol = max(10.0, float(np.linalg.norm(feat_rgb - ground_rgb)) * 0.35)

    def depths_for(names, side: str) -> list:
        """按「从外侧向内」的方向扫描三张边格，收集条带深度（像素）。"""
        out = []
        for n in names:
            arr = np.asarray(by_name[n].convert("RGB"), dtype=np.float32)
            h, w = arr.shape[:2]
            if side == "top":
                profiles = [arr[:, x] for x in range(w)]
            elif side == "bottom":
                profiles = [arr[::-1, x] for x in range(w)]
            elif side == "left":
                profiles = [arr[y, :] for y in range(h)]
            else:
                profiles = [arr[y, ::-1] for y in range(h)]
            for prof in profiles:
                depth = _first_feature_depth(prof, feat_rgb, tol)
                if depth is not None and 0 < depth < s:
                    out.append(depth)
        return out

    depths = (
        depths_for(("top", "tl", "tr"), "top")
        + depths_for(("bottom", "bl", "br"), "bottom")
        + depths_for(("left", "tl", "bl"), "left")
        + depths_for(("right", "tr", "br"), "right")
    )
    band_frac = float(np.median(depths)) / s if depths else float(default_band_frac)
    band_frac = min(0.5, max(0.08, band_frac))

    # 描边：边格外侧条带内「明显暗于基础地形」的像素
    band_px = max(1, int(round(band_frac * s)))
    rim_px: list = []
    rim_cols: list = []
    for n in ("top", "bottom", "left", "right"):
        arr = np.asarray(by_name[n].convert("RGB"), dtype=np.float32)
        h, w = arr.shape[:2]
        if n == "top":
            strip = arr[:band_px, :, :]
        elif n == "bottom":
            strip = arr[h - band_px:, :, :]
        elif n == "left":
            strip = arr[:, :band_px, :]
        else:
            strip = arr[:, w - band_px:, :]
        lum = 0.299 * strip[..., 0] + 0.587 * strip[..., 1] + 0.114 * strip[..., 2]
        ground_lum = float(0.299 * ground_rgb[0] + 0.587 * ground_rgb[1] + 0.114 * ground_rgb[2])
        dark = lum < ground_lum * 0.72
        if not dark.any():
            continue
        rim_cols.append(np.median(strip[dark], axis=0))
        # 宽度：每列（行）里暗像素的数量取中位数
        counts = dark.sum(axis=0) if n in ("top", "bottom") else dark.sum(axis=1)
        counts = counts[counts > 0]
        if counts.size:
            rim_px.append(float(np.median(counts)))
    rim_frac = (float(np.median(rim_px)) / s) if rim_px else float(default_rim_frac)
    rim_frac = min(0.25, max(0.015, rim_frac))
    rim_rgb = (
        tuple(int(c) for c in np.median(np.stack(rim_cols, axis=0), axis=0))
        if rim_cols else (24, 22, 26)
    )
    return {
        "band_frac": band_frac,
        "rim_frac": rim_frac,
        "rim_rgb": rim_rgb,
        "feature_rgb": tuple(int(c) for c in feat_rgb),
        "ground_rgb": tuple(int(c) for c in ground_rgb),
        "measured": bool(depths) and bool(rim_cols),
    }


def measure_edge_profile(
    edges: Dict[str, Image.Image],
    band_px: int,
    ground_rgb: np.ndarray,
    feat_rgb: np.ndarray,
    max_outline: int = 5,
    max_bevel: int = 3,
) -> Dict:
    """实测「边界剖面」：地面侧的**分层描边色调** + 特征侧的**倒角色调**。

    手绘 47 图块之所以耐看，关键在边界不是一条平色线，而是由外向内的一组层次：
    地面 → 描边暗色（可能 2~3 阶）→ 特征边的高光/暗部 → 特征纹理。这里把这组层次
    从 AI 的边格上量出来（每一「离边界距离」取四侧三格的中位色），渲染时按
    **到边界的像素距离**上色 —— 因此圆弧转角、直边、内凹角都自动得到等宽的层次，
    相邻瓦片的共享边依旧逐像素一致（层次只依赖距离）。

    返回 {"outline": [(rgb), ...]（下标 0 = 紧贴特征的一圈）,
          "bevel": [(rgb), ...]（下标 0 = 紧贴边界的一圈特征侧）,
          "outline_px": int, "bevel_px": int}
    """
    ground_rgb = np.asarray(ground_rgb, dtype=np.float32)
    feat_rgb = np.asarray(feat_rgb, dtype=np.float32)
    ground_lum = float(0.299 * ground_rgb[0] + 0.587 * ground_rgb[1] + 0.114 * ground_rgb[2])
    feat_lum = float(0.299 * feat_rgb[0] + 0.587 * feat_rgb[1] + 0.114 * feat_rgb[2])
    band = max(1, int(band_px))
    outline_samples: List[List[np.ndarray]] = [[] for _ in range(max_outline)]
    bevel_samples: List[List[np.ndarray]] = [[] for _ in range(max_bevel)]

    def profiles(arr: np.ndarray, side: str):
        h, w = arr.shape[:2]
        if side == "top":
            return [arr[:, x] for x in range(w)]
        if side == "bottom":
            return [arr[::-1, x] for x in range(w)]
        if side == "left":
            return [arr[y, :] for y in range(h)]
        return [arr[y, ::-1] for y in range(h)]

    for name in ("top", "bottom", "left", "right"):
        tile = edges.get(name)
        if tile is None:
            continue
        arr = np.asarray(tile.convert("RGB"), dtype=np.float32)
        for prof in profiles(arr, name):
            for k in range(1, max_outline + 1):
                i = band - k                      # 地面侧：距边界 k 像素
                if 0 <= i < prof.shape[0]:
                    outline_samples[k - 1].append(prof[i])
            for j in range(1, max_bevel + 1):
                i = band - 1 + j                  # 特征侧：距边界 j 像素
                if 0 <= i < prof.shape[0]:
                    bevel_samples[j - 1].append(prof[i])

    outline: List[Tuple[int, int, int]] = []
    for samples in outline_samples:
        if not samples:
            break
        tone = np.median(np.stack(samples, axis=0), axis=0)
        lum = float(0.299 * tone[0] + 0.587 * tone[1] + 0.114 * tone[2])
        if outline and lum > ground_lum * 0.94:
            break                             # 已经回到地面亮度 -> 描边到此为止
        if not outline and lum > ground_lum * 0.88:
            break                             # 第一圈就不暗 -> 该底图没有描边
        outline.append(tuple(int(c) for c in tone))

    bevel: List[Tuple[int, int, int]] = []
    for samples in bevel_samples:
        if not samples:
            break
        tone = np.median(np.stack(samples, axis=0), axis=0)
        lum = float(0.299 * tone[0] + 0.587 * tone[1] + 0.114 * tone[2])
        if abs(lum - feat_lum) < 4.0:
            break                             # 与特征纹理一致 -> 没有倒角
        bevel.append(tuple(int(c) for c in tone))
    return {"outline": outline, "bevel": bevel, "outline_px": len(outline), "bevel_px": len(bevel)}


def _resized_edges(base: BaseTileSet, tile_size: int) -> Dict[str, Image.Image]:
    from .tiles import resize_tile

    return {n: resize_tile(base.edges[n], int(tile_size)) for n in EDGE_NAMES if n in base.edges}


def align_terrain_set(
    base: BaseTileSet,
    base_texture: Optional[Image.Image] = None,
    tile_size: int = 32,
    ground_rgb: Optional[np.ndarray] = None,
    max_colors: int = 64,
    plain: bool = False,
) -> BaseTileSet:
    """把 AI 九宫格艺术「对齐化」成一套可构造性无缝拼接的地形艺术。

    - `center` = 该地形的无缝特征纹理（网格对齐平铺，边缘逐像素可接）；
    - `base_texture` + `band`/`radius` + `line_color`/`line_width` = 另一方地形
      的纹理与实测条带/描边参数，供 `autotile.compose_art_tile` 程序化构图。

    这样 47 张瓦片共享同一份纹理与同一条几何规则，**相邻瓦片的共享边逐像素
    相等**（无缝由构造保证），不再依赖 AI 把边界画在哪个深度、是否连续。

    plain=True（基础地形块）时不实测条带/描边（它本来就没有边界），用默认几何，
    这样地图外缘的边框线也不会被底图上的脏像素污染。
    """
    texture = make_tile_texture(base.center, tile_size, max_colors=max_colors)
    if plain:
        darkest = _darkest_color(texture)
        band_px = max(2, tile_size // 4)
        soft = tuple(
            int(c) for c in (np.asarray(darkest, np.float32) * 0.35 + np.asarray(texture.convert("RGB").getpixel((0, 0)), np.float32) * 0.65)
        )
        return BaseTileSet(
            size=texture.size[0],
            center=texture,
            edges={n: texture for n in EDGE_NAMES},
            corners={n: texture for n in ("tl", "tr", "bl", "br")},
            line_color=darkest,
            line_width=2,
            band=band_px,
            radius=band_px,
            base_texture=(base_texture or texture),
            art_meta={"band_px": band_px, "rim_px": 2, "plain": True, "tile_size": tile_size,
                      "outline": [list(darkest), list(soft)], "bevel": [],
                      "outline_px": 2, "bevel_px": 0},
        )
    measured = measure_terrain_art(base, ground_rgb=ground_rgb)
    band = max(1, min(tile_size // 2, int(round(measured["band_frac"] * tile_size))))
    # 目标尺度上实测边界剖面（描边分层色调 + 特征侧倒角），渲染时按「到边界的距离」上色
    profile = measure_edge_profile(
        _resized_edges(base, tile_size),
        band,
        ground_rgb=np.asarray(
            ground_rgb if ground_rgb is not None else measured["ground_rgb"], dtype=np.float32
        ),
        feat_rgb=np.asarray(measured["feature_rgb"], dtype=np.float32),
    )
    outline = list(profile["outline"])
    if not outline:                     # 底图没有明显描边 -> 用「实测描边色 + 过渡色」两层兜底
        lc = np.asarray(measured["rim_rgb"], dtype=np.float32)
        gc = np.asarray(measured["ground_rgb"], dtype=np.float32)
        outline = [
            tuple(int(c) for c in lc),
            tuple(int(c) for c in (lc * 0.35 + gc * 0.65)),
        ]
    outline = outline[: max(1, band - 1)]
    bevel = list(profile["bevel"])
    return BaseTileSet(
        size=texture.size[0],
        center=texture,
        edges={n: texture for n in EDGE_NAMES},
        corners={n: texture for n in ("tl", "tr", "bl", "br")},
        line_color=outline[0],
        line_width=len(outline),
        band=band,
        radius=band,
        base_texture=(base_texture or texture),
        art_meta=dict(
            measured,
            band_px=band,
            rim_px=len(outline),
            tile_size=tile_size,
            outline=[list(c) for c in outline],
            bevel=[list(c) for c in bevel],
            outline_px=len(outline),
            bevel_px=len(bevel),
        ),
    )


def _darkest_color(img: Image.Image) -> Tuple[int, int, int]:
    """纹理中最暗 15% 像素的中位色（当描边色用，和纹理同色系）。"""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32).reshape(-1, 3)
    if not len(arr):
        return (0, 0, 0)
    lum = 0.299 * arr[:, 0] + 0.587 * arr[:, 1] + 0.114 * arr[:, 2]
    k = max(1, int(len(lum) * 0.15))
    dark = arr[np.argsort(lum)[:k]]
    return tuple(int(c) for c in np.median(dark, axis=0))


def process_base_set(
    base: BaseTileSet,
    max_colors: int = 32,
    detail_keep: float = 0.3,
    line_width: Optional[int] = None,
) -> BaseTileSet:
    """把 AI 裁出的 9 张基础瓦片处理成自洽无缝瓦片组。

    - 中心：全向无缝；
    - 四边：沿边界方向无缝 + 统一边界线色；
    - 四角：由中心纹理 + 统一线色构图推导（与墙面/中心零接缝），
      可选混合 AI 角瓦片内部细节；
    - 全部重量化到「中心主色 + 线色」调色板，保证整组颜色一致。
    """
    from .autotile import derive_corners

    s = base.size
    lw = max(1, int(line_width or base.line_width or max(1, s // 16)))
    center = make_texture_seamless(base.center, max_colors=max_colors)

    edge_tops: Dict[str, Image.Image] = {}
    band_colors: Dict[str, RGB] = {}
    for name in EDGE_NAMES:
        rot_in, rot_out = _EDGE_ROT[name]
        oriented = base.edges[name].transpose(rot_in) if rot_in else base.edges[name]
        processed, band_color = make_edge_seamless(oriented, line_width=lw)
        edge_tops[name] = processed.transpose(rot_out) if rot_out else processed
        band_colors[name] = band_color

    # 统一线色：四边线带颜色取中位数，再回填到全部边瓦片线带
    colors = np.array([band_colors[n] for n in EDGE_NAMES], dtype=np.float32)
    line_color: RGB = tuple(int(c) for c in np.median(colors, axis=0))
    edges: Dict[str, Image.Image] = {}
    for name in EDGE_NAMES:
        tile = edge_tops[name]
        arr = np.asarray(tile.convert("RGBA")).copy()
        band = {"top": (slice(0, lw), slice(None)),
                "bottom": (slice(s - lw, s), slice(None)),
                "left": (slice(None), slice(0, lw)),
                "right": (slice(None), slice(s - lw, s))}[name]
        arr[band][..., :3] = np.array(line_color, dtype=np.uint8)
        arr[band][..., 3] = 255
        edges[name] = _requantize_rgba(
            Image.fromarray(arr, "RGBA"), center, extra=line_color, max_colors=max_colors
        )

    corners = derive_corners(
        center, line_color, line_width=lw,
        ai_corners=base.corners, detail_keep=detail_keep,
    )
    return BaseTileSet(
        size=s,
        center=center,
        edges=edges,
        corners=corners,
        line_color=line_color,
        line_width=lw,
    )
