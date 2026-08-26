"""像素画布数据模型：铅笔 / 橡皮 / 取色 / 填充 + 撤销重做（纯数据层，不依赖 Qt）。

UI 层（PixelEditorWidget）负责渲染与鼠标事件换算，本模块只处理像素数据，
便于单元测试与复用。
"""
from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

RGBA = Tuple[int, int, int, int]
TRANSPARENT: RGBA = (0, 0, 0, 0)


def _as_rgba(color) -> RGBA:
    """把 (r,g,b) / (r,g,b,a) / [r,g,b] 等统一为 RGBA 四元组。"""
    if color is None:
        return TRANSPARENT
    c = tuple(int(v) for v in color)
    if len(c) == 3:
        return c[0], c[1], c[2], 255
    if len(c) == 4:
        return c[0], c[1], c[2], c[3]
    raise ValueError(f"非法颜色: {color!r}")


class PixelCanvas:
    """包装一张 RGBA 帧，提供像素级编辑与撤销/重做。"""

    def __init__(self, image: Optional[Image.Image] = None, max_history: int = 50):
        if image is None:
            image = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
        self._img: Image.Image = image.convert("RGBA")
        self.max_history = int(max_history)
        self._undo_stack: List[np.ndarray] = []
        self._redo_stack: List[np.ndarray] = []
        self._palette: Optional[List[RGBA]] = None  # 锁定调色板（None = 不锁定）

    # ------------------------------------------------------------------ #
    @property
    def image(self) -> Image.Image:
        return self._img

    @property
    def width(self) -> int:
        return self._img.width

    @property
    def height(self) -> int:
        return self._img.height

    @property
    def size(self) -> Tuple[int, int]:
        return self._img.size

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    # ------------------------------------------------------------------ #
    def get_pixel(self, x: int, y: int) -> RGBA:
        """取像素颜色；越界返回全透明。"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return tuple(self._img.getpixel((x, y)))
        return TRANSPARENT

    # ------------------------------------------------------------------ #
    # 调色板锁定
    # ------------------------------------------------------------------ #
    @property
    def palette(self) -> Optional[List[RGBA]]:
        return list(self._palette) if self._palette else None

    def set_palette(self, colors) -> None:
        """锁定调色板：此后绘制/填充的颜色会吸附到最近锁定色。"""
        self._palette = [_as_rgba(c) for c in colors] if colors else None

    def clear_palette(self) -> None:
        self._palette = None

    def snap_color(self, color) -> RGBA:
        """把颜色吸附到锁定调色板的最近色（未锁定时原样返回）。"""
        return self._snap(color)

    def _snap(self, color) -> RGBA:
        c = _as_rgba(color)
        if not self._palette:
            return c
        return min(self._palette, key=lambda p: sum((p[i] - c[i]) ** 2 for i in range(4)))

    def _snapshot(self) -> None:
        """把当前状态压入撤销栈（画之前调用），并清空重做栈。"""
        self._undo_stack.append(np.asarray(self._img).copy())
        if len(self._undo_stack) > self.max_history:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _commit(self, arr: np.ndarray) -> None:
        self._img = Image.fromarray(arr, "RGBA")

    # ------------------------------------------------------------------ #
    def set_pixel(self, x: int, y: int, color, size: int = 1) -> None:
        """单点/方形笔刷绘制（铅笔）。越界或颜色未变化时不做任何事。"""
        c = self._snap(color)
        if size <= 1:
            if not (0 <= x < self.width and 0 <= y < self.height):
                return
            if self._img.getpixel((x, y)) == c:
                return
            self._snapshot()
            self._img.putpixel((x, y), c)
            return
        self._brush_rect(x, y, size, c)

    def _brush_rect(self, cx: int, cy: int, size: int, c: RGBA) -> None:
        """以 (cx,cy) 为中心盖 size×size 方形笔刷（裁剪到画布）。"""
        r = size // 2
        x0 = max(0, cx - r)
        x1 = min(self.width - 1, cx + (size - 1 - r))
        y0 = max(0, cy - r)
        y1 = min(self.height - 1, cy + (size - 1 - r))
        if x1 < x0 or y1 < y0:
            return
        arr = np.asarray(self._img).copy()
        sub = arr[y0 : y1 + 1, x0 : x1 + 1]
        target = np.array(c, dtype=np.uint8)
        if not (sub != target).any():
            return
        self._snapshot()
        sub[:] = target
        self._commit(arr)

    def draw_line(self, p0: Tuple[int, int], p1: Tuple[int, int], color, size: int = 1) -> None:
        """Bresenham 连线（铅笔快速拖动避免断点）；size>1 时为方形笔刷盖章。"""
        c = self._snap(color)
        pts = list(self._line_points(p0, p1))
        if size <= 1:
            changed = any(
                0 <= x < self.width and 0 <= y < self.height and self._img.getpixel((x, y)) != c
                for x, y in pts
            )
            if not changed:
                return
            self._snapshot()
            for x, y in pts:
                if 0 <= x < self.width and 0 <= y < self.height:
                    self._img.putpixel((x, y), c)
            return
        # 方形笔刷：收集整条线的盖章格
        r = size // 2
        cells = set()
        for x, y in pts:
            for yy in range(max(0, y - r), min(self.height, y + size - r)):
                for xx in range(max(0, x - r), min(self.width, x + size - r)):
                    cells.add((xx, yy))
        if not cells:
            return
        arr = np.asarray(self._img).copy()
        if not any(tuple(arr[yy, xx]) != c for xx, yy in cells):
            return
        self._snapshot()
        for xx, yy in cells:
            arr[yy, xx] = np.array(c, dtype=np.uint8)
        self._commit(arr)

    @staticmethod
    def _line_points(p0, p1):
        """Bresenham 直线点序列。"""
        x0, y0 = int(p0[0]), int(p0[1])
        x1, y1 = int(p1[0]), int(p1[1])
        pts = []
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            pts.append((x0, y0))
            if (x0, y0) == (x1, y1):
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return pts

    def flood_fill(self, x: int, y: int, color) -> None:
        """四连通泛洪填充：把与 (x,y) 同色的连通区域替换为目标色。"""
        c = self._snap(color)
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        target = self._img.getpixel((x, y))
        if target == c:
            return
        self._snapshot()
        arr = np.asarray(self._img).copy()
        h, w = arr.shape[:2]
        visited = np.zeros((h, w), dtype=bool)
        q: deque = deque([(y, x)])
        visited[y, x] = True
        t = np.array(target)
        while q:
            cy, cx = q.popleft()
            arr[cy, cx] = c
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    if np.array_equal(arr[ny, nx], t):
                        visited[ny, nx] = True
                        q.append((ny, nx))
        self._commit(arr)

    # ------------------------------------------------------------------ #
    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self._redo_stack.append(np.asarray(self._img).copy())
        self._img = Image.fromarray(self._undo_stack.pop(), "RGBA")
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(np.asarray(self._img).copy())
        self._img = Image.fromarray(self._redo_stack.pop(), "RGBA")
        return True

    def replace_image(self, image: Image.Image) -> None:
        """整体替换画布内容（切换帧时用），清空历史。"""
        self._img = image.convert("RGBA")
        self._undo_stack.clear()
        self._redo_stack.clear()

    def replace_color(self, old, new) -> int:
        """把画布中所有等于 old 的像素替换为 new（Krita 式全局换色）。

        返回替换的像素数；无变化时返回 0 且不产生撤销记录。
        """
        return self.replace_colors({_as_rgba(old): _as_rgba(new)})

    def replace_colors(self, mapping) -> int:
        """批量换色（色族整体替换用）：mapping = {旧色: 新色}。

        一次性快照、一次提交，返回替换的总像素数；无变化返回 0 且不产生撤销记录。
        """
        mapping = {_as_rgba(k): _as_rgba(v) for k, v in mapping.items()}
        arr = np.asarray(self._img).copy()  # PIL 视图只读，须复制后写入
        n = 0
        for old_c, new_c in mapping.items():
            if old_c == new_c:
                continue
            mask = (arr == np.array(old_c, dtype=np.uint8)).all(axis=-1)
            if mask.any():
                arr[mask] = np.array(new_c, dtype=np.uint8)
                n += int(mask.sum())
        if n == 0:
            return 0
        self._snapshot()
        self._commit(arr)
        return n

    def paste_image(self, img: Image.Image, x: int, y: int) -> int:
        """把 RGBA 图以 alpha 合成到画布 (x,y)（图层合并用）。

        一次性快照一次提交；越界自动裁剪；返回发生变化的像素数，无变化返回 0。
        """
        img = img.convert("RGBA")
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(self.width, int(x) + img.width)
        y1 = min(self.height, int(y) + img.height)
        if x1 <= x0 or y1 <= y0:
            return 0
        src = np.asarray(img.crop((x0 - x, y0 - y, x1 - x, y1 - y))).astype(np.float64)
        arr = np.asarray(self._img).copy()
        sub = arr[y0:y1, x0:x1].astype(np.float64)
        a = src[..., 3:4] / 255.0
        da = sub[..., 3:4] / 255.0
        out_a = a + da * (1.0 - a)
        rgb = (src[..., :3] * a + sub[..., :3] * da * (1.0 - a)) / np.maximum(out_a, 1e-9)
        new_sub = np.concatenate([rgb, out_a * 255.0], axis=-1).astype(np.uint8)
        changed = (new_sub != arr[y0:y1, x0:x1]).any(axis=-1)
        n = int(changed.sum())
        if n == 0:
            return 0
        self._snapshot()
        arr[y0:y1, x0:x1] = new_sub
        self._commit(arr)
        return n

    def fill_rect(self, x0, y0, x1, y1, color) -> int:
        """矩形区域填充（框选填充）：把 [x0..x1]×[y0..y1] 全部格子替换为 color。

        一次性快照一次提交；越界自动裁剪；返回替换像素数，无变化返回 0。
        """
        c = self._snap(color)
        x0, x1 = sorted((int(x0), int(x1)))
        y0, y1 = sorted((int(y0), int(y1)))
        x0 = max(0, min(self.width - 1, x0))
        x1 = max(0, min(self.width - 1, x1))
        y0 = max(0, min(self.height - 1, y0))
        y1 = max(0, min(self.height - 1, y1))
        if x1 < x0 or y1 < y0:
            return 0
        arr = np.asarray(self._img).copy()  # PIL 视图只读，须复制后写入
        sub = arr[y0 : y1 + 1, x0 : x1 + 1]
        target = np.array(c, dtype=np.uint8)
        mask = (sub != target).any(axis=-1)
        n = int(mask.sum())
        if n == 0:
            return 0
        self._snapshot()
        sub[mask] = target
        self._commit(arr)
        return n
