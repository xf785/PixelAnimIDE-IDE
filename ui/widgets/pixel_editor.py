"""像素编辑画布控件：基于 PixelCanvas 数据模型的 QWidget 编辑器。

支持：铅笔 / 橡皮 / 取色 / 填充；整数倍缩放（NEAREST）+ 滚轮缩放；透明棋盘格背景；
像素网格线；撤销 / 重做（Ctrl+Z / Ctrl+Shift+Z 或按钮）。
编辑结果通过 `edited` 信号通知外部（用于标记项目已修改、刷新时间轴缩略图）。
调色板按「色族」显示：相近颜色聚为一族（白色族/红色族/淡红色族…），
右键以整族为单位替换并保留族内渐变。
"""
from __future__ import annotations

import colorsys
import logging
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from PySide6.QtCore import QPoint, QPointF, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPalette, QPen, QWheelEvent
from PySide6.QtWidgets import (
    QColorDialog,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.editing import PixelCanvas
from core.editing.canvas import SYM_BOTH, SYM_H, SYM_NONE, SYM_V
from ui.i18n import T, tr
from ui.layout import scaled

logger = logging.getLogger("PixelFoundry.ui.pixel_editor")


class Tool(Enum):
    PENCIL = "pencil"
    ERASER = "eraser"
    EYEDROPPER = "eyedropper"
    FILL = "fill"
    LINE = "line"
    RECT = "rect"
    ELLIPSE = "ellipse"
    SELECT = "select"


#: 形状工具（拖动预览，松开落笔）
SHAPE_TOOLS = (Tool.LINE, Tool.RECT, Tool.ELLIPSE)

# 调色板显示数量（前 N 个高频色族 + … 弹窗看完整）
PALETTE_SHOW = 6

TRANSPARENT = (0, 0, 0, 0)


def parse_gpl(text: str) -> List[Tuple[int, int, int, int]]:
    """解析 GIMP .gpl / 纯文本调色板："R G B[ 名称]" 每行一色。

    忽略 GIMP Palette / Name: / Columns: / # 注释等头部行；解析失败的行跳过。
    """
    colors: List[Tuple[int, int, int, int]] = []
    for raw in str(text).splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        parts = line.replace("\t", " ").split()
        if len(parts) < 3:
            continue
        try:
            r, g, b = (int(float(parts[0])), int(float(parts[1])), int(float(parts[2])))
        except ValueError:
            continue
        c = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)), 255)
        if c not in colors:
            colors.append(c)
    return colors

# 色族聚类阈值（RGB 欧氏距离，超过则另起一族）
FAMILY_THRESHOLD = 70

# 右侧图标控制列宽度（展开 / 收起）
SIDE_W = 46
SIDE_MIN = 24

# 画布背景模式：key -> (图标, 名称)
BG_MODES = [
    ("checker", "▦", "灰黑网格"),
    ("white", "⬜", "纯白"),
    ("black", "⬛", "纯黑"),
    ("green", "🟩", "纯绿"),
]


# --------------------------------------------------------------------------- #
# 色族算法：聚类 + 中文命名 + 渐变保留的整体换色映射（数值算法，暴力可接受）
# --------------------------------------------------------------------------- #
def cluster_color_families(counts: List[Tuple[int, Tuple[int, int, int, int]]], threshold: int = FAMILY_THRESHOLD):
    """把按频率降序的 (count, rgba) 颜色统计聚类为色族。

    贪心聚类，O(唯一色数 × 族数)：与已有族「代表色（最高频成员）」的 RGB 距离
    ≤ threshold 归入该族，否则新建一族。返回 [(total, rep, members)] 按总像素降序。
    """
    families: List[Tuple[Tuple[int, int, int, int], int, list]] = []  # (rep, total, members)
    for cnt, rgba in counts:
        best_i, best_d = -1, float("inf")
        for i, (frep, _ft, _fm) in enumerate(families):
            d = sum((frep[k] - rgba[k]) ** 2 for k in range(3))
            if d < best_d:
                best_i, best_d = i, d
        if best_i >= 0 and best_d <= threshold * threshold:
            frep, ftotal, fmembers = families[best_i]
            fmembers.append(rgba)
            families[best_i] = (frep, ftotal + cnt, fmembers)
        else:
            families.append((rgba, cnt, [rgba]))
    return sorted(families, key=lambda x: -x[1])


def color_family_name(rgba: Tuple[int, int, int, int]) -> str:
    """按代表色生成中文色族名：白色族 / 红色族 / 淡红色族 / 深蓝色族 / 灰色族 …"""
    r, g, b, a = rgba
    if a < 16:
        return tr("透明族")
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    if v < 0.16:
        return tr("黑色族")
    if s < 0.12:
        if v > 0.88:
            return tr("白色族")
        if v > 0.55:
            return tr("浅灰族")
        return tr("灰色族")
    hue = h * 360.0
    if hue >= 335 or hue < 20:
        name = tr("红色族")
    elif hue < 45:
        name = tr("橙色族")
    elif hue < 70:
        name = tr("黄色族")
    elif hue < 160:
        name = tr("绿色族")
    elif hue < 200:
        name = tr("青色族")
    elif hue < 250:
        name = tr("蓝色族")
    elif hue < 290:
        name = tr("紫色族")
    else:
        name = tr("品红色族")
    if v > 0.85 and s < 0.55:
        name = tr("淡{0}").format(tr(name))
    elif v < 0.35:
        name = tr("深{0}").format(tr(name))
    return name


def family_replace_mapping(members: List[Tuple[int, int, int, int]], rep, new_base) -> dict:
    """色族整体换色映射：族内各色保留与代表色的相对差（渐变保留）。

    new = clamp(new_base + (old - rep))，alpha 保持成员原值。
    """
    mapping = {}
    for c in members:
        mapping[c] = (
            max(0, min(255, new_base[0] + c[0] - rep[0])),
            max(0, min(255, new_base[1] + c[1] - rep[1])),
            max(0, min(255, new_base[2] + c[2] - rep[2])),
            c[3],
        )
    return mapping


def _swatch_style(rgba: Tuple[int, int, int, int]) -> str:
    r, g, b, a = rgba
    return (
        f"background-color: rgba({r},{g},{b},{a});"
        "border: 1px solid rgba(255,255,255,0.25); border-radius: 4px;"
    )


def _family_tooltip(name: str, rep: Tuple[int, int, int, int], members: List[Tuple[int, int, int, int]]) -> str:
    if rep[3] == 0:
        return tr("{0} · {1} 色（左键选色，右键替换色族）").format(tr(name), len(members))
    return (
        tr("{0} · {1} 色 · 代表 #{2}（左键选色，右键替换色族）").format(
            tr(name), len(members), f"{rep[0]:02x}{rep[1]:02x}{rep[2]:02x}"
        )
    )


def _to_qimage(img: Image.Image) -> QImage:
    arr = np.asarray(img.convert("RGBA")).copy()
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()


def _to_qimage_arr(arr: np.ndarray) -> QImage:
    """RGBA uint8 数组 -> QImage（调用方保证连续）。"""
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()


def _lasso_mask(points: List[Tuple[int, int]], w: int, h: int) -> np.ndarray:
    """把套索折线围成的多边形填为布尔掩膜。"""
    from PIL import Image, ImageDraw

    if len(points) < 3:
        return np.zeros((h, w), dtype=bool)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).polygon([(x, y) for x, y in points], fill=255)
    return np.asarray(mask, dtype=bool)


class PixelEditorWidget(QWidget):
    """像素画布编辑器。"""

    edited = Signal()                 # 像素被修改
    color_picked = Signal(object)     # 取色结果（RGBA tuple）

    def __init__(self, parent=None):
        super().__init__(parent)
        self._canvas = PixelCanvas(Image.new("RGBA", (16, 16), TRANSPARENT))
        self._qimg = _to_qimage(self._canvas.image)
        self._tool: Tool = Tool.PENCIL
        self._color: Tuple[int, int, int, int] = (0, 0, 0, 255)
        self._zoom = 1
        self._pan_x = 0  # 滚轮/缩放后的视图平移（相对居中基准），支持以光标为焦点缩放
        self._pan_y = 0
        self._drawing = False
        self._last_cell: Optional[Tuple[int, int]] = None
        self._onion_enabled = False
        self._onion_prev_qimg: Optional[QImage] = None
        self._onion_next_qimg: Optional[QImage] = None
        self._palette_locked = False
        self._recent_colors: List[Tuple[int, int, int, int]] = []
        self._wheel_popup = None
        # 画布平移（Ctrl+左键拖动）
        self._panning = False
        self._pan_start: Optional[QPoint] = None
        self._pan_origin = (0, 0)
        # 右键框选填充（拖动）与取色圆盘（按住/快速点击）
        self._rb_active = False
        self._rb_anchor: Optional[QPoint] = None
        self._rb_cell0: Optional[Tuple[int, int]] = None
        self._rb_cell1: Optional[Tuple[int, int]] = None
        self._wheel_timer: Optional[QTimer] = None
        self._wheel_opened = False
        # 画布背景 / 网格 / 面板收起状态
        self._bg_mode = "checker"   # checker | white | black | green
        self._grid_visible = True
        self._side_collapsed = False  # 右侧图标列默认展开显示
        # 笔刷 / 填充 / 选区 / 浮动图层
        self._brush_size = 1
        self._fill_global = False
        self._sel_mode = "rect"      # rect | lasso
        # 形状工具 / 对称 / 环绕（无缝瓦片）
        self._symmetry = SYM_NONE
        self._wrap = False
        self._shape_start: Optional[Tuple[int, int]] = None
        self._shape_cur: Optional[Tuple[int, int]] = None
        self._shape_filled = False    # 矩形/椭圆：填充 or 描边
        self._selection: Optional[np.ndarray] = None   # bool (h, w)
        self._sel_hl_qimg: Optional[QImage] = None
        self._sel_border_segments: List[Tuple[int, int, int, int]] = []  # 屏幕空间细线（蓝虚线）
        self._sel_dragging = False
        self._sel_anchor_cell: Optional[Tuple[int, int]] = None
        self._sel_rect_cur: Optional[Tuple[int, int]] = None
        self._sel_lasso_points: List[Tuple[int, int]] = []
        self._float_layer: Optional[Image.Image] = None  # Ctrl+V 粘贴的浮动图层（半透明）
        self._float_pos = (0, 0)
        self._float_qimg: Optional[QImage] = None
        self._float_opacity = 0.55                        # 浮动层半透明显示
        self._clipboard: Optional[Image.Image] = None     # Ctrl+C 复制的内容
        self._clipboard_pos = (0, 0)
        self._moving_float = False
        self._float_grab = (0, 0)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(160, 160)
        self._build_toolbar()

    # ------------------------------------------------------------------ #
    def _build_toolbar(self) -> None:
        """布局：画布占满主体，控件收敛到右侧图标列 + 底部调色板条（均可收起）。"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        main = QHBoxLayout()
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)
        self._canvas_host = _CanvasView(self)
        main.addWidget(self._canvas_host, 1)
        self._side_panel = self._build_side_panel()
        main.addWidget(self._side_panel)
        layout.addLayout(main, 1)

        self._palette_bar = self._build_palette_bar()
        layout.addWidget(self._palette_bar)
        self._refresh_palette()
        self._apply_side_collapsed()  # 初始为收起态（三角图标 + 隐藏图标列）

    def _icon_btn(self, kind: str, tip: str, checkable: bool = False) -> QToolButton:
        """右侧图标按钮：QPainter 自绘 16px 扁平线条图标（DSH 风格统一）。

        tip 为中文原文（i18n ID），经 T() 注册，语言切换时自动重译。
        """
        from ui.icons import editor_icon

        btn = QToolButton()
        btn.setIcon(editor_icon(kind, "#9aa0a8", size=16))
        btn.setIconSize(QSize(scaled(16), scaled(16)))
        T(btn, tip, attr="tooltip")
        btn.setCheckable(checkable)
        btn.setFixedSize(scaled(34), scaled(34))
        return btn
    def _build_side_panel(self) -> QWidget:
        """右侧图标控制列：默认收起（仅剩一个三角展开钮），点击展开全部图标。

        左键选择工具；右键弹出第二级详细选项（笔刷大小 / 选择方式 / 填充方式 / 背景档）。
        """
        panel = QWidget()
        panel.setObjectName("EditorSidePanel")
        panel.setFixedWidth(SIDE_MIN)  # 默认收起
        # 图标列放进滚动区：窗口不高时也不会把底部按钮挤掉（Krita 的工具箱同理）
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        scroll = QScrollArea()
        scroll.setObjectName("EditorToolScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        inner = QWidget()
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        v = QVBoxLayout(inner)
        v.setContentsMargins(3, 4, 3, 4)
        v.setSpacing(4)

        self._side_buttons: List[QWidget] = []

        self._toggle_side_btn = self._icon_btn("chevron_left", "展开控制面板")
        self._toggle_side_btn.clicked.connect(self._on_toggle_side)
        v.addWidget(self._toggle_side_btn, 0, Qt.AlignmentFlag.AlignHCenter)

        self._color_swatch = QLabel()
        self._color_swatch.setFixedSize(scaled(30), scaled(30))
        T(self._color_swatch, "当前颜色", attr="tooltip")
        self._color_swatch.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._update_swatch()
        self._side_buttons.append(self._color_swatch)
        v.addWidget(self._color_swatch, 0, Qt.AlignmentFlag.AlignHCenter)

        # 工具（图标，左键选择 / 右键二级菜单）
        self._tool_buttons = {}
        tool_defs = [
            (Tool.PENCIL, "pencil", tr("铅笔（右键笔刷大小）")),
            (Tool.ERASER, "eraser", tr("橡皮（右键笔刷大小）")),
            (Tool.EYEDROPPER, "eyedropper", tr("取色")),
            (Tool.FILL, "fill", tr("填充（右键填充方式）")),
            (Tool.LINE, "line", tr("直线（拖动预览，右键笔刷粗细）")),
            (Tool.RECT, "rect", tr("矩形（右键：描边 / 填充）")),
            (Tool.ELLIPSE, "ellipse", tr("椭圆（右键：描边 / 填充）")),
            (Tool.SELECT, "select", tr("选择（右键框选/套索；Ctrl+左键加点；Ctrl+C 复制，Ctrl+V 粘贴半透明新图层；Ctrl+右键拖拽移动；Ctrl+M 合并）")),
        ]
        for tool, kind, label in tool_defs:
            btn = self._icon_btn(kind, label, checkable=True)
            btn.clicked.connect(lambda _=False, t=tool: self.set_tool(t))
            btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            btn.customContextMenuRequested.connect(lambda pos, t=tool, b=btn: self._on_tool_menu(t, b, pos))
            self._tool_buttons[tool] = btn
            self._side_buttons.append(btn)
            v.addWidget(btn, 0, Qt.AlignmentFlag.AlignHCenter)
        self._tool_buttons[Tool.PENCIL].setChecked(True)
        v.addSpacing(6)

        # 对称 / 环绕（无缝瓦片）：两个开关
        self._sym_btn = self._icon_btn("symmetry", "对称绘制：关闭（左键切换，右键选轴）", checkable=True)
        self._sym_btn.clicked.connect(self._on_symmetry_clicked)
        self._sym_btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._sym_btn.customContextMenuRequested.connect(lambda pos, b=self._sym_btn: self._on_symmetry_menu(b, pos))
        self._side_buttons.append(self._sym_btn)
        v.addWidget(self._sym_btn, 0, Qt.AlignmentFlag.AlignHCenter)

        self._wrap_btn = self._icon_btn("wrap", tr("环绕绘制：笔刷越过边界回到另一侧（画无缝瓦片）"), checkable=True)
        self._wrap_btn.toggled.connect(self._on_wrap_toggled)
        self._side_buttons.append(self._wrap_btn)
        v.addWidget(self._wrap_btn, 0, Qt.AlignmentFlag.AlignHCenter)

        self._transform_btn = self._icon_btn("transform", tr("画布变换：翻转 / 旋转 / 裁剪 / 尺寸 / 缩放"))
        self._transform_btn.clicked.connect(lambda: self._on_transform_menu(self._transform_btn))
        self._side_buttons.append(self._transform_btn)
        v.addWidget(self._transform_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        v.addSpacing(6)

        # 撤销 / 重做
        self._undo_btn = self._icon_btn("undo", "撤销（Ctrl+Z）")
        self._undo_btn.clicked.connect(self.undo)
        self._side_buttons.append(self._undo_btn)
        v.addWidget(self._undo_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        self._redo_btn = self._icon_btn("redo", "重做（Ctrl+Shift+Z）")
        self._redo_btn.clicked.connect(self.redo)
        self._side_buttons.append(self._redo_btn)
        v.addWidget(self._redo_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        v.addSpacing(6)

        # 缩放
        zoom_out = self._icon_btn("zoom_out", "缩小")
        zoom_out.clicked.connect(lambda: self._set_zoom(self._zoom - 1))
        self._side_buttons.append(zoom_out)
        v.addWidget(zoom_out, 0, Qt.AlignmentFlag.AlignHCenter)
        self._zoom_label = QLabel("1x")
        self._zoom_label.setFixedWidth(scaled(40))
        self._zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._side_buttons.append(self._zoom_label)
        v.addWidget(self._zoom_label, 0, Qt.AlignmentFlag.AlignHCenter)
        zoom_in = self._icon_btn("zoom_in", "放大")
        zoom_in.clicked.connect(lambda: self._set_zoom(self._zoom + 1))
        self._side_buttons.append(zoom_in)
        v.addWidget(zoom_in, 0, Qt.AlignmentFlag.AlignHCenter)
        v.addSpacing(6)

        # 洋葱皮 / 调色板锁定 / 提取调色板
        self._onion_btn = self._icon_btn("onion", "洋葱皮：显示相邻帧半透明幽灵", checkable=True)
        self._onion_btn.toggled.connect(self._on_onion_toggled)
        self._side_buttons.append(self._onion_btn)
        v.addWidget(self._onion_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        self._palette_lock_btn = self._icon_btn("lock", "锁定调色板：绘制/填充吸附到当前帧调色板", checkable=True)
        self._palette_lock_btn.toggled.connect(self._on_palette_lock_toggled)
        self._side_buttons.append(self._palette_lock_btn)
        v.addWidget(self._palette_lock_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        self._extract_palette_btn = self._icon_btn("palette", "提取调色板并锁定")
        self._extract_palette_btn.clicked.connect(self._on_extract_palette)
        self._side_buttons.append(self._extract_palette_btn)
        v.addWidget(self._extract_palette_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        v.addSpacing(6)
        v.addSpacing(6)

        # 网格显示/隐藏 + 背景切换 + 调色板条开关
        self._grid_btn = self._icon_btn("grid", "显示/隐藏像素网格", checkable=True)
        self._grid_btn.setChecked(self._grid_visible)
        self._grid_btn.toggled.connect(self._on_grid_toggled)
        self._side_buttons.append(self._grid_btn)
        v.addWidget(self._grid_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        self._bg_btn = self._icon_btn("background", "背景：灰黑网格（点击切换，右键选档）")
        self._bg_btn.clicked.connect(self._on_bg_cycle)
        self._bg_btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._bg_btn.customContextMenuRequested.connect(lambda pos, b=self._bg_btn: self._on_bg_menu(b, pos))
        self._side_buttons.append(self._bg_btn)
        v.addWidget(self._bg_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        self._toggle_palette_btn = self._icon_btn("palette", "显示/隐藏调色板", checkable=True)
        self._toggle_palette_btn.setChecked(True)
        self._toggle_palette_btn.toggled.connect(self._on_toggle_palette)
        self._side_buttons.append(self._toggle_palette_btn)
        v.addWidget(self._toggle_palette_btn, 0, Qt.AlignmentFlag.AlignHCenter)

        v.addStretch(1)
        return panel

    def _build_palette_bar(self) -> QWidget:
        """底部调色板条：当前色族 swatches + … 弹窗 + 收起按钮（可整体隐藏）。"""
        bar = QWidget()
        bar.setObjectName("EditorPaletteBar")
        h = QHBoxLayout(bar)
        h.setContentsMargins(6, 4, 6, 4)
        h.setSpacing(2)
        self._palette_row = h
        self._palette_swatches: list = []
        self._palette_more_btn = QToolButton()
        self._palette_more_btn.setText("…")
        self._palette_more_btn.setFixedSize(scaled(22), scaled(22))
        T(self._palette_more_btn, "查看完整色族调色板", attr="tooltip")
        self._palette_more_btn.clicked.connect(self._on_palette_dialog)
        h.addWidget(self._palette_more_btn)
        h.addStretch(1)
        self._palette_bar_collapse_btn = QToolButton()
        self._palette_bar_collapse_btn.setText("▼")
        self._palette_bar_collapse_btn.setFixedSize(scaled(20), scaled(20))
        T(self._palette_bar_collapse_btn, "收起调色板", attr="tooltip")
        self._palette_bar_collapse_btn.clicked.connect(self._on_collapse_palette)
        h.addWidget(self._palette_bar_collapse_btn)
        return bar

    # ------------------------------------------------------------------ #
    # 面板收起 / 背景 / 网格
    # ------------------------------------------------------------------ #
    def _on_toggle_side(self) -> None:
        """右侧图标列展开/收起（收起后画布更大，仅剩三角展开钮）。"""
        self._side_collapsed = not self._side_collapsed
        self._apply_side_collapsed()

    def _apply_side_collapsed(self) -> None:
        """按 _side_collapsed 应用：隐藏图标列只留三角钮；三角方向指示展开/收起。"""
        from ui.icons import editor_icon

        for w in self._side_buttons:
            w.setVisible(not self._side_collapsed)
        self._side_panel.setFixedWidth(scaled(SIDE_MIN) if self._side_collapsed else scaled(SIDE_W))
        kind = "chevron_left" if self._side_collapsed else "chevron_right"
        self._toggle_side_btn.setIcon(editor_icon(kind, "#9aa0a8", size=16))
        T(self._toggle_side_btn, "展开控制面板" if self._side_collapsed else "收起控制面板", attr="tooltip")

    def retranslate_ui(self) -> None:
        """语言切换后重刷动态文案（色族色块提示、对称按钮提示等）。"""
        self._refresh_palette()
        self.set_symmetry(self._symmetry)
        T(self._wrap_btn, "环绕绘制：笔刷越过边界回到另一侧（画无缝瓦片）", attr="tooltip")

    def apply_ui_scale(self, scale: float) -> None:
        """按界面布局比例同步缩放编辑器全部固定尺寸（按钮/面板/调色板）。"""
        self._color_swatch.setFixedSize(scaled(30), scaled(30))
        self._zoom_label.setFixedWidth(scaled(40))
        for b in [self._toggle_side_btn] + list(self._side_buttons):
            if isinstance(b, QToolButton):
                b.setFixedSize(scaled(34), scaled(34))
        self._palette_more_btn.setFixedSize(scaled(22), scaled(22))
        self._palette_bar_collapse_btn.setFixedSize(scaled(20), scaled(20))
        self._apply_side_collapsed()
        self._refresh_palette()
        self._canvas_host.update()

    def _on_toggle_palette(self, checked: bool) -> None:
        self._palette_bar.setVisible(checked)
        self._toggle_palette_btn.setChecked(checked)

    def _on_collapse_palette(self) -> None:
        self._toggle_palette_btn.setChecked(False)
        self._palette_bar.setVisible(False)

    def _on_grid_toggled(self, checked: bool) -> None:
        self._grid_visible = bool(checked)
        self._canvas_host.update()

    def _on_bg_cycle(self) -> None:
        """循环切换背景：灰黑网格 -> 纯白 -> 纯黑 -> 纯绿。"""
        idx = next((i for i, (k, _g, _n) in enumerate(BG_MODES) if k == self._bg_mode), 0)
        key, _glyph, label = BG_MODES[(idx + 1) % len(BG_MODES)]
        self._set_bg_mode(key)
        # 注册具体化 ID（如「背景：纯白（点击切换，右键选档）」），语言切换自动重译
        T(self._bg_btn, "背景：{0}（点击切换，右键选档）".format(label), attr="tooltip")

    def _set_bg_mode(self, key: str) -> None:
        self._bg_mode = key if key in (k for k, _g, _n in BG_MODES) else "checker"
        self._canvas_host.update()

    def _on_bg_menu(self, btn, pos: QPoint) -> None:
        """右键背景按钮：直接选择背景档。"""
        menu = QMenu(self)
        for key, _glyph, label in BG_MODES:
            act = menu.addAction(tr(label))
            act.setCheckable(True)
            act.setChecked(self._bg_mode == key)
            act.triggered.connect(lambda _=False, k=key: self._set_bg_mode(k))
        menu.exec(btn.mapToGlobal(pos))

    # ------------------------------------------------------------------ #
    # 工具右键二级菜单（笔刷大小 / 选择方式 / 填充方式）
    # ------------------------------------------------------------------ #
    def _on_tool_menu(self, tool: Tool, btn, pos: QPoint) -> None:
        menu = QMenu(self)
        if tool in (Tool.PENCIL, Tool.ERASER):
            menu.addSection(tr("笔刷大小"))
            for sz in (1, 2, 3, 4, 6, 8):
                act = menu.addAction(f"{sz}×{sz}")
                act.setCheckable(True)
                act.setChecked(self._brush_size == sz)
                act.triggered.connect(lambda _=False, s=sz: self._set_brush_size(s))
        elif tool == Tool.SELECT:
            menu.addSection(tr("选择方式"))
            act = menu.addAction(tr("矩形框选"))
            act.setCheckable(True)
            act.setChecked(self._sel_mode == "rect")
            act.triggered.connect(lambda _=False: self._set_sel_mode("rect"))
            act = menu.addAction(tr("套索选择"))
            act.setCheckable(True)
            act.setChecked(self._sel_mode == "lasso")
            act.triggered.connect(lambda _=False: self._set_sel_mode("lasso"))
            menu.addSeparator()
            menu.addAction(tr("全部选择")).triggered.connect(self._select_all)
            menu.addAction(tr("取消选择")).triggered.connect(self._clear_selection)
            menu.addAction(tr("复制 (Ctrl+C)")).triggered.connect(self._copy_selection)
            menu.addAction(tr("粘贴为新图层 (Ctrl+V)")).triggered.connect(self._paste_layer)
            menu.addAction(tr("合并图层 (Ctrl+M)")).triggered.connect(self._merge_float_layer)
        elif tool == Tool.FILL:
            menu.addSection(tr("填充方式"))
            act = menu.addAction(tr("连通区域填充"))
            act.setCheckable(True)
            act.setChecked(not self._fill_global)
            act.triggered.connect(lambda _=False: self._set_fill_global(False))
            act = menu.addAction(tr("全局同色替换"))
            act.setCheckable(True)
            act.setChecked(self._fill_global)
            act.triggered.connect(lambda _=False: self._set_fill_global(True))
        elif tool == Tool.LINE:
            menu.addSection(tr("笔刷粗细"))
            for sz in (1, 2, 3, 4, 6, 8):
                act = menu.addAction(f"{sz}×{sz}")
                act.setCheckable(True)
                act.setChecked(self._brush_size == sz)
                act.triggered.connect(lambda _=False, s=sz: self._set_brush_size(s))
        elif tool in (Tool.RECT, Tool.ELLIPSE):
            menu.addSection(tr("形状样式"))
            for filled, label in ((False, tr("描边")), (True, tr("填充"))):
                act = menu.addAction(label)
                act.setCheckable(True)
                act.setChecked(self._shape_filled is filled)
                act.triggered.connect(lambda _=False, f=filled: self._set_shape_filled(f))
            menu.addSeparator()
            menu.addSection(tr("笔刷粗细"))
            for sz in (1, 2, 3, 4, 6, 8):
                act = menu.addAction(f"{sz}×{sz}")
                act.setCheckable(True)
                act.setChecked(self._brush_size == sz)
                act.triggered.connect(lambda _=False, s=sz: self._set_brush_size(s))
        menu.exec(btn.mapToGlobal(pos))

    def _set_shape_filled(self, filled: bool) -> None:
        """矩形/椭圆：填充 or 仅描边（下次落笔生效）。"""
        self._shape_filled = bool(filled)

    # ------------------------------------------------------------------ #
    # 对称绘制 / 环绕绘制（无缝瓦片）/ 画布变换
    # ------------------------------------------------------------------ #
    def symmetry(self) -> str:
        return self._symmetry

    def set_symmetry(self, mode: str) -> None:
        """设置对称模式（none / h / v / both），并同步按钮状态与提示。"""
        self._symmetry = mode if mode in (SYM_NONE, SYM_H, SYM_V, SYM_BOTH) else SYM_NONE
        self._sym_btn.setChecked(self._symmetry != SYM_NONE)
        tip = {
            SYM_NONE: "对称绘制：关闭（左键切换，右键选轴）",
            SYM_H: "对称绘制：左右镜像（左键关闭，右键选轴）",
            SYM_V: "对称绘制：上下镜像（左键关闭，右键选轴）",
            SYM_BOTH: "对称绘制：四象限镜像（左键关闭，右键选轴）",
        }[self._symmetry]
        T(self._sym_btn, tip, attr="tooltip")
        self._canvas_host.update()

    def _on_symmetry_clicked(self, checked: bool) -> None:
        """左键：关闭 -> 左右 -> 上下 -> 四象限 循环。"""
        if not checked:
            self.set_symmetry(SYM_NONE)
            return
        order = (SYM_H, SYM_V, SYM_BOTH)
        try:
            nxt = order[(order.index(self._symmetry) + 1) % len(order)]
        except ValueError:
            nxt = SYM_H
        self.set_symmetry(nxt)

    def _on_symmetry_menu(self, btn, pos: QPoint) -> None:
        menu = QMenu(self)
        menu.addSection(tr("对称轴"))
        for key, label in ((SYM_NONE, "关闭"), (SYM_H, "左右镜像"), (SYM_V, "上下镜像"), (SYM_BOTH, "四象限镜像")):
            act = menu.addAction(tr(label))
            act.setCheckable(True)
            act.setChecked(self._symmetry == key)
            act.triggered.connect(lambda _=False, k=key: self.set_symmetry(k))
        menu.exec(btn.mapToGlobal(pos))

    def wrap_enabled(self) -> bool:
        return self._wrap

    def set_wrap(self, enabled: bool) -> None:
        """环绕绘制：笔刷越过边界后从另一侧继续（画无缝瓦片）。"""
        self._wrap = bool(enabled)
        self._wrap_btn.setChecked(self._wrap)
        self._canvas_host.update()

    def _on_wrap_toggled(self, checked: bool) -> None:
        self.set_wrap(checked)

    def _on_transform_menu(self, btn) -> None:
        """画布变换菜单：翻转 / 旋转 / 裁剪 / 尺寸 / 缩放。"""
        menu = QMenu(self)
        menu.addAction(tr("水平翻转")).triggered.connect(lambda: self.transform("flip_h"))
        menu.addAction(tr("垂直翻转")).triggered.connect(lambda: self.transform("flip_v"))
        menu.addSeparator()
        menu.addAction(tr("顺时针旋转 90°")).triggered.connect(lambda: self.transform("rotate_cw"))
        menu.addAction(tr("逆时针旋转 90°")).triggered.connect(lambda: self.transform("rotate_ccw"))
        menu.addSeparator()
        crop = menu.addAction(tr("裁剪到选区"))
        crop.setEnabled(bool(self._selection is not None and self._selection.any()))
        crop.triggered.connect(lambda: self.transform("crop_selection"))
        menu.addAction(tr("画布尺寸…")).triggered.connect(lambda: self.transform("resize"))
        menu.addAction(tr("内容整数倍缩放…")).triggered.connect(lambda: self.transform("scale"))
        menu.exec(btn.mapToGlobal(btn.rect().bottomLeft()))

    def transform(self, action: str) -> bool:
        """执行画布变换（供按钮 / 主窗口菜单调用），返回是否发生了变化。"""
        before = self._canvas.size
        ok = False
        if action == "flip_h":
            ok = self._canvas.flip_horizontal()
        elif action == "flip_v":
            ok = self._canvas.flip_vertical()
        elif action == "rotate_cw":
            ok = self._canvas.rotate_90(clockwise=True)
        elif action == "rotate_ccw":
            ok = self._canvas.rotate_90(clockwise=False)
        elif action == "crop_selection":
            ok = self._crop_to_selection()
        elif action == "resize":
            ok = self._resize_canvas_dialog()
        elif action == "scale":
            ok = self._scale_content_dialog()
        if ok:
            self._clear_selection()
            self._rebuild()
            self._refresh_palette()
            self.edited.emit()
            self._canvas_host.update()
        if self._canvas.size != before:
            self._clamp_pan()
        return ok

    def _crop_to_selection(self) -> bool:
        sel = self._selection
        if sel is None or not sel.any():
            return False
        ys, xs = np.nonzero(sel)
        return self._canvas.crop_to((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))

    def _resize_canvas_dialog(self) -> bool:
        from PySide6.QtWidgets import QInputDialog

        w, ok = QInputDialog.getInt(self, tr("画布尺寸"), tr("宽度(px)"), self._canvas.width, 1, 4096, 1)
        if not ok:
            return False
        h, ok = QInputDialog.getInt(self, tr("画布尺寸"), tr("高度(px)"), self._canvas.height, 1, 4096, 1)
        if not ok:
            return False
        anchor, ok = QInputDialog.getItem(
            self, tr("画布尺寸"), tr("内容对齐"),
            [tr("居中"), tr("左上"), tr("右上"), tr("左下"), tr("右下")], 0, False,
        )
        if not ok:
            return False
        key = {"居中": "center", "左上": "tl", "右上": "tr", "左下": "bl", "右下": "br"}.get(anchor, "center")
        return self._canvas.resize_canvas(w, h, key)

    def _scale_content_dialog(self) -> bool:
        from PySide6.QtWidgets import QInputDialog

        factor, ok = QInputDialog.getInt(self, tr("内容缩放"), tr("放大整数倍"), 2, 1, 16, 1)
        if not ok:
            return False
        return self._canvas.scale_content(factor)

    def _set_brush_size(self, size: int) -> None:
        self._brush_size = max(1, min(8, int(size)))

    def _set_sel_mode(self, mode: str) -> None:
        self._sel_mode = "lasso" if mode == "lasso" else "rect"

    def _set_fill_global(self, global_fill: bool) -> None:
        self._fill_global = bool(global_fill)

    def _select_all(self) -> None:
        self._selection = np.ones((self._canvas.height, self._canvas.width), dtype=bool)
        self._rebuild_sel_overlay()
        self._canvas_host.update()

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #
    def set_tool(self, tool: Tool) -> None:
        self._tool = tool
        for t, btn in self._tool_buttons.items():
            btn.setChecked(t == tool)

    def tool(self) -> Tool:
        return self._tool

    def set_color(self, color, record: bool = True) -> None:
        """设置当前颜色；record=True 时记入最近使用色（取色圆盘底部显示）。"""
        c = tuple(int(v) for v in color)
        if len(c) == 3:
            c = (c[0], c[1], c[2], 255)
        c = (c[0], c[1], c[2], c[3])
        if record:
            self._record_recent(c)
        self._color = c
        self._update_swatch()

    def color(self) -> Tuple[int, int, int, int]:
        return self._color

    def _record_recent(self, c) -> None:
        if c not in self._recent_colors:
            self._recent_colors.insert(0, c)
        elif self._recent_colors[0] != c:
            self._recent_colors.remove(c)
            self._recent_colors.insert(0, c)
        del self._recent_colors[10:]

    # ------------------------------------------------------------------ #
    # Krita 式右键取色圆盘
    # ------------------------------------------------------------------ #
    def open_color_wheel(self, global_pos: QPoint) -> None:
        """在 global_pos 处弹出取色圆盘；按住移动取色，松开提交。"""
        from ui.widgets.color_wheel import ColorWheelPopup

        if self._wheel_popup is not None:
            try:
                self._wheel_popup.close()
            except RuntimeError:  # noqa: BLE001
                pass
            self._wheel_popup = None
        popup = ColorWheelPopup(self._color, self._recent_colors, self)
        popup.color_preview.connect(lambda c: self.set_color(c, record=False))
        popup.color_selected.connect(self._on_wheel_commit)
        popup.destroyed.connect(lambda: setattr(self, "_wheel_popup", None))
        self._wheel_popup = popup
        popup.show_at(global_pos)

    def _on_wheel_commit(self, color) -> None:
        c = tuple(int(v) for v in color)
        if len(c) == 3:
            c = (c[0], c[1], c[2], 255)
        c = self._canvas.snap_color(c)  # 锁定调色板时吸附到最近锁定色
        self.set_color(c)
        self.color_picked.emit(c)

    # ------------------------------------------------------------------ #
    # 画布平移（Ctrl + 左键拖动）
    # ------------------------------------------------------------------ #
    def _on_pan_press(self, event) -> None:
        self._panning = True
        self._pan_start = event.position().toPoint()
        self._pan_origin = (self._pan_x, self._pan_y)
        self._canvas_host.setCursor(Qt.CursorShape.ClosedHandCursor)
        event.accept()

    def _on_pan_move(self, pos: QPoint) -> None:
        if not self._panning or self._pan_start is None:
            return
        self._pan_x = self._pan_origin[0] + (pos.x() - self._pan_start.x())
        self._pan_y = self._pan_origin[1] + (pos.y() - self._pan_start.y())
        self._clamp_pan()
        self._canvas_host.update()

    def _on_pan_release(self) -> None:
        self._panning = False
        self._pan_start = None
        self._canvas_host.setCursor(Qt.CursorShape.ArrowCursor)

    def _clamp_pan(self) -> None:
        """限制平移幅度：保证画布仍与视口相交（不会完全拖出视野）。"""
        fw, fh = self._canvas.width, self._canvas.height
        z = self._zoom
        self._pan_x = max(-fw * z, min(fw * z, self._pan_x))
        self._pan_y = max(-fh * z, min(fh * z, self._pan_y))

    # ------------------------------------------------------------------ #
    # 右键：拖动 = 框选区域填充；按住/快速点击 = 取色圆盘
    # ------------------------------------------------------------------ #
    def _on_right_press(self, event) -> None:
        pos = event.position().toPoint()
        self._rb_active = True
        self._rb_anchor = pos
        cell = self._cell_at(pos)
        self._rb_cell0 = cell
        self._rb_cell1 = cell
        self._wheel_opened = False
        self._canvas_host.setCursor(Qt.CursorShape.CrossCursor)
        self._start_wheel_timer(event.globalPosition().toPoint())

    def _start_wheel_timer(self, global_pos: QPoint) -> None:
        """按住不动超过阈值即弹出取色圆盘（Krita 式按住取色）。"""
        self._stop_wheel_timer()
        self._wheel_timer = QTimer(self)
        self._wheel_timer.setSingleShot(True)
        self._wheel_timer.timeout.connect(lambda: self._open_wheel_from_hold(global_pos))
        self._wheel_timer.start(220)

    def _stop_wheel_timer(self) -> None:
        if self._wheel_timer is not None:
            try:
                self._wheel_timer.stop()
            except RuntimeError:  # noqa: BLE001
                pass
            self._wheel_timer.deleteLater()
            self._wheel_timer = None

    def _open_wheel_from_hold(self, global_pos: QPoint) -> None:
        self._wheel_opened = True
        self._rb_active = False  # 圆盘模式，取消框选
        self._stop_wheel_timer()
        self._canvas_host.setCursor(Qt.CursorShape.ArrowCursor)
        self.open_color_wheel(global_pos)
        self._canvas_host.update()

    def _on_right_move(self, pos: QPoint) -> None:
        if not self._rb_active:
            return
        if self._rb_anchor is not None and (pos - self._rb_anchor).manhattanLength() > 4:
            self._stop_wheel_timer()  # 拖动 -> 框选模式（不再弹圆盘）
        cell = self._cell_at(pos)
        if cell is not None:
            self._rb_cell1 = cell
            self._canvas_host.update()

    def _on_right_release(self, pos: QPoint, global_pos: QPoint) -> None:
        if self._wheel_opened:
            self._rb_active = False  # 取色圆盘已接管
            return
        self._stop_wheel_timer()
        self._canvas_host.setCursor(Qt.CursorShape.ArrowCursor)
        dragged = self._rb_anchor is not None and (pos - self._rb_anchor).manhattanLength() > 4
        if not dragged:
            # 快速点击（未拖动）-> 取色圆盘
            self._rb_active = False
            self._canvas_host.update()
            self.open_color_wheel(global_pos)
            return
        # 框选区域填充：用当前颜色填充选中矩形
        if self._rb_cell0 is not None and self._rb_cell1 is not None:
            n = self._canvas.fill_rect(*self._rb_cell0, *self._rb_cell1, self._color)
            if n:
                self._rebuild()
                self.edited.emit()
        self._rb_active = False
        self._canvas_host.update()

    def set_frame(self, image: Image.Image) -> None:
        """切换/载入一帧（清空历史与选区/浮动图层）。"""
        self._canvas.replace_image(image)
        self._clear_selection()
        self._rebuild()

    # ------------------------------------------------------------------ #
    # 本地导入 / 导出图片
    # ------------------------------------------------------------------ #
    def import_image(self, path=None):
        """从本地导入图片替换当前帧（QFileDialog 选文件；可传 path 免弹窗）。"""
        from PySide6.QtWidgets import QFileDialog

        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, tr("导入图片"), "", tr("图片 (*.png *.jpg *.jpeg *.bmp *.gif *.webp);;所有文件 (*)")
            )
            if not path:
                return None
        try:
            img = Image.open(path).convert("RGBA")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, tr("导入失败"), f"{tr('无法读取图片：')}{exc}")
            return None
        self.set_frame(img)
        self.edited.emit()
        return img

    def export_image(self, path=None):
        """导出当前帧为 PNG（QFileDialog 选路径；可传 path 免弹窗）。"""
        from PySide6.QtWidgets import QFileDialog

        if not path:
            path, _ = QFileDialog.getSaveFileName(
                self, tr("导出当前帧为 PNG"), "frame.png", tr("PNG 图片 (*.png)")
            )
            if not path:
                return None
        try:
            self.frame().save(path, format="PNG")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, tr("导出失败"), f"{tr('保存失败：')}{exc}")
            return None
        return Path(path)

    def frame(self) -> Image.Image:
        return self._canvas.image.copy()

    def set_onion(self, prev: Optional[Image.Image], nxt: Optional[Image.Image]) -> None:
        """设置洋葱皮幽灵帧（相邻帧）；None 表示该侧无帧。"""
        self._onion_prev_qimg = _to_qimage(prev) if prev is not None else None
        self._onion_next_qimg = _to_qimage(nxt) if nxt is not None else None
        self._canvas_host.update()

    def set_onion_enabled(self, enabled: bool) -> None:
        self._onion_enabled = bool(enabled)
        self._onion_btn.setChecked(self._onion_enabled)
        self._canvas_host.update()

    def onion_enabled(self) -> bool:
        return self._onion_enabled

    def canvas(self) -> PixelCanvas:
        return self._canvas

    def undo(self) -> None:
        if self._canvas.undo():
            self._rebuild()

    def redo(self) -> None:
        if self._canvas.redo():
            self._rebuild()

    # ---- 对外公开的选区/图层命令（主窗口菜单与工具条用） ----
    def copy_selection(self) -> None:
        """复制当前选区到剪贴板。"""
        self._copy_selection()

    def paste_layer(self) -> None:
        """把剪贴板内容粘贴成半透明浮动图层。"""
        self._paste_layer()

    def merge_float_layer(self) -> None:
        """把浮动图层合并进当前帧。"""
        self._merge_float_layer()

    def zoom_in(self) -> None:
        self._set_zoom(self._zoom + 1)

    def zoom_out(self) -> None:
        self._set_zoom(self._zoom - 1)

    def zoom(self) -> int:
        return self._zoom

    def fit_zoom(self) -> None:
        """按视图大小选一个合适的整数倍缩放（新建/载入画布后让画布占满视图）。"""
        host = self._canvas_host
        w = max(1, self._canvas.width)
        h = max(1, self._canvas.height)
        if host.width() < 16 or host.height() < 16:
            return
        z = min((host.width() - 24) // w, (host.height() - 24) // h)
        self._pan_x = 0
        self._pan_y = 0
        self._set_zoom(max(1, min(32, int(z))))
        self._canvas_host.update()

    def _set_zoom(self, zoom: int, focus: Optional[QPoint] = None) -> None:
        """整数倍缩放；focus 给出时以其为焦点（保持焦点处画布格不动）。"""
        zoom = max(1, min(32, int(zoom)))
        if zoom != self._zoom:
            if focus is not None:
                host = self._canvas_host
                fx = (focus.x() - host._ox) / self._zoom
                fy = (focus.y() - host._oy) / self._zoom
                self._zoom = zoom
                cw, ch = self._canvas.width, self._canvas.height
                base_ox = (host.width() - cw * zoom) // 2
                base_oy = (host.height() - ch * zoom) // 2
                self._pan_x = focus.x() - fx * zoom - base_ox
                self._pan_y = focus.y() - fy * zoom - base_oy
            else:
                self._zoom = zoom
        self._zoom_label.setText(f"{self._zoom}x")
        self._canvas_host.update()

    # ------------------------------------------------------------------ #
    def _update_swatch(self) -> None:
        r, g, b, a = self._color
        self._color_swatch.setStyleSheet(
            f"background-color: rgba({r},{g},{b},{a});"
            "border: 1px solid rgba(255,255,255,0.35);"
        )

    def _on_onion_toggled(self, checked: bool) -> None:
        self.set_onion_enabled(checked)

    def _on_palette_lock_toggled(self, checked: bool) -> None:
        self._palette_locked = bool(checked)
        if checked:
            self._canvas.set_palette(self._extract_frame_palette())
        else:
            self._canvas.clear_palette()

    def _on_extract_palette(self) -> None:
        colors = self._extract_frame_palette()
        self._canvas.set_palette(colors)
        self._palette_locked = True
        self._palette_lock_btn.setChecked(True)

    def _extract_frame_palette(self, max_colors: int = 64) -> List[Tuple[int, int, int, int]]:
        """从当前帧提取去重颜色（含透明色），作为锁定调色板。"""
        colors: List[Tuple[int, int, int, int]] = []
        seen = set()
        arr = np.asarray(self._canvas.image).reshape(-1, 4)
        for px in arr:
            t = (int(px[0]), int(px[1]), int(px[2]), int(px[3]))
            if t not in seen and len(colors) < max_colors:
                seen.add(t)
                colors.append(t)
        if (0, 0, 0, 0) not in seen:
            colors.append((0, 0, 0, 0))
        return colors

    def _extract_color_counts(self, max_colors: int = 256) -> List[Tuple[int, Tuple[int, int, int, int]]]:
        """当前帧颜色按出现频率降序，返回 [(count, rgba)]。"""
        arr = np.asarray(self._canvas.image).reshape(-1, 4)
        packed = (
            arr[:, 0].astype(np.int64) << 24
            | arr[:, 1].astype(np.int64) << 16
            | arr[:, 2].astype(np.int64) << 8
            | arr[:, 3].astype(np.int64)
        )
        uniq, counts = np.unique(packed, return_counts=True)
        order = np.argsort(counts)[::-1][:max_colors]
        out = []
        for i in order:
            p = int(uniq[i])
            out.append((int(counts[i]), ((p >> 24) & 255, (p >> 16) & 255, (p >> 8) & 255, p & 255)))
        return out

    # ------------------------------------------------------------------ #
    # 调色板（本图高频「色族」+ 右键整族替换）
    # ------------------------------------------------------------------ #
    def _families(self):
        """当前帧色族列表：[(rep, total, members)] 按总像素降序。"""
        return cluster_color_families(self._extract_color_counts())

    def _refresh_palette(self) -> None:
        """本图色族按频率排序显示前 PALETTE_SHOW 个 + …；左键选色、右键整族替换。"""
        for sw in self._palette_swatches:
            sw.deleteLater()
        self._palette_swatches.clear()
        families = self._families()
        for idx, (rep, total, members) in enumerate(families[:PALETTE_SHOW]):
            sw = self._make_swatch(rep, members)
            self._palette_swatches.append(sw)
            self._palette_row.insertWidget(1 + idx, sw)
        if not families:
            hint = QLabel(tr("（画布为空）"))
            hint.setObjectName("HintLabel")
            self._palette_swatches.append(hint)
            self._palette_row.insertWidget(1, hint)

    def _make_swatch(self, rep: Tuple[int, int, int, int], members: List[Tuple[int, int, int, int]]) -> QToolButton:
        family = (rep, members)
        sw = QToolButton()
        sw.setFixedSize(scaled(22), scaled(22))
        sw.setToolTip(_family_tooltip(color_family_name(rep), rep, members))
        sw.setStyleSheet(_swatch_style(rep))
        sw.clicked.connect(lambda _=False, c=rep: self.set_color(c))
        sw.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        sw.customContextMenuRequested.connect(lambda pos, f=family: self._on_swatch_menu(f, pos))
        return sw

    def _on_swatch_menu(self, family, pos: QPoint) -> None:
        rep, members = family
        menu = QMenu(self)
        act_replace = menu.addAction(tr("替换{0}…").format(tr(color_family_name(rep))))
        act_set = menu.addAction(tr("设为当前颜色"))
        chosen = menu.exec(self.sender().mapToGlobal(pos) if self.sender() else QPoint(0, 0))
        if chosen == act_replace:
            self._on_replace_family(family)
        elif chosen == act_set:
            self.set_color(rep)

    def _on_replace_family(self, family) -> None:
        """右键整族替换：把该色族全部颜色整体换成新色，保留族内相对渐变。"""
        rep, members = family
        if rep[3] == 0:
            QMessageBox.information(self, tr("提示"), tr("透明色族不支持整体替换（可用橡皮擦除）"))
            return
        c = QColorDialog.getColor(QColor(*rep[:3]), self, tr("替换{0}").format(tr(color_family_name(rep))))
        if not c.isValid():
            return
        new_base = (c.red(), c.green(), c.blue(), 255)
        mapping = family_replace_mapping(members, rep, new_base)
        n = self._canvas.replace_colors(mapping)
        if n:
            self._rebuild()
            self.edited.emit()
            self._log_replace(n, color_family_name(rep), rep, new_base, len(members))

    def _log_replace(self, n: int, name: str, old_rep, new_base, n_colors: int) -> None:
        def _hex(rgba):
            return tr("透明") if rgba[3] == 0 else f"#{rgba[0]:02x}{rgba[1]:02x}{rgba[2]:02x}"

        try:
            from ui.pages.ide_page import IdePage

            parent = self.window()
            if isinstance(parent, IdePage):
                parent._log(
                    tr("已替换 {0} 像素：{1}（{2} 色）{3} → {4}").format(n, tr(name), n_colors, _hex(old_rep), _hex(new_base)),
                    "info",
                )
        except Exception:  # noqa: BLE001
            pass

    def _on_palette_dialog(self) -> None:
        families = self._families()
        if not families:
            QMessageBox.information(self, tr("提示"), tr("画布为空"))
            return
        dialog = _PaletteDialog(families, self)
        dialog.color_selected.connect(self.set_color)
        dialog.color_replaced.connect(self._on_replace_family)
        dialog.palette_loaded.connect(self.apply_imported_palette)
        dialog.exec()

    def apply_imported_palette(self, colors) -> None:
        """导入调色板：锁定为这些颜色（绘制/填充自动吸附到最近色）。"""
        colors = [tuple(int(v) for v in c) for c in colors]
        if not colors:
            return
        # 先勾选锁定（其槽函数会先把本图调色板写进画布），再用导入的调色板覆盖
        self._palette_lock_btn.setChecked(True)
        self._palette_locked = True
        self._canvas.set_palette(colors)

    def _rebuild(self) -> None:
        self._qimg = _to_qimage(self._canvas.image)
        self._undo_btn.setEnabled(self._canvas.can_undo)
        self._redo_btn.setEnabled(self._canvas.can_redo)
        self._refresh_palette()
        self._canvas_host.update()

    # ------------------------------------------------------------------ #
    # 坐标换算（widget -> 画布像素）
    # ------------------------------------------------------------------ #
    def _cell_at(self, pos: QPoint) -> Optional[Tuple[int, int]]:
        x = int((pos.x() - self._canvas_host._ox) / self._zoom)
        y = int((pos.y() - self._canvas_host._oy) / self._zoom)
        if 0 <= x < self._canvas.width and 0 <= y < self._canvas.height:
            return x, y
        return None

    def _apply(self, cell: Tuple[int, int], from_cell: Optional[Tuple[int, int]] = None) -> None:
        x, y = cell
        if self._float_layer is not None and self._tool != Tool.SELECT:
            self._merge_float_layer()  # 绘制/填充前先合并浮动图层
        sym, wrap = self._symmetry, self._wrap
        if self._tool == Tool.PENCIL:
            if from_cell:
                self._canvas.draw_line(from_cell, cell, self._color, size=self._brush_size,
                                       symmetry=sym, wrap=wrap)
            else:
                self._canvas.set_pixel(x, y, self._color, size=self._brush_size,
                                       symmetry=sym, wrap=wrap)
        elif self._tool == Tool.ERASER:
            if from_cell:
                self._canvas.draw_line(from_cell, cell, TRANSPARENT, size=self._brush_size,
                                       symmetry=sym, wrap=wrap)
            else:
                self._canvas.set_pixel(x, y, TRANSPARENT, size=self._brush_size,
                                       symmetry=sym, wrap=wrap)
        elif self._tool == Tool.FILL:
            if self._fill_global:
                target = self._canvas.get_pixel(x, y)
                if target[3] > 0:
                    self._canvas.replace_color(target, self._color)
            else:
                self._canvas.flood_fill(x, y, self._color)
        elif self._tool == Tool.EYEDROPPER:
            c = self._canvas.get_pixel(x, y)
            if c[3] > 0:  # 透明像素不取色
                c = self._canvas.snap_color(c)  # 锁定调色板时吸附到最近锁定色
                self.set_color(c)
                self.color_picked.emit(c)
        if self._tool in (Tool.PENCIL, Tool.ERASER, Tool.FILL):
            self._rebuild()
            self.edited.emit()

    # ------------------------------------------------------------------ #
    # 形状工具：拖动预览 -> 松开落笔（直线 / 矩形 / 椭圆）
    # ------------------------------------------------------------------ #
    def _shape_commit(self) -> None:
        """把预览中的形状真正画进画布。"""
        if self._shape_start is None or self._shape_cur is None:
            return
        p0, p1 = self._shape_start, self._shape_cur
        sym, wrap = self._symmetry, self._wrap
        color = TRANSPARENT if self._tool == Tool.ERASER else self._color
        n = 0
        if self._tool == Tool.LINE:
            n = self._canvas.draw_line(p0, p1, color, size=self._brush_size, symmetry=sym, wrap=wrap)
        elif self._tool == Tool.RECT:
            n = self._canvas.draw_rect(p0, p1, color, size=self._brush_size,
                                       filled=self._shape_filled, symmetry=sym, wrap=wrap)
        elif self._tool == Tool.ELLIPSE:
            n = self._canvas.draw_ellipse(p0, p1, color, size=self._brush_size,
                                          filled=self._shape_filled, symmetry=sym, wrap=wrap)
        self._shape_start = None
        self._shape_cur = None
        self._rebuild()
        self._canvas_host.update()
        if n:
            self.edited.emit()

    def _shape_preview_cells(self):
        """预览用的格子集合（与落笔结果一致，含镜像与环绕）。"""
        if self._shape_start is None or self._shape_cur is None:
            return set()
        p0, p1 = self._shape_start, self._shape_cur
        if self._tool == Tool.LINE:
            base = set(self._canvas._line_points(p0, p1))
        elif self._tool == Tool.RECT:
            base = set(self._canvas.rect_cells(p0, p1, self._shape_filled))
        else:
            base = set(self._canvas.ellipse_cells(p0, p1, self._shape_filled))
        cells = set()
        for x, y in base:
            cells.update(self._canvas.mirror_cells(x, y, self._symmetry))
        return set(self._canvas._normalize(cells, self._wrap))

    # ------------------------------------------------------------------ #
    # 选择工具：框选 / 套索 / Ctrl+点选 / 浮动图层（Ctrl+C 复制、Ctrl+M 合并）
    # ------------------------------------------------------------------ #
    def _on_select_left_press(self, event: QMouseEvent) -> None:
        pos = event.position().toPoint()
        cell = self._cell_at(pos)
        if cell is None:
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Ctrl+左键：切换单个像素的选择状态（多点选择）
            if self._selection is None:
                self._selection = np.zeros((self._canvas.height, self._canvas.width), dtype=bool)
            self._selection[cell[1], cell[0]] = not self._selection[cell[1], cell[0]]
            self._rebuild_sel_overlay()
            self._canvas_host.update()
            return
        self._sel_dragging = True
        self._sel_anchor_cell = cell
        self._selection = np.zeros((self._canvas.height, self._canvas.width), dtype=bool)
        if self._sel_mode == "lasso":
            self._sel_lasso_points = [cell]
        else:
            self._sel_rect_cur = cell
            self._update_rect_sel()
        self._canvas_host.update()

    def _on_select_right_press(self, event: QMouseEvent) -> None:
        """Ctrl+右键（任意工具下）：移动浮动图层；有选区时先自动提起为浮动层再移动。"""
        if self._float_layer is not None:
            # 移动浮动图层（粘贴出的图层 / 已提起的选区）
            cell = self._cell_at(event.position().toPoint())
            if cell is None:
                return
            self._moving_float = True
            self._float_grab = (cell[0] - self._float_pos[0], cell[1] - self._float_pos[1])
            return
        if self._selection is not None and self._selection.any():
            # 有选区但未复制：Ctrl+右键直接提起选区内容为浮动层（复制，原图保留）再移动
            self._copy_selection()
            self._paste_layer()
            if self._float_layer is None:
                return
            cell = self._cell_at(event.position().toPoint())
            if cell is None:
                return
            self._moving_float = True
            self._float_grab = (cell[0] - self._float_pos[0], cell[1] - self._float_pos[1])
            return
        # 选择档：无选区/浮动层时，右键拖拽 = 重新框选
        self._on_select_left_press(event)

    def _on_select_move(self, pos: QPoint) -> None:
        cell = self._cell_at(pos)
        if cell is None:
            return
        if self._sel_mode == "lasso":
            last = self._sel_lasso_points[-1]
            if cell != last:
                self._sel_lasso_points.append(cell)
                self._selection = _lasso_mask(self._sel_lasso_points, self._canvas.width, self._canvas.height)
                self._rebuild_sel_overlay()
        else:
            self._sel_rect_cur = cell
            self._update_rect_sel()
        self._canvas_host.update()

    def _on_select_release(self, pos: QPoint) -> None:
        self._sel_dragging = False
        if self._sel_mode == "rect":
            self._update_rect_sel()
        self._canvas_host.update()

    def _update_rect_sel(self) -> None:
        if self._sel_anchor_cell is None or self._sel_rect_cur is None:
            return
        a, b = self._sel_anchor_cell, self._sel_rect_cur
        x0, x1 = sorted((a[0], b[0]))
        y0, y1 = sorted((a[1], b[1]))
        sel = np.zeros((self._canvas.height, self._canvas.width), dtype=bool)
        sel[y0 : y1 + 1, x0 : x1 + 1] = True
        self._selection = sel
        self._rebuild_sel_overlay()

    def _on_float_move(self, pos: QPoint) -> None:
        if not self._moving_float or self._float_layer is None:
            return
        cell = self._cell_at(pos)
        if cell is None:
            return
        fx = cell[0] - self._float_grab[0]
        fy = cell[1] - self._float_grab[1]
        w, h = self._canvas.width, self._canvas.height
        self._float_pos = (
            max(-self._float_layer.width + 1, min(w - 1, fx)),
            max(-self._float_layer.height + 1, min(h - 1, fy)),
        )
        self._canvas_host.update()

    def _copy_selection(self) -> None:
        """Ctrl+C：把选区内容复制到剪贴板（原图与选区保留）。"""
        sel = self._selection
        if sel is None or not sel.any():
            return
        arr = np.asarray(self._canvas.image)
        ys, xs = np.nonzero(sel)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        h, w = sel.shape
        content = np.zeros((h, w, 4), dtype=np.uint8)
        content[sel] = arr[sel]
        self._clipboard = Image.fromarray(content[y0 : y1 + 1, x0 : x1 + 1], "RGBA")
        self._clipboard_pos = (x0, y0)

    def _paste_layer(self) -> None:
        """Ctrl+V：把剪贴板内容粘贴为「半透明」浮动新图层（显示在原选区位置）。

        半透明便于看清新图层与底图的叠放；Ctrl+右键拖拽实时移动，Ctrl+M 合并。
        """
        if self._clipboard is None:
            return
        if self._float_layer is not None:
            self._merge_float_layer()  # 先合并旧浮动层再粘贴新的
        self._float_layer = self._clipboard.copy()
        self._float_pos = self._clipboard_pos
        self._float_qimg = _to_qimage(self._float_layer)
        # 浮动层成为唯一突出对象：清空选区
        self._selection = None
        self._sel_dragging = False
        self._rebuild_sel_overlay()
        self._canvas_host.update()

    def _merge_float_layer(self) -> None:
        """Ctrl+M：把浮动图层合成进主图层并清除选区。"""
        if self._float_layer is None:
            return
        n = self._canvas.paste_image(self._float_layer, *self._float_pos)
        self._float_layer = None
        self._float_pos = (0, 0)
        self._float_qimg = None
        self._selection = None
        self._rebuild_sel_overlay()
        self._rebuild()
        if n:
            self.edited.emit()

    def _clear_selection(self) -> None:
        self._selection = None
        self._sel_dragging = False
        self._float_layer = None
        self._float_pos = (0, 0)
        self._float_qimg = None
        self._moving_float = False
        self._rebuild_sel_overlay()
        self._canvas_host.update()

    def _rebuild_sel_overlay(self) -> None:
        """重建选区：半透明高亮（逐格）+ 蓝色细虚线边框（屏幕空间线段）。

        边框用「格边线段」表示，绘制时以 1px cosmetic 蓝虚线画在屏幕坐标，
        不随原图分辨率放大成粗块，任意缩放都清晰。
        """
        sel = self._selection
        if sel is None or not sel.any():
            self._sel_hl_qimg = None
            self._sel_border_segments = []
            return
        h, w = sel.shape
        hl = np.zeros((h, w, 4), dtype=np.uint8)
        hl[sel] = (80, 160, 255, 55)
        self._sel_hl_qimg = _to_qimage_arr(hl)
        segs: List[Tuple[int, int, int, int]] = []
        for y in range(h):
            for x in range(w):
                if not sel[y, x]:
                    continue
                if y == 0 or not sel[y - 1, x]:
                    segs.append((x, y, x + 1, y))
                if y == h - 1 or not sel[y + 1, x]:
                    segs.append((x, y + 1, x + 1, y + 1))
                if x == 0 or not sel[y, x - 1]:
                    segs.append((x, y, x, y + 1))
                if x == w - 1 or not sel[y, x + 1]:
                    segs.append((x + 1, y, x + 1, y + 1))
        self._sel_border_segments = segs

    # ------------------------------------------------------------------ #
    # 键盘快捷键（绑定可在 设置 → 快捷键 中自定义；像素编辑器固定使用 pixel 键位）
    # ------------------------------------------------------------------ #
    def keyPressEvent(self, event) -> None:  # noqa: N802
        from ui import shortcuts as sc

        # 编辑类
        if sc.match(event, sc.get("undo", "pixel")):
            self.undo()
            event.accept()
            return
        if sc.match(event, sc.get("redo", "pixel")):
            self.redo()
            event.accept()
            return
        if sc.match(event, sc.get("copy", "pixel")):
            self._copy_selection()
            event.accept()
            return
        if sc.match(event, sc.get("paste", "pixel")):
            self._paste_layer()
            event.accept()
            return
        if sc.match(event, sc.get("merge", "pixel")):
            self._merge_float_layer()
            event.accept()
            return
        # 选择类
        if sc.match(event, sc.get("select_all", "pixel")):
            self._select_all()
            event.accept()
            return
        if sc.match(event, sc.get("deselect", "pixel")):
            self._clear_selection()
            event.accept()
            return
        # 视图类
        if sc.match(event, sc.get("zoom_in", "pixel")):
            self._set_zoom(self._zoom + 1)
            event.accept()
            return
        if sc.match(event, sc.get("zoom_out", "pixel")):
            self._set_zoom(self._zoom - 1)
            event.accept()
            return
        # 工具类（仅当用户配置了快捷键时生效）
        for aid, tool in (
            ("tool_pencil", Tool.PENCIL),
            ("tool_eraser", Tool.ERASER),
            ("tool_eyedropper", Tool.EYEDROPPER),
            ("tool_fill", Tool.FILL),
            ("tool_line", Tool.LINE),
            ("tool_rect", Tool.RECT),
            ("tool_ellipse", Tool.ELLIPSE),
            ("tool_select", Tool.SELECT),
        ):
            seq = sc.get(aid, "pixel")
            if seq and sc.match(event, seq):
                self.set_tool(tool)
                event.accept()
                return
        # 对称 / 环绕：无默认快捷键，但用户可在设置里绑定
        if sc.match(event, sc.get("toggle_symmetry", "pixel")):
            self.set_symmetry(SYM_NONE if self._symmetry != SYM_NONE else SYM_H)
            event.accept()
            return
        if sc.match(event, sc.get("toggle_wrap", "pixel")):
            self.set_wrap(not self._wrap)
            event.accept()
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------ #
    def _on_mouse_press(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        cell = self._cell_at(event.position().toPoint())
        if cell is None:
            return
        if self._tool in SHAPE_TOOLS:
            # 形状工具：按下只记起点，拖动预览，松开才落笔
            if self._float_layer is not None:
                self._merge_float_layer()
            self._shape_start = cell
            self._shape_cur = cell
            self._drawing = True
            self._canvas_host.update()
            return
        self._drawing = True
        self._last_cell = cell
        self._apply(cell)

    def _on_mouse_move(self, event: QMouseEvent) -> None:
        if not self._drawing:
            return
        cell = self._cell_at(event.position().toPoint())
        if self._tool in SHAPE_TOOLS:
            if cell is not None and cell != self._shape_cur:
                self._shape_cur = cell
                self._canvas_host.update()
            return
        if cell is None:
            return
        if self._tool in (Tool.EYEDROPPER, Tool.FILL):
            return  # 取色/填充只在按下时生效（拖动不重复触发）
        if cell != self._last_cell:
            self._apply(cell, self._last_cell)
            self._last_cell = cell

    def _on_mouse_release(self, event: QMouseEvent) -> None:
        if self._tool in SHAPE_TOOLS and self._shape_start is not None:
            cell = self._cell_at(event.position().toPoint())
            if cell is not None:
                self._shape_cur = cell
            self._shape_commit()
        self._drawing = False
        self._last_cell = None


class _PaletteDialog(QDialog):
    """完整色族调色板弹窗：网格显示本图全部色族；左键选代表色、右键整族替换（保留渐变）。

    另外支持**导出 / 导入 GIMP .gpl 调色板**（Aseprite、Krita、GIMP 都能互相读写）。
    """

    color_selected = Signal(object)   # 左键选中（传代表色）
    color_replaced = Signal(object)   # 右键替换（传 (rep, members) 色族）
    palette_loaded = Signal(object)   # 导入调色板（传颜色列表）

    def __init__(self, families, parent=None):
        """families: [(rep, total, members)] 色族列表。紧凑网格：只显示色块，族名悬停显示。"""
        super().__init__(parent)
        self._families = list(families)
        self.setWindowTitle(tr("色族调色板"))
        self.setMinimumSize(340, 220)
        layout = QVBoxLayout(self)

        header = QLabel(
            tr("共 {0} 个色族 · 悬停看族名 · 左键选色 · 右键替换整个色族（保留族内渐变）").format(len(families))
        )
        header.setObjectName("HintLabel")
        layout.addWidget(header)

        grid_host = QWidget()
        grid = QGridLayout(grid_host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(2)
        for idx, (rep, total, members) in enumerate(families):
            name = color_family_name(rep)
            sw = QToolButton()
            sw.setFixedSize(24, 24)
            sw.setToolTip(tr("{0} · {1} 色 · {2} 像素").format(tr(name), len(members), total))
            sw.setStyleSheet(_swatch_style(rep))
            sw.clicked.connect(lambda _=False, c=rep: self._on_select(c))
            sw.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            sw.customContextMenuRequested.connect(
                lambda pos, f=(rep, members): self._on_context(f, pos)
            )
            grid.addWidget(sw, idx // 14, idx % 14)
        layout.addWidget(grid_host, 1)

        btn_row = QHBoxLayout()
        custom = QPushButton(tr("自定义颜色…"))
        custom.clicked.connect(self._on_custom)
        btn_row.addWidget(custom)
        btn_export = QPushButton(tr("导出调色板…"))
        T(btn_export, "导出为 GIMP .gpl 调色板（Aseprite / Krita / GIMP 可直接导入）", attr="tooltip")
        btn_export.clicked.connect(self._on_export_palette)
        btn_row.addWidget(btn_export)
        btn_import = QPushButton(tr("导入调色板…"))
        T(btn_import, "导入 .gpl / 纯文本调色板并锁定，之后绘制自动吸附到这些颜色", attr="tooltip")
        btn_import.clicked.connect(self._on_import_palette)
        btn_row.addWidget(btn_import)
        close_btn = QPushButton(tr("关闭"))
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _on_select(self, color) -> None:
        self.color_selected.emit(color)

    def _on_context(self, family, pos: QPoint) -> None:
        rep, _members = family
        menu = QMenu(self)
        act = menu.addAction(tr("替换{0}…").format(tr(color_family_name(rep))))
        if menu.exec(self.sender().mapToGlobal(pos) if self.sender() else QPoint(0, 0)) == act:
            self.color_replaced.emit(family)

    def _on_custom(self) -> None:
        c = QColorDialog.getColor(QColor(0, 0, 0), self, tr("选择颜色"))
        if c.isValid():
            self.color_selected.emit((c.red(), c.green(), c.blue(), 255))

    # ------------------------------------------------------------------ #
    # 调色板导入 / 导出（GIMP .gpl，纯文本 "R G B" 也认）
    # ------------------------------------------------------------------ #
    def palette_colors(self) -> List[Tuple[int, int, int, int]]:
        """当前调色板里的全部颜色（族内成员去重，按出现顺序）。"""
        seen: List[Tuple[int, int, int, int]] = []
        for rep, _total, members in self._families:
            for c in list(members) + [rep]:
                c = tuple(int(v) for v in c)
                if len(c) == 3:
                    c = (c[0], c[1], c[2], 255)
                if c[3] > 0 and c not in seen:
                    seen.append(c)
        return seen

    def _on_export_palette(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        colors = self.palette_colors()
        if not colors:
            return
        path, _f = QFileDialog.getSaveFileName(
            self, tr("导出调色板"), "palette.gpl", tr("GIMP 调色板 (*.gpl);;所有文件 (*)")
        )
        if not path:
            return
        lines = ["GIMP Palette", "Name: PixelFoundry", "Columns: 16", "#"]
        for r, g, b, _a in colors:
            lines.append(f"{r:3d} {g:3d} {b:3d}\t#{r:02X}{g:02X}{b:02X}")
        try:
            Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
            self._status(tr("已导出调色板：{0}（{1} 色）").format(path, len(colors)))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, tr("导出调色板失败"), str(exc))

    def _on_import_palette(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _f = QFileDialog.getOpenFileName(
            self, tr("导入调色板"), "", tr("调色板 (*.gpl *.txt *.pal);;所有文件 (*)")
        )
        if not path:
            return
        try:
            colors = parse_gpl(Path(path).read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, tr("导入调色板失败"), str(exc))
            return
        if not colors:
            QMessageBox.information(self, tr("导入调色板"), tr("这个文件里没有解析到颜色"))
            return
        self.palette_loaded.emit(colors)
        self._status(tr("已导入调色板：{0} 色（已锁定）").format(len(colors)))

    def _status(self, message: str) -> None:
        win = self.window()
        if win is not None and hasattr(win, "statusBar"):
            try:
                win.statusBar().showMessage(message)
            except Exception:  # noqa: BLE001
                pass


class _CanvasView(QWidget):
    """实际渲染像素网格的内层控件（透明棋盘 + 帧 + 网格线 + 缩放）。"""

    def __init__(self, editor: PixelEditorWidget):
        super().__init__()
        self._editor = editor
        self._ox = 0
        self._oy = 0

    # ------------------------------------------------------------------ #
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)

        editor = self._editor
        zoom = editor._zoom
        fw, fh = editor._canvas.width, editor._canvas.height
        # 居中偏移 + 滚轮缩放产生的平移
        ox = (self.width() - fw * zoom) // 2 + editor._pan_x
        oy = (self.height() - fh * zoom) // 2 + editor._pan_y
        self._ox, self._oy = ox, oy

        target = QRect(ox, oy, fw * zoom, fh * zoom)

        # 背景：灰黑网格 / 纯白 / 纯黑 / 纯绿（透明像素露出背景色）
        # 画布「外部」用工作区底色（Krita 的深灰台面），画布「内部」才画棋盘格，
        # 这样画布边界一眼可见；配色随主题（深/浅）自动切换。
        dark = self.palette().color(QPalette.ColorRole.Window).lightness() < 128
        workspace = QColor("#232327") if dark else QColor("#c9ced6")
        check_a = QColor("#3a3a3e") if dark else QColor("#ffffff")
        check_b = QColor("#2c2c2e") if dark else QColor("#dadde3")
        bg = editor._bg_mode
        if bg == "white":
            painter.fillRect(self.rect(), QColor(255, 255, 255))
        elif bg == "black":
            painter.fillRect(self.rect(), QColor(8, 8, 8))
        elif bg == "green":
            painter.fillRect(self.rect(), QColor(0, 255, 0))
        else:  # checker
            painter.fillRect(self.rect(), workspace)
            painter.save()
            painter.setClipRect(target)
            painter.fillRect(target, check_b)
            cell = max(4, 8 * zoom)
            for gy in range((target.height() // cell) + 2):
                for gx in range((target.width() // cell) + 2):
                    if (gx + gy) % 2 == 0:
                        painter.fillRect(ox + gx * cell, oy + gy * cell, cell, cell, check_a)
            painter.restore()

        # 洋葱皮幽灵帧（半透明，绘制在当前帧之下）
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        if editor._onion_enabled:
            if editor._onion_prev_qimg is not None:
                painter.setOpacity(0.35)
                painter.drawImage(target, editor._onion_prev_qimg)
                painter.setOpacity(1.0)
            if editor._onion_next_qimg is not None:
                painter.setOpacity(0.35)
                painter.drawImage(target, editor._onion_next_qimg)
                painter.setOpacity(1.0)

        # 帧图（NEAREST）
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawImage(target, editor._qimg)

        # 浮动图层（Ctrl+V 粘贴的选区内容）：半透明显示，便于看清叠放；+ 蓝色细虚线框
        if editor._float_layer is not None and editor._float_qimg is not None:
            fl = editor._float_layer
            fx, fy = editor._float_pos
            fl_rect = QRect(ox + fx * zoom, oy + fy * zoom, fl.width * zoom, fl.height * zoom)
            painter.setOpacity(editor._float_opacity)
            painter.drawImage(fl_rect, editor._float_qimg)
            painter.setOpacity(1.0)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            pen = QPen(QColor("#3d9bff"), 1)
            pen.setCosmetic(True)  # 1px 屏幕线宽，不随缩放变粗
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(fl_rect.adjusted(-1, -1, 1, 1))

        # 选区高亮（半透明）+ 蓝色细虚线边框（屏幕空间线段，实时绘制）
        if editor._sel_hl_qimg is not None:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            painter.drawImage(QRect(ox, oy, fw * zoom, fh * zoom), editor._sel_hl_qimg)
        if editor._sel_border_segments:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            pen = QPen(QColor("#3d9bff"), 1)
            pen.setCosmetic(True)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            for x0, y0, x1, y1 in editor._sel_border_segments:
                painter.drawLine(
                    QPointF(ox + x0 * zoom, oy + y0 * zoom),
                    QPointF(ox + x1 * zoom, oy + y1 * zoom),
                )

        # 框选区域（右键拖动）：半透明高亮 + 边框
        if editor._rb_active and editor._rb_cell0 is not None and editor._rb_cell1 is not None:
            cx0, cy0 = editor._rb_cell0
            cx1, cy1 = editor._rb_cell1
            rx0, rx1 = sorted((cx0, cx1))
            ry0, ry1 = sorted((cy0, cy1))
            sel = QRect(
                ox + rx0 * zoom, oy + ry0 * zoom,
                (rx1 - rx0 + 1) * zoom, (ry1 - ry0 + 1) * zoom,
            )
            painter.fillRect(sel, QColor(80, 160, 255, 70))
            painter.setPen(QPen(QColor(80, 160, 255), 1))
            painter.drawRect(sel)

        # 形状工具预览（直线/矩形/椭圆的拖动虚影）
        preview = editor._shape_preview_cells() if editor._tool in SHAPE_TOOLS else set()
        if preview:
            col = (0, 0, 0, 0) if editor._tool == Tool.ERASER else editor._color
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(col[0], col[1], col[2], 150))
            for cx, cy in preview:
                painter.drawRect(QRect(ox + cx * zoom, oy + cy * zoom, zoom, zoom))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            pen = QPen(QColor("#3d9bff"), 1)
            pen.setCosmetic(True)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            x0, y0 = editor._shape_start
            x1, y1 = editor._shape_cur
            painter.drawRect(QRect(
                ox + min(x0, x1) * zoom, oy + min(y0, y1) * zoom,
                (abs(x1 - x0) + 1) * zoom, (abs(y1 - y0) + 1) * zoom,
            ))

        # 对称轴（开启对称绘制时显示中轴虚线）
        if editor._symmetry != SYM_NONE:
            pen = QPen(QColor("#ff9d3d"), 1)
            pen.setCosmetic(True)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            if editor._symmetry in (SYM_H, SYM_BOTH):
                cx = ox + (fw / 2.0) * zoom
                painter.drawLine(QPointF(cx, oy), QPointF(cx, oy + fh * zoom))
            if editor._symmetry in (SYM_V, SYM_BOTH):
                cy = oy + (fh / 2.0) * zoom
                painter.drawLine(QPointF(ox, cy), QPointF(ox + fw * zoom, cy))

        # 像素网格（可显示/隐藏；深底用浅线、浅底用深线）
        if editor._grid_visible:
            if bg == "white":
                painter.setPen(QPen(QColor(0, 0, 0, 70)))
            else:
                painter.setPen(QPen(QColor(255, 255, 255, 70)))
            for x in range(fw + 1):
                painter.drawLine(ox + x * zoom, oy, ox + x * zoom, oy + fh * zoom)
            for y in range(fh + 1):
                painter.drawLine(ox, oy + y * zoom, ox + fw * zoom, oy + y * zoom)

        painter.end()

    # ------------------------------------------------------------------ #
    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        """滚轮缩放：以鼠标位置为焦点（焦点处的画布格保持不动）。"""
        delta = event.angleDelta().y()
        if delta == 0:
            return
        step = 1 if delta > 0 else -1
        self._editor._set_zoom(self._editor._zoom + step, focus=event.position().toPoint())
        event.accept()

    # ------------------------------------------------------------------ #
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._editor.setFocus()
        if event.button() == Qt.MouseButton.RightButton:
            e = self._editor
            has_float = e._float_layer is not None
            has_sel = e._selection is not None and e._selection.any()
            if has_float or has_sel or e._tool == Tool.SELECT:
                # 移动浮动层 / 提起选区移动 / 选择档框选
                e._on_select_right_press(event)
            else:
                # 其它工具：右键 = 框选填充 / 取色圆盘
                e._on_right_press(event)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self._editor._tool == Tool.SELECT:
                self._editor._on_select_left_press(event)
                return
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                # Ctrl + 左键拖动 = 平移画布
                self._editor._on_pan_press(event)
                return
        self._editor._on_mouse_press(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        e = self._editor
        pos = event.position().toPoint()
        if e._panning:
            e._on_pan_move(pos)
            return
        if e._rb_active:
            e._on_right_move(pos)
            return
        if e._sel_dragging:
            e._on_select_move(pos)
            return
        if e._moving_float:
            e._on_float_move(pos)
            return
        e._on_mouse_move(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        e = self._editor
        pos = event.position().toPoint()
        if e._panning:
            e._on_pan_release()
            return
        if e._rb_active:
            e._on_right_release(pos, event.globalPosition().toPoint())
            return
        if e._sel_dragging:
            e._on_select_release(pos)
            return
        if e._moving_float:
            e._moving_float = False
            return
        e._on_mouse_release(event)
