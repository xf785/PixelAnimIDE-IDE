"""瓦片包 / 素材包浏览器（像素独立画布左栏用）。

功能：
- 导入：`.tilepack`、导出 zip、导出文件夹（`load_tileset()` 通吃）；
- **切换器**：全部 / 地形 / 建筑 / 素材 / 底图 分段按钮 + 搜索框（清晰快捷地过滤可视化）；
- 包列表带复选框（控制显隐）、点选切换当前包、可移除；
- 缩略图网格：最近邻放大预览，双击即"送入画布"，也支持按钮操作；
- 统计行：包 / 地形 / 拼件 / 素材 / 底图 数量一目了然。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.tilemap.pack import TilePack, load_tileset
from ui.i18n import T, tr

logger = logging.getLogger("PixelFoundry.ui.pack_browser")

#: 切换器分类：(键, 中文标签)
CATEGORIES: Tuple[Tuple[str, str], ...] = (
    ("all", "全部"),
    ("terrain", "地形"),
    ("building", "建筑"),
    ("prop", "素材"),
    ("sheet", "底图"),
)


class PackBrowser(QWidget):
    """包浏览器：加载瓦片包 / 素材包并把里面的资源可视化、可送入画布。"""

    assetChosen = Signal(object, str)          # (PIL.Image, 名字)
    assetReplaceRequested = Signal(object, str)  # 双击/按钮：替换画布
    packChanged = Signal()                     # 包列表变化（增删/清空）-> 便于持久化

    def __init__(self, parent=None):
        super().__init__(parent)
        self._packs: List[TilePack] = []
        self._paths: List[str] = []
        self._assets: List[Tuple[str, str, Image.Image]] = []   # (类别, 名字, 图)
        self._build_ui()
        self._refresh()

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        # 切换器：分段按钮（全部/地形/建筑/素材/底图）
        switcher = QHBoxLayout()
        switcher.setSpacing(4)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        for key, label in CATEGORIES:
            btn = T(QPushButton(), label)
            btn.setCheckable(True)
            btn.setObjectName("SegmentButton")
            btn.setProperty("category", key)
            self._group.addButton(btn)
            switcher.addWidget(btn)
            if key == "all":
                btn.setChecked(True)
        self._group.buttonClicked.connect(lambda _b: self._refresh_assets())
        root.addLayout(switcher)

        self._search = QLineEdit()
        T(self._search, "搜索资源…", attr="placeholder")
        self._search.textChanged.connect(lambda _t: self._refresh_assets())
        root.addWidget(self._search)

        self._pack_list = QListWidget()
        self._pack_list.setObjectName("PackList")
        self._pack_list.setMaximumHeight(96)
        self._pack_list.currentRowChanged.connect(lambda _r: self._refresh_assets())
        root.addWidget(self._pack_list)

        self._grid = QListWidget()
        self._grid.setObjectName("AssetGrid")
        self._grid.setViewMode(QListWidget.ViewMode.IconMode)
        self._grid.setIconSize(QSize(48, 48))
        self._grid.setGridSize(QSize(74, 84))
        self._grid.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._grid.setMovement(QListWidget.Movement.Static)
        self._grid.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._grid.itemDoubleClicked.connect(lambda _i: self._emit(True))
        self._grid.setMinimumHeight(150)
        root.addWidget(self._grid, 1)

        rows = QVBoxLayout()
        rows.setSpacing(4)
        self._btn_add = T(QPushButton(), "添加包…")
        T(self._btn_add, "导入瓦片包/素材包（.tilepack、导出 zip 或导出文件夹）", attr="tooltip")
        self._btn_add.clicked.connect(self.pick_and_add)
        self._btn_put = T(QPushButton(), "放入画布")
        T(self._btn_put, "把选中的资源居中合成到当前画布（保持画布尺寸）", attr="tooltip")
        self._btn_put.clicked.connect(lambda: self._emit(False))
        self._btn_replace = T(QPushButton(), "替换画布")
        T(self._btn_replace, "用选中的资源替换整张画布（画布尺寸随之改变）", attr="tooltip")
        self._btn_replace.clicked.connect(lambda: self._emit(True))
        self._btn_remove = T(QPushButton(), "移除包")
        self._btn_remove.clicked.connect(self.remove_current_pack)
        self._btn_clear = T(QPushButton(), "清空")
        self._btn_clear.clicked.connect(self.clear)
        for b in (self._btn_add, self._btn_put, self._btn_replace):
            rows.addWidget(b)
        row2 = QHBoxLayout()
        row2.addWidget(self._btn_remove)
        row2.addWidget(self._btn_clear)
        rows.addLayout(row2)
        root.addLayout(rows)

        self._stats = QLabel()
        self._stats.setWordWrap(True)
        root.addWidget(self._stats)

    # ------------------------------------------------------------------ #
    # 对外 API
    # ------------------------------------------------------------------ #
    def packs(self) -> List[TilePack]:
        return list(self._packs)

    def pack_paths(self) -> List[str]:
        return list(self._paths)

    def stats(self) -> Dict[str, int]:
        """按**可见包**（勾选状态）统计，与缩略图网格展示的内容保持一致。"""
        packs = self.visible_packs()
        return {
            "packs": len(packs),
            "terrains": sum(len(p.terrains) for p in packs),
            "pieces": sum(len(p.pieces) for p in packs),
            "props": sum(len(p.pieces) for p in packs if p.category == "prop"),
            "sheets": sum(len(p.sheets) for p in packs),
        }

    def add_pack_from_path(self, path) -> Tuple[int, int]:
        """加载一个包（zip / .tilepack / 文件夹），返回 (地形数, 拼件数)。"""
        pack = load_tileset(path)
        self._packs.append(pack)
        self._paths.append(str(path))
        item = QListWidgetItem(f"{pack.name}  ·  {pack.category}")
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        item.setToolTip(str(path))
        self._pack_list.addItem(item)
        if self._pack_list.currentRow() < 0:
            self._pack_list.setCurrentRow(0)
        self._refresh()
        self.packChanged.emit()
        return len(pack.terrains), len(pack.pieces)

    def add_paths(self, paths) -> int:
        """批量恢复（打开项目/上次会话时用），返回成功条数。"""
        ok = 0
        for path in paths or []:
            try:
                self.add_pack_from_path(path)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("瓦片包加载失败 %s: %s", path, exc)
        return ok

    def clear(self) -> None:
        self._packs.clear()
        self._paths.clear()
        self._pack_list.clear()
        self._refresh()
        self.packChanged.emit()

    def remove_current_pack(self) -> None:
        row = self._pack_list.currentRow()
        if row < 0:
            return
        del self._packs[row]
        del self._paths[row]
        self._pack_list.takeItem(row)
        self._refresh()
        self.packChanged.emit()

    def current_asset(self) -> Optional[Tuple[str, Image.Image]]:
        item = self._grid.currentItem()
        if item is None:
            return None
        data = item.data(Qt.ItemDataRole.UserRole)
        if not data:
            return None
        name, img = data
        return name, img

    def visible_packs(self) -> List[TilePack]:
        """按包列表复选框筛出"可见"的包（可用于只看某几个包的资源）。"""
        out: List[TilePack] = []
        for row, pack in enumerate(self._packs):
            item = self._pack_list.item(row)
            if item is None or item.checkState() == Qt.CheckState.Checked:
                out.append(pack)
        return out

    # ------------------------------------------------------------------ #
    def pick_and_add(self) -> None:
        start = str(Path.home())
        path, _f = QFileDialog.getOpenFileName(
            self, tr("添加瓦片集（zip / tilepack，或先选文件夹按钮）"), start,
            tr("瓦片集 (*.zip *.tilepack);;所有文件 (*)"),
        )
        if not path:
            path = QFileDialog.getExistingDirectory(self, tr("添加瓦片集文件夹"), start)
        if not path:
            return
        try:
            terrains, pieces = self.add_pack_from_path(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, tr("加载瓦片包失败"), str(exc))
            return
        logger.info("已加载瓦片包 %s（地形 %d、拼件 %d）", path, terrains, pieces)

    def _emit(self, replace: bool) -> None:
        got = self.current_asset()
        if got is None:
            QMessageBox.information(self, tr("瓦片包"), tr("请先在列表里选择一个资源"))
            return
        name, img = got
        (self.assetReplaceRequested if replace else self.assetChosen).emit(img, name)

    # ------------------------------------------------------------------ #
    def _active_category(self) -> str:
        btn = self._group.checkedButton()
        return str(btn.property("category")) if btn is not None else "all"

    def _refresh(self) -> None:
        packs = self.visible_packs()
        self._assets = []
        for pack in packs:
            for tid, tset in sorted(pack.terrains.items()):
                label = pack.terrain_names.get(tid, f"terrain{tid}")
                self._assets.append(("terrain", f"{pack.name}·{label}", tset.center))
            for name, piece in sorted(pack.pieces.items()):
                kind = "prop" if pack.category == "prop" else "building"
                self._assets.append((kind, f"{pack.name}·{name}", piece))
            for name, sheet in sorted(pack.sheets.items()):
                self._assets.append(("sheet", f"{pack.name}·{name}", sheet))
        self._refresh_assets()
        st = self.stats()
        self._stats.setText(
            tr("包 {0} · 地形 {1} · 拼件 {2} · 素材 {3} · 底图 {4}").format(
                st["packs"], st["terrains"], st["pieces"], st["props"], st["sheets"]
            )
        )

    def _refresh_assets(self) -> None:
        category = self._active_category()
        keyword = self._search.text().strip().lower()
        self._grid.clear()
        for kind, name, img in self._assets:
            if category != "all" and kind != category:
                continue
            if keyword and keyword not in name.lower():
                continue
            item = QListWidgetItem(QIcon(self._pixmap(img)), name.split("·")[-1])
            item.setToolTip(name)
            item.setData(Qt.ItemDataRole.UserRole, (name, img))
            self._grid.addItem(item)
        if self._grid.count() and self._grid.currentRow() < 0:
            self._grid.setCurrentRow(0)

    @staticmethod
    def _pixmap(img: Image.Image, size: int = 48) -> QPixmap:
        rgba = img.convert("RGBA")
        scale = max(1, min(4, size // max(1, max(rgba.size))))
        big = rgba.resize((rgba.width * scale, rgba.height * scale), Image.Resampling.NEAREST)
        from ui.widgets.tilemap_view import pil_to_qpixmap

        return pil_to_qpixmap(big)
