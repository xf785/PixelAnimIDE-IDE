"""瓦片包 / 素材包浏览器（像素独立画布左栏用）。

功能：
- 导入：`.tilepack`、导出 zip、导出文件夹，**以及没有 manifest 的普通图片文件夹**；
- **逐级目录浏览**：左侧目录树展开包内**每一级目录**（`atlas/`、`textures/`、
  `tiles/terrain_1/`、`source/`、`pieces/`、`props/` …），点任意一级即过滤右侧缩略图，
  于是 47 图集、逐张单瓦片、原始底图、纹理全都能被浏览并送入画布；
- **切换器**：全部 / 地形 / 建筑 / 素材 / 底图 / 图集 / 单瓦片 / 纹理 + 搜索框；
- 包列表带复选框（控制显隐）、点选切换当前包、可移除；
- 缩略图网格：最近邻放大预览，双击即「送入画布」，也支持按钮操作；
- 统计行：包 / 地形 / 拼件 / 素材 / 底图 / 当前目录数量一目了然。

面板拆分：本控件内部把界面拆成两个 :class:`~ui.widgets.dock.Docker`
（「瓦片包」与「资源浏览」），宿主页面可以用 :meth:`PackBrowser.panels`
取走它们放进自己的可拖拽停靠栏。
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
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.tilemap.pack import (
    KIND_ATLAS,
    KIND_DATA,
    KIND_IMAGE,
    KIND_PIECE,
    KIND_PROP,
    KIND_SOURCE,
    KIND_TEXTURE,
    KIND_TILE,
    PackHandle,
    dir_image_count,
    file_tree,
    is_image_path,
    load_pack_handle,
)
from ui.i18n import T, tr
from ui.widgets.dock import Docker, DockerColumn

logger = logging.getLogger("PixelFoundry.ui.pack_browser")

#: 切换器分类：(键, 中文标签)
CATEGORIES: Tuple[Tuple[str, str], ...] = (
    ("all", "全部"),
    ("terrain", "地形"),
    ("building", "建筑"),
    ("prop", "素材"),
    ("sheet", "底图"),
    ("atlas", "图集"),
    ("tile", "单瓦片"),
    ("texture", "纹理"),
)

#: 目录树里图片资源的分类图标
_KIND_ICON = {
    KIND_ATLAS: "layers",
    KIND_TILE: "grid",
    KIND_TEXTURE: "background",
    KIND_SOURCE: "import_image",
    KIND_PIECE: "tiles",
    KIND_PROP: "copy",
    KIND_IMAGE: "import_image",
    KIND_DATA: "export",
}

#: 网格一次最多渲染的条目（超大包只显示前 N 个，避免卡顿）
MAX_GRID_ITEMS = 600

#: 目录树节点数据：("logical"|"files", 包序号, 相对目录, 相对文件)
_NodeData = Tuple[str, int, str, str]


class _Asset:
    """一条可送入画布的资源（逻辑资源直接持有图，目录资源按需解码）。"""

    __slots__ = ("kind", "label", "full", "image", "pack_index", "rel")

    def __init__(self, kind: str, label: str, full: str, image: Optional[Image.Image] = None,
                 pack_index: int = -1, rel: str = ""):
        self.kind = kind
        self.label = label
        self.full = full
        self.image = image
        self.pack_index = pack_index
        self.rel = rel


class PackBrowser(QWidget):
    """包浏览器：加载瓦片包 / 素材包并把里面的资源可视化、可送入画布。"""

    assetChosen = Signal(object, str)          # (PIL.Image, 名字)
    assetReplaceRequested = Signal(object, str)  # 双击/按钮：替换画布
    packChanged = Signal()                     # 包列表变化（增删/清空）-> 便于持久化

    def __init__(self, parent=None, *, standalone: bool = True):
        super().__init__(parent)
        if not standalone:
            # 面板由宿主页面接管：本控件只是个「面板与信号的持有者」，
            # 明确禁止它自己成为顶层窗口（否则可能被当成空白窗口显示出来）。
            self.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            self.hide()
        self._packs: List[PackHandle] = []
        self._paths: List[str] = []
        self._assets: List[_Asset] = []
        self._thumbs: Dict[Tuple[int, str], QPixmap] = {}
        self._raw_cache: Dict[Tuple[int, str], Image.Image] = {}
        # 当前浏览范围：("logical", 包序号或 -1, "", "") / ("files", 包序号, 目录, "")
        self._scope: _NodeData = ("logical", -1, "", "")
        self._build_ui()
        if standalone:
            column = DockerColumn(self)
            root = QVBoxLayout(self)
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(0)
            root.addWidget(column)
            for panel in self._dockers:
                column.add_panel(panel, stretch=1 if panel is self._dockers[-1] else 0)
        self._refresh()

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        # ============ 面板 1：瓦片包（导入 / 显隐 / 移除） ============
        pack_body = QWidget()
        pv = QVBoxLayout(pack_body)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(6)

        self._pack_list = QListWidget()
        self._pack_list.setObjectName("PackList")
        self._pack_list.setMaximumHeight(78)
        T(self._pack_list, "勾选控制显隐；点选切换当前包", attr="tooltip")
        self._pack_list.currentRowChanged.connect(lambda _r: self._refresh())
        pv.addWidget(self._pack_list)

        self._btn_add = T(QPushButton(), "添加包…")
        self._btn_add.setObjectName("PrimaryButton")
        T(self._btn_add, "导入瓦片包/素材包（.tilepack、导出 zip、导出文件夹或普通图片文件夹）", attr="tooltip")
        self._btn_add.clicked.connect(self.pick_and_add)
        pv.addWidget(self._btn_add)

        row2 = QHBoxLayout()
        row2.setSpacing(6)
        self._btn_remove = T(QPushButton(), "移除包")
        self._btn_remove.setObjectName("ToolBtn")
        self._btn_remove.clicked.connect(self.remove_current_pack)
        self._btn_clear = T(QPushButton(), "清空")
        self._btn_clear.setObjectName("ToolBtn")
        self._btn_clear.clicked.connect(self.clear)
        row2.addWidget(self._btn_remove, 1)
        row2.addWidget(self._btn_clear, 1)
        pv.addLayout(row2)

        self._stats = QLabel()
        self._stats.setObjectName("HintLabel")
        self._stats.setWordWrap(True)
        pv.addWidget(self._stats)

        pack_docker = Docker("瓦片包", pack_body, icon_kind="tiles")
        pack_docker.set_icon("tiles")

        # ============ 面板 2：资源浏览（目录树 + 缩略图） ============
        asset_body = QWidget()
        av = QVBoxLayout(asset_body)
        av.setContentsMargins(0, 0, 0, 0)
        av.setSpacing(6)

        # ---- 分类切换器：4 列网格，窄栏也不挤 ----
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        sw = QWidget()
        grid = QGridLayout(sw)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(4)
        for i, (key, label) in enumerate(CATEGORIES):
            btn = T(QPushButton(), label)
            btn.setCheckable(True)
            btn.setObjectName("SegmentButton")
            btn.setProperty("category", key)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._group.addButton(btn)
            grid.addWidget(btn, i // 4, i % 4)
            if key == "all":
                btn.setChecked(True)
        for col in range(4):
            grid.setColumnStretch(col, 1)
        self._group.buttonClicked.connect(lambda _b: self._refresh_assets())
        av.addWidget(sw)

        self._search = QLineEdit()
        self._search.setObjectName("SearchBox")
        T(self._search, "搜索资源…", attr="placeholder")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(lambda _t: self._refresh_assets())
        av.addWidget(self._search)

        # ---- 目录树：包内每一级目录 ----
        self._tree = QTreeWidget()
        self._tree.setObjectName("PackTree")
        self._tree.setHeaderHidden(True)
        self._tree.setUniformRowHeights(True)
        self._tree.setMinimumHeight(120)
        self._tree.setMaximumHeight(220)
        T(self._tree, "包内目录：展开到任意一级即可浏览该层资源", attr="tooltip")
        self._tree.currentItemChanged.connect(lambda cur, _prev: self._on_tree_changed(cur))
        av.addWidget(self._tree)

        # ---- 缩略图网格 ----
        self._grid = QListWidget()
        self._grid.setObjectName("AssetGrid")
        self._grid.setViewMode(QListWidget.ViewMode.IconMode)
        self._grid.setIconSize(QSize(48, 48))
        self._grid.setGridSize(QSize(74, 84))
        self._grid.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._grid.setMovement(QListWidget.Movement.Static)
        self._grid.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._grid.itemDoubleClicked.connect(lambda _i: self._emit(True))
        self._grid.setMinimumHeight(140)
        av.addWidget(self._grid, 1)

        # ---- 操作按钮 ----
        row3 = QHBoxLayout()
        row3.setSpacing(6)
        self._btn_put = T(QPushButton(), "放入画布")
        self._btn_put.setObjectName("PrimaryButton")
        T(self._btn_put, "把选中的资源居中合成到当前画布（保持画布尺寸）", attr="tooltip")
        self._btn_put.clicked.connect(lambda: self._emit(False))
        self._btn_replace = T(QPushButton(), "替换画布")
        self._btn_replace.setObjectName("ToolBtn")
        T(self._btn_replace, "用选中的资源替换整张画布（画布尺寸随之改变）", attr="tooltip")
        self._btn_replace.clicked.connect(lambda: self._emit(True))
        row3.addWidget(self._btn_put, 1)
        row3.addWidget(self._btn_replace, 1)
        av.addLayout(row3)

        self._scope_label = QLabel()
        self._scope_label.setObjectName("HintLabel")
        self._scope_label.setWordWrap(True)
        av.addWidget(self._scope_label)

        asset_docker = Docker("资源浏览", asset_body, icon_kind="grid")
        asset_docker.set_icon("grid")

        self._dockers: List[Docker] = [pack_docker, asset_docker]

    # ------------------------------------------------------------------ #
    # 对外 API
    # ------------------------------------------------------------------ #
    def panels(self) -> List[Docker]:
        """内部两个 docker（宿主可放进自己的停靠栏）。"""
        return list(self._dockers)

    def packs(self) -> List:
        return [h.pack for h in self._packs]

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

    def file_count_total(self) -> int:
        """**可见包**里的图片文件总数（含目录资源，如逐张瓦片与图集）。"""
        return sum(len(h.archive.image_paths()) for h in self._visible_handles())

    def add_pack_from_path(self, path) -> Tuple[int, int]:
        """加载一个包（zip / .tilepack / 文件夹 / 图片文件夹），返回 (地形数, 拼件数)。"""
        handle = load_pack_handle(path)
        self._packs.append(handle)
        self._paths.append(str(path))
        pack = handle.pack
        suffix = tr("（图片文件夹）") if handle.raw else f"  ·  {pack.category}"
        item = QListWidgetItem(f"{handle.name}{suffix}")
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        item.setToolTip(str(path))
        self._pack_list.addItem(item)
        if self._pack_list.currentRow() < 0:
            self._pack_list.setCurrentRow(0)
        self._thumbs.clear()
        self._raw_cache.clear()
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
        for handle in self._packs:
            handle.close()
        self._packs.clear()
        self._paths.clear()
        self._pack_list.clear()
        self._thumbs.clear()
        self._raw_cache.clear()
        self._scope = ("logical", -1, "", "")
        self._refresh()
        self.packChanged.emit()

    def remove_current_pack(self) -> None:
        row = self._pack_list.currentRow()
        if row < 0:
            return
        self._packs[row].close()
        del self._packs[row]
        del self._paths[row]
        self._pack_list.takeItem(row)
        self._thumbs.clear()
        self._raw_cache.clear()
        self._scope = ("logical", -1, "", "")
        self._refresh()
        self.packChanged.emit()

    def current_asset(self) -> Optional[Tuple[str, Image.Image]]:
        return self.asset_of(self._grid.currentItem())

    def asset_of(self, item: Optional[QListWidgetItem]) -> Optional[Tuple[str, Image.Image]]:
        """取某个网格条目的 (完整名, 原图)；目录资源按需从包内解码。"""
        if item is None:
            return None
        asset = item.data(Qt.ItemDataRole.UserRole)
        if asset is None:
            return None
        return asset.full, self._image_for(asset)

    def visible_packs(self) -> List:
        """按包列表复选框筛出"可见"的包（可用于只看某几个包的资源）。"""
        return [h.pack for h in self._visible_handles()]

    def _visible_handles(self) -> List[PackHandle]:
        """按复选框筛出可见包的句柄（含原始目录）。"""
        out: List[PackHandle] = []
        for row, handle in enumerate(self._packs):
            item = self._pack_list.item(row)
            if item is None or item.checkState() == Qt.CheckState.Checked:
                out.append(handle)
        return out

    def scope_text(self) -> str:
        """当前浏览范围的可读名（用于标题/统计）。"""
        kind, pack_index, rel_dir, _rel_file = self._scope
        if kind == "logical":
            if pack_index < 0:
                return tr("全部包 · 逻辑资源")
            return f"{self._name_of(pack_index)} · {tr('逻辑资源')}"
        return f"{self._name_of(pack_index)}/{rel_dir}" if rel_dir else self._name_of(pack_index)

    def retranslate_ui(self) -> None:
        """语言切换后重刷统计行与目录树/范围标签（它们含动态数字与译名）。"""
        try:
            self._refresh()
        except Exception as exc:  # noqa: BLE001
            logger.debug("包浏览器重译失败: %s", exc)

    def file_count(self, pack_index: int, rel_dir: str = "") -> int:
        """某个包（或包内某目录）里的图片总数。"""
        handle = self._handle(pack_index)
        if handle is None:
            return 0
        paths = handle.archive.image_paths()
        if rel_dir:
            prefix = rel_dir.rstrip("/") + "/"
            paths = [p for p in paths if p.startswith(prefix)]
        return len(paths)

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
            QMessageBox.warning(self, tr("加载瓦片包失败"), tr(str(exc)))
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
    # 目录树
    # ------------------------------------------------------------------ #
    def _handle(self, index: int) -> Optional[PackHandle]:
        if 0 <= index < len(self._packs):
            return self._packs[index]
        return None

    def _name_of(self, index: int) -> str:
        handle = self._handle(index)
        return handle.name if handle is not None else ""

    def _build_tree(self) -> None:
        """按可见包重建目录树（保留当前选中范围）。"""
        self._tree.blockSignals(True)
        self._tree.clear()

        root = QTreeWidgetItem([tr("全部包 · 逻辑资源")])
        root.setData(0, Qt.ItemDataRole.UserRole, ("logical", -1, "", ""))
        root.setIcon(0, _icon("layers"))
        self._tree.addTopLevelItem(root)

        visible = {i for i, h in enumerate(self._packs)
                   if (self._pack_list.item(i) is None
                       or self._pack_list.item(i).checkState() == Qt.CheckState.Checked)}

        for index, handle in enumerate(self._packs):
            if index not in visible:
                continue
            pack = handle.pack
            total = len(handle.archive.image_paths())
            tag = tr("（图片文件夹）") if handle.raw else f"· {pack.category}"
            node = QTreeWidgetItem([f"{handle.name}  {tag}  ({total})"])
            node.setData(0, Qt.ItemDataRole.UserRole, ("files", index, "", ""))
            node.setIcon(0, _icon("tiles"))
            self._tree.addTopLevelItem(node)

            logical = QTreeWidgetItem([f"{tr('逻辑资源')}  ({len(pack.terrains) + len(pack.pieces) + len(pack.sheets)})"])
            logical.setData(0, Qt.ItemDataRole.UserRole, ("logical", index, "", ""))
            logical.setIcon(0, _icon("layers"))
            node.addChild(logical)

            tree = file_tree(handle.archive.rel_paths())
            self._fill_tree(node, tree, index, "")

        self._tree.blockSignals(False)
        node = self._find_scope_node() or root
        self._tree.setCurrentItem(node)
        _expand_to(self._tree, node)

    def _iter_tree_nodes(self):
        """深度优先遍历树上所有节点。"""
        stack = [self._tree.topLevelItem(i) for i in range(self._tree.topLevelItemCount())]
        while stack:
            node = stack.pop()
            if node is None:
                continue
            yield node
            stack.extend(node.child(i) for i in range(node.childCount()))

    def _find_scope_node(self) -> Optional[QTreeWidgetItem]:
        """按当前浏览范围（self._scope）找到对应节点（重建后保持选中）。"""
        kind, pack_index, rel_dir, _rel_file = self._scope
        for node in self._iter_tree_nodes():
            data = node.data(0, Qt.ItemDataRole.UserRole)
            if not data:
                continue
            if kind == "logical" and data[0] == "logical" and data[1] == pack_index:
                return node
            if kind == "files" and data[0] == "files" and data[1] == pack_index and data[2] == rel_dir:
                return node
        return None

    def _fill_tree(self, parent: QTreeWidgetItem, tree: Dict[str, object], index: int, rel_dir: str) -> None:
        """递归填充目录树（只列目录；文件在选择目录后于网格中查看）。"""
        for key in sorted(k for k in tree if k != "__files__"):
            value = tree[key]
            if not isinstance(value, dict):
                continue
            rel = f"{rel_dir}/{key}" if rel_dir else key
            count = dir_image_count(value)
            item = QTreeWidgetItem([f"{key}  ({count})" if count else key])
            item.setData(0, Qt.ItemDataRole.UserRole, ("files", index, rel, ""))
            item.setIcon(0, _icon("grid" if count else "export"))
            parent.addChild(item)
            self._fill_tree(item, value, index, rel)

    def _on_tree_changed(self, item: Optional[QTreeWidgetItem]) -> None:
        if item is None:
            return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        self._scope = tuple(data)  # type: ignore[assignment]
        self._apply_scope()

    def _apply_scope(self) -> None:
        """浏览范围变化：重新收集资源并刷新缩略图网格。"""
        self._collect_assets()
        self._refresh_assets()

    def set_scope(self, pack_index: int, rel_dir: str = "") -> None:
        """外部跳转到某个包的某个目录（rel_dir 为空 = 整个包）。"""
        self._scope = ("files", pack_index, rel_dir, "")
        item = self._find_scope_node() if self._tree.topLevelItemCount() else None
        if item is not None:
            self._tree.setCurrentItem(item)
            _expand_to(self._tree, item)
        self._apply_scope()

    # ------------------------------------------------------------------ #
    def _active_category(self) -> str:
        btn = self._group.checkedButton()
        return str(btn.property("category")) if btn is not None else "all"

    def _refresh(self) -> None:
        """重建目录树 + 资源列表 + 统计。"""
        self._build_tree()
        self._collect_assets()
        self._refresh_assets()
        st = self.stats()
        self._stats.setText(
            tr("包 {0} · 地形 {1} · 拼件 {2} · 素材 {3} · 底图 {4} · 图片 {5}").format(
                st["packs"], st["terrains"], st["pieces"], st["props"], st["sheets"],
                self.file_count_total(),
            )
        )

    def _collect_assets(self) -> None:
        """按当前浏览范围收集资源（逻辑资源 / 某个目录下的全部图片）。"""
        scope, pack_index, rel_dir, _rel_file = self._scope
        self._assets = []
        if scope == "logical":
            indices = range(len(self._packs)) if pack_index < 0 else [pack_index]
            for i in indices:
                handle = self._handle(i)
                item = self._pack_list.item(i)
                if handle is None or (item is not None and item.checkState() != Qt.CheckState.Checked):
                    continue
                pack = handle.pack
                for tid, tset in sorted(pack.terrains.items()):
                    label = pack.terrain_names.get(tid, f"terrain{tid}")
                    self._assets.append(_Asset("terrain", label, f"{handle.name}·{label}", tset.center))
                for name, piece in sorted(pack.pieces.items()):
                    kind = "prop" if pack.category == "prop" else "building"
                    self._assets.append(_Asset(kind, name, f"{handle.name}·{name}", piece))
                for name, sheet in sorted(pack.sheets.items()):
                    self._assets.append(_Asset("sheet", name, f"{handle.name}·{name}", sheet))
            return
        handle = self._handle(pack_index)
        if handle is None:
            return
        prefix = rel_dir.rstrip("/") + "/" if rel_dir else ""
        paths = [p for p in handle.archive.rel_paths()
                 if is_image_path(p) and (not prefix or p.startswith(prefix))]
        for rel in paths:
            kind = _kind_of(rel)
            label = rel[len(prefix):] if prefix else rel
            self._assets.append(_Asset(kind, label, f"{handle.name}/{rel}",
                                       pack_index=pack_index, rel=rel))

    def _refresh_assets(self) -> None:
        category = self._active_category()
        keyword = self._search.text().strip().lower()
        self._grid.clear()
        shown = 0
        for asset in self._assets:
            if category != "all" and asset.kind != category:
                continue
            if keyword and keyword not in asset.full.lower():
                continue
            if shown >= MAX_GRID_ITEMS:
                break
            shown += 1
            item = QListWidgetItem(QIcon(self._thumb(asset)), Path(asset.label).stem or asset.label)
            item.setToolTip(f"{asset.full}\n{tr('类别')}: {tr(_CATEGORY_LABEL.get(asset.kind, asset.kind))}")
            item.setData(Qt.ItemDataRole.UserRole, asset)
            self._grid.addItem(item)
        if self._grid.count() and self._grid.currentRow() < 0:
            self._grid.setCurrentRow(0)
        self._update_scope_label(shown)

    def _update_scope_label(self, shown: int) -> None:
        total = len(self._assets)
        text = tr("目录：{0}").format(self.scope_text())
        detail = tr("显示 {0}/{1}").format(shown, total)
        if shown < total:
            detail += tr("（已达到显示上限 {0}，缩小范围或用搜索）").format(MAX_GRID_ITEMS)
        self._scope_label.setText(f"{text}\n{detail}")

    # ------------------------------------------------------------------ #
    # 取图 / 缩略图
    # ------------------------------------------------------------------ #
    def _image_for(self, asset: _Asset) -> Image.Image:
        """取得资源原图（目录资源按需从包内解码，带缓存）。"""
        if asset.image is not None:
            return asset.image
        key = (asset.pack_index, asset.rel)
        cached = self._raw_cache.get(key)
        if cached is not None:
            return cached
        handle = self._handle(asset.pack_index)
        if handle is None:
            return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        img = handle.archive.load_image(asset.rel)
        if len(self._raw_cache) > 512:
            self._raw_cache.clear()
        self._raw_cache[key] = img
        return img

    def _thumb(self, asset: _Asset, size: int = 48) -> QPixmap:
        key = (asset.pack_index, asset.rel) if asset.image is None else (-1, asset.full)
        pm = self._thumbs.get(key)
        if pm is not None:
            return pm
        try:
            img = self._image_for(asset)
        except Exception as exc:  # noqa: BLE001
            logger.warning("缩略图解码失败 %s: %s", asset.full, exc)
            img = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
        pm = self._pixmap(img, size)
        if len(self._thumbs) > 1200:
            self._thumbs.clear()
        self._thumbs[key] = pm
        return pm

    @staticmethod
    def _pixmap(img: Image.Image, size: int = 48) -> QPixmap:
        rgba = img.convert("RGBA")
        scale = max(1, min(4, size // max(1, max(rgba.size))))
        big = rgba.resize((rgba.width * scale, rgba.height * scale), Image.Resampling.NEAREST)
        from ui.widgets.tilemap_view import pil_to_qpixmap

        return pil_to_qpixmap(big)


# --------------------------------------------------------------------------- #
# 分类工具
# --------------------------------------------------------------------------- #
#: 包内路径 -> 切换器分类（terrain/building/prop/sheet 由逻辑资源直接给出）
_CATEGORY_LABEL = {
    "terrain": "地形",
    "building": "建筑",
    "prop": "素材",
    "sheet": "底图",
    KIND_ATLAS: "图集",
    KIND_TILE: "单瓦片",
    KIND_TEXTURE: "纹理",
    KIND_SOURCE: "底图",
    KIND_PIECE: "建筑",
    KIND_PROP: "素材",
    KIND_IMAGE: "图片",
    KIND_DATA: "数据",
}


def _kind_of(rel: str) -> str:
    """目录资源的分类键（吃进切换器的键）。"""
    from core.tilemap.pack import classify_asset

    kind = classify_asset(rel)
    return {
        KIND_SOURCE: "sheet",
        KIND_PIECE: "building",
        KIND_PROP: "prop",
    }.get(kind, kind)


def _icon(kind: str) -> QIcon:
    from ui.icons import editor_icon

    return editor_icon(_KIND_ICON.get(kind, "grid"), "#9aa0a8", size=14)


def _expand_to(tree: QTreeWidget, item: QTreeWidgetItem) -> None:
    """展开到指定节点（把它的所有祖先都展开）。"""
    node = item.parent()
    while node is not None:
        node.setExpanded(True)
        node = node.parent()
