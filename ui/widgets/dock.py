"""Krita 风格停靠面板（docker）基础设施。

参考 Krita 的 docker 体系，提供三件可复用的东西：

- :class:`Docker` —— 单个停靠面板：**标题栏**（小号图标 + 标题 + 右侧附加控件 +
  折叠箭头）+ 内容区。点击标题栏或箭头即可折叠成一条细标题（Krita 的 docker
  可以折起来腾出画布空间）；标题经 ``T()`` 注册，语言切换后自动重译。
- :class:`DockerColumn` —— 一列 docker，内部用 **垂直 QSplitter** 串起来，
  分隔条可拖动，从而「每个面板的高度」都能自己调。
- :class:`SideDock` —— 侧边停靠栏：包住一列 docker，支持**整体收起**（收起后
  变成一条竖排标签，宽度只占 22px），并可在外层 ``QSplitter`` 里**拖动调宽**。

布局原则（与 Krita 一致）：画布永远是主体，docker 尽量窄、可折、可拖；
所有宽度/高度都交给 splitter，而不是写死 ``setFixedWidth``。
"""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QAbstractButton,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSplitter,
    QStackedLayout,
    QStyle,
    QStyleOption,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ui.i18n import T, tr
from ui.layout import scaled

#: 收起后竖排标签的宽度
RAIL_W = 22

#: 侧栏默认/最小宽度（可被 splitter 覆盖）
DOCK_DEFAULT_W = 232
DOCK_MIN_W = 150


class Docker(QWidget):
    """Krita 风格停靠面板：标题头 + 内容区，可折叠。

    用法::

        d = Docker("画布设置", some_widget)
        d.add_header_widget(btn)      # 标题右侧（折叠箭头左边）的附加按钮
        column.add_panel(d, stretch=1)
    """

    toggled = Signal(bool)  # 折叠状态变化（True = 已折叠）

    def __init__(
        self,
        title: str = "",
        content: Optional[QWidget] = None,
        *,
        icon_kind: str = "",
        collapsible: bool = True,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setObjectName("Docker")
        self._collapsed = False
        self._icon_kind = icon_kind
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---------- 标题栏 ----------
        self._header = QFrame()
        self._header.setObjectName("DockerHeader")
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        hh = QHBoxLayout(self._header)
        hh.setContentsMargins(scaled(9), scaled(4), scaled(5), scaled(4))
        hh.setSpacing(scaled(5))

        self._icon_label = QLabel()
        self._icon_label.setObjectName("DockerIcon")
        self._icon_label.setFixedSize(scaled(14), scaled(14))
        self._icon_label.setVisible(bool(icon_kind))
        hh.addWidget(self._icon_label)

        self._title_label = QLabel()
        self._title_label.setObjectName("DockerTitle")
        if title:
            T(self._title_label, title)
        hh.addWidget(self._title_label)
        hh.addStretch(1)

        self._header_widgets: List[QWidget] = []
        self._chevron = QToolButton()
        self._chevron.setObjectName("DockerChevron")
        self._chevron.setAutoRaise(True)
        self._chevron.setCursor(Qt.CursorShape.PointingHandCursor)
        self._chevron.setFixedSize(scaled(18), scaled(18))
        self._chevron.clicked.connect(self.toggle)
        self._chevron.setVisible(collapsible)
        hh.addWidget(self._chevron)

        root.addWidget(self._header)

        # ---------- 内容区 ----------
        self._body = QWidget()
        self._body.setObjectName("DockerBody")
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(scaled(8), scaled(6), scaled(8), scaled(8))
        self._body_layout.setSpacing(scaled(6))
        root.addWidget(self._body, 1)

        self.set_content(content)
        self._set_chevron()
        if icon_kind:
            self.set_icon(icon_kind)
        # 点标题栏也能折叠（子控件如箭头/附加按钮自行处理点击，不会冒泡到这里）
        self._header.mousePressEvent = self._on_header_press  # type: ignore[method-assign]
        self.set_collapsible(collapsible)

    # ------------------------------------------------------------------ #
    def _on_header_press(self, event) -> None:  # noqa: ANN001
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle()
        else:
            QFrame.mousePressEvent(self._header, event)

    # ------------------------------------------------------------------ #
    # 内容
    # ------------------------------------------------------------------ #
    def set_content(self, content: Optional[QWidget]) -> None:
        """设置内容控件（替换旧的）。"""
        old = self._body_layout.itemAt(0).widget() if self._body_layout.count() else None
        if old is not None:
            self._body_layout.removeWidget(old)
            old.setParent(None)
        self._content = content
        if content is not None:
            self._body_layout.addWidget(content, 1)

    def content(self) -> Optional[QWidget]:
        return self._content

    def body(self) -> QWidget:
        """内容区容器（想直接往 body 里塞控件时用）。"""
        return self._body

    # ------------------------------------------------------------------ #
    # 标题栏
    # ------------------------------------------------------------------ #
    def set_title(self, title: str) -> None:
        T(self._title_label, title)

    def title(self) -> str:
        return self._title_label.text()

    def set_icon(self, kind: str, color: str = "#9aa0a8") -> None:
        """设置标题左侧的小图标（editor_icon 的 kind）。"""
        from ui.icons import editor_icon

        self._icon_kind = kind
        self._icon_label.setVisible(bool(kind))
        if kind:
            self._icon_label.setPixmap(editor_icon(kind, color, size=scaled(14)).pixmap(scaled(14), scaled(14)))

    def add_header_widget(self, widget: QWidget) -> QWidget:
        """把控件放到标题右侧（折叠箭头左边），例如计数标签、小按钮。"""
        layout: QHBoxLayout = self._header.layout()  # type: ignore[assignment]
        layout.insertWidget(layout.count() - 1, widget)
        self._header_widgets.append(widget)
        return widget

    def set_collapsible(self, collapsible: bool) -> None:
        self._chevron.setVisible(bool(collapsible))
        self._header.setCursor(
            Qt.CursorShape.PointingHandCursor if collapsible else Qt.CursorShape.ArrowCursor
        )

    # ------------------------------------------------------------------ #
    # 折叠
    # ------------------------------------------------------------------ #
    def is_collapsed(self) -> bool:
        return self._collapsed

    def toggle(self) -> None:
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed: bool) -> None:
        collapsed = bool(collapsed)
        if collapsed == self._collapsed:
            return
        self._collapsed = collapsed
        self._body.setVisible(not collapsed)
        self._header.setProperty("collapsed", collapsed)
        self._header.style().unpolish(self._header)
        self._header.style().polish(self._header)
        self._set_chevron()
        self.toggled.emit(collapsed)

    def _set_chevron(self) -> None:
        self._chevron.setText("▸" if self._collapsed else "▾")
        T(self._chevron, "展开面板" if self._collapsed else "收起面板", attr="tooltip")

    # ------------------------------------------------------------------ #
    def apply_ui_scale(self) -> None:
        """界面比例变化后重排内部固定尺寸。"""
        hh: QHBoxLayout = self._header.layout()  # type: ignore[assignment]
        hh.setContentsMargins(scaled(9), scaled(4), scaled(5), scaled(4))
        hh.setSpacing(scaled(5))
        self._icon_label.setFixedSize(scaled(14), scaled(14))
        self._chevron.setFixedSize(scaled(18), scaled(18))
        self._body_layout.setContentsMargins(scaled(8), scaled(6), scaled(8), scaled(8))
        if self._icon_kind:
            self.set_icon(self._icon_kind)


class DockerColumn(QWidget):
    """一列 docker：内部垂直 QSplitter，面板高度可拖动调整。"""

    def __init__(self, parent: Optional[QWidget] = None, *, spacing: int = 6):
        super().__init__(parent)
        self.setObjectName("DockerColumn")
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.setObjectName("DockerSplitter")
        self._splitter.setChildrenCollapsible(True)   # 折叠后允许缩到只剩标题条
        self._splitter.setHandleWidth(max(3, scaled(spacing)))
        self._splitter.splitterMoved.connect(self._on_handle_moved)
        v.addWidget(self._splitter)
        self._panels: List[Docker] = []
        self._stretch: List[int] = []      # 每个面板的伸缩权重（add_panel 传入）
        self._remembered: dict = {}        # index -> 展开时记住的高度
        self._redistributing = False

    def add_panel(self, panel: Docker, stretch: int = 0) -> Docker:
        """追加一个面板；stretch>0 的面板优先吃掉多余空间。

        折叠某个面板时，它只保留标题条高度，腾出的空间**自动分给其余展开的面板**
        （堆叠填充）；再次展开时按上次的高度还原。
        """
        panel.setParent(self._splitter)
        self._splitter.addWidget(panel)
        self._panels.append(panel)
        self._stretch.append(max(1, int(stretch)) if stretch else 0)
        idx = len(self._panels) - 1
        self._splitter.setStretchFactor(idx, self._effective_stretch(idx))
        panel.toggled.connect(lambda collapsed, i=idx: self._on_panel_toggled(i, collapsed))
        QTimer.singleShot(0, self, self.redistribute)
        return panel

    def add_docker(self, title: str, content: Optional[QWidget] = None, *,
                   icon_kind: str = "", stretch: int = 0, collapsible: bool = True) -> Docker:
        """快捷方式：直接建一个面板并追加。"""
        panel = Docker(title, content, icon_kind=icon_kind, collapsible=collapsible)
        self.add_panel(panel, stretch=stretch)
        return panel

    def panels(self) -> List[Docker]:
        return list(self._panels)

    # ------------------------------------------------------------------ #
    # 堆叠填充：折叠让位 / 展开收回
    # ------------------------------------------------------------------ #
    def _effective_stretch(self, index: int) -> int:
        """折叠的面板不参与伸缩（多余空间全部给展开的面板）；指定 stretch 的面板优先长。"""
        if index >= len(self._panels):
            return 0
        if self._panels[index].is_collapsed():
            return 0
        weight = self._stretch[index]
        return 3 * weight if weight else 1

    def _min_height(self, index: int) -> int:
        panel = self._panels[index]
        try:
            hint = panel.minimumSizeHint().height()
        except RuntimeError:
            hint = scaled(28)
        return max(scaled(24), int(hint))

    def _expanded_hint(self, index: int) -> int:
        """展开面板的「期望高度」：记住的高度 > 内容建议高度。"""
        remembered = self._remembered.get(index)
        if remembered:
            return int(remembered)
        panel = self._panels[index]
        try:
            header = scaled(26)
            body = panel.body().sizeHint().height() if panel.body() is not None else 0
            return max(scaled(72), header + int(body))
        except RuntimeError:
            return scaled(160)

    def _on_panel_toggled(self, index: int, collapsed: bool) -> None:
        if index >= len(self._panels):
            return
        sizes = self._splitter.sizes()
        if collapsed and index < len(sizes):
            self._remembered[index] = sizes[index]      # 收起前记住高度，展开时还原
        self._splitter.setStretchFactor(index, self._effective_stretch(index))
        self.redistribute()

    def _on_handle_moved(self, *_args) -> None:
        """用户手动拖过分隔条：把展开面板的当前高度记为「期望高度」。"""
        sizes = self._splitter.sizes()
        for i, size in enumerate(sizes):
            if i < len(self._panels) and not self._panels[i].is_collapsed():
                self._remembered[i] = int(size)

    def redistribute(self) -> None:
        """按「折叠=标题条、展开=填充剩余」重排所有面板高度。"""
        if self._redistributing or not self._panels:
            return
        sizes = self._splitter.sizes()
        total = sum(sizes)
        if total <= 0:
            total = max(self.height(), scaled(200))
        if total <= 0:
            return
        self._redistributing = True
        try:
            collapsed = [i for i, p in enumerate(self._panels) if p.is_collapsed()]
            expanded = [i for i, p in enumerate(self._panels) if not p.is_collapsed()]
            wants = [0] * len(self._panels)
            for i in collapsed:
                wants[i] = self._min_height(i)          # 只剩标题条
            budget = total - sum(wants)
            if expanded:
                if budget <= 0:
                    budget = total
                    wants = [0] * len(self._panels)
                # 先给每个展开面板它想要的高度
                desired = {i: self._expanded_hint(i) for i in expanded}
                desired_sum = sum(desired.values())
                if desired_sum <= budget:
                    # 还有富余：按权重（可伸缩面板权重更高）分掉
                    for i in expanded:
                        wants[i] = desired[i]
                    extra = budget - desired_sum
                    weights = [max(1, self._effective_stretch(i)) for i in expanded]
                    weight_sum = sum(weights)
                    given = 0
                    for pos, i in enumerate(expanded):
                        add = extra * weights[pos] // weight_sum
                        wants[i] += add
                        given += add
                    wants[expanded[-1]] += extra - given
                else:
                    # 空间不够：按期望高度等比压缩，但不低于最小高度
                    scale = budget / float(desired_sum)
                    mins = {}
                    for i in expanded:
                        mins[i] = self._min_height(i)
                        wants[i] = max(mins[i], int(desired[i] * scale))
                    over = sum(wants) - total
                    # 仍然超了就继续从「还有余量」的面板里扣
                    guard = 0
                    while over > 0 and guard < 40:
                        guard += 1
                        room = [i for i in expanded if wants[i] > mins[i]]
                        if not room:
                            break
                        step = max(1, over // len(room))
                        for i in room:
                            cut = min(step, wants[i] - mins[i])
                            wants[i] -= cut
                            over -= cut
                            if over <= 0:
                                break
            self._splitter.setSizes(wants)
        finally:
            self._redistributing = False

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        # 显示时统一重排一次：构造期 splitter 还没有真实高度，setSizes 会被缩放
        self.redistribute()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        # 高度变化后保持「折叠的面板只占标题条」的不变量
        if self._redistributing or not self._panels:
            return
        sizes = self._splitter.sizes()
        if not sizes:
            return
        collapsed = [i for i, p in enumerate(self._panels) if p.is_collapsed()]
        if not collapsed:
            return
        for i in collapsed:
            if i < len(sizes) and sizes[i] > self._min_height(i) + 2:
                self.redistribute()
                return

    def apply_ui_scale(self) -> None:
        self._splitter.setHandleWidth(max(3, scaled(6)))
        for p in self._panels:
            p.apply_ui_scale()
        self.redistribute()


class _RailButton(QAbstractButton):
    """收起状态的竖排标签（文字旋转 90°，Krita 折叠 docker 的观感）。"""

    def __init__(self, text: str, arrow: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("DockRail")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedWidth(scaled(RAIL_W))
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self._arrow = arrow
        self._text = text
        self.setToolTip(text)

    def set_rail_text(self, text: str) -> None:
        self._text = text
        self.setToolTip(text)
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(scaled(RAIL_W), scaled(120))

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        opt = QStyleOption()
        opt.initFrom(self)
        self.style().drawPrimitive(QStyle.PrimitiveElement.PE_Widget, opt, p, self)
        col = self.palette().color(self.foregroundRole())
        p.setPen(col)
        f = self.font()
        f.setPointSizeF(max(7.5, f.pointSizeF() - 0.5))
        p.setFont(f)
        p.translate(self.width(), 0)
        p.rotate(90)
        rect = self.rect()
        rect.setWidth(self.height())
        p.drawText(rect.adjusted(0, 0, -self.width(), 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   f"{self._arrow}  {self._text}")
        p.end()


class SideDock(QWidget):
    """侧边停靠栏：一列 docker + 整体收起/展开 + 由外层 splitter 控制宽度。

    - ``set_collapsed(True)`` -> 收起成一条 22px 竖排标签（点它再展开）；
    - ``bind_splitter(splitter, index)`` -> 记住/恢复在 splitter 中的宽度；
    - ``default_width`` 为首次展开时的宽度。
    """

    collapsedChanged = Signal(bool)

    def __init__(
        self,
        title: str = "",
        *,
        side: str = "left",
        default_width: int = DOCK_DEFAULT_W,
        min_width: int = DOCK_MIN_W,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setObjectName("SideDock")
        self._title = title
        self._side = "right" if side == "right" else "left"
        self._default_width = scaled(default_width)
        self._min_width = scaled(min_width)
        self._collapsed = False
        self._saved_width = self._default_width
        self._width_applied = False
        self._splitter: Optional[QSplitter] = None
        self._splitter_index = 0

        self._stack = QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)

        self._column = DockerColumn(self)
        self._stack.addWidget(self._column)

        arrow = "▶" if self._side == "left" else "◀"
        self._rail = _RailButton(tr(title), arrow, self)
        self._rail.clicked.connect(lambda: self.set_collapsed(False))
        self._stack.addWidget(self._rail)
        self._stack.setCurrentWidget(self._column)

        self.setMinimumWidth(0)
        self.setMaximumWidth(16777215)

    # ------------------------------------------------------------------ #
    def column(self) -> DockerColumn:
        return self._column

    def add_panel(self, panel: Docker, stretch: int = 0) -> Docker:
        return self._column.add_panel(panel, stretch=stretch)

    def add_docker(self, title: str, content: Optional[QWidget] = None, *,
                   icon_kind: str = "", stretch: int = 0, collapsible: bool = True) -> Docker:
        return self._column.add_docker(title, content, icon_kind=icon_kind,
                                       stretch=stretch, collapsible=collapsible)

    def panels(self) -> List[Docker]:
        return self._column.panels()

    def set_title(self, title: str) -> None:
        self._title = title
        self._rail.set_rail_text(tr(title))

    def bind_splitter(self, splitter: QSplitter, index: int, *, default_width: Optional[int] = None) -> None:
        """绑定外层横向 splitter，用于记忆/恢复宽度。"""
        self._splitter = splitter
        self._splitter_index = index
        if default_width:
            self._default_width = scaled(default_width)
            self._saved_width = self._default_width

    def default_width(self) -> int:
        return self._default_width

    # ------------------------------------------------------------------ #
    def is_collapsed(self) -> bool:
        return self._collapsed

    def toggle(self) -> None:
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed: bool) -> None:
        collapsed = bool(collapsed)
        if collapsed == self._collapsed:
            return
        self._collapsed = collapsed
        if collapsed:
            self._remember_width()
            self._stack.setCurrentWidget(self._rail)
            self.setFixedWidth(scaled(RAIL_W))
            self._rebalance()
        else:
            self._stack.setCurrentWidget(self._column)
            self.setMinimumWidth(0)
            self.setMaximumWidth(16777215)
            self._apply_width(self._saved_width or self._default_width)
        # 页面常在构造期/恢复期随后才调用 splitter.setSizes(...)，那一手会把收起栏的槽位又撑开；
        # 因此布局落定（事件循环下一拍）后再收一次，保证收起 = 真的让出宽度。
        QTimer.singleShot(0, self, self._settle)
        self.collapsedChanged.emit(collapsed)

    def _settle(self) -> None:
        """布局落定后校正：收起时让出槽位，展开时恢复记忆宽度。"""
        if self._collapsed:
            self._rebalance()
        else:
            self._apply_width(self._saved_width or self._default_width)

    def showEvent(self, event) -> None:  # noqa: N802
        """首次显示后校正宽度。

        构造期 ``splitter.setSizes()`` 时 splitter 还没有真实宽度，Qt 会等比压缩；
        等真正显示出来后再按「默认/上次」宽度摆正一次，避免侧栏被压得过窄。
        """
        super().showEvent(event)
        if not self._width_applied:
            self._width_applied = True
            QTimer.singleShot(0, self, self._settle)

    def _neighbor_index(self, sizes: List[int]) -> Optional[int]:
        """挑一个「邻居」来吸收宽度差：优先画布（非停靠栏），其次最大的那一栏。

        不能简单地取「最大的一栏」——左右两个停靠栏会互相补偿、来回抢宽度，
        最终把中间的画布挤成一条缝。
        """
        others = [i for i in range(len(sizes)) if i != self._splitter_index]
        if not others:
            return None
        plain = [i for i in others if not isinstance(self._splitter.widget(i), SideDock)]
        pool = plain or others
        return max(pool, key=lambda i: sizes[i])

    def _rebalance(self) -> None:
        """收起后让 splitter 把腾出的空间还给相邻栏（否则会留一条空白）。"""
        if self._splitter is None:
            return
        sizes = self._splitter.sizes()
        if not (0 <= self._splitter_index < len(sizes)):
            return
        rail = scaled(RAIL_W)
        delta = sizes[self._splitter_index] - rail
        if delta <= 0:
            return
        sizes[self._splitter_index] = rail
        target = self._neighbor_index(sizes)
        if target is not None:
            sizes[target] = max(scaled(120), sizes[target] + delta)
        self._splitter.setSizes(sizes)

    def _remember_width(self) -> None:
        w = self.width()
        if w > scaled(RAIL_W) + 4:
            self._saved_width = w

    def _apply_width(self, width: int) -> None:
        """在 splitter 中恢复宽度（没有 splitter 时用固定宽）。

        宽度差一律由**画布**（非停靠栏那一栏）吸收 —— 左右两个停靠栏若互相补偿会来回抢宽度。
        """
        width = max(self._min_width, int(width))
        if self._splitter is not None:
            sizes = self._splitter.sizes()
            if self._splitter_index < len(sizes):
                delta = width - sizes[self._splitter_index]
                if delta == 0:
                    return
                sizes[self._splitter_index] = width
                target = self._neighbor_index(sizes)
                if target is not None:
                    sizes[target] = max(scaled(120), sizes[target] - delta)
                self._splitter.setSizes(sizes)
                return
        self.setMinimumWidth(width)
        self.setMaximumWidth(width)

    # ------------------------------------------------------------------ #
    def add_toggle_button(self, icon_kind: str, tooltip: str) -> QToolButton:
        """生成一个「收起本栏」按钮（放在面板标题里或工具栏上）。"""
        from ui.icons import editor_icon

        btn = QToolButton()
        btn.setObjectName("DockToggle")
        btn.setAutoRaise(True)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setIcon(editor_icon(icon_kind, "#9aa0a8", size=scaled(14)))
        btn.setIconSize(QSize(scaled(14), scaled(14)))
        btn.setFixedSize(scaled(20), scaled(20))
        T(btn, tooltip, attr="tooltip")
        btn.clicked.connect(self.toggle)
        return btn

    def apply_ui_scale(self) -> None:
        self._column.apply_ui_scale()
        self._rail.setFixedWidth(scaled(RAIL_W))
        if self._collapsed:
            self.setFixedWidth(scaled(RAIL_W))
