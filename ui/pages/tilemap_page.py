"""瓦片地图模式页（第 5 模式）。

分类：
- 地块生态（ground）：2×2 生态图（1 基础 + 3 特征），生成多地形 47 艺术集，
  多地形画笔铺设；
- 建筑类（building）：2×2 建筑图（墙体/顶面/开口/立柱），生成透明拼件，
  像图层一样叠放在地块上（可旋转）；
- 经典 3×3（classic）：保留旧链路。

链路：提示词 → 文生底图 → 裁切 → 无缝化 → 瓦片集/拼件 → 地图预览铺设 → 导出。
"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from PIL import Image

from config.settings import DEFAULT_OUTPUT_DIR
from core.tilemap import TileMapModel
from core.tilemap.pack import (
    export_tileset_dir,
    load_tileset,
    pack_from_session,
    save_tilepack,
)
from core.workflow.tilemap_workflow import TilemapParams, TilemapWorkflow
from core.workflow.solo_workflow import WorkflowError
from ui.app_context import AppContext
from ui.i18n import T, tr
from ui.widgets.tile_editor import TileEditorDialog, base_set_with_edits
from ui.widgets.tilemap_view import TilemapView, pil_to_qpixmap
from ui.workers import TilemapWorker

logger = logging.getLogger("PixelAnimIDE.ui.tilemap_page")

FORM_WIDTH = 380
STYLE_PRESETS = ["game sprite", "retro", "pixel", "top-down RPG", "platformer", "16-bit"]

CATEGORY_LABELS = {
    "ground": "地块生态",
    "building": "建筑类",
    "prop": "素材（道具）",
}

# 基础地形块位置（AI 常不遵守「左上」要求，默认自动识别）
BASE_BLOCK_LABELS = {
    "auto": "自动识别",
    "tl": "左上块",
    "tr": "右上块",
    "bl": "左下块",
    "br": "右下块",
}


class TilemapPage(QWidget):
    """瓦片地图模式页。"""

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self._ctx = ctx
        self._worker: TilemapWorker | None = None
        self._result = None
        self._session = None
        self._params: TilemapParams | None = None
        self._prop_pack = None      # 素材包（跨多次生成累积）
        self._local_wf = TilemapWorkflow(image_api=None)  # 编辑后本地重跑（无需 API）
        self._build_ui()

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(10)

        top = QHBoxLayout()
        top.setSpacing(14)

        # ---------- 左：参数表单 ----------
        left = QWidget()
        lp = QVBoxLayout(left)
        lp.setContentsMargins(0, 0, 0, 0)
        lp.setSpacing(10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setFixedWidth(FORM_WIDTH)
        host = QWidget()
        fl = QVBoxLayout(host)
        fl.setContentsMargins(0, 0, 6, 0)
        fl.setSpacing(10)

        input_box = T(QGroupBox(), "瓦片集参数")
        f = QFormLayout(input_box)
        f.setContentsMargins(12, 18, 12, 12)
        f.setVerticalSpacing(10)

        self._category_combo = QComboBox()
        for key, zh in CATEGORY_LABELS.items():
            self._category_combo.addItem(tr(zh), key)
        self._category_combo.currentIndexChanged.connect(self._on_category_changed)
        f.addRow(T(QLabel(), "瓦片类别"), self._category_combo)

        self._desc_edit = QTextEdit()
        T(self._desc_edit, "例如：草地、石砖墙、熔岩地面、水面……", attr="placeholder")
        self._desc_edit.setMaximumHeight(64)
        f.addRow(T(QLabel(), "纹理描述"), self._desc_edit)

        # 地块生态特征槽（3 个：名称 + 描述）
        self._feature_rows: list = []
        self._feature_labels: list = []
        for i, default in enumerate(
            (("水塘", "a clear pond"), ("稀疏草地", "sparse patchy grass"), ("岩石", "a rocky outcrop"))
        ):
            name_edit = QLineEdit(default[0])
            desc_edit = QLineEdit(default[1])
            row = QHBoxLayout()
            row.addWidget(name_edit, 1)
            row.addWidget(desc_edit, 2)
            label = T(QLabel(), f"特征 {i + 1}")
            f.addRow(label, row)
            self._feature_rows.append((name_edit, desc_edit))
            self._feature_labels.append(label)

        self._base_pos_label = T(QLabel(), "基础地形块")
        self._base_pos_combo = QComboBox()
        for key, zh in BASE_BLOCK_LABELS.items():
            self._base_pos_combo.addItem(T(None, zh), key)
        self._base_pos_combo.setToolTip(
            tr("生图返回的 2×2 底图中哪一块是纯基础地形；默认自动识别（AI 常不遵守位置要求）")
        )
        f.addRow(self._base_pos_label, self._base_pos_combo)

        self._prop_name_label = T(QLabel(), "素材名称")
        self._prop_name_edit = QLineEdit("tree")
        f.addRow(self._prop_name_label, self._prop_name_edit)
        self._prop_count_label = T(QLabel(), "变体数")
        self._prop_count_spin = QSpinBox()
        self._prop_count_spin.setRange(1, 9)
        self._prop_count_spin.setValue(4)
        f.addRow(self._prop_count_label, self._prop_count_spin)

        self._style_combo = QComboBox()
        self._style_combo.setEditable(True)
        self._style_combo.addItems(STYLE_PRESETS)
        self._style_combo.setCurrentText("game sprite")
        f.addRow(T(QLabel(), "风格"), self._style_combo)

        self._tile_spin = QSpinBox()
        self._tile_spin.setRange(8, 128)
        self._tile_spin.setSingleStep(8)
        self._tile_spin.setValue(32)
        f.addRow(T(QLabel(), "瓦片尺寸"), self._tile_spin)

        self._sheet_spin = QSpinBox()
        self._sheet_spin.setRange(256, 1536)
        self._sheet_spin.setSingleStep(128)
        self._sheet_spin.setValue(768)
        f.addRow(T(QLabel(), "生图边长"), self._sheet_spin)

        self._line_label = T(QLabel(), "边界线宽")
        self._line_spin = QSpinBox()
        self._line_spin.setRange(1, 4)
        self._line_spin.setValue(1)
        f.addRow(self._line_label, self._line_spin)

        # 边缘噪声（手绘感）：占瓦片尺寸的百分比，0 = 边界完全平直
        self._noise_label = T(QLabel(), "边缘噪声")
        self._noise_spin = QSpinBox()
        self._noise_spin.setRange(0, 25)
        self._noise_spin.setSingleStep(1)
        self._noise_spin.setValue(9)
        self._noise_spin.setSuffix(" %")
        self._noise_spin.setToolTip(
            tr("非内部瓦片边缘的不规则起伏幅度（占瓦片尺寸百分比；0=平直，越大越像手绘）")
        )
        f.addRow(self._noise_label, self._noise_spin)

        self._blend_label = T(QLabel(), "交界融合")
        self._blend_spin = QSpinBox()
        self._blend_spin.setRange(0, 100)
        self._blend_spin.setSingleStep(10)
        self._blend_spin.setValue(50)
        self._blend_spin.setSuffix(" %")
        self._blend_spin.setToolTip(
            tr("不同地形交界的渗透咬合强度：100% 最自然，0% 为平滑描边硬边")
        )
        f.addRow(self._blend_label, self._blend_spin)

        self._mode_label = T(QLabel(), "瓦片集模式")
        self._mode_combo = QComboBox()
        self._mode_combo.addItem(T(None, "47-tile 瓦片集"), "47")
        self._mode_combo.addItem(T(None, "双网格地图"), "dual")
        self._mode_combo.addItem(T(None, "FrameRonin 47（3×24 布局）"), "blob47")
        self._mode_combo.addItem(T(None, "16 图块族（4×4）"), "tile16")
        f.addRow(self._mode_label, self._mode_combo)

        self._map_src_label = T(QLabel(), "演示地图来源")
        self._map_src_combo = QComboBox()
        self._map_src_combo.addItem(T(None, "展示地形（覆盖 47 类）"), "showcase")
        self._map_src_combo.addItem(T(None, "铺满示例"), "filled")
        f.addRow(self._map_src_label, self._map_src_combo)

        size_row = QHBoxLayout()
        self._map_w_spin = QSpinBox()
        self._map_w_spin.setRange(4, 64)
        self._map_w_spin.setValue(14)
        size_row.addWidget(T(QLabel(), "地图宽"), 1)
        size_row.addWidget(self._map_w_spin, 1)
        self._map_h_spin = QSpinBox()
        self._map_h_spin.setRange(4, 64)
        self._map_h_spin.setValue(10)
        size_row.addWidget(T(QLabel(), "高"), 1)
        size_row.addWidget(self._map_h_spin, 1)
        f.addRow(T(QLabel(), "演示地图"), size_row)

        fl.addWidget(input_box)

        actions = QHBoxLayout()
        self._gen_btn = T(QPushButton(), "生成瓦片集")
        self._gen_btn.setObjectName("PrimaryButton")
        self._gen_btn.clicked.connect(self._on_generate)
        actions.addWidget(self._gen_btn, 1)
        fl.addLayout(actions)

        actions2 = QHBoxLayout()
        self._accept_btn = T(QPushButton(), "接受并生成瓦片集")
        self._accept_btn.setObjectName("PrimaryButton")
        self._accept_btn.clicked.connect(self._on_accept)
        self._accept_btn.setVisible(False)
        actions2.addWidget(self._accept_btn, 1)
        self._edit_btn = T(QPushButton(), "编辑瓦片")
        self._edit_btn.clicked.connect(self._on_edit_tiles)
        self._edit_btn.setEnabled(False)
        self._map_btn = T(QPushButton(), "地图预览")
        self._map_btn.clicked.connect(self._on_map_preview)
        # 预览无需先生成：直接进入后用「添加瓦片包」加载内容
        self._map_btn.setEnabled(True)
        self._map_btn.setToolTip(tr("无需先生成瓦片集：进入后可用「添加瓦片包」加载地块/建筑/素材包"))
        actions2.addWidget(self._edit_btn)
        actions2.addWidget(self._map_btn)
        self._pack_btn = T(QPushButton(), "导出瓦片集")
        self._pack_btn.clicked.connect(self._on_save_pack)
        self._pack_btn.setEnabled(False)
        actions2.addWidget(self._pack_btn)
        fl.addLayout(actions2)

        self._status = QLabel(tr("就绪"))
        self._status.setWordWrap(True)
        fl.addWidget(self._status)
        fl.addStretch(1)
        scroll.setWidget(host)
        lp.addWidget(scroll)
        top.addWidget(left)

        # ---------- 右：预览 ----------
        right = QWidget()
        rp = QVBoxLayout(right)
        rp.setContentsMargins(0, 0, 0, 0)
        rp.setSpacing(8)
        head = QHBoxLayout()
        head.addWidget(T(QLabel(), "瓦片集预览"))
        self._preview_caption = QLabel("")
        head.addWidget(self._preview_caption)
        head.addStretch(1)
        rp.addLayout(head)
        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._preview_label.setMinimumSize(320, 320)
        self._preview_label.setStyleSheet("background: rgba(0,0,0,0.12); border-radius: 8px;")
        rp.addWidget(self._preview_label, 1)
        top.addWidget(right, 1)

        root.addLayout(top, 1)
        self._on_category_changed()

    # ------------------------------------------------------------------ #
    def _on_category_changed(self, *_args) -> None:
        cat = self._category_combo.currentData()
        is_ground = cat == "ground"
        is_building = cat == "building"
        for label in self._feature_labels:
            label.setVisible(is_ground)
        for name_edit, desc_edit in self._feature_rows:
            name_edit.setVisible(is_ground)
            desc_edit.setVisible(is_ground)
        self._line_label.setVisible(cat == "building")
        self._line_spin.setVisible(cat == "building")
        self._base_pos_label.setVisible(is_ground)
        self._base_pos_combo.setVisible(is_ground)
        self._mode_label.setVisible(cat != "building")
        self._mode_combo.setVisible(cat != "building")
        self._map_src_label.setVisible(is_ground)
        self._map_src_combo.setVisible(is_ground)
        is_prop = cat == "prop"
        for widget in (self._prop_name_label, self._prop_name_edit,
                       self._prop_count_label, self._prop_count_spin):
            widget.setVisible(is_prop)
        self._desc_edit.setPlaceholderText(
            tr("例如：草地、沙漠、雪原……（生态基础地形）")
            if is_ground
            else tr("例如：石墙、木栅栏、房屋……（建筑主体）")
            if is_building
            else tr("例如：一棵松树、一丛野花、一块石头……（单个素材描述）")
            if cat == "prop"
            else tr("例如：草地、石砖墙、熔岩地面、水面……")
        )
        # 切换类别：收起「接受」按钮、恢复生成按钮文案
        self._accept_btn.setVisible(False)
        self._gen_btn.setText(tr("生成瓦片集"))

    def _collect_params(self) -> TilemapParams:
        cat = self._category_combo.currentData()
        desc = self._desc_edit.toPlainText().strip()
        if not desc:
            raise WorkflowError(tr("请先填写纹理描述"), step="瓦片提示词")
        features: dict = {}
        if cat == "ground":
            for name_edit, desc_edit in self._feature_rows:
                name = name_edit.text().strip()
                if name:
                    features[name] = desc_edit.text().strip() or name
        return TilemapParams(
            description=desc,
            style=self._style_combo.currentText().strip() or "game sprite",
            category=cat,
            features=features,
            base_block=self._base_pos_combo.currentData() or "auto",
            tile_size=self._tile_spin.value(),
            sheet_size=self._sheet_spin.value(),
            atlas_mode=self._mode_combo.currentData(),
            map_source=self._map_src_combo.currentData() or "showcase",
            prop_name=self._prop_name_edit.text().strip() or "prop",
            prop_variants=self._prop_count_spin.value(),
            line_width=self._line_spin.value(),
            edge_noise=self._noise_spin.value() / 100.0,
            edge_blend=self._blend_spin.value() / 100.0,
            map_width=self._map_w_spin.value(),
            map_height=self._map_h_spin.value(),
            output_dir=Path(DEFAULT_OUTPUT_DIR) / "tilemap",
        )

    # ------------------------------------------------------------------ #
    def _on_generate(self) -> None:
        try:
            params = self._collect_params()
        except WorkflowError as exc:
            self._status.setText(exc.message)
            return
        if self._worker and self._worker.isRunning():
            return
        self._params = params
        self._stage = "base" if params.category != "classic" else "full"
        self._gen_btn.setEnabled(False)
        self._accept_btn.setVisible(False)
        self._edit_btn.setEnabled(False)
        self._map_btn.setEnabled(False)
        self._pack_btn.setEnabled(False)
        self._status.setText(tr("生成中…"))
        self._worker = TilemapWorker(self._ctx.api, params, parent=self, stages=self._stage)
        self._worker.succeeded.connect(self._on_worker_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_worker_done(self, obj) -> None:
        """按阶段分发：base 阶段 = 底图待确认；full = 全部完成。"""
        if self._stage == "base":
            self._on_base_done(obj)
        else:
            self._on_done(obj)

    def _on_base_done(self, session) -> None:
        """底图（2×2 生态图/建筑图）已生成：保留中间结果，展示并等待用户确认。"""
        self._session = session
        self._gen_btn.setEnabled(True)
        self._gen_btn.setText(tr("重新生成"))
        self._accept_btn.setVisible(True)
        self._accept_btn.setEnabled(True)
        self._edit_btn.setEnabled(False)
        self._map_btn.setEnabled(False)
        self._pack_btn.setEnabled(False)
        self._show_sheet(session.sheet_clean or session.sheet_image)
        frames = (session.frame_report or {}).get("count", 0)
        note = tr("（已抹除 {0} 条格线框）").format(frames) if frames else ""
        self._status.setText(
            tr("底图已生成并保存（{0}）{1}。确认满意后点「接受并生成瓦片集」，不满意可「重新生成」").format(
                session.sheet_path, note
            )
        )

    def _on_accept(self) -> None:
        """用户确认底图：本地继续 裁切→无缝→瓦片集→导出（无需 API）。"""
        if self._session is None:
            return
        # 基础块位置在「确认底图」这一刻才真正生效：用户看着底图即可纠正误判，
        # 无需重新生图（裁切/无缝/瓦片集全是本地步骤）。
        self._session.params.base_block = self._base_pos_combo.currentData() or "auto"
        self._accept_btn.setEnabled(False)
        self._gen_btn.setEnabled(False)
        self._status.setText(tr("处理中…"))
        try:
            result = self._local_wf.finish_from_base(self._session)
        except Exception as exc:  # noqa: BLE001
            self._gen_btn.setEnabled(True)
            self._accept_btn.setEnabled(True)
            QMessageBox.warning(self, tr("瓦片集生成失败"), str(exc))
            return
        self._on_done(result)

    def _show_sheet(self, img) -> None:
        """预览「清理后」的生图底图（确认中间结果）。"""
        if img is None:
            return
        zoom = max(1, min(3, 900 // max(img.size)))
        pix = pil_to_qpixmap(img).scaled(
            img.width * zoom, img.height * zoom,
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation,
        )
        self._preview_label.setPixmap(pix)
        self._preview_caption.setText(tr("生图底图（待确认，格线框已抹除）") + f"  ·  {img.size[0]}×{img.size[1]}")

    def _on_done(self, result) -> None:
        self._result = result
        self._session = result.session
        # 素材模式：把本次生成的素材**并入素材包**（可继续生成别的素材，最后导出包）
        if result.category == "prop" and getattr(result.session, "props", None):
            from core.tilemap.pack import TilePack

            if self._prop_pack is None:
                self._prop_pack = TilePack(name=result.session.params.prop_name or "props",
                                           category="prop", tile_size=result.session.params.tile_size)
            self._prop_pack.pieces.update(result.session.props)
        self._gen_btn.setEnabled(True)
        self._gen_btn.setText(tr("生成瓦片集"))
        self._accept_btn.setVisible(False)
        self._edit_btn.setEnabled(True)
        self._map_btn.setEnabled(True)
        self._pack_btn.setEnabled(True)
        self._show_preview()
        self._status.setText(tr("瓦片集已生成: {0}").format(result.output_dir / "export"))

    def _on_failed(self, message: str) -> None:
        self._gen_btn.setEnabled(True)
        self._status.setText(tr("生成失败: {0}").format(message))
        QMessageBox.warning(self, tr("瓦片集生成失败"), message)

    def _show_preview(self) -> None:
        if self._session is None:
            return
        cat = self._session.params.category
        sheet = None
        if cat == "ground":
            first_tid = sorted(self._session.terrain_sheets)[0]
            sheet, _meta = self._session.terrain_sheets[first_tid]
            caption = tr("地块生态（{0} 套地形，显示基础地形 47 集）").format(len(self._session.terrain_sets))
        elif cat == "building":
            pieces = self._session.pieces["pieces"]
            names = list(pieces)
            s = self._session.params.tile_size
            
            sheet = Image.new("RGBA", (s * len(names), s), (0, 0, 0, 0))
            for i, name in enumerate(names):
                sheet.paste(pieces[name], (i * s, 0), pieces[name])
            caption = tr("建筑拼件（{0}）").format(" / ".join(names))
        else:
            sheet = self._session.atlas_sheet
            caption = (
                tr("47-tile（8×6）") if self._session.params.atlas_mode == "47" else tr("双网格（16 块）")
            )
        if cat == "prop" and getattr(self._session, "props", None):
            # 素材模式：把抠底后的素材横排展示（浅色底 + 棋盘格衬托透明区域）
            props = list(self._session.props.items())
            s = self._session.params.tile_size
            sheet = Image.new("RGBA", (max(1, len(props)) * s, s), (0, 0, 0, 0))
            bg = Image.new("RGBA", sheet.size, (236, 240, 246, 255))
            for i in range(0, sheet.width, 8):
                for j in range(0, sheet.height, 8):
                    if (i // 8 + j // 8) % 2 == 0:
                        bg.paste((214, 220, 228, 255), (i, j, min(i + 8, sheet.width), min(j + 8, sheet.height)))
            for i, (_name, prop) in enumerate(props):
                sheet.alpha_composite(prop.convert("RGBA"), (i * s, 0))
            bg.alpha_composite(sheet)
            sheet = bg
            caption = tr("素材（已抠背景，共 {0} 个）").format(len(props))
        if sheet is None:
            return
        zoom = max(1, min(4, 1024 // max(sheet.size)))
        pix = pil_to_qpixmap(sheet).scaled(
            sheet.width * zoom,
            sheet.height * zoom,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self._preview_label.setPixmap(pix)
        self._preview_caption.setText(f"{caption}  ·  {sheet.size[0]}×{sheet.size[1]}")

    # ------------------------------------------------------------------ #
    def _rerun_local_steps(self) -> None:
        if self._session is None:
            return
        try:
            for name in ("seamless", "atlas", "export"):
                self._local_wf.step(name, self._session.params, self._session)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, tr("重新生成失败"), str(exc))
            return
        self._result = self._session.result
        self._show_preview()
        self._status.setText(tr("瓦片已更新并重新生成瓦片集"))

    def _block_base_set(self):
        """返回 (名称列表, {名称: 可编辑 BaseTileSet}) 供编辑对话框使用（按类别）。"""
        session = self._session
        cat = session.params.category
        if cat == "ground":
            names = ["基础地形"] + list(session.ecosystem.features)
            return names, {names[0]: session.ecosystem.base, **session.ecosystem.features}
        if cat == "building":
            names = ["墙体", "顶面", "开口", "立柱"]
            return names, {
                "墙体": session.building.wall,
                "顶面": session.building.top,
                "开口": session.building.opening,
                "立柱": session.building.pillar,
            }
        return ["九宫格"], {"九宫格": session.base}

    def _editor_note(self) -> str:
        """编辑对话框的说明：对齐式构图下只有中心格会成为地形纹理。"""
        cat = self._session.params.category if self._session else "classic"
        if cat == "building":
            return tr("四个组的中心格分别是墙体/顶面/开口/立柱拼件（白底会被抠除）。")
        return tr("47 拼接采用「对齐式构图」：只有**中心格**会作为该地形的无缝纹理，"
                  "条带与描边由算法按实测参数生成（其余 8 格仅供参考/留存）。")

    def _on_edit_tiles(self) -> None:
        if self._session is None:
            QMessageBox.information(self, tr("编辑瓦片"), tr("请先生成瓦片集"))
            return
        names, sets = self._block_base_set()
        name, ok = QInputDialog.getItem(self, tr("编辑瓦片"), tr("选择瓦片组"), names, 0, False)
        if not ok or name not in sets:
            return
        dialog = TileEditorDialog(sets[name], parent=self, note=self._editor_note())
        if dialog.exec() == QDialog.DialogCode.Accepted:
            edited = dialog.result()
            sets[name] = base_set_with_edits(sets[name], edited)
            cat = self._session.params.category
            if cat == "ground":
                if name == names[0]:
                    self._session.ecosystem.base = sets[name]
                else:
                    self._session.ecosystem.features[name] = sets[name]
            elif cat == "building":
                mapping = {"墙体": "wall", "顶面": "top", "开口": "opening", "立柱": "pillar"}
                setattr(self._session.building, mapping[name], sets[name])
            else:
                self._session.base = sets[name]
            self._rerun_local_steps()

    def _terrain_labels(self) -> dict:
        if self._session is None or self._session.ecosystem is None:
            return {}
        labels = {1: tr("基础地形")}
        for i, name in enumerate(self._session.ecosystem.features, start=2):
            labels[i] = name
        return labels

    def _on_save_pack(self) -> None:
        """把当前会话保存成瓦片包（.tilepack），之后可在任意预览里叠加。"""
        if self._session is None:
            QMessageBox.information(self, tr("保存瓦片包"), tr("请先生成瓦片集"))
            return
        if self._prop_pack is not None and self._session.params.category == "prop":
            default = Path(DEFAULT_OUTPUT_DIR) / "tilemap" / f"{self._prop_pack.name}.tilepack"
            path, _f = QFileDialog.getSaveFileName(
                self, tr("保存瓦片包"), str(default), tr("瓦片包 (*.tilepack)")
            )
            if not path:
                return
            try:
                saved = save_tilepack(path, self._prop_pack)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, tr("保存瓦片包失败"), str(exc))
                return
            self._status.setText(tr("瓦片包已保存：{0}（素材 {1} 个）").format(saved, len(self._prop_pack.pieces)))
            return
        default = str(Path(DEFAULT_OUTPUT_DIR) / "tilemap")
        name, ok = QInputDialog.getText(
            self, tr("导出瓦片集"), tr("导出文件夹名"), text=self._session.params.description or "tileset"
        )
        if not ok or not name.strip():
            return
        try:
            saved = export_tileset_dir(Path(default) / name.strip(), pack_from_session(self._session))["dir"]
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, tr("保存瓦片包失败"), str(exc))
            return
        self._status.setText(tr("瓦片集已导出：{0}（含 47 图集/逐张瓦片/元信息，另附 zip）").format(saved))

    def scratch_map_model(self) -> TileMapModel:
        """未生成瓦片集时用的空白预览模型（尺寸/瓦片大小取当前参数）。"""
        return TileMapModel(
            int(self._map_w_spin.value()), int(self._map_h_spin.value()),
            tile_size=int(self._tile_spin.value()),
        )

    def _on_map_preview(self) -> None:
        """打开地图预览：**无需先生成**，可直接进入并用「添加瓦片包」加载素材/地形/建筑包。"""
        session = self._session
        has_session = session is not None and session.map_model is not None
        pieces = session.pieces["pieces"] if has_session and session.pieces else None
        if has_session:
            # 带上拼件注册表：建筑 overlay 按名称恢复（此前从 dict 重建会丢失全部拼件）
            model = TileMapModel.from_dict(session.map_model.to_dict(), pieces=pieces)
            if session.map_model.terrain_sets:
                for tid, tset in session.terrain_sets.items():
                    model.set_terrain(tid, tset)
                model.base_terrain = session.map_model.base_terrain
        else:
            model = self.scratch_map_model()
        dialog = QDialog(self)
        dialog.setWindowTitle(tr("地图预览（左键铺设 / 右键擦除 / Ctrl+左键平移 / 滚轮缩放）"))
        dialog.resize(900, 680)
        layout = QVBoxLayout(dialog)
        center = None
        if has_session and not model.terrain_sets and session.processed is not None:
            center = session.processed.center  # 仅经典单地形程序化渲染需要
        view = TilemapView(
            model,
            center,
            line_color=(0, 0, 0),
            line_width=1,
            atlas_mode="47",
            terrain_labels=self._terrain_labels() if has_session else {},
            pieces=pieces,
        )
        layout.addWidget(view, 1)

        def _add_pack() -> None:
            start = str(Path(DEFAULT_OUTPUT_DIR) / "tilemap")
            path, _f = QFileDialog.getOpenFileName(
                dialog, tr("添加瓦片集（zip / tilepack，或先选文件夹按钮）"), start,
                tr("瓦片集 (*.zip *.tilepack);;所有文件 (*)"),
            )
            if not path:
                path = QFileDialog.getExistingDirectory(dialog, tr("添加瓦片集文件夹"), start)
            if not path:
                return
            try:
                pack = load_tileset(path)
                view.add_pack(pack)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(dialog, tr("加载瓦片包失败"), str(exc))
                return
            self._status.setText(tr("已添加瓦片包：{0}（地形 {1}、拼件 {2}）").format(
                pack.name, len(pack.terrains), len(pack.pieces)))

        def _save_pack() -> None:
            if self._session is None:
                return
            path, _f = QFileDialog.getSaveFileName(
                dialog, tr("保存瓦片包"),
                str(Path(DEFAULT_OUTPUT_DIR) / "tilemap" / f"{session.params.description or 'tilepack'}.tilepack"),
                tr("瓦片包 (*.tilepack)"),
            )
            if not path:
                return
            try:
                saved = save_tilepack(path, pack_from_session(self._session))
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(dialog, tr("保存瓦片包失败"), str(exc))
                return
            self._status.setText(tr("瓦片包已保存：{0}").format(saved))

        def _big_map() -> None:
            from core.tilemap.bigmap import generate_perlin_map, scatter_walls

            dlg = QDialog(dialog)
            dlg.setWindowTitle(tr("柏林噪声大地图"))
            form = QFormLayout(dlg)
            w_spin = QSpinBox(); w_spin.setRange(24, 400); w_spin.setValue(min(220, max(24, model.width * 8)))
            h_spin = QSpinBox(); h_spin.setRange(16, 400); h_spin.setValue(min(160, max(16, model.height * 8)))
            seed_edit = QLineEdit(session.params.description or "pixelgifide")
            sea = QDoubleSpinBox(); sea.setRange(0.1, 0.9); sea.setSingleStep(0.02); sea.setValue(0.38)
            mtn = QDoubleSpinBox(); mtn.setRange(0.1, 0.95); mtn.setSingleStep(0.02); mtn.setValue(0.55)
            walls_check = QCheckBox(tr("散布建筑（需先添加建筑瓦片包）"))
            walls_check.setEnabled(bool(view._pieces))
            form.addRow(tr("宽度（格）"), w_spin)
            form.addRow(tr("高度（格）"), h_spin)
            form.addRow(tr("种子"), seed_edit)
            form.addRow(tr("海平面"), sea)
            form.addRow(tr("山地阈值"), mtn)
            form.addRow("", walls_check)
            btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            btns.accepted.connect(dlg.accept)
            btns.rejected.connect(dlg.reject)
            form.addRow(btns)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            big = TileMapModel(int(w_spin.value()), int(h_spin.value()), tile_size=model.tile_size)
            for tid, tset in view._model.terrain_sets.items():
                big.set_terrain(tid, tset)
            big.base_terrain = view._model.base_terrain or 1
            fids = sorted(t for t in big.terrain_sets if t != (big.base_terrain or 1))
            kinds = generate_perlin_map(
                big, seed=seed_edit.text().strip() or "pixelgifide",
                sea_level=float(sea.value()), mountain_threshold=float(mtn.value()),
                water_id=fids[0] if fids else (big.base_terrain or 1),
                plain_id=big.base_terrain or 1,
                mountain_id=fids[1] if len(fids) > 1 else None,
            )
            if walls_check.isChecked() and view._pieces:
                scatter_walls(big, dict(view._pieces), kinds=kinds, every=11)
            self._session.map_model = big
            self._status.setText(tr("柏林噪声大地图已生成：{0}×{1} 格").format(big.width, big.height))
            dlg2 = QDialog(dialog)
            dlg2.setWindowTitle(tr("柏林噪声大地图预览（Ctrl+左键拖动平移 / 滚轮缩放）"))
            dlg2.resize(1000, 720)
            lay2 = QVBoxLayout(dlg2)
            view2 = TilemapView(big, pieces=dict(view._pieces),
                                terrain_labels=dict(view._terrain_labels))
            lay2.addWidget(view2, 1)
            ok = T(QPushButton(), "应用到会话")
            row2 = QHBoxLayout(); row2.addStretch(1); row2.addWidget(ok)
            lay2.addLayout(row2)
            ok.clicked.connect(dlg2.accept)
            if dlg2.exec() == QDialog.DialogCode.Accepted:
                dialog.accept()
            return

        row = QHBoxLayout()
        pack_btn = T(QPushButton(), "添加瓦片包")
        pack_btn.clicked.connect(_add_pack)
        save_btn = T(QPushButton(), "保存瓦片包")
        save_btn.clicked.connect(_save_pack)
        big_btn = T(QPushButton(), "柏林噪声大地图")
        big_btn.clicked.connect(_big_map)
        row.addWidget(pack_btn)
        row.addWidget(save_btn)
        row.addWidget(big_btn)
        row.addStretch(1)
        close_btn = T(QPushButton(), "应用到会话" if has_session else "关闭")
        close_btn.setEnabled(has_session)
        hint = T(QLabel(), tr("提示：点「添加瓦片包」加载地块/建筑/素材包后即可直接铺设"))
        row.addWidget(hint)
        row.addWidget(close_btn)
        layout.addLayout(row)
        close_btn.clicked.connect(dialog.accept)
        code = dialog.exec()
        if code == QDialog.DialogCode.Accepted and has_session:
            session.map_model = model
            try:
                self._local_wf.step("export", session.params, session)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, tr("导出失败"), str(exc))
                return
            self._status.setText(tr("地图已更新并重新导出预览"))
