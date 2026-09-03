# 瓦片地图模式（第 5 模式）—— 规划与进度存档

> 状态：**已暂停**（2026-09-03）。核心功能全部完成并通过全量测试（411 项）与打包验证；
> 剩余为可选/验收类工作。恢复时先读本文件与 `docs/tilemap_mode_design.md`。

## 1. 总体进度

| 模块 | 状态 | 说明 |
|---|---|---|
| 严格瓦片集提示词（文生 3×3 底图） | ✅ 完成 | `core/tilemap/prompts.py`（内置严格规范，无需 LLM） |
| 自适应裁切 + 规格归一 | ✅ 完成 | `core/tilemap/tiles.py`（单格=短边/3 取偶，居中裁 3×3） |
| 逐瓦片像素重绘 | ✅ 完成 | `ui/widgets/tile_editor.py`（复用 PixelEditorWidget） |
| 无缝化算法 | ✅ 完成 | `core/tilemap/seamless.py`（纹理全向/墙面轴向/转角推导，构造性零接缝） |
| 47-tile 瓦片集生成 | ✅ 完成 | `core/tilemap/autotile.py`（256 掩码→47 槽映射，含 JSON 元数据） |
| 双网格（可选） | ✅ 完成 | `build_dual_pieces_sheet` + `dual_grid_map` + 2× 渲染 |
| 大网格地图预览与铺设 | ✅ 完成 | `ui/widgets/tilemap_view.py`（左键铺/右键擦/滚轮缩放/网格线） |
| 工作流层 | ✅ 完成 | `core/workflow/tilemap_workflow.py`（六步，自动/手动，编辑后本地重跑） |
| 第 5 模式 UI 与注册 | ✅ 完成 | `ui/pages/tilemap_page.py` + `ui/main_window.py`（侧栏第三行按钮，index 4） |
| i18n | ✅ 完成（基础） | `ui/i18n.py` 新增约 40 条 EN 文案 |
| mock 端到端测试 | ✅ 完成 | `tests/test_tilemap_workflow.py`（6 项）+ `tests/test_tilemap_gui.py`（4 项） |
| 打包验证 | ✅ 完成 | PyInstaller 重建 + 离屏启动冒烟通过 |

## 2. 关键文件清单

- 核心算法：`core/tilemap/{__init__,prompts,tiles,seamless,autotile,map}.py`
- 工作流：`core/workflow/tilemap_workflow.py`、`ui/workers.py`（`TilemapWorker`）
- UI：`ui/pages/tilemap_page.py`、`ui/widgets/tile_editor.py`、
  `ui/widgets/tilemap_view.py`、`ui/icons.py`（`tiles` 图标）、`ui/main_window.py`
- 测试：`tests/test_tilemap_core.py`（14）、`tests/test_tilemap_workflow.py`（6）、
  `tests/test_tilemap_gui.py`（4）；`tests/test_gui.py` 已更新为 5 模式断言
- 文档：`docs/tilemap_mode_design.md`（算法推导/质量保障）、本文件

## 3. 算法要点（恢复时快速回顾）

- 47-tile = 8 邻域位掩码（TL=1…BR=128）按视觉等价归并：16+16+8+2+4+1=47；
  构图规则：空侧描边线；四角「两侧同空（外角）或两侧同满且对角空（内角）」
  切半径 s/2 四分之一圆盘（弧线过边中点，弧描边、弦不描边）；地形取中心纹理。
- 孤立足与「四内角」图像重合：47 槽 / 46 独立图，模板惯例保留双槽。
- 无缝不变量（有测试）：相邻格共享边不透明像素逐像素一致，差异仅允许在
  共享边两端 r+2 像素的转角区（缺口弦为 Godot 同款外观）。
- 纹理无缝 = 顺序镜像平均（先水平后垂直），边缘逐像素相等，重量化回原调色板。

## 4. 恢复后的可选工作（按优先级）

1. **实机人工验收**：真实生图服务商跑一次（看 3×3 底图裁切是否对齐、发灰背景
   是否影响白底规范）；真实素材看无缝效果与线色取样；必要时微调提示词/参数。
2. **手动逐步模式 UI**（可选）：给瓦片页加自动/手动开关，复用 `step()` 与
   `TILEMAP_STEP_LABELS`（工作流层已支持）。
3. **地图模型保存/加载 UI**：`TileMapModel.to_json/from_json` 已就绪，
   可在瓦片页加「保存地图/载入地图」按钮（导出目录 `map_demo.json` 已生成）。
4. **细节打磨**：瓦片页更多 i18n 文案、地图预览对话框「应用到会话」仅在
   Accept 时生效（当前 X 关闭也会应用）、预览图滚动容器、鼠标中键平移。
5. **版本发布**：若验收通过，按 v0.3.0 流程（改 APP_VERSION → 打包 → Release）
   发布第 5 模式。

## 5. 如何恢复

- 目标 ID：`goal-2d93a8e8-dd65-4602-95e9-a3ff36d79c6e`（当前 paused，revision 4）。
- 会话中对我说「继续瓦片地图任务」即可；或直接按第 4 节清单逐项推进。
- 运行验证：`.venv\Scripts\python.exe -m pytest tests -q`；
  打包：`.venv\Scripts\pyinstaller.exe --noconfirm PixelAnimIDE.spec`。
