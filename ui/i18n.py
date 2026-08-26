"""中英文适配：轻量 i18n（ID 序列 + 语言包解耦架构）。

设计：
- 所有界面文案以「中文原文」为稳定 ID（即文案序列本身）；
- 每个语言是一个 {id: text} 语言包（LANG_PACKS / ui/lang/xx.py）；
- tr(id) 按当前语言取词：中文直接返回 ID（默认包为空），
  其它语言查对应语言包，未收录回退中文（便于增量迁移）；
- 新增语言：复制任意语言包按序翻译，注册到 LANG_PACKS，
  并（可选）放入 ui/lang/xx.py 让 _load_pack 自动加载；设置页语言下拉
  自动读取 available_languages()。

用法：
    from ui.i18n import tr
    btn.setText(tr("开始生成"))
    btn = T(QPushButton(), "开始生成")   # 注册后可立即重译
"""
from __future__ import annotations

_LANG: str = "zh"  # 当前语言（LANG_PACKS 的 key）

# 语言包注册表：{语言代码: {id: text}}。zh 为默认（ID 即原文，包可为空）。
LANG_PACKS: dict = {
    "zh": {},
    "en": {
    # ---------- 主窗口 / 模式 ----------
    "Solo": "Solo",
    "IDE": "IDE",
    "精灵图": "Sprites",
    "像素": "Pixel",
    "Solo — 一键生成": "Solo — One-click generation",
    "IDE — 分步工作区": "IDE — Step-by-step workspace",
    "精灵图 — 文生图网格精灵图": "Sprites — grid sprite sheets from text",
    "像素 — 独立像素画布": "Pixel — standalone pixel canvas",
    "Solo 模式 — 一键生成": "Solo mode — one-click generation",
    "IDE 模式 — 分步工作区": "IDE mode — step-by-step workspace",
    "精灵图模式 — 文生图网格精灵图": "Sprite mode — grid sprite sheets from text",
    "像素模式 — 独立像素画布": "Pixel mode — standalone pixel canvas",
    "设置 — API 配置 / 常规": "Settings — API config / general",
    "切换主题（深色 / 浅色）": "Toggle theme (dark / light)",
    "就绪": "Ready",
    "已同步 Solo 结果到 IDE": "Solo result synced to IDE",
    "已同步精灵图结果到 IDE": "Sprite result synced to IDE",
    "已从 IDE 同步当前帧到像素画布": "Current frame synced from IDE to pixel canvas",
    "IDE 暂无帧可同步，请先在 IDE 生成或导入图片": "No IDE frame to sync yet — generate or import an image in IDE first",
    "已同步像素画布到 IDE（首帧 + 图生图参考）": "Pixel canvas synced to IDE (first frame + i2i reference)",
    "已设置图生视频首帧（Solo），点「开始生成」即可": "Video first frame set (Solo) — click Start to generate",
    "文本": "Text",
    "图片": "Image",
    "动画": "Animation",
    "像素化": "Pixelate",
    "背景": "Background",
    "导出": "Export",
    "文本生成": "Text generation",
    "图片生成": "Image generation",
    "动画生成": "Animation generation",
    "像素化处理": "Pixelate",
    "背景去除": "Background removal",
    "生成提示词": "Generate prompts",
    "生成首帧图片": "Generate first frame",
    "生成动画": "Generate animation",
    "去除背景": "Remove background",
    "新建": "New",
    "打开": "Open",
    "保存": "Save",
    "预览": "Preview",
    "编辑": "Edit",
    "提示词": "Prompts",
    "项目": "Project",
    "参考图 / 首帧图": "Reference / first frame",
    "点击添加自备参考图": "Click to add your own reference image",
    "日志": "Log",
    "帧序列": "Frames",
    "播放": "Play",
    "暂停": "Pause",
    "适应": "Fit",
    "缩小预览": "Zoom out",
    "放大预览": "Zoom in",
    "重置为适应窗口": "Reset to fit window",
    "文字描述": "Description",
    "文本描述": "Description",
    "动作类型(可选)": "Action (optional)",
    "动作(可选)": "Action (optional)",
    "宽高比": "Aspect ratio",
    "像素尺寸": "Pixel size",
    "单格尺寸": "Cell size",
    "最大颜色数": "Max colors",
    "帧数": "Frames",
    "帧率(fps)": "FPS",
    "播放速度": "Playback speed",
    "首尾帧一致（循环闭合）": "Loop close (first = last)",
    "背景强制纯色": "Force solid background",
    "背景容差": "BG tolerance",
    "内缩(px)": "Shrink (px)",
    "羽化(px)": "Feather (px)",
    "输出目录": "Output dir",
    "打开输出目录": "Open output dir",
    "完美像素化": "Perfect pixelate",
    "去除背景": "Remove background",
    "浏览…": "Browse…",
    "搜索": "Search",
    "查看完整色族调色板": "View full color-family palette",
    "收起调色板": "Collapse palette",
    "显示/隐藏调色板": "Show/hide palette",
    "从本地导入图片（替换当前帧）": "Import image from disk (replaces frame)",
    "导出当前帧为 PNG": "Export current frame as PNG",
    "从本地导入图片": "Import image from disk",
    "当前颜色": "Current color",
    # ---------- 像素编辑器 ----------
    "铅笔": "Pencil",
    "橡皮": "Eraser",
    "取色": "Eyedropper",
    "填充": "Fill",
    "选择": "Select",
    "铅笔（右键笔刷大小）": "Pencil (right-click: brush size)",
    "橡皮（右键笔刷大小）": "Eraser (right-click: brush size)",
    "填充（右键填充方式）": "Fill (right-click: fill mode)",
    "取色": "Eyedropper",
    "选择（右键框选/套索；Ctrl+左键加点；Ctrl+C 复制，Ctrl+V 粘贴半透明新图层；Ctrl+右键拖拽移动；Ctrl+M 合并）": "Select (right-click: rect/lasso; Ctrl+click: add pixels; Ctrl+C copy, Ctrl+V paste semi-transparent layer; Ctrl+right-drag move; Ctrl+M merge)",
    "撤销（Ctrl+Z）": "Undo (Ctrl+Z)",
    "重做（Ctrl+Shift+Z）": "Redo (Ctrl+Shift+Z)",
    "缩小": "Zoom out",
    "放大": "Zoom in",
    "洋葱皮：显示相邻帧半透明幽灵": "Onion skin: show adjacent frames as ghosts",
    "锁定调色板：绘制/填充吸附到当前帧调色板": "Lock palette: snap drawing/fill to frame palette",
    "提取调色板并锁定": "Extract palette and lock",
    "显示/隐藏像素网格": "Show/hide pixel grid",
    "背景：灰黑网格（点击切换，右键选档）": "Background: checker (click to cycle, right-click to pick)",
    "收起/展开控制面板": "Collapse/expand controls",
    "展开控制面板": "Expand controls",
    "收起控制面板": "Collapse controls",
    "笔刷大小": "Brush size",
    "选择方式": "Selection mode",
    "矩形框选": "Rectangle select",
    "套索选择": "Lasso select",
    "全部选择": "Select all",
    "取消选择": "Deselect",
    "复制 (Ctrl+C)": "Copy (Ctrl+C)",
    "粘贴为新图层 (Ctrl+V)": "Paste as new layer (Ctrl+V)",
    "合并图层 (Ctrl+M)": "Merge layer (Ctrl+M)",
    "填充方式": "Fill mode",
    "连通区域填充": "Flood fill",
    "全局同色替换": "Replace all same color",
    "（画布为空）": "(empty canvas)",
    "查看完整色族调色板": "View full color-family palette",
    "色族调色板": "Color-family palette",
    "当前颜色": "Current color",
    "替换此颜色…": "Replace this color…",
    "设为当前颜色": "Set as current color",
    "自定义颜色…": "Custom color…",
    "关闭": "Close",
    "透明色不支持整体替换（可用橡皮擦除）": "Transparent cannot be replaced (use eraser)",
    "画布为空": "Canvas is empty",
    "已替换 {n} 像素：": "Replaced {n} pixels: ",
    "（左键选色，右键替换色族）": "(left: pick, right: replace family)",
    "提示": "Notice",
    "导入图片": "Import image",
    "导入失败": "Import failed",
    "无法读取图片：": "Cannot read image: ",
    "导出失败": "Export failed",
    "保存失败：": "Save failed: ",
    "导出当前帧为 PNG": "Export current frame as PNG",
    # ---------- Solo 页 ----------
    "开始生成": "Start generating",
    "取消": "Cancel",
    "同步到 IDE": "Sync to IDE",
    "参数": "Parameters",
    "中间结果": "Progress",
    "参考图": "Reference",
    "一键抠图（扣除纯色背景）": "Key out background (solid color)",
    "完美像素化": "Perfect pixelate",
    "背景强制纯色（主体浅色→黑底，否则白底）": "Force solid background (pale subject → black, else white)",
    "生成 GIF / APNG / 精灵图": "Export GIF / APNG / sprite sheet",
    "输出": "Output",
    "生成提示词": "Generate prompts",
    "无参考图": "No reference",
    # ---------- 精灵图页 ----------
    "输入参数": "Input",
    "网格 i×j": "Grid i×j",
    "帧数 ≤ 行×列；多余格不裁切。如 4×4 网格、16 帧": "Frames ≤ rows×cols; extra cells are skipped. e.g. 4×4 grid, 16 frames",
    "处理选项": "Options",
    "生成精灵图": "Generate sprite sheet",
    "对象底图": "Base image",
    "精灵图": "Sprite sheet",
    "同步到 IDE": "Sync to IDE",
    "开始生成精灵图：": "Start generating sprite sheet: ",
    "精灵图完成：": "Sprite sheet done: ",
    # ---------- 像素页 ----------
    "画布设置": "Canvas settings",
    "预设": "Presets",
    "分辨率(宽×高)": "Resolution (W×H)",
    "背景": "Background",
    "透明": "Transparent",
    "白色": "White",
    "黑色": "Black",
    "新建画布": "New canvas",
    "操作": "Actions",
    "导入图片…": "Import image…",
    "从本地导入图片替换当前帧": "Import image from disk, replacing the frame",
    "从 IDE 同步": "Sync from IDE",
    "把 IDE 当前帧/首帧拉进画布精细编辑": "Pull the IDE current frame/first frame for fine editing",
    "同步到 IDE": "Sync to IDE",
    "把画布图作为首帧 + 图生图参考导入 IDE": "Use canvas image as first frame + i2i reference in IDE",
    "用作图生视频首帧": "Use as video first frame",
    "把画布图作为首帧走图生视频（Solo）；过小会自动最近邻放大到 API 最低要求": "Use canvas as video first frame (Solo); auto NEAREST-upscaled to API minimum if too small",
    "导出 PNG": "Export PNG",
    "已新建 {w}×{h} 画布": "Created {w}×{h} canvas",
    "已载入 {w}×{h} 图片": "Loaded {w}×{h} image",
    "已导出：": "Exported: ",
    # ---------- 设置 ----------
    "通用文本 API": "LLM API",
    "图片生成 API": "Image API",
    "图转视频 API": "Video API",
    "常规设置": "General",
    "界面": "UI",
    "主题": "Theme",
    "深色": "Dark",
    "浅色": "Light",
    "界面布局比例": "UI scale",
    "小（0.8×）": "Small (0.8×)",
    "标准（1.0×）": "Standard (1.0×)",
    "大（1.25×）": "Large (1.25×)",
    "特大（1.5×）": "Extra large (1.5×)",
    "缩放界面字体与布局，适配高分辨率/小屏幕设备": "Scale fonts and layout for high-DPI / small screens",
    "语言": "Language",
    "中文": "Chinese",
    "English": "English",
    "保存": "Save",
    "已保存": "Saved",
    "选择输出目录": "Choose output directory",
    "切换界面语言（重启后全局生效）": "Switch UI language (applies immediately)",
    # ---------- IDE 分步面板 ----------
    "步骤 1 · 文本生成": "Step 1 · Text generation",
    "步骤 2 · 图片生成": "Step 2 · Image generation",
    "步骤 3 · 动画生成": "Step 3 · Animation generation",
    "步骤 4 · 像素化": "Step 4 · Pixelize",
    "步骤 5 · 背景去除": "Step 5 · Background removal",
    "步骤 6 · 导出": "Step 6 · Export",
    "动作类型(可选)": "Action (optional)",
    "动作(可选)": "Action (optional)",
    "选择或输入动作…": "Choose or type an action…",
    "步行": "Walk",
    "奔跑": "Run",
    "跳跃": "Jump",
    "突进": "Dash",
    "爬行": "Crawl",
    "攻击": "Attack",
    "格挡": "Block",
    "昏迷": "Stun",
    "图片提示词": "Image prompt",
    "动画提示词": "Animation prompt",
    "负面提示词": "Negative prompt",
    "帧序列：": "Frames: ",
    "等待生成…": "Waiting…",
    "复制": "Copy",
    "暂无预览": "No preview",
    "例如：一只拿着剑的橙色小猫，Q 版，侧身站立": "e.g. an orange kitten holding a sword, chibi, side view",
    "点击添加自备参考图": "Click to add your own reference image",
    "主题: {theme}；请先在「设置」中配置 API（或勾选模拟 API）": "Theme: {theme}; configure APIs in Settings first (or enable mock APIs)",
    "开始 Solo 流程：{desc}": "Starting Solo pipeline: {desc}",
    "Solo 流程完成": "Solo pipeline finished",
    "提示词生成成功": "Prompts generated",
    "—— 步骤 {step}/{total}：{name} ——": "—— Step {step}/{total}: {name} ——",
    "插入": "Insert",
    "复制当前帧": "Duplicate current frame",
    "删除当前帧": "Delete current frame",
    "在当前帧后插入一帧（复制当前帧）": "Insert a frame after the current one (copy)",
    "追加一个空白帧": "Append a blank frame",
    "拖动缩略图可调整帧顺序": "Drag thumbnails to reorder frames",
    "帧 {i}": "Frame {i}",
    "IDE 模式：逐步执行或直接编辑帧序列": "IDE mode: run steps one by one or edit the frame sequence directly",
    "精灵图模式：仅用文生图生成网格精灵图（帧数 / i×j 网格 / 一键抠图）": "Sprite mode: grid sprite sheets from text only (frames / i×j grid / one-click keying)",
    "+ 空白帧": "+ Blank frame",
    "GIF 播放速度:": "GIF speed: ",
    "正在编辑帧 {cur}/{total}": "Editing frame {cur}/{total}",
    "暂无帧，先生成动画或添加空白帧": "No frames yet — generate an animation or add a blank frame",
    "帧 {cur}/{n}（共 {n} 帧）· {w}×{h} · {fps}fps": "Frame {cur}/{n} ({n} total) · {w}×{h} · {fps}fps",
    "就绪 · {w}×{h}": "Ready · {w}×{h}",
    "● 未保存": "● Unsaved",
    "应用提示词到工作区": "Apply prompts to workspace",
    "GIF 播放速度": "GIF speed",
    "帧序列缩略图": "Frame strip",
    "首帧图": "First frame",
    "选择参考图": "Choose reference image",
    "点击添加参考图（图生图）；已有图时点击可更换": "Click to add a reference image (i2i); click again to replace",
    "移除参考图": "Remove reference",
    "＋\n参考图": "＋\nReference",
    "服务商预设": "Provider presets",
    "查询模型": "Query models",
    "测试连接": "Test connection",
    "新建配置": "New config",
    "删除配置": "Delete config",
    "设为默认": "Set default",
    "选择模型": "Choose model",
    "模型名称": "Model name",
    "端点路径(可选)": "Endpoint path (optional)",
    "温度": "Temperature",
    "超时(秒)": "Timeout (s)",
    "代理(可选)": "Proxy (optional)",
    "校验 SSL 证书": "Verify SSL",
    "使用模拟 API（无需密钥）": "Use mock API (no key)",
    "返回格式": "Response format",
    "图片 URL": "Image URL",
    "参考图字段名(图生图)": "Reference image field (i2i)",
    "默认尺寸(宽x高)": "Default size (WxH)",
    "采样步数": "Sampling steps",
    "种子(-1 随机)": "Seed (-1 random)",
    "参考图上传方式": "Reference upload mode",
    "JSON 内嵌 base64 (data URI)": "JSON base64 (data URI)",
    "multipart 文件上传（gpt.ge 等要求）": "multipart file upload (gpt.ge etc.)",
    "服务商适配": "Provider adapter",
    "默认帧数": "Default frames",
    "默认帧率": "Default FPS",
    "首帧同时作为尾帧传入（首尾帧一致）": "Send first frame as last frame (loop)",
    "提示词模板": "Prompt template",
    "提交端点(可选)": "Submit endpoint (optional)",
    "轮询端点(可选, 含 {id})": "Poll endpoint (optional, {id})",
    "轮询方法": "Poll method",
    "轮询间隔(秒)": "Poll interval (s)",
    "最大轮询次数": "Max polls",
    "任务ID字段路径": "Job ID path",
    "状态字段路径": "Status path",
    "成功状态(逗号分隔)": "Success statuses (comma)",
    "失败状态(逗号分隔)": "Failure statuses (comma)",
    "视频URL字段路径": "Video URL path",
    "请求体模板(JSON, 可选)": "Request template (JSON, optional)",
    "额外字段(JSON, 可选)": "Extra fields (JSON, optional)",
    "首帧图最小边(过小则最近邻放大)": "First-frame min side (NEAREST upscale)",
    "首帧图长边上限": "First-frame max side",
    "使用该模型": "Use this model",
    "输入关键词过滤（如 seedance / kling / image）…": "Filter by keyword (seedance / kling / image)…",
    "（自定义）": "(custom)",
    "新建": "New",
    "删除": "Delete",
    "保存配置": "Save config",
    "高级选项（收起）": "Advanced (collapsed)",
    "高级选项（展开）": "Advanced (expanded)",
    "火山方舟 Ark": "Volcengine Ark",
    "通义千问 DashScope": "Qwen DashScope",
    "腾讯混元": "Tencent Hunyuan",
    "硅基流动 SiliconFlow": "SiliconFlow",
    "智谱 Zhipu": "Zhipu",
    "Ollama（本地）": "Ollama (local)",
    "火山方舟 Seedream": "Volcengine Seedream",
    "智谱 CogView": "Zhipu CogView",
    "硅基流动 SiliconFlow": "SiliconFlow",
    "通义万相 DashScope": "DashScope Wanx",
    "选择或输入动作（每帧的动作循环）…": "Choose or type an action (per-frame loop)…",
    "生成图片/动画提示词（LLM 失败自动用本地模板）": "Generates image/animation prompts (local fallback if LLM fails)",
    "添加参考图后即走图生图（i2i）；像素风自动按像素分辨率出图": "With a reference image this becomes i2i; pixel prompts auto-use pixel resolution",
    "参考图将作为首帧图传入视频 API；背景强制纯色等选项在「背景」步骤": "The reference image is sent as the video first frame; background options live in the Background step",
    "首帧自动检测网格大小，全部帧按同一网格精确采样；非像素风自动跳过": "Auto-detects the grid from frame 0 and samples all frames on it; non-pixel art is skipped",
    "颜色键 + 容差 + 内缩去白边 + 羽化；强制纯色影响动画生成时的背景稳定约束": "Color key + tolerance + shrink (removes fringe) + feather; forced solid background also stabilizes animation backgrounds",
    "预览抠图效果…": "Preview keying…",
    "实时预览背景扣除效果并调整容差/内缩/羽化": "Live keying preview; adjust tolerance/shrink/feather",
    "前景内缩像素：消掉对象边缘残留的白边/白晕": "Shrink the subject by N px to remove white fringe/halo",
    "导出 GIF / APNG / PNG 序列 / 雪碧图 / JSON 元数据 / 项目文件": "Export GIF / APNG / PNG sequence / sprite sheet / JSON metadata / project file",
    "同时作为首帧图：可直接走「动画生成」步骤，\n或走「图片生成」步骤做图生图。": "Also used as the first frame: run the Animation step directly,\nor the Image step for i2i.",
    "参考图（图生图，可选）\n点击添加图片": "Reference image (i2i, optional)\nClick to add an image",
    "选择预设动作会自动按建议时长设置帧数（AI 视频动作较慢）": "Preset actions auto-set the frame count from the suggested duration (AI video is slow)",
    "图转视频参数（AI 可自动调整）": "Video parameters (AI may auto-tune)",
    "0.5x（慢放）": "0.5x (slow)",
    "1x（原速）": "1x (normal)",
    "2x（提速）": "2x (faster)",
    "AI 视频动作通常偏慢，可提速播放让动作更利落": "AI video is usually slow; speed up playback for snappier motion",
    "留默认值时，LLM 会按动作自动评估时长（如 步行→2s、挥砍→1s）": "At defaults, the LLM estimates duration from the action (e.g. walk→2s, slash→1s)",
    "完美像素化（Perfect Pixel 网格采样）": "Perfect pixelate (grid sampling)",
    "去除背景（默认白色）": "Remove background (white default)",
    "背景强制纯色（主体浅色→黑底，否则白底）": "Force solid background (pale subject → black, else white)",
    "检测与画面边缘相连的背景并刷成纯色；对象本身是浅色系时自动用纯黑背景保证对比度": "Detects edge-connected background and fills it solid; pale subjects get a black background for contrast",
    "首尾帧一致（循环闭合）": "Loop close (first = last)",
    "AI 视频首尾帧常不闭合，勾选后强制首尾一致，循环播放无跳变": "AI videos rarely loop; when checked, the last frame is forced to equal the first",
    "导出 GIF 动画": "Export GIF animation",
    "导出 PNG 序列帧": "Export PNG frames",
    "导出 APNG 动画": "Export APNG animation",
    "导出雪碧图 (Sprite Sheet)": "Export sprite sheet",
    "生成精灵图": "Generate sprite sheet",
    "一键抠图（扣除纯色背景）": "Key out background (solid color)",
    "首尾帧一致（循环无缝）": "Loop close (seamless)",
    "末帧强制等于首帧；角色形象逐格一致、仅动作平滑变化": "Last frame forced to equal the first; character stays identical across cells",
    "帧数 ≤ 行×列；多余格不裁切。如 4×4 网格、16 帧": "Frames ≤ rows×cols; extra cells skipped. e.g. 4×4 grid, 16 frames",
    "把精灵图帧序列同步到 IDE 模式继续编辑": "Sync the sprite frames into IDE mode for further editing",
    "把生成的首帧图与最终帧序列同步到 IDE 模式继续编辑": "Sync the generated first frame and final frames into IDE mode",
    "完成": "Done",
    "失败": "Failed",
    "预览": "Preview",
    "播放": "Play",
    "暂停": "Pause",
    "适应": "Fit",
    "编辑": "Edit",
    "提示词": "Prompts",
    "对象底图": "Base image",
    "缩进": "Shrink",
    "已新建 {w}×{h} 画布": "Created {w}×{h} canvas",
    "已载入 {w}×{h} 图片": "Loaded {w}×{h} image",
    "已导出：": "Exported: ",
    "选择或输入动作…": "Choose or type an action…",
    "输出": "Output",
    "默认输出目录": "Default output dir",
    "数据目录：": "Data dir: ",
    "已保存": "Saved",
    "Base URL": "Base URL",
    "API Key": "API Key",
    "模型名称": "Model name",
    "服务商预设": "Provider presets",
    "查询模型": "Query models",
    "测试连接": "Test connection",
    "新建配置": "New config",
    "删除配置": "Delete config",
    "设为默认": "Set default",
    # ---------- 背景抠图预览 ----------
    "背景扣除预览": "Background key preview",
    "参数调整（实时预览）": "Adjust parameters (live preview)",
    "先强制纯色背景（自适应归一化）": "Normalize background to solid color first",
    "显示原图对照": "Show original",
    "应用到全部帧": "Apply to all frames",
    "请先生成动画或导入首帧图再预览抠图": "Generate animation or import a first frame before previewing",
    # ---------- 取色圆盘 ----------
    "色相": "Hue",
    },
}

# 语言显示名（新增语言包时在此补充）
_LANG_NAMES = {"zh": "中文", "en": "English"}


def _load_pack(name: str) -> dict:
    """从 ui/lang/<name>.py 自动加载语言包（STRINGS: {id: text}），失败则用注册表。"""
    pack = dict(LANG_PACKS.get(name, {}))
    try:
        mod = __import__(f"ui.lang.{name}", fromlist=["STRINGS"])
        pack.update(dict(getattr(mod, "STRINGS", {})))
    except ImportError:
        pass
    return pack


def available_languages() -> list:
    """可用语言列表 [(code, 显示名)]：注册表 + ui/lang/ 目录下的包。"""
    langs = [("zh", _LANG_NAMES.get("zh", "中文"))]
    for code in LANG_PACKS:
        if code == "zh":
            continue
        langs.append((code, _LANG_NAMES.get(code, code)))
    import os
    from pathlib import Path

    lang_dir = Path(__file__).resolve().parent / "lang"
    if lang_dir.is_dir():
        for p in sorted(lang_dir.glob("*.py")):
            if p.name.startswith("_") or p.stem in ("zh",) or p.stem in dict(langs):
                continue
            langs.append((p.stem, _LANG_NAMES.get(p.stem, p.stem)))
    return langs


def set_language(lang: str) -> None:
    """设置语言（zh / en / 其它语言包 key）；未知语言回退中文。"""
    global _LANG
    lang = str(lang or "zh").lower()
    if lang in LANG_PACKS or lang == "zh":
        _LANG = lang
    else:
        _LANG = "zh"


def language() -> str:
    return _LANG


def tr(text: str) -> str:
    """按当前语言翻译；未收录的 ID 回退中文原文（中文即默认包）。"""
    if _LANG == "zh":
        return text
    pack = _load_pack(_LANG)
    return pack.get(text, text)


# --------------------------------------------------------------------------- #
# 立即重译：T() 设置文本并注册，语言切换后 retranslate_all() 全局重刷。
# --------------------------------------------------------------------------- #
_REGISTRY: list = []  # (widget, attr, zh, index)


def T(widget, zh_text, attr: str = "text", index: int = None):
    """设置控件文本并注册（attr: text / tooltip / placeholder / tab）。

    widget=None 时仅返回翻译文本（等价 tr）；否则设置文本、注册并返回控件本身，
    便于链式创建：btn = T(QPushButton(), "开始生成")；f.addRow(T(QLabel(), "帧数"), spin)。
    """
    zh_text = str(zh_text)
    if widget is None:
        return tr(zh_text)
    _apply_text(widget, attr, index, tr(zh_text))
    _REGISTRY.append((widget, attr, zh_text, index))
    return widget


def _apply_text(widget, attr: str, index, text: str) -> None:
    try:
        if attr == "text":
            if hasattr(widget, "setTitle") and not hasattr(widget, "setText"):
                widget.setTitle(text)  # QGroupBox 等用 setTitle
            else:
                widget.setText(text)
        elif attr == "tooltip":
            widget.setToolTip(text)
        elif attr == "placeholder":
            widget.setPlaceholderText(text)
        elif attr == "tab":
            widget.setTabText(index, text)
    except RuntimeError:
        pass


def retranslate_all() -> None:
    """按当前语言重刷所有已注册文本（语言切换后立即生效）。"""
    for widget, attr, zh, index in list(_REGISTRY):
        _apply_text(widget, attr, index, tr(zh))
