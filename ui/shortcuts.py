"""快捷键注册表与解析：按模式（Solo/IDE/精灵图/像素）分组的键位配置。

设计：
- MODES：四个工作模式，每个模式有独立的键位条目（类别 / 条目 / 默认快捷键）；
- 用户配置存 ui_settings["shortcuts"]：{mode: {action_id: "Ctrl+Z"}}（旧格式
  {action_id: ...} 自动迁移到 pixel 模式）；
- 运行期经 set_shortcuts() 载入模块缓存；控件用 get(action_id) 读取
  「当前生效模式」（set_active_mode）的绑定，match(event, seq) 判断按键；
- 修改立即生效（缓存即配置），点「保存」后持久化。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt

# 四个工作模式（与主窗口模式一致）
MODES = ("solo", "ide", "sprite", "pixel")

# mode -> {action_id: (类别, 条目名(zh ID), 默认快捷键)}
# 默认快捷键为空字符串表示默认不绑定（用户可自行设置）
SHORTCUT_DEFS: Dict[str, Dict[str, Tuple[str, str, str]]] = {
    # ---- Solo：预览 ----
    "solo": {
        "preview_play": ("预览", "播放/暂停", "Space"),
        "preview_fit": ("预览", "适应窗口", "F"),
    },
    # ---- IDE：预览 + 时间轴 ----
    "ide": {
        "preview_play": ("预览", "播放/暂停", "Space"),
        "preview_fit": ("预览", "适应窗口", "F"),
        "timeline_insert": ("时间轴", "插入帧", "I"),
        "timeline_duplicate": ("时间轴", "复制帧", "D"),
        "timeline_delete": ("时间轴", "删除帧", "Delete"),
    },
    # ---- 精灵图：预览 ----
    "sprite": {
        "preview_play": ("预览", "播放/暂停", "Space"),
    },
    # ---- 像素编辑器：编辑 / 选择 / 视图 / 工具 ----
    "pixel": {
        "undo": ("编辑", "撤销", "Ctrl+Z"),
        "redo": ("编辑", "重做", "Ctrl+Shift+Z"),
        "copy": ("编辑", "复制选区", "Ctrl+C"),
        "paste": ("编辑", "粘贴为图层", "Ctrl+V"),
        "merge": ("编辑", "合并图层", "Ctrl+M"),
        "select_all": ("选择", "全部选择", "Ctrl+A"),
        "deselect": ("选择", "取消选择", "Esc"),
        "zoom_in": ("视图", "放大", "Ctrl+="),
        "zoom_out": ("视图", "缩小", "Ctrl+-"),
        "tool_pencil": ("工具", "铅笔", ""),
        "tool_eraser": ("工具", "橡皮", ""),
        "tool_eyedropper": ("工具", "取色", ""),
        "tool_fill": ("工具", "填充", ""),
        "tool_select": ("工具", "选择", ""),
    },
}

DEFAULTS: Dict[str, Dict[str, str]] = {
    mode: {aid: default for aid, (_cat, _name, default) in defs.items()}
    for mode, defs in SHORTCUT_DEFS.items()
}

_config: Dict[str, Dict[str, str]] = {}  # 用户配置（非默认值）
_active_mode: str = "pixel"


# --------------------------------------------------------------------------- #
# 配置读写
# --------------------------------------------------------------------------- #
def set_shortcuts(mapping: Optional[dict]) -> None:
    """载入用户配置。

    支持两种格式：
    - 新格式 {mode: {action_id: seq}}（键为模式名）；
    - 旧格式 {action_id: seq}（键为条目 id，自动归入 pixel 模式）。
    """
    global _config
    _config = {}
    if not mapping:
        return
    if any(k not in SHORTCUT_DEFS for k in mapping):
        # 旧格式：{action_id: seq} -> pixel 模式（键不是模式名）
        _config["pixel"] = _sanitize("pixel", mapping)
        return
    for mode, items in mapping.items():
        if mode in SHORTCUT_DEFS and isinstance(items, dict):
            _config[mode] = _sanitize(mode, items)


def _sanitize(mode: str, items: dict) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for aid, seq in items.items():
        if aid in SHORTCUT_DEFS[mode] and isinstance(seq, str) and seq:
            out[aid] = seq
    return out


def get(action_id: str, mode: Optional[str] = None) -> str:
    """当前生效的快捷键文本（用户配置优先，否则默认）。

    mode=None 时使用 set_active_mode 设置的模式（推荐）。
    """
    m = mode if mode in SHORTCUT_DEFS else _active_mode
    cfg = _config.get(m, {})
    return cfg.get(action_id, DEFAULTS[m].get(action_id, ""))


def set_active_mode(mode: str) -> None:
    """设置当前生效模式（主窗口切换模式时调用）。"""
    global _active_mode
    if mode in SHORTCUT_DEFS:
        _active_mode = mode


def active_mode() -> str:
    return _active_mode


def to_settings() -> Dict[str, Dict[str, str]]:
    """导出非默认配置（用于持久化，保持文件精简）。"""
    out: Dict[str, Dict[str, str]] = {}
    for mode, cfg in _config.items():
        non_default = {aid: s for aid, s in cfg.items() if s != DEFAULTS[mode].get(aid)}
        if non_default:
            out[mode] = non_default
    return out


# --------------------------------------------------------------------------- #
# 条目元数据（设置面板用）
# --------------------------------------------------------------------------- #
def categories(mode: Optional[str] = None) -> Tuple[str, ...]:
    """某模式下的类别顺序（一级下拉）。"""
    m = mode if mode in SHORTCUT_DEFS else _active_mode
    seen: List[str] = []
    for _aid, (cat, _name, _d) in SHORTCUT_DEFS[m].items():
        if cat not in seen:
            seen.append(cat)
    return tuple(seen)


def actions_in(category: str, mode: Optional[str] = None) -> List[str]:
    """某类别下的条目 action_id 列表（二级下拉）。"""
    m = mode if mode in SHORTCUT_DEFS else _active_mode
    return [aid for aid, (cat, _n, _d) in SHORTCUT_DEFS[m].items() if cat == category]


def action_name(action_id: str, mode: Optional[str] = None) -> str:
    m = mode if mode in SHORTCUT_DEFS else _active_mode
    return SHORTCUT_DEFS[m][action_id][1]


def all_entries(mode: Optional[str] = None) -> Dict[str, str]:
    """某模式全部条目的当前生效快捷键（冲突检测用）。"""
    m = mode if mode in SHORTCUT_DEFS else _active_mode
    return {aid: get(aid, m) for aid in SHORTCUT_DEFS[m]}


# --------------------------------------------------------------------------- #
# 解析与匹配
# --------------------------------------------------------------------------- #
_MOD_NAMES = {
    "Ctrl": Qt.KeyboardModifier.ControlModifier,
    "Control": Qt.KeyboardModifier.ControlModifier,
    "Shift": Qt.KeyboardModifier.ShiftModifier,
    "Alt": Qt.KeyboardModifier.AltModifier,
    "Meta": Qt.KeyboardModifier.MetaModifier,
    "Super": Qt.KeyboardModifier.MetaModifier,
}

_KEY_NAMES = {
    "Esc": Qt.Key.Key_Escape,
    "Escape": Qt.Key.Key_Escape,
    "Space": Qt.Key.Key_Space,
    "Enter": Qt.Key.Key_Return,
    "Return": Qt.Key.Key_Return,
    "Tab": Qt.Key.Key_Tab,
    "Backspace": Qt.Key.Key_Backspace,
    "Delete": Qt.Key.Key_Delete,
    "Del": Qt.Key.Key_Delete,
    "Insert": Qt.Key.Key_Insert,
    "Home": Qt.Key.Key_Home,
    "End": Qt.Key.Key_End,
    "PageUp": Qt.Key.Key_PageUp,
    "PageDown": Qt.Key.Key_PageDown,
    "Up": Qt.Key.Key_Up,
    "Down": Qt.Key.Key_Down,
    "Left": Qt.Key.Key_Left,
    "Right": Qt.Key.Key_Right,
    "Plus": Qt.Key.Key_Plus,
    "Minus": Qt.Key.Key_Minus,
    "=": Qt.Key.Key_Equal,
    "-": Qt.Key.Key_Minus,
    "+": Qt.Key.Key_Plus,
    ".": Qt.Key.Key_Period,
    ",": Qt.Key.Key_Comma,
    "/": Qt.Key.Key_Slash,
    ";": Qt.Key.Key_Semicolon,
    "'": Qt.Key.Key_Apostrophe,
    "[": Qt.Key.Key_BracketLeft,
    "]": Qt.Key.Key_BracketRight,
    "`": Qt.Key.Key_QuoteLeft,
    "\\": Qt.Key.Key_Backslash,
}
for _i in range(12):
    _KEY_NAMES[f"F{_i + 1}"] = getattr(Qt.Key, f"Key_F{_i + 1}")
for _i in range(10):
    _KEY_NAMES[str(_i)] = getattr(Qt.Key, f"Key_{_i}")


def parse_shortcut(text: str) -> Optional[Tuple[Qt.KeyboardModifier, Qt.Key]]:
    """'Ctrl+Shift+Z' -> (mods, key)；解析失败返回 None。"""
    if not text:
        return None
    parts = [p.strip() for p in text.split("+") if p.strip()]
    if not parts:
        return None
    mods = Qt.KeyboardModifier.NoModifier
    key_name = parts[-1]
    for p in parts[:-1]:
        m = _MOD_NAMES.get(p)
        if m is None:
            return None
        mods = mods | m
    if key_name in _KEY_NAMES:
        key = _KEY_NAMES[key_name]
    elif len(key_name) == 1 and key_name.isalpha():
        key = getattr(Qt.Key, f"Key_{key_name.upper()}")
    else:
        return None
    return mods, key


def match(event, text: str) -> bool:
    """按键事件是否匹配快捷键文本。

    事件必须包含快捷键的全部修饰键，且不得包含其他修饰键；
    等号/加号在部分键盘布局需要 Shift，匹配时允许额外 Shift/小键盘修饰。
    """
    seq = parse_shortcut(text)
    if seq is None:
        return False
    mods, key = seq
    ev_mods = event.modifiers()
    missing = mods & ~ev_mods
    extra = ev_mods & ~mods
    allowed = Qt.KeyboardModifier.KeypadModifier
    if key in (Qt.Key.Key_Equal, Qt.Key.Key_Plus):
        allowed = allowed | Qt.KeyboardModifier.ShiftModifier
    if missing or (extra & ~allowed):
        return False
    return event.key() == key


def format_shortcut(text: str) -> str:
    """展示用：空串显示 —。"""
    return text or "—"


def key_to_text(mods: Qt.KeyboardModifier, key: Qt.Key) -> str:
    """按键捕获用：把 (mods, key) 转成 'Ctrl+Shift+Z' 文本。"""
    names: List[str] = []
    if mods & Qt.KeyboardModifier.ControlModifier:
        names.append("Ctrl")
    if mods & Qt.KeyboardModifier.ShiftModifier:
        names.append("Shift")
    if mods & Qt.KeyboardModifier.AltModifier:
        names.append("Alt")
    if mods & Qt.KeyboardModifier.MetaModifier:
        names.append("Meta")
    rev = {v: k for k, v in _KEY_NAMES.items()}
    if key in rev:
        names.append(rev[key])
    else:
        ch = chr(key) if 32 <= key < 127 else None
        names.append(ch if ch else f"Key{int(key)}")
    return "+".join(names)
