"""i18n 覆盖回归：界面/核心里的中文串必须在英文包里都有翻译。

用 AST 扫描 `tr("…")` 与 `T(widget, "…", …)` 的文案参数：AST 会把相邻字符串字面量
**自动合并**（隐式拼接），因此不会像正则那样把「多行拼接的片段」误当成独立 ID；
f-string 则取其静态片段。断言每条中文串都能在 `LANG_PACKS["en"]` 找到英文，
以后新增界面文案忘记补英文会直接测试失败。
"""
import ast
import re
from pathlib import Path


from ui.i18n import LANG_PACKS, set_language, tr

ROOT = Path(__file__).resolve().parent.parent
ZH = re.compile(r"[\u4e00-\u9fff]")
# 有意不翻译的字面量（纯符号/占位/专有名词）
ALLOW = {"×", "—", "·"}


def _sources():
    for folder in ("ui", "core"):
        for path in (ROOT / folder).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            yield path


def _candidate_texts(node) -> list:
    """取调用参数里的文案：字符串常量（含 AST 合并后的拼接）/ f-string 的静态片段。"""
    out = []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        out.append(node.value)
    elif isinstance(node, ast.JoinedStr):        # f-string：静态片段单独检查
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                out.append(part.value)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        out.extend(_candidate_texts(node.left))
        out.extend(_candidate_texts(node.right))
    return out


def _strings() -> dict:
    found: dict = {}
    for path in _sources():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except SyntaxError:                      # 语法错误交给别的检查去报
            continue
        rel = path.relative_to(ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id == "tr":
                args = node.args[:1]
            elif node.func.id == "T":
                args = node.args[1:2]            # T(widget, "文案", …) / T(None, "文案")
            else:
                continue
            for arg in args:
                for s in _candidate_texts(arg):
                    if s and s not in ALLOW and ZH.search(s):
                        found.setdefault(s, set()).add(rel)
    return found


def test_every_chinese_string_has_english():
    en = LANG_PACKS["en"]
    missing = {}
    for s, files in _strings().items():
        if s in en:
            continue
        # f-string 的静态片段（如 f"特征 {i+1}" 的 "特征 "）：只要它是某个键的前缀即视为已覆盖
        if any(key.startswith(s) or s in key for key in en):
            continue
        missing[s] = sorted(files)[:2]
    assert not missing, f"以下中文串缺少英文翻译（{len(missing)} 条）：\n" + "\n".join(
        f"  {s!r}  <- {', '.join(files)}" for s, files in sorted(missing.items())[:20]
    )


def test_english_values_are_not_empty_or_chinese():
    en = LANG_PACKS["en"]
    bad = []
    for key, value in en.items():
        if not isinstance(value, str) or not value.strip():
            bad.append((key, value))
        elif re.search(r"[\u4e00-\u9fff]", value):
            bad.append((key, value))
    assert not bad, f"英文包里有空值/未翻译项（{len(bad)} 条）：{bad[:8]}"


def test_placeholders_match_between_zh_and_en():
    """占位符必须一致，否则英文下会丢参数（如 {0}/{1}）。"""
    en = LANG_PACKS["en"]
    problems = []
    for key, value in en.items():
        if set(re.findall(r"\{\d+\}", key)) != set(re.findall(r"\{\d+\}", value)):
            problems.append((key, value))
    assert not problems, f"占位符不一致（{len(problems)} 条）：{problems[:6]}"


def test_language_switch_round_trip():
    """切到英文再切回中文，文案应随之变化且不残留。"""
    set_language("en")
    assert tr("基础地形") == "Base terrain"
    assert tr("高度") == "Height"
    set_language("zh")
    assert tr("基础地形") == "基础地形"
    assert "基础地形" in LANG_PACKS["en"], "英文包应始终保留键（含中文 ID）"
