"""i18n 覆盖回归：界面/核心里的中文串必须在英文包里都有翻译。

扫描 `tr("…")` 与 `T(widget, "…", …)` 两种调用形式（含多行拼接后的片段），
断言每条中文串都能在 `LANG_PACKS["en"]` 找到对应英文；这样以后新增界面文案时，
忘记补英文会直接测试失败。
"""
import ast
import re
from pathlib import Path

import pytest

from ui.i18n import LANG_PACKS, set_language, tr

ROOT = Path(__file__).resolve().parent.parent
CALL = re.compile(r'(?:T\(\s*[^,)]*,\s*|tr\(\s*)"((?:[^"\\]|\\.)*[\u4e00-\u9fff](?:[^"\\]|\\.)*)"')
# 有意不翻译的字面量（纯符号/占位/专有名词）
ALLOW = {"×", "—", "·"}


def _sources():
    for folder in ("ui", "core"):
        for path in (ROOT / folder).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            yield path


def _strings() -> dict:
    found: dict = {}
    for path in _sources():
        text = path.read_text(encoding="utf-8")
        for raw in CALL.findall(text):
            if not raw:
                continue
            try:                      # 源码里是转义写法（\n 等），按 Python 字面量解码后再比对
                s = ast.literal_eval(f'"{raw}"')
            except (SyntaxError, ValueError):
                s = raw
            if s and s not in ALLOW:
                found.setdefault(s, set()).add(path.relative_to(ROOT).as_posix())
    return found


def test_every_chinese_string_has_english():
    en = LANG_PACKS["en"]
    missing = {s: sorted(v)[:2] for s, v in _strings().items() if s not in en}
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
