"""语言包目录（可选，文件式扩展点）。

约定：本目录下每个 <code>.py 导出一个 `STRINGS: dict = {id: text}`，
id 为中文原文（稳定 ID 序列），text 为该语言译文。
ui.i18n._load_pack 会在 LANG_PACKS 注册表基础上自动合并本目录下的包，
新增语言无需改代码：加一个文件 + 在 LANG_PACKS/_LANG_NAMES 注册即可，
设置页「语言」下拉会自动出现该语言。
"""
