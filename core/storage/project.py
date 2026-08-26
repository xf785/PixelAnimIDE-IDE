"""项目文件保存/加载（JSON）。

阶段1：保存 Solo 工作流的输出元数据（帧序列目录、GIF 路径、参数等），
为阶段2 的 IDE 工作区持久化打基础。
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("PixelAnimIDE.storage.project")

PROJECT_FORMAT_VERSION = 1


@dataclass
class Project:
    name: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    fps: int = 8
    frame_count: int = 0
    width: int = 0
    height: int = 0
    frames_dir: Optional[str] = None
    gif_path: Optional[str] = None
    apng_path: Optional[str] = None
    sprite_path: Optional[str] = None
    source_video: Optional[str] = None
    first_frame: Optional[str] = None
    video_duration: Optional[float] = None
    native_width: int = 0
    native_height: int = 0
    native_gif_path: Optional[str] = None
    native_png_dir: Optional[str] = None
    prompts: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return {"format_version": PROJECT_FORMAT_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        known = {f.name for f in cls.__dataclass_fields__.values() if f.name != "format_version"}
        cleaned = {k: v for k, v in data.items() if k in known}
        return cls(**cleaned)


def save_project(project: Project, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(project.to_dict(), f, ensure_ascii=False, indent=2)
    logger.info("项目已保存: %s", path)
    return path


def load_project(path: Path | str) -> Project:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return Project.from_dict(data)
