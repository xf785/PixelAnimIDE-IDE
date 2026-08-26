"""项目文件保存/加载测试。"""
from core.storage.project import Project, load_project, save_project


def test_save_load_roundtrip(tmp_path):
    project = Project(
        name="solo_test",
        fps=8,
        frame_count=6,
        width=128,
        height=128,
        frames_dir="C:/out/frames",
        gif_path="C:/out/anim.gif",
        prompts={"image_prompt": "x"},
        params={"description": "cat"},
    )
    path = save_project(project, tmp_path / "p" / "project.json")
    assert path.exists()
    loaded = load_project(path)
    assert loaded.name == "solo_test"
    assert loaded.fps == 8
    assert loaded.frame_count == 6
    assert loaded.prompts["image_prompt"] == "x"
    assert loaded.params["description"] == "cat"


def test_defaults():
    project = Project(name="x")
    assert project.frame_count == 0
    assert project.created_at  # 自动时间戳
