import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _deps():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]


def test_every_runtime_dependency_has_upper_bound():
    unbounded = [d for d in _deps() if "<" not in d]
    assert unbounded == []


def test_langchain_is_1x():
    deps = " ".join(_deps())
    assert "langchain-core>=1.7,<2" in deps
    assert "langchain-community>=0.4,<0.5" in deps


def test_requirements_txt_only_installs_package():
    lines = [line.strip() for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()]
    pkgs = [line for line in lines if line and not line.startswith("#")]
    assert pkgs == ["-e ."]
