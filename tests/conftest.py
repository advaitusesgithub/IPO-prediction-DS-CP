from pathlib import Path


def pytest_configure(config):
    root = Path(__file__).resolve().parents[1]
    (root / ".runtime").mkdir(exist_ok=True)
    (root / "outputs/part1").mkdir(parents=True, exist_ok=True)
