import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config

DATA_DIR = Path(__file__).parent.parent / "data"


def test_every_project_has_a_description():
    missing = [name for name in config.PROJECTS if name not in config.PROJECT_DESCRIPTIONS]
    assert missing == []


def test_every_project_has_a_saved_readme_fallback():
    missing = [name for name in config.PROJECTS if not (DATA_DIR / f"{name}_readme_fixture.md").exists()]
    assert missing == []
