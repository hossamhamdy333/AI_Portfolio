import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))
import config


def read_part(project, file_path):
    if project == "portfolio_overview":
        return (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    repo = config.PROJECT_REPO_OVERRIDES.get(project)
    if repo is None:
        return (REPO_ROOT / project / file_path).read_text(encoding="utf-8")
    branch = config.PROJECT_BRANCH_OVERRIDES.get(project, config.GITHUB_BRANCH)
    url = f"https://raw.githubusercontent.com/{config.GITHUB_ORG}/{repo}/{branch}/{file_path}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.text


def build(project):
    files = config.PROJECT_FILES.get(project, ["README.md"])
    parts = []
    for i, file_path in enumerate(files):
        text = read_part(project, file_path)
        if i == 0:
            parts.append(text)
        else:
            parts.append(f"\n\n---\n\n# Supplementary document: `{file_path}`\n\n{text}")
    return "".join(parts)


def main():
    write = "--write" in sys.argv
    for project in config.PROJECTS:
        path = DATA / f"{project}_readme_fixture.md"
        new = build(project)
        old = path.read_text(encoding="utf-8") if path.exists() else None
        status = "same" if old == new else "CHANGED"
        print(f"{status:8}{project}")
        if write and status == "CHANGED":
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(new)


main()
