"""Every relative link and repository path mentioned in the Markdown files must exist."""

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACKED = subprocess.check_output(["git", "ls-files", "-z", "*.md"], cwd=ROOT, text=True).split("\0")
DOC_FILES = sorted(ROOT / name for name in TRACKED
                   if name and not name.startswith("third_party/") and (ROOT / name).is_file())

LINK = re.compile(r"\]\(([^)\s]+)\)")
CODE_PATH = re.compile(
    r"`((?:tools|docs|f1tenth_[a-z]+|sdu_apex_[a-z]+|raceline_optimizer|third_party|docker|config)"
    r"/[A-Za-z0-9_./-]+)`")
RUNTIME_ROOTS = ("live_runs/", "build/", "install/")


class DocumentationPathTests(unittest.TestCase):
    def test_markdown_links_resolve(self):
        missing = []
        for doc in DOC_FILES:
            for target in LINK.findall(doc.read_text()):
                if re.match(r"[a-z]+:", target) or target.startswith("#"):
                    continue
                path = (doc.parent / target.split("#")[0]).resolve()
                if not path.exists():
                    missing.append(f"{doc.relative_to(ROOT)}: {target}")
        self.assertEqual(missing, [])

    def test_repository_paths_in_code_spans_exist(self):
        missing = []
        for doc in DOC_FILES:
            for target in CODE_PATH.findall(doc.read_text()):
                if target.startswith(RUNTIME_ROOTS) or "*" in target or ".venv" in target:
                    continue
                if re.fullmatch(r"sdu_apex_msgs/[A-Z]\w+", target):  # ROS message type, not a path
                    continue
                if not (ROOT / target.rstrip("/")).exists():
                    missing.append(f"{doc.relative_to(ROOT)}: {target}")
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
