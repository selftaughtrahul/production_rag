"""Throwaway: extract classic script blocks from the HTML for `node --check`."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

raw = Path("PROJECT_WORKFLOW.html").read_text(encoding="utf-8")
blocks = re.findall(r"<script>(.*?)</script>", raw, flags=re.S)
print("classic script blocks:", len(blocks))

for index, block in enumerate(blocks):
    tmp = Path(f"_block{index}.js")
    tmp.write_text(block, encoding="utf-8")
    result = subprocess.run(["node", "--check", str(tmp)], capture_output=True, text=True)
    print(f"block {index}: {'ok' if result.returncode == 0 else result.stderr.strip()[:400]}")
    tmp.unlink()
