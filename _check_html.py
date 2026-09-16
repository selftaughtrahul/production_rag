"""Throwaway structural check for PROJECT_WORKFLOW.html."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

VOID = {"br", "hr", "img", "input", "link", "meta", "source", "path", "circle", "rect"}


class Checker(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, int]] = []
        self.problems: list[str] = []
        self.ids: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        for key, value in attrs:
            if key == "id" and value:
                self.ids.append(value)
        if tag not in VOID:
            self.stack.append((tag, self.getpos()[0]))

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID:
            return
        if not self.stack:
            self.problems.append(f"line {self.getpos()[0]}: stray </{tag}>")
            return
        open_tag, line = self.stack.pop()
        if open_tag != tag:
            self.problems.append(
                f"line {self.getpos()[0]}: </{tag}> closes <{open_tag}> opened at line {line}"
            )


def main() -> None:
    path = Path("PROJECT_WORKFLOW.html")
    raw = path.read_text(encoding="utf-8")

    body = re.sub(r"<script.*?</script>", "", raw, flags=re.S)
    body = re.sub(r"<style.*?</style>", "", body, flags=re.S)
    checker = Checker()
    checker.feed(body)
    for tag, line in checker.stack:
        checker.problems.append(f"unclosed <{tag}> opened at line {line}")

    dupes = {i for i in checker.ids if checker.ids.count(i) > 1}
    if dupes:
        checker.problems.append(f"duplicate ids: {sorted(dupes)}")

    for href in re.findall(r'href="#([^"]+)"', raw):
        if href not in checker.ids:
            checker.problems.append(f"dangling anchor #{href}")

    for block in re.findall(r'<pre class="mermaid">(.*?)</pre>', raw, flags=re.S):
        stray = [ln for ln in block.splitlines() if "<" in ln or (">" in ln and "&gt;" not in ln)]
        if stray:
            checker.problems.append(f"unescaped mermaid line: {stray[0].strip()[:70]}")

    nums = re.findall(r'class="sec-num">(\d+)<', raw)
    if nums != [f"{i:02d}" for i in range(1, len(nums) + 1)]:
        checker.problems.append(f"section numbering out of order: {nums}")

    nav = re.findall(r'<a href="#[^"]+"><span class="num">(\d+)</span>', raw)
    if nav != nums:
        checker.problems.append(f"nav numbering {nav} != section numbering {nums}")

    print("sections:", len(nums))
    print("problems:", len(checker.problems))
    for problem in checker.problems:
        print("  -", problem)


if __name__ == "__main__":
    main()
