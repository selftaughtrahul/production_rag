"""HTML pages for the jQuery frontend. Files live in web/ at the repo root."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

WEB_ROOT = Path(__file__).resolve().parents[2] / "web"

router = APIRouter(tags=["Pages"], include_in_schema=False)

_PAGES: dict[str, str] = {
    "/": "index.html",
    "/login": "login.html",
    "/dashboard": "dashboard.html",
    "/app": "chat.html",
}


def _serve(file_name: str):
    def page() -> FileResponse:
        # Pages carry the auth gate, so never let the browser cache a stale copy.
        return FileResponse(
            WEB_ROOT / file_name,
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    return page


for _path, _file in _PAGES.items():
    router.add_api_route(_path, _serve(_file), methods=["GET"], name=_file)
