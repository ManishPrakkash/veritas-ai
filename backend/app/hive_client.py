"""
Thin wrapper around Hive's image AI-detection endpoint.

This calls the same unauthenticated "plugin" endpoint the existing static
site (js/main.js) already calls client-side through public CORS proxies —
we're just doing it server-side now, with no proxy needed since server-to-
server requests aren't subject to browser CORS.

Caveat: this endpoint isn't a documented/keyed developer API, it's the one
Hive's own browser extension talks to. It could change or start rejecting
non-browser-like requests without notice. If that happens, or once this
needs to be production-reliable, switch to Hive's official developer API
(requires an account + API key) instead of reverse-engineering the plugin
endpoint further.
"""

import uuid
from typing import Optional

import httpx

HIVE_URL = "https://plugin.hivemoderation.com/api/v1/image/ai_detection"

# Same two "nothing to see here" labels js/main.js excludes when picking
# the headline class/score.
EXCLUDED_CLASSES = {"not_ai_generated", "none"}


class HiveError(Exception):
    pass


def pick_top_class(classes: list[dict]) -> tuple[Optional[str], float]:
    """Highest score among classes that aren't the excluded "not AI" labels.

    Mirrors the reduction in js/main.js:fillResults exactly (that function's
    loop only advances `highest.score` past 0 once it hits a non-excluded
    class, so the net result is: max score among non-excluded classes, and
    its class name).
    """
    best_name: Optional[str] = None
    best_score = 0.0

    for entry in classes:
        if entry.get("class") in EXCLUDED_CLASSES:
            continue

        score = entry.get("score", 0)
        if score > best_score:
            best_name, best_score = entry.get("class"), score

    return best_name, best_score


async def analyze(image_bytes: bytes, filename: str) -> tuple[Optional[str], float]:
    files = {"media": (filename, image_bytes, "image/jpeg")}
    data = {"request_id": str(uuid.uuid4())}

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(HIVE_URL, files=files, data=data)

    response.raise_for_status()
    payload = response.json()

    if payload.get("status_code") != 200:
        raise HiveError(payload.get("message", "Hive returned an error"))

    classes = payload.get("data", {}).get("classes", [])
    return pick_top_class(classes)
