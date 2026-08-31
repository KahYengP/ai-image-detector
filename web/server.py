"""Local UI server for the UM detector pages.

Serves `web/` and runs the same scoring path as `predict.py`.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
WEB = Path(__file__).resolve().parent
UPLOADS = WEB / "uploads"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.augmentations import ModelTransform
from predict import load_checkpoint, predict_image
from utils import IMAGE_EXTS, get_device, list_images, load_config, project_root, resolve_checkpoint

_MODEL = None
_LOCK = threading.Lock()


def _engine():
    global _MODEL
    with _LOCK:
        if _MODEL is None:
            cfg = load_config()
            device = get_device(None)
            ckpt = resolve_checkpoint(cfg)
            model, saved = load_checkpoint(ckpt, device)
            transform = ModelTransform(saved, augment=False)
            thresholds = saved.get("thresholds") or cfg.get("thresholds") or {}
            clip_name = saved.get("model", {}).get("clip_name") or "openai/clip-vit-base-patch32"
            _MODEL = {
                "model": model,
                "transform": transform,
                "device": device,
                "thresholds": thresholds,
                "clip_name": clip_name,
            }
        return _MODEL


def _safe_name(name: str) -> str:
    stem = Path(name).name
    stem = re.sub(r"[^A-Za-z0-9._\- ()]", "_", stem)
    return stem[:180] or "image.jpg"


def _enrich(record: dict, path: Path, url: str) -> dict:
    out = dict(record)
    out["image_url"] = url
    try:
        out["file_size"] = path.stat().st_size
    except OSError:
        out["file_size"] = None
    try:
        from PIL import Image

        with Image.open(path) as im:
            out["width"], out["height"] = im.size
    except Exception:
        out["width"] = out["height"] = None
    return out


def _score_path(path: Path, image_dir: Path, url: str) -> dict:
    eng = _engine()
    rec = predict_image(
        eng["model"],
        eng["transform"],
        path,
        eng["device"],
        eng["thresholds"],
        clip_name=eng["clip_name"],
        image_dir=image_dir,
    )
    rec["image_path"] = path.name
    return _enrich(rec, path, url)


def _decode_data_url(data: str) -> bytes:
    if "," in data:
        data = data.split(",", 1)[1]
    return base64.b64decode(data)


def _pick_sample_paths() -> list[Path]:
    roots = [project_root() / "test-images", project_root() / "train images"]
    images: list[Path] = []
    for root in roots:
        if root.is_dir():
            images.extend(list_images(root))
    if not images:
        return []

    def kind(path: Path) -> str:
        stem = path.stem.lower().replace("_", "-")
        if stem.startswith("ai-generated"):
            return "ai"
        if stem.startswith("filtered-edited") or stem.startswith("edited-filtered"):
            return "edit"
        if stem.startswith("real") and not stem.startswith("real-live"):
            return "real"
        return "other"

    picked: list[Path] = []
    used: set[Path] = set()
    for want in ("ai", "real", "edit"):
        candidates = [p for p in images if p not in used and kind(p) == want]
        candidates.sort(key=lambda p: (-len(p.stem), p.name))
        if candidates:
            picked.append(candidates[0])
            used.add(candidates[0])
    for path in images:
        if len(picked) >= 3:
            break
        if path not in used:
            picked.append(path)
            used.add(path)
    return picked[:3]


def _media_url(path: Path) -> str:
    path = path.resolve()
    train = (project_root() / "train images").resolve()
    test = (project_root() / "test-images").resolve()
    try:
        path.relative_to(train.resolve())
        return f"/media/train/{path.name}"
    except ValueError:
        pass
    try:
        path.relative_to(test.resolve())
        return f"/media/test/{path.name}"
    except ValueError:
        pass
    return f"/media/uploads/{path.name}"


def _write_predictions(records: list[dict]) -> None:
    dest = project_root() / "outputs" / "predictions.json"
    slim = []
    for rec in records:
        slim.append({k: v for k, v in rec.items() if k not in {"image_url", "preview"}})
    dest.write_text(json.dumps(slim, indent=2), encoding="utf-8")


class Handler(SimpleHTTPRequestHandler):
    timeout = 600

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def end_headers(self) -> None:
        path = unquote((self.path or "/").split("?", 1)[0])
        if path.endswith((".css", ".js", ".html")) or path in {"/", "/index.html"}:
            self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def _json(self, payload: dict, status: int = 200) -> None:
        blob = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def do_GET(self) -> None:
        path = unquote(self.path.split("?", 1)[0])
        if path == "/api/health":
            return self._json({"ok": True, "engine": "ready"})
        if path.startswith("/media/"):
            return self._serve_media(path)
        if path in {"/", "/index.html"}:
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:
        path = unquote(self.path.split("?", 1)[0])
        try:
            if path == "/api/analyze":
                return self._analyze()
            if path == "/api/sample":
                return self._sample()
        except Exception as exc:
            return self._json({"error": str(exc)}, 500)
        self._json({"error": "not found"}, 404)

    def _serve_media(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) < 3:
            self.send_error(404)
            return
        kind, name = parts[1], "/".join(parts[2:])
        name = _safe_name(unquote(name))
        roots = {
            "uploads": UPLOADS,
            "test": project_root() / "test-images",
            "train": project_root() / "train images",
        }
        root = roots.get(kind)
        if root is None:
            self.send_error(404)
            return
        file_path = (root / name).resolve()
        try:
            file_path.relative_to(root.resolve())
        except ValueError:
            self.send_error(403)
            return
        if not file_path.is_file():
            self.send_error(404)
            return
        ctype = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _analyze(self) -> None:
        body = self._read_json()
        files = body.get("files") or []
        if not files:
            return self._json({"error": "No images uploaded."}, 400)
        UPLOADS.mkdir(parents=True, exist_ok=True)
        records = []
        for item in files:
            name = _safe_name(item.get("name") or "image.jpg")
            if Path(name).suffix.lower() not in IMAGE_EXTS:
                name = name + ".jpg"
            dest = UPLOADS / name
            dest.write_bytes(_decode_data_url(item.get("data") or ""))
            records.append(_score_path(dest, UPLOADS, f"/media/uploads/{name}"))
        _write_predictions(records)
        self._json({"records": records})

    def _sample(self) -> None:
        picked = _pick_sample_paths()
        if not picked:
            return self._json({"error": "No sample images found in test-images or train images."}, 400)
        records = []
        for path in picked:
            records.append(_score_path(path, path.parent, _media_url(path)))
        _write_predictions(records)
        self._json({"records": records})


def main() -> None:
    UPLOADS.mkdir(parents=True, exist_ok=True)
    host, port = "127.0.0.1", 8765
    ThreadingHTTPServer.allow_reuse_address = False
    try:
        httpd = ThreadingHTTPServer((host, port), Handler)
    except OSError as exc:
        raise SystemExit(
            f"Port {port} is already in use. Close the other python web/server.py "
            f"window, or in PowerShell run:\n"
            f"  netstat -ano | findstr :{port}\n"
            f"  taskkill /PID <pid> /F\n({exc})"
        ) from exc
    print(f"UM detector UI  http://{host}:{port}")
    print("Open that URL, then Analyze or Load Sample Test Batch.")
    print("Leave this window open. Stop with Ctrl+C.")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
