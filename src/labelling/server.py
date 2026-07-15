"""Minimal HTTP server for the manual affordance labelling tool."""

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import torch


class _Handler(BaseHTTPRequestHandler):
    data_root: Path
    labels_root: Path
    samples: list[dict]

    def log_message(self, fmt, *args):  # silence access log
        pass

    # ------------------------------------------------------------------
    # routing
    # ------------------------------------------------------------------

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path in ("", "/"):
                self._serve_file("static/index.html", "text/html; charset=utf-8")
            elif path == "/api/samples":
                self._json(self._samples_summary())
            elif path.startswith("/api/vertices/"):
                self._serve_tensor(path[len("/api/vertices/"):], "vertex_positions.pt", reshape=(-1, 3))
            elif path.startswith("/api/normals/"):
                self._serve_tensor(path[len("/api/normals/"):], "vertex_normals.pt", reshape=(-1, 3))
            elif path.startswith("/api/pseudolabels/"):
                stem, verb = self._split2(path[len("/api/pseudolabels/"):])
                self._serve_tensor(stem, f"vertex_pseudolabels_{verb}.pt")
            elif path.startswith("/api/manuallabels/"):
                stem, verb = self._split2(path[len("/api/manuallabels/"):])
                self._serve_manual(stem, verb)
            elif path.startswith("/api/flag/"):
                stem = unquote(path[len("/api/flag/"):])
                flag = self.data_root / "reconstructions" / stem / ".bad_reconstruction"
                self._json({"flagged": flag.exists()})
            elif path.startswith("/api/mesh/"):
                stem = unquote(path[len("/api/mesh/"):])
                mesh_path = self.data_root / "reconstructions" / stem / "mesh.glb"
                if not mesh_path.exists():
                    self.send_response(204)
                    self._cors()
                    self.end_headers()
                    return
                self._respond(200, mesh_path.read_bytes(), "model/gltf-binary")
            else:
                self.send_error(404)
        except Exception as exc:
            self.send_error(500, str(exc))

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path.startswith("/api/manuallabels/"):
                stem, verb = self._split2(path[len("/api/manuallabels/"):])
                length = int(self.headers["Content-Length"])
                raw = self.rfile.read(length)
                arr = np.frombuffer(raw, dtype=np.float32).copy()
                out = self._gt_path(stem, verb)
                out.parent.mkdir(parents=True, exist_ok=True)
                torch.save(torch.from_numpy(arr), out)
                self._respond(200, b"ok", "text/plain")
            elif path.startswith("/api/flag/"):
                stem = unquote(path[len("/api/flag/"):])
                flag = self.data_root / "reconstructions" / stem / ".bad_reconstruction"
                if flag.exists():
                    flag.unlink()
                    flagged = False
                else:
                    flag.touch()
                    flagged = True
                self._json({"flagged": flagged})
            else:
                self.send_error(404)
        except Exception as exc:
            self.send_error(500, str(exc))

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _gt_path(self, stem: str, verb: str) -> Path:
        """Canonical path for a saved GT label file (the layout eval/finetune read)."""
        return self.labels_root / stem / f"vertex_manuallabels_{verb}.pt"

    def _serve_manual(self, stem: str, verb: str):
        """Serve manual labels from gt_labels/ folder."""
        path = self._gt_path(stem, verb)
        if not path.exists():
            self.send_response(204)
            self._cors()
            self.end_headers()
            return
        t = torch.load(path, map_location="cpu", weights_only=True)
        arr = t.numpy().astype(np.float32)
        self._respond(200, arr.tobytes(), "application/octet-stream")

    def _samples_summary(self) -> list[dict]:
        out = []
        for s in self.samples:
            stem = Path(s["sam3d_reconstruction_dir"]).name
            verb = s["verb"]
            has_manual = self._gt_path(stem, verb).exists()
            is_flagged = (
                self.data_root / "reconstructions" / stem / ".bad_reconstruction"
            ).exists()
            out.append(
                {
                    "sample_id": s["sample_id"],
                    "stem": stem,
                    "verb": verb,
                    "category": s["category"],
                    "split": s.get("split", ""),
                    "has_manual": has_manual,
                    "is_flagged": is_flagged,
                }
            )
        return out

    def _serve_tensor(self, stem: str, filename: str, reshape=None):
        stem = unquote(stem)
        path = self.data_root / "reconstructions" / stem / filename
        if not path.exists():
            self.send_response(204)
            self._cors()
            self.end_headers()
            return
        t = torch.load(path, map_location="cpu", weights_only=True)
        arr = t.numpy().astype(np.float32)
        if reshape:
            arr = arr.reshape(reshape)
        data = arr.tobytes()
        self._respond(200, data, "application/octet-stream")

    def _json(self, obj):
        data = json.dumps(obj).encode()
        self._respond(200, data, "application/json")

    def _respond(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _serve_file(self, rel: str, ctype: str):
        full = Path(__file__).parent / rel
        if not full.exists():
            self.send_error(404)
            return
        self._respond(200, full.read_bytes(), ctype)

    @staticmethod
    def _split2(s: str) -> tuple[str, str]:
        s = unquote(s)
        idx = s.index("/")
        return s[:idx], s[idx + 1:]


def run(data_root: Path, port: int = 8765, split: str | None = None, labels_root: Path | None = None):
    manifest = data_root / "manifest.pseudolabeled.vsem.jsonl"
    if not manifest.exists():
        manifest = data_root / "manifest.jsonl"
    with open(manifest) as f:
        samples = [json.loads(l) for l in f]
    if split:
        samples = [s for s in samples if s.get("split") == split]

    class Handler(_Handler):
        pass

    Handler.data_root = data_root
    Handler.labels_root = labels_root or (data_root / "human_gt_labels")
    Handler.samples = samples

    server = HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"Labeller running at {url}  ({len(samples)} samples)")
    print(f"  reading/writing labels in: {Handler.labels_root}")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
