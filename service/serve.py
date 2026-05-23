#!/usr/bin/env python3
"""face-py HTTP service.

骨架阶段 (Phase A.1): /api/v1/health 可用, /api/v1/face/identity_check + /api/v1/face/compare
返 501 not_implemented 占位. Phase A.2 接 module/face/pipeline.py 实现.

跟 cloth/service/serve.py + day_night/service/serve.py 同款 stdlib http.server 骨架.
跟 face C++ face_http_server.cpp 客户契约对齐 (响应 envelope + auth 协议).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from module.face.pipeline import FacePipeline, build_pipeline_from_env  # noqa: E402


DEFAULT_PORT = 32192   # face C++ 占 32186, face-py 走 32192


# =============================================================================
# Env helpers — 同款 cloth/serve.py style
# =============================================================================

def env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"env {name} must be int, got {raw!r}")


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise SystemExit(f"env {name} must be float, got {raw!r}")


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


# =============================================================================
# Logging
# =============================================================================

def configure_logging() -> logging.Logger:
    level_name = env_str("FACE_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":%(message)s}',
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    return logging.getLogger("face-py")


# =============================================================================
# HTTP handler
# =============================================================================

class Handler(BaseHTTPRequestHandler):
    server_version = "FacePyHTTP/0.0.1"

    # Class attrs filled by main()
    auth_token: str = ""
    auth_required: bool = False
    started_at: float = 0.0
    log: logging.Logger
    log_request_bodies: bool = False
    pipeline: FacePipeline  # set by main() before serve_forever

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # 默认 access log 太吵, 我们用 self.log 自己 emit 结构化日志.
        return

    # ----- 响应工具 -----

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str,
                request_id: str = "", error_code: str = "") -> None:
        # 跟 face C++ HttpError 输出 shape 对齐 — 客户端按 error / error_code 字段
        # 取消息, 别加 envelope.
        payload: dict[str, Any] = {"error": message, "request_id": request_id}
        if error_code:
            payload["error_code"] = error_code
        self._write_json(status, payload)

    def _check_auth(self) -> bool:
        if not self.auth_required:
            return True
        token = self.auth_token
        header = self.headers.get("Authorization", "")
        if header.startswith("Bearer ") and header[7:] == token:
            return True
        if self.headers.get("X-API-Key", "") == token:
            return True
        return False

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            raise ValueError("empty request body")
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
        return body

    # ----- 路由 -----

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/v1/health":
            self._handle_health(); return
        self._error(HTTPStatus.NOT_FOUND, f"unknown path: {self.path}")

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/v1/face/identity_check":
            self._handle_identity_check(); return
        if self.path == "/api/v1/face/compare":
            self._handle_compare(); return
        self._error(HTTPStatus.NOT_FOUND, f"unknown path: {self.path}")

    # ----- handlers -----

    def _handle_health(self) -> None:
        uptime = time.time() - self.started_at
        self._write_json(HTTPStatus.OK, {
            "status": "ok",
            "version": _read_version(),
            "uptime_seconds": round(uptime, 3),
            "auth_required": self.auth_required,
        })

    def _handle_identity_check(self) -> None:
        request_id = ""
        if not self._check_auth():
            self._error(HTTPStatus.UNAUTHORIZED, "missing authentication token",
                        error_code="unauthorized"); return
        try:
            body = self._read_json_body()
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc), request_id, "invalid_request"); return
        request_id = str(body.get("request_id", "") or uuid.uuid4())

        image_path = str(body.get("image_path", ""))
        ref_image_path = str(body.get("ref_image_path", ""))
        if not image_path:
            self._error(HTTPStatus.BAD_REQUEST,
                         "Field 'image_path' required (non-empty string)",
                         request_id, "invalid_request"); return
        if not ref_image_path:
            self._error(HTTPStatus.BAD_REQUEST,
                         "Field 'ref_image_path' required (non-empty string)",
                         request_id, "invalid_request"); return

        # face C++ identity_check 返 200 + error_code 字段 (业务级失败不抛 HTTP 5xx).
        result = self.pipeline.identity_check(image_path, ref_image_path)
        self._write_json(HTTPStatus.OK, result.to_json())

    def _handle_compare(self) -> None:
        request_id = ""
        if not self._check_auth():
            self._error(HTTPStatus.UNAUTHORIZED, "missing authentication token",
                        error_code="unauthorized"); return
        try:
            body = self._read_json_body()
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc), request_id, "invalid_request"); return
        request_id = str(body.get("request_id", "") or uuid.uuid4())

        image_a = str(body.get("image_a_path", ""))
        image_b = str(body.get("image_b_path", ""))
        if not image_a or not image_b:
            self._error(HTTPStatus.BAD_REQUEST,
                         "Fields 'image_a_path' and 'image_b_path' required",
                         request_id, "invalid_request"); return

        result = self.pipeline.compare(image_a, image_b)
        self._write_json(HTTPStatus.OK, result.to_json())


# =============================================================================
# Utilities
# =============================================================================

def _read_version() -> str:
    try:
        return (_REPO / "VERSION").read_text().strip()
    except OSError:
        return "0.0.0"


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    log = configure_logging()
    host = env_str("FACE_HTTP_HOST", "127.0.0.1")
    port = env_int("FACE_HTTP_PORT", DEFAULT_PORT)
    auth_token = env_str("FACE_HTTP_AUTH_TOKEN", "")
    auth_required = env_bool("FACE_HTTP_AUTH_REQUIRED", bool(auth_token))
    if auth_required and not auth_token:
        raise SystemExit("FACE_HTTP_AUTH_REQUIRED is set but FACE_HTTP_AUTH_TOKEN is empty")

    Handler.auth_token = auth_token
    Handler.auth_required = auth_required
    Handler.started_at = time.time()
    Handler.log = log
    Handler.log_request_bodies = env_bool("FACE_LOG_REQUEST_BODIES", False)

    # Build pipeline (loads ONNX models + creates CUDA session). This is slow
    # (~1-2s GPU warmup), so we do it once at startup, not per request.
    log.info(json.dumps({"event": "pipeline_init"}))
    Handler.pipeline = build_pipeline_from_env()
    log.info(json.dumps({
        "event": "pipeline_ready",
        "providers": Handler.pipeline.recognizer.providers,
    }))

    log.info(json.dumps({
        "event": "starting", "host": host, "port": port,
        "auth_required": auth_required,
        "version": _read_version(),
    }))

    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info(json.dumps({"event": "shutting_down"}))
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
