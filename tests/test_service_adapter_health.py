"""Health endpoint contract for the optional identity domain adapter."""
from __future__ import annotations

import json
import time
import unittest
from types import SimpleNamespace

from module.face.domain_adapter import DomainAdapterRuntime
from service.serve import Handler


class _HealthRequest:
    """Minimal handler surface needed to exercise the real health response."""

    auth_required = False
    started_at = time.time() - 1

    def __init__(self, adapter: DomainAdapterRuntime) -> None:
        self.pipeline = SimpleNamespace(domain_adapter=adapter)
        self.status: int | None = None
        self.body: dict | None = None

    def _write_json(self, status: int, body: dict) -> None:
        self.status = status
        self.body = body


class ServiceAdapterHealthTests(unittest.TestCase):
    def health(
        self,
        *,
        adapter_mode: str,
        ready: bool,
        version: str = "",
        sha256: str = "",
        load_error: str = "",
    ) -> dict:
        request = _HealthRequest(
            DomainAdapterRuntime(
                mode=adapter_mode,
                ready=ready,
                version=version,
                artifact_sha256=sha256,
                load_error=load_error,
            )
        )

        Handler._handle_health(request)

        self.assertEqual(request.status, 200)
        self.assertIsNotNone(request.body)
        return request.body

    def test_health_reports_adapter_readiness_without_paths(self) -> None:
        """Removing adapter readiness or exposing an artifact path breaks health."""
        body = self.health(adapter_mode="shadow", ready=True, version="v1", sha256="abc")

        self.assertEqual(body.get("domain_adapter"), {
            "mode": "shadow",
            "ready": True,
            "version": "v1",
            "sha256": "abc",
            "load_error": "",
        })
        self.assertNotIn("path", json.dumps(body).lower())

    def test_health_reports_only_safe_adapter_load_error(self) -> None:
        """Artifact load failures are observable without including failure details."""
        body = self.health(
            adapter_mode="active",
            ready=False,
            load_error="adapter_artifact_invalid",
        )

        self.assertEqual(body.get("domain_adapter"), {
            "mode": "active",
            "ready": False,
            "version": "",
            "sha256": "",
            "load_error": "adapter_artifact_invalid",
        })


if __name__ == "__main__":
    unittest.main()
