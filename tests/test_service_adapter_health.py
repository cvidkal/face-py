"""Health endpoint contract for the optional identity domain adapter."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from module.face.domain_adapter import DomainAdapterRuntime
from service.serve import Handler


_REPO = Path(__file__).resolve().parent.parent


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
    def render_compose(self, *, dev_mode: str = "off", dev_path: str | None = None) -> str:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dev_env = root / "dev.env"
            prod_env = root / "prod.env"
            shutil.copyfile(_REPO / "deploy/env/dev.env.example", dev_env)
            shutil.copyfile(_REPO / "deploy/env/prod.env.example", prod_env)
            dev_env.write_text(
                dev_env.read_text(encoding="utf-8").replace(
                    "FACE_DOMAIN_ADAPTER_MODE=off",
                    f"FACE_DOMAIN_ADAPTER_MODE={dev_mode}",
                )
                + (f"\nFACE_DOMAIN_ADAPTER_PATH={dev_path}\n" if dev_path else ""),
                encoding="utf-8",
            )
            compose = root / "docker-compose.yml"
            compose.write_text(
                (_REPO / "docker-compose.yml").read_text(encoding="utf-8")
                .replace("deploy/env/dev.env", str(dev_env))
                .replace("deploy/env/prod.env", str(prod_env)),
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment.pop("FACE_DOMAIN_ADAPTER_MODE", None)
            environment.pop("FACE_DOMAIN_ADAPTER_PATH", None)
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose), "config"],
                cwd=_REPO,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

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

    def test_compose_uses_adapter_mode_and_path_from_its_service_env_file(self) -> None:
        """A shadow operator setting must reach the container without shell interpolation."""
        body = self.render_compose(
            dev_mode="shadow",
            dev_path="/models/identity-domain-adapter/candidate.onnx",
        )

        self.assertIn('FACE_DOMAIN_ADAPTER_MODE: shadow', body)
        self.assertIn(
            'FACE_DOMAIN_ADAPTER_PATH: /models/identity-domain-adapter/candidate.onnx',
            body,
        )

    def test_compose_examples_default_the_adapter_to_off(self) -> None:
        """The checked-in examples preserve raw face behavior until an operator opts in."""
        body = self.render_compose()

        self.assertEqual(body.count('FACE_DOMAIN_ADAPTER_MODE: "off"'), 2)

    def test_default_adapter_mount_directory_is_tracked(self) -> None:
        """The default bind source exists before Docker can create it as root."""
        self.assertTrue((_REPO / "models/identity-domain-adapter").is_dir())


if __name__ == "__main__":
    unittest.main()
