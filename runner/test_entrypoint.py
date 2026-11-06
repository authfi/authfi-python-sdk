"""The runner's entrypoint refuses a pod that cannot sign in as an instance. Offline; needs no image and no network.

python -I runner/test_entrypoint.py
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load(framework: str = "langgraph"):
    os.environ["FW_FRAMEWORK"] = framework
    import entrypoint

    return importlib.reload(entrypoint)


class Entrypoint(unittest.TestCase):
    def setUp(self) -> None:
        self.tok = Path(tempfile.mkdtemp()) / "token"
        self.tok.write_text("x")
        self.good = {
            "AUTHFI_ISSUER": "https://probe-c1.np.authfi.io",
            "AUTHFI_SERVICE_ACCOUNT_TOKEN_FILE": str(self.tok),
            "AUTHFI_AGENT_ID": "agt_" + "ab" * 16,
            "AUTHFI_CONFIG_VERSION": "3",
            "AUTHFI_CONFIG_DIGEST": "sha256:" + "ab" * 32,
        }

    def test_a_complete_pod_has_no_problem(self) -> None:
        self.assertEqual(load().problems(self.good), [])

    def test_each_empty_value_is_named_and_none_is_printed(self) -> None:
        ep = load()
        for name in ep.REQUIRED:
            env = dict(self.good, **{name: ""})
            found = ep.problems(env)
            self.assertTrue(any(name in line for line in found), (name, found))
            self.assertFalse(any(self.good["AUTHFI_AGENT_ID"] in line for line in found))

    def test_a_key_file_is_refused_so_the_pod_cannot_fall_back(self) -> None:
        found = load().problems(dict(self.good, AUTHFI_AGENT_KEY_FILE="/k"))
        self.assertTrue(any("holds no key" in line for line in found), found)

    def test_a_missing_token_file_a_bad_version_and_a_bad_digest(self) -> None:
        ep = load()
        self.assertTrue(ep.problems(dict(self.good, AUTHFI_SERVICE_ACCOUNT_TOKEN_FILE="/nowhere/token")))
        self.assertTrue(ep.problems(dict(self.good, AUTHFI_CONFIG_VERSION="v3")))
        self.assertTrue(ep.problems(dict(self.good, AUTHFI_CONFIG_DIGEST="md5:00")))

    def test_an_unknown_framework_is_refused(self) -> None:
        self.assertTrue(load("claude-agent-sdk").problems(self.good), "claude-agent-sdk has no instance path yet")
        load()


class Image(unittest.TestCase):
    here = Path(__file__).resolve().parent

    def test_every_framework_script_is_in_the_image_context(self) -> None:
        for rel in load().SCRIPT.values():
            self.assertTrue((self.here / "scripts" / rel).is_file(), rel)

    def test_the_copied_scripts_are_the_ones_that_were_recorded(self) -> None:
        recorded = [line.split(None, 1) for line in (self.here / "scripts.sha256").read_text().splitlines() if line]
        self.assertTrue(recorded)
        for digest, name in recorded:
            data = (self.here / "scripts" / name.strip().lstrip("*")).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), digest, name)

    def test_the_image_runs_as_non_root_and_bakes_no_credential(self) -> None:
        text = (self.here / "Dockerfile").read_text()
        self.assertIn("USER 65532:65532", text)
        for forbidden in ("AUTHFI_AGENT_KEY", "AUTHFI_AGENT_PRIVATE_KEY", "ANTHROPIC_API_KEY", "ARG TOKEN"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
