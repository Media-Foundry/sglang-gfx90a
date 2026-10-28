"""CPU-only migration contract tests; no ROCm import or GPU allocation."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("dsv4_deploy", Path(__file__).with_name("deploy.py"))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg_path = self.root / "config.json"
        self.cfg_path.write_text(json.dumps({"model_path": "model", "aiter_root": "aiter", "rocm_root": "rocm"}))
        self.cfg = d.load_config(self.cfg_path)

    def fixture(self):
        root = self.root / "model"
        root.mkdir()
        (root / "config.json").write_text('{"model_type":"deepseek_v4"}')
        for file in ("tokenizer.json", "tokenizer_config.json"):
            (root / file).write_text("{}")
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "part.safetensors"}}))
        header = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        (root / "part.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + bytes(4))
        return root

    def test_paths_relative_to_config_not_cwd(self):
        self.assertEqual(self.cfg["model_path"], str(self.root / "model"))
        self.assertEqual(d.REPO / "deploy/dsv4", d.KIT)

    def test_duplicate_gpus_rejected(self):
        self.cfg_path.write_text(json.dumps({"gpus": [0] * 8}))
        with self.assertRaises(ValueError):
            d.load_config(self.cfg_path)

    def test_checkpoint_headers_and_incomplete_ignored(self):
        root = self.fixture()
        (root / "unrelated.incomplete").touch()
        self.assertEqual(d.checkpoint(root)["shards"], 1)

    def test_truncated_checkpoint_rejected(self):
        root = self.fixture()
        f = root / "part.safetensors"
        f.write_bytes(f.read_bytes()[:-1])
        with self.assertRaises(ValueError):
            d.checkpoint(root)

    def test_missing_shard_rejected(self):
        root = self.fixture()
        (root / "part.safetensors").unlink()
        with self.assertRaises(ValueError):
            d.checkpoint(root)

    def test_v41_rejected(self):
        root = self.fixture()
        (root / "config.json").write_text('{"model_type":"deepseek_v41"}')
        with self.assertRaises(ValueError):
            d.checkpoint(root)

    def test_inherited_experiments_removed(self):
        with patch.dict(os.environ, {"SGLANG_DSV4_GFX90A_DSPARK_M128_ANCHOR_ONLY_ROUTED": "1",
                                     "SPECULATIVE_DSPARK_BLOCK_SIZE": "9", "CUDA_VISIBLE_DEVICES": "0"}):
            env = d.environment(self.cfg, "ar")
        self.assertEqual(env["SGLANG_DSV4_GFX90A_DSPARK_M128_ANCHOR_ONLY_ROUTED"], "0")
        self.assertNotIn("SPECULATIVE_DSPARK_BLOCK_SIZE", env)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", env)

    def test_best_requires_artifacts(self):
        with self.assertRaises(ValueError):
            d.settings(self.cfg, "ar-best")

    def test_best_enables_only_when_explicit(self):
        self.cfg.update({k: "/new/" + k for k in d.MANIFESTS})
        best = d.settings(self.cfg, "ar-best")
        self.assertTrue(all(best[k] == "1" for k in d.BEST))
        self.assertTrue(all(d.settings(self.cfg, "ar")[k] == "0" for k in d.BEST))

    def test_strict_dspark_scope(self):
        env = d.settings(self.cfg, "dspark")
        self.assertTrue(all(env[k] == "0" for k in d.BEST + d.NATIVE))
        self.assertEqual(env["SPECULATIVE_DSPARK_BLOCK_SIZE"], "3")
        self.assertEqual(env["SGLANG_DSPARK_SYNC_ACCEPT_ACROSS_TP"], "1")
        self.assertEqual(env["SGLANG_DSV4_GFX90A_FP16_MHC_DOT"], "0")
        self.assertNotIn("64", env["CUDA_GRAPH_BS_DECODE"].split())
        self.assertTrue(all(k not in env for k in d.MANIFESTS.values()))

    def test_plan_does_not_probe_or_launch(self):
        with patch("sys.argv", ["deploy", "plan", "--config", str(self.cfg_path)]), \
             patch.object(d.subprocess, "run", side_effect=AssertionError("must not run")), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            d.main()
        self.assertEqual(json.loads(out.getvalue())["command"][-1], "serve")

    def test_launch_requires_reservation_before_checks(self):
        with patch("sys.argv", ["deploy", "launch", "--config", str(self.cfg_path)]), \
             patch.object(d, "doctor", side_effect=AssertionError("must not probe")):
            with self.assertRaisesRegex(ValueError, "预约"):
                d.main()

    def test_manifest_modified_source_rejected(self):
        source = self.root / "source.cu"
        module = self.root / "test.so"
        source.write_text("test")
        module.write_bytes(b"fixture; not imported")
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"status": "complete", "module": str(module),
                                       "module_sha256": d.sha(module), "sources": {str(source): d.sha(source)}}))
        d.validate_manifest(manifest)
        source.write_text("changed")
        with self.assertRaises(ValueError):
            d.validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
