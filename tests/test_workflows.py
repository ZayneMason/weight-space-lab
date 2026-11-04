"""Public workflow checks: offline evidence, collisions, and incompatible inputs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from weight_space_lab import workflows
from weight_space_lab.evidence import digest, verify_experiment
from weight_space_lab.legacy import helper


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_offline_demo_records_exact_patch_behavior_and_hashes(self):
        with patch("socket.create_connection", side_effect=AssertionError("Network forbidden")):
            result = workflows.run_demo(self.root / "demo")
        out = self.root / "demo"
        self.assertEqual(result["summary"]["baseline_accuracy"], 0)
        self.assertEqual(result["summary"]["candidate_accuracy"], 1)
        self.assertEqual(result["summary"]["changed_entries"], 2)
        self.assertEqual(len(result["summary"]["fixed_ids"]), 6)
        self.assertEqual(result["summary"]["regressed_ids"], [])
        receipt = json.loads((out / "candidate/receipt.json").read_text())
        self.assertEqual(digest(out / "original.npz"), receipt["base_sha256"])
        self.assertEqual(digest(out / "candidate/model.npz"), receipt["candidate_sha256"])
        self.assertIn(receipt["reason"], (out / "report.html").read_text())
        self.assertIn(receipt["base_sha256"], (out / "report.html").read_text())
        self.assertTrue(verify_experiment(out)["verified"])

    def test_collision_preserves_every_previous_artifact(self):
        out = self.root / "demo"
        workflows.run_demo(out)
        before = {str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file()}
        with self.assertRaisesRegex(FileExistsError, "new output directory"):
            workflows.run_demo(out)
        self.assertEqual(before, {str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file()})

    def test_evidence_rejects_tampering_and_path_escape(self):
        out = self.root / "demo"
        workflows.run_demo(out)
        (out / "report.html").write_text("fabricated")
        with self.assertRaisesRegex(ValueError, "report.html"):
            verify_experiment(out)
        manifest = json.loads((out / "experiment.json").read_text())
        manifest["artifacts"]["../outside"] = "0" * 64
        (out / "experiment.json").write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            verify_experiment(out)

    def test_recomputed_receipt_detects_false_changed_entry_claim(self):
        out = self.root / "demo"
        workflows.run_demo(out)
        receipt_file = out / "candidate/receipt.json"
        receipt = json.loads(receipt_file.read_text())
        receipt["changed"] = 1
        receipt_file.write_text(json.dumps(receipt))
        manifest_file = out / "experiment.json"
        manifest = json.loads(manifest_file.read_text())
        manifest["artifacts"]["candidate/receipt.json"] = digest(receipt_file)
        manifest_file.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "changed"):
            verify_experiment(out)

    def test_generic_comparison_rejects_different_inputs_actionably(self):
        out = self.root / "demo"
        workflows.run_demo(out)
        report = json.loads((out / "candidate.json").read_text())
        report["dataset_sha256"] = "different"
        (out / "candidate.json").write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "dataset_sha256"):
            helper("weight_lab").compare_reports(out / "baseline.json", out / "candidate.json")

    def test_export_has_relative_paths_and_verifiable_metadata(self):
        out = self.root / "demo"
        workflows.run_demo(out)
        export = self.root / "export"
        workflows.export_evidence(out, export)
        payload = (export / "evidence.json").read_text()
        self.assertNotIn(str(out), payload)
        self.assertFalse(list(export.rglob("*.npz")))
        self.assertTrue(verify_experiment(export)["verified"])

    def test_failed_breakout_keeps_failure_record_and_does_not_retry(self):
        with patch.object(workflows, "helper") as mocked:
            mocked.return_value.fetch.side_effect = OSError("network unavailable")
            with self.assertRaisesRegex(OSError, "network unavailable"):
                workflows.run_breakout(self.root / "failed", episodes=2, max_steps=20)
            mocked.return_value.fetch.assert_called_once()
        failure = json.loads((self.root / "failed/failure.json").read_text())
        self.assertEqual(failure["status"], "failed")
        self.assertFalse((self.root / "failed/experiment.json").exists())

    def test_root_help_and_qwen_help_do_not_run_a_gpu_job(self):
        for args in (["--help"], ["qwen", "--help"]):
            run = subprocess.run([sys.executable, str(ROOT / "lab.py"), *args], cwd=self.root,
                                 capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("usage", run.stdout.lower())

    def test_breakout_rejects_forged_metrics_and_incomplete_records(self):
        import copy
        engine = helper("reproduce_breakout")
        report = {"schema": "weight-space-breakout-v1", "game": "breakout", "episodes": 2,
                  "settings": {"max_steps": 20}, "versions": {}, "checkpoint_sha256": "x",
                  "evaluator_sha256": "y", "mean": 2.0, "se": 1.0,
                  "ci95_normal": [.04, 3.96], "mean_length": 20.,
                  "results": [{"episode": i, "score": 1 + 2*i, "steps": 20, "truncated": True} for i in range(2)]}
        engine.validate_report(report)
        for field, value in (("mean", 999), ("se", None), ("results", report["results"][:1])):
            wrong = copy.deepcopy(report); wrong[field] = value
            with self.assertRaises(ValueError):
                engine.validate_report(wrong)

    def test_full_breakout_scope_requires_full_protocol_before_fetch(self):
        with patch.object(workflows, "helper") as mocked:
            with self.assertRaisesRegex(ValueError, "exactly 1000"):
                workflows.run_breakout(self.root / "invalid", mode="full", episodes=8)
            mocked.assert_not_called()
        self.assertFalse((self.root / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
