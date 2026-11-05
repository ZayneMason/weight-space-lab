"""Benchmark controls and matched instances, never API model research results."""
import json
from pathlib import Path
import tempfile
import unittest

from weight_space_lab import benchmark_tasks, controller, editor_session
from weight_space_lab.evidence import digest, write_json


class BenchmarkTaskTests(unittest.TestCase):
    def test_all_families_have_nontrivial_baselines_and_feasible_one_entry_controls(self):
        with tempfile.TemporaryDirectory() as folder:
            for family in benchmark_tasks.FAMILIES:
                with self.subTest(family=family):
                    spec = benchmark_tasks.generate(family, 11, 91)
                    run = Path(folder) / family
                    controller.create(run, _task=spec)
                    path = run / "control.json"
                    write_json(path, {"base_sha256": digest(run / "editor/original.npz"), "reason": "Feasible scripted test control.",
                        "operations": [{"op": "set", "tensor": spec["editable"]["tensor"],
                            "indices": spec["editable"]["flat_indices"], "values": [spec["control_value"]]}]})
                    candidate = controller.submit(run, path)["candidate_id"]
                    feedback = controller.feedback(run, candidate)
                    self.assertEqual(feedback["baseline_accuracy"], .5)
                    self.assertEqual(feedback["candidate_accuracy"], 1)
                    controller.freeze(run, candidate)
                    result = controller.finalize(run)
                    self.assertTrue(result["success_rule_met"])
                    self.assertEqual(result["comparison"]["regressed_ids"], [])
                    self.assertTrue(controller.audit(run)["verified"])

    def test_practice_and_final_draws_are_separate_and_matched_by_seed(self):
        a = benchmark_tasks.generate("hidden", 7, 31)
        b = benchmark_tasks.generate("hidden", 7, 31)
        c = benchmark_tasks.generate("hidden", 7, 32)
        self.assertEqual(a["practice"], b["practice"])
        self.assertEqual(a["heldout"], b["heldout"])
        self.assertNotEqual(a["heldout"], c["heldout"])
        self.assertEqual(len(a["practice"]["examples"]), 32)
        self.assertEqual(len(a["heldout"]["examples"]), 256)
        practice_inputs = {tuple(row["x"]) for row in a["practice"]["examples"]}
        self.assertFalse(any(tuple(row["x"]) in practice_inputs for row in a["heldout"]["examples"]))


class BenchmarkCampaignTests(unittest.TestCase):
    def config(self):
        return {"models": [{"id": "test-editor", "provider": "anthropic", "model": "fixture-model",
                    "model_revision": "test-only", "settings": {}, "input_usd_per_million": 2, "output_usd_per_million": 10}],
                "families": ["threshold", "hidden"], "task_seeds": [11],
                "budgets": {"proposals": 5, "feedback_calls": 3, "tool_requests": 32, "api_calls": 8, "max_output_tokens": 1024, "max_request_bytes": 65536},
                "limits": {"max_total_api_calls": 16, "max_total_output_tokens": 16384, "max_estimated_usd": 10}}

    def test_control_suite_preserves_matched_instances_and_counts_every_planned_trial(self):
        from weight_space_lab import benchmark
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "suite"
            benchmark.create(root, self.config())
            before = benchmark.summarize(root)
            self.assertFalse(before["external_comparison_complete"])
            self.assertEqual(before["models"][0]["planned"], 2)
            benchmark.run(root)
            result = benchmark.summarize(root)
            self.assertEqual(result["controls"][0]["successes"], 2)
            self.assertEqual(result["controls"][1]["successes"], 0)
            self.assertEqual(result["controls"][1]["statuses"], {"completed": 2})
            self.assertEqual(result["controls"][1]["mean_accuracy_completed_only"], .5)
            self.assertEqual(result["api_calls_reserved"], 0)
            self.assertFalse(result["external_comparison_complete"])
            self.assertTrue(benchmark.audit(root)["verified"])
            plan = json.loads((root / "plan.json").read_text())
            hashes = [digest(root / trial["path"] / "private/heldout.json") for trial in plan["trials"] if trial["case_id"] == "hidden-11"]
            self.assertEqual(len(set(hashes)), 1)
            old = digest(root / "plan.json")
            with self.assertRaises(FileExistsError):
                benchmark.create(root, self.config())
            self.assertEqual(old, digest(root / "plan.json"))

    def test_live_api_requires_explicit_opt_in_and_keys_before_any_call(self):
        from weight_space_lab import benchmark
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "suite"
            benchmark.create(root, self.config())
            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(ValueError, "ANTHROPIC_API_KEY"):
                    benchmark.run(root, allow_paid_api=True)
            self.assertEqual(benchmark.summarize(root)["api_calls_reserved"], 0)

    def test_rejects_duplicate_editors_seeds_and_inadequate_global_token_budget(self):
        from weight_space_lab import benchmark
        with tempfile.TemporaryDirectory() as folder:
            config = self.config(); config["models"] *= 2
            with self.assertRaises(ValueError):
                benchmark.create(Path(folder) / "duplicate", config)
            config = self.config(); config["task_seeds"] *= 2
            with self.assertRaises(ValueError):
                benchmark.create(Path(folder) / "seeds", config)
            config = self.config(); config["limits"]["max_total_output_tokens"] = 1
            with self.assertRaises(ValueError):
                benchmark.create(Path(folder) / "budget", config)

    def test_modified_completed_result_cannot_be_reported_as_valid(self):
        from weight_space_lab import benchmark
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "suite"
            config = self.config(); config["families"] = ["threshold"]
            benchmark.create(root, config)
            benchmark.run(root)
            result = root / "trials/control-feasible/threshold-11/final/result.json"
            result.write_text('{}')
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                benchmark.summarize(root)

    def test_missing_outcome_cannot_hide_a_started_trial_and_budget_reset_is_detected(self):
        from weight_space_lab import benchmark
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "suite"
            config = self.config(); config["families"] = ["threshold"]
            benchmark.create(root, config)
            benchmark.run(root)
            state_file = root / "state.json"
            state = json.loads(state_file.read_text()); state["api_calls_reserved"] = 1
            state_file.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "counters"):
                benchmark.summarize(root)
            state["api_calls_reserved"] = 0
            del state["outcomes"]["control-feasible/threshold-11"]
            state_file.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "Missing outcome"):
                benchmark.summarize(root)
