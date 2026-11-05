"""Protocol gates for independent experiments; proposals here are hand-authored tests."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from weight_space_lab import controller
from weight_space_lab.evidence import digest


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "run"
        controller.create(self.run, feedback_budget=1, proposal_budget=2)

    def proposal(self):
        path = Path(self.temp.name) / "patch.json"
        path.write_text(json.dumps({"base_sha256": digest(self.run / "editor/original.npz"),
            "reason": "Hand-authored controller verification; not agent discovery.",
            "operations": [{"op": "set", "tensor": "bias", "indices": [1], "values": [0]}]}))
        return path

    def test_budget_freeze_and_matched_final_evaluation(self):
        source_hash = digest(self.run / "editor/original.npz")
        candidate = controller.submit(self.run, self.proposal())["candidate_id"]
        feedback = controller.feedback(self.run, candidate)
        self.assertGreater(feedback["candidate_accuracy"], feedback["baseline_accuracy"])
        with self.assertRaisesRegex(ValueError, "feedback budget"):
            controller.feedback(self.run, candidate)
        with self.assertRaisesRegex(ValueError, "Freeze"):
            controller.finalize(self.run)
        controller.freeze(self.run, candidate)
        with self.assertRaisesRegex(ValueError, "editing"):
            controller.submit(self.run, self.proposal())
        with self.assertRaisesRegex(ValueError, "editing"):
            controller.feedback(self.run, candidate)
        result = controller.finalize(self.run)
        self.assertEqual(result["comparison"]["candidate_accuracy"], 1)
        self.assertEqual(result["comparison"]["regressed_ids"], [])
        self.assertEqual(digest(self.run / "editor/original.npz"), source_hash)
        self.assertTrue(controller.audit(self.run)["verified"])
        with self.assertRaisesRegex(ValueError, "one-time"):
            controller.finalize(self.run)

    def test_rejected_proposals_consume_budget_and_preserve_failure(self):
        path = self.proposal()
        bad = json.loads(path.read_text()); bad["operations"][0]["tensor"] = "weight"
        path.write_text(json.dumps(bad))
        for _ in range(2):
            with self.assertRaises(ValueError):
                controller.submit(self.run, path)
        with self.assertRaisesRegex(ValueError, "proposal budget"):
            controller.submit(self.run, self.proposal())
        state = controller.status(self.run)
        self.assertEqual(state["proposals_used"], 2)
        self.assertEqual(len(list((self.run / "proposals").glob("*/failure.json"))), 2)
        self.assertTrue(controller.audit(self.run)["verified"])

    def test_frozen_checkpoint_or_input_tampering_blocks_final(self):
        candidate = controller.submit(self.run, self.proposal())["candidate_id"]
        controller.feedback(self.run, candidate)
        controller.freeze(self.run, candidate)
        with (self.run / "frozen/model.npz").open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "hash"):
            controller.finalize(self.run)

    def test_final_set_never_appears_in_feedback_and_final_cannot_select_candidate(self):
        candidate = controller.submit(self.run, self.proposal())["candidate_id"]
        feedback = controller.feedback(self.run, candidate)
        payload = json.dumps(feedback)
        self.assertNotIn("heldout", payload)
        self.assertNotIn("final-", payload)
        with self.assertRaises(ValueError):
            controller.freeze(self.run, "unknown")
        self.assertEqual(controller.status(self.run)["phase"], "editing")

    def test_concurrent_controller_lock_refuses_second_writer(self):
        lock = self.run / ".controller.lock"
        lock.write_text("another action")
        with self.assertRaisesRegex(RuntimeError, "locked"):
            controller.submit(self.run, self.proposal())
        self.assertEqual(controller.status(self.run)["proposals_used"], 0)

    def test_regressions_fail_success_rule_even_when_repairs_succeed(self):
        path = self.proposal()
        proposal = json.loads(path.read_text()); proposal["operations"][0]["values"] = [.6]
        path.write_text(json.dumps(proposal))
        candidate = controller.submit(self.run, path)["candidate_id"]
        controller.feedback(self.run, candidate)
        controller.freeze(self.run, candidate)
        result = controller.finalize(self.run)
        self.assertEqual(result["repair_group_accuracy"], 1)
        self.assertEqual(len(result["protected_group_regressed_ids"]), 64)
        self.assertFalse(result["success_rule_met"])

    def test_feedback_failure_consumes_budget_and_records_error(self):
        candidate = controller.submit(self.run, self.proposal())["candidate_id"]
        with patch.object(controller, "helper") as mocked:
            mocked.return_value.evaluate_mlp.side_effect = ValueError("evaluation failed")
            with self.assertRaisesRegex(ValueError, "evaluation failed"):
                controller.feedback(self.run, candidate)
        self.assertEqual(controller.status(self.run)["feedback_used"], 1)
        self.assertTrue((self.run / "feedback/f001/failure.json").exists())
        self.assertTrue(controller.audit(self.run)["verified"])

    def test_counter_reset_cannot_create_additional_budget(self):
        candidate = controller.submit(self.run, self.proposal())["candidate_id"]
        controller.feedback(self.run, candidate)
        state_file = self.run / "state.json"
        state = json.loads(state_file.read_text()); state["feedback_used"] = 0
        state_file.write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "counters"):
            controller.feedback(self.run, candidate)

    def test_freeze_failure_is_journaled_before_partial_files_and_is_terminal(self):
        candidate = controller.submit(self.run, self.proposal())["candidate_id"]
        controller.feedback(self.run, candidate)
        with patch.object(controller.shutil, "copyfile", side_effect=OSError("disk unavailable")):
            with self.assertRaisesRegex(OSError, "disk unavailable"):
                controller.freeze(self.run, candidate)
        state = json.loads((self.run / "state.json").read_text())
        self.assertEqual(state["phase"], "freeze_failed")
        self.assertEqual([e["outcome"] for e in state["events"] if e["action"] == "freeze"], ["started", "failed"])
        self.assertTrue((self.run / "freeze-failure.json").exists())
        self.assertFalse(controller.audit(self.run)["final_complete"])
        with self.assertRaises(ValueError):
            controller.finalize(self.run)

    def test_no_selection_is_recorded_without_final_feedback(self):
        result = controller.close(self.run, "Editor abstained before using its budget")
        self.assertFalse(result["heldout_evaluated"])
        self.assertFalse(result["budget_exhausted"])
        self.assertEqual(controller.status(self.run)["phase"], "closed_without_final")
        self.assertTrue(controller.audit(self.run)["verified"])
        with self.assertRaises(ValueError):
            controller.finalize(self.run)

    def test_final_directory_failure_consumes_one_time_attempt(self):
        candidate = controller.submit(self.run, self.proposal())["candidate_id"]
        controller.feedback(self.run, candidate)
        controller.freeze(self.run, candidate)
        original = Path.mkdir
        def fail_final(path, *args, **kwargs):
            if path == self.run / "final":
                raise OSError("directory unavailable")
            return original(path, *args, **kwargs)
        with patch.object(Path, "mkdir", fail_final):
            with self.assertRaisesRegex(OSError, "directory unavailable"):
                controller.finalize(self.run)
        state = json.loads((self.run / "state.json").read_text())
        self.assertEqual(state["phase"], "final_failed")
        self.assertEqual([e["outcome"] for e in state["events"] if e["action"] == "final"], ["started", "failed"])
        self.assertTrue((self.run / "final-failure.json").exists())
        self.assertTrue(controller.audit(self.run)["verified"])
        with self.assertRaisesRegex(ValueError, "one-time"):
            controller.finalize(self.run)

    def test_proposal_directory_failure_still_consumes_reserved_budget(self):
        original = Path.mkdir
        def fail_candidate(path, *args, **kwargs):
            if path.name == "c001":
                raise OSError("directory unavailable")
            return original(path, *args, **kwargs)
        with patch.object(Path, "mkdir", fail_candidate):
            with self.assertRaisesRegex(OSError, "directory unavailable"):
                controller.submit(self.run, self.proposal())
        self.assertEqual(controller.status(self.run)["proposals_used"], 1)
        self.assertTrue((self.run / "proposals/c001-failure.json").exists())
        self.assertTrue(controller.audit(self.run)["verified"])

    def test_feedback_directory_failure_still_consumes_reserved_budget(self):
        candidate = controller.submit(self.run, self.proposal())["candidate_id"]
        original = Path.mkdir
        def fail_feedback(path, *args, **kwargs):
            if path.name == "f001":
                raise OSError("directory unavailable")
            return original(path, *args, **kwargs)
        with patch.object(Path, "mkdir", fail_feedback):
            with self.assertRaisesRegex(OSError, "directory unavailable"):
                controller.feedback(self.run, candidate)
        self.assertEqual(controller.status(self.run)["feedback_used"], 1)
        self.assertTrue((self.run / "feedback/f001-failure.json").exists())
        self.assertTrue(controller.audit(self.run)["verified"])


if __name__ == "__main__":
    unittest.main()
