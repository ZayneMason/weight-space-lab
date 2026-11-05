"""Scripted transport checks; these are not external-agent research trials."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from weight_space_lab import controller, editor_session
from weight_space_lab.evidence import digest


class EditorSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "run"
        controller.create(self.run, proposal_budget=2, feedback_budget=1, editor_id="scripted-test-editor")
        self.metadata = {"editor_id": "scripted-test-editor", "provider": "none",
                         "model": "scripted-validation", "model_revision": "test-v1",
                         "settings": {"tool_policy": "scripted-test-only"},
                         "prompt": "Prewritten transport verification. Not agent discovery."}

    def request(self, i, method, params=None):
        return {"id": i, "method": method, "params": params or {}}

    def patch(self):
        return {"base_sha256": digest(self.run / "editor/original.npz"),
                "reason": "Prewritten positive control for bridge validation only.",
                "operations": [{"op": "set", "tensor": "bias", "indices": [1], "values": [0]}]}

    def serve(self, packets, session="s1", limit=32):
        wire = b''.join((json.dumps(p).encode() + b'\n') if isinstance(p, dict) else p for p in packets)
        output = io.StringIO()
        editor_session.serve(self.run, session, self.metadata, io.BytesIO(wire), output, max_requests=limit)
        return [json.loads(line) for line in output.getvalue().splitlines()]

    def test_scripted_editor_freezes_without_exposing_final_or_positive_control(self):
        rows = self.serve([self.request(1, "describe"), self.request(2, "inspect"),
                          self.request(3, "propose", {"patch": self.patch()}),
                          self.request(4, "feedback", {"candidate_id": "c001"}),
                          self.request(5, "freeze", {"candidate_id": "c001"})])
        self.assertEqual(controller.status(self.run)["phase"], "frozen")
        self.assertFalse((self.run / "final").exists())
        text = json.dumps(rows)
        self.assertNotIn(str(self.run), text)
        self.assertNotIn("positive_control", text)
        self.assertNotIn("heldout_seed", text)
        self.assertNotIn("final-repair-", text)
        self.assertEqual(rows[-1]["reason"], "frozen")
        self.assertTrue(editor_session.audit(self.run, "s1")["verified"])
        controller.finalize(self.run)  # Operator action, outside the editor transport.
        self.assertTrue(editor_session.audit(self.run, "s1")["verified"])

    def test_finalization_and_arbitrary_paths_are_not_available_methods(self):
        rows = self.serve([self.request(1, "finalize"),
                          self.request(2, "tensor", {"name": "../../private/heldout.json"}),
                          self.request(3, "inspect", {"path": "private/heldout.json"})])
        errors = [r for r in rows if "error" in r]
        self.assertEqual(len(errors), 3)
        self.assertFalse((self.run / "final").exists())
        self.assertEqual(controller.status(self.run)["feedback_used"], 0)
        self.assertTrue(editor_session.audit(self.run, "s1")["verified"])

    def test_bad_patch_and_repeated_feedback_preserve_controller_budgets(self):
        bad = self.patch(); bad["operations"][0]["tensor"] = "weight"
        rows = self.serve([self.request(1, "propose", {"patch": bad}),
                          self.request(2, "propose", {"patch": self.patch()}),
                          self.request(3, "feedback", {"candidate_id": "c002"}),
                          self.request(4, "feedback", {"candidate_id": "c002"})])
        self.assertEqual(controller.status(self.run)["proposals_used"], 2)
        self.assertEqual(controller.status(self.run)["feedback_used"], 1)
        self.assertEqual(rows[-2]["error"]["code"], "feedback_budget_exhausted")
        self.assertTrue(editor_session.audit(self.run, "s1")["verified"])

    def test_malformed_and_duplicate_requests_consume_transport_allowance(self):
        rows = self.serve([b'not json\n', self.request(1, "status"), self.request(1, "status"),
                          self.request(2, "propose", {"patch": self.patch()})], limit=3)
        self.assertEqual(rows[-1]["reason"], "request_limit")
        self.assertEqual(controller.status(self.run)["proposals_used"], 0)
        self.assertEqual(editor_session.audit(self.run, "s1")["requests"], 3)

    def test_oversized_input_terminates_without_reading_unbounded_data(self):
        rows = self.serve([b'x' * (editor_session.MAX_LINE_BYTES + 10)])
        self.assertEqual(rows[-1]["reason"], "oversized_request")
        self.assertEqual(rows[-2]["error"]["code"], "request_too_large")
        self.assertEqual(editor_session.audit(self.run, "s1")["requests"], 1)

    def test_recorded_metadata_and_transcript_tampering_is_detected(self):
        self.serve([self.request(1, "status")])
        path = self.run / "sessions/s1/events.jsonl"
        with path.open("a") as stream:
            stream.write('{"forged":true}\n')
        with self.assertRaises(ValueError):
            editor_session.audit(self.run, "s1")

    def test_credentials_are_not_supported_metadata_and_ids_cannot_escape(self):
        self.metadata["api_key"] = "not-a-real-key"
        with self.assertRaises(ValueError):
            self.serve([])
        del self.metadata["api_key"]
        with self.assertRaises(ValueError):
            self.serve([], session="../escape")

    def test_session_name_collision_preserves_previous_transcript(self):
        self.serve([self.request(1, "status")])
        before = digest(self.run / "sessions/s1/events.jsonl")
        with self.assertRaises(FileExistsError):
            self.serve([])
        self.assertEqual(before, digest(self.run / "sessions/s1/events.jsonl"))

    def test_restart_keeps_global_quotas_and_pins_editor_configuration(self):
        self.serve([self.request(1, "propose", {"patch": self.patch()})])
        self.serve([self.request(1, "feedback", {"candidate_id": "c001"})], session="s2")
        self.assertEqual(controller.status(self.run)["proposals_used"], 1)
        self.assertEqual(controller.status(self.run)["feedback_used"], 1)
        self.assertTrue(editor_session.audit(self.run, "s1")["verified"])
        self.assertTrue(editor_session.audit(self.run, "s2")["verified"])
        self.metadata["model"] = "different-editor"
        with self.assertRaisesRegex(ValueError, "pinned"):
            self.serve([], session="s3")

    def test_out_of_band_controller_mutation_invalidates_session(self):
        session = editor_session.Session(self.run, "s1", self.metadata)
        try:
            session.process(json.dumps(self.request(1, "status")).encode())
            path = Path(self.temp.name) / "outside-patch.json"
            path.write_text(json.dumps(self.patch()))
            controller.submit(self.run, path)
            response = session.process(json.dumps(self.request(2, "status")).encode())
            self.assertEqual(response["error"]["code"], "controller_changed")
        finally:
            session.finish()
        with self.assertRaisesRegex(ValueError, "outside"):
            editor_session.audit(self.run, "s1")

    def test_invalid_encoding_keys_and_method_types_are_recorded_errors(self):
        rows = self.serve([b'\xff\n', b'{"id":1,"id":2,"method":"status","params":{}}\n',
                          self.request(3, ["status"]), self.request(4, {"tool": "status"})])
        self.assertEqual(len([r for r in rows if "error" in r]), 4)
        self.assertTrue(editor_session.audit(self.run, "s1")["verified"])

    def test_only_one_live_session_can_write_and_lock_releases_at_eof(self):
        session = editor_session.Session(self.run, "s1", self.metadata)
        try:
            with self.assertRaisesRegex(RuntimeError, "active"):
                editor_session.Session(self.run, "s2", self.metadata)
        finally:
            session.finish()
        self.serve([], session="s2")
        self.assertTrue(editor_session.audit(self.run, "s2")["verified"])

    def test_interleaved_operator_feedback_cannot_be_attributed_to_invalid_request(self):
        patch_file = Path(self.temp.name) / "outside-patch.json"
        patch_file.write_text(json.dumps(self.patch()))
        controller.submit(self.run, patch_file)
        session = editor_session.Session(self.run, "s1", self.metadata)
        original = editor_session._state
        calls = 0
        def interleave(run):
            nonlocal calls
            calls += 1
            if calls == 2:
                controller.feedback(self.run, "c001")
            return original(run)
        try:
            with patch.object(editor_session, "_state", interleave):
                response = session.process(json.dumps(self.request(1, "feedback", {"candidate_id": "invalid"})).encode())
            self.assertEqual(response["error"]["code"], "controller_changed")
        finally:
            session.finish()
        with self.assertRaises(ValueError):
            editor_session.audit(self.run, "s1")

    def test_controller_checks_expected_anchor_under_mutation_lock(self):
        patch_file = Path(self.temp.name) / "outside-patch.json"
        patch_file.write_text(json.dumps(self.patch()))
        controller.submit(self.run, patch_file)
        session = editor_session.Session(self.run, "s1", self.metadata)
        original = controller.feedback
        def interleave(run, candidate_id, **kwargs):
            original(run, "c001")  # Changes state after dispatch begins, before its lock.
            return original(run, candidate_id, **kwargs)
        try:
            with patch.object(controller, "feedback", interleave):
                response = session.process(json.dumps(self.request(1, "feedback", {"candidate_id": "c001"})).encode())
            self.assertEqual(response["error"]["code"], "controller_changed")
            self.assertEqual(controller.status(self.run)["feedback_used"], 1)
        finally:
            session.finish()
        with self.assertRaises(ValueError):
            editor_session.audit(self.run, "s1")

    def test_interleaved_operator_freeze_invalidates_editor_session(self):
        patch_file = Path(self.temp.name) / "outside-patch.json"
        patch_file.write_text(json.dumps(self.patch()))
        controller.submit(self.run, patch_file)
        controller.feedback(self.run, "c001")
        session = editor_session.Session(self.run, "s1", self.metadata)
        original = editor_session._state
        calls = 0
        def interleave(run):
            nonlocal calls
            calls += 1
            if calls == 2:
                controller.freeze(self.run, "c001")
            return original(run)
        try:
            with patch.object(editor_session, "_state", interleave):
                response = session.process(json.dumps(self.request(1, "freeze", {"candidate_id": "c002"})).encode())
            self.assertEqual(response["error"]["code"], "controller_changed")
            self.assertEqual(controller.status(self.run)["phase"], "frozen")
        finally:
            session.finish()
        with self.assertRaises(ValueError):
            editor_session.audit(self.run, "s1")
