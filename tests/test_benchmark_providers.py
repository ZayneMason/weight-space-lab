"""Recorded provider-shaped fixtures only; these tests never send HTTP."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from weight_space_lab import benchmark, benchmark_providers as providers
from weight_space_lab.evidence import digest
from tests import test_benchmark


def response(provider, name=None, params=None):
    if provider == "anthropic":
        return {"id": "fixture", "model": "fixture-revision", "content": [
            {"type": "tool_use", "id": "call-fixture", "name": name, "input": params or {}}] if name else [{"type": "text", "text": "Done"}],
            "usage": {"input_tokens": 100, "output_tokens": 10}, "stop_reason": "tool_use" if name else "end_turn"}
    return {"id": "fixture", "model": "fixture-revision", "choices": [{"message": {"role": "assistant", "content": None,
        "tool_calls": [{"id": "call-fixture", "type": "function", "function": {"name": name, "arguments": json.dumps(params or {})}}] if name else []}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10}}


class ProviderTests(unittest.TestCase):
    def test_native_formats_preserve_tool_ids_and_thinking_blocks(self):
        for provider in providers.KEYS:
            model = {"provider": provider, "model": "fixture", "settings": {}}
            conversation = providers.Conversation(model, "Fixture prompt")
            body = conversation.request(512)
            self.assertEqual(body["max_tokens"], 512)
            self.assertNotIn("temperature", body)
            self.assertEqual(len(body["tools"]), 9)
            raw = response(provider, "inspect")
            if provider == "anthropic":
                raw["content"].insert(0, {"type": "thinking", "thinking": "fixture", "signature": "fixture-signature"})
            calls, usage = conversation.receive(raw)
            conversation.results([(calls[0]["id"], {"result": {"sha256": "fixture"}})])
            self.assertEqual(usage, {"input_tokens": 100, "output_tokens": 10})
            messages = conversation.request(512)["messages"]
            if provider == "anthropic":
                self.assertEqual(messages[-2]["content"][0]["signature"], "fixture-signature")
                self.assertEqual(messages[-1]["content"][0]["tool_use_id"], "call-fixture")
            else:
                self.assertEqual(messages[-1]["tool_call_id"], "call-fixture")

    def test_both_provider_fixture_loops_freeze_before_operator_final_and_account_usage(self):
        config = test_benchmark.BenchmarkCampaignTests().config()
        config["families"] = ["routing"]
        second = deepcopy(config["models"][0]); second.update(id="fireworks-editor", provider="fireworks", model="fixture-fireworks")
        config["models"].append(second)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "suite"
            benchmark.create(root, config)
            cases = json.loads((root / "private/cases.json").read_text())
            plan = json.loads((root / "plan.json").read_text())
            counters = {provider: 0 for provider in providers.KEYS}
            def fixture(provider, body):
                counters[provider] += 1
                trial = next(t for t in plan["trials"] if t["provider"] == provider)
                run = root / trial["path"]
                self.assertFalse((run / "final").exists())
                step = counters[provider]
                if step == 1:
                    return response(provider, "inspect")
                if step == 2:
                    return response(provider, "propose", {"patch": {"base_sha256": digest(run / "editor/original.npz"), "reason": "Privileged fixture, not an API result.",
                        "operations": [{"op": "set", "tensor": "weight", "indices": [3], "values": [cases[0]["control_value"]]}]}})
                return response(provider, "feedback" if step == 3 else "freeze", {"candidate_id": "c001"})
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fixture-only", "FIREWORKS_API_KEY": "fixture-only"}), patch.object(providers, "send", fixture):
                result = benchmark.run(root, allow_paid_api=True)
            self.assertTrue(result["external_comparison_complete"])
            self.assertEqual([g["successes"] for g in result["models"]], [1, 1])
            self.assertEqual(result["api_calls_reserved"], 8)
            self.assertEqual(result["output_tokens_reserved"], 8192)
            self.assertEqual(result["usage"]["output_tokens"], 80)
            for request in root.glob("trials/*/*/provider-calls/*/request.json"):
                text = request.read_text()
                self.assertNotIn("fixture-only", text)
                self.assertNotIn("final-repair-", text)
                self.assertNotIn("positive_control", text)
            self.assertTrue(benchmark.audit(root)["verified"])

    def test_transport_failure_consumes_reservation_does_not_retry_or_drop_trial(self):
        config = test_benchmark.BenchmarkCampaignTests().config(); config["families"] = ["threshold"]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "suite"
            benchmark.create(root, config)
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fixture-only"}), patch.object(providers, "send", side_effect=RuntimeError("fixture failure")) as send:
                result = benchmark.run(root, allow_paid_api=True)
            self.assertEqual(send.call_count, 1)
            self.assertEqual(result["api_calls_reserved"], 1)
            self.assertEqual(result["models"][0]["planned"], 1)
            self.assertEqual(result["models"][0]["successes"], 0)
            self.assertEqual(result["models"][0]["statuses"], {"execution_error": 1})
            self.assertTrue(benchmark.audit(root)["verified"])

    def test_malformed_native_arguments_remain_invalid_instead_of_becoming_empty_params(self):
        c = providers.Conversation({"provider": "fireworks", "model": "fixture", "settings": {}}, "Prompt")
        raw = response("fireworks", "describe")
        raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "[]"
        calls, _ = c.receive(raw)
        self.assertEqual(calls[0]["params"], [])

    def test_usage_anomaly_blocks_all_later_paid_calls(self):
        config = test_benchmark.BenchmarkCampaignTests().config()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "suite"
            benchmark.create(root, config)
            raw = response("anthropic")
            raw["usage"]["input_tokens"] = 1000000
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fixture-only"}), patch.object(providers, "send", return_value=raw) as send:
                result = benchmark.run(root, allow_paid_api=True)
            self.assertEqual(send.call_count, 1)
            self.assertEqual(result["models"][0]["statuses"], {"execution_error": 2})
            self.assertFalse(result["models"][0]["ranking_eligible"])

    def test_malformed_tool_envelopes_do_not_erase_known_usage(self):
        for provider in providers.KEYS:
            with self.subTest(provider=provider), tempfile.TemporaryDirectory() as folder:
                config = test_benchmark.BenchmarkCampaignTests().config(); config["families"] = ["threshold"]
                config["models"][0]["provider"] = provider
                root = Path(folder) / "suite"
                benchmark.create(root, config)
                raw = response(provider, "inspect")
                if provider == "anthropic":
                    del raw["content"][0]["name"]
                else:
                    del raw["choices"][0]["message"]["tool_calls"][0]["function"]
                with patch.dict("os.environ", {providers.KEYS[provider]: "fixture-only"}), patch.object(providers, "send", return_value=raw):
                    result = benchmark.run(root, allow_paid_api=True)
                self.assertEqual(result["usage"]["input_tokens"], 100)
                self.assertEqual(result["usage"]["output_tokens"], 10)
                self.assertEqual(result["models"][0]["statuses"], {"execution_error": 1})
                self.assertTrue(benchmark.audit(root)["verified"])

    def test_excessively_nested_fireworks_arguments_are_recordable_invalid_params(self):
        c = providers.Conversation({"provider": "fireworks", "model": "fixture", "settings": {}}, "Prompt")
        raw = response("fireworks", "describe")
        args = '[' * 2000 + '0' + ']' * 2000
        raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = args
        calls, usage = c.receive(raw)
        self.assertEqual(calls[0]["params"], args)
        self.assertEqual(usage["output_tokens"], 10)
