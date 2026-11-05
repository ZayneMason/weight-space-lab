"""CLI transport fixtures; no signed-in model is invoked by unit tests."""
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

from weight_space_lab import benchmark, benchmark_cli
from weight_space_lab.evidence import digest
from tests import test_benchmark


class CliTests(unittest.TestCase):
    def test_commands_disable_native_tools_and_secrets_are_not_inherited(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            model = {"provider": "codex-cli", "model": "fixture-model", "settings": {}}
            command = benchmark_cli.command(model, path, executable="codex.exe")
            self.assertIn("--ignore-user-config", command)
            self.assertIn("--ignore-rules", command)
            self.assertIn("--ephemeral", command)
            self.assertIn("shell_tool", command)
            self.assertIn("read-only", command)
            model["provider"] = "claude-cli"
            command = benchmark_cli.command(model, path, executable="claude.exe")
            self.assertIn("--tools", command)
            self.assertEqual(command[command.index("--tools")+1], "")
            self.assertIn("--strict-mcp-config", command)
            self.assertIn("--no-session-persistence", command)
            with patch.dict("os.environ", {"FIREWORKS_API_KEY": "fixture", "ANTHROPIC_API_KEY": "fixture", "OPENAI_API_KEY": "fixture",
                "ANTHROPIC_AUTH_TOKEN": "fixture", "ANTHROPIC_CUSTOM_HEADERS": "Authorization: fixture", "HOUSEHOLD_AGENT_TOKEN": "fixture", "NODE_OPTIONS": "fixture"}):
                env = benchmark_cli.environment()
            self.assertFalse(any(k in env for k in ("FIREWORKS_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                "ANTHROPIC_CUSTOM_HEADERS", "HOUSEHOLD_AGENT_TOKEN", "NODE_OPTIONS")))

    def test_codex_event_log_preserves_usage_and_rejects_native_tool_execution(self):
        action = {"actions": [{"method": "inspect", "params": {}}]}
        events = [{"type": "thread.started", "thread_id": "fixture"}, {"type": "turn.started"},
                  {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(action)}},
                  {"type": "turn.completed", "usage": {"input_tokens": 100, "cached_input_tokens": 10, "output_tokens": 20}}]
        result = benchmark_cli.decode("codex-cli", '\n'.join(json.dumps(e) for e in events))
        self.assertEqual(result["usage"]["completion_tokens"], 20)
        self.assertEqual(result["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "inspect")
        events.insert(2, {"type": "item.completed", "item": {"type": "command_execution", "command": "fixture"}})
        with self.assertRaisesRegex(ValueError, "Native"):
            benchmark_cli.decode("codex-cli", '\n'.join(json.dumps(e) for e in events))

    def test_claude_structured_output_and_missing_usage_are_explicit(self):
        raw = {"type": "result", "is_error": False, "structured_output": {"actions": [{"method": "status", "params": {}}]},
               "usage": {"input_tokens": 80, "output_tokens": 10, "cache_read_input_tokens": 20}, "modelUsage": {"fixture-revision": {}}}
        result = benchmark_cli.decode("claude-cli", json.dumps(raw))
        self.assertEqual(result["usage"]["prompt_tokens"], 100)
        del raw["usage"]
        with self.assertRaisesRegex(ValueError, "usage"):
            benchmark_cli.decode("claude-cli", json.dumps(raw))

    def test_cli_opt_in_is_independent_of_paid_api_opt_in(self):
        config = test_benchmark.BenchmarkCampaignTests().config()
        config["models"][0].update(provider="codex-cli", input_usd_per_million=0, output_usd_per_million=0)
        config["families"] = ["threshold"]
        config["limits"]["max_total_cli_calls"] = 8
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "suite"
            benchmark.create(root, config)
            with patch.object(benchmark_cli, "send", side_effect=AssertionError("No CLI launch without opt-in")):
                result = benchmark.run(root)
            self.assertEqual(result["cli_calls_reserved"], 0)
            self.assertEqual(result["api_calls_reserved"], 0)
            self.assertEqual(result["models"][0]["statuses"], {"pending": 1})

    def test_cli_fixture_rounds_freeze_and_keep_allowance_usage_separate(self):
        from copy import deepcopy
        config = test_benchmark.BenchmarkCampaignTests().config()
        config["models"][0].update(provider="codex-cli", input_usd_per_million=0, output_usd_per_million=0)
        other = deepcopy(config["models"][0]); other.update(id="claude-editor", provider="claude-cli", model="fixture-claude")
        config["models"].append(other)
        config["families"] = ["routing"]
        config["limits"]["max_total_cli_calls"] = 16
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "suite"
            benchmark.create(root, config)
            plan = json.loads((root / "plan.json").read_text())
            case = json.loads((root / "private/cases.json").read_text())[0]
            calls = {p: 0 for p in benchmark_cli.PROVIDERS}
            def send(model, body, target):
                provider = model["provider"]
                calls[provider] += 1
                trial = next(t for t in plan["trials"] if t["provider"] == provider)
                run = root / trial["path"]
                self.assertFalse((run / "final").exists())
                step = calls[provider]
                method, params = "inspect", {}
                if step == 2:
                    method, params = "propose", {"patch": {"base_sha256": digest(run / "editor/original.npz"), "reason": "Privileged CLI fixture only.",
                        "operations": [{"op": "set", "tensor": "weight", "indices": [3], "values": [case["control_value"]]}]}}
                elif step > 2:
                    method, params = "feedback" if step == 3 else "freeze", {"candidate_id": "c001"}
                return {"model": "fixture", "usage": {"prompt_tokens": 100, "completion_tokens": 10}, "choices": [{"message": {
                    "role": "assistant", "tool_calls": [{"id": "fixture", "function": {"name": method, "arguments": json.dumps(params)}}]}}]}
            with patch.object(benchmark_cli, "preflight", side_effect=lambda p: {"provider": p, "version": "fixture"}), patch.object(benchmark_cli, "send", send):
                result = benchmark.run(root, allow_cli=True)
            self.assertEqual(result["api_calls_reserved"], 0)
            self.assertEqual(result["output_tokens_reserved"], 0)
            self.assertEqual(result["cli_calls_reserved"], 8)
            self.assertEqual(result["cli_usage"]["output_tokens"], 80)
            self.assertEqual([m["successes"] for m in result["models"]], [1, 1])
            self.assertIsNone(result["models"][0]["known_usage"]["estimated_usd"])
            self.assertTrue(benchmark.audit(root)["verified"])

    def test_bad_actions_keep_known_cli_usage_in_normalized_failure(self):
        events = [{"type": "item.completed", "item": {"type": "agent_message", "text": "not JSON"}},
                  {"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 20}}]
        result = benchmark_cli.decode("codex-cli", '\n'.join(json.dumps(e) for e in events), validate=False)
        self.assertEqual(result["usage"]["completion_tokens"], 20)
        self.assertIsNotNone(result["cli_decode_error"])

    def test_real_subprocess_capture_uses_stdin_and_masks_env_without_invoking_models(self):
        raw = {"type": "result", "usage": {"input_tokens": 10, "output_tokens": 2}, "structured_output": {"actions": []}}
        script = 'import os,sys; assert not any(os.getenv(k) for k in ("FIREWORKS_API_KEY","ANTHROPIC_AUTH_TOKEN","ANTHROPIC_CUSTOM_HEADERS")); assert "fixture prompt" in sys.stdin.read(); print(' + repr(json.dumps(raw)) + ')'
        with tempfile.TemporaryDirectory() as folder, patch.dict("os.environ", {"FIREWORKS_API_KEY": "fixture-key", "ANTHROPIC_AUTH_TOKEN": "fixture-token", "ANTHROPIC_CUSTOM_HEADERS": "Authorization: fixture"}), patch.object(benchmark_cli, "command", return_value=[sys.executable, "-c", script]):
            result = benchmark_cli.send({"provider": "claude-cli", "model": "fixture", "settings": {}}, {"prompt": "fixture prompt"}, folder)
            self.assertEqual(result["usage"]["prompt_tokens"], 10)
            self.assertIsNone(result["cli_decode_error"])
            self.assertNotIn("fixture-key", (Path(folder)/"cli-prompt.txt").read_text())

    def test_timeout_after_completed_cli_usage_retains_usage_and_never_returns_actions(self):
        raw = {"type": "result", "usage": {"input_tokens": 10, "output_tokens": 2}, "structured_output": {"actions": []}}
        script = 'import sys,time; sys.stdin.read(); print(' + repr(json.dumps(raw)) + ',flush=True); time.sleep(10)'
        with tempfile.TemporaryDirectory() as folder, patch.object(benchmark_cli, "command", return_value=[sys.executable, "-c", script]):
            result = benchmark_cli.send({"provider": "claude-cli", "model": "fixture", "settings": {}}, {}, folder, timeout=1)
            self.assertEqual(result["usage"]["completion_tokens"], 2)
            self.assertEqual(result["cli_decode_error"], "timeout")

    def test_capture_limit_also_applies_to_fast_exiting_child(self):
        script = 'import sys; sys.stdin.read(); print("x"*8192)'
        with tempfile.TemporaryDirectory() as folder, patch.object(benchmark_cli, "MAX_CAPTURE", 4096), patch.object(benchmark_cli, "command", return_value=[sys.executable, "-c", script]):
            with self.assertRaises(ValueError):
                benchmark_cli.send({"provider": "claude-cli", "model": "fixture", "settings": {}}, {}, folder)
            self.assertEqual((Path(folder)/"stdout.txt").stat().st_size, 4096)
            self.assertTrue(json.loads((Path(folder)/"cli-exit.json").read_text())["capture_truncated"])

    def test_partial_codex_tail_preserves_completed_usage_and_invalidates_actions(self):
        events = [{"type":"item.completed","item":{"type":"agent_message","text":'{"actions":[]}'}},
                  {"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":20}}]
        text = '\n'.join(json.dumps(e) for e in events) + '\n{"type":'
        result = benchmark_cli.decode("codex-cli", text, validate=False)
        self.assertEqual(result["usage"]["completion_tokens"], 20)
        self.assertIsNotNone(result["cli_decode_error"])

    def test_cancellation_stops_and_reaps_process_and_retains_exit_record(self):
        import subprocess
        seen = []
        original = subprocess.Popen.communicate
        def interrupt(process, *args, **kwargs):
            seen.append(process)
            raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as folder, patch.object(benchmark_cli, "command", return_value=[sys.executable, "-c", "import time; time.sleep(30)"]), patch.object(subprocess.Popen, "communicate", interrupt):
            # taskkill's own subprocess must use the real communicate implementation.
            def stop(process):
                with patch.object(subprocess.Popen, "communicate", original):
                    return original_stop(process)
            original_stop = benchmark_cli._stop
            with patch.object(benchmark_cli, "_stop", stop):
                with self.assertRaises(KeyboardInterrupt):
                    benchmark_cli.send({"provider":"codex-cli","model":"fixture","settings":{}}, {}, folder)
            self.assertIsNotNone(seen[0].poll())
            record = json.loads((Path(folder)/"cli-exit.json").read_text())
            self.assertEqual(record["interrupted_by"], "KeyboardInterrupt")

    def test_key_entry_transfers_only_through_child_environment(self):
        import importlib.util
        import os
        spec = importlib.util.spec_from_file_location("fireworks_key_test", Path(__file__).resolve().parents[1]/"scripts/fireworks_key.py")
        helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
        seen = {}
        def spawn(args, **kwargs):
            seen.update(args=args, env=dict(kwargs['env']))
            return object()
        with patch.dict(os.environ, {"FIREWORKS_API_KEY":"previous-fixture"}), patch.object(helper.os, "name", "nt"), patch.object(helper.subprocess, "Popen", spawn):
            helper.open_terminal("new-fixture")
            self.assertEqual(os.environ['FIREWORKS_API_KEY'], "previous-fixture")
        self.assertEqual(seen['env']['FIREWORKS_API_KEY'], "new-fixture")
        self.assertNotIn("new-fixture", json.dumps(seen['args']))
