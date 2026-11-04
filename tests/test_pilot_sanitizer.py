import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pilot_sanitizer", ROOT / "scripts/sanitize_qwen_pilot.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PilotExportTests(unittest.TestCase):
    def test_published_numeric_pilot_agrees_with_aggregate_evidence(self):
        payload = json.loads((ROOT / "evidence/qwen/pilot-samples.json").read_text())
        text = json.dumps(payload)
        self.assertNotIn("/home/", text)
        self.assertNotIn("C:\\", text)
        for arm in ("baseline", "p50c"):
            summary = json.loads((ROOT / f"evidence/qwen/pilot-{arm}.json").read_text())
            self.assertEqual(payload["arms"][arm]["summary"], summary)
            samples = [s for p in payload["arms"][arm]["problems"] for s in p["samples"]]
            self.assertEqual(sum(s["correct"] for s in samples) / len(samples), summary["accuracy"])
            self.assertEqual(sum(s["tokens"] for s in samples), summary["tokens"])
            self.assertTrue(all(len(s["text_sha256"]) == 64 for s in samples))


if __name__ == "__main__":
    unittest.main()
