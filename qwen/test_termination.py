import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("termination", Path(__file__).with_name("termination.py"))
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
last_boxed = t.grader().last_boxed  # pure-python extractor; no math-verify needed


def exact(text, gold):
    """String-equality stand-in for the math-verify grader."""
    return last_boxed(text) == gold


def noise(n, seed=0):
    rng = random.Random(seed)
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789 +-=") for _ in range(n))


def features(closed, correct, found=False, boxed=True):
    return {"closed": closed, "correct": correct, "found": found, "boxed": boxed, "lenient_correct": None,
            "tail_ratio": .5, "loop_onset": None, "regrade_agrees": True}


class TextMeasures(unittest.TestCase):
    def test_boxed_values_are_distinct_and_ordered(self):
        text = "a \\boxed{1} b \\boxed{2} c \\boxed{1} d \\boxed{ 3 }</think> \\boxed{\\frac{1}{2}}"
        found = t.boxed_values(text)
        self.assertEqual([v for _, v in found], ["1", "2", "3", "\\frac{1}{2}"])
        self.assertEqual(found[0][0], text.index("\\boxed{1}"))
        self.assertEqual(len(t.boxed_values(text, limit=2)), 2)
        self.assertEqual(t.boxed_at("\\boxed 42$ rest", 0), "42")
        self.assertIsNone(t.boxed_at("\\boxed{unclosed", 0))

    def test_repetition_loop_is_detected_near_its_start(self):
        text = noise(30000) + "Wait, let me re-check the sum. " * 2000
        self.assertLess(t.tail_ratio(text), t.LOOP_THRESHOLD)
        onset = t.loop_onset(text)
        self.assertAlmostEqual(onset, 30000 / len(text), delta=4000 / len(text))
        self.assertGreater(t.tail_ratio(noise(30000, 1)), t.LOOP_THRESHOLD)
        self.assertIsNone(t.loop_onset(noise(30000, 2)))

    def test_lenient_answer_takes_latest_phrase_with_a_number(self):
        self.assertEqual(t.lenient_answer("The answer is 4.\nFinal Answer: $7$."), "7")
        self.assertIsNone(t.lenient_answer("Final Answer: none"))

    def test_sample_features_separate_closure_answer_and_found(self):
        text = "<think>try \\boxed{7} then loop</think>\nFinal Answer: 7."
        f = t.sample_features(text, "7", recorded_correct=0, grade=exact, last_boxed=last_boxed)
        self.assertTrue(f["closed"])
        self.assertFalse(f["boxed"])           # nothing boxed after </think>
        self.assertFalse(f["correct"])         # strict grade
        self.assertTrue(f["regrade_agrees"])
        self.assertTrue(f["found"])            # but the answer was written in the reasoning
        self.assertTrue(f["lenient_correct"])
        open_text = "<think>still thinking \\boxed{9}"
        g = t.sample_features(open_text, "9", recorded_correct=1, grade=exact, last_boxed=last_boxed)
        self.assertFalse(g["closed"])
        self.assertTrue(g["correct"])          # graded on the unfinished reasoning, like the released extractor
        self.assertNotIn("text", g)


class Statistics(unittest.TestCase):
    def test_closure_cells_attribute_the_whole_difference(self):
        base = {("p1", 0): features(True, True), ("p2", 0): features(False, False, found=True),
                ("p3", 0): features(False, False), ("p4", 0): features(False, False),
                ("p5", 0): features(True, True), ("p6", 0): features(False, True)}
        edit = {("p1", 0): features(True, True), ("p2", 0): features(True, True),
                ("p3", 0): features(True, True), ("p4", 0): features(False, False),
                ("p5", 0): features(True, False), ("p6", 0): features(True, False)}
        r = t.paired_cells(base, edit, replicates=500)
        cells = r["cells"]
        self.assertEqual(cells["baseline_closed/edited_closed"]["n"], 2)
        self.assertEqual(cells["baseline_open/edited_closed"]["n"], 3)
        self.assertEqual(cells["baseline_open/edited_open"]["n"], 1)
        self.assertEqual(cells["baseline_open/edited_closed"]["edited_correct"], 2)
        self.assertEqual(cells["baseline_open/edited_closed"]["edited_correct_where_baseline_never_wrote_it"], 1)
        self.assertAlmostEqual(r["delta"], (3 - 3) / 6)
        self.assertAlmostEqual(sum(c["contribution"] for c in cells.values()), r["delta"])
        self.assertAlmostEqual(r["both_closed_accuracy_difference"], -0.5)
        lo, hi = r["delta_ci95"]
        self.assertLessEqual(lo, r["delta"])
        self.assertGreaterEqual(hi, r["delta"])

    def test_aggregate_bounds_match_published_unedited_numbers(self):
        b = t.aggregate_bounds({"ok": 158, "total": 2000, "cut": 1860, "tokens_sum": 2000})
        self.assertAlmostEqual(b["accuracy"], .079)
        self.assertAlmostEqual(b["finish_rate"], .07)
        self.assertAlmostEqual(b["correct_and_capped_at_least"], .009)
        self.assertEqual(b["accuracy_given_finished_at_most"], 1.0)

    def test_accuracy_intervals_cluster_by_problem(self):
        r = t.accuracy_ci({"a": [1, 1, 1, 1], "b": [0, 0, 0, 0]}, replicates=400)
        self.assertEqual(r["accuracy"], .5)
        self.assertEqual(r["samples"], 8)
        p = t.paired_accuracy({"a": [0, 0], "b": [0, 0]}, {"a": [1, 1], "b": [0, 1]}, replicates=400)
        self.assertAlmostEqual(p["delta"], .75)


class BudgetCurves(unittest.TestCase):
    def test_cutting_at_a_budget_regrades_the_prefix(self):
        encode = lambda text: [ord(c) for c in text]
        decode = lambda ids: "".join(map(chr, ids))
        early = "<think>short</think> \\boxed{5}"
        late = "<think>" + "x" * 80 + "</think> \\boxed{6}"
        samples = [{"text": early, "gold": "5", "recorded_correct": 1, "token_ids": None},
                   {"text": late, "gold": "6", "recorded_correct": 1, "token_ids": encode(late)}]
        curve = t.budget_curve(samples, [10, 40, 200], decode, exact, encode=encode, cap=200)
        self.assertEqual(curve[10], {"closed": 0.0, "accuracy": 0.0})
        self.assertEqual(curve[40], {"closed": .5, "accuracy": .5})
        self.assertEqual(curve[200], {"closed": 1.0, "accuracy": 1.0})


def _write_upstream(root, arm, problems):
    for shard in range(10):
        folder = root / arm / "parts" / f"math500_s{shard}of10"
        folder.mkdir(parents=True)
        rows = problems[shard::10]
        ok = sum(sum(r["correct"]) for r in rows)
        total = sum(len(r["correct"]) for r in rows)
        (folder / "scores.json").write_text(json.dumps({"math500": {"ok": ok, "total": total, "cut": total - 1, "tokens_sum": 10 * total}}))
        (folder / "math500_per_problem.json").write_text(json.dumps([{"id": r["id"], "correct": r["correct"], "pred_first": None} for r in rows]))
        with gzip.open(folder / "math500_first_samples.jsonl.gz", "wt", encoding="utf-8") as stream:
            for r in rows:
                stream.write(json.dumps({"id": r["id"], "answer": r["answer"], "text": r["text"]}) + "\n")


class EndToEnd(unittest.TestCase):
    def test_upstream_cli_reports_without_leaking_text(self):
        loop = "<think>" + "Let me reconsider the problem. " * 1500
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "eval"
            base = [{"id": f"test/algebra/{i}.json", "answer": str(i), "text": loop if i % 3 else f"<think>ok</think> \\boxed{{{i}}}",
                     "correct": [0 if i % 3 else 1, 0, 0, 0]} for i in range(20)]
            edit = [{**r, "text": f"<think>fine</think> \\boxed{{{r['answer']}}}", "correct": [1, 1, 0, 0]} for r in base]
            _write_upstream(root, "unedited", base)
            _write_upstream(root, "p50c", edit)
            out = Path(tmp) / "out"
            t.main(["upstream", str(root), str(out), "--arms", "unedited", "p50c", "--replicates", "200"])
            summary = json.loads((out / "summary.json").read_text())
            arms = summary["arms"]
            self.assertAlmostEqual(arms["unedited"]["all_samples_accuracy"]["accuracy"], 7 / 80)
            self.assertAlmostEqual(arms["p50c"]["all_samples_accuracy"]["accuracy"], .5)
            self.assertEqual(arms["unedited"]["text_samples"]["regrade_agreement"], 1.0)
            self.assertAlmostEqual(arms["unedited"]["text_samples"]["open_looping_share_of_open"], 1.0)
            cells = summary["paired"]["p50c"]["text_samples_by_closure"]["cells"]
            self.assertEqual(cells["baseline_open/edited_closed"]["n"], 13)
            self.assertEqual(cells["baseline_open/edited_closed"]["edited_correct"], 13)
            self.assertAlmostEqual(summary["paired"]["p50c"]["text_samples_by_closure"]["share_of_delta_from_baseline_open_edited_closed"], 1.0)
            written = (out / "summary.json").read_text() + (out / "samples.json").read_text()
            self.assertNotIn("Let me reconsider", written)
            self.assertNotIn("test/algebra", written)
            self.assertIn(hashlib.sha256(b"test/algebra/1.json").hexdigest(), written)

    def test_upstream_loader_rejects_inconsistent_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "eval"
            rows = [{"id": f"p{i}", "answer": "1", "text": "\\boxed{1}", "correct": [1, 0]} for i in range(10)]
            _write_upstream(root, "unedited", rows)
            scores = root / "unedited/parts/math500_s0of10/scores.json"
            data = json.loads(scores.read_text())
            data["math500"]["ok"] += 1
            scores.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                t.load_upstream(root, "unedited")

    def test_local_runner_outputs_load_every_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "baseline"
            run.mkdir()
            (run / "job.json").write_text(json.dumps({"arm": "baseline", "problem_ids": ["q1"], "sampling": {"max_tokens": 64}}))
            gens = [{"sample": k, "text": f"<think>a</think> \\boxed{{{k}}}", "correct": k == 1, "finish_reason": "stop",
                     "tokens": 5, "token_ids": [1, 2, 3, 4, 5]} for k in range(4)]
            name = hashlib.sha256(b"q1").hexdigest()[:20] + ".json"
            (run / name).write_text(json.dumps({"id": "q1", "answer": "1", "generations": gens}))
            data = t.load_local(run)
            self.assertEqual(data["aggregate"], {"ok": 1, "total": 4, "cut": 0, "tokens_sum": 20})
            self.assertEqual(len(data["samples"]), 4)
            self.assertEqual(data["max_tokens"], 64)


if __name__ == "__main__":
    unittest.main()
