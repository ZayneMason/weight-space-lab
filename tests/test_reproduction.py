import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reproduce_breakout", ROOT / "skills/weight-space-lab/scripts/reproduce_breakout.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


class ReproductionTests(unittest.TestCase):
    def test_parser_literal_assignment_and_augmentation(self):
        arrays, count = lab.parse_edits('''```edits
W1[0:2,0:2] = [ [1,2], [3,4] ]
W1[0:2,0] *= -2
b1[:] = 1
b1[0] += 4
b1[1] -= 2
```''')
        self.assertEqual(count, 5)
        np.testing.assert_array_equal(arrays['actor/dense1/kernel'][:2,:2], [[-2,2],[-6,4]])
        np.testing.assert_array_equal(arrays['actor/dense1/bias'][:2], [5,-1])

    def test_parser_rejects_execution_and_invalid_shapes(self):
        for code in ('import os', 'b1[:] = __import__("os").system("echo bad")',
                     'b1[:] = [1] * 32', 'for i in range(32): b1[i] = 1',
                     'b1[-1] = 1', 'b1[32] = 1', 'b1[0:99] = 1', 'b1[0] = 1e100',
                     'b1[0:2] = [1,2,3]', 'b1[:] = float("nan")'):
            with self.subTest(code=code), self.assertRaises((ValueError, SyntaxError)):
                lab.parse_edits(f'```edits\n{code}\n```')

    def test_only_last_edits_block_applies(self):
        arrays, _ = lab.parse_edits('```edits\nb1[0] = 1\n```\nText\n```edits\nb1[1] = 2\n```')
        np.testing.assert_array_equal(arrays['actor/dense1/bias'][:2], [0,2])

    def test_channel_mapping_and_flattening(self):
        state = np.zeros((10,10,4), dtype=bool)
        state[9,4,0] = True
        state[3,5,1] = True
        state[2,4,2] = True
        state[1,6,3] = True
        x = lab.observation(state)
        self.assertEqual(list(np.flatnonzero(x)), [(1*10+6)*6+2, (2*10+4)*6+4, (3*10+5)*6+3, (9*10+4)*6])

    def test_dense_formula(self):
        rng = np.random.default_rng(42)
        arrays = [rng.normal(size=s).astype(np.float32) for _, s in lab.TENSORS.values()]
        x = rng.integers(0,2,600).astype(np.float32)
        expected = np.tanh(np.tanh(x@arrays[0]+arrays[1])@arrays[2]+arrays[3])@arrays[4]+arrays[5]
        np.testing.assert_array_equal(lab.policy_logits(x, arrays), expected)

    def test_audit_fails_on_tool_calls_and_failed_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'events.jsonl'
            good = [{'type':'thread.started'}, {'type':'turn.started'},
                    {'type':'item.completed','item':{'type':'agent_message','text':'response'}},
                    {'type':'turn.completed'}]
            path.write_text('\n'.join(json.dumps(x) for x in good))
            self.assertTrue(lab.audit_events(path)['tool_free'])
            for extra in ({'type':'item.completed','item':{'type':'command_execution'}}, {'type':'error','message':'failed'}):
                path.write_text('\n'.join(json.dumps(x) for x in good+[extra]))
                self.assertFalse(lab.audit_events(path)['tool_free'])
            path.write_text(json.dumps(good[2]))
            self.assertFalse(lab.audit_events(path)['tool_free'])

    def test_audit_records_known_startup_notices_only_before_turn(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'events.jsonl'
            notice = {'type':'item.completed','item':{'type':'error','message':'Code Mode is unavailable because code-mode host is disabled. Code mode will fail closed.'}}
            start, answer, end = {'type':'turn.started'}, {'type':'item.completed','item':{'type':'agent_message'}}, {'type':'turn.completed'}
            path.write_text('\n'.join(json.dumps(x) for x in [notice,start,answer,end]))
            self.assertTrue(lab.audit_events(path)['tool_free'])
            self.assertEqual(len(lab.audit_events(path)['startup_notices']),1)
            path.write_text('\n'.join(json.dumps(x) for x in [start,notice,answer,end]))
            self.assertFalse(lab.audit_events(path)['tool_free'])

    def test_published_response_reconstructs_exact_weights_when_available(self):
        reference = ROOT/'runs/breakout-reproduction/reference'
        if not reference.exists():
            self.skipTest('Download pinned reference assets to run integration test')
        result = lab.verify_reference(reference)
        self.assertTrue(result['exact_tensor_match'])
        self.assertEqual(result['statements'],92)


if __name__ == '__main__':
    unittest.main()
