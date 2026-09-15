"""CPU-only timing/input-contract tests for the frozen service client."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('pair_client', Path(__file__).with_name('client.py'))
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)

class ClientTest(unittest.TestCase):
    def invoke(self, *, wrong_echo=False, missing_ids=False):
        args = SimpleNamespace(inputs=Path('unused'), request_count=2, request_offset=0,
                               rounds=1, tokens=1, timeout=30, base_url='http://unused', output=None)
        manifest = {'requests': [{'input_ids': [i, i+10]} for i in range(2)]}
        def response(url, payload, timeout):
            self.assertTrue(payload['return_prompt_token_ids'])
            i = payload['input_ids'][0]
            return ({'prompt_token_ids': [99] if wrong_echo else payload['input_ids'],
                     'output_ids': [] if missing_ids else [7], 'text': 'code',
                     'meta_info': {'prompt_tokens': 2, 'cached_tokens': 0}},
                    10+i, 20+i, 30+i)
        with patch.object(client, 'parse_args', return_value=args), \
             patch.object(Path, 'read_text', return_value=json.dumps(manifest)), \
             patch.object(Path, 'read_bytes', return_value=json.dumps(manifest).encode()), \
             patch.object(client, 'post_stream', side_effect=response), \
             contextlib.redirect_stdout(io.StringIO()) as stdout:
            client.main()
        return json.loads(stdout.getvalue().split('\n', 1)[1])

    def test_uses_last_first_not_drain(self):
        result = self.invoke()
        wave = result['rounds'][0]
        self.assertEqual(wave['prefill_wall_s'], 11)
        self.assertEqual(wave['group_wall_s'], 21)
        self.assertAlmostEqual(result['median_input_tok_s'], 4/11)
        self.assertTrue(wave['input_echo_exact'])
        self.assertEqual(len(wave['raw_times']), 2)

    def test_wrong_input_echo_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'input IDs'):
            self.invoke(wrong_echo=True)

    def test_empty_output_ids_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'did not complete'):
            self.invoke(missing_ids=True)

if __name__ == '__main__': unittest.main()
