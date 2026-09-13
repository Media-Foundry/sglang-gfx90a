import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.rocm.compare_dsv41_row_trace import compare_records, row_metrics
from scripts.rocm.dsv41_trace_hooks import gather_packed_kv_rows, make_hook


class TestRowTrace(unittest.TestCase):
    def test_capture_call_selection_preserves_forward_counter(self):
        with tempfile.TemporaryDirectory() as folder, patch('torch.cuda.is_current_stream_capturing', return_value=False):
            hook = make_hook({'folder':folder,'label':'select','max_calls':4,'capture_calls':[2],
                              'capture_parameters':False})
            module = torch.nn.Identity()
            for i in range(6):
                value = torch.tensor([[i]])
                self.assertIsNone(hook(module,(value,),value))
            files=list(Path(folder).glob('*.pt'))
            self.assertEqual(len(files),1)
            record=torch.load(files[0],weights_only=True)
            self.assertEqual(record['call'],2)
            self.assertEqual(record['output'].item(),2)

    def test_page_planar_cache_capture(self):
        payload = torch.arange(2 * 4 * 576).reshape(2, 4, 576).to(torch.uint8)
        scales = (torch.arange(2 * 4 * 8) + 129).reshape(2, 4, 8).to(torch.uint8)
        pages = torch.cat((payload.flatten(1), scales.flatten(1)), -1)
        ids = torch.tensor([7, 0, 3, 4, 3])
        expected = torch.cat((payload.reshape(-1, 576)[ids], scales.reshape(-1, 8)[ids]), -1)
        self.assertTrue(torch.equal(gather_packed_kv_rows(pages, ids, 4), expected))

    def test_last_valid_not_padded_tail(self):
        full = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        cached = full[2:3].clone()
        a = {"args": [], "output": cached}
        b = {"args": [], "output": full}
        rows = compare_records(a, b, 1, 4, 0, 2)
        self.assertTrue(rows[0]["exact"])
        self.assertFalse(compare_records(a, b, 1, 4, 0, 3)[0]["exact"])

    def test_dtype_mismatch_is_not_hidden(self):
        self.assertFalse(row_metrics(torch.ones(4), torch.ones(4).bfloat16())["compatible"])

    def test_non_row_tensor_is_skipped(self):
        a = {"args": [], "output": torch.ones(6, 4)}
        b = {"args": [], "output": torch.ones(6, 4)}
        self.assertEqual(compare_records(a, b, 1, 4, 0, 3)[0]["skipped"], "not a token-row tensor")

    def test_nonfinite_does_not_compare_as_pass(self):
        result = row_metrics(torch.tensor([float("nan")]), torch.tensor([float("nan")]))
        self.assertFalse(result["finite"])
        self.assertFalse(result["exact"])


if __name__ == "__main__":
    unittest.main()
