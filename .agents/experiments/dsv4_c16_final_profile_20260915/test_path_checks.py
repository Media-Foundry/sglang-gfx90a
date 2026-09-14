import unittest
from path_checks import check_paths

def fixture():
    return '\n'.join(
        ['[DSV4 indexer] prefill empty tiles selected: rows=32767'] * 8 +
        [f'[TP{rank}] {hit}' for rank in range(8) for hit in (
            'prefill post-fused4 selected: rows=32767',
            'prefill mix-reuse4 selected: rows=32767',
            'prefill query-reuse4 selected: query_group=16 runtime_m=1')])

class Paths(unittest.TestCase):
    def test_legacy_unranked_empty(self):
        check_paths(fixture(), True)
    def test_missing_rank(self):
        with self.assertRaises(AssertionError):
            check_paths('\n'.join(l for l in fixture().splitlines() if '[TP7]' not in l), True)
    def test_wrong_runtime(self):
        with self.assertRaises(AssertionError):
            check_paths(fixture(), False)
    def test_empty(self):
        with self.assertRaises(AssertionError): check_paths('', True)

if __name__ == '__main__': unittest.main()
