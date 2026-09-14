import unittest
from trace_checks import check_shapes

def lines(runtime, different=False, missing=False):
    out=[]
    for rank in range(7 if missing else 8):
        for m in (32766,32768):
            suffix=(m if different or not runtime else 0)
            out.append(f'[TP{rank}] DSV4 indexer compile-shape rows={m} width=2048 '
                       f'pages=32 page_stride=32 group=16 runtime_m={runtime} '
                       f'align=0,0,0,0,0 artifact=abc{suffix:x} object={1000+suffix}')
    return '\n'.join(out)

class Contracts(unittest.TestCase):
    def test_reuse_and_distinct_control(self):
        self.assertEqual(len(check_shapes(lines(1),1)),8)
        self.assertEqual(len(check_shapes(lines(0),0)),8)
    def test_must_reject_multiple_runtime_artifacts(self):
        with self.assertRaises(AssertionError):check_shapes(lines(1,different=True),1)
    def test_missing_rank_is_not_a_pass(self):
        with self.assertRaises(AssertionError):check_shapes(lines(1,missing=True),1)
    def test_empty_is_not_a_pass(self):
        with self.assertRaises(AssertionError):check_shapes('',1)

if __name__=='__main__':unittest.main()
