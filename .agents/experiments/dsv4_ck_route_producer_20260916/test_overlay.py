"""CPU source-contract tests only; no compiled-kernel correctness claim."""
from pathlib import Path
import unittest
from overlay import patch,GRID,DESCRIPTOR,SCATTER


class Overlay(unittest.TestCase):
    def test_round_trip_and_input_gather_preserved(self):
        original=(Path('/home/pc/pytorch/third_party/aiter/3rdparty/composable_kernel/include')/GRID).read_text()
        candidate=patch(original)
        self.assertEqual(candidate.count('#if DSV4_ROUTE_MAJOR_STAGE1'),4)
        # Both Run variants have a descriptor and scatter preprocessor branch.
        parts=candidate.split('#if DSV4_ROUTE_MAJOR_STAGE1')
        restored=parts[0]
        for part in parts[1:]:
            self.assertIn('#else\n',part)
            discarded,rest=part.split('#else\n',1)
            baseline,tail=rest.split('\n#endif',1)
            restored+=baseline+tail
        self.assertEqual(original,restored)
        # All input gathers, MFMA pipelines and SwiGLU bodies remain untouched.
        begin=original.index('        const index_t token_pos')
        end=original.index('        const IndexType expert_stride',begin)
        self.assertIn(original[begin:end],candidate)

    def test_changed_contract_rejected(self):
        with self.assertRaises(AssertionError):
            patch('different header')

    def test_entry_geometry(self):
        entry=Path(__file__).with_name('stage1.cu').read_text()
        self.assertIn('256,64,64,128,1,4,false,false,false,2,false>',entry)
        self.assertIn('DSV4_ROUTE_MAJOR_STAGE1 ? ids.numel() : hidden.size(0)*6',entry)
        self.assertIn('out.numel()*out.element_size()<(int64_t(1)<<31)',entry)


if __name__=='__main__':
    unittest.main()
