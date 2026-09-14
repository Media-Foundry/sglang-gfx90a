import unittest
from metadata import canonical_tie_ids,tile_stats

class Metadata(unittest.TestCase):
    def test_tie_membership_and_descending_emission(self):
        self.assertEqual(canonical_tie_ids(0,4),[-1]*4)
        self.assertEqual(canonical_tie_ids(1,4),[0,-1,-1,-1])
        self.assertEqual(canonical_tie_ids(3,4),[2,1,0,-1])
        self.assertEqual(canonical_tie_ids(9,4),[3,2,1,0])
    def test_full_group(self):
        d=tile_stats([4096]*16,[0]*16,4096)
        self.assertEqual(d['initial_shared_page_fraction'],1)
        self.assertEqual(d['logical_k_load_reuse'],16)
    def test_two_owners_with_different_causal_lengths(self):
        d=tile_stats([4096]*8+[2048]*8,[0]*8+[1]*8,4096)
        self.assertEqual(d['initial_shared_page_group_tiles'],128)
        self.assertEqual(d['candidate_logical_k_tile_loads'],2176)
        self.assertEqual(d['baseline_logical_k_tile_loads'],3072)
    def test_trivial_and_partial_tiles(self):
        d=tile_stats([0,512,513],[0]*3,4096)
        self.assertEqual(d['nonempty_group_tiles'],33)
        self.assertEqual(d['candidate_logical_k_tile_loads'],33)

if __name__=='__main__':unittest.main()
