"""CPU-only ownership checks, including partial groups and empty work."""
import unittest
from owner_oracle import plan


class PlanTest(unittest.TestCase):
    def test_rows_and_padding(self):
        for m in (0,1,15,16,17,127,128,129,8191,8192,32767,32768):
            for mode in ('trivial','active','causal'):
                lens=[0 if mode=='trivial' else 2048 if mode=='active' else (i%8192+1)//4 for i in range(m)]
                owners,inverse=plan(lens)
                self.assertEqual(len(owners),8)
                self.assertEqual(len(set(map(len,owners))),1)
                self.assertEqual(len(owners[0])%16,0)
                seen=[]
                for rank,rows in enumerate(owners):
                    for slot,row in enumerate(rows):
                        if row>=0:
                            self.assertEqual(inverse[row],rank*len(rows)+slot)
                            seen.append(row)
                self.assertEqual(len(seen),len(set(seen)))
                self.assertTrue(all(l<=512 or inverse[i]>=0 for i,l in enumerate(lens)))
                self.assertTrue(all(inverse[i]==-1 for i in set(range(m))-set(seen)))


if __name__=='__main__':unittest.main()
