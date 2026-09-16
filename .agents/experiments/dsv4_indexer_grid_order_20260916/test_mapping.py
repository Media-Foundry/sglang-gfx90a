"""Pure-CPU grid ownership and unchanged post-coordinate kernel body checks."""
import ast
from pathlib import Path
import unittest

root=Path(__file__).resolve().parent
repo=root.parents[2]

class Mapping(unittest.TestCase):
    def test_exact_cover(self):
        for m in (1,15,16,17,3072,3584,3840,4095,4096,8193):
            nq=(m+15)//16
            for width in (512,576,2048,4096,8192):
                nk=(width+15)//16
                for gq in (1,4,8,16,32):
                    size=((nq+gq-1)//gq)*gq*nk
                    seen=set()
                    for pid in range(size):
                        q=(pid//(gq*nk))*gq+pid%gq
                        k=(pid%(gq*nk))//gq
                        if q>=nq:continue
                        self.assertNotIn((q,k),seen)
                        seen.add((q,k))
                    self.assertEqual(len(seen),nq*nk)

    def test_math_body_identical(self):
        old=ast.parse((repo/'python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_runtime_m.py').read_text())
        new=ast.parse((root/'kernel.py').read_text())
        before=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='reuse_runtime_m')
        after=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='reuse_grouped_grid')
        # Both bodies from rows=first+... onward must remain unchanged.
        def suffix(fn):
            i=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='rows')
            return [ast.dump(n) for n in fn.body[i:]]
        self.assertEqual(suffix(before),suffix(after))

if __name__=='__main__':unittest.main()
