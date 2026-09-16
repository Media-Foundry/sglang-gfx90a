"""CPU arithmetic proof checks for the wave-uniform route address contract."""
import unittest


class Mapping(unittest.TestCase):
    def test_each_wave_has_one_token_and_whole_vectors(self):
        stride=1664*256
        for v in (4,8):
            width=4096//v
            self.assertEqual(width%64,0)
            self.assertEqual(stride%width,0)
            for m in (8192,8193,16384,32765,32767,32768,36864):
                limit=m*width
                self.assertEqual(limit%64,0)
                for block in (0,1,103,415,831,1663):
                    for wave in range(4):
                        initial=block*256+wave*64
                        last=(limit-1-initial)//stride
                        for step in sorted({0,1,max(0,last-1),last}):
                            j=initial+step*stride
                            if step<0 or j>=limit:continue
                            self.assertEqual(j//width,(j+63)//width)
                            self.assertLess(j+63,limit)
                            for lane in (0,1,63):
                                col=((j+lane)%width)*v
                                self.assertLessEqual(col+v,4096)
                                self.assertEqual(col*4%(4*v),0)
                                # grid-stride mapping has exactly one owner.
                                linear=(j+lane)%stride
                                self.assertEqual(linear//256,block)
                                self.assertEqual(linear%256,wave*64+lane)


if __name__=='__main__':unittest.main()
