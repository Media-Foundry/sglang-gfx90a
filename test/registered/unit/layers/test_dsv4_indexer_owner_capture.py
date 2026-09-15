import hashlib
import os
import unittest
from unittest.mock import patch
import numpy as np
from sglang.kernels.ops.debug import dsv4_indexer_owner_capture as capture

class OwnerCaptureTest(unittest.TestCase):
    def test_logical_pages_partial_and_preshuffle(self):
        rng=np.random.default_rng(8)
        keys=rng.integers(0,256,(2,64,128),dtype=np.uint8)
        scales=rng.integers(0,256,(2,64,4),dtype=np.uint8)
        for tile in (0,8,16,32,64):
            # Independent element-wise production load address, not the inverse
            # of the helper's own reshape (which can self-confirm a wrong ABI).
            packed=np.empty((2,8192),dtype=np.uint8)
            for row in range(64):
                for col in range(128):
                    offset=(row//tile)*(tile*128)+(col//tile)*tile*tile+(row%tile)*tile+col%tile if tile else row*128+col
                    packed[:,offset]=keys[:,row,col]
            pages=np.concatenate((packed,scales.reshape(2,256)),axis=1)
            for count in (0,1,63,64,65,127,128):
                k,s=capture.logical_rows(pages,count,tile)
                np.testing.assert_array_equal(k,keys.reshape(-1,128)[:count])
                np.testing.assert_array_equal(s,scales.reshape(-1,4)[:count])
                self.assertEqual(capture.digest(k),hashlib.sha256(k.tobytes()).hexdigest())

    def test_tail_garbage_does_not_affect_valid_logical_hash(self):
        pages=np.zeros((2,8448),dtype=np.uint8)
        k,s=capture.logical_rows(pages,65,0)
        pages[1,128:8192]=71;pages[1,8196:]=13
        k2,s2=capture.logical_rows(pages,65,0)
        self.assertEqual(capture.digest(k),capture.digest(k2))
        self.assertEqual(capture.digest(s),capture.digest(s2))

    def test_scope_returns_before_touching_gpu(self):
        values=dict(layer_id=2,rank=0,batch=None,x=None,q_lora=None,q=None,weights=None,
                    positions=None,seq_lens=None,page_table=None,cache=None,
                    preshuffle_tile=16,dot_fp16=False,fp8_fnuz=True)
        with patch.dict(os.environ,{'SGLANG_DSV4_DEBUG_INDEXER_OWNER_DIR':''}), \
             patch.object(capture.torch.cuda,'is_current_stream_capturing') as gpu:
            capture.capture(**values);gpu.assert_not_called()
        with patch.dict(os.environ,{'SGLANG_DSV4_DEBUG_INDEXER_OWNER_DIR':'unused'}), \
             patch('sglang.srt.layers.dsv4_prefill_experiments.mix_pair_active',return_value=False), \
             patch.object(capture.torch.cuda,'is_current_stream_capturing') as gpu:
            capture.capture(**values);gpu.assert_not_called()
        with patch.dict(os.environ,{'SGLANG_DSV4_DEBUG_INDEXER_OWNER_DIR':'unused'}), \
             patch('sglang.srt.layers.dsv4_prefill_experiments.mix_pair_active',return_value=True), \
             patch.object(capture.torch.cuda,'is_current_stream_capturing',return_value=True):
            capture.capture(**values)

if __name__=='__main__':unittest.main()
