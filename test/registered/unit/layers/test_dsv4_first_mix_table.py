import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from sglang.kernels.ops.layernorm import gfx90a_first_mix_table as table


class FirstMixTableTest(unittest.TestCase):
    def inputs(self):
        model=NS(start_layer=0,end_layer=43,layers=[NS(config=NS(model_type='deepseek_v4'))])
        batch=NS(forward_mode=NS(name='EXTEND'),lora_ids=[None],mm_inputs=None,input_ids=torch.zeros(64,dtype=torch.int64))
        return model,batch

    def test_scope_and_mode(self):
        model,batch=self.inputs()
        with patch.dict(os.environ,{table.ENV:'1'}),patch.object(table,'common_unfused_fp32',return_value=True):
            self.assertTrue(table.eligible(model,batch,None))
            batch.mm_inputs=[None]*16
            self.assertTrue(table.eligible(model,batch,None))
            batch.mm_inputs=[]
            self.assertTrue(table.eligible(model,batch,None))
            self.assertFalse(table.eligible(model,batch,torch.empty(1)))
            for mode in ('DECODE','TARGET_VERIFY','DRAFT_EXTEND','MIXED','IDLE'):
                batch.forward_mode.name=mode;self.assertFalse(table.eligible(model,batch,None))
            batch.forward_mode.name='EXTEND'
            for field,value in [('lora_ids',['adapter']),('mm_inputs',[object()])]:
                old=getattr(batch,field);setattr(batch,field,value)
                self.assertFalse(table.eligible(model,batch,None));setattr(batch,field,old)
            model.layers[0].config.model_type='deepseek_v41'
            self.assertFalse(table.eligible(model,batch,None))
        model,batch=self.inputs()
        with patch.dict(os.environ,{table.ENV:'0'}):self.assertFalse(table.eligible(model,batch,None))
        with patch.dict(os.environ,{table.ENV:'1'}),patch.object(table,'common_unfused_fp32',return_value=False):
            self.assertFalse(table.eligible(model,batch,None))

    def test_padded_vocab_contract(self):
        for padded in (129280,131072):
            for rank in range(8):
                n=padded//8
                embedding=NS(tp_size=8,num_added_embeddings=0,num_embeddings=129280,org_vocab_size=129280,
                    num_embeddings_padded=padded,org_vocab_size_padded=padded,num_embeddings_per_partition=n,
                    shard_indices=NS(padded_org_vocab_start_index=rank*n,padded_org_vocab_end_index=(rank+1)*n,
                                     org_vocab_start_index=min(rank*n,129280),org_vocab_end_index=min((rank+1)*n,129280)))
                self.assertEqual(table.shard_contract(embedding,rank,8),(n,padded))
                embedding.num_added_embeddings=1
                with self.assertRaises(AssertionError):table.shard_contract(embedding,rank,8)

    def test_consume_once_and_invalidate(self):
        fn=torch.zeros(24,16);mixes=torch.zeros(64,1,24);batch=NS()
        batch.__dict__[table.ENTRY]=(fn,mixes)
        self.assertIsNone(table.take(1,batch,fn,64))
        self.assertIs(table.take(0,batch,fn,64),mixes)
        self.assertIsNone(table.take(0,batch,fn,64))
        batch.__dict__[table.ENTRY]=(fn,mixes)
        with self.assertRaises(AssertionError):table.take(0,batch,fn.clone(),64)
        model=NS();setattr(model,table.CACHE,object());table.invalidate(model)
        self.assertIsNone(getattr(model,table.CACHE))

    def test_versions_change_keys_and_default_clears_pending(self):
        embedding=NS(weight=torch.zeros(8,16),num_embeddings_padded=8);fn=torch.zeros(24,64)
        old=table.weight_key(embedding,fn,1e-6);fn.add_(1)
        self.assertNotEqual(old,table.weight_key(embedding,fn,1e-6))
        model,batch=self.inputs();batch.__dict__[table.ENTRY]=object()
        with patch.dict(os.environ,{table.ENV:'0'}),patch.object(table,'build') as build:
            table.prepare(model,batch.input_ids,batch,None,None);build.assert_not_called()
        self.assertNotIn(table.ENTRY,batch.__dict__)

    def test_loader_hooks_and_online_update_guard(self):
        root=Path(__file__).resolve().parents[4]
        model_tree=ast.parse((root/'python/sglang/srt/models/deepseek_v4.py').read_text())
        model=next(n for n in model_tree.body if isinstance(n,ast.ClassDef) and n.name=='DeepseekV4ForCausalLM')
        for name in ('load_weights','post_load_weights','set_embed_and_head'):
            fn=next(n for n in model.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertIn('_first_mix_table.invalidate(self.model)',ast.unparse(fn.body[0]))
        source=root/'python/sglang/srt/model_executor/model_runner_components/weight_updater.py'
        tree=ast.parse(source.read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_unsupported_derived_weight_cache_error')
        self.assertIn(table.ENV,ast.unparse(fn))
        namespace={'os':os,'Optional':__import__('typing').Optional}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),namespace)
        with patch.dict(os.environ,{table.ENV:'1'}):
            self.assertIn('restart',namespace['_unsupported_derived_weight_cache_error']())


if __name__=='__main__':unittest.main()
