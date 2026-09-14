import unittest
from bench_prefix import make_plan, verify_response, validate_cache_pattern


class PrefixContract(unittest.TestCase):
    def test_plan_keeps_full_input_and_exact_prefix(self):
        manifest={'requests':[{'input_ids':[i]*16384} for i in range(16)]}
        plan=make_plan(manifest)
        self.assertEqual([r['prefix_tokens'] for r in plan],[0,4096,8192,12288]*4)
        for r,source in zip(plan,manifest['requests']):
            self.assertEqual(r['input_ids'],source['input_ids'])
            self.assertEqual(r['prefix_ids'],source['input_ids'][:r['prefix_tokens']])

    def test_ragged_page_alignment(self):
        p=make_plan({'requests':[{'input_ids':[i]*32767} for i in range(16)]})
        self.assertEqual([r['prefix_tokens'] for r in p[:4]],[0,7936,16128,24320])

    def test_reject_changed_input_and_bad_cached_count(self):
        response=dict(prompt_token_ids=[1,2,3],output_ids=[4],
                      meta_info=dict(id='case',prompt_tokens=3,completion_tokens=1,cached_tokens=2))
        self.assertEqual(verify_response(response,[1,2,3],'case',2),2)
        with self.assertRaises(AssertionError):verify_response(response,[1,2,9],'case',2)
        with self.assertRaises(AssertionError):verify_response(response,[1,2,3],'case',1)
        with self.assertRaises(AssertionError):verify_response(response,[1,2,3],'other',2)

    def test_empty_output_is_not_correctness(self):
        response=dict(prompt_token_ids=[1],output_ids=[],
                      meta_info=dict(id='case',prompt_tokens=1,completion_tokens=1,cached_tokens=0))
        with self.assertRaises(AssertionError):verify_response(response,[1],'case',0)

    def test_quality_length_is_explicit(self):
        response=dict(prompt_token_ids=[1],output_ids=[4]*128,
                      meta_info=dict(id='case',prompt_tokens=1,completion_tokens=128,cached_tokens=0))
        self.assertEqual(verify_response(response,[1],'case',0,128),0)
        with self.assertRaises(AssertionError):verify_response(response,[1],'case',0)

    def test_changing_prefix_hits_are_not_a_speedup(self):
        planned=[0,4096,8192,12288]*4
        validate_cache_pattern(planned,planned,planned)
        partial=[0,3840,7936,12032]*4
        validate_cache_pattern(partial,planned,partial)
        with self.assertRaises(AssertionError):validate_cache_pattern(partial,planned,planned)
        with self.assertRaises(AssertionError):validate_cache_pattern([0]*16,planned)
        with self.assertRaises(AssertionError):validate_cache_pattern([True]+planned[1:],planned)
        with self.assertRaises(AssertionError):validate_cache_pattern(planned[:-1],planned)


if __name__=='__main__':unittest.main()
