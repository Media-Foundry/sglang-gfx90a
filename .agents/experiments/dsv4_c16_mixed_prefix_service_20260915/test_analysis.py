"""Synthetic CPU fixtures only; never reported as service measurements."""
import copy
import unittest
from analyze import validate_wave_timing

class TimingContract(unittest.TestCase):
    def fixture(self):
        return dict(responses=[dict(begin=10.,first=15.+i/16,end=30.) for i in range(16)],
            prime_wall_s=2.,wave_ttft_s=5.9375,total_input_tokens=1000,
            newly_computed_tokens=600,full_input_tok_s=1000/5.9375,
            newly_computed_tok_s=600/5.9375)

    def test_raw_timestamp_identity(self):validate_wave_timing(self.fixture())

    def test_wrong_rate_or_drain_time(self):
        wave=self.fixture();wave['wave_ttft_s']=20.
        wave['full_input_tok_s']=50.;wave['newly_computed_tok_s']=30.
        with self.assertRaises(AssertionError):validate_wave_timing(wave)

    def test_missing_nonfinite_reversed(self):
        for value in (float('nan'),float('inf'),9.,31.):
            wave=self.fixture();wave['responses'][0]['first']=value
            with self.assertRaises(AssertionError):validate_wave_timing(wave)
        wave=self.fixture();wave['responses'].pop()
        with self.assertRaises(AssertionError):validate_wave_timing(wave)

    def test_cache_work_not_full_input_rate(self):
        wave=self.fixture();wave['newly_computed_tok_s']=wave['full_input_tok_s']
        with self.assertRaises(AssertionError):validate_wave_timing(wave)
        for value in (0,1001,True):
            wave=self.fixture();wave['newly_computed_tokens']=value
            with self.assertRaises(AssertionError):validate_wave_timing(wave)

if __name__=='__main__':unittest.main()
