"""CPU/mock tests for unified DSV4 C4 request-state lifecycle."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.disaggregation.decode import DecodeReqToTokenPool
from sglang.srt.mem_cache.allocation import alloc_req_slots
from sglang.srt.mem_cache.deepseek_v4_compress_state import KVAndScore
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _request(req_pool_idx=None, *, reused=False):
    return SimpleNamespace(
        # This fork's Req exposes the allocation fields directly. Keep the
        # lifecycle fixture aligned with the allocator's actual public API.
        req_pool_idx=req_pool_idx,
        kv_committed_len=1 if reused else 0,
        kv_allocated_len=1 if reused else 0,
        kv=SimpleNamespace(
            req_pool_idx=req_pool_idx,
            kv_committed_len=1 if reused else 0,
            kv_allocated_len=1 if reused else 0,
            holds_kv=reused,
        ),
        inflight_middle_chunks=1 if reused else 0,
    )


def _mark_reused(req):
    req.kv_committed_len = 1
    req.kv_allocated_len = 1
    req.kv.kv_committed_len = 1
    req.kv.kv_allocated_len = 1
    req.kv.holds_kv = True
    req.inflight_middle_chunks = 1


def _c4_pool(rows: int, width: int, ring_size: int):
    return SimpleNamespace(
        ratio=4,
        ring_size=ring_size,
        kv_score_buffer=KVAndScore(torch.full((rows, width), 7.0)),
    )


def _token_pool(unified: bool, ring_size: int = 8):
    logical_rows = 4 * ring_size
    physical_rows = logical_rows + ring_size + 4
    attn = _c4_pool(physical_rows, width=12, ring_size=ring_size)
    indexer = _c4_pool(physical_rows, width=8, ring_size=ring_size)
    c128 = SimpleNamespace(
        ratio=128,
        ring_size=128,
        kv_score_buffer=KVAndScore(torch.full((physical_rows, 8), 9.0)),
    )
    token_pool = object.__new__(DeepSeekV4TokenToKVPool)
    token_pool._unified_kv = unified
    token_pool.compress_state_pools = [attn, c128]
    token_pool.indexer_compress_state_pools = [indexer, None]
    token_pool.get_ring_size = MagicMock(return_value=ring_size)
    return token_pool, attn, indexer, c128, logical_rows


class TestUnifiedC4StateLifecycle(unittest.TestCase):
    def test_unified_prices_compressed_latents_not_linear_swa(self):
        configurator = object.__new__(DSV4PoolConfigurator)
        configurator._unified = True
        configurator.qk_nope_head_dim = 448
        configurator.qk_rope_head_dim = 64
        configurator.indexer_head_dim = 128
        configurator.num_layers_ca4 = 21
        configurator.num_layers_ca128 = 20
        configurator.c4_shrink_factor = 1
        configurator.low_ratio_source_counts = {1: 0, 2: 0}
        self.assertEqual(configurator._get_bytes_per_full_token(), 6229.)

    def test_unified_budget_subtracts_fixed_state_and_keeps_one_million(self):
        configurator = object.__new__(DSV4PoolConfigurator)
        configurator._unified = True
        configurator.requested_max_running_requests_per_worker = 256
        configurator.bytes_per_full_token = 6229.
        configurator.swa_ratio = 0.65
        configurator.c4_shrink_factor = 1
        configurator._get_c128_state_fixed_bytes = MagicMock(return_value=5 * 1024**3)
        configurator._fixed_swa_bytes = MagicMock(return_value=2 * 1024**3)
        configurator._fixed_c4_state_bytes = MagicMock(return_value=1024**3)
        budget = 16 * 1024**3
        result = configurator.calculate_pool_sizes(budget, page_size=256)
        self.assertEqual(result.max_total_num_tokens, int(8 * 1024**3 / 6229) // 256 * 256)
        self.assertGreaterEqual(result.max_total_num_tokens, 1048576)
        self.assertEqual(result.c4_state_pool_size, 0)

    def test_non_unified_v41_ratio_source_coefficient_unchanged(self):
        configurator = object.__new__(DSV4PoolConfigurator)
        configurator._unified = False
        configurator.qk_nope_head_dim = 448
        configurator.qk_rope_head_dim = 64
        configurator.indexer_head_dim = 128
        configurator.swa_ratio = 0.1
        configurator.num_layers_total = 40
        configurator.num_layers_ca4 = configurator.num_layers_ca128 = 0
        configurator.c4_ring_size = 8
        configurator.swa_page_size = 256
        configurator.c4_shrink_factor = 1
        configurator.low_ratio_source_counts = {1: 1, 2: 3}
        self.assertAlmostEqual(configurator._get_bytes_per_full_token(), 3966.)

    def test_pool_size_is_exact_request_ring_product(self):
        configurator = object.__new__(DSV4PoolConfigurator)
        configurator.disaggregation_mode = "decode"
        configurator.disaggregation_decode_extra_slots = 3
        configurator.c4_ring_size = 16

        self.assertEqual(configurator._unified_c4_state_pool_size(10), 14 * 16)

    def test_clear_resets_only_selected_request_rings(self):
        ring_size = 8
        token_pool, attn, indexer, c128, logical_rows = _token_pool(
            unified=True, ring_size=ring_size
        )

        token_pool.clear_c4_req_states([1, 3])

        selected = torch.tensor(list(range(8, 16)) + list(range(24, 32)))
        untouched = torch.tensor(list(range(0, 8)) + list(range(16, 24)))
        for pool in (attn, indexer):
            state = pool.kv_score_buffer.kv_score
            half = state.shape[-1] // 2
            self.assertTrue(
                torch.equal(
                    state[selected, :half], torch.zeros_like(state[selected, :half])
                )
            )
            self.assertTrue(torch.isneginf(state[selected, half:]).all())
            self.assertTrue((state[untouched] == 7).all())
            self.assertTrue((state[logical_rows:] == 7).all())
        self.assertTrue((c128.kv_score_buffer.kv_score == 9).all())

    def test_clear_is_noop_off_the_unified_path(self):
        """The non-unified (fp8) pool addresses C4 state by SWA page, so a
        req-slot reset must not touch it."""
        token_pool, attn, indexer, _, _ = _token_pool(unified=False)

        token_pool.clear_c4_req_states([1, 3])

        for pool in (attn, indexer):
            self.assertTrue((pool.kv_score_buffer.kv_score == 7).all())

    def test_req_pool_hook_fires_for_new_slots_only(self):
        req_pool = ReqToTokenPool(3, 16, "cpu", enable_memory_saver=False)
        hook = MagicMock()
        req_pool.register_on_alloc_rows(hook)
        reused = _request()

        # First admission: a brand-new slot, so its C4 ring must be cleared.
        (reused_idx,) = alloc_req_slots(req_pool, [reused], None)
        hook.assert_called_once_with([reused_idx])

        # Chunked continuation reuses the same slot -- clearing it here would
        # wipe the state captured by the previous chunk.
        hook.reset_mock()
        _mark_reused(reused)
        self.assertEqual(alloc_req_slots(req_pool, [reused], None), [reused_idx])
        hook.assert_not_called()

        # Mixed batch: only the newly allocated slot is reported.
        fresh = _request()
        indices = alloc_req_slots(req_pool, [reused, fresh], None)
        self.assertEqual(indices[0], reused_idx)
        self.assertNotEqual(indices[1], reused_idx)
        hook.assert_called_once_with([indices[1]])

    def test_decode_req_pool_hook_fires_for_new_slots_only(self):
        """PD decode pre-allocates through DecodeReqToTokenPool, which has its
        own alloc; it must report fresh rows the same way."""
        req_pool = DecodeReqToTokenPool(
            2, 16, "cpu", enable_memory_saver=False, pre_alloc_size=2
        )
        hook = MagicMock()
        req_pool.register_on_alloc_rows(hook)

        first = _request()
        (first_idx,) = req_pool.alloc([first])
        hook.assert_called_once_with([first_idx])

        hook.reset_mock()
        _mark_reused(first)
        second = _request()
        indices = req_pool.alloc([first, second])
        self.assertEqual(indices[0], first_idx)
        hook.assert_called_once_with([indices[1]])

        hook.reset_mock()
        self.assertEqual(req_pool.alloc([first]), [first_idx])
        hook.assert_not_called()

    def test_req_pool_without_hook_is_unchanged(self):
        req_pool = ReqToTokenPool(2, 16, "cpu", enable_memory_saver=False)
        (idx,) = alloc_req_slots(req_pool, [_request()], None)
        self.assertGreater(idx, 0)


if __name__ == "__main__":
    unittest.main()
