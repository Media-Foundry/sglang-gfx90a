import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.prefill_delayer import (
    PrefillDelayer,
    PrefillDelayerSinglePassExecutor,
    RecentPrefillBatchSizeTracker,
    _NegotiateOutput,
    _State,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPrefillDelayerHighWatermark(unittest.TestCase):
    def test_wall_timeout_releases_slot_condition(self):
        delayer = object.__new__(PrefillDelayer)
        delayer._token_usage_low_watermark = None
        delayer._queue_trigger_enabled = False
        delayer._prefill_max_requests = 16
        delayer._max_delay_ms = 5.0
        delayer._max_delay_passes = 10000
        delayer.enable_dp_attention = True
        delayer.skip_first_delayer = False
        delayer._gather_info = MagicMock(
            return_value=torch.tensor([[1, 0, 15, 16, 1, 1]], dtype=torch.int64)
        )

        with patch(
            "sglang.srt.managers.prefill_delayer.time.perf_counter",
            return_value=10.006,
        ):
            result = delayer._negotiate_should_allow_prefill_pure(
                prev_state=_State(delayed_count=7, start_time=10.0),
                local_prefillable=True,
                token_usage=0.5,
                running_batch=15,
                max_prefill_bs=16,
                max_running_requests=16,
                waiting_queue_len=1,
            )

        self.assertTrue(result.output_allow)
        self.assertEqual(result.output_reason, "wait_timeout")
        self.assertEqual(result.wait_forward_passes, 7)

    def test_rank_clock_skew_cannot_split_admission(self):
        for shared_timeout in (False, True):
            decisions = []
            for local_now in (10.004, 10.006):
                delayer = object.__new__(PrefillDelayer)
                delayer._token_usage_low_watermark = None
                delayer._queue_trigger_enabled = True
                delayer._queue_min_ratio = 1.0
                delayer._prefill_max_requests = 16
                delayer._max_delay_ms = 5.0
                delayer._max_delay_passes = 10000
                delayer.enable_dp_attention = True
                delayer.skip_first_delayer = False
                delayer._gather_info = MagicMock(return_value=torch.tensor(
                    [[1, 0, 0, 16, 1, int(shared_timeout)]], dtype=torch.int64
                ))
                with patch('sglang.srt.managers.prefill_delayer.time.perf_counter',
                           return_value=local_now):
                    result = delayer._negotiate_should_allow_prefill_pure(
                        prev_state=_State(delayed_count=7, start_time=10.0),
                        local_prefillable=True, token_usage=0.5,
                        running_batch=0, max_prefill_bs=16,
                        max_running_requests=256, waiting_queue_len=1,
                    )
                self.assertEqual(
                    delayer._gather_info.call_args.kwargs['wall_timeout'],
                    local_now >= 10.005,
                )
                decisions.append(result.output_allow)
            self.assertEqual(decisions, [shared_timeout, shared_timeout])

    def test_timeout_field_uses_existing_gather_and_tp0_selection(self):
        delayer = object.__new__(PrefillDelayer)
        delayer._gather_device = "cpu"
        delayer._gather_group = object()
        delayer._global_info_buffer = torch.empty((2, 2, 6), dtype=torch.int64)
        gathered = torch.tensor([
            [[1, 0, 0, 16, 1, 0], [1, 0, 0, 16, 1, 1]],
            [[1, 0, 0, 16, 1, 1], [1, 0, 0, 16, 1, 0]],
        ])

        def gather(output, local, group):
            self.assertIs(group, delayer._gather_group)
            self.assertEqual(local.tolist(), [1, 0, 0, 16, 1, 1])
            output.copy_(gathered.flatten())

        with patch("torch.distributed.all_gather_into_tensor", side_effect=gather) as call:
            result = delayer._gather_info(
                True, False, max_prefill_bs=16, waiting_queue_len=1,
                wall_timeout=True,
            )
        call.assert_called_once()
        self.assertTrue(torch.equal(result, gathered[:, 0, :]))

    def test_peak_expires_after_recent_attempt_window(self):
        tracker = RecentPrefillBatchSizeTracker(window_size=4)

        self.assertEqual(tracker.observe_attempt(100), 100)
        for attempted_prefill_bs in [2, 1, 2]:
            self.assertEqual(tracker.observe_attempt(attempted_prefill_bs), 100)

        self.assertEqual(tracker.observe_attempt(2), 2)

    def test_recurring_small_peak_remains_effective(self):
        tracker = RecentPrefillBatchSizeTracker(window_size=4)

        for attempted_prefill_bs in [2, 1, 1, 2, 1, 1, 2]:
            self.assertEqual(tracker.observe_attempt(attempted_prefill_bs), 2)

    def test_rejected_non_empty_attempt_advances_window(self):
        tracker = RecentPrefillBatchSizeTracker(window_size=4)
        self.assertEqual(tracker.observe_attempt(10), 10)

        delayer = MagicMock()
        delayer.enable_dp_attention = True
        delayer.dp_size = 1
        delayer._metrics_collector = None
        delayer._debug_log_enabled = False
        delayer._negotiate_should_allow_prefill.return_value = _NegotiateOutput(
            next_state=None,
            input_estimation="all",
            output_allow=False,
            output_reason="delay",
            num_prefillable=1,
            num_token_watermark_force_allow=0,
        )

        for _ in range(4):
            executor = PrefillDelayerSinglePassExecutor(delayer, token_usage=0.9)
            self.assertFalse(
                executor.negotiate_should_allow_prefill(
                    local_prefillable=True,
                    running_batch=15,
                    max_prefill_bs=tracker.max_prefill_bs,
                    max_running_requests=20,
                    waiting_queue_len=2,
                )
            )
            attempted_prefill_bs = executor.finalize(actual_prefill_bs=0)
            self.assertEqual(attempted_prefill_bs, 2)
            tracker.observe_attempt(attempted_prefill_bs)

        self.assertEqual(tracker.max_prefill_bs, 2)

    def test_print_rejected_attempt_estimates_after_unusual_peak(self):
        window_size = 16

        for steady_prefill_bs in (2, 3, 4):
            with self.subTest(steady_prefill_bs=steady_prefill_bs):
                tracker = RecentPrefillBatchSizeTracker(window_size=window_size)
                self.assertEqual(tracker.observe_attempt(100), 100)

                delayer = MagicMock()
                delayer.enable_dp_attention = True
                delayer.dp_size = 1
                delayer._metrics_collector = None
                delayer._debug_log_enabled = False
                delayer._negotiate_should_allow_prefill.return_value = _NegotiateOutput(
                    next_state=None,
                    input_estimation="all",
                    output_allow=False,
                    output_reason="delay",
                    num_prefillable=1,
                    num_token_watermark_force_allow=0,
                )

                print(
                    f"\nunusual_prefill_bs=100, "
                    f"subsequent_prefill_bs={steady_prefill_bs}, "
                    f"window_size={window_size}",
                    flush=True,
                )
                print(
                    "round | waiting_bs | estimated_rejected_bs | "
                    "high_watermark_before | high_watermark_after",
                    flush=True,
                )

                for round_index in range(1, window_size + 1):
                    high_watermark_before = tracker.max_prefill_bs
                    executor = PrefillDelayerSinglePassExecutor(
                        delayer, token_usage=0.9
                    )
                    self.assertFalse(
                        executor.negotiate_should_allow_prefill(
                            local_prefillable=True,
                            running_batch=20,
                            max_prefill_bs=high_watermark_before,
                            max_running_requests=128,
                            waiting_queue_len=steady_prefill_bs,
                        )
                    )
                    estimated_prefill_bs = executor.finalize(actual_prefill_bs=0)
                    high_watermark_after = tracker.observe_attempt(estimated_prefill_bs)

                    print(
                        f"{round_index:>5} | {steady_prefill_bs:>10} | "
                        f"{estimated_prefill_bs:>21} | "
                        f"{high_watermark_before:>21} | "
                        f"{high_watermark_after:>20}",
                        flush=True,
                    )
                    self.assertEqual(estimated_prefill_bs, steady_prefill_bs)
                    expected_high_watermark = (
                        100 if round_index < window_size else steady_prefill_bs
                    )
                    self.assertEqual(high_watermark_after, expected_high_watermark)

    def test_rejects_empty_attempts(self):
        with self.assertRaisesRegex(ValueError, "window_size must be positive"):
            RecentPrefillBatchSizeTracker(window_size=0)

        tracker = RecentPrefillBatchSizeTracker(window_size=4)

        with self.assertRaisesRegex(ValueError, "non-empty attempt"):
            tracker.observe_attempt(0)

        self.assertEqual(tracker.max_prefill_bs, 0)


if __name__ == "__main__":
    unittest.main()
