"""CPU-only checks for timestamp DMA ownership and nonblocking polling."""

import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.utils.realtime_trace_readback import RealtimeTraceReadback


class TestRealtimeTraceReadback(unittest.TestCase):
    def test_pending_buffer_is_not_read_or_reused(self):
        markers, host, event, stream = (MagicMock() for _ in range(4))
        with patch("torch.empty_like", return_value=host) as alloc, patch(
            "torch.cuda.Event", return_value=event
        ), patch("torch.cuda.current_stream", return_value=stream):
            rb = RealtimeTraceReadback(markers)
            alloc.assert_called_once_with(markers, device="cpu", pin_memory=True)
            self.assertIsNone(rb.sample(markers, 16))
            host.copy_.assert_called_once_with(markers.detach(), non_blocking=True)
            host.tolist.assert_not_called()
            event.record.assert_called_once_with(stream)
            event.query.return_value = False
            self.assertIsNone(rb.sample(markers, 32))
            self.assertEqual(host.copy_.call_count, 1)
            host.tolist.assert_not_called()
            event.synchronize.assert_not_called()
            event.query.return_value = True
            host.tolist.return_value = [100, 120]
            self.assertEqual(rb.sample(markers, 48), (16, [100, 120]))
            self.assertEqual(host.copy_.call_count, 2)
            self.assertEqual(rb.sequence, 48)
            markers.cpu.assert_not_called()

    def test_rejects_stream_switch_instead_of_racing_source(self):
        markers, host, event, a, b = (MagicMock() for _ in range(5))
        with patch("torch.empty_like", return_value=host), patch(
            "torch.cuda.Event", return_value=event
        ), patch("torch.cuda.current_stream", side_effect=[a, b]):
            rb = RealtimeTraceReadback(markers)
            rb.sample(markers, 16)
            with self.assertRaisesRegex(RuntimeError, "one replay stream"):
                rb.sample(markers, 32)
            self.assertEqual(host.copy_.call_count, 1)


if __name__ == "__main__":
    unittest.main()
