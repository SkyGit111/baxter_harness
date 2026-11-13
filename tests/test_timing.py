import unittest
from unittest.mock import Mock,patch
from cix_harness.timing import timed_rpc,TimingSummary


class TimingTests(unittest.TestCase):
    def test_rpc_failure_timed_without_retry_or_payload(self):
        client=Mock();client.call.side_effect=RuntimeError('private-body')
        log=Mock();rpc=timed_rpc(client,log)
        with patch('cix_harness.timing.time.monotonic',side_effect=[4.,6.5]):
            with self.assertRaises(RuntimeError):rpc.call('primitive.execute',('token',))
        client.call.assert_called_once_with('primitive.execute',('token',))
        fields=log.call_args.kwargs
        self.assertFalse(fields['succeeded']);self.assertEqual(fields['duration_s'],2.5)
        self.assertNotIn('private-body',str(fields));self.assertNotIn('token',str(fields))
        self.assertIs(timed_rpc(rpc,log),rpc)

    def test_success_preserves_receipt_and_server_timings(self):
        reply={'status':'ready','timings_s':{'ik_preflight':.003}}
        client=Mock();client.call.return_value=reply;log=Mock()
        self.assertIs(timed_rpc(client,log).call('primitive.prepare'),reply)
        self.assertEqual(log.call_args.kwargs['server_timings_s'],reply['timings_s'])

    def test_summary_missing_milestone_is_not_zero_and_nested_not_added(self):
        summary=TimingSummary()
        summary.record('planner_api',{'seconds':4.,'first_sse_event_seconds':1.,'first_output_text_delta_seconds':None})
        summary.record('planner_api',{'seconds':8.,'first_sse_event_seconds':2.,'first_output_text_delta_seconds':7.})
        summary.record('component_timing',{'component':'robot_rpc','method':'primitive.prepare','duration_s':.1,'server_timings_s':{'ik_preflight':.01}})
        result=summary.snapshot(20.)
        self.assertEqual(result['wall_seconds'],20.)
        stats=result['components']
        self.assertEqual(stats['api.seconds']['mean_s'],6.)
        self.assertEqual(stats['api.first_output_text_delta_seconds']['count'],1)
        self.assertEqual(stats['api.first_output_text_delta_seconds']['mean_s'],7.)
        self.assertEqual(stats['robot_rpc.primitive.prepare']['total_s'],.1)
        self.assertEqual(stats['robot_rpc.primitive.prepare.server.ik_preflight']['total_s'],.01)
