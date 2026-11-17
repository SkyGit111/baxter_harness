import unittest
from cix_harness.timing import CallChain


class CallChainTests(unittest.TestCase):
    def test_backend_switch_and_unfinished_call_remain_distinguishable(self):
        log=CallChain()
        events=[{'event':'task_started','episode':'e','route':'hybrid'},
                {'event':'task_step_dispatch','step':{'id':'a','backend':'primitives','skill':'pick','arm':'right'}},
                {'event':'primitive_plan','plan_id':'p'},
                {'event':'rpc_started','call_id':'c','method':'primitive.execute'},
                {'event':'component_timing','call_id':'c','method':'primitive.execute','duration_s':4.,'succeeded':False},
                {'event':'task_step_dispatch','step':{'id':'b','backend':'code','skill':'compute','arm':'right'}},
                {'event':'rpc_started','call_id':'d','method':'compute'}]
        for i,item in enumerate(events):log.record(dict(item,t=i,seq=i))
        rows=log.snapshot()['calls']
        self.assertEqual(rows[3]['plan_id'],'p');self.assertEqual(rows[3]['backend'],'primitives')
        self.assertFalse(rows[4]['succeeded'])
        self.assertEqual(rows[-1]['backend'],'code');self.assertNotIn('plan_id',rows[-1])
        self.assertNotIn('succeeded',rows[-1])
