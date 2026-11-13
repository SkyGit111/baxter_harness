import json
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from cix_harness.api_planner import ResponsesTaskPlanner
from cix_harness.task_plan import validate_plan, validate_check
from cix_harness.executive import TaskExecutive
from cix_harness.memory import EpisodeMemory
from cix_harness.executive import ExecutiveJournal
from cix_harness.code_tool import CodeTool
from cix_harness.hybrid import run as run_hybrid
from test_executive import step, check, obs, ScriptedPlanner


def hybrid_step(skill,backend):return dict(step(skill),backend=backend)
def hybrid_check(kind='act',uv=None,code=''):return dict(check(kind,uv),code=code)


class APIPlanner(unittest.TestCase):
    def test_api_vla_route_does_not_require_local_vlm(self):
        from cix_harness.__main__ import check_models
        from cix_harness.contract import ACTION_SPEC
        policy=Mock();policy.call.return_value={'action_spec':ACTION_SPEC}
        with patch('cix_harness.__main__.local_json') as local, patch('cix_harness.__main__.HttpRpcClient',return_value=policy):
            check_models(NS(planner_provider='api',vla_port=9101))
        local.assert_not_called()

    def planner(self):
        with patch.dict('os.environ',{'HARNESS_TEST_KEY':'dummy-test-key'}):
            return ResponsesTaskPlanner(key_env='HARNESS_TEST_KEY')

    def test_response_request_and_parsing_with_reasoning_items(self):
        client=self.planner();stream=Mock()
        stream.read.return_value=json.dumps({'status':'completed','output':[
            {'type':'reasoning','summary':[]},
            {'type':'message','content':[{'type':'output_text','text':'{"steps":[]}'}]}]}).encode()
        opener=Mock();opener.open.return_value.__enter__=Mock(return_value=stream)
        opener.open.return_value.__exit__=Mock(return_value=False)
        with patch('cix_harness.api_planner.build_opener',return_value=opener):
            self.assertEqual(client.plan(obs(),{'route':'hybrid'}),{'steps':[]})
        request=opener.open.call_args.args[0];body=json.loads(request.data)
        self.assertFalse(body['store']);self.assertEqual(body['model'],'gpt-6-astra')
        self.assertEqual(body['input'][1]['content'][1]['type'],'input_image')
        self.assertIn('backend',body['text']['format']['schema']['properties']['steps']['items']['required'])

    def test_no_key_bad_url_and_error_body_never_exposed(self):
        with patch.dict('os.environ',{},clear=True):
            with self.assertRaisesRegex(ValueError,'environment'):ResponsesTaskPlanner()
        for url in ('http://example.com','https://user:secret@example.com','https://example.com?key=secret'):
            with self.assertRaises(ValueError):ResponsesTaskPlanner(base_url=url)
        client=self.planner();opener=Mock()
        opener.open.side_effect=HTTPError('https://example.com',401,'dummy-test-key',{},None)
        with patch('cix_harness.api_planner.build_opener',return_value=opener):
            with self.assertRaisesRegex(RuntimeError,'^Planner API HTTP 401$'):client.plan(obs(),{})
        self.assertEqual(opener.open.call_count,1)


class HybridExecutive(unittest.TestCase):
    def test_full_executive_code_then_primitive_then_vla(self):
        from test_hybrid import Robot, Policy, Dispatch
        robot=Robot();model=Policy()
        planner=ScriptedPlanner([[hybrid_step('compute','code'),hybrid_step('pick','primitives'),hybrid_step('place','vla')]],
            [hybrid_check(code='result = sum([1, 2])'),hybrid_check('step_done'),hybrid_check(uv=[.4,.5]),
             hybrid_check('step_done'),hybrid_check(),hybrid_check('step_done'),hybrid_check('task_done')])
        memory=EpisodeMemory(':memory:');self.addCleanup(memory.close)
        journal=Mock();e=TaskExecutive('把青色块放到桌面','hybrid',planner,memory,'test',journal,mode='live')
        result=run_hybrid(Dispatch().args(),ExecutiveJournal(journal,e),e,robot,model,CodeTool())
        self.assertEqual(result,0);self.assertTrue(e.task_verified)
        self.assertEqual([s['backend'] for s in e.verified],['code','primitives','vla'])
        self.assertEqual(memory.db.execute('SELECT status FROM episodes').fetchone()[0],'verified')

    def frame(self,i=1,held=False):
        o=obs(i,held);o['backend_skills']={'primitives':['pick','place'],'vla':['pick','place'],'code':['compute']}
        return o

    def test_backend_must_exist_and_uv_follows_selected_backend(self):
        with self.assertRaises(ValueError):validate_plan({'steps':[hybrid_step('pick','vla')]},['pick'],False,{'primitives':['pick']})
        validate_check(hybrid_check(uv=[.4,.5]),'hybrid',hybrid_step('pick','primitives'))
        with self.assertRaises(ValueError):validate_check(hybrid_check(uv=[.4,.5]),'hybrid',hybrid_step('pick','vla'))

    def test_compute_result_survives_into_motor_step_context(self):
        planner=ScriptedPlanner([[hybrid_step('compute','code'),hybrid_step('pick','primitives')]],
                  [hybrid_check(code='result = 2 + 3'),hybrid_check('step_done'),hybrid_check(uv=[.4,.5])])
        memory=EpisodeMemory(':memory:');self.addCleanup(memory.close)
        e=TaskExecutive('抓起青色块','hybrid',planner,memory,None,Mock(),mode='live')
        self.assertEqual(e.choose(self.frame())['backend'],'code')
        e.event('code_result',{'status':'completed','result':5,'applied':False})
        result=e.choose(self.frame(2))
        self.assertEqual(result['backend'],'primitives')
        self.assertEqual(planner.contexts[-1][1]['computed_results'][0]['result'],5)
        self.assertFalse(e.task_verified)

    def test_code_only_cannot_certify_a_physical_goal(self):
        planner=ScriptedPlanner([[hybrid_step('compute','code')]],
                 [hybrid_check(code='result = True'),hybrid_check('step_done'),hybrid_check('task_done'),hybrid_check('task_done')])
        memory=EpisodeMemory(':memory:');self.addCleanup(memory.close)
        e=TaskExecutive('抓起方块','hybrid',planner,memory,None,Mock(),mode='live')
        e.choose(self.frame());e.event('code_result',{'status':'completed','result':True,'applied':False})
        self.assertEqual(e.choose(self.frame(2))['decision'],'blocked')
        self.assertFalse(e.task_verified)

    def test_replan_can_switch_physical_backend(self):
        first=hybrid_step('pick','primitives');first['on_failure']='replan'
        planner=ScriptedPlanner([[first],[hybrid_step('pick','vla')]],
                              [hybrid_check(uv=[.4,.5]),hybrid_check()])
        memory=EpisodeMemory(':memory:');self.addCleanup(memory.close)
        e=TaskExecutive('抓起方块','hybrid',planner,memory,None,Mock(),mode='live')
        self.assertEqual(e.choose(self.frame())['backend'],'primitives')
        e.event('primitive_result',{'skill':'pick','status':'failed','failure_code':'empty_grasp','applied':True})
        self.assertEqual(e.choose(self.frame(2))['backend'],'vla')
