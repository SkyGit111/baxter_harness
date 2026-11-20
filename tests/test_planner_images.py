import json
from pathlib import Path
import unittest
from unittest.mock import Mock,patch

from cix_harness.api_planner import ResponsesTaskPlanner
from cix_harness.contract import image_data
from cix_harness.planner_images import select_images,image_info

IMAGE=image_data((Path(__file__).parent/'fixtures/white-pixel.png').read_bytes())


class PlannerImages(unittest.TestCase):
    def frame(self):
        return {'id':'current','image':IMAGE,'wrist_image':IMAGE,
                'state':{'arm':'right','wrist_age_s':.2},
                'camera_metadata':{'scene':{'age_s':.05},'wrist':{'arm':'right','age_s':.25}}}

    def test_current_two_views_have_distinct_roles_and_bounded_dimensions(self):
        images,meta=select_images(self.frame())
        self.assertEqual(len(images),2)
        self.assertEqual([r['source'] for r in meta['views']],['d455_scene','right_wrist'])
        self.assertEqual([r['role'] for r in meta['views']],['uv_reference','auxiliary_rgb'])
        self.assertEqual(meta['views'][1]['capture_age_at_observation_s'],.25)
        self.assertEqual(meta['views'][0]['pixels'],1)

    def test_stale_unknown_mismatched_or_invalid_wrist_falls_back_to_scene(self):
        for change,reason in [(lambda o:o['camera_metadata']['wrist'].update(age_s=.9),'stale_at_observation'),
                              (lambda o:o['camera_metadata']['wrist'].update(age_s=None),'unknown_age'),
                              (lambda o:o['camera_metadata']['wrist'].update(arm='left'),'arm_mismatch'),
                              (lambda o:o.update(wrist_image='not an image'),'invalid_image')]:
            obs=self.frame();change(obs)
            images,meta=select_images(obs)
            self.assertEqual(images,[IMAGE]);self.assertEqual(meta['wrist_omitted_reason'],reason)
        obs=self.frame();del obs['state']['arm']
        self.assertEqual(select_images(obs)[1]['wrist_omitted_reason'],'unknown_arm')

    def test_old_observation_format_and_explicit_scene_only_are_supported(self):
        obs=self.frame();del obs['camera_metadata']
        self.assertEqual(select_images(obs)[1]['views'][1]['capture_age_at_observation_s'],.2)
        images,meta=select_images(obs,'scene')
        self.assertEqual(images,[IMAGE]);self.assertEqual(meta['wrist_omitted_reason'],'disabled')
        del obs['wrist_image']
        self.assertEqual(select_images(obs)[1]['wrist_omitted_reason'],'missing')

    def test_api_sends_both_images_and_logs_metadata_without_image_payload(self):
        opener=Mock();stream=Mock();log=Mock()
        stream.read.return_value=json.dumps({'status':'completed','model':'provider-model-2026-11-20','output':[{'type':'message','content':[
            {'type':'output_text','text':'{"ok":true}'}]}]}).encode()
        opener.open.return_value.__enter__=Mock(return_value=stream)
        opener.open.return_value.__exit__=Mock(return_value=False)
        with patch.dict('os.environ',{'TEST_KEY':'synthetic'}):planner=ResponsesTaskPlanner(key_env='TEST_KEY',log=log)
        with patch('cix_harness.api_planner.build_opener',return_value=opener):
            planner.request('test','system',{},self.frame(),{},200)
        body=json.loads(opener.open.call_args.args[0].data)
        content=body['input'][1]['content']
        self.assertEqual([x['type'] for x in content],['input_text','input_image','input_image'])
        self.assertEqual([x['image_url'] for x in content if x['type']=='input_image'],[IMAGE,IMAGE])
        self.assertEqual(log.call_args.kwargs['response_model'],'provider-model-2026-11-20')
        self.assertEqual(log.call_args.kwargs['model'],body['model'])
        self.assertIn('ONLY pixel coordinate frame for uv',body['input'][0]['content'])
        self.assertEqual(log.call_args.kwargs['images']['image_count'],2)
        self.assertNotIn('base64',json.dumps(log.call_args.kwargs))

    def test_invalid_scene_is_rejected_before_network(self):
        with patch.dict('os.environ',{'TEST_KEY':'synthetic'}):planner=ResponsesTaskPlanner(key_env='TEST_KEY')
        with patch('cix_harness.api_planner.build_opener') as network:
            with self.assertRaises(ValueError):planner.request('test','system',{}, {'image':'garbage'},{},200)
        network.assert_not_called()

    def test_image_dimension_parser_rejects_corrupt_or_unbounded_images(self):
        self.assertEqual(image_info(IMAGE)['width'],1)
        for raw in (b'\xff\xd8\xff',b'\x89PNG\r\n\x1a\n'+b'\0'*20):
            with self.assertRaises(ValueError):image_info(image_data(raw))

    def test_offline_comparison_never_creates_api_client(self):
        from scripts.compare_planner_views import compare
        with patch('scripts.compare_planner_views.ResponsesTaskPlanner') as api:
            result=compare(self.frame(),{'route':'hybrid','current':{'skill':'pick'}})
        api.assert_not_called()
        self.assertEqual(result['robot_calls'],0)
        self.assertEqual(result['trials'],[])
        self.assertEqual(result['images']['scene']['image_count'],1)
        self.assertEqual(result['images']['scene_wrist']['image_count'],2)

    def test_comparison_does_not_spend_tokens_on_two_identical_single_view_inputs(self):
        from scripts.compare_planner_views import compare
        obs=self.frame();obs.pop('wrist_image')
        with patch('scripts.compare_planner_views.ResponsesTaskPlanner') as api:
            with self.assertRaisesRegex(ValueError,'Cannot compare'):
                compare(obs,{},call_api=True)
        api.assert_not_called()

    def test_api_rejects_legacy_wrist_primary_before_robot_calls(self):
        from types import SimpleNamespace
        from cix_harness.harness import run
        robot=Mock()
        with self.assertRaisesRegex(ValueError,'D455 as the primary'):
            run(SimpleNamespace(planner_provider='api',planner_view='wrist'),Mock(),robot=robot)
        robot.call.assert_not_called()
