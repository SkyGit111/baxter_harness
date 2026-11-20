import json
import tempfile
import unittest
from pathlib import Path

from cix_harness.motion_report import summarize


class MotionReport(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)

    def write(self, name, events):
        path = self.folder/name
        path.write_text(''.join(json.dumps(e)+'\n' for e in events), encoding='utf-8')
        return path

    def test_legacy_receipts_report_unknown_clock_split_and_measured_translation(self):
        target = {'kind':'move','tcp_m':[.1,0,0],'xyzw':[1,0,0,0]}
        self.write('events.jsonl', [
            {'event':'observation','obs_id':'o','state':{'eef_m':[0,0,0], 'eef_xyzw':[0,1,0,0]}},
            {'event':'primitive_plan','plan_id':'p','obs_id':'o','status':'ready','skill':'pick','stages':[target]},
            {'event':'planner_api','phase':'task_checkpoint','seconds':3.2},
            {'event':'primitive_result','plan_id':'p','status':'completed','completed_stages':[
                {'stage':target,'duration_s':8.1,'eef_m':[.09,0,0]}]},
            {'event':'task_finished','status':'verified'}])
        report = summarize(self.folder)
        stage = report['primitives'][0]['stages'][0]
        self.assertIsNone(stage['scheduled_seconds'])
        self.assertEqual(stage['duration_s'], 8.1)
        self.assertEqual(stage['requested_rotation_deg'], 180.)
        self.assertAlmostEqual(stage['measured_eef_displacement_m'], .09)
        self.assertIsNone(report['api_calls'][0]['first_output_text_delta_seconds'])

    def test_failed_rpc_uses_matched_robot_detail_and_does_not_call_it_completion(self):
        stage = {'kind':'move','motion_phase':'approach'}
        self.write('events.jsonl', [
            {'event':'primitive_plan','plan_id':'p','skill':'pick','status':'ready','stages':[stage]},
            {'event':'component_timing','component':'robot_rpc','method':'dual.execute','succeeded':False,'duration_s':5},
            {'event':'task_finished','status':'unknown'},
            {'event':'stop','reason':'tracking'}])
        self.assertEqual(summarize(self.folder)['primitives'][0]['execution_status'], 'unknown')
        audit = self.write('robot.jsonl', [
            {'event':'primitive_stage_applied','plan_id':'other','stage':stage},
            {'event':'segment_execution_failed','context':{'plan_id':'p','stage_index':0},
             'scheduled_seconds':5.6,'bottleneck_joint':'right_s1','feedback_slowdown_s':.9},
            {'event':'primitive_failed','plan_id':'p','error':'tracking','failed_stage':{
                'stage_index':0,'stage':stage,'status':'failed','duration_s':4.3}}])
        report = summarize(self.folder,audit)
        row = report['primitives'][0]
        self.assertEqual(report['status'], 'unknown')
        self.assertEqual(row['execution_status'], 'failed_in_robot_audit')
        self.assertEqual(len(row['stages']), 1)
        self.assertEqual(row['stages'][0]['status'], 'failed')
        self.assertEqual(row['stages'][0]['bottleneck_joint'], 'right_s1')
        self.assertEqual(row['stages'][0]['feedback_slowdown_s'], .9)
        self.assertEqual(report['failures'][-1]['reason'], 'tracking')

    def test_untagged_legacy_segment_requires_a_single_dispatch_and_matching_stage_receipt(self):
        target = {'kind':'move'}
        self.write('events.jsonl', [
            {'event':'primitive_plan','plan_id':'p','skill':'pick','status':'ready','stages':[target]}])
        failure = {'event':'primitive_failed','plan_id':'p',
                   'failed_stage':{'stage':target, 'status':'failed', 'duration_s':63.1}}
        segment = {'event':'segment_arrival_timeout','retimed_seconds':47.5,'scheduled_seconds':30.3}
        for active, seconds in [([], None), (['p'], 47.5), (['p','q'], None)]:
            with self.subTest(active=active):
                audit = self.write('audit.jsonl', [
                    *[{'event':'primitive_dispatch','plan_id':k} for k in active], segment, failure])
                row = summarize(self.folder,audit)['primitives'][0]['stages'][0]
                self.assertEqual(row['retimed_seconds'],seconds)
