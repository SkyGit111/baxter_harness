"""Primitive completion evidence comes from measured stages, not sparse frame deltas."""
import unittest
from unittest.mock import Mock

from cix_harness.executive import TaskExecutive, primitive_measurements
from cix_harness.memory import EpisodeMemory


def stage(kind,z,gripping,**fields):
    return {'stage':dict(kind=kind,**fields),'eef_m':[.6,.2,z],'gripping':gripping}


def pick_receipt(lift=.2):
    return {'skill':'pick','status':'completed','applied':True,'gripping':True,'completed_stages':[
        stage('move',.1,False),stage('move',.1,False),stage('move',.1,False),
        stage('gripper',.1,False,open=True),stage('move',-.1,False),
        stage('gripper',-.1,True,open=False),stage('verify_grasp',-.1,True,held=True),
        stage('move',-.1+lift,True),stage('verify_grasp',-.1+lift,True,held=True)]}


def place_receipt():
    return {'skill':'place','status':'completed','applied':True,'gripping':False,'completed_stages':[
        stage('move',.1,True),stage('move',.1,True),stage('move',.1,True),stage('move',-.1,True),
        stage('gripper',-.1,False,open=True),stage('verify_grasp',-.1,False,held=False),stage('move',.1,False)]}


def observation(held,z=.1):
    return {'id':'after','state':{'eef_m':[.6,.2,z],'gripping':held,'gripper_percent':20 if held else 100}}


class PrimitiveEvidence(unittest.TestCase):
    def executive(self,skill):
        memory=EpisodeMemory(':memory:');self.addCleanup(memory.close)
        e=TaskExecutive('synthetic task','primitives',Mock(),memory,None,Mock(),mode='live')
        e.steps=[{'id':'s1','skill':skill,'target':'synthetic cube','expect':'test',
                  'instruction':'test','on_failure':'stop','status':'active'}]
        e.dispatch='s1';e.last_action_obs='before';e.origin=[.6,.2,.1]
        e.monitor.observe(observation(skill=='place')['state'])
        return e

    def test_pick_returns_to_start_height_but_measured_close_to_lift_is_success(self):
        e=self.executive('pick');e.event('primitive_result',pick_receipt())
        obs=observation(True);e.observe(obs);evidence=e.step_evidence(obs)
        self.assertEqual(evidence['distance_from_start_m'],0)
        self.assertEqual(evidence['commands_applied'],7)
        self.assertEqual(evidence['lift_distance_m'],.2)
        self.assertTrue(evidence['held_and_lifted']);self.assertIsNone(evidence['confirmed_release'])
        self.assertTrue(e.can_verify(obs))

    def test_place_exposes_confirmed_release_and_actual_operation_count(self):
        e=self.executive('place');e.event('primitive_result',place_receipt())
        obs=observation(False);e.observe(obs);evidence=e.step_evidence(obs)
        self.assertEqual(evidence['commands_applied'],6)
        self.assertTrue(evidence['confirmed_release']);self.assertTrue(e.released)
        self.assertIsNone(evidence['held_and_lifted']);self.assertIsNone(evidence['released_after_lift'])
        self.assertFalse(evidence['drop_detected']);self.assertTrue(e.can_verify(obs))

    def test_completed_pick_with_insufficient_measured_lift_is_not_verified(self):
        e=self.executive('pick');e.event('primitive_result',pick_receipt(.01))
        obs=observation(True,-.09);e.observe(obs)
        self.assertFalse(e.step_evidence(obs)['held_and_lifted'])
        self.assertFalse(e.can_verify(obs))

    def test_release_requires_open_followed_by_empty_verification(self):
        for mutation in ('missing','reordered'):
            receipt=place_receipt()
            if mutation=='missing':del receipt['completed_stages'][5]
            else:receipt['completed_stages'][4],receipt['completed_stages'][5]=receipt['completed_stages'][5],receipt['completed_stages'][4]
            e=self.executive('place');e.event('primitive_result',receipt)
            obs=observation(False);e.observe(obs)
            self.assertFalse(e.step_evidence(obs)['confirmed_release']);self.assertFalse(e.can_verify(obs))

    def test_final_receipt_grasp_state_must_match_measured_pick_or_release(self):
        for skill,receipt,state in [('pick',pick_receipt(),True),('place',place_receipt(),False)]:
            receipt['gripping']=not state
            e=self.executive(skill);e.event('primitive_result',receipt)
            obs=observation(state);e.observe(obs)
            self.assertFalse(e.can_verify(obs))

    def test_fresh_grasp_loss_overrides_prior_successful_lift(self):
        e=self.executive('pick');e.event('primitive_result',pick_receipt())
        obs=observation(False);e.observe(obs)
        self.assertTrue(e.step_evidence(obs)['drop_detected']);self.assertFalse(e.can_verify(obs))

    def test_legacy_completion_has_unknown_measurements_not_false_or_zero(self):
        e=self.executive('pick')
        e.event('primitive_result',{'skill':'pick','status':'completed','applied':True,'gripping':True})
        obs=observation(True);e.observe(obs);evidence=e.step_evidence(obs)
        self.assertEqual(evidence['source'],'primitive_completion_contract')
        for key in ('commands_applied','held_and_lifted','confirmed_release'):
            self.assertIsNone(evidence[key])
        self.assertTrue(e.can_verify(obs))

    def test_invalid_measured_position_does_not_become_success_evidence(self):
        receipt=pick_receipt();receipt['completed_stages'][-1]['eef_m'][2]=float('nan')
        e=self.executive('pick')
        with self.assertRaisesRegex(ValueError,'finite measured'):e.event('primitive_result',receipt)
        self.assertFalse(e.primitive_ok)


if __name__=='__main__':unittest.main()
