import json
import math
from pathlib import Path
import unittest

from cix_harness.planning import candidate_profiles
from cix_harness.primitives import eef_from_tcp,plan


class PlacementOrientation(unittest.TestCase):
    def test_selected_pick_yaw_is_preserved_by_later_place_and_stack(self):
        profile=json.loads(Path('config/baxter-primitives.example.json').read_text(encoding='utf-8'))['primitives']
        profile['tcp_workspace_min_m'][2]=-.17
        # A non-axial measured TCP makes an unintended wrist rotation translate
        # the fingers as well as rotating them.
        profile['tcp_offset_tool_m']=[.01,-.005,.025]
        initial={'eef_m':[.6,.2,.1],'eef_xyzw':[1.,0.,0.,0.],'gripping':False}
        _,selected=next(candidate_profiles(profile,initial,'pick',{'grasp_yaw_offsets_rad':[math.pi/2]}))
        picked=plan('pick',initial,selected,[.7,.25,-.12])
        last=next(s for s in reversed(picked) if s['kind']=='move')
        held={'eef_m':eef_from_tcp(last['tcp_m'],last['xyzw'],profile),
              'eef_xyzw':last['xyzw'],'gripping':True}
        self.assertGreater(math.dist(held['eef_xyzw'],profile['tool_orientation_xyzw']),.5)
        for skill in ('place','stack'):
            # Runtime restores the original config; the offline evaluator has
            # the selected config. Both must produce exactly the same motions.
            online=plan(skill,held,profile,[.6,.15,-.12])
            offline=plan(skill,held,selected,[.6,.15,-.12])
            self.assertEqual(online,offline)
            for stage in online:
                if stage['kind']=='move':self.assertEqual(stage['xyzw'],held['eef_xyzw'])
            self.assertEqual([s for s in online if s['kind']=='gripper'],[{'kind':'gripper','open':True}])

    def test_preserving_orientation_does_not_authorize_empty_gripper_placement(self):
        profile=json.loads(Path('config/baxter-primitives.example.json').read_text(encoding='utf-8'))['primitives']
        state={'eef_m':[.6,.2,.1],'eef_xyzw':[1.,0.,0.,0.],'gripping':False}
        for skill in ('place','stack'):
            with self.assertRaisesRegex(ValueError,'held object'):plan(skill,state,profile,[.6,.2,-.1])
