import contextlib
import io
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from cix_harness import handeye


def pose(theta,axis,position):
    v=np.asarray(axis,dtype=float)/np.linalg.norm(axis)
    return handeye.transform({'xyz':list(position),'xyzw':(math.sin(theta/2.)*v).tolist()+[math.cos(theta/2.)]})


class HandeyeTest(unittest.TestCase):
    def setUp(self):
        self.x=pose(2.4,[4,-3,1],[.45,-.1,.9])
        self.y=pose(-.3,[1,.2,.6],[.01,.095,.04])
        self.pairs=[]
        for i,axis in enumerate(([1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1],[1,2,3],[2,3,1])):
            e=pose(.1*i+.25,axis,[.4+i*.027,.1-i*.015,.2+.05*math.sin(i)])
            c=np.linalg.solve(self.x,e @ self.y)
            self.pairs.append((e,c))

    def test_cli_solves_heldout_with_fixed_tool_and_correct_direction(self):
        with TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            fit,checks,out=[Path(folder)/n for n in ('fit.json','all.json','result.json')]
            data=[{'base_T_effector':e.tolist(),'camera_T_tag':c.tolist()} for e,c in self.pairs]
            fit.write_text(json.dumps(data[:-3]));checks.write_text(json.dumps({'samples':data}))
            self.assertEqual(handeye.main(['--fit',str(fit),'--holdout',str(checks),'--holdout-includes-fit',
                                          '--output',str(out),'--max-mm','1','--max-deg','.2']),0)
            result=json.loads(out.read_text())
            self.assertTrue(result['consistency_passed'])
            self.assertFalse(result['commissioned'])
            np.testing.assert_allclose(handeye.transform(result['base_T_camera']),self.x,atol=1e-12)
            np.testing.assert_allclose(handeye.transform(result['effector_T_tag']),self.y,atol=1e-12)
            self.assertEqual(result['holdout']['count'],3)

    def test_independent_outlier_and_wrong_existing_camera_are_rejected(self):
        fit,held=self.pairs[:-3],self.pairs[-3:]
        e,c=held[0];bad=c.copy();bad[:3,3]+=[0,.07,0]
        result=handeye.solve(fit,[(e,bad),*held[1:]],5,2)
        self.assertFalse(result['consistency_passed'])
        self.assertAlmostEqual(result['holdout']['max_mm'],70.,places=6)
        self.assertFalse(handeye.solve(fit,held,5,2,np.linalg.inv(self.x))['consistency_passed'])

    def test_inconsistent_pose_formats_and_nonindependent_samples(self):
        m=np.eye(4);m[1,1]=-1
        with self.assertRaisesRegex(ValueError,'right-handed'):handeye.transform(m)
        with self.assertRaisesRegex(ValueError,'Duplicate pose'):
            handeye.solve(self.pairs[:-3],self.pairs[:3],5,1)

    def test_single_axis_is_unobservable(self):
        pairs=[]
        for i in range(7):
            e=pose(.2*i,[0,0,1],[i*.03,.1,.2])
            pairs.append((e,np.linalg.solve(self.x,e@self.y)))
        with self.assertRaisesRegex(ValueError,'independent rotation axes'):
            handeye.solve(pairs[:-3],pairs[-3:],5,1)

    def test_easy_handeye_direction_for_real_service_format(self):
        def ros(m):
            t=handeye.packed(m)
            return {'translation':dict(zip('xyz',t['xyz'])),'rotation':dict(zip('xyzw',t['xyzw']))}
        a,b=self.pairs[2]
        (got_arm,got_tag),=handeye.pose_arrays({'samples':{'hand_world_samples':[ros(np.linalg.inv(a))],
                                                          'camera_marker_samples':[ros(b)]}})
        np.testing.assert_allclose(got_arm,a,atol=1e-12)
        np.testing.assert_allclose(got_tag,b,atol=1e-12)
