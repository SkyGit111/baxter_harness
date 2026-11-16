import math
import unittest
from cix_harness.pi05_contract import rotate
from cix_harness.tcp_calibration import fit


class PivotCalibration(unittest.TestCase):
    def samples(self):
        offset=[.012,-.007,.11];point=[.6,-.2,-.17];result=[]
        for axis,angle in [(0,0),(0,.7),(1,.8),(2,.6),(0,-.6),(1,-.7)]:
            q=[0.,0.,0.,math.cos(angle/2)];q[axis]=math.sin(angle/2)
            p=[x-y for x,y in zip(point,rotate(q,offset))]
            result.append({'arm':'right','eef_m':p,'eef_xyzw':q})
        return result,offset
    def test_recovers_offset_without_numpy_or_motion(self):
        samples,expected=self.samples();r=fit(samples)
        for a,b in zip(r['tcp_offset_tool_m'],expected):self.assertAlmostEqual(a,b,places=9)
        self.assertFalse(r['commissioned'])
    def test_rejects_same_orientation_and_wrong_contact(self):
        samples,_=self.samples()
        with self.assertRaises(ValueError):fit([samples[0]]*6)
        samples[-1]['eef_m'][0]+=.08
        with self.assertRaises(ValueError):fit(samples)
