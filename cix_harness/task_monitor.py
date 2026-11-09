"""Measured grasp/lift evidence and bounded progress, separate from VLM claims."""
import math


class TaskMonitor:
    def __init__(self, kind='generic'):
        self.kind=kind
        self.hold_origin=None
        self.held_and_lifted=False
        self.released_after_lift=False
        self.drop_detected=False
        self.positions=[]
        self.last_state={}

    def observe(self,state, release_commanded=False):
        previously_held=self.last_state.get('gripping') is True
        self.last_state=state
        position=state.get('eef_m')
        if position is None:
            return
        if state.get('gripping') is True:
            if self.hold_origin is None:
                self.hold_origin=position[2]
                self.held_and_lifted=False
                self.released_after_lift=False
            if position[2]-self.hold_origin >= .025:
                self.held_and_lifted=True
        else:
            if previously_held:
                if release_commanded and state.get('gripper_percent',0)>70:
                    self.released_after_lift=self.held_and_lifted
                else:
                    self.drop_detected=True
                    self.released_after_lift=False
                self.held_and_lifted=False
            self.hold_origin=None
        self.positions.append(list(position))
        self.positions=self.positions[-8:]

    def applied(self,receipt,action):
        if 'gripping' in receipt and 'endpoint_after_m' in receipt:
            self.observe({'eef_m':receipt['endpoint_after_m'],
                          'gripping':receipt['gripping'],
                          'gripper_percent':receipt.get('gripper_percent')},
                         release_commanded=action[6]>=.5)

    def require_progress(self):
        if self.kind != 'generic' and self.drop_detected:
            raise RuntimeError('Grasp feedback was lost without a confirmed release')

    def evidence(self):
        return {'held_and_lifted':self.held_and_lifted,'released_after_lift':self.released_after_lift,
                'drop_detected':self.drop_detected,'gripping':self.last_state.get('gripping'),
                'gripper_percent':self.last_state.get('gripper_percent')}

    def can_complete(self):
        if self.kind=='generic': return True
        if self.kind=='pick':
            return self.held_and_lifted and self.last_state.get('gripping') is True
        return (not self.drop_detected and self.released_after_lift and
                self.last_state.get('gripping') is False and self.last_state.get('gripper_percent',0)>70)

    def stalled(self):
        return len(self.positions)>=6 and max(math.dist(self.positions[0],p) for p in self.positions)<.005
