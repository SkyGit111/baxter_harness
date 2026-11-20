"""Select an arm on the serialized shared-camera bridge without changing geometry."""
class ArmClient:
    def __init__(self, client, health, arms='auto'):
        self.client=client
        self.dual=health.get('dual_protocol')=='serialized_dual_primitives_v1'
        available=health.get('available_arms',[]) if self.dual else [health.get('arm')]
        requested=available if arms in ('auto','both') else [arms]
        if arms=='both' and not self.dual:raise ValueError('Both arms require the shared dual bridge')
        if arms=='both' and set(available)!={'left','right'}:
            raise ValueError('Both arms must be available for a two-arm task')
        if not requested or any(a not in ('left','right') or a not in available for a in requested):
            raise ValueError('Requested arm is not available on this bridge')
        self.allowed=list(requested);self.arm=self.allowed[0]

    def select(self,arm):
        if arm not in self.allowed:raise ValueError('Planner selected an unavailable arm')
        self.arm=arm

    def call(self,method,args=(),kwargs=None,**options):
        if method=='healthz':
            health=dict(self.client.call(method,args,kwargs=kwargs,**options))
            if self.dual:
                if health.get('dual_protocol')!='serialized_dual_primitives_v1':
                    raise RuntimeError('Dual bridge protocol changed')
                health.update(motion_scope='supervised_primitive_task',arm=','.join(self.allowed))
            return health
        if self.dual and method.startswith('primitive.'):
            result=self.client.call('dual.'+method.split('.',1)[1],(self.arm,*args),kwargs=kwargs,**options)
            if result.get('selected_arm')!=self.arm:raise RuntimeError('RPC response belongs to another arm')
            if method=='primitive.observe':
                if result.get('state',{}).get('arm')!=self.arm:raise RuntimeError('Observation belongs to another arm')
                result={**result,'available_arms':self.allowed,
                        'arm_states':{a:result['arm_states'][a] for a in self.allowed}}
            return result
        return self.client.call(method,args,kwargs=kwargs,**options)
