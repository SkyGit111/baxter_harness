"""Monotonic component timings. Nested durations must not be added together."""
import math
import time


class TimedRpc:
    def __init__(self, client, log, component):
        self.client, self.log, self.component = client, log, component

    def call(self, method, *args, **kwargs):
        began=time.monotonic();result=None;okay=False
        try:
            result=self.client.call(method,*args,**kwargs);okay=True
            return result
        finally:
            fields={'component':self.component,'method':method,
                    'duration_s':round(time.monotonic()-began,6),'succeeded':okay}
            if isinstance(result,dict) and isinstance(result.get('timings_s'),dict):
                fields['server_timings_s']=result['timings_s']
            self.log('component_timing',**fields)


def timed_rpc(client,log,component='robot_rpc'):
    return client if isinstance(client,TimedRpc) else TimedRpc(client,log,component)


class TimingSummary:
    def __init__(self):self.samples={}

    def add(self,name,value):
        if type(value) in (int,float) and math.isfinite(value) and value>=0:
            self.samples.setdefault(name,[]).append(value)

    def record(self,event,fields):
        if event=='component_timing':
            name=fields['component']+'.'+fields.get('method','total')
            self.add(name,fields.get('duration_s'))
            for key,value in fields.get('server_timings_s',{}).items():
                self.add(name+'.server.'+key,value)
        elif event=='planner_api':
            for key in ('seconds','total_seconds','auth_seconds','first_sse_event_seconds','first_output_text_delta_seconds'):
                self.add('api.'+key,fields.get(key))
        elif event=='primitive_result':
            for stage in fields.get('completed_stages',[]):
                self.add('primitive_stage.'+stage.get('stage',{}).get('kind','unknown'),stage.get('duration_s'))

    def snapshot(self,wall_seconds):
        result={}
        for name,samples in self.samples.items():
            ordered=sorted(samples);count=len(ordered)
            median=(ordered[(count-1)//2]+ordered[count//2])/2
            result[name]={'count':count,'total_s':round(sum(samples),6),
                          'mean_s':round(sum(samples)/count,6),'p50_s':round(median,6),
                          'p95_s':round(ordered[max(0,math.ceil(.95*count)-1)],6),
                          'min_s':round(ordered[0],6),'max_s':round(ordered[-1],6)}
        return {'wall_seconds':round(wall_seconds,3),'components':result,
                'note':'RPC, server, planner and stage durations overlap. Do not sum nested categories. Null/unobserved milestones are omitted, not zero.'}
