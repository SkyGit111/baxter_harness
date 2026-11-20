"""Monotonic component timings. Nested durations must not be added together."""
import math
import time
import uuid


class TimedRpc:
    def __init__(self, client, log, component):
        self.client, self.log, self.component = client, log, component

    def select(self,arm):
        return self.client.select(arm)

    def call(self, method, *args, **kwargs):
        began=time.monotonic();result=None;okay=False;call_id=uuid.uuid4().hex
        self.log('rpc_started',call_id=call_id,component=self.component,method=method)
        try:
            result=self.client.call(method,*args,**kwargs);okay=True
            return result
        finally:
            fields={'component':self.component,'method':method,
                    'call_id':call_id,'duration_s':round(time.monotonic()-began,6),'succeeded':okay}
            if isinstance(result,dict) and isinstance(result.get('timings_s'),dict):
                fields['server_timings_s']=result['timings_s']
            self.log('component_timing',**fields)


class CallChain:
    """Ordered task/backend/RPC/phase ledger, without image or credential payloads."""
    def __init__(self):self.context={};self.rows=[]

    def record(self,item):
        event=item['event']
        if event=='task_started':self.context={'episode':item.get('episode'),'route':item.get('route')}
        if event in ('task_step_dispatch','task_step_executed','task_step_failed'):
            step=item.get('step',{})
            self.context.update(step_id=step.get('id'),backend=step.get('backend'),
                                skill=step.get('skill'),arm=step.get('arm'))
            if event=='task_step_dispatch':self.context.pop('plan_id',None)
        if event=='primitive_plan' and item.get('plan_id'):self.context['plan_id']=item['plan_id']
        keys=('t','seq','event','call_id','component','method','duration_s','succeeded',
              'phase','seconds','total_seconds','auth_seconds','first_output_text_delta_seconds',
              'status','failure_code','reason','plan_id','nominal_motion_seconds','selected_approach_path')
        if event in ('task_started','task_plan','task_step_dispatch','rpc_started','component_timing',
                     'planner_api','primitive_plan','primitive_result','recovery','task_step_executed',
                     'task_step_failed','finish','stop','task_finished'):
            row={**self.context,**{k:item[k] for k in keys if k in item}}
            if event=='task_plan':row['steps']=item.get('steps',[])
            if event=='primitive_plan' and item.get('moveit_full_path'):
                row['moveit_full_path']=item['moveit_full_path']
            if event=='primitive_result':
                row.update({k:item[k] for k in ('applied','outcome_uncertain','stopped','stop_attempted','retryable','message','failed_stage','confirmed_commands') if k in item})
                row['stages']=[{k:s[k] for k in ('stage_index','phase','duration_s','timings_s','motion_detail','moveit') if k in s}
                               for s in item.get('completed_stages',[])]
            self.rows.append(row)

    def snapshot(self):
        return {'format':'harness-call-chain-v1','calls':self.rows,
                'note':'Chronological calls; RPC/server/stage timings overlap. Join robot audit by plan_id. A missing completion is unknown, not retried.'}


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
            failed=fields.get('failed_stage')
            if isinstance(failed,dict):self.add('primitive_failed_stage.'+failed.get('phase','unknown'),failed.get('duration_s'))

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
