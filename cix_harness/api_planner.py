"""Responses API planner, sharing contracts with the board-local planner."""
import json
import os
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, build_opener

from .rpc import NoRedirect, MAX_BYTES
from .task_plan import VisualTaskPlanner
from .planner_images import AUX_WRIST_MAX_AGE_S, IMAGE_SYSTEM, select_images

DEFAULT_API_MODEL = 'gpt-6.2-astra'


def completed_stream(stream, timing_observer=None):
    """Read through completion; notify timing milestones without exposing content.

    Milestones mean a complete parsed SSE data event and a nonempty output-text
    delta respectively. Neither measures the first network byte or reasoning token.
    """
    total = 0
    data = []
    milestones = set()
    while True:
        line = stream.readline(MAX_BYTES + 1)
        total += len(line)
        if total > MAX_BYTES:
            raise ValueError('Planner API response exceeds limit')
        if not line:
            raise ValueError('Planner API stream ended before response.completed')
        line = line.decode('utf-8').rstrip('\r\n')
        if line.startswith('data:'):
            data.append(line[5:].lstrip(' '))
        elif not line and data:
            payload = '\n'.join(data)
            data = []
            if payload == '[DONE]':
                raise ValueError('Planner API stream ended before response.completed')
            event = json.loads(payload)
            kind = event.get('type')
            markers = ['first_sse_event']
            if kind == 'response.output_text.delta' and isinstance(event.get('delta'), str) and event['delta']:
                markers.append('first_output_text_delta')
            for marker in markers:
                if marker not in milestones:
                    milestones.add(marker)
                    if timing_observer is not None:
                        timing_observer(marker)
            if kind == 'response.completed':
                return event['response']
            if kind in ('response.failed', 'response.incomplete', 'error'):
                # Provider messages may contain request data. Log only a fixed status.
                raise RuntimeError('Planner API stream failed or incomplete')


class ResponsesTaskPlanner(VisualTaskPlanner):
    def __init__(self, model=DEFAULT_API_MODEL, base_url='https://api.openai.com/v1',
                 key_env='OPENAI_API_KEY', timeout=120, log=None,
                 auth='api-key', credential_path=None, image_mode='scene_wrist', wrist_max_age_s=AUX_WRIST_MAX_AGE_S):
        parsed=urlparse(base_url)
        if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('API base URL must be HTTPS without credentials/query/fragment')
        self.model=model;self.base_url=base_url.rstrip('/');self.timeout=timeout;self.log=log
        if image_mode not in ('scene','scene_wrist'):raise ValueError('Unknown API image mode')
        self.image_mode=image_mode;self.wrist_max_age_s=wrist_max_age_s;self._observation_seen=None
        if auth not in ('api-key', 'chatgpt'):raise ValueError('Unknown API authentication mode')
        self.auth=auth;self.credential_path=credential_path
        if auth=='chatgpt' and self.base_url!='https://api.openai.com/v1':
            raise ValueError('ChatGPT authorization requires the official Responses API endpoint')
        self.key=os.environ.get(key_env) if auth=='api-key' else None
        if auth=='api-key' and not self.key:raise ValueError('Set API credential in environment variable '+key_env)

    def request(self, name, system, schema, obs, context, tokens):
        total_began=time.monotonic()
        began=None;received=None;reply=None;auth_seconds=0.;error_type=None
        stage='request_prepare';completed=False;milestones={};image_metadata=None

        def observe_timing(marker):
            milestones[marker+'_seconds']=round(time.monotonic()-began,3)

        try:
            images,image_metadata=select_images(obs,self.image_mode,self.wrist_max_age_s)
            # The bridge clock is remote. Only report the known lower bound on
            # local reuse time, never subtract unsynchronised monotonic clocks.
            identity=obs.get('id')
            if identity is None or self._observation_seen is None or self._observation_seen[0]!=identity:
                self._observation_seen=(identity,total_began)
            image_metadata['observation_reuse_elapsed_s']=round(total_began-self._observation_seen[1],3)
            visual_context=dict(context,planner_images=image_metadata)
            body={'model':self.model,'store':False,'max_output_tokens':max(4096,tokens),
                  'input':[{'role':'developer','content':system+'\n'+IMAGE_SYSTEM},{'role':'user','content':[
                      {'type':'input_text','text':json.dumps(visual_context,ensure_ascii=False,separators=(',',':'))}]+
                      [{'type':'input_image','image_url':url} for url in images]}],
                  'text':{'format':{'type':'json_schema','name':name,'strict':True,'schema':schema}}}
            key=self.key
            if self.auth=='chatgpt':
                from .chatgpt_auth import get_access_token
                stage='authorization';auth_began=time.monotonic()
                try:key=get_access_token(self.credential_path)
                finally:auth_seconds=time.monotonic()-auth_began
                body.pop('max_output_tokens');body['stream']=True
            stage='request_prepare'
            request=Request(self.base_url+'/responses',data=json.dumps(body,allow_nan=False).encode(),
                            headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
            stage='request';began=time.monotonic()
            try:
                with build_opener(NoRedirect()).open(request,timeout=self.timeout) as stream:
                    stage='response_read'
                    if self.auth=='chatgpt':
                        reply=completed_stream(stream,observe_timing)
                    else:
                        raw=stream.read(MAX_BYTES+1)
                        if len(raw)>MAX_BYTES:raise ValueError('Planner API response exceeds limit')
                        reply=json.loads(raw)
                received=time.monotonic()
            except HTTPError as error:
                # Do not log the request, key or an arbitrary provider response body.
                raise RuntimeError('Planner API HTTP '+str(error.code)) from None
            except (URLError,TimeoutError):
                raise RuntimeError('Planner API connection failed; no motor call issued by planner') from None
            stage='decision_parse'
            if reply.get('status')!='completed':raise ValueError('Planner API response incomplete or failed')
            parts=[]
            for item in reply.get('output',[]):
                if item.get('type')=='message':
                    for part in item.get('content',[]):
                        if part.get('type')=='refusal':raise RuntimeError('Planner API declined the request')
                        if part.get('type')=='output_text':parts.append(part['text'])
            if not parts:raise ValueError('Planner API returned no decision text')
            result=json.loads(''.join(parts));completed=True
            return result
        except BaseException as error:
            error_type=type(error).__name__
            raise
        finally:
            ended=time.monotonic()
            if self.log:
                usage=reply.get('usage') if isinstance(reply,dict) else None
                usage=usage if isinstance(usage,dict) else {}
                counts={k:usage[k] for k in ('input_tokens','output_tokens','total_tokens')
                        if type(usage.get(k)) is int and usage[k]>=0}
                if 'total_tokens' not in counts and 'input_tokens' in counts and 'output_tokens' in counts:
                    counts['total_tokens']=counts['input_tokens']+counts['output_tokens']
                response_id=reply.get('id') if isinstance(reply,dict) else None
                if (not isinstance(response_id,str) or not response_id.startswith('resp_') or len(response_id)>200
                        or any(not (c.isascii() and (c.isalnum() or c in '_-')) for c in response_id)):
                    response_id=None
                response_model=reply.get('model') if isinstance(reply,dict) else None
                if not isinstance(response_model,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}',response_model):
                    response_model=None
                fields={'phase':name,'model':self.model,'response_model':response_model,'status':'completed' if completed else 'error',
                        'seconds':round((received if received is not None else ended)-began,3) if began is not None else None,
                        'total_seconds':round(ended-total_began,3),'auth_seconds':round(auth_seconds,3),
                        'first_sse_event_seconds':milestones.get('first_sse_event_seconds'),
                        'first_output_text_delta_seconds':milestones.get('first_output_text_delta_seconds'),
                        'usage':counts,'response_id':response_id,'images':image_metadata}
                if error_type:fields.update(error_type=error_type,error_stage=stage)
                self.log('planner_api',**fields)
