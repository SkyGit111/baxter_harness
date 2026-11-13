"""Responses API planner, sharing contracts with the board-local planner."""
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, build_opener

from .rpc import NoRedirect, MAX_BYTES
from .task_plan import VisualTaskPlanner


def completed_stream(stream):
    """Read SSE through its terminal event; partial JSON is never a decision."""
    total = 0
    data = []
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
            if kind == 'response.completed':
                return event['response']
            if kind in ('response.failed', 'response.incomplete', 'error'):
                # Provider messages may contain request data. Log only a fixed status.
                raise RuntimeError('Planner API stream failed or incomplete')


class ResponsesTaskPlanner(VisualTaskPlanner):
    def __init__(self, model='gpt-6-astra', base_url='https://api.openai.com/v1',
                 key_env='OPENAI_API_KEY', timeout=120, log=None,
                 auth='api-key', credential_path=None):
        parsed=urlparse(base_url)
        if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('API base URL must be HTTPS without credentials/query/fragment')
        self.model=model;self.base_url=base_url.rstrip('/');self.timeout=timeout;self.log=log
        if auth not in ('api-key', 'chatgpt'):raise ValueError('Unknown API authentication mode')
        self.auth=auth;self.credential_path=credential_path
        if auth=='chatgpt' and self.base_url!='https://api.openai.com/v1':
            raise ValueError('ChatGPT authorization requires the official Responses API endpoint')
        self.key=os.environ.get(key_env) if auth=='api-key' else None
        if auth=='api-key' and not self.key:raise ValueError('Set API credential in environment variable '+key_env)

    def request(self, name, system, schema, obs, context, tokens):
        body={'model':self.model,'store':False,'max_output_tokens':max(4096,tokens),
              'input':[{'role':'developer','content':system},{'role':'user','content':[
                  {'type':'input_text','text':json.dumps(context,ensure_ascii=False,separators=(',',':'))},
                  {'type':'input_image','image_url':obs['image']}]}],
              'text':{'format':{'type':'json_schema','name':name,'strict':True,'schema':schema}}}
        key=self.key
        if self.auth=='chatgpt':
            from .chatgpt_auth import get_access_token
            key=get_access_token(self.credential_path)
            body.pop('max_output_tokens')
            body['stream']=True
        request=Request(self.base_url+'/responses',data=json.dumps(body,allow_nan=False).encode(),
                        headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
        began=time.monotonic()
        try:
            with build_opener(NoRedirect()).open(request,timeout=self.timeout) as stream:
                if self.auth=='chatgpt':
                    reply=completed_stream(stream)
                else:
                    raw=stream.read(MAX_BYTES+1)
                    if len(raw)>MAX_BYTES:raise ValueError('Planner API response exceeds limit')
                    reply=json.loads(raw)
        except HTTPError as error:
            # Do not log the request, key or an arbitrary provider response body.
            raise RuntimeError('Planner API HTTP '+str(error.code)) from None
        except (URLError,TimeoutError):
            raise RuntimeError('Planner API connection failed; no motor call issued by planner') from None
        if self.log:
            usage=reply.get('usage') or {}
            counts={k:usage[k] for k in ('input_tokens','output_tokens','total_tokens') if k in usage}
            self.log('planner_api',phase=name,model=self.model,seconds=round(time.monotonic()-began,3),
                     usage=counts,response_id=reply.get('id'))
        if reply.get('status')!='completed':raise ValueError('Planner API response incomplete or failed')
        parts=[]
        for item in reply.get('output',[]):
            if item.get('type')=='message':
                for part in item.get('content',[]):
                    if part.get('type')=='refusal':raise RuntimeError('Planner API declined the request')
                    if part.get('type')=='output_text':parts.append(part['text'])
        if not parts:raise ValueError('Planner API returned no decision text')
        return json.loads(''.join(parts))
