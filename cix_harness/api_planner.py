"""Responses API planner, sharing contracts with the board-local planner."""
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, build_opener

from .rpc import NoRedirect, MAX_BYTES
from .task_plan import VisualTaskPlanner


class ResponsesTaskPlanner(VisualTaskPlanner):
    def __init__(self, model='gpt-6-astra', base_url='https://api.openai.com/v1',
                 key_env='OPENAI_API_KEY', timeout=120, log=None):
        parsed=urlparse(base_url)
        if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('API base URL must be HTTPS without credentials/query/fragment')
        self.model=model;self.base_url=base_url.rstrip('/');self.timeout=timeout;self.log=log
        self.key=os.environ.get(key_env)
        if not self.key:raise ValueError('Set API credential in environment variable '+key_env)

    def request(self, name, system, schema, obs, context, tokens):
        body={'model':self.model,'store':False,'max_output_tokens':max(4096,tokens),
              'input':[{'role':'developer','content':system},{'role':'user','content':[
                  {'type':'input_text','text':json.dumps(context,ensure_ascii=False,separators=(',',':'))},
                  {'type':'input_image','image_url':obs['image']}]}],
              'text':{'format':{'type':'json_schema','name':name,'strict':True,'schema':schema}}}
        request=Request(self.base_url+'/responses',data=json.dumps(body,allow_nan=False).encode(),
                        headers={'Content-Type':'application/json','Authorization':'Bearer '+self.key})
        began=time.monotonic()
        try:
            with build_opener(NoRedirect()).open(request,timeout=self.timeout) as stream:
                raw=stream.read(MAX_BYTES+1)
        except HTTPError as error:
            # Do not log the request, key or an arbitrary provider response body.
            raise RuntimeError('Planner API HTTP '+str(error.code)) from None
        except (URLError,TimeoutError):
            raise RuntimeError('Planner API connection failed; no motor call issued by planner') from None
        if len(raw)>MAX_BYTES:raise ValueError('Planner API response exceeds limit')
        reply=json.loads(raw)
        if self.log:self.log('planner_api',phase=name,model=self.model,seconds=round(time.monotonic()-began,3),
                             usage=reply.get('usage'),response_id=reply.get('id'))
        if reply.get('status')!='completed':raise ValueError('Planner API response incomplete or failed')
        parts=[]
        for item in reply.get('output',[]):
            if item.get('type')=='message':
                for part in item.get('content',[]):
                    if part.get('type')=='refusal':raise RuntimeError('Planner API declined the request')
                    if part.get('type')=='output_text':parts.append(part['text'])
        if not parts:raise ValueError('Planner API returned no decision text')
        return json.loads(''.join(parts))
