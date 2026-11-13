import io
import json
import unittest
from unittest.mock import Mock, patch

from cix_harness.api_planner import ResponsesTaskPlanner, completed_stream


def sse(*events):
    return io.BytesIO(b''.join(b'data: '+json.dumps(event).encode()+b'\n\n' for event in events))


class ChatGPTPlanner(unittest.TestCase):
    def test_account_token_and_streaming_contract(self):
        completed={'type':'response.completed','response':{'status':'completed','output':[
            {'type':'message','content':[{'type':'output_text','text':'{"ok":true}'}]}]}}
        opener=Mock()
        opener.open.return_value=sse({'type':'response.output_text.delta','delta':'invalid partial'},completed)
        planner=ResponsesTaskPlanner(auth='chatgpt',credential_path='own-project-store')
        with patch('cix_harness.chatgpt_auth.get_access_token',return_value='test-project-token') as token, \
                patch('cix_harness.api_planner.build_opener',return_value=opener):
            result=planner.request('check','decide',{}, {'image':'data:image/jpeg;base64,AA=='},{},200)
        self.assertEqual(result,{'ok':True})
        token.assert_called_once_with('own-project-store')
        request=opener.open.call_args.args[0]
        self.assertEqual(request.get_header('Authorization'),'Bearer test-project-token')
        body=json.loads(request.data)
        self.assertTrue(body['stream']);self.assertFalse(body['store'])
        self.assertNotIn('max_output_tokens',body)

    def test_partial_or_failed_stream_never_returns_a_decision(self):
        partial={'type':'response.output_text.delta','delta':'{"ok":true}'}
        for end in (None,'response.failed','response.incomplete','error'):
            stream=sse(partial,*([{'type':end,'message':'sensitive provider content'}] if end else []))
            with self.assertRaises((RuntimeError,ValueError)) as error:
                completed_stream(stream)
            self.assertNotIn('sensitive',str(error.exception))
        with self.assertRaises(ValueError):completed_stream(io.BytesIO(b'data: [DONE]\n\n'))

    def test_stream_byte_limit_and_endpoint(self):
        with patch('cix_harness.api_planner.MAX_BYTES',32):
            with self.assertRaisesRegex(ValueError,'limit'):completed_stream(io.BytesIO(b'data: '+b'x'*64))
        for url in ('https://example.org/v1','https://api.openai.com/v1/other'):
            with self.assertRaisesRegex(ValueError,'official'):
                ResponsesTaskPlanner(auth='chatgpt',base_url=url)

