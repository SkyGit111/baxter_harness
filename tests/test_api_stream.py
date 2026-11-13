import io
import json
import unittest
from unittest.mock import Mock, patch

from cix_harness.api_planner import ResponsesTaskPlanner, completed_stream


def sse(*events):
    return io.BytesIO(b''.join(b'data: '+json.dumps(event).encode()+b'\n\n' for event in events))


class Clock:
    def __init__(self):self.now=100.
    def __call__(self):return self.now


class ScheduledStream(io.BytesIO):
    """Advance only when a complete simulated SSE event arrives."""
    def __init__(self,clock,schedule):
        super().__init__(b'')
        self.clock=clock;self.lines=[]
        for when,event in schedule:
            self.lines.extend([(when,b'data: '+json.dumps(event).encode()+b'\n'),(when,b'\n')])
    def readline(self,_limit=-1):
        if not self.lines:return b''
        when,line=self.lines.pop(0);self.clock.now=when
        return line


def completed_reply(**changes):
    value={'status':'completed','id':'resp_test','output':[
        {'type':'message','content':[{'type':'output_text','text':'{"ok":true}'}]}]}
    value.update(changes)
    return {'type':'response.completed','response':value}


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


class APIRequestTiming(unittest.TestCase):
    def test_stream_milestones_are_complete_event_and_nonempty_output_delta(self):
        log=Mock();clock=Clock();opener=Mock()
        stream=ScheduledStream(clock,[(106,{'type':'response.created'}),
            (108,{'type':'response.output_text.delta','delta':''}),
            (109,{'type':'response.reasoning_text.delta','delta':'private reasoning'}),
            (110,{'type':'response.output_text.delta','delta':'private answer'}),
            (112,{'type':'response.output_text.delta','delta':'more text'}),
            (116,completed_reply(usage={'input_tokens':100,'output_tokens':20,'provider_detail':'private'}))])
        def authorize(_path):clock.now=102;return 'private credential'
        def open_response(*_args,**_kwargs):clock.now=105;return stream
        opener.open.side_effect=open_response
        planner=ResponsesTaskPlanner(auth='chatgpt',log=log)
        with patch('cix_harness.api_planner.time.monotonic',clock), \
                patch('cix_harness.chatgpt_auth.get_access_token',side_effect=authorize), \
                patch('cix_harness.api_planner.build_opener',return_value=opener):
            self.assertEqual(planner.request('check','private prompt',{}, {'image':'private camera frame'},{},200),{'ok':True})
        log.assert_called_once()
        self.assertEqual(log.call_args.args[0],'planner_api')
        fields=log.call_args.kwargs
        self.assertEqual(fields['seconds'],14.)
        self.assertEqual(fields['total_seconds'],16.)
        self.assertEqual(fields['auth_seconds'],2.)
        self.assertEqual(fields['first_sse_event_seconds'],4.)
        self.assertEqual(fields['first_output_text_delta_seconds'],8.)
        self.assertEqual(fields['status'],'completed')
        self.assertEqual(fields['usage'],{'input_tokens':100,'output_tokens':20,'total_tokens':120})
        self.assertNotIn('private',json.dumps(fields))

    def test_observer_receives_only_two_fixed_milestones_never_payload(self):
        observer=Mock()
        completed_stream(sse({'type':'response.created','secret':'never pass me'},
            {'type':'response.output_text.delta','delta':''},
            {'type':'response.output_text.delta','delta':'first'},
            {'type':'response.output_text.delta','delta':'second'},completed_reply()),observer)
        self.assertEqual([call.args for call in observer.call_args_list],
                         [('first_sse_event',),('first_output_text_delta',)])

    def test_error_stream_records_elapsed_time_and_does_not_fabricate_ttft(self):
        clock=Clock();log=Mock();opener=Mock()
        opener.open.return_value=ScheduledStream(clock,[(103,{'type':'response.created'}),
            (107,{'type':'response.failed','message':'private provider failure'})])
        planner=ResponsesTaskPlanner(auth='chatgpt',log=log)
        with patch('cix_harness.api_planner.time.monotonic',clock), \
                patch('cix_harness.chatgpt_auth.get_access_token',return_value='private credential'), \
                patch('cix_harness.api_planner.build_opener',return_value=opener):
            with self.assertRaises(RuntimeError):planner.request('check','decide',{}, {'image':'test'},{},200)
        fields=log.call_args.kwargs
        self.assertEqual(fields['seconds'],7.)
        self.assertEqual(fields['first_sse_event_seconds'],3.)
        self.assertIsNone(fields['first_output_text_delta_seconds'])
        self.assertEqual(fields['status'],'error')
        self.assertEqual(fields['error_stage'],'response_read')
        self.assertNotIn('private',json.dumps(fields))

    def test_auth_failure_has_auth_total_and_no_http_duration(self):
        clock=Clock();log=Mock();planner=ResponsesTaskPlanner(auth='chatgpt',log=log)
        def authorize(_path):
            clock.now=104
            raise RuntimeError('private auth failure')
        with patch('cix_harness.api_planner.time.monotonic',clock), \
                patch('cix_harness.chatgpt_auth.get_access_token',side_effect=authorize), \
                patch('cix_harness.api_planner.build_opener') as opener:
            with self.assertRaises(RuntimeError):planner.request('check','decide',{}, {'image':'test'},{},200)
        opener.assert_not_called()
        fields=log.call_args.kwargs
        self.assertEqual(fields['auth_seconds'],4.)
        self.assertEqual(fields['total_seconds'],4.)
        self.assertIsNone(fields['seconds'])
        self.assertEqual(fields['error_stage'],'authorization')
        self.assertNotIn('private',json.dumps(fields))

    def test_nonstreaming_logs_no_stream_milestones_and_filters_usage(self):
        clock=Clock();log=Mock();opener=Mock();stream=Mock()
        reply=completed_reply(id={'private':'provider payload'},usage={'input_tokens':'private','output_tokens':True,'total_tokens':-1})['response']
        def read_response(_limit):clock.now=105;return json.dumps(reply).encode()
        stream.read.side_effect=read_response
        opener.open.return_value.__enter__=Mock(return_value=stream)
        opener.open.return_value.__exit__=Mock(return_value=False)
        with patch.dict('os.environ',{'TEST_KEY':'dummy'}):planner=ResponsesTaskPlanner(key_env='TEST_KEY',log=log)
        with patch('cix_harness.api_planner.time.monotonic',clock), \
                patch('cix_harness.api_planner.build_opener',return_value=opener):
            planner.request('plan','decide',{}, {'image':'test'},{},200)
        fields=log.call_args.kwargs
        self.assertEqual(fields['seconds'],5.)
        self.assertEqual(fields['auth_seconds'],0.)
        self.assertIsNone(fields['first_sse_event_seconds'])
        self.assertIsNone(fields['first_output_text_delta_seconds'])
        self.assertEqual(fields['usage'],{})
        self.assertIsNone(fields['response_id'])
        self.assertNotIn('private',json.dumps(fields))

    def test_complete_response_with_invalid_decision_logs_parse_failure(self):
        log=Mock();opener=Mock()
        opener.open.return_value=sse(completed_reply(output=[{'type':'message','content':[
            {'type':'output_text','text':'not valid json'}]}]))
        planner=ResponsesTaskPlanner(auth='chatgpt',log=log)
        with patch('cix_harness.chatgpt_auth.get_access_token',return_value='synthetic'), \
                patch('cix_harness.api_planner.build_opener',return_value=opener):
            with self.assertRaises(ValueError):planner.request('check','decide',{}, {'image':'test'},{},200)
        self.assertEqual(log.call_args.kwargs['status'],'error')
        self.assertEqual(log.call_args.kwargs['error_stage'],'decision_parse')

