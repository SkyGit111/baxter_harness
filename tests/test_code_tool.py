import unittest

from cix_harness.code_tool import CodeTool


class ComputationalCode(unittest.TestCase):
    def test_computes_selection_and_keeps_input_unchanged(self):
        context={'objects':[{'name':'red','size':4},{'name':'cyan','size':2},{'name':'yellow','size':3}]}
        source='''best = data["objects"][0]
for item in data["objects"]:
    if item["size"] < best["size"]:
        best = item
result = {"target": best["name"], "count": len(data["objects"])}
'''
        output=CodeTool().execute(source,context)
        self.assertEqual(output,{'status':'completed','result':{'target':'cyan','count':3},'applied':False})
        self.assertEqual(context['objects'][0]['name'],'red')

    def test_bounded_arithmetic_sequences_and_builtins(self):
        source='''values = []
for i in range(4):
    values = values + [i * 2]
result = {"values": sorted(values)[1:3], "total": sum(values), "span": max(values)-min(values), "half": round(7 / 2, 1)}
'''
        self.assertEqual(CodeTool().execute(source,{})['result'],{'values':[2,4],'total':12,'span':6,'half':3.5})

    def test_rejects_host_access_even_in_dead_branch(self):
        for source in ['import os\nresult = 0','result = open("secret")','result = data.__class__',
                       'result = __import__("os")','if False:\n    import socket\nresult = 1',
                       'result = eval("1")','result = (lambda: 1)()', 'result = [v for v in range(2)]']:
            with self.subTest(source=source),self.assertRaises(ValueError):CodeTool().execute(source,{})

    def test_rejects_unbounded_work_before_large_allocation(self):
        for source in ['while True:\n    pass','result = 2 ** 1000000000','result = [0] * 1000000000',
                       'result = "x" * 1000000000','result = range(1000000000)','result = 1e309',
                       'result = 1000000000000 * 1000000000000']:
            with self.subTest(source=source),self.assertRaises(ValueError):CodeTool().execute(source,{})

    def test_loop_fuel_and_value_depth_are_bounded(self):
        with self.assertRaisesRegex(ValueError,'budget'):
            CodeTool(max_steps=80).execute('result=0\nfor i in range(100):\n    result=result+i',{})
        with self.assertRaisesRegex(ValueError,'deep'):
            CodeTool().execute('result=0\nfor i in range(30):\n    result=[result]',{})

    def test_rejects_non_json_context_missing_result_and_mutation(self):
        for context in [{'object':object()},{'nonfinite':float('nan')},{'key':{1:'bad'}}]:
            with self.assertRaises(ValueError):CodeTool().execute('result=data',context)
        for source in ['value=1','data={}','len=2','data["a"]=2','result = range(3,0,0)']:
            with self.subTest(source=source),self.assertRaises(ValueError):CodeTool().execute(source,{'a':1})

    def test_unknown_calls_cannot_be_supplied_by_context(self):
        with self.assertRaises(ValueError):CodeTool().execute('result=robot(data)',{'robot':'not callable'})
        self.assertEqual(CodeTool().execute('result = 1 if "a" in data and not False else 0',{'a':2})['result'],1)

    def test_result_and_context_serialization_limits(self):
        with self.assertRaises(ValueError):CodeTool().execute('result=data',{'values':['x'*4096]*5})
        with self.assertRaises(ValueError):CodeTool().execute('result=["x"*4096]*5',{})


if __name__=='__main__':unittest.main()
