"""Bounded Python-like data computation, interpreted without eval/exec or host tools.

The model reads JSON-compatible ``data`` and assigns JSON-compatible ``result``.
This is a computational helper, never a robot actuator or general Python sandbox.
"""
import ast
import json
import math
import operator
import time


class CodeTool:
    def __init__(self, max_steps=10000, timeout_s=1.0):
        if type(max_steps) is not int or not 1 <= max_steps <= 100000:
            raise ValueError('Invalid code step budget')
        if type(timeout_s) not in (int, float) or not 0 < timeout_s <= 5:
            raise ValueError('Invalid code time budget')
        self.max_steps, self.timeout_s = max_steps, timeout_s

    def execute(self, source, context):
        if not isinstance(source, str) or not 0 < len(source.encode('utf-8')) <= 8192:
            raise ValueError('Code must contain 1-8192 UTF-8 bytes')
        interpreter = _Interpreter(self.max_steps, self.timeout_s)
        try:
            interpreter.check_value(context)
            payload = json.dumps(context, allow_nan=False)
            if len(payload.encode('utf-8')) > 16384: raise ValueError('Code context too large')
            interpreter.names = {'data': json.loads(payload)}
            tree = ast.parse(source, mode='exec')
            interpreter.validate(tree)
            interpreter.statements(tree.body)
            if 'result' not in interpreter.names: raise ValueError('Code must assign result')
            result = interpreter.names['result']
            interpreter.check_value(result)
            encoded = json.dumps(result, allow_nan=False)
            if len(encoded.encode('utf-8')) > 16384: raise ValueError('Code result too large')
            return {'status': 'completed', 'result': json.loads(encoded), 'applied': False}
        except (SyntaxError, TypeError, KeyError, IndexError, ZeroDivisionError,
                OverflowError, RecursionError) as error:
            raise ValueError('Invalid computational code: '+str(error)[:160]) from error


class _Interpreter:
    FUNCTIONS = {'len', 'min', 'max', 'sum', 'sorted', 'range', 'abs', 'round'}
    ALLOWED = (ast.Module, ast.Assign, ast.For, ast.If, ast.Expr, ast.Pass,
               ast.Name, ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Subscript,
               ast.Slice, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
               ast.Call, ast.Load, ast.Store, ast.Add, ast.Sub, ast.Mult, ast.Div,
               ast.FloorDiv, ast.Mod, ast.UAdd, ast.USub, ast.Not, ast.And, ast.Or,
               ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn)

    def __init__(self, steps, timeout_s):
        self.remaining = steps
        self.deadline = time.monotonic()+timeout_s
        self.names = {}

    def tick(self):
        self.remaining -= 1
        if self.remaining < 0 or time.monotonic() > self.deadline:
            raise ValueError('Computational code budget exhausted')

    def validate(self, tree):
        pending = [(tree, 0)]; count = 0
        while pending:
            node, depth = pending.pop(); count += 1
            if count > 1000 or depth > 40: raise ValueError('Code AST too large or deep')
            # Python 3.8 wraps a subscription in Index; Python 3.9+ does not.
            if not isinstance(node, self.ALLOWED+(ast.Index,)):
                raise ValueError('Unsupported code syntax: '+type(node).__name__)
            if isinstance(node, ast.Name) and node.id.startswith('_'):
                raise ValueError('Private names are not available')
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in self.FUNCTIONS or node.keywords:
                    raise ValueError('Only named computational builtins without keywords are available')
            if isinstance(node, ast.Assign):
                if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                    raise ValueError('Only simple local assignment is supported')
                self.check_target(node.targets[0].id)
            if isinstance(node, ast.For):
                if not isinstance(node.target, ast.Name) or node.orelse:
                    raise ValueError('Only simple bounded for loops are supported')
                self.check_target(node.target.id)
            pending.extend((child, depth+1) for child in ast.iter_child_nodes(node))

    def check_target(self, name):
        if name == 'data' or name in self.FUNCTIONS or name.startswith('_'):
            raise ValueError('Cannot overwrite a reserved code name')

    def check_value(self, value):
        pending = [(value, 0)]; count = 0
        while pending:
            item, depth = pending.pop(); count += 1
            if count > 4096 or depth > 16: raise ValueError('Code value too large or deep')
            if count % 64 == 0: self.tick()
            kind = type(item)
            if item is None or kind is bool: continue
            if kind in (int, float):
                if not math.isfinite(item) or abs(item) > 1e12: raise ValueError('Code number out of bounds')
            elif kind is str:
                if len(item) > 4096: raise ValueError('Code string too large')
            elif kind in (list, dict):
                if len(item) > 256: raise ValueError('Code container too large')
                if kind is dict:
                    if any(type(key) is not str for key in item): raise ValueError('Code object keys must be strings')
                    pending.extend((key, depth+1) for key in item)
                    pending.extend((child, depth+1) for child in item.values())
                else: pending.extend((child, depth+1) for child in item)
            else: raise ValueError('Only JSON-native code values are supported')
        return value

    def statements(self, body):
        for node in body:
            self.tick()
            if isinstance(node, ast.Assign): self.names[node.targets[0].id] = self.expression(node.value)
            elif isinstance(node, ast.If): self.statements(node.body if self.expression(node.test) else node.orelse)
            elif isinstance(node, ast.For):
                items = self.expression(node.iter)
                if type(items) not in (list, dict, str) or len(items) > 256:
                    raise ValueError('For loop needs a bounded JSON sequence')
                for value in items:
                    self.tick(); self.names[node.target.id] = value; self.statements(node.body)
            elif isinstance(node, ast.Expr): self.expression(node.value)
            elif not isinstance(node, ast.Pass): raise ValueError('Unsupported statement')

    def expression(self, node):
        self.tick()
        if isinstance(node, ast.Constant): value = node.value
        elif isinstance(node, ast.Name):
            if node.id not in self.names: raise ValueError('Unknown code variable: '+node.id)
            value = self.names[node.id]
        elif isinstance(node, (ast.List, ast.Tuple)): value = [self.expression(v) for v in node.elts]
        elif isinstance(node, ast.Dict):
            if any(key is None for key in node.keys): raise ValueError('Dictionary unpacking is unavailable')
            value = {self.expression(k): self.expression(v) for k, v in zip(node.keys, node.values)}
        elif isinstance(node, ast.Subscript):
            container = self.expression(node.value)
            index = node.slice.value if isinstance(node.slice, ast.Index) else node.slice
            if isinstance(index, ast.Slice):
                bounds = [self.expression(v) if v is not None else None for v in (index.lower,index.upper,index.step)]
                if any(v is not None and type(v) is not int for v in bounds): raise ValueError('Slice bounds must be integers')
                if type(container) not in (list,str): raise ValueError('Only sequences can be sliced')
                value = container[slice(*bounds)]
            else:
                key = self.expression(index)
                if type(container) is dict and type(key) is str: value = container[key]
                elif type(container) in (list,str) and type(key) is int: value = container[key]
                else: raise ValueError('Invalid JSON subscription')
        elif isinstance(node, ast.BinOp): value = self.binary(node.op, self.expression(node.left), self.expression(node.right))
        elif isinstance(node, ast.UnaryOp):
            item = self.expression(node.operand)
            if isinstance(node.op, ast.Not): value = not item
            else:
                self.numeric(item)
                value = -item if isinstance(node.op, ast.USub) else +item
        elif isinstance(node, ast.BoolOp):
            for operand in node.values:
                value = self.expression(operand)
                if isinstance(node.op, ast.And) and not value or isinstance(node.op, ast.Or) and value: break
        elif isinstance(node, ast.Compare):
            left = self.expression(node.left); value = True
            compare = {ast.Eq:operator.eq, ast.NotEq:operator.ne, ast.Lt:operator.lt, ast.LtE:operator.le,
                       ast.Gt:operator.gt, ast.GtE:operator.ge, ast.In:lambda a,b:a in b, ast.NotIn:lambda a,b:a not in b}
            for op, right_node in zip(node.ops,node.comparators):
                right = self.expression(right_node)
                if not compare[type(op)](left,right): value = False; break
                left = right
        elif isinstance(node, ast.IfExp): value = self.expression(node.body if self.expression(node.test) else node.orelse)
        elif isinstance(node, ast.Call): value = self.call(node.func.id, [self.expression(a) for a in node.args])
        else: raise ValueError('Unsupported expression')
        return self.check_value(value)

    @staticmethod
    def numeric(*values):
        if any(type(value) not in (int,float) for value in values): raise ValueError('Numeric operands required')

    def binary(self, op, left, right):
        if isinstance(op, ast.Add) and type(left) in (str,list) and type(right) is type(left):
            limit = 4096 if type(left) is str else 256
            if len(left)+len(right) > limit: raise ValueError('Concatenation too large')
            return left+right
        if isinstance(op, ast.Mult) and (type(left) in (str,list) or type(right) in (str,list)):
            sequence, count = (left,right) if type(left) in (str,list) else (right,left)
            limit = 4096 if type(sequence) is str else 256
            if type(count) is not int or count < 0 or count > limit or len(sequence)*count > limit:
                raise ValueError('Sequence repetition too large')
            return sequence*count
        self.numeric(left,right)
        functions = {ast.Add:operator.add, ast.Sub:operator.sub, ast.Mult:operator.mul,
                     ast.Div:operator.truediv, ast.FloorDiv:operator.floordiv, ast.Mod:operator.mod}
        return functions[type(op)](left,right)

    def call(self, name, args):
        if name == 'range':
            if not 1 <= len(args) <= 3 or any(type(a) is not int for a in args): raise ValueError('Invalid bounded range')
            values = range(*args)
            if len(values) > 256: raise ValueError('Range too large')
            return list(values)
        if name == 'len' and len(args) == 1: return len(args[0])
        if name == 'abs' and len(args) == 1: self.numeric(args[0]); return abs(args[0])
        if name == 'round' and 1 <= len(args) <= 2:
            self.numeric(args[0])
            if len(args) == 2 and (type(args[1]) is not int or not -12 <= args[1] <= 12): raise ValueError('Invalid rounding precision')
            return round(*args)
        if name in ('sum','sorted') and len(args) == 1 and type(args[0]) is list:
            if name == 'sum': self.numeric(*args[0]); return sum(args[0])
            return sorted(args[0])
        if name in ('min','max'):
            items = args[0] if len(args) == 1 else args
            if type(items) not in (list,str) or not items: raise ValueError('min/max requires nonempty bounded values')
            return min(items) if name == 'min' else max(items)
        raise ValueError('Invalid computational builtin arguments')
