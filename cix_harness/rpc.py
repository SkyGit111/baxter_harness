# Copyright 2026 The RPent Authors. Licensed under Apache-2.0.
# Adapted from rpent/utils/rpc/http_rpc.py. See third_party/RPent-LICENSE.
# This edition uses loopback only, JSON-native fields and serialized dispatch.
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .upstream.rpc_client import RpcClient, check_response

MAX_BYTES = 12 * 1024 * 1024


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def local_json(port, path, body=None, timeout=120):
    url = "http://127.0.0.1:{}{}".format(port, path)
    payload = None if body is None else json.dumps(body, allow_nan=False).encode()
    req = Request(url, data=payload, headers={"Content-Type": "application/json"})
    opener = build_opener(ProxyHandler({}), NoRedirect())
    with opener.open(req, timeout=timeout) as response:
        raw = response.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("Response exceeds limit")
    return json.loads(raw)


class HttpRpcClient(RpcClient):
    def __init__(self, port):
        super().__init__()
        self.port = port

    def call(self, method, args=(), kwargs=None, *, timeout_s=120):
        response = local_json(self.port, "/call", {
            "method": method, "args": list(args), "kwargs": kwargs or {}, "session_id": None,
        }, timeout_s)
        return check_response(response, method)


class RpcHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        if self.path != "/call":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_BYTES:
                raise ValueError("Invalid request length")
            self.connection.settimeout(10)
            request = json.loads(self.rfile.read(size))
            method = self.server.methods[request["method"]]
            value = method(*request.get("args", []), **request.get("kwargs", {}))
            result = json.dumps({"ok": True, "result": value}, allow_nan=False)
        except Exception as exc:
            print("RPC error:", repr(exc), file=sys.stderr, flush=True)
            result = json.dumps({"ok": False, "error": str(exc)})
        payload = result.encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except ConnectionError:
            # The caller may already have timed out. Do not re-run the operation.
            pass


def serve(port, methods):
    # One inference or robot operation at a time. No concurrent writes to hardware.
    with HTTPServer(("127.0.0.1", port), RpcHandler) as server:
        server.methods = methods
        print("Ready on 127.0.0.1:{}".format(port), flush=True)
        server.serve_forever()
