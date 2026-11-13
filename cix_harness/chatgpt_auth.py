"""Sign in with ChatGPT for this OSS app, independent of Codex credentials.

Official protocol: https://developers.openai.com/siwc/token-sharing-open-source
Install PyJWT[crypto] for ID-token verification. No robot access or inference here.
"""
import argparse
import base64
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import secrets
import subprocess
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import Request, build_opener
import uuid
import webbrowser

from .rpc import NoRedirect

ISSUER = 'https://auth.openai.com'
AUTHORIZE = ISSUER + '/api/accounts/authorize'
TOKEN = ISSUER + '/api/accounts/oauth/token'
JWKS = ISSUER + '/.well-known/jwks.json'
RESOURCE = 'https://api.openai.com/v1'
DIRECT_SCOPE = 'chatgpt.tokens.use.direct'
SCOPES = 'openid profile email offline_access resource.invoke ' + DIRECT_SCOPE
DYNAMIC_CLIENT = 'dynamic_agent_client'
MAX_AUTH_BYTES = 1024 * 1024


def default_credentials_path():
    root = Path(os.environ.get('LOCALAPPDATA', Path.home() / '.config')) if os.name == 'nt' else Path.home() / '.config'
    return root / 'baxter_harness' / 'chatgpt.json'


def _private_file(path):
    if os.name != 'nt':
        os.chmod(str(path), 0o600)
        return
    # Apply an owner-only DACL before any secret is written to the temporary file.
    who = subprocess.run(['whoami'], check=True, capture_output=True, text=True).stdout.strip()
    subprocess.run(['icacls', str(path), '/inheritance:r', '/grant:r', who + ':(F)'],
                   check=True, capture_output=True)


def _save(path, value):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temp = tempfile.mkstemp(prefix='.chatgpt-', dir=str(path.parent))
    try:
        os.close(fd)
        _private_file(temp)
        with open(temp, 'w', encoding='utf-8') as output:
            json.dump(value, output, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _load(path):
    path = Path(path).expanduser()
    if not path.exists():
        return {'ext_agent_host_id': 'urn:uuid:' + str(uuid.uuid4()), 'accounts': {}}
    with path.open(encoding='utf-8') as source:
        value = json.load(source)
    if not isinstance(value.get('accounts'), dict) or not value.get('ext_agent_host_id'):
        raise ValueError('Invalid baxter_harness ChatGPT credential store')
    return value


@contextmanager
def _lock(path, timeout=10):
    lock = Path(str(Path(path).expanduser()) + '.lock')
    lock.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RuntimeError('ChatGPT credential store is locked; another process may be refreshing') from None
            time.sleep(.1)
    try:
        yield
    finally:
        lock.unlink()


def _json_request(url, form=None, bearer=None):
    headers = {'Accept': 'application/json'}
    data = None
    if form is not None:
        data = urlencode(form).encode('ascii')
        headers['Content-Type'] = 'application/x-www-form-urlencoded'
    if bearer:
        headers['Authorization'] = 'Bearer ' + bearer
    request = Request(url, data=data, headers=headers)
    try:
        with build_opener(NoRedirect()).open(request, timeout=30) as response:
            raw = response.read(MAX_AUTH_BYTES + 1)
    except HTTPError as exc:
        # Never expose arbitrary provider response bodies, codes, or credentials.
        raise RuntimeError('ChatGPT authorization service HTTP ' + str(exc.code)) from None
    except (URLError, TimeoutError):
        raise RuntimeError('ChatGPT authorization service connection failed') from None
    if len(raw) > MAX_AUTH_BYTES:
        raise ValueError('ChatGPT authorization response exceeds limit')
    return json.loads(raw) if raw else {}


def new_attempt(host_id, port, client_id=None):
    if not 0 < port < 65536:
        raise ValueError('Invalid callback port')
    verifier = secrets.token_urlsafe(64)
    return {'state': secrets.token_urlsafe(32), 'nonce': secrets.token_urlsafe(32),
            'code_verifier': verifier,
            'code_challenge': base64.urlsafe_b64encode(hashlib.sha256(verifier.encode('ascii')).digest()).rstrip(b'=').decode('ascii'),
            'redirect_uri': 'http://127.0.0.1:{}/auth/callback'.format(port),
            'client_id': client_id or DYNAMIC_CLIENT, 'ext_agent_host_id': host_id,
            'expires_at': time.monotonic() + 600, 'consumed': False}


def authorization_url(attempt):
    params = {key: attempt[key] for key in ('client_id', 'ext_agent_host_id', 'redirect_uri', 'state', 'nonce', 'code_challenge')}
    params.update(response_type='code', scope=SCOPES, resource=RESOURCE, code_challenge_method='S256')
    if attempt['client_id'] == DYNAMIC_CLIENT:
        params['agent_name_hint'] = 'baxter_harness'
    # Deliberately omit id_token_hint so the printed URL never includes a token.
    return AUTHORIZE + '?' + urlencode(params)


def consume_callback(attempt, callback):
    parsed = urlsplit(callback)
    if parsed.path != '/auth/callback':
        raise ValueError('Unexpected ChatGPT callback path')
    params = parse_qs(parsed.query, keep_blank_values=True)
    if any(len(values) != 1 for values in params.values()):
        raise ValueError('Duplicate ChatGPT callback parameters')
    state = params.get('state', [''])[0]
    if not secrets.compare_digest(state, attempt['state']):
        raise ValueError('ChatGPT callback state did not match')
    if attempt['consumed'] or time.monotonic() >= attempt['expires_at']:
        raise ValueError('ChatGPT authorization attempt expired or already consumed')
    attempt['consumed'] = True
    if 'error' in params:
        raise ValueError('ChatGPT authorization was declined or failed; no token exchange')
    client_id = params.get('client_id', [attempt['client_id']])[0]
    if client_id == DYNAMIC_CLIENT or not client_id:
        raise ValueError('ChatGPT registration did not return an issued client ID')
    if attempt['client_id'] != DYNAMIC_CLIENT and client_id != attempt['client_id']:
        raise ValueError('ChatGPT callback changed the selected client registration')
    code = params.get('code', [''])[0]
    if not code:
        raise ValueError('ChatGPT callback omitted the authorization code')
    return code, client_id


def validate_id_token(id_token, client_id, nonce=None):
    try:
        import jwt
    except ImportError:
        raise RuntimeError('Install PyJWT[crypto] to use Sign in with ChatGPT') from None
    try:
        key = jwt.PyJWKClient(JWKS).get_signing_key_from_jwt(id_token)
        claims = jwt.decode(id_token, key.key, algorithms=['RS256'], audience=client_id,
                            issuer=ISSUER, leeway=5,
                            options={'require': ['sub', 'exp', 'iat']})
        if not isinstance(claims['sub'], str) or not claims['sub']:
            raise ValueError('Missing subject')
        if nonce is not None and not secrets.compare_digest(str(claims.get('nonce', '')), nonce):
            raise ValueError('Nonce mismatch')
        if isinstance(claims.get('aud'), list) and len(claims['aud']) > 1 and claims.get('azp') != client_id:
            raise ValueError('Authorized party mismatch')
        return claims
    except Exception:
        raise RuntimeError('ChatGPT ID token failed signature or identity validation') from None


def _token_record(tokens, client_id, claims, previous=None):
    previous = previous or {}
    scopes = tokens.get('scope')
    scopes = scopes.split() if isinstance(scopes, str) else previous.get('scopes', [])
    if DIRECT_SCOPE not in scopes:
        raise RuntimeError('ChatGPT plan use was not granted; authorize this app in the browser')
    if str(tokens.get('token_type', '')).lower() != 'bearer' or not tokens.get('access_token'):
        raise RuntimeError('ChatGPT token response omitted a bearer credential')
    if not tokens.get('refresh_token'):
        raise RuntimeError('ChatGPT token response omitted the rotating refresh token')
    try:
        expires_in = float(tokens['expires_in'])
        if not 0 < expires_in <= 86400:
            raise ValueError('Invalid expiry')
    except (ValueError, KeyError, TypeError):
        raise RuntimeError('ChatGPT token response has invalid expiry') from None
    return {'client_id': client_id, 'issuer': ISSUER, 'subject': claims['sub'],
            'email': claims.get('email', previous.get('email', '')),
            'id_token': tokens.get('id_token') or previous.get('id_token', ''),
            'access_token': tokens['access_token'], 'refresh_token': tokens['refresh_token'],
            'token_type': 'Bearer', 'scopes': scopes, 'expires_at': time.time() + expires_in}


def complete_login(path, attempt, callback):
    code, client_id = consume_callback(attempt, callback)
    tokens = _json_request(TOKEN, {'grant_type': 'authorization_code', 'client_id': client_id,
                                 'code': code, 'code_verifier': attempt['code_verifier'],
                                 'redirect_uri': attempt['redirect_uri'], 'resource': RESOURCE})
    claims = validate_id_token(tokens.get('id_token', ''), client_id, attempt['nonce'])
    record = _token_record(tokens, client_id, claims)
    with _lock(path):
        store = _load(path)
        previous = store['accounts'].get(client_id)
        if previous and previous['subject'] != record['subject']:
            raise RuntimeError('ChatGPT sign-in changed the selected account identity')
        store['accounts'][client_id] = record
        store['active_client_id'] = client_id
        _save(path, store)
    return {key: record[key] for key in ('client_id', 'email', 'subject')}


def get_access_token(credential_path=None):
    """Return this app's selected bearer token, refreshing under a process lock."""
    path = credential_path or default_credentials_path()
    with _lock(path):
        store = _load(path)
        client_id = store.get('active_client_id')
        record = store['accounts'].get(client_id, {})
        if DIRECT_SCOPE not in record.get('scopes', []) or not record.get('access_token'):
            raise RuntimeError('Run python -m cix_harness.chatgpt_auth login first')
        if record.get('expires_at', 0) > time.time() + 60:
            return record['access_token']
        tokens = _json_request(TOKEN, {'grant_type': 'refresh_token', 'client_id': client_id,
                                      'refresh_token': record['refresh_token'], 'resource': RESOURCE})
        claims = {'sub': record['subject'], 'email': record.get('email', '')}
        if tokens.get('id_token'):
            claims = validate_id_token(tokens['id_token'], client_id)
            if claims['sub'] != record['subject']:
                raise RuntimeError('ChatGPT refresh changed account identity')
        record = _token_record(tokens, client_id, claims, record)
        store['accounts'][client_id] = record
        _save(path, store)
        return record['access_token']


def login(path, new_account=False, client_id=None, open_browser=True, timeout=600):
    try:
        import jwt
        if not jwt.algorithms.has_crypto:
            raise ImportError('Cryptographic JWT support missing')
    except ImportError:
        raise RuntimeError('Install PyJWT[crypto] to use Sign in with ChatGPT') from None
    with _lock(path):
        store = _load(path)
        _save(path, store)  # Host identity must persist before opening authorization.
    selected = None if new_account else client_id or store.get('active_client_id')
    if selected and selected not in store['accounts']:
        raise ValueError('Unknown saved ChatGPT client registration')
    result = {}

    class CallbackHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            if urlsplit(self.path).path != '/auth/callback':
                self.send_error(404)
                return
            try:
                # Validate callback here; complete_login performs the one-time consume.
                state = parse_qs(urlsplit(self.path).query).get('state', [''])[0]
                if not secrets.compare_digest(state, attempt['state']):
                    raise ValueError('State mismatch')
            except ValueError:
                self.send_error(400)
                return
            result['callback'] = self.path
            body = b'Authorization received. Return to the baxter_harness terminal for the verified result.'
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    class LoopbackServer(HTTPServer):
        def get_request(self):
            connection, address = super().get_request()
            connection.settimeout(3)
            return connection, address

    with LoopbackServer(('127.0.0.1', 0), CallbackHandler) as server:
        server.timeout = .5
        attempt = new_attempt(store['ext_agent_host_id'], server.server_port, selected)
        deadline = time.monotonic() + min(timeout, 600)
        url = authorization_url(attempt)
        print('Continue with ChatGPT: authorize baxter_harness and its ChatGPT plan usage in your browser.')
        print(url, flush=True)  # Contains PKCE challenge/state, never a token or verifier.
        if open_browser:
            webbrowser.open(url)
        while 'callback' not in result and time.monotonic() < deadline:
            server.handle_request()
        if 'callback' not in result:
            raise RuntimeError('ChatGPT authorization timed out; run login again')
    return complete_login(path, attempt, result['callback'])


def status(path):
    store = _load(path)
    return {'credentials': str(path), 'active_client_id': store.get('active_client_id'),
            'accounts': [{'client_id': key, 'email': account.get('email'),
                          'signed_in': bool(account.get('access_token')),
                          'plan_usage_authorized': DIRECT_SCOPE in account.get('scopes', [])}
                         for key, account in store['accounts'].items()]}


def list_models(path):
    reply = _json_request(RESOURCE + '/models', bearer=get_access_token(path))
    return [{'slug': item['slug'], 'display_name': item.get('display_name', item['slug'])}
            for item in reply.get('models', []) if item.get('visibility') == 'list']


def import_credentials(source, destination):
    """Import this app's SSH-transferred session while retaining this host's ID.

    Stop the source's token-refreshing processes before transferring a session;
    the destination becomes its sole refresh owner. This never reads Codex files.
    """
    incoming = _load(source)
    client_id = incoming.get('active_client_id')
    record = incoming['accounts'].get(client_id, {})
    if not record.get('access_token') or DIRECT_SCOPE not in record.get('scopes', []):
        raise ValueError('Source has no authorized baxter_harness ChatGPT session')
    with _lock(destination):
        store = _load(destination)
        store['accounts'][client_id] = record
        store['active_client_id'] = client_id
        _save(destination, store)
    return status(destination)


def logout(path):
    remote_revoked = False
    with _lock(path):
        store = _load(path)
        record = store['accounts'].get(store.get('active_client_id'), {})
        if record.get('refresh_token'):
            try:
                discovery = _json_request(ISSUER + '/.well-known/openid-configuration')
                endpoint = discovery.get('revocation_endpoint', '')
                parsed = urlsplit(endpoint)
                if parsed.scheme != 'https' or parsed.netloc != 'auth.openai.com' or parsed.query or parsed.fragment:
                    raise ValueError('Unexpected revocation endpoint')
                _json_request(endpoint, {'token': record['refresh_token'], 'token_type_hint': 'refresh_token',
                                         'client_id': record['client_id']})
                remote_revoked = True
            except (RuntimeError, ValueError):
                pass
        for key in ('access_token', 'refresh_token', 'id_token', 'scopes', 'expires_at'):
            record.pop(key, None)
        _save(path, store)
    return {'signed_out_locally': True, 'remote_revocation_confirmed': remote_revoked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['login', 'status', 'models', 'logout'])
    parser.add_argument('--credentials', type=Path, default=default_credentials_path())
    parser.add_argument('--new-account', action='store_true')
    parser.add_argument('--client-id', help='Saved registration to reauthorize')
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    try:
        if args.command == 'login':
            result = login(args.credentials, args.new_account, args.client_id, not args.no_browser, args.timeout)
        else:
            result = {'status': status, 'models': list_models, 'logout': logout}[args.command](args.credentials)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (RuntimeError, ValueError, OSError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
