"""OAuth trust-boundary checks: synthetic credentials, no network or robot access."""
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch
from urllib.parse import parse_qs, urlencode, urlsplit

from cix_harness import chatgpt_auth as auth


def token_reply(**changes):
    value = {'access_token': 'synthetic-access', 'refresh_token': 'synthetic-refresh',
             'id_token': 'synthetic-id', 'token_type': 'Bearer',
             'expires_in': 3600, 'scope': auth.SCOPES}
    value.update(changes)
    return value


def callback(attempt, **changes):
    values = {'state': attempt['state'], 'code': 'synthetic-code', 'client_id': 'oaiapp_test'}
    values.update(changes)
    return '/auth/callback?' + urlencode(values)


class ChatGPTAuthTests(unittest.TestCase):
    def setUp(self):
        # Python 3.13's Windows mkdir(0700) excludes the restricted test token.
        # Use an inherited workspace DACL; every value written here is synthetic.
        root = Path(__file__).resolve().parents[1] / '.local'
        folder = root / ('test-chatgpt-auth-' + uuid.uuid4().hex)
        folder.mkdir(parents=True)
        def cleanup():
            if folder.resolve().parent != root.resolve():
                raise RuntimeError('Test cleanup escaped workspace')
            shutil.rmtree(folder)
        self.addCleanup(cleanup)
        self.path = folder / 'credentials.json'
        # Tests never write real secrets; avoid platform-specific ACL subprocesses.
        private = patch.object(auth, '_private_file')
        private.start()
        self.addCleanup(private.stop)

    def attempt(self, client_id=None):
        return auth.new_attempt('urn:uuid:synthetic-host', 1455, client_id)

    def seed(self, expires_at=None):
        record = auth._token_record(token_reply(), 'oaiapp_test', {'sub': 'verified-subject', 'email': 'example@example.test'})
        if expires_at is not None:
            record['expires_at'] = expires_at
        store = {'ext_agent_host_id': 'urn:uuid:existing-host', 'active_client_id': 'oaiapp_test',
                 'accounts': {'oaiapp_test': record}}
        auth._save(self.path, store)
        return store

    def test_dynamic_registration_and_pkce(self):
        attempt = self.attempt()
        params = parse_qs(urlsplit(auth.authorization_url(attempt)).query)
        self.assertEqual(params['client_id'], ['dynamic_agent_client'])
        self.assertEqual(params['agent_name_hint'], ['baxter_harness'])
        self.assertEqual(params['resource'], [auth.RESOURCE])
        self.assertEqual(params['redirect_uri'], ['http://127.0.0.1:1455/auth/callback'])
        self.assertEqual(params['code_challenge_method'], ['S256'])
        self.assertNotIn('code_verifier', params)
        self.assertNotEqual(attempt['code_verifier'], attempt['code_challenge'])
        self.assertNotEqual(attempt['state'], self.attempt()['state'])

    def test_returning_registration_omits_app_name_and_tokens_from_url(self):
        params = parse_qs(urlsplit(auth.authorization_url(self.attempt('oaiapp_saved'))).query)
        self.assertEqual(params['client_id'], ['oaiapp_saved'])
        self.assertNotIn('agent_name_hint', params)
        self.assertNotIn('id_token_hint', params)

    def test_rejects_unverified_expired_and_replayed_callbacks(self):
        attempt = self.attempt()
        with self.assertRaisesRegex(ValueError, 'state'):
            auth.consume_callback(attempt, callback(attempt, state='wrong'))
        self.assertFalse(attempt['consumed'])
        self.assertEqual(auth.consume_callback(attempt, callback(attempt)), ('synthetic-code', 'oaiapp_test'))
        with self.assertRaisesRegex(ValueError, 'consumed'):
            auth.consume_callback(attempt, callback(attempt))
        expired = self.attempt()
        expired['expires_at'] = time.monotonic() - 1
        with self.assertRaisesRegex(ValueError, 'expired'):
            auth.consume_callback(expired, callback(expired))

    def test_decline_missing_client_and_account_swap_never_exchange(self):
        for client_id, change in [(None, {'error': 'access_denied'}),
                                  (None, {'client_id': auth.DYNAMIC_CLIENT}),
                                  ('oaiapp_original', {'client_id': 'oaiapp_other'})]:
            attempt = self.attempt(client_id)
            with patch.object(auth, '_json_request') as request:
                with self.assertRaises(ValueError):
                    auth.complete_login(self.path, attempt, callback(attempt, **change))
                request.assert_not_called()

    def test_duplicate_state_and_wrong_path_rejected(self):
        attempt = self.attempt()
        for raw in [callback(attempt) + '&state=' + attempt['state'], callback(attempt).replace('/auth/callback', '/other')]:
            with self.assertRaises(ValueError):
                auth.consume_callback(attempt, raw)

    def test_login_uses_issued_client_and_validates_before_storage(self):
        attempt = self.attempt()
        with patch.object(auth, '_json_request', return_value=token_reply()) as request, \
                patch.object(auth, 'validate_id_token', return_value={'sub': 'verified-subject'}) as validate:
            result = auth.complete_login(self.path, attempt, callback(attempt))
        self.assertEqual(result['client_id'], 'oaiapp_test')
        form = request.call_args.args[1]
        self.assertEqual(form['client_id'], 'oaiapp_test')
        self.assertEqual(form['code_verifier'], attempt['code_verifier'])
        self.assertEqual(form['redirect_uri'], attempt['redirect_uri'])
        validate.assert_called_once_with('synthetic-id', 'oaiapp_test', attempt['nonce'])
        self.assertEqual(auth.get_access_token(self.path), 'synthetic-access')

    def test_bad_signature_or_missing_plan_scope_leaves_existing_account_untouched(self):
        self.seed()
        before = self.path.read_text()
        for validation_error, scope in [(RuntimeError('signature invalid'), auth.SCOPES), (None, 'openid email')]:
            attempt = self.attempt()
            with patch.object(auth, '_json_request', return_value=token_reply(scope=scope)), \
                    patch.object(auth, 'validate_id_token', side_effect=validation_error,
                                 return_value={'sub': 'verified-subject'}):
                with self.assertRaises(RuntimeError):
                    auth.complete_login(self.path, attempt, callback(attempt))
            self.assertEqual(self.path.read_text(), before)

    def test_returning_account_subject_cannot_change(self):
        self.seed()
        before = self.path.read_text()
        attempt = self.attempt('oaiapp_test')
        with patch.object(auth, '_json_request', return_value=token_reply()), \
                patch.object(auth, 'validate_id_token', return_value={'sub': 'different-subject'}):
            with self.assertRaisesRegex(RuntimeError, 'identity'):
                auth.complete_login(self.path, attempt, callback(attempt))
        self.assertEqual(before, self.path.read_text())

    def test_refresh_rotates_both_tokens_and_preserves_host(self):
        self.seed(expires_at=1)
        with patch.object(auth, '_json_request', return_value=token_reply(
                access_token='new-access', refresh_token='new-refresh', id_token=None)) as request:
            self.assertEqual(auth.get_access_token(self.path), 'new-access')
        form = request.call_args.args[1]
        self.assertEqual(form['grant_type'], 'refresh_token')
        self.assertEqual(form['client_id'], 'oaiapp_test')
        self.assertNotIn('scope', form)
        store = auth._load(self.path)
        self.assertEqual(store['ext_agent_host_id'], 'urn:uuid:existing-host')
        self.assertEqual(store['accounts']['oaiapp_test']['refresh_token'], 'new-refresh')
        self.assertFalse(Path(str(self.path) + '.lock').exists())

    def test_refresh_failure_does_not_destroy_session_or_leave_lock(self):
        self.seed(expires_at=1)
        before = self.path.read_text()
        with patch.object(auth, '_json_request', side_effect=RuntimeError('network failure')):
            with self.assertRaises(RuntimeError):
                auth.get_access_token(self.path)
        self.assertEqual(before, self.path.read_text())
        self.assertFalse(Path(str(self.path) + '.lock').exists())

    def test_model_catalog_filters_visibility_preserves_server_order(self):
        self.seed()
        with patch.object(auth, '_json_request', return_value={'models': [
            {'slug': 'hidden', 'visibility': 'hidden'},
            {'slug': 'astra-slug', 'display_name': 'Astra', 'visibility': 'list'},
            {'slug': 'another', 'visibility': 'list'}]}) as request:
            self.assertEqual(auth.list_models(self.path), [
                {'slug': 'astra-slug', 'display_name': 'Astra'},
                {'slug': 'another', 'display_name': 'another'}])
        self.assertEqual(request.call_args.args[0], auth.RESOURCE + '/models')

    def test_status_never_exposes_credentials(self):
        self.seed()
        result = json.dumps(auth.status(self.path))
        for secret in ['synthetic-access', 'synthetic-refresh', 'synthetic-id']:
            self.assertNotIn(secret, result)

    def test_import_preserves_destination_host_id(self):
        self.seed()
        target = self.path.parent / 'board.json'
        auth._save(target, {'ext_agent_host_id': 'urn:uuid:board-host', 'accounts': {}})
        auth.import_credentials(self.path, target)
        self.assertEqual(auth._load(target)['ext_agent_host_id'], 'urn:uuid:board-host')
        self.assertEqual(auth.get_access_token(target), 'synthetic-access')

    def test_logout_clears_tokens_when_remote_unreachable_and_keeps_registration(self):
        self.seed()
        with patch.object(auth, '_json_request', side_effect=RuntimeError('offline')):
            result = auth.logout(self.path)
        self.assertFalse(result['remote_revocation_confirmed'])
        record = auth._load(self.path)['accounts']['oaiapp_test']
        self.assertEqual(record['subject'], 'verified-subject')
        self.assertNotIn('refresh_token', record)
        self.assertNotIn('access_token', record)
        self.assertNotIn('id_token', record)

    def test_real_signed_id_token_rejects_invalid_signature_identity_and_nonce(self):
        try:
            import jwt
            from cryptography.hazmat.primitives.asymmetric import rsa
        except ImportError:
            self.skipTest('PyJWT[crypto] optional login dependency is not installed')
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        base = {'sub': 'verified-subject', 'iss': auth.ISSUER, 'aud': 'oaiapp_test',
                'iat': int(time.time()), 'exp': int(time.time()) + 300, 'nonce': 'expected-nonce'}
        with patch.object(jwt.PyJWKClient, 'get_signing_key_from_jwt',
                          return_value=SimpleNamespace(key=key.public_key())):
            valid = jwt.encode(base, key, algorithm='RS256')
            self.assertEqual(auth.validate_id_token(valid, 'oaiapp_test', 'expected-nonce')['sub'], 'verified-subject')
            for change, signing_key in [({'nonce': 'wrong'}, key), ({'aud': 'other-client'}, key),
                                        ({'iss': 'https://untrusted.example'}, key),
                                        ({'exp': int(time.time()) - 100}, key), ({}, other),
                                        ({'aud': ['oaiapp_test', 'other-client'], 'azp': 'other-client'}, key)]:
                claims = dict(base, **change)
                invalid = jwt.encode(claims, signing_key, algorithm='RS256')
                with self.assertRaisesRegex(RuntimeError, 'validation'):
                    auth.validate_id_token(invalid, 'oaiapp_test', 'expected-nonce')


if __name__ == '__main__':
    unittest.main()
