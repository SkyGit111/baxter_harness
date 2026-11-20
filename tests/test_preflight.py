"""Offline preflight and deployment boundary tests; no account or robot access."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import uuid

from cix_harness.preflight import check_config, check_online, report, verify_manifest, save_workspace_view
from cix_harness.contract import image_data

SPEC = importlib.util.spec_from_file_location('build_deploy_bundle', Path(__file__).resolve().parents[1] / 'scripts' / 'build_deploy_bundle.py')
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / '.local' / ('test-preflight-' + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        def cleanup():
            if self.root.resolve().parent.name != '.local':raise RuntimeError('Cleanup escaped workspace')
            shutil.rmtree(self.root)
        self.addCleanup(cleanup)

    def test_offline_only_checks_credential_presence_never_reads_tokens_or_network(self):
        credentials = self.root / 'chatgpt-session.json'; credentials.write_text('private-invalid-content')
        with patch('cix_harness.preflight.Path.read_text', side_effect=AssertionError('No credential read')), \
                patch('cix_harness.chatgpt_auth.list_models') as account, \
                patch('cix_harness.rpc.HttpRpcClient') as rpc:
            result = report(credentials=credentials)
        self.assertTrue(result['credentials']['present']);self.assertFalse(result['credentials']['contents_read'])
        self.assertFalse(result['ready_for_motion'])
        self.assertNotIn('private-invalid-content', json.dumps(result))
        account.assert_not_called();rpc.assert_not_called()
        with self.assertRaisesRegex(ValueError, '--online'):report(check_account=True)
        with self.assertRaisesRegex(ValueError, '--online'):report(view_dir=self.root/'views')

    def test_candidate_config_remains_uncommissioned_and_pending(self):
        original = Path('config/baxter-primitives.example.json')
        before = original.read_bytes()
        result = check_config(original)
        self.assertEqual(result['json_syntax'], 'ok');self.assertEqual(result['schema'], 'ok')
        self.assertEqual(result['status'], 'pending')
        self.assertFalse(result['arms']['left']['commissioned'])
        self.assertEqual(before, original.read_bytes())
        bad = self.root / 'bad.json';bad.write_text('{invalid json and private value')
        invalid = check_config(bad)
        self.assertEqual(invalid['status'], 'error');self.assertNotIn('private value', json.dumps(invalid))

    def test_online_rpc_whitelist_and_no_image_content_in_report(self):
        state = {'eef_m':[.4,.2,.1], 'eef_xyzw':[0,0,0,1], 'joint_rad':[0]*7,
                 'joint_names':['j'+str(i) for i in range(7)], 'gripping':False}
        observation = {'image':'data:image/jpeg;base64,PRIVATE_IMAGE',
            'arm_states':{'left':dict(state,arm='left'),'right':dict(state,arm='right')},
            'workspaces':{'left':{'frame':'base'},'right':{'frame':'base'}}}
        robot = Mock();robot.call.side_effect = [
            {'motion_scope':'supervised_dual_primitive_task','dual_protocol':'serialized_dual_primitives_v1','armed':False},
            dict(observation,selected_arm='left'),dict(observation,selected_arm='right')]
        result = check_online(robot)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual([call.args for call in robot.call.call_args_list], [('healthz',),('dual.observe',('left',)),('dual.observe',('right',))])
        self.assertFalse(result['motor_commands_sent']);self.assertEqual(result['ik_preflight'],'pending_task_preview')
        self.assertNotIn('PRIVATE_IMAGE', json.dumps(result))

    def test_wrong_bridge_scope_stops_before_any_observation_or_motor_call(self):
        robot = Mock();robot.call.return_value={'motion_scope':'supervised_primitive_task'}
        result = check_online(robot)
        self.assertEqual(result['status'],'error');robot.call.assert_called_once()

    def test_private_reference_view_preserves_original_image_and_escapes_notes(self):
        image=image_data(Path('tests/fixtures/white-pixel.png').read_bytes())
        reference={'polygon_uv':[[.1,.2],[.8,.2],[.8,.7],[.1,.7]],'status':'in_image'}
        path=self.root/'view.html'
        save_workspace_view({'image':image,'workspaces':{'left':{
            'table_reference':reference,'calibration_note':'<script>unsafe()</script>'}}},path)
        output=path.read_text(encoding='utf-8')
        self.assertIn(image,output)
        self.assertIn('points="0.1,0.2 0.8,0.2 0.8,0.7 0.1,0.7"',output)
        self.assertNotIn('<script>',output)
        self.assertIn('&lt;script&gt;',output)

    def test_manifest_verification_detects_change_and_path_escape(self):
        file = self.root / 'source.py';file.write_bytes(b'print(1)\n')
        manifest = {'format':'baxter-harness-bundle-v1','kind':'source','files':{
            'source.py':{'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'bytes':file.stat().st_size}}}
        path = self.root / 'bundle.manifest.json';path.write_text(json.dumps(manifest))
        self.assertEqual(verify_manifest(self.root,path)['checked_files'],1)
        file.write_text('changed')
        with self.assertRaisesRegex(ValueError,'checksum'):verify_manifest(self.root,path)
        manifest['files']={'../outside':{'sha256':'a'*64,'bytes':1}};path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'relative'):verify_manifest(self.root,path)

    def test_source_selection_excludes_secrets_images_weights_and_site_data(self):
        for bad in ('.local/chatgpt-session.json','runs/run/events.jsonl','models/model.safetensors',
                    'config/baxter.local.json','config/site.json','deploy/wheels/file.py',
                    'docs/scene.jpg','../outside.py','.env','deploy/auth.json','scripts/unrelated.cpp'):
            self.assertFalse(bundle.allowed_source(bad),bad)
        for good in ('cix_harness/chatgpt_auth.py','config/baxter-hybrid.example.json','scripts/build_deploy_bundle.py','scripts/offline_scene_check.cpp',
                     'third_party/RPent-LICENSE','README.md','deploy/Dockerfile.vlm','deploy/Dockerfile.vla',
                     'tests/fixtures/white-pixel.png'):
            self.assertTrue(bundle.allowed_source(good),good)

    def test_only_git_tracked_allowed_source_is_read(self):
        (self.root/'cix_harness').mkdir();(self.root/'cix_harness'/'one.py').write_text('private unstaged content\n')
        (self.root/'cix_harness'/'untracked.py').write_text('x=2\n')
        rows = b'100644 aaaaa 0\tcix_harness/one.py\0' + b'100644 bbbbb 0\t.local/chatgpt-session.json\0'
        with patch.object(bundle.subprocess,'run',side_effect=[NS(stdout=rows),NS(stdout=b'indexed=1\n')]) as git:
            selected=bundle.tracked_sources(self.root)
        self.assertEqual(set(selected),{'cix_harness/one.py'})
        self.assertEqual(selected['cix_harness/one.py'],b'indexed=1\n')
        self.assertEqual(git.call_args_list[1].args[0][-3:],['cat-file','blob','aaaaa'])

    def test_site_bundle_is_separate_and_rejects_credential_keys(self):
        path=self.root/'baxter.local.json';path.write_text(json.dumps({'primitives':{'calibrated':False}}))
        self.assertEqual(set(bundle.site_configs([path])),{'config/baxter.local.json'})
        path.write_text(json.dumps({'primitives':{},'refresh_token':'private'}))
        with self.assertRaisesRegex(ValueError,'Credential'):bundle.site_configs([path])
        (self.root/'deploy').mkdir();secret=self.root/'deploy'/'unsafe.json'
        secret.write_text(json.dumps({'access_token':'private'}))
        with self.assertRaisesRegex(ValueError,'Credential'):bundle.validate_source('deploy/unsafe.json',secret.read_bytes())

    def test_only_fixed_synthetic_fixture_is_allowed(self):
        raw=Path('tests/fixtures/white-pixel.png').read_bytes()
        self.assertEqual(bundle.validate_source(bundle.SYNTHETIC_FIXTURE,raw),raw)
        with self.assertRaisesRegex(ValueError,'synthetic'):bundle.validate_source(bundle.SYNTHETIC_FIXTURE,b'photo')

    def test_bundles_reproducible_and_self_verifiable(self):
        source={'cix_harness/example.py':b'print("offline")\n'}
        a=self.root/'a.zip';b=self.root/'b.zip'
        bundle.write_bundle(a,source,'source','revision')
        bundle.write_bundle(b,source,'source','revision')
        self.assertEqual(a.read_bytes(),b.read_bytes())
        self.assertEqual(bundle.verify_archive(a)['verified_files'],1)
        self.assertTrue(a.with_name('a.zip.sha256').is_file())


if __name__ == '__main__':unittest.main()
