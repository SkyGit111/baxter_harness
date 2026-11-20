import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from cix_harness.__main__ import Journal
from cix_harness.executive import TaskExecutive
from cix_harness.contract import image_data
from scripts.compare_planner_views import compare, list_frames, load_run, read_events


class SavedCheckpoints(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        quiet = redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)
        self.log = Journal(folder.name)
        self.image = image_data((Path(__file__).parent/'fixtures/white-pixel.png').read_bytes())

    def frame(self, name, arm):
        return {'id': name, 'image': self.image, 'wrist_image': self.image,
                'state': {'arm': arm}, 'arm_states': {'left': {}, 'right': {}},
                'camera_metadata': {'wrist': {'age_s': .3, 'arm': arm}}}

    def test_same_frame_multiple_checkpoints_keep_holding_state_and_arm(self):
        planner = Mock()
        planner.check.return_value = {'decision': 'step_done'}
        memory = Mock()
        memory.start.return_value = 'test'
        memory.retrieve.return_value = []
        task = TaskExecutive('stack cubes', 'hybrid', planner, memory, None, self.log)
        self.log.camera(self.frame('a', 'left'), 'hybrid')
        right = self.frame('b', 'right')
        self.log.camera(right, 'hybrid')
        before = {'route':'hybrid', 'current':{'arm':'right', 'skill':'pick'}, 'verified':[]}
        after = {'route':'hybrid', 'current':None, 'verified':[before['current']]}
        task.call_planner('check', right, before)
        task.call_planner('check', right, after)
        for index, expected in enumerate((before, after)):
            obs, context = load_run(self.log.folder, 1, index)
            self.assertEqual(context, expected)
            result = compare(obs, context)
            self.assertEqual(result['selection']['context_source'], 'recorded_checkpoint')
            self.assertEqual(result['images']['scene_wrist']['views'][1]['source'], 'right_wrist')
            self.assertEqual(obs['arm_states'], right['arm_states'])
        self.assertEqual(list_frames(read_events(self.log.folder))[1]['saved_checkpoint_inputs'], 2)
        self.assertNotIn('base64', (self.log.folder/'events.jsonl').read_text(encoding='utf-8'))

    def test_old_later_frame_allows_image_manifest_and_never_fabricates_task_state(self):
        first = self.frame('a', 'left')
        self.log.camera(first, 'hybrid')
        self.log.camera(self.frame('b','right'), 'hybrid')
        self.log('task_model_reply', phase='check', obs_id='b', reply={'decision':'task_done'})
        obs, context = load_run(self.log.folder, 1)
        self.assertIsNone(context)
        self.assertEqual(compare(obs, context)['images']['scene_wrist']['image_count'], 2)
        with patch('scripts.compare_planner_views.ResponsesTaskPlanner') as api:
            with self.assertRaisesRegex(ValueError, 'no saved checkpoint'):
                compare(obs, context, call_api=True)
            api.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'Checkpoint index'):
            load_run(self.log.folder, 1, 2)
