"""Protocol and physical-command regressions. All endpoints are ephemeral loopback fixtures."""
import contextlib
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from cix_harness.__main__ import main
from cix_harness.contract import ACTION_SPEC, image_data
from cix_harness.rpc import RpcHandler

ROOT = Path(__file__).resolve().parents[1]


class VlmHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, value):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(value).encode())

    def do_GET(self):
        assert self.path == "/v1/models"
        self.reply({"data": [{"id": "cix-vlm"}]})

    def do_POST(self):
        assert self.path == "/v1/chat/completions"
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requested_model = request["model"]
        image = request["messages"][-1]["content"][1]["image_url"]["url"]
        assert image.startswith("data:image/png;base64,")
        decision = self.server.decisions.pop(0)
        self.reply({"choices": [{"message": {"content": json.dumps(decision)}, "finish_reason": "stop"}]})


def start(cls, cleanup):
    server = HTTPServer(("127.0.0.1", 0), cls)
    threading.Thread(target=lambda: server.serve_forever(poll_interval=.02), daemon=True).start()
    cleanup(server.server_close)
    cleanup(server.shutdown)
    return server


class HarnessTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.vlm = start(VlmHandler, self.addCleanup)
        self.vla = start(RpcHandler, self.addCleanup)
        self.robot = start(RpcHandler, self.addCleanup)
        self.frame = 0
        self.steps = []
        self.inputs = []
        self.act = {"decision": "act", "subtask": "pick up the red block", "evidence": "block on table"}
        self.done = {"decision": "done", "subtask": "", "evidence": "block in tray"}
        self.vla.methods = {"healthz": lambda: {"action_spec": ACTION_SPEC}, "vla.predict": self.predict}
        self.robot.methods = {"healthz": lambda: {"armed": True, "action_spec": ACTION_SPEC},
                              "env.observe": self.observe, "env.step": self.step}

    def observe(self):
        self.frame += 1
        return {"id": str(self.frame), "image": image_data((ROOT / "tests/fixtures/white-pixel.png").read_bytes())}

    def predict(self, obs, options):
        self.inputs.append((obs["id"], obs["task_descriptions"]))
        return {"obs_id": obs["id"], "actions": [[0.006, 0, 0, 0, 0, 0, 1]], "action_spec": ACTION_SPEC}

    def step(self, frame, spec, action):
        self.assertEqual(frame, str(self.frame))
        self.assertEqual(spec, ACTION_SPEC)
        self.steps.append(frame)
        return {"obs_id": frame, "applied": True}

    def invoke(self, *args):
        ports = ["--vlm-port", str(self.vlm.server_port), "--vla-port", str(self.vla.server_port),
                 "--robot-port", str(self.robot.server_port)]
        with contextlib.redirect_stdout(io.StringIO()):
            return main(ports + ["run", "--task", "把红方块放到蓝色托盘里", "--logs", self.dir.name] + list(args))

    def test_feedback_uses_new_frames_and_stops_on_done(self):
        self.vlm.decisions = [self.act, self.done]
        self.assertEqual(self.invoke("--execute", "--steps", "2"), 0)
        self.assertEqual(self.steps, ["2", "3"])
        self.assertEqual(self.inputs, [(s, self.act["subtask"]) for s in self.steps])
        self.assertEqual(self.frame, 4)
        events = next(Path(self.dir.name).rglob("events.jsonl")).read_text(encoding="utf-8")
        self.assertEqual(json.loads(events.splitlines()[-1])["status"], "done")

    def test_file_preview_never_contacts_robot(self):
        self.vlm.decisions = [self.act]
        self.assertEqual(self.invoke("--image", str(ROOT / "tests/fixtures/white-pixel.png")), 0)
        self.assertEqual(self.frame, 0)
        self.assertEqual(self.steps, [])
        self.assertEqual(len(self.inputs), 1)

    def test_plan_selects_model_without_vla_or_robot(self):
        self.vlm.decisions = [self.act]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = main(["--vlm-port", str(self.vlm.server_port), "--vlm-model", "local-9b",
                         "--robot-port", "0", "--vla-port", "0", "plan",
                         "--image", str(ROOT / "tests/fixtures/white-pixel.png"), "--task", "拿起红色物体"])
        self.assertEqual(code, 0)
        self.assertEqual(self.vlm.requested_model, "local-9b")
        self.assertEqual(json.loads(out.getvalue())["subtask"], self.act["subtask"])

    def test_malformed_policy_never_reaches_robot(self):
        self.vlm.decisions = [self.act]
        self.vla.methods["vla.predict"] = lambda *args: {"actions": [[1]*6], "obs_id": "2", "action_spec": ACTION_SPEC}
        self.assertEqual(self.invoke("--execute"), 1)
        self.assertEqual(self.steps, [])

    def test_live_plan_reads_camera_without_vla_or_motion(self):
        self.vlm.decisions = [self.act]
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(["--vlm-port", str(self.vlm.server_port), "--robot-port", str(self.robot.server_port),
                         "--vla-port", "0", "plan", "--robot-camera", "--task", "拿起红色方块",
                         "--logs", self.dir.name])
        self.assertEqual(code, 0)
        self.assertEqual(self.frame, 1)
        self.assertEqual(self.steps, [])
        self.assertEqual(self.inputs, [])
        self.assertEqual(len(list(Path(self.dir.name).rglob('000-planner.jpg'))), 1)

    def test_untranslated_subtask_never_reaches_english_policy(self):
        self.vlm.decisions = [{**self.act, "subtask": "抓取红色方块"}]
        self.assertEqual(self.invoke("--execute"), 1)
        self.assertEqual(self.inputs, [])

    def test_ack_loss_is_never_retried(self):
        self.vlm.decisions = [self.act]

        def lose_receipt(*args):
            self.step(*args)
            raise RuntimeError("connection lost after applying step")

        self.robot.methods["env.step"] = lose_receipt
        self.assertEqual(self.invoke("--execute"), 1)
        self.assertEqual(self.steps, ["2"])

    def test_final_visual_check_cannot_exceed_motion_budget(self):
        self.vlm.decisions = [self.act, self.act]
        self.assertEqual(self.invoke("--execute", "--steps", "1", "--max-turns", "1"), 2)
        self.assertEqual(self.steps, ["2"])
        self.assertEqual(self.frame, 3)

    def test_final_pose_failure_records_applied_action_and_stops(self):
        self.vlm.decisions = [self.act]
        def miss_target(*args):
            return {**self.step(*args), 'target_reached':False}
        self.robot.methods['env.step'] = miss_target
        self.assertEqual(self.invoke('--execute','--steps','2'),1)
        self.assertEqual(len(self.steps),1)
        events=[json.loads(line) for line in next(Path(self.dir.name).rglob('events.jsonl')).read_text(encoding='utf-8').splitlines()]
        self.assertEqual(events[-2]['event'],'applied')
        self.assertEqual(events[-1]['event'],'stop')

    def test_cartesian_diagnostic_bridge_cannot_execute_policy_loop(self):
        self.robot.methods['healthz'] = lambda:{'armed':True,'action_spec':ACTION_SPEC,
                                              'motion_scope':'supervised_cartesian_diagnostic'}
        self.assertEqual(self.invoke('--execute'),1)
        self.assertEqual(self.steps,[])

    def test_expiring_window_reobserves_without_extra_planner_round(self):
        self.vlm.decisions = [self.act,self.done]
        self.robot.methods['healthz']=lambda:{'armed':True,'action_spec':ACTION_SPEC,'max_action_window':2}
        self.vla.methods['healthz']=lambda:{'action_spec':ACTION_SPEC,'max_action_window':2}
        def predict_window(obs,options):
            result=self.predict(obs,options)
            result['actions']*=options['action_window']
            return result
        self.vla.methods['vla.predict']=predict_window
        def short_window(frame,spec,actions):
            return {**self.step(frame,spec,actions[0]),'applied_actions':1,
                    'reobserve_reason':'observation_deadline','target_reached':True}
        self.robot.methods['env.step_chunk']=short_window
        self.assertEqual(self.invoke('--execute','--steps','4','--action-window','2'),0)
        self.assertEqual(self.steps,['2','3','4','5'])
        self.assertEqual(self.frame,6)

    def test_wrist_planning_requires_real_wrist_before_any_motion(self):
        self.vlm.decisions=[self.act]
        self.assertEqual(self.invoke('--execute','--planner-view','wrist'),1)
        self.assertEqual(self.steps,[])
        self.assertEqual(self.inputs,[])

    def test_segment_endpoint_error_returns_to_vision_instead_of_aborting_task(self):
        self.vlm.decisions=[self.act,self.done]
        self.robot.methods['healthz']=lambda:{'armed':True,'action_spec':ACTION_SPEC,'max_action_segment':16}
        self.robot.methods['env.step_segment']=lambda frame,spec,actions:{
            **self.step(frame,spec,actions[0]),'applied_actions':1,'execution_mode':'policy_segment',
            'target_reached':False,'position_error_m':.0108}
        self.assertEqual(self.invoke('--execute','--execution-mode','segment','--steps','4'),0)
        self.assertEqual(len(self.steps),1)
        self.assertEqual(self.frame,3)

    def test_segment_budget_ends_burst_and_keeps_final_visual_check(self):
        self.vlm.decisions=[self.act,self.done]
        self.robot.methods['healthz']=lambda:{'armed':True,'action_spec':ACTION_SPEC,
            'max_action_segment':16,'motion_remaining':1-len(self.steps)}
        self.robot.methods['env.step_segment']=lambda frame,spec,actions:{
            **self.step(frame,spec,actions[0]),'applied_actions':1,'target_reached':True}
        self.assertEqual(self.invoke('--execute','--execution-mode','segment','--steps','4'),0)
        self.assertEqual(len(self.steps),1)
        self.assertEqual(self.frame,3)
