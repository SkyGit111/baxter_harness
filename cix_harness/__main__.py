import argparse
import json
import sys
import time
import uuid
from pathlib import Path

from .contract import ACTION_SPEC, image_bytes, image_data, instruction
from .planner import decide
from .rollout import predict, rollout, validate_motion_model
from .rpc import HttpRpcClient, local_json
from .task_monitor import TaskMonitor
from .timing import TimingSummary, timed_rpc


class Journal:
    def __init__(self, root):
        self.folder = Path(root) / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
        self.folder.mkdir(parents=True)
        print("Run files:", self.folder, flush=True)
        self.start = time.monotonic()
        self.frames = 0
        self.timings = TimingSummary()

    def __call__(self, event, **fields):
        item = {"t": round(time.monotonic() - self.start, 3), "event": event, **fields}
        line = json.dumps(item, ensure_ascii=False, allow_nan=False)
        with (self.folder / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        self.timings.record(event,fields)
        if event in ('component_timing','planner_api','primitive_result','finish','stop','task_finished'):
            (self.folder/'timing-summary.json').write_text(
                json.dumps(self.timings.snapshot(item['t']),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(line, flush=True)

    def camera(self, obs, role):
        ext = ".png" if obs["image"].startswith("data:image/png;") else ".jpg"
        name = "{:03d}-{}{}".format(self.frames, role, ext)
        self.frames += 1
        (self.folder / name).write_bytes(image_bytes(obs["image"]))
        if "wrist_image" in obs:
            (self.folder / (name.rsplit('.', 1)[0] + '-wrist.jpg')).write_bytes(image_bytes(obs['wrist_image']))
        self("observation", file=name, obs_id=obs["id"], state=obs.get("state", {}))


def check_models(args, log=None):
    if getattr(args,'planner_provider','local') != 'api':
        models = local_json(args.vlm_port, "/v1/models", timeout=8)
        if args.vlm_model not in [model.get("id") for model in models["data"]]:
            raise RuntimeError("Local planner does not serve requested model: " + args.vlm_model)
    vla = HttpRpcClient(args.vla_port)
    if log:vla=timed_rpc(vla,log,'policy_rpc')
    meta = vla.call("healthz", timeout_s=8)
    if meta.get("action_spec") != ACTION_SPEC:
        raise RuntimeError("Local VLA action specification mismatch")
    return vla, meta


def run(args, log, planner=decide):
    goal = instruction(args.task)
    model, meta = check_models(args, log)
    monitor=TaskMonitor(getattr(args,'task_kind','generic'))
    robot = HttpRpcClient(args.robot_port) if args.image is None else None
    if robot:robot=timed_rpc(robot,log)
    if robot:
        status = robot.call("healthz", timeout_s=8)
        if args.execute and (status.get("armed") is not True or status.get("action_spec") != ACTION_SPEC):
            raise RuntimeError("Robot bridge is read-only or its action contract differs")
        if args.execute:
            if status.get('motion_scope') == 'supervised_cartesian_diagnostic':
                raise RuntimeError('This bridge is reserved for the supervised Cartesian diagnostic')
            validate_motion_model(meta, status)
            capacity=status.get('max_action_segment',0) if args.execution_mode=='segment' else status.get('max_action_window',1)
            if args.action_window > min(meta.get('max_action_window',1),capacity):
                raise RuntimeError('Requested action window is unsupported by the loaded services')
            if (meta.get('motion_scope') == 'local_clearance_single_step' or
                    status.get('motion_scope') == 'local_clearance_single_step'):
                if args.steps != 1 or args.max_turns != 1 or status.get('motion_remaining') != 1:
                    raise RuntimeError('Local clearance test requires one turn, one step and robot budget one')
            if meta.get('warmed') is False:
                warm_obs=robot.call('env.observe')
                predict(model,warm_obs,'observe the scene')
                log('warmup',discarded_prediction=True)
    log("start", goal=goal, mode="live" if args.execute else "preview", vla=meta)
    history = []
    # Include one final visual check after the last allowed motion burst.
    for turn in range(args.max_turns + 1):
        obs = (robot.call("env.observe") if robot else
               {"id": "file", "image": image_data(Path(args.image).read_bytes())})
        planner_obs=obs
        if args.planner_view == 'wrist':
            if 'wrist_image' not in obs:
                raise RuntimeError('Wrist planner view requires a fresh wrist image')
            planner_obs={**obs,'image':obs['wrist_image']}
            planner_obs.pop('wrist_image',None)
        log.camera(planner_obs, "planner" if args.planner_view == "scene" else "planner-wrist")
        monitor.observe(obs.get('state',{}))
        if args.execute:
            monitor.require_progress()
        if args.execute and monitor.stalled() and not getattr(args, 'managed_harness', False):
            log('finish',status='no_measured_progress',evidence=monitor.evidence())
            return 2
        if getattr(args, 'managed_harness', False):
            planner_obs = {**planner_obs, 'harness_feedback': {'stalled': monitor.stalled(),
                           'measured': monitor.evidence()}}
        decision = planner(args.vlm_port, goal, planner_obs, history, model=args.vlm_model)
        log("decision", **decision)
        if decision['decision']=='done' and args.execute and not monitor.can_complete():
            log('finish',status='completion_not_supported_by_grasp_feedback',evidence=monitor.evidence())
            return 2
        if decision["decision"] != "act":
            log("finish", status=decision["decision"], source="vlm", evidence=decision["evidence"])
            return 0 if decision["decision"] == "done" else 2
        if not args.execute:
            if robot:
                obs = robot.call("env.observe")
                log.camera(obs, "vla")
            action = predict(model, obs, decision["subtask"])
            log("preview", obs_id=obs["id"], subtask=decision["subtask"], action=action)
            if robot and status.get("can_preview_ik"):
                log("ik_preview", **robot.call("env.preview", (obs["id"], ACTION_SPEC, action)))
            elif robot and status.get("can_inspect_ik"):
                log("ik_inspection", **robot.call("env.inspect", (obs["id"], ACTION_SPEC, action)))
            return 0
        if turn == args.max_turns:
            log("finish", status="motion_budget_exhausted")
            return 2
        if robot.call('healthz').get('motion_remaining') == 0:
            log('finish',status='robot_motion_budget_exhausted')
            return 2
        used=rollout(robot, model, decision["subtask"], args.steps, log,args.action_window,monitor,args.execution_mode)
        history.append({**decision, "commands_applied": used,'measured_evidence':monitor.evidence()})
    raise AssertionError("unreachable")


def main(argv=None):
    parser = argparse.ArgumentParser(description="TY1100 local VLM + local VLA")
    parser.add_argument("--vlm-port", type=int, default=9100)
    parser.add_argument("--vlm-model", default="cix-vlm", help="Model ID advertised by the local VLM")
    parser.add_argument("--vla-port", type=int, default=9101)
    parser.add_argument("--robot-port", type=int, default=9102)
    cmds = parser.add_subparsers(dest="command", required=True)
    cmds.add_parser("check", help="Check both loaded models")
    cmds.add_parser("check-vla", help="Check only the action model service")
    plan = cmds.add_parser("plan", help="Preview a local VLM decision on a file or live robot camera")
    source = plan.add_mutually_exclusive_group(required=True)
    source.add_argument("--image")
    source.add_argument("--robot-camera", action="store_true", help="Read the robot camera and telemetry without commanding motion")
    plan.add_argument("--task", required=True)
    plan.add_argument("--logs", default="runs")
    probe = cmds.add_parser("predict", help="Run one local VLA forward pass on an image")
    probe_source = probe.add_mutually_exclusive_group(required=True)
    probe_source.add_argument("--image")
    probe_source.add_argument("--robot-camera", action="store_true")
    probe.add_argument("--subtask", required=True)
    probe.add_argument("--logs", default="runs")
    p = cmds.add_parser("run", help="Preview one decision or run a bounded task")
    p.add_argument("--task", required=True)
    p.add_argument("--image", help="Preview on a JPEG/PNG; omit to read the robot camera")
    p.add_argument("--execute", action="store_true", help="Apply policy outputs via an armed bridge")
    p.add_argument('--action-window',type=int,default=1,choices=range(1,17),help='Bounded policy horizon before a fresh image')
    p.add_argument('--execution-mode',choices=['waypoints','segment'],default='waypoints')
    p.add_argument('--task-kind',choices=['generic','pick','place','stack'],default='generic')
    p.add_argument('--planner-view',choices=['scene','wrist'],default='scene',help='Planning image; VLA still receives both views')
    p.add_argument("--steps", type=int, default=3, choices=range(1, 65), metavar="1..64")
    p.add_argument("--max-turns", type=int, default=8, choices=range(1, 33), metavar="1..32")
    p.add_argument("--logs", default="runs")
    args = parser.parse_args(argv)
    log = None
    try:
        if args.command == "check-vla":
            meta = HttpRpcClient(args.vla_port).call("healthz", timeout_s=8)
            if meta.get("action_spec") != ACTION_SPEC:
                raise RuntimeError("Local VLA action specification mismatch")
            print(json.dumps(meta))
            return 0
        if args.command == "check":
            print(json.dumps(check_models(args)[1]))
            return 0
        if args.command == "plan":
            if args.robot_camera:
                obs = HttpRpcClient(args.robot_port).call("env.observe")
                log = Journal(args.logs)
                log.camera(obs, "planner")
            else:
                obs = {"id": "file", "image": image_data(Path(args.image).read_bytes())}
            start = time.monotonic()
            decision = decide(args.vlm_port, instruction(args.task), obs, [], model=args.vlm_model)
            if log:
                log("decision", model=args.vlm_model, **decision)
            print(json.dumps({"model": args.vlm_model, "seconds": round(time.monotonic() - start, 3),
                              **decision}, ensure_ascii=False))
            return 0
        if args.command == "predict":
            if args.robot_camera:
                obs = HttpRpcClient(args.robot_port).call("env.observe")
                log = Journal(args.logs)
                log.camera(obs, "vla")
            else:
                obs = {"id": "file", "image": image_data(Path(args.image).read_bytes())}
            start = time.monotonic()
            action = predict(HttpRpcClient(args.vla_port), obs, instruction(args.subtask))
            result = {"action_spec": ACTION_SPEC, "action": action, "seconds": round(time.monotonic()-start,3)}
            if log:
                log("prediction", obs_id=obs["id"], subtask=args.subtask, **result)
            print(json.dumps(result, allow_nan=False))
            return 0
        if args.image and args.execute:
            parser.error("A file image cannot be used for live execution")
        log = Journal(args.logs)
        return run(args, log)
    except (Exception, KeyboardInterrupt) as exc:
        if log:
            log("stop", reason=str(exc) or type(exc).__name__)
        else:
            print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
