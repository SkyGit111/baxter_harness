"""One visual decision at task start and after each bounded VLA rollout."""
import json
import re
from urllib.error import HTTPError

from .contract import instruction
from .rpc import local_json

SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {"type": "string"},
        "decision": {"type": "string", "enum": ["act", "done", "blocked"]},
        "subtask": {"type": "string", "pattern": "^[A-Za-z0-9 ,.'-]{0,160}$"},
    },
    "required": ["decision", "subtask", "evidence"],
    "additionalProperties": False,
}

SYSTEM = """You supervise one Baxter arm viewing a tabletop. Inspect the current camera image
and the measured gripper state. First describe the target's visible color, shape and
location in evidence, then choose decision and subtask based on those observations.
gripper_percent is 0 when closed and 100 when fully open;
gripping=false means no object is detected as held. An empty closed gripper can still
start a pick: the motor policy handles approaching and opening it. Do not infer that a
target is unreachable merely because the gripper is currently far away in the image.
Robot arms appearing in the view do not by themselves make the scene unusable.
The controller checks motion feasibility; do not invent reachability measurements.
The user may speak Chinese. Your only motor tool is a
vision-language-action policy. Choose act and give ONE short English instruction naming
the visible object and immediate physical goal. For a pick-and-place task with an empty
gripper, first say e.g. "pick up the blue cup". After observing it held, give the next
instruction. Write the motor subtask in English even for a Chinese user. Translate
the user's object description faithfully; preserve color and shape without guessing its use.
Continue an unfinished subtask across feedback turns if progress is visible or plausible.
If gripping is true, do not repeat a pick instruction: lift the held object before
transporting it. For a pick task, visible closure alone is not success; the object
must be held and raised above the table. For stacking, observe release and the final
object arrangement; motor acknowledgements alone are not evidence of completion.
If the goal is visibly satisfied choose done. If the target cannot be identified, the view
is unusable, or repeated attempts show no progress, choose blocked. For either terminal
choice use an empty subtask. Give brief visual evidence. An accepted action only means
a command ran. Judge its outcome from the current image.
Return one JSON object with keys evidence, decision (act/done/blocked), subtask. No markdown."""


def decide(port, task, obs, recent, model="cix-vlm"):
    state=dict(obs.get('state',{}))
    opening=state.get('gripper_percent')
    if isinstance(opening,(int,float)):
        state['gripper_opening_description']=('nearly closed' if opening<10 else
                                             'open' if opening>70 else 'partially open')
    context = {"goal": task, "recent": recent[-4:], "state": state}
    body = {
        "model": model, "temperature": 0, "max_tokens": 200,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": json.dumps(context, ensure_ascii=False)},
                {"type": "image_url", "image_url": {"url": obs["image"]}},
            ]},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "decision", "strict": True, "schema": SCHEMA}},
    }
    try:
        response = local_json(port, "/v1/chat/completions", body, timeout=180)
    except HTTPError as exc:
        raise RuntimeError("VLM HTTP {}: {}".format(exc.code, exc.read(1024).decode(errors="replace"))) from exc
    choice = response["choices"][0]
    if choice.get("finish_reason") == "length":
        raise ValueError("VLM decision was truncated")
    reply = json.loads(choice["message"]["content"])
    if not isinstance(reply, dict) or set(reply) != set(SCHEMA["required"]):
        raise ValueError("VLM returned an incomplete decision")
    if reply["decision"] not in ("act", "done", "blocked"):
        raise ValueError("Unknown planner decision")
    instruction(reply["evidence"])
    if reply["decision"] == "act":
        reply["subtask"] = instruction(reply["subtask"])
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9 ,.'-]{0,159}", reply["subtask"]):
            raise ValueError("The motor instruction must start with an English verb and use ASCII text")
    elif reply["subtask"] != "":
        raise ValueError("A terminal decision must have an empty subtask")
    return reply
