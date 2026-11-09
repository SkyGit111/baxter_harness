"""Native RPent tools backed by the project's existing robot and policy services."""
from io import BytesIO

from PIL import Image
from rpent.session import EnvState
from rpent.tools import Toolkit, ToolResult, tool

from cix_harness.contract import ACTION_SPEC, image_bytes, image_data, instruction
from cix_harness.rollout import predict, validate_motion_model
from cix_harness.rpc import HttpRpcClient


class BaxterToolkit(Toolkit):
    def __init__(self, *, image, robot_port, vla_port, budget, folder, dashboard_events, memory):
        state = EnvState(folder)
        state.reset()
        super().__init__(state=state, dashboard_events=dashboard_events, memory=memory)
        self.image = image
        self.robot = None if image else HttpRpcClient(robot_port)
        self.policy = HttpRpcClient(vla_port)
        self.remaining = budget
        if budget:
            meta = self.robot.call("healthz", timeout_s=8)
            if meta.get("armed") is not True or meta.get("action_spec") != ACTION_SPEC:
                raise ValueError("Live steps require a commissioned and armed Baxter bridge")
            if meta.get('motion_scope') == 'supervised_cartesian_diagnostic':
                raise ValueError('Diagnostic bridge cannot execute policy tools')
            policy_meta = self.policy.call("healthz", timeout_s=8)
            if policy_meta.get("action_spec") != ACTION_SPEC:
                raise ValueError("Policy and robot action contracts differ")
            validate_motion_model(policy_meta, meta)
            self.add_tool(self.apply_one_step)
        self.add_tool(self.observe_table)
        self.add_tool(self.preview_action)

    def observation(self):
        if self.image:
            return {"id": "file", "image": image_data(self.image.read_bytes()), "state": {}}
        return self.robot.call("env.observe")

    def get_env_state(self, *, command, result, elapsed_s):
        try:
            obs = self.observation()
        except Exception:
            self.remaining = 0
            raise
        with Image.open(BytesIO(image_bytes(obs["image"]))) as frame:
            png = BytesIO()
            rgb = frame.convert("RGB")
            rgb.save(png, format="PNG")
        data = {"obs_id": obs["id"], "robot": obs["state"], "remaining": self.remaining}
        with self.state.record_step(state=data, command=command, result=result, elapsed_s=elapsed_s):
            if not self.state.save("table.png", rgb):
                self.remaining = 0
                raise RuntimeError("Could not record Baxter observation")
        return ToolResult(data={**data, "result": result}, images=[png.getvalue()])

    @tool(readonly=False)
    def observe_table(self) -> ToolResult:
        """Capture a fresh camera image and robot state without moving."""
        return ToolResult(data={"observed": True})

    @tool(readonly=False)
    def preview_action(self, subtask: str) -> ToolResult:
        """Predict one VLA step for a short English subtask, without moving."""
        obs = self.observation()
        action = predict(self.policy, obs, instruction(subtask))
        data = {"prediction_obs_id": obs["id"], "action": action, "applied": False}
        if self.robot and self.robot.call("healthz").get("can_preview_ik"):
            data["ik_preview"] = self.robot.call("env.preview", (obs["id"], ACTION_SPEC, action))
        elif self.robot and self.robot.call("healthz").get("can_inspect_ik"):
            data["ik_inspection"] = self.robot.call("env.inspect", (obs["id"], ACTION_SPEC, action))
        return ToolResult(data=data)

    @tool(readonly=False)
    def apply_one_step(self, subtask: str) -> ToolResult:
        """Apply one fresh policy prediction using one physical step from the operator's budget."""
        if self.remaining <= 0:
            raise RuntimeError("Motion budget exhausted")
        self.remaining -= 1
        try:
            obs = self.observation()
            action = predict(self.policy, obs, instruction(subtask))
            self.raise_if_cancelled()
            receipt = self.robot.call("env.step", (obs["id"], ACTION_SPEC, action), timeout_s=12)
            if receipt.get("applied") is not True or receipt.get("obs_id") != obs["id"]:
                raise RuntimeError("Invalid motion receipt")
            if receipt.get('target_reached') is False:
                raise RuntimeError('Action executed but measured arrival failed; do not replay')
        except BaseException:
            self.remaining = 0
            raise
        return ToolResult(data={"receipt": receipt, "action": action})
