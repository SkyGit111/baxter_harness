"""Exercise the upstream registry, tool dispatch and trace using the supplied still image."""
import argparse
import logging
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project))
import robots
robots.__path__.append(str(project / "integrations/rpent/robots"))
from rpent.dashboard.events import NullDashboardEventSink
from rpent.robots.base import enumerate_robots, get_robot_spec, get_toolkit
from rpent.utils.logging import init_output_dir
from cix_harness.contract import ACTION_SPEC
from cix_harness.contract import image_data

def main():
    assert "baxter" in enumerate_robots()
    spec = get_robot_spec("baxter")
    parser = argparse.ArgumentParser()
    spec.add_cli_args(parser, False)
    with TemporaryDirectory() as output:
        init_output_dir(output)
        args = parser.parse_args(["--baxter-task", "拿起红色长条", "--baxter-image",
                                  str(project / "examples/baxter-d455.jpg")])
        args.output_dir, args.memory_dir = output, str(Path(output) / "memory")
        config = spec.parse_config(args)
        spec.prepare_memory(args, config)
        events = NullDashboardEventSink()
        _, runtime = spec.init_runtime(args, config.output_dir, events)
        toolkit = get_toolkit("baxter", runtime_kwargs=runtime, dashboard_events=events, config=config)
        try:
            assert "拿起红色长条" in spec.prompts.render("user", variables=config.prompt_vars)
            assert "apply_one_step" not in {item.name for item in toolkit.list_tools()}
            view = toolkit.execute_tool("observe_table", {})
            assert not view.is_error and view.images[0].startswith(b"\x89PNG"), view.data
            toolkit.policy = Mock()
            toolkit.policy.call.return_value = {"action_spec": ACTION_SPEC, "obs_id":"file",
                                                "actions":[[0,0,0,0,0,0,1]]}
            preview = toolkit.execute_tool("preview_action", {"subtask":"pick up the red object"})
            assert not preview.is_error and preview.data["result"]["applied"] is False, preview.data
            assert toolkit.state.latest_step == 1
            assert all(toolkit.state.exists("table.png", step=i) for i in (0, 1))
            policy = toolkit.policy
            image = image_data(args.baxter_image.read_bytes())
            calls = []
            def robot_call(method, *args, **kwargs):
                calls.append(method)
                if method == "healthz":
                    return {"armed": True, "action_spec": ACTION_SPEC}
                if method == "env.observe":
                    return {"id": "file", "image": image, "state": {}}
                if method == "env.step":
                    raise ConnectionError("fixture: lost motion receipt")
                raise AssertionError(method)
            robot = Mock()
            robot.call.side_effect = robot_call
            with patch("robots.baxter.toolkit.HttpRpcClient", side_effect=[robot, policy]):
                live_fixture = get_toolkit("baxter", runtime_kwargs={**runtime, "image": None, "budget": 2},
                                           dashboard_events=events, config=config)
            try:
                for _ in range(2):
                    result = live_fixture.execute_tool("apply_one_step", {"subtask": "pick up the red object"})
                    assert result.is_error
                assert calls.count("env.step") == 1 and live_fixture.remaining == 0
            finally:
                live_fixture.close()
        finally:
            toolkit.close()
            logging.shutdown()
    print("RPent registered Baxter, recorded the preview and refused a repeat after a lost motion receipt.")

if __name__ == "__main__":
    main()
