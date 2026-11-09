"""Baxter extension for the pinned RPent CLI."""
from pathlib import Path
from uuid import uuid4

from rpent.memory import MemoryManager
from rpent.robots.prompt_bundle import PromptBundle
from rpent.robots.robot_spec import RobotSpec, RunConfig

from cix_harness.contract import instruction


def add_args(parser, use_dashboard):
    parser.add_argument("--baxter-task", required=True)
    parser.add_argument("--baxter-image", type=Path, help="Use a still image for preview")
    parser.add_argument("--baxter-robot-port", type=int, default=9102)
    parser.add_argument("--baxter-vla-port", type=int, default=9101)
    parser.add_argument("--baxter-motion-budget", type=int, choices=range(0, 9), default=0)


def config(args):
    goal = instruction(args.baxter_task)
    if args.baxter_image and args.baxter_motion_budget:
        raise ValueError("File input is only available for preview")
    if args.baxter_image and not args.baxter_image.is_file():
        raise ValueError("Image file unavailable")
    folder = Path(args.output_dir or Path("runs") / ("rpent-" + uuid4().hex[:12]))
    return RunConfig("baxter", folder, {"goal": goal, "budget": args.baxter_motion_budget},
                     {"task_name": goal})


def runtime(args, output_dir, dashboard_events, components=None):
    return [], {"image": args.baxter_image, "robot_port": args.baxter_robot_port,
                "vla_port": args.baxter_vla_port, "budget": args.baxter_motion_budget}


def prepare_memory(args, config):
    root = Path(args.memory_dir or "baxter-memory").resolve()
    root.mkdir(exist_ok=True, parents=True)
    config.prompt_vars["memory_dir"] = str(root)
    args.memory_dir = str(root)


def get_robot_spec():
    system = ("You supervise one Baxter arm. Always call observe_table first. Choose short English "
              "motor subtasks. In preview use preview_action then finish with a preview summary. "
              "In live mode call apply_one_step once per fresh observation and inspect the image after "
              "each action. Only declare physical success if the current image and state support it. "
              "Stop on an error, a missing target or an exhausted budget. Never invent motor actions.")
    prompts = PromptBundle(system=lambda variables: system,
                           user=lambda variables: {"Task": variables["goal"],
                                                   "Maximum physical steps": str(variables["budget"])})
    return RobotSpec(name="baxter", prompts=prompts, add_cli_args=add_args, parse_config=config,
                     init_runtime=runtime, prepare_memory=prepare_memory, is_real_robot=True)


def get_toolkit(*, runtime_kwargs, dashboard_events, config):
    from .toolkit import BaxterToolkit

    return BaxterToolkit(**runtime_kwargs, folder=config.output_dir, dashboard_events=dashboard_events,
                         memory=MemoryManager(config.prompt_vars["memory_dir"]))
