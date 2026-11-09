# 上游来源

当前板端 π0.5 使用 [LeRobot v0.4.4](https://github.com/huggingface/lerobot/tree/8fff0fde7c79f23a93d845d1a50e985de01f8b8a)、[OpenPI 兼容 Transformers 提交](https://github.com/huggingface/transformers/tree/dcddb970176382c0fcf4521b0c0e6fc15894dfe0) 和 [π0.5-LIBERO 固定检查点](https://huggingface.co/lerobot/pi05-libero/tree/9ba5431256f4136a01c224ab0307b0d3035cc92c)。运行源码归档保留各自许可；模型文件哈希在 `deploy/pi05-manifest.json`。`cix_harness/pi05.py` 调用上游真实模型、前后处理器，并增加低主存流式加载与 Baxter 观测/动作适配。

`cix_harness/upstream/kdl_parser.py` 来自控制主机已安装的 Baxter SDK `baxter_pykdl/src/baxter_kdl/kdl_parser.py`，保留 Georgia Tech Research Corporation 的 BSD 许可全文。项目仅移植 Python 3 语法、移除旧演示入口；`kinematics.py` 调用 PyKDL 的带关节限位数值逆解，并增加正解一致性与关节连续性检查。控制机旧工作区中的 Python 2 URDF 副本不适用，运行时优先使用 ROS Noetic 自带解析器。

RPent 以 Apache-2.0 发布。选用提交 [0fa63c5a191df7665b03a1569d7a46d85df6b175](https://github.com/RLinf/RPent/tree/0fa63c5a191df7665b03a1569d7a46d85df6b175)。许可全文在 [RPent-LICENSE](RPent-LICENSE)。

| 本地文件 | 来源及改动 |
| --- | --- |
| `cix_harness/upstream/rpc_client.py` | 直接复用 `rpent/utils/rpc/rpc_client.py` 中的客户端基类及响应校验。 |
| `cix_harness/rpc.py` | 参考同版本 `rpent/utils/rpc/http_rpc.py` 改编。只接受本机回环连接，使用带大小限制的 JSON，将图像做成可复用的 data URL，机器人 RPC 串行执行。 |
| `cix_harness/rollout.py` | 改编自 `robots/franka/tools.py` 中的 `vla_grasp` 有界预测与执行循环。这里接收通用子任务，每次取 Baxter 新观测、检查动作契约和回执。 |

VLM 运行时取自 [llama.cpp b12121](https://github.com/ggml-org/llama.cpp/releases/tag/b12121)。部署包保留上游发布的原始 ARM64 归档及内含许可。Qwen 量化权重和视觉投影来自 [ggml-org](https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/tree/5037fcf163dd95d1e41d1974465f0898ed108ca2)。OpenVLA 的权重、处理器和自定义建模代码来自[固定模型快照](https://huggingface.co/openvla/openvla-7b/tree/47a0ec7fc4ec123775a391911046cf33cf9ed83f)。模型以各自仓库的许可和模型卡为准。

TY1100 厂商镜像来自随设备手册第 5.4.6 节。已查阅该镜像内的 `/root/README.md` 和 `/root/apps/openvla_infer.py`。项目按其加载方式编写服务，通过 Docker 引用原始镜像。公开权重所需的 Transformers、tokenizers 在派生镜像中单独锁定，底层 torch 和驱动使用厂商版本。

D455 的采集代码改编自用户已有 [Baxter-claw 驱动](https://github.com/SkyGit111/Baxter-claw/blob/cf923aea646d483b8dbe162eda4855ad560f9797/bridge/drivers/realsense_driver.py)，保留 640×480、30 FPS 和 BGR 颜色流，将深度处理替换为有时效检查的 JPEG 观测。示例 `examples/baxter-d455.jpg` 为同一提交中的[历史实拍原图](https://github.com/SkyGit111/Baxter-claw/blob/cf923aea646d483b8dbe162eda4855ad560f9797/debug_d455_phase1_raw.jpg)。两者沿用 [Baxter-claw 的 MIT 许可](Baxter-claw-LICENSE)。


2026-11-10 原语路线：`cix_harness/primitives.py` 的接近、抓取、抬升、放置阶段顺序参考用户 Baxter-claw 固定提交 `cf923aea646d483b8dbe162eda4855ad560f9797` 的 `bridge/primitives.py`；SDK 语义参考 `bridge/drivers/baxter_driver.py`。MIT 许可保留在 Baxter-claw-LICENSE。新代码重写为独立阶段计划、真实 RGB-D/base/TCP 变换和现有连续 IK 执行，不移植固定桌面深度、自动校准或双臂模块。

HarnessVLA 是架构参考，不存在未经注明的 HarnessVLA 代码复制。其高层 agent、解析原语和冻结 VLA 的职责分工，与本项目当前两条独立路线的差异见 README 文首。
