# 双 G1 本体 ScaleBFM Residual 部署

本体推理支持无线直连和双有线 namespace 两种传输。本文的启动例子使用无线配置。
如果本体没有无线网卡，请使用 [双有线本体部署流程](ONBOARD_WIRED_ZH.md)，
通过现有 `g1a/g1b` 网卡分发 Vive；两种模式都在本体计算策略和关节控制。

本入口用于 `Dual_G1_MJ` 的离线 `dual_g1_scalebfm_residual_object`
任务：29 维 whole-body residual、201 维 current actor、actual box。
每台本体独立执行自身 FK、batch=1 ScaleBFM 和自己的 residual Actor。
主机仅运行 Vive 获取和位姿分发。在线 command 任务的 238 维 Actor、在线规划器、
partner-proprio 和 b1 观测不属于这个入口支持的契约。

## 两项训练观测变更

1. 对方 pelvis 和箱子均在自身实测 pelvis 坐标系表达。位置 3 维，
   相对旋转矩阵前两列按行展开为 6 维。完全不读取对方关节或对方 torso FK。
2. 离线任务的 reference tracking anchor 仍是 `torso_link`。
   观测中的参考角速度现在是
   `R(reference_torso_world).T @ reference_torso_angular_velocity_world`。
   这里既不是实测 torso 的旋转，也不是 pelvis 的旋转。自身 IMU 角速度保持本体帧。

ScaleBFM 自身的历史、参考关键刚体、关节顺序与 residual 合成动作历史保持训练语义。
观测位置：参考角速度 `[61:64]`，partner `[154:163]`，箱子 `[192:201]`。
标准化使用 checkpoint 内冻结的均值/方差。

## 文件与进程

主机：`src/publish_dual_vive.py`；本体：`src/deploy_onboard_scalebfm.py --robot a|b`。
本体 C++ bridge：`../g1_sim2real/scripts/run_onboard_bridge.sh`。
公共配置：`config/g1/onboard_scalebfm.yaml`，引用已有策略/PD 配置。

每个本体的 Python 和 C++ bridge 使用 `127.0.0.1:55001/55002`。
本体之间可复用端口；DDS 网口必须选择各自内部控制器网口，脚本默认 `eth0`。
外接主机网线、主机上的 g1a/g1b namespace 和 veth 不参与此部署。

先把 `onboard_scalebfm.yaml` 中三个示例 WLAN IP 换为实际地址。
把相同的配置、校准 JSON、模型、NPZ 和运动学 XML 复制到 A/B。
校准只在主机应用一次；本体保存 JSON 仅用于 SHA256 一致性检查。
相对路径以配置文件所在目录解析。允许两台通过 `--device` 选择不同设备；
控制/模型/参考/标定配置必须一致。
两台 `--actuate` 模式也必须一致，避免一台正常输出、另一台仅阻尼却共同进入任务。

## 导出新训练权重

旧 torso/world checkpoint 与新观测同维但不兼容，不能靠重标 manifest 转换。
必须使用实际按新契约训练的 checkpoint。以下命令的 checkpoint 路径需要替换：

```bash
cd motion_tracking/sim2real
uv run --extra onboard python scripts/prepare_dual_scalebfm_artifacts.py \
  --source-root /home/bcj/wyr/Dual_G1_MJ \
  --residual-checkpoint /path/to/new_pelvis_training/checkpoints/best_agent.pt \
  --reference-bundle /home/bcj/wyr/Dual_G1_MJ/results/cfgen_box1m_lateral_0p5m_minus90.npz \
  --interaction-frame pelvis \
  --anchor-angular-velocity-frame reference-anchor \
  --output config/g1/dual_policy_artifacts_pelvis
```

这些参数是对真实训练设置的声明；仅从权重形状无法自动判断坐标系。
导出器保留旧契约默认值，因此必须显式给出新契约及新 checkpoint 路径。
本体入口校验 SHA256，并拒绝缺少 pelvis/reference-anchor 契约的 artifact。
旧的集中式入口也会读取 manifest 的两项字段；无字段的历史 artifact 保持 torso/world。

策略配置中的 control_mode、future_step、residual_scale 和 reference 起始帧需匹配训练。
本入口使用 50 Hz 和共同的 `motion_world`；不执行各机器人独立的参考重对齐。

## 安装和启动

在每个本体按原仓库方式安装 Python 环境、编译本体架构的 bridge：

```bash
cd motion_tracking/sim2real
uv sync --extra onboard
cd ../g1_sim2real
bash scripts/build.sh
G1_NET=eth0 bash scripts/run_onboard_bridge.sh
```

`onboard` extra 添加 PyTorch，不添加 OpenVR。机器人不需要运行 SteamVR。
Jetson/CUDA 的 PyTorch 安装需匹配本体已有系统环境；CPU/GPU 能否满足周期需在本体实测。
`cpu_affinity: []` 默认不绑定核，配置后可沿用原仓库分核运行方式。

在 Vive 主机启动：

```bash
cd motion_tracking/sim2real
bash scripts/run_vive_pose_publisher.sh --config config/g1/onboard_scalebfm.yaml
```

分别在 A/B 本体启动，首先不加 `--actuate` 做观测和通信检查：

```bash
# A 本体
bash scripts/run_onboard_scalebfm.sh --config config/g1/onboard_scalebfm.yaml --robot a
# B 本体
bash scripts/run_onboard_scalebfm.sh --config config/g1/onboard_scalebfm.yaml --robot b
```

两端都准备好后，任一本体遥控器 Start → DefaultPose，B → standing，A → task。
终端也支持 `s`、`b`、`a`、`x` 加 Enter。启动时已经按住的按钮被忽略，需释放后再按。
未加 `--actuate` 时所有输出 kp=0；进入非零控制阶段时使用现有禁用输出的阻尼语义，
不是完全断开 bridge。悬空/支撑条件下检查日志与观测，不能用此模式评估站立性能。

实际驱动时，在两台各自命令末尾添加：

```text
--actuate --confirm-actuation ENABLE_MOTORS
```

任务末帧后本地返回 ScaleBFM standing，双方完成确认后进入 finished。
每次任务部署只执行一遍参考，重新执行需重启部署并重新完成握手。
`--duration` 结束、Ctrl-C、Stop/x 或异常都会发送故障通知及本地阻尼命令。

## 网络数据与时钟

主机每 10 ms 尝试发送同一完整快照到 A/B，核心为 21 个 float：
三组 `position_w[3] + quaternion_xyzw[4]`。JSON 还携带箱子半尺寸、
采集时间、snapshot 序号、stream ID 和 calibration SHA256。
传输不包含关节目标、关节状态、torso 位姿、动作历史或 residual。
Tracker 暂时无效时，保留原快照的采集时间和序号，不把重发当成新采样。
超过 pose_timeout_s 会终止任务，当前 standing 也依赖 Vive，不能作为断 Vive 降级策略。

两台本体互发心跳、阶段、epoch、任务帧、就绪/停止状态和配置指纹。
A 协调未来时刻的 PREPARE → ACK → COMMIT → commit ACK，B 的关节控制始终独立。
消息重发、乱序丢弃、进程重启检测及有限失联期限避免静默使用旧会话。
任意网络分区下无法保证绝对同时动作，持续心跳和帧/阶段误差检查负责限制失步。

三个节点使用四时间戳 ping/pong 测量远端 wall clock 偏移；不直接相减不同机器的
monotonic 时间。保留最近 2 秒的低 RTT 样本，将半 RTT 作为时间不确定度。
默认只接受 RTT ≤10 ms 的样本，不满足时不进入控制。明显本地 wall clock 跳变会终止。
每个本体主循环从后台接收缓存取值，不阻塞等待无线数据；本地 bridge 读取最多等待 1 ms。

默认失效阈值：本地状态 60 ms、Vive 100 ms、对端 120 ms，帧偏差最多 2 帧。
推理/处理预算 18 ms，连续 3 次超限终止；命令间隔/调度停顿 100 ms 终止。
C++ bridge 自身 200 ms watchdog 保留，未增加无限重发旧目标的喂狗线程。
阈值可在配置中调整，但应先测量延迟和控制表现。

## 日志与验证

两台分别生成 `logs/onboard_scalebfm/<timestamp>_<side>/rollout.npz` 与 metadata。
metadata 保存 A stream 作为共同 run_id、配置指纹、观测契约和退出原因。
rollout 保存 epoch、参考帧、pose 序号、本地状态、三个位姿、观测、residual、
最终限幅目标、PD 增益和时间数据，便于跨机器关联。

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_onboard_network.py tests/test_onboard_policy.py tests/test_onboard_processes.py
```

测试包括两个真实部署进程通过本机 UDP 完成 Start/B/A/任务结束，以及停止 Vive 分发
后的故障退出；低层机器人状态由测试 bridge 模拟，所有命令验证 enable=0、kp=0。
这是软件链路验证，不代替本体 50 Hz 性能和双机带载行为验收。
