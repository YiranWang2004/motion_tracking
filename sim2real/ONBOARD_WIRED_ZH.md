# 双有线 namespace 下的 G1 本体推理

已准备的横移、抬升任务配置和本体启动命令见 [两台本体任务清单](ONBOARD_INSTALLED_TASKS_ZH.md)。

保留现有两块有线网卡和 `g1a/g1b` 隔离网络。每台本体运行自身的 C++ bridge、
ScaleBFM 和 residual；主机只采集 Vive、分发位姿和中转协同消息。
外接有线链路不传输关节角、关节速度、PD 参数、目标关节角或策略输出。
两项新观测约定（pelvis 交互位姿、reference-anchor 角速度）与无线本体模式完全相同。

## 数据路径

```text
Vive 主机默认 namespace
  ├─ 位姿 publisher ─ veth-g1a ─ g1a 位姿转发 ─ 有线网卡 A ─ A 本体策略
  └─ 位姿 publisher ─ veth-g1b ─ g1b 位姿转发 ─ 有线网卡 B ─ B 本体策略

A 本体策略 ─ g1a 协同转发 ─ 主机 team hub ─ g1b 协同转发 ─ B 本体策略

每个本体内部：策略 ⇄ 127.0.0.1 UDP ⇄ 本体 C++ bridge ⇄ 内部 DDS 控制器
```

因为 A/B 所在网段隔离且可使用相同 IP，协同消息通过主机中转。
中转器只允许 `g1-onboard-v1` 的 pose、team、ping/pong 格式，拒绝关节状态和命令包。
pose 包包含三组 pelvis/箱子世界位姿、采集时间、序号、标定标识、箱子尺寸；
team 包包含阶段、帧号、就绪、停止等状态。
时钟包和原始时间戳不改写，测得的 RTT 覆盖完整转发路径。
中转器不会生成心跳或替机器人确认启动，断链会触发原有数据年龄/失联保护。

## 配置

使用 `config/g1/onboard_scalebfm_wired.yaml`：

```yaml
network:
  transport: wired_namespace
  wired_topology: dual_network.yaml
```

网络地址直接从现有 `dual_network.yaml` 读取，不另建一套网卡配置：

| 项目 | A | B |
| --- | --- | --- |
| 主机默认 namespace 的 veth 地址 | 10.201.1.1 | 10.201.2.1 |
| namespace 内的 veth 地址 | 10.201.1.2 | 10.201.2.2 |
| namespace 内的物理网口地址 | 192.168.123.201 | 192.168.123.201 |
| 本体地址（默认 robot_ip） | 192.168.123.164 | 192.168.123.164 |
| 位姿：主机发送/本体接收端口 | 55300 / 55310 | 55301 / 55310 |
| 协同消息端口 | 55320 | 55320 |

`robot_ip` 必须确实是运行 Python 推理的本体计算机地址。如果其地址与机器人控制器
地址不同，可在 wired 配置中分别设置 `network.a.host` / `network.b.host` 为本体地址。
物理接口/namespace/veth 的变更统一修改 `dual_network.yaml` 并重新应用网络配置。
不要为本模式使用与 YAML 不一致的临时 G1_IFACE/G1_NETNS 等环境变量覆盖。

将 wired 配置、`dual_network.yaml`、所引用的策略 YAML、标定 JSON、artifact 和
standing/限幅配置一起复制到两台本体。本体只读取 topology YAML 中的地址，
不创建 namespace，也不需要主机网卡存在。模型准备和本体环境安装参见
[ONBOARD_SCALEBFM_ZH.md](ONBOARD_SCALEBFM_ZH.md)。

## 启动顺序

### 1. 结束旧的主机控制

先让旧任务停止，再结束主机 `run_dual_bridges.sh`。
不要同时运行主机 DDS bridge 和本体 DDS bridge。
保留现有 namespace，不执行 `--teardown`。
新的转发启动器发现 namespace 里仍有 `g1_udp_bridge` 会拒绝启动，不会自动杀掉它。

### 2. 主机检查/建立原有网络

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
sudo bash scripts/setup_dual_network.sh --check
# 如果本次开机尚未建立，先执行：
# sudo bash scripts/setup_dual_network.sh --apply
```

### 3. 主机启动有线转发组，保持终端运行

```bash
bash scripts/run_onboard_wired_relays.sh \
  --config config/g1/onboard_scalebfm_wired.yaml
```

此命令先在当前终端通过 sudo 提权监督进程，再检查现有网络，在 `g1a/g1b`
各启动一个轻量转发进程，并在默认 namespace 启动 team hub（以原用户身份运行）。
提权继续使用 uv 已选定的 Python 解释器；无需通过 sudo 运行 uv，也无需配置免密 sudo。
子进程不再单独调用 sudo，避免新会话无法复用终端认证缓存而出现
`sudo: a password is required`。监督进程退出时直接清理子进程组，不依赖 sudo 缓存有效期。
需要 sudo 进入 namespace，但不修改路由、NAT 或接口。
看到 `a`、`b`、`hub` 三条 `relay ready` 后，才表示三个转发进程都已完成初始化。
任一转发进程退出会停止整个转发组。Ctrl-C 只停止转发，不删除 namespace。

### 4. 主机启动 Vive publisher

另一终端运行：

```bash
cd /home/bcj/wyr/motion_tracking/sim2real
bash scripts/run_vive_pose_publisher.sh \
  --config config/g1/onboard_scalebfm_wired.yaml
```

仍在默认 namespace 中运行 SteamVR/OpenVR，不把 Vive 进程放进 `g1a/g1b`。

### 5. 各本体启动本地 bridge 和策略

通过各自 namespace SSH 登录本体，例如（首次连接仍需核对主机密钥）：

```bash
sudo ip netns exec g1a ssh -o HostKeyAlias=unitree-g1-a unitree@192.168.123.164
sudo ip netns exec g1b ssh -o HostKeyAlias=unitree-g1-b unitree@192.168.123.164
```

下面命令在本体上执行，目录按本体实际仓库位置调整：

```bash
# A 在 g1_sim2real/ 下运行 bridge（已检查本体接口名）
G1_NET=enP8p1s0 bash scripts/run_onboard_bridge.sh

# B 在 g1_sim2real/ 下运行 bridge
G1_NET=eth0 bash scripts/run_onboard_bridge.sh

# A 本体另一个终端，在 sim2real/ 下
bash scripts/run_onboard_scalebfm.sh \
  --config config/g1/onboard_scalebfm_wired.yaml --robot a

# B 本体另一个终端，在 sim2real/ 下
bash scripts/run_onboard_scalebfm.sh \
  --config config/g1/onboard_scalebfm_wired.yaml --robot b
```

初次检查不加 `--actuate`；这使用零刚度/阻尼输出，不是站立控制。
准备实际驱动时，重启两台策略，均添加 `--actuate --confirm-actuation ENABLE_MOTORS`。
操作仍为 Start/s → DefaultPose → B/b → standing → A/a → task，Stop/x 停止。
端口 55001/55002 只在本体 loopback 使用。

## 切回无线

停下当前任务和转发组，两台本体及主机 publisher 都改用
`config/g1/onboard_scalebfm.yaml`（`transport: wireless`），填写三台无线 IP。
无线模式无需有线转发组，机器人协同消息恢复直接互发。
不指定 transport 的旧无线配置仍按 wireless 处理。

## 验证与限制

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_onboard_wired.py tests/test_onboard_processes.py tests/test_onboard_network.py
```

测试用不同 loopback 地址模拟接口，运行完整转发路径和两个实际部署进程，
验证时钟、Start/B/A、任务完成、Vive 丢失退出和无线兼容。
测试不创建真实 namespace、不连接机器人，真实网卡和带载行为仍需现场验证。
本体闭环不再依赖外接线传递关节状态/动作，但线断后 Vive 和协同消息仍会丢失，
系统会终止任务；机内推理不意味着断线后可以继续搬运。
