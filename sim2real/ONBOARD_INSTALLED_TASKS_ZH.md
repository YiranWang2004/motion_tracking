# 两台本体已准备的任务

横移、抬升的有线配置现已启用 **TensorRT/CUDA GPU 推理**。安装方式、两台实测耗时、
新模型编译和回退方法见 [ONBOARD_TENSORRT_ZH.md](ONBOARD_TENSORRT_ZH.md)。
下方 CPU 性能表是首次安装时的历史对照，当前默认后端以 TensorRT 文档为准。

两台本体的仓库目录均为 `/home/unitree/wyr/motion_tracking`。
主机上的对应目录为 `/home/bcj/wyr/motion_tracking`。

| 任务 | 机载有线配置（相对于 sim2real） | 模型目录（相对于 config/g1） | 原始动作文件 |
| --- | --- | --- | --- |
| 横移 | `config/g1/onboard_scalebfm_wired_lateral.yaml` | `dual_policy_artifacts_box1m_lateral_20260908` | `cfgen_box1m_lateral_0p5m_minus90.npz` |
| 抬升/放下 | `config/g1/onboard_scalebfm_wired_lift.yaml` | `dual_policy_artifacts_lift1m_new` | `cfgen_box1m_lift_drop_minus90.npz` |

每套模型目录内的 `reference_bundle.npz` 就是该任务的动作文件，内容与训练仓库原文件一致。
配置已指定对应的策略参数和模型目录，正常启动不需要再传 `--reference-bundle`。
回放日志中的 `--replay-start-frame` 和 `--replay-speed` 只影响主机回放，不作为机器人控制参数。

复制排除了 logs、Git 忽略文件和 `.git`，并清除了本次同步的其他任务模型、动作。
本体独立创建 Python 环境并编译 ARM64 bridge，没有使用主机的虚拟环境或编译产物。
本体的 `sim2real/pyproject.toml` 为 PyTorch 增加了显式 CPU wheel 源，对应 `uv.lock` 在本体解析；
主机的这两个文件没有改动。以后重新同步这两个文件时应保留本体 CPU 源设置，避免下载 CUDA 依赖。

## 2026-09-12 安装验证

A（`unitree-g1-nx`，Ubuntu 22.04）和 B（`ubuntu`，Ubuntu 20.04）均完成 ARM64 bridge 编译，
动态库解析和 bridge `--help` 检查通过。Python 环境使用 PyTorch `2.14.0+cpu`。
两套 artifact 均通过 manifest SHA256 校验，分别执行了 100 次单机器人离线推理，
201 维观测、29 维输出及有限值检查通过。

| 本体 | 任务 | 推理中位数 | P95 |
| --- | --- | --- | --- |
| A | 横移 | 35.62 ms | 42.95 ms |
| A | 抬升 | 39.46 ms | 55.89 ms |
| B | 横移 | 53.65 ms | 61.13 ms |
| B | 抬升 | 53.67 ms | 62.05 ms |

这些是参考姿态构造输入的离线检查，不是实机闭环测试；安装验证期间也有其他检查进程。
当前 CPU 配置明显超过 50 Hz 的 20 ms 周期及配置中的 18 ms 处理预算。
编译与模型加载已完成，但实时性能尚不满足部署要求，应先优化推理后端并重新测量，
再进行实际任务控制。此次没有启动 DDS bridge、Vive 转发或电机控制。

B 的旧版 ARM glibc 曾在正常入口加载 PyTorch 时出现 static TLS block 错误。
部署入口现提前导入 PyTorch，使其 OpenMP 库先于其他原生库加载。

## 启动

### 同步仓库与标定

在主机 `sim2real/` 目录下执行：

```bash
# 默认同步整个 motion_tracking 仓库中未被 Git 忽略的文件到 A、B
bash scripts/sync_onboard_vive.sh

# 先查看差异，不写入；存在差异时退出码为 1
bash scripts/sync_onboard_vive.sh --check

# 保留机器人单独配置过的 ARM Python 依赖文件，其余照常同步
bash scripts/sync_onboard_vive.sh \
  --exclude sim2real/pyproject.toml \
  --exclude sim2real/uv.lock

# 兼容原来的仅 Vive 标定同步
bash scripts/sync_onboard_vive.sh --calibration-only --task lift
bash scripts/sync_onboard_vive.sh --calibration-only --task lift --check
```

默认包含已跟踪文件和未提交的新文件，遵守各级 `.gitignore` 及 Git 标准忽略规则，
包括 `!` 例外；即使文件已被 Git 跟踪，匹配忽略规则仍不传输。
`.gitignore` 文件本身会同步，`.git` 元数据不传输。源端已经删除的文件不传输，
也不删除本体文件。远端独有的虚拟环境、编译目录和 TensorRT 引擎会保留。
若主机存在同路径且未被忽略的文件，则按主机版本覆盖。
`--exclude` 接受仓库相对文件/目录路径，可重复指定，不是通配符。

默认会同步 `pyproject.toml` 和 `uv.lock`；本体这两个文件之前为 ARM/CPU wheel
单独配置过，覆盖后 `uv run` 可能重新解析或安装依赖。需要保留时使用上面的排除命令。
同步通过 rsync 增量传输并按内容校验，覆盖前备份至仓库外的
`/home/unitree/wyr/motion_tracking.sync-backups/<时间戳>/`，保留相对路径。
重复同步相同内容不会重复备份。备份不会自动清理。

也支持 `--task lateral` 或 `--config config/g1/onboard_scalebfm_wired_lift.yaml`。
全仓库模式下任务选择只决定连接配置，两个任务的未忽略文件都会同步。
脚本复用 namespace SSH 连接，需要时在当前终端认证；先检查两端连接和 rsync，
再依次同步。若第二台失败，第一台已完成的更新保留，修复后重跑即可。

仅标定模式仍读取两端任务 YAML 的 `vive_config`，保留原有 SHA256 校验、
并发修改检查和同目录 `.bak.<时间戳>` 备份。

同步不编译、不安装依赖、不重启进程。代码/配置更新后重新启动两台策略；
标定更新后还需重启主机 Vive publisher。bridge 和转发器无需因标定更新而重启。
模型源文件变化时，需要在各机器人重新构建匹配的 TensorRT 引擎。

### 主机一键打开四个本体终端

主机已在 `~/.local/opt/terminator` 安装 Terminator，命令入口为
`~/.local/bin/terminator`，也可以从桌面应用菜单打开。使用系统已有的 GTK/VTE 依赖，
没有替换系统默认终端。四分屏脚本使用主机 `/usr/bin/python3`、PyYAML 和 ConfigObj。

在主机 `sim2real/` 下运行：

```bash
# 仅预览四分屏，不连接机器人
bash scripts/run_onboard_terminals.sh --task lateral --preview

# A/B 各启动一个 bridge 和一个策略进程，策略默认不启用电机输出
bash scripts/run_onboard_terminals.sh --task lateral

# 或选择抬升任务
bash scripts/run_onboard_terminals.sh --task lift
```

窗口布局：左上 A bridge、左下 A 策略、右上 B bridge、右下 B 策略。
两份任务配置的 `network.a.robot_interface` 为 `enP8p1s0`，
`network.b.robot_interface` 为 `eth0`，这是两台本体实际的 DDS 网卡名。
脚本分别读取它们；需要临时覆盖时使用
`--robot-interface-a enP8p1s0 --robot-interface-b eth0`。
`--robot-interface` 可以统一覆盖两台，单台参数优先。
脚本复用已有 namespace SSH 连接；连接失效时，在调用脚本的终端通过 sudo 和 SSH
完成认证。策略窗口等待本体 bridge 的 55002 UDP 端口就绪后启动。
键盘广播默认关闭，各窗口分别接收输入；点击对应窗口即可操作。
进程退出后保留输出便于查看。策略模式支持 `s/b/a/x` 加 Enter。

此脚本启动本体的四个进程，主机的有线转发和 Vive publisher 仍按下文分别启动，
并应先于本体策略运行。不要重复打开实际运行的任务窗口组。
需要停止时先停止任务和两台策略，再关闭两台 bridge；脚本不会自动重启退出的进程。
只有显式同时传入 `--actuate --confirm-actuation ENABLE_MOTORS` 才启用电机输出；
实际运行前需确认启动日志显示 TensorRT 后端，并观察真实通信负载下的完整周期耗时。

### 分别启动各进程

网络检查、旧 bridge 的停止要求和完整操作顺序见 [有线部署文档](ONBOARD_WIRED_ZH.md)。
下面以横移任务为例。切换抬升任务时，将各处配置名改成 `onboard_scalebfm_wired_lift.yaml`，
两台本体必须选择相同任务。

主机 `sim2real/` 下，两个终端分别运行：

```bash
bash scripts/run_onboard_wired_relays.sh --config config/g1/onboard_scalebfm_wired_lateral.yaml
```

```bash
bash scripts/run_vive_pose_publisher.sh --config config/g1/onboard_scalebfm_wired_lateral.yaml
```

A 本体第一个终端：

```bash
cd /home/unitree/wyr/motion_tracking/g1_sim2real
G1_NET=enP8p1s0 bash scripts/run_onboard_bridge.sh
```

B 本体第一个终端：

```bash
cd /home/unitree/wyr/motion_tracking/g1_sim2real
G1_NET=eth0 bash scripts/run_onboard_bridge.sh
```

A 本体第二个终端：

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/unitree/wyr/motion_tracking/sim2real
bash scripts/run_onboard_scalebfm.sh --config config/g1/onboard_scalebfm_wired_lateral.yaml --robot a
```

B 本体第二个终端：

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/unitree/wyr/motion_tracking/sim2real
bash scripts/run_onboard_scalebfm.sh --config config/g1/onboard_scalebfm_wired_lateral.yaml --robot b
```

以上策略命令未启用 actuation。实际电机控制需在两台策略命令中均添加
`--actuate --confirm-actuation ENABLE_MOTORS`，并遵循有线部署文档的准备和操作流程。
复制、编译和离线推理检查本身不启动机器人控制。
