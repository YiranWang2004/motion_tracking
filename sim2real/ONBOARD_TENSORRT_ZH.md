# 双 G1 TensorRT/CUDA 推理

横移和抬升的机载有线配置已默认启用 `acceleration.backend: tensorrt`。
两台本体均已安装对应源码，并分别在本机编译、校验了两套任务的 engine。
正常启动仍使用 `bash scripts/run_onboard_terminals.sh --task lift` 或 `--task lateral`。
策略启动时显示 `Inference backend: TensorRT/CUDA (local worker)`。

## 为什么使用本地 TensorRT worker

ScaleBFM 的 ScaleBridge 通过 Torch-TensorRT 编译、加载 GPU 模型。
本项目采用相同的 TensorRT/CUDA 执行后端，但为兼容两台现有系统，使用
PyTorch → ONNX → 本机 TensorRT engine，而不是安装相同的 Torch-TensorRT wheel。

| 本体 | 当前环境 | TensorRT Python |
| --- | --- | --- |
| A | JetPack 6.2.1、CUDA 12.6、TensorRT 10.3.0 | 系统 Python 3.10 |
| B | L4T R35.3.1、CUDA 11.4、TensorRT 8.5.2.2 | 系统 Python 3.8 |

策略进程继续使用已有 Python 3.10 CPU PyTorch 环境，计算运动学、观测及动作合成。
每台策略启动一个 `/usr/bin/python3` 子进程，使用系统已有 TensorRT 和 CUDA 执行两个网络。
进程间通过本机匿名管道传输张量；不经过主机、局域网或 UDP，不增加跨机器人数据依赖。
这里 `device: cpu` 指观测侧 PyTorch 的设备，网络实际由 TensorRT worker 在 GPU 上运行；
不要为这个后端另加 `--device cuda`。

ScaleBFM 的神经网络（含 task embedder）和本机器人 residual Actor 均在 GPU 上执行。
residual engine 包含冻结的均值/方差、输入限幅、标准化、MLP 和输出限幅。
pelvis 交互观测、reference-anchor 角速度、动作历史、residual_scale 与 CPU 实现一致。
engine 使用静态 batch=1、FP32 I/O，构建时关闭 TF32，没有启用 FP16。

## 配置和校验

```yaml
acceleration:
  backend: tensorrt
  directory: tensorrt_lift  # 横移为 tensorrt_lateral
  python: /usr/bin/python3
  torch_num_threads: 1
```

每个目录包含 `scalebfm.engine`、`residual_a.engine` 或 `residual_b.engine`、
`engine_manifest.json`，以及用于重建的 ONNX 和 `export.json`。
不同本体不共用 engine；启动检查源模型/参考等文件 SHA256、engine SHA256、
机器人角色、TensorRT 版本和输入输出形状。更换模型、参考或升级 TensorRT 后需要重新构建。
缺失或不兼容时直接报错，不静默回退到慢速 CPU。

运行时管道收发有 80 ms 总超时，原控制循环仍执行 18 ms 连续超时保护；
80 ms 只是 worker 故障时的等待上限，不是放宽控制周期。启动预热有独立较长超时，
发生在打开 bridge 通信之前。策略退出时清理 worker。

## 本机离线验证结果

每组使用参考姿态构造输入，先运行 PyTorch 基线，再运行 100 次 TensorRT 完整任务推理，
另测 100 次站立。完整任务推理含运动学、观测构造、两个网络、本地管道往返和动作合成。
不含真实桥接状态读取、Vive/协同通信及命令发送，不是带载闭环测试。

| 本体 | 任务 | 任务中位数 | 任务 P95 | 任务最大 | 站立中位数 |
| --- | --- | --- | --- | --- | --- |
| A | 抬升 | 15.14 ms | 15.64 ms | 16.72 ms | 9.64 ms |
| A | 横移 | 13.45 ms | 14.55 ms | 14.90 ms | 9.83 ms |
| B | 抬升 | 10.84 ms | 12.28 ms | 12.91 ms | 8.67 ms |
| B | 横移 | 11.86 ms | 12.37 ms | 13.81 ms | 8.49 ms |

四组最大目标关节角误差均小于 `5.4e-6 rad`。原始报告见
[`deployment_reports/tensorrt_onboard/`](deployment_reports/tensorrt_onboard/)。
这一轮所有任务与站立测量均低于 18 ms；A 的余量较小，真实通信负载下仍需观察
完整 `processing_ms`、最大耗时与连续超时计数。没有修改 GPU 功耗模式、系统时钟或控制保护阈值。
GPU 加速不能解决 Vive tracker 自身的跟踪失效。

## 新权重的导出与构建

主机 `sim2real/`：

```bash
uv pip install --python .venv/bin/python 'onnx>=1.16,<1.18'
.venv/bin/python scripts/export_onboard_tensorrt.py \
  --artifacts config/g1/dual_policy_artifacts_lift1m_new \
  --output /tmp/tensorrt_lift
```

将生成目录内容同步到每台本体的 `sim2real/config/g1/tensorrt_lift/`，随后在各本体构建：

```bash
cd /home/unitree/wyr/motion_tracking/sim2real
# A 使用 a；B 使用 b
/usr/bin/python3 scripts/build_onboard_tensorrt.py \
  --directory config/g1/tensorrt_lift --robot a
```

构建日志保存在该目录 `*.build.log`。每台仅构建自己的 residual engine。
构建完成后必须做数值和速度校验：

```bash
.venv/bin/python scripts/benchmark_onboard_tensorrt.py \
  --config config/g1/onboard_scalebfm_wired_lift.yaml \
  --engines config/g1/tensorrt_lift --robot a \
  --output /tmp/tensorrt_validation.json
```

横移任务将对应名称换成 `dual_policy_artifacts_box1m_lateral_20260908`、
`tensorrt_lateral` 和 `onboard_scalebfm_wired_lateral.yaml`。
上述导出、构建、验证命令均不会启动机器人控制。

## 回退

需要对照 CPU 实现时，将任务配置的 `acceleration.backend` 改为 `pytorch`，
两台策略需选择一致的后端并重新启动。CPU 对照用于诊断，之前测量不满足实时预算。
通用有线配置和无线配置仍保留原 PyTorch 路径；要启用加速，在对应配置中加入上面的
acceleration 段并提供本机 engine 即可，传输方式与推理后端独立。
