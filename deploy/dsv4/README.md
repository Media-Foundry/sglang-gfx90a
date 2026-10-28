# DeepSeek-V4-Flash / MI250 迁移入口

这是本 fork 的部署工具，不是任意显卡通用的安装器。目标为 **原版 V4 Flash、TP8、4 张 MI250（8 个 gfx90a GCD）、原始 checkpoint**。不包含 V4.1 / Engram，也不启用历史 anchor-only 近似验证。

提供预检、启动计划、本机扩展重建、前台启动和两条 smoke 请求。不会安装系统包、覆盖已有构建、终止其他任务，也不会自动占用 GPU。`launch` 必须显式确认已预约 GPU。

## 最短使用流程

先迁移完整源码、匹配的环境和 checkpoint，再在新机器执行：

```bash
cd /srv/src/sglang
cp deploy/dsv4/config.example.json deploy/dsv4/config.local.json
# 编辑 config.local.json：模型、AIter、ROCm、端口等路径。
export PYTHON_BIN=/srv/envs/DS/bin/python

# 默认只读文件；不初始化 GPU、不启动模型。
bash deploy/dsv4/run.sh doctor --profile ar
bash deploy/dsv4/run.sh plan --profile ar

# 在已独占预约的机器上启动；保持前台，Ctrl-C 停止本服务。
bash deploy/dsv4/run.sh launch --profile ar --gpu-reserved
```

另一个终端在日志明确 ready 后执行：

```bash
PYTHON_BIN=/srv/envs/DS/bin/python bash deploy/dsv4/run.sh smoke
```

`smoke` 先访问 health，再各发一次法国首都和 Python 去重代码请求；绕过 HTTP 代理。它只验证请求链路和非空输出，打印完整答案供人工检查，**不代表精度、确定性或性能验收**。首次 JIT 可耗时很久；不要把冷启动当作稳态性能。

## 环境要求与准备

| 项目 | 要求 / 注意事项 |
| --- | --- |
| GPU | 8 个 gfx90a GCD，每 GCD 约 64 GiB；本入口不自动降到 TP4、不支持 MI300/NVIDIA |
| 拓扑 | 最佳配置使用相邻 rank 配对；必须核实设备顺序，使每对为同一 MI250 OAM 双 GCD |
| OS / 驱动 | Linux x86-64，能访问 `/dev/kfd`、`/dev/dri`，ROCm runtime、hipcc、开发头文件与 PyTorch 匹配 |
| Python | 本机参考环境是 Python 3.12；扩展必须用实际部署的 Python/PyTorch 重编译 |
| PyTorch / Triton | ROCm 版本；不能装 CUDA wheel 替代。当前版本参考见 `environment.reference.json` |
| AIter / CK | 使用本项目验证过的定制源码、子模块及补丁。普通 PyPI AIter 或同名 git tag 不保证等价 |
| AOT kernels | 本 fork 的 ROCm `sgl_kernel`；不能误装默认 CUDA 版本 |
| 构建工具 | C/C++ 编译工具、hipcc、Python 开发头、Ninja、pybind11；部分 AIter 构建还依赖 CMake |
| CPU / NUMA | 安装 `numactl`，默认 `--interleave=all`；容器需允许相应 NUMA 策略 |
| RAM | 建议 256–512 GiB 级别并保留加载余量；这是迁移容量预算，不是已证明的最低需求 |
| 磁盘 | 权重本身约 167 GB（十进制）；另给环境、源码、编译/JIT 缓存预留空间，建议总空闲至少 250–350 GB |
| 容器 / IPC | 需 GPU 设备权限、足够共享内存、跨 GCD peer/IPC 能力；不能仅以 `torch.cuda.is_available()` 通过代替 |

Ubuntu 系统依赖示例（用户自行执行；脚本不会 sudo）：

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build numactl python3-dev pkg-config
```

使用 Conda Python 时还应确认其自身 include 目录完整；系统 `python3-dev` 不一定匹配 Conda 的 Python ABI。

**不能提供一个可信的通用 `pip install -r` 就重建全部环境。** 当前 PyTorch/Triton 含开发版构建，AIter/CK 有本地合同。优先迁移自建 wheelhouse 或同一基础镜像，并保存 AIter 的实际工作树。`environment.reference.json` 只是 2026-10-28 读到的安装元数据，不是 9 月跑分环境的完整锁文件，不应直接当 PyPI requirements 使用。

当前 `python/pyproject.toml` 含 CUDA 依赖；不要在 ROCm 环境直接无约束安装其全部依赖。先准备匹配的 ROCm 软件栈和其余 Python 依赖，再按需执行：

```bash
# 可选：源码通过本入口的 PYTHONPATH 即可访问；此命令不自动补依赖。
"$PYTHON_BIN" -m pip install --no-deps -e python

# AOT ROCm wheel 构建示例；在目标环境编译，不复用旧 .so。
cd python/sglang/kernels/aot
AMDGPU_TARGET=gfx90a "$PYTHON_BIN" setup_rocm.py bdist_wheel
# 检查 dist/ 中实际 wheel 名称，再 pip install --no-deps 那一个 wheel。
```

AIter 应按所迁移定制版本的构建说明安装。`aiter_root` 是源码/构建合同检查路径，**不是**自动安装命令。`doctor --imports` 隐藏 GPU 后检查核心导入；如果该版本模块本身要求 GPU 初始化，错误需单独评估，不能当作硬件坏了。

```bash
bash deploy/dsv4/run.sh doctor --imports
bash deploy/dsv4/run.sh versions > /tmp/dsv4-installed-versions.json
```

若 `sgl_kernel` 不在环境中而使用源码 build 目录，把其实际 Python 目录及扩展目录加入配置的 `aot_python_paths`。不要照抄旧机器的 `lib.linux-x86_64-cpython-312` 路径。

## 三种配置

| profile | 用途 | 限制 |
| --- | --- | --- |
| `ar` | 原生 AR，当前 launcher 的 TP8 多请求＋大 prefill 配置；图档位 C1/2/4/8/16/32/64 | 不启用需要外部 manifest 的最佳 prefill 扩展；不声称复现历史最高吞吐 |
| `ar-best` | 在 AR 上启用已记录的 unique-slot CK、H16 peer、route producer、MHC 等组合 | 必须本机重建三个 manifest、核实配对拓扑，再重新 E2E 验收 |
| `dspark` | 严格 full-target、γ=3、FP32 MHC fusion，保留跨 TP draft/accept 同步 | 仅提供 C1～C32 图档位；不是历史 TP4 anchor-only 约 1.5k 路径，也不是旧私有 C64 launcher |

三者默认 1,048,576 logical KV tokens、32K prefill chunk。容量由目标硬件实际验收；**1M logical pool 不表示已验证 C64×1M 上下文同时驻留**。更小 pool 可在配置改 `kv_tokens`，但不能继续使用 1M 的结果标签。

入口隔离继承环境，只保留基础系统/动态库变量，并显式指定上述 profile。旧 shell 的 `SGLANG_*`、`AITER_*`、`SPECULATIVE_*` 开关不会偷偷混入。`plan` 打印的是显式输入；launcher 仍会补充默认值，只有服务日志/路径检查才能证明最终实际命中。

## 最佳 prefill 扩展：本机两步重建

```bash
# 两个输出目录必须尚不存在；不覆盖旧构建、不修改 AIter 源码。
bash deploy/dsv4/run.sh build-extras --output-dir /srv/builds/dsv4-extras-v1
bash deploy/dsv4/run.sh build-route  --output-dir /srv/builds/dsv4-route-v1
```

构建 helper 对 AIter/CK 相关头文件执行 SHA256 合同检查。失败应核对源码和补丁，**不要删检查来强行编译**。编译日志在各模块输出目录的 `command-*.log`。

更新配置：

```json
{
  "unique_manifest": "/srv/builds/dsv4-extras-v1/unique/manifest.json",
  "ipc_manifest": "/srv/builds/dsv4-extras-v1/ipc/manifest.json",
  "route_manifest": "/srv/builds/dsv4-route-v1/route/manifest.json",
  "paired_topology_verified": true
}
```

这里只展示要修改的字段，保留完整配置的其他字段。确认 `rocm-smi --showtopo` / `amd-smi topology` 与可见设备顺序相符后，才设置拓扑确认项。

```bash
bash deploy/dsv4/run.sh doctor --profile ar-best --imports
bash deploy/dsv4/run.sh launch --profile ar-best --gpu-reserved
```

manifest 记录模块 hash、源码/头文件绝对路径及工具链。迁移目录、改依赖、换 Python/PyTorch/ROCm 后重新构建；不要修改 JSON 路径伪造复现。编译成功本身不是 GPU correctness oracle 通过。

`route/` 来自 `.agents/experiments/dsv4_ck_route_producer_20260916/{build.py,overlay.py,stage1.cu}`。迁移版本只新增显式 `--rocm-root`，去除 `/opt/rocm` 硬编码；保留两种 stage1 产物及头文件合同。

## 需要迁移哪些文件

原机参考位置：

- 本 fork 工作树：`/home/pc/Code/sglang`。
- 模型/DSpark 所用 checkpoint：`/home/pc/models/modelscope`；原索引引用 48 个 safetensors 分片。
- AIter 工作树：`/home/pc/pytorch/third_party/aiter`，包括 CK 子模块和本地补丁。
- Python 环境：`/home/pc/anaconda3/envs/DS`，建议用 wheelhouse/可重定位环境或镜像迁移，不直接复制后假定前缀可用。
- 历史交接：`.agents/HANDOFF_DSV4_FLASH_20261015.md`；详细实验及数值限制仍以原记录为准。

**当前工作树含未提交修改和未跟踪运行时代码，只有 `git clone` 不够。** 至少完整保存 `python/`、`scripts/`、`deploy/dsv4/` 及相关构建源码。迁移前记录 `git status --short`、`git rev-parse HEAD`、外部 AIter 的提交与 dirty 状态，并审阅待传文件，避免带走凭据。不要仅复制本文的小目录，却省略其依赖的 fork。

权重可单独用 `rsync -a --info=progress2` 迁移；不要加 `--delete`。入口按 index 查找真正分片，检查 safetensors header、索引 tensor 和文件长度；旁边存在 `.incomplete` 文件不会被误认为需要加载。这个检查不替代来源提供的全量 checksum。

## 新机器验收顺序

1. `doctor`：文件完整性、路径、构建合同；`doctor --imports`：核心模块可导入。
2. 管理员确认 8 GCD 空闲并独占预约；`launch` 额外检查 gfx90a、显存容量/占用及端口。此检查不是调度锁，不能阻止他人随后抢占。
3. 新进程启动后核对 TP8、原版模型、实际 pool、graph tiers、prefill/DSpark backend 与 manifest 命中。
4. smoke 后测真实异质代码、固定前缀 logits、长输出；法国首都通过不能证明全模型无 drift。
5. P：明确输入长度、零/混合 prefix hit、波次口径；D：1024 input、max2048 output、natural EOS，分别报完整请求和 common resident window。AR 与 DSpark 分开。
6. ABBA 对照，记录 cold/warm、峰值 HBM、实际 M、提交 token/acceptance。不能把换 workload 后的速度直接除以旧数字。

历史最佳 P 约 10.4k input tok/s 是特定 C16×8K、TP8、1M、组合开关的结果；不是新机器承诺。1024-in/max2048-out 的 AR C32 resident 约 1022 tok/s、C64 约 1297 tok/s；对应严格 DSpark C32 约 876 tok/s。历史约 1127 的 DSpark 使用不同短输入/输出协议；旧 TP4 约 1.5k 还有近似 target，不能混作迁移验收基线。

## 本工具验证范围

CPU 单元测试：

```bash
python3 -m unittest discover -s deploy/dsv4 -p 'test_*.py' -v
```

这些测试不安装软件、不启动服务、不申请 GPU。新增部署入口仍需要目标机器上的真实安装、重建和 E2E 验证；文档中未将其描述成已完成的迁移实测。

2026-10-28 本次已执行：13 项 CPU 单元测试通过、Shell/Python 语法检查通过；真实 checkpoint 的 48 个分片通过 header/size/index 检查，共 166,886,535,336 字节。迁移的 `overlay.py` 和 `stage1.cu` 与原实验源码逐字节相同。未执行本机扩展编译、GPU import/launch 或新机器 E2E。
