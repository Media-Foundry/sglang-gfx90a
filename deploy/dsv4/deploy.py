#!/usr/bin/env python3
"""原版 V4 / gfx90a 的迁移入口。默认不安装、不启动、不初始化 GPU。"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import urllib.request

KIT = Path(__file__).resolve().parent
REPO = KIT.parents[1]
PROFILES = ("ar", "ar-best", "dspark")
MANIFESTS = {
    "unique_manifest": "SGLANG_DSV4_DEBUG_CK_UNIQUE_MANIFEST",
    "ipc_manifest": "SGLANG_DSV4_DEBUG_H16_IPC_MANIFEST",
    "route_manifest": "SGLANG_DSV4_DEBUG_CK_ROUTE_MANIFEST",
}
BEST = """
SGLANG_DSV4_GFX90A_BF16_CK_FIXED_SLOT
SGLANG_DSV4_DEBUG_CK_REDUCE_VEC4
SGLANG_DSV4_DEBUG_PREFILL_POST_WAVE
SGLANG_DSV4_DEBUG_PREFILL_MIX_OWNER
SGLANG_DSV4_DEBUG_CK_DIRECT_DEQUANT
SGLANG_DSV4_C4_PREFILL_QUERY_WIDE
SGLANG_DSV4_C4_PREFILL_QUERY_OWNER_WIDE
SGLANG_DSV4_C4_PREFILL_OWNER_K32
SGLANG_DSV4_PREFILL_MHC_COMMON_FP32
SGLANG_DSV4_PREFILL_MHC_COMMON_SMALL
SGLANG_DSV4_PREFILL_MHC_COMMON_UNFUSED
SGLANG_DSV4_PREFILL_CK_ROUTE_PRODUCER
SGLANG_DSV4_PREFILL_FIRST_MIX_TABLE
""".split()
NATIVE = """
SGLANG_DSV4_PREFILL_POST_FUSED4
SGLANG_DSV4_PREFILL_MIX_REUSE4
SGLANG_DSV4_PREFILL_MIX_PAIR_COLUMNS
SGLANG_DSV4_PREFILL_ATTN_STAGE1
SGLANG_DSV4_C4_PREFILL_QUERY_OWNER
SGLANG_DSV4_C4_PREFILL_QUERY_REUSE4
SGLANG_DSV4_GFX90A_MHC_LARGE_M_BF16_GEMM
""".split()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    template = json.loads((KIT / "config.example.json").read_text())
    require(not (set(cfg) - set(template)), "配置中存在未知字段")
    template.update(cfg)
    cfg = template
    for key in ("model_path", "aiter_root", "rocm_root", *MANIFESTS):
        if cfg[key]:
            p = Path(cfg[key]).expanduser()
            cfg[key] = str((path.parent / p).resolve())
    cfg["aot_python_paths"] = [
        str((path.parent / Path(p).expanduser()).resolve())
        for p in cfg["aot_python_paths"]
    ]
    require(len(cfg["gpus"]) == 8 and len(set(cfg["gpus"])) == 8
            and all(type(x) is int and x >= 0 for x in cfg["gpus"]),
            "本入口只支持 8 个不重复的物理 GPU ordinal / TP8")
    require(type(cfg["port"]) is int and 1 <= cfg["port"] <= 65535, "port 无效")
    require(type(cfg["kv_tokens"]) is int and cfg["kv_tokens"] > 0, "kv_tokens 无效")
    # 不默认暴露未鉴权服务到公网。
    require(cfg["host"] == "127.0.0.1", "本入口只监听 127.0.0.1；远程访问请使用 SSH 隧道")
    return cfg


def settings(cfg, profile):
    result = dict(
        SGLANG_DIR=str(REPO), PYTHON_BIN=sys.executable,
        MODEL_PATH=cfg["model_path"], ROCM_ROOT=cfg["rocm_root"],
        HIP_VISIBLE_DEVICES=",".join(map(str, cfg["gpus"])),
        TP_SIZE="8", EP_SIZE="1", MOE_A2A_BACKEND="none",
        HOST=cfg["host"], PORT=str(cfg["port"]),
        MAX_TOTAL_TOKENS=str(cfg["kv_tokens"]), MEM_FRACTION_STATIC="0.96",
        CHUNKED_PREFILL_SIZE="32768", MAX_PREFILL_TOKENS="32768",
        PREFILL_MAX_REQUESTS="16", RANDOM_SEED="20260913", ENABLE_METRICS="1",
        NUMA_INTERLEAVE_ALL=str(int(cfg["numa_interleave"])),
        ENABLE_DETERMINISTIC_INFERENCE="0",
        SGLANG_DSV4_GFX90A_TP8_MULTI_REQUEST_PROFILE="1",
        SGLANG_DSV4_GFX90A_PREFILL_THROUGHPUT_PROFILE="1",
        SGLANG_DSV4_NATIVE_SMALL_M_FP32_GEMV="0",
        SGLANG_DSV4_C4_PREFILL_QUERY_PRODUCER="0",
        SGLANG_DSV4_C4_PREFILL_QUERY_PRODUCER_FIXED_LT="0",
        SGLANG_DSV4_GFX90A_FP4_RUNTIME_M="1",
        SGLANG_DSV4_C4_TRIVIAL_LOGITS_SKIP="1",
        SGLANG_DSV4_GFX90A_AR_INDEXER_EMPTY_TILE_SKIP="1",
        SGLANG_DSV4_GFX90A_TP8_BS1_WOA_GEMV="1",
        SGLANG_DSV4_GFX90A_TP8_M32_ATTN_MULTISTREAM="1",
        SGLANG_DSV4_GFX90A_TP8_M32_LEGACY_AR="1",
        SGLANG_DSV4_GFX90A_TP8_M32_AR_BLOCKS="4",
        SGLANG_DSV4_GFX90A_TP8_M32_GATE_PREFETCH="1",
        SGLANG_DSV4_GFX90A_TP8_M32_DOWN_UNIFORM="1",
        SGLANG_DSV4_GFX90A_TP8_M32_DOWN_FIXED_WARMUP="1",
        SGLANG_DSV4_GFX90A_TP8_M32_DOWN_PAIRED_GRAPHS="0",
        AITER_GFX90A_MXFP4_QUANT_MAX_ROWS="0",
        SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK="4096",
        SGLANG_MORI_DECODE_MAX_DISPATCH_TOKENS_PER_RANK="128",
        CUDA_GRAPH_MAX_BS_DECODE="64", CUDA_GRAPH_BS_DECODE="1 2 4 8 16 32 64",
    )
    result.update({key: "0" for key in BEST})
    for suffix in ("TP4_BS32_PROFILE", "DSPARK_TP8_BS32_PROFILE",
                   "DSPARK_M64_ANCHOR_ONLY_ROUTED", "DSPARK_M128_ANCHOR_ONLY_ROUTED",
                   "DSPARK_M128_PRE_ROUTER_COMPACT"):
        result["SGLANG_DSV4_GFX90A_" + suffix] = "0"
    result["SGLANG_DSV4_DSA_DENSE_ONLY_GRAPH"] = "0"
    result["SGLANG_DSV4_GFX90A_DSPARK_TP8_FULL_TARGET_PROFILE"] = "0"
    if profile == "ar-best":
        result.update({key: "1" for key in BEST})
        result.update({key: "1" for key in NATIVE})
        for key, env_key in MANIFESTS.items():
            require(cfg[key], f"ar-best 缺少 {key}；必须本机重建，不能静默回退")
            result[env_key] = cfg[key]
    if profile == "dspark":
        result.update({key: "0" for key in NATIVE})
        result.update(
            SGLANG_DSV4_GFX90A_DSPARK_TP8_FULL_TARGET_PROFILE="1",
            SGLANG_DSV4_GFX90A_DSPARK_M128_MHC_FUSION="1",
            SGLANG_DSV4_GFX90A_FP16_MHC_DOT="0",
            SGLANG_DSPARK_SYNC_ACCEPT_ACROSS_TP="1",
            SGLANG_DSPARK_SYNC_DRAFT_ACROSS_TP="1",
            SGLANG_RAGGED_VERIFY_MODE="static", SPECULATIVE_DSPARK_BLOCK_SIZE="3",
            SPECULATIVE_DRAFT_MODEL_PATH=cfg["model_path"],
            SGLANG_DSV4_GFX90A_TP8_M32_DOWN_FIXED_WARMUP="0",
            CUDA_GRAPH_MAX_BS_DECODE="32", CUDA_GRAPH_BS_DECODE="1 2 4 8 16 32",
        )
    result["PYTHONPATH"] = os.pathsep.join([str(REPO / "python"), *cfg["aot_python_paths"]])
    return result


def environment(cfg, profile):
    # 不继承旧试验的任意环境开关。仅保留运行/编译所需的基础环境。
    allowed = {"PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TERM", "TMPDIR",
               "CONDA_PREFIX", "VIRTUAL_ENV", "LD_LIBRARY_PATH", "LIBRARY_PATH", "CPATH"}
    env = {k: v for k, v in os.environ.items() if k in allowed}
    env.update(settings(cfg, profile))
    for key, suffix in (("LD_LIBRARY_PATH", "lib"), ("LIBRARY_PATH", "lib"), ("CPATH", "include")):
        env[key] = str(Path(cfg["rocm_root"]) / suffix) + (
            os.pathsep + env[key] if env.get(key) else "")
    return env


def checkpoint(path):
    root = Path(path).resolve()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
        require((root / name).is_file(), f"缺少模型文件：{root / name}")
    config = json.loads((root / "config.json").read_text())
    require(config.get("model_type") == "deepseek_v4", "仅支持原版 deepseek_v4，不支持 V4.1/其他模型")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values()))
    require(names, "空 weight_map")
    total = 0
    for name in names:
        file = (root / name).resolve()
        require(file.is_relative_to(root) and file.is_file(), f"分片缺失或越界：{name}")
        size = file.stat().st_size
        require(size > 8, f"分片为空：{name}")
        # 只读 safetensors 头；验证文件未截断，不把 167GB 权重加载进 RAM。
        with file.open("rb") as stream:
            length = int.from_bytes(stream.read(8), "little")
            require(0 < length <= min(size - 8, 128 * 1024**2), f"非法 safetensors header：{name}")
            header = json.loads(stream.read(length))
        ends = [v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"]
        require(ends and max(ends) + 8 + length == size, f"分片截断或长度异常：{name}")
        for tensor, shard in index["weight_map"].items():
            if shard == name:
                require(tensor in header, f"索引中的 tensor 不存在：{tensor}")
        total += size
    return {"shards": len(names), "file_bytes": total, "validation": "header/size/index；非全量 checksum"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(path):
    data = json.loads(Path(path).read_text())
    require(data.get("status") == "complete", f"构建未完成：{path}")
    require(sha(data["module"]) == data["module_sha256"], f"扩展 hash 不匹配：{path}")
    require(data.get("sources"), f"缺少源码 provenance：{path}")
    for source, expected in data["sources"].items():
        require(sha(source) == expected, f"构建依赖变更/路径失效：{source}；请本机重新构建")


def doctor(cfg, profile, imports=False):
    report = {"profile": profile, "python": sys.version, "checkpoint": checkpoint(cfg["model_path"])}
    require((REPO / "scripts/rocm_dsv4_flash.sh").is_file(), "需放在本 fork 完整源码树内")
    require((Path(cfg["rocm_root"]) / "bin/hipcc").is_file(), "缺少 ROCm hipcc")
    require((Path(cfg["aiter_root"]) / "aiter").is_dir(), "缺少 AIter 源码")
    if cfg["numa_interleave"]:
        require(shutil.which("numactl"), "缺少 numactl")
    settings(cfg, profile)
    if profile == "ar-best":
        for key in MANIFESTS:
            validate_manifest(cfg[key])
        require(cfg["paired_topology_verified"] is True,
                "先确认所选 ordinal 的相邻 rank 是同一 OAM 双 GCD，再设 paired_topology_verified=true")
    if imports:
        env = environment(cfg, profile)
        env.update(HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="")
        env["DSV4_EXPECTED_AITER"] = cfg["aiter_root"]
        env["DSV4_EXPECTED_MANIFESTS"] = json.dumps(
            [cfg[k] for k in MANIFESTS] if profile == "ar-best" else [])
        code = """import json, os
from pathlib import Path
import torch, triton, aiter, sglang, sgl_kernel, tvm_ffi
assert torch.version.hip, '必须使用 ROCm PyTorch'
assert Path(aiter.__file__).resolve().is_relative_to(Path(os.environ['DSV4_EXPECTED_AITER']).resolve()), '导入的 AIter 不是配置中的源码树；请安装该源码或纠正配置'
for filename in json.loads(os.environ['DSV4_EXPECTED_MANIFESTS']):
    toolchain = json.loads(Path(filename).read_text())['toolchain']
    assert toolchain['torch_version'] == torch.__version__, 'PyTorch 已变，重建扩展'
    assert toolchain['torch_hip'] == torch.version.hip, 'HIP 已变，重建扩展'
    assert toolchain['cxx11_abi'] == int(torch._C._GLIBCXX_USE_CXX11_ABI), 'C++ ABI 不匹配'
print('torch=',torch.__version__,'HIP=',torch.version.hip)
print('aiter=',aiter.__file__)
print('sgl_kernel=',sgl_kernel.__file__)
"""
        subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=180)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "doctor", "launch", "build-extras", "build-route", "smoke", "versions"))
    p.add_argument("--config", default=str(KIT / "config.local.json"))
    p.add_argument("--profile", choices=PROFILES, default="ar")
    p.add_argument("--imports", action="store_true", help="doctor：隐藏 GPU 后检查 imports")
    p.add_argument("--gpu-reserved", action="store_true", help="launch：确认这些 GPU 已获独占使用许可")
    p.add_argument("--output-dir", help="构建输出必须是不存在的新目录")
    args = p.parse_args()
    if args.command == "versions":
        print(json.dumps({d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}, indent=2, sort_keys=True))
        return
    cfg = load_config(args.config)
    env = environment(cfg, args.profile)
    command = ["bash", str(REPO / "scripts/rocm_dsv4_flash.sh"),
               "serve-dspark" if args.profile == "dspark" else "serve"]
    if args.command == "plan":
        print(json.dumps({"command": command, "explicit_settings": settings(cfg, args.profile),
                          "note": "其余默认值由当前 launcher 解析；这不是实际路径命中证明"}, indent=2, ensure_ascii=False))
    elif args.command == "doctor":
        print(json.dumps(doctor(cfg, args.profile, args.imports), indent=2, ensure_ascii=False))
    elif args.command in ("build-extras", "build-route"):
        require(args.output_dir, "请指定新的 --output-dir；不会覆盖现有构建")
        build = (REPO / "scripts/rocm/build_dsv4_prefill_extras.py" if args.command == "build-extras"
                 else KIT / "route/build.py")
        subprocess.run([sys.executable, str(build), "--aiter-root", cfg["aiter_root"],
                        "--rocm-root", cfg["rocm_root"], "--output-dir", str(Path(args.output_dir).resolve())],
                       env=env, check=True)
    elif args.command == "launch":
        require(args.gpu_reserved, "请先确认独占预约及空闲，再传 --gpu-reserved；脚本不会终止其他进程")
        doctor(cfg, args.profile, imports=True)
        with socket.socket() as sock:
            sock.bind((cfg["host"], cfg["port"]))
        # 此检查会初始化小型 GPU context，仅 launch 执行。不是资源预约锁。
        code = """import torch
assert torch.cuda.device_count() == 8, '需要 8 个可见 GCD'
for i in range(8):
    prop = torch.cuda.get_device_properties(i)
    assert prop.gcnArchName.split(':')[0] == 'gfx90a', (i, prop)
    free, total = torch.cuda.mem_get_info(i)
    assert total >= 60 * 1024**3, (i, '需要约64GiB/GCD', total)
    assert total - free < 1024**3, (i, '显存已占用，请检查其他任务', total-free)
print('8 GCD 基本检查通过；拓扑及独占预约仍由管理员保证。')
"""
        subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=120)
        os.chdir(REPO)
        print("前台启动；首次编译可能较久。Ctrl-C 停止本服务。", flush=True)
        os.execvpe(command[0], command, env)
    elif args.command == "smoke":
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        base = f"http://{cfg['host']}:{cfg['port']}"
        with opener.open(base + "/health", timeout=10) as response:
            require(response.status == 200, "health 未通过")
        for prompt in ("What is the capital of France? Answer in one sentence.",
                       "Write a Python function that deduplicates a list while preserving order. Explain briefly."):
            body = json.dumps({"text": prompt, "sampling_params": {"temperature": 0, "max_new_tokens": 128}}).encode()
            request = urllib.request.Request(base + "/generate", data=body, headers={"Content-Type": "application/json"})
            with opener.open(request, timeout=600) as response:
                result = json.load(response)
            require(isinstance(result.get("text"), str) and result["text"].strip(), "输出为空")
            print(json.dumps({"prompt": prompt, "response": result}, ensure_ascii=False))
        print("请求链路通过；请人工检查答案。这不是数值精度、长输出或性能验收。")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        print(f"部署检查失败：{error}", file=sys.stderr)
        sys.exit(2)
