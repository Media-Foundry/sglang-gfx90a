from pathlib import Path
from sglang.kernels.jit.utils import load_jit

def load():
    root=Path(__file__).resolve().parent;repo=root.parents[2]
    return load_jit('dsv4_premix_mfma_screen',cuda_files=[str(root/'premix.cuh')],
        cuda_wrappers=[(name,'sglang::MhcMfmaScreen::'+name) for name in ('map0','map1','u4','u8')],
        extra_include_paths=[str(repo/'python/sglang/kernels/jit/csrc')],
        extra_cuda_cflags=['-O3','-fno-fast-math','-ffp-contract=off'])

def load_split():
    root=Path(__file__).resolve().parent;repo=root.parents[2]
    return load_jit('dsv4_premix_mfma_split_screen',cuda_files=[str(root/'split.cuh')],
        cuda_wrappers=[(name,'sglang::MhcSplitScreen::'+name) for name in ('s4','s16','s32')],
        extra_include_paths=[str(root),str(repo/'python/sglang/kernels/jit/csrc')],
        extra_cuda_cflags=['-O3','-fno-fast-math','-ffp-contract=off'])
