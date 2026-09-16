from pathlib import Path
from sglang.kernels.jit.utils import load_jit
def load():
    root=Path(__file__).resolve().parent;repo=root.parents[2]
    return load_jit('dsv4_exact_supply_screen',cuda_files=[str(root/'exact.cuh')],
        cuda_wrappers=[(n,'sglang::ExactSupplyScreen::'+n) for n in ['d0','d1','s2','s4','s6','s12']],
        extra_include_paths=[str(repo/'python/sglang/kernels/jit/csrc')],
        extra_cuda_cflags=['-O3','-fno-fast-math','-ffp-contract=off'])
