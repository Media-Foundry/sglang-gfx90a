"""Clean, isolated stage1 token/route output modules. Run after service ABBA.

Does not edit AIter headers or production loaders. A new output directory is
mandatory; manifests capture transitive headers and both command logs.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sysconfig

from overlay import GRID,patch


def main():
    root=Path(__file__).resolve().parent
    repo=root.parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aiter-root',type=Path,default=Path('/home/pc/pytorch/third_party/aiter'))
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    output=args.output_dir.resolve(); aiter=args.aiter_root.resolve()
    assert not output.exists(),f'Refusing to overwrite {output}'
    builder_path=repo/'scripts/rocm/build_dsv4_prefill_extras.py'
    spec=importlib.util.spec_from_file_location('prefill_build',builder_path)
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    contract_path=repo/'scripts/rocm/dsv4_prefill_extras/build-contract.json'
    contract=json.loads(contract_path.read_text())
    for relative,expected in contract['aiter_contract'].items():
        assert builder.digest(aiter/relative)==expected,relative
    original=aiter/'3rdparty/composable_kernel/include'/GRID
    content=patch(original.read_text())
    import torch
    import pybind11
    assert torch.version.hip
    rocm=Path('/opt/rocm'); hipcc=rocm/'bin/hipcc'
    torch_root=Path(torch.__file__).resolve().parent
    abi=int(torch._C._GLIBCXX_USE_CXX11_ABI)
    provenance=dict(compiler=str(hipcc),compiler_version=subprocess.check_output([str(hipcc),'--version'],text=True),
        torch_version=torch.__version__,torch_hip=torch.version.hip,cxx11_abi=abi,
        python_include=sysconfig.get_path('include'),target='gfx90a')
    output.mkdir(parents=True)
    overlay=output/'overlay'/GRID
    overlay.parent.mkdir(parents=True); overlay.write_text(content)
    system=[torch_root/'include',torch_root/'include/torch/csrc/api/include',
            rocm/'include',Path(sysconfig.get_path('include'))]
    ck=[output/'overlay',aiter/'3rdparty/ck_helper',aiter/'3rdparty/composable_kernel/include',
        aiter/'3rdparty/composable_kernel/library/include',aiter/'csrc/include',
        aiter/'csrc/ck_gemm_moe_2stages_codegen',aiter/'csrc/include/torch']
    includes=[f'-I{x}' for x in ck]+[f'-I{pybind11.get_include()}']+[
        x for path in system for x in ('-isystem',str(path))]
    flags=['-fPIC','-std=c++20',f'-D_GLIBCXX_USE_CXX11_ABI={abi}',
           '-D__HIP_PLATFORM_AMD__=1','-DUSE_ROCM=1',*contract['cuda_flags']]
    link=['-shared','-mcmodel=large','-ffunction-sections','-fdata-sections','-Wl,--gc-sections','-Wl,--cref',
          '-L'+str(torch_root/'lib'),'-lc10','-lc10_hip','-ltorch_cpu','-ltorch_hip','-ltorch','-ltorch_python',
          '-L'+str(rocm/'lib'),'-lamdhip64']
    sources={str(p):builder.digest(p) for p in [Path(__file__).resolve(),root/'overlay.py',
        root/'stage1.cu',builder_path,contract_path,original,overlay,
        *[aiter/relative for relative in contract['aiter_contract']]]}
    manifests={}
    for mode,value in [('token',0),('route',1)]:
        manifest=builder.compile_module('producer_'+mode,root/'stage1.cu',output/mode,
            [*flags,f'-DDSV4_ROUTE_MAJOR_STAGE1={value}'],includes,link,hipcc,provenance,sources)
        manifests[mode]=str(manifest)
    report=dict(status='complete',scope='isolated stage1 output ownership; GPU oracle still required',
                manifests=manifests,production_changed=False)
    (output/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
