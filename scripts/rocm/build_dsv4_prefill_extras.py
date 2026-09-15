#!/usr/bin/env python3
"""Offline clean build of the validated gfx90a prefill helpers, without cached Ninja.

Original-V4 TP8 experimental profile only. Does not change installed AIter/CK or
enable a service feature. The output directory must not already exist.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sysconfig


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def overlays(aiter, output, contract):
    for relative, expected in contract['aiter_contract'].items():
        if digest(aiter/relative) != expected:
            raise ValueError(f'Unsupported AIter/CK contract: {relative}; review before rebuilding')
    base = aiter/'3rdparty/composable_kernel/include'
    device = 'ck/tensor_operation/gpu/device/impl/device_moe_gemm.hpp'
    scatter = 'ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_v7r3_scatter.hpp'
    raw = (base/device).read_text()
    needle = 'IsInputGemm ? InMemoryDataOperationEnum::Set : InMemoryDataOperationEnum::AtomicAdd;'
    assert raw.count(needle) == 1
    first = raw.replace(needle, 'InMemoryDataOperationEnum::Set; // EXPERIMENT: unique assignment slots only.')
    raw = (base/scatter).read_text()
    needle = ('                dst_bufs(i).template Update<DstInMemOp, dst_vector_t>(\n'
              '                    dst_offset, is_dst_valid, dst_vectors[i].template AsType<dst_vector_t>()[I0]);')
    assert raw.count(needle) == 1
    replacement = '''                // Unique-slot Set must not encode invalid writes as offset+2GiB:
                // the addition wraps into a live slot when the output exceeds2GiB.
                if constexpr(DstInMemOp == InMemoryDataOperationEnum::Set)
                {
                    if(is_dst_valid)
                        dst_bufs(i).template Update<DstInMemOp, dst_vector_t>(
                            dst_offset, true, dst_vectors[i].template AsType<dst_vector_t>()[I0]);
                }
                else
                {
                    dst_bufs(i).template Update<DstInMemOp, dst_vector_t>(
                        dst_offset, is_dst_valid, dst_vectors[i].template AsType<dst_vector_t>()[I0]);
                }'''
    second = raw.replace(needle, replacement)
    for relative, content in ((device, first), (scatter, second)):
        path = output/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return [output/device, output/scatter]


def compile_module(kind, source, directory, flags, includes, link, hipcc, provenance, sources):
    directory.mkdir()
    tag = hashlib.sha256(json.dumps(dict(kind=kind, source=digest(source), flags=flags,
        includes=includes, provenance=provenance, sources=sources), sort_keys=True).encode()).hexdigest()[:12]
    name = f'dsv4_prefill_{kind}_{tag}'
    obj, module, deps = directory/'entry.o', directory/(name+'.so'), directory/'entry.d'
    commands = [[str(hipcc), *flags, f'-DTORCH_EXTENSION_NAME={name}', *includes,
                 '-MMD', '-MF', str(deps), '-c', str(source), '-o', str(obj)],
                [str(hipcc), str(obj), *link, '-o', str(module)]]
    record = dict(status='building',name=name,module=str(module),sources=dict(sources),
                  commands=commands,toolchain=provenance)
    manifest = directory/'manifest.json'
    def save(): manifest.write_text(json.dumps(record,indent=2)+'\n')
    save()
    for i, command in enumerate(commands):
        print(f'{kind}: command {i+1}/{len(commands)} -> {directory}', flush=True)
        with (directory/f'command-{i}.log').open('w') as log:
            status = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
        if status:
            record.update(status='failed',failed_command=i,returncode=status);save()
            raise RuntimeError(f'Build failed; inspect {directory}/command-{i}.log')
    # Track non-system transitive headers, not just the three directly patched files.
    dependency_text = deps.read_text().replace('\\\n',' ')
    for dependency in shlex.split(dependency_text.split(':',1)[1]):
        path = Path(dependency).resolve()
        record['sources'][str(path)] = digest(path)
    record.update(status='complete',module_sha256=digest(module))
    if kind == 'ipc':
        record.update(source=str(source),source_sha256=digest(source))
    save()
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--aiter-root',type=Path,required=True)
    p.add_argument('--rocm-root',type=Path,default=Path('/opt/rocm'))
    p.add_argument('--output-dir',type=Path,required=True)
    args = p.parse_args()
    aiter, rocm, output = args.aiter_root.resolve(), args.rocm_root.resolve(), args.output_dir.resolve()
    if output.exists(): raise FileExistsError(f'Refusing to overwrite {output}')
    source_root = Path(__file__).resolve().parent/'dsv4_prefill_extras'
    contract_path = source_root/'build-contract.json'
    contract = json.loads(contract_path.read_text())
    hipcc = rocm/'bin/hipcc'
    version = subprocess.check_output([str(hipcc),'--version'],text=True)
    import torch
    import pybind11
    if not torch.version.hip: raise RuntimeError('Use a ROCm PyTorch environment')
    torch_root = Path(torch.__file__).resolve().parent
    abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    provenance = dict(compiler=str(hipcc),compiler_version=version,
        torch_version=torch.__version__,torch_hip=torch.version.hip,cxx11_abi=abi,
        python_include=sysconfig.get_path('include'),target='gfx90a')
    for relative, expected in contract['aiter_contract'].items():
        if digest(aiter/relative) != expected: raise ValueError(f'Unsupported AIter/CK contract: {relative}')
    output.mkdir(parents=True)
    patched = overlays(aiter,output/'overlay',contract)
    system = [torch_root/'include',torch_root/'include/torch/csrc/api/include',
              rocm/'include',Path(sysconfig.get_path('include'))]
    includes = [f'-I{pybind11.get_include()}'] + [x for path in system for x in ('-isystem',str(path))]
    common = ['-fPIC','-std=c++20',f'-D_GLIBCXX_USE_CXX11_ABI={abi}',
              '-D__HIP_PLATFORM_AMD__=1','-DUSE_ROCM=1']
    link = ['-shared','-mcmodel=large','-ffunction-sections','-fdata-sections','-Wl,--gc-sections','-Wl,--cref',
            '-L'+str(torch_root/'lib'),'-lc10','-lc10_hip','-ltorch_cpu','-ltorch_hip','-ltorch','-ltorch_python',
            '-L'+str(rocm/'lib'),'-lamdhip64']
    sources = {str(path):digest(path) for path in [Path(__file__).resolve(),contract_path,*patched,
        *[aiter/relative for relative in contract['aiter_contract']]]}
    ck_includes = [output/'overlay',aiter/'3rdparty/ck_helper',aiter/'3rdparty/composable_kernel/include',
        aiter/'3rdparty/composable_kernel/library/include',aiter/'csrc/include',
        aiter/'csrc/ck_gemm_moe_2stages_codegen',aiter/'csrc/include/torch']
    unique = compile_module('unique',source_root/'unique_stage2.cu',output/'unique',
        [*common,*contract['cuda_flags']], [*[f'-I{x}' for x in ck_includes],*includes],
        link,hipcc,provenance,sources)
    ipc = compile_module('ipc',source_root/'hip_ipc.cu',output/'ipc',
        [*common,'-O2','--offload-arch=gfx90a'],includes,link,hipcc,provenance,
        {str(Path(__file__).resolve()):digest(__file__),str(contract_path):digest(contract_path)})
    result = dict(status='complete',unique_manifest=str(unique),ipc_manifest=str(ipc),toolchain=provenance)
    (output/'build.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'profile.env').write_text(
        'export SGLANG_DSV4_GFX90A_BF16_CK_FIXED_SLOT=1\n'
        f'export SGLANG_DSV4_DEBUG_CK_UNIQUE_MANIFEST={shlex.quote(str(unique))}\n'
        f'export SGLANG_DSV4_DEBUG_H16_IPC_MANIFEST={shlex.quote(str(ipc))}\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__': main()
