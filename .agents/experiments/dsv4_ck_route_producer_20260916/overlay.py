"""Pure text overlay: stage1 output ownership only, no installed CK mutation."""
import hashlib

GRID='ck/tensor_operation/gpu/grid/gridwise_moe_gemm.hpp'
EXPECTED='0d04e36b18b4a50e9331d5ad696f021e9b75bb2d23fc26163e37151794b5f7ac'
DESCRIPTOR='            IsInputGemm ? problem.NumTokens * problem.TopK : problem.NumTokens,'
SCATTER='''                    if constexpr(IsInputGemm)
                    {
                        token_offset = token_offset * problem.TopK + (fused_token >> 24);
                    }'''


def patch(source):
    assert hashlib.sha256(source.encode()).hexdigest()==EXPECTED
    # CK has both Run and Run_2Lds: keep their output contracts consistent.
    assert source.count(DESCRIPTOR)==source.count(SCATTER)==2
    source=source.replace(DESCRIPTOR,'''#if DSV4_ROUTE_MAJOR_STAGE1
            IsInputGemm ? problem.M : problem.NumTokens,
#else
'''+DESCRIPTOR+'''
#endif''')
    source=source.replace(SCATTER,'''                    if constexpr(IsInputGemm)
                    {
#if DSV4_ROUTE_MAJOR_STAGE1
                        token_offset = c_token_pos + m0;
#else
                        token_offset = token_offset * problem.TopK + (fused_token >> 24);
#endif
                    }''')
    return source
