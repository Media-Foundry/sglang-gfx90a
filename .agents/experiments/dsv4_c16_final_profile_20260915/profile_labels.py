"""Name the work actually inside marker bounds, not the old implementation."""


def index_intervals(paths, layer):
    owner = paths.get(f'{layer}:53') == 'indexer_owner_chain_done'
    return [(49,50,'index_weights'),(50,51,'index_query'),
            (51,52,'index_compressor'),
            (52,53,'index_owner_chain' if owner else 'index_logits_plus_metadata'),
            (53,54,'index_owner_tail_marker' if owner else 'index_topk_plus_metadata')]
