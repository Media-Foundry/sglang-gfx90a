from profile_labels import index_intervals


def test_owner_chain_is_not_mislabeled_as_logits():
    result=index_intervals({'20:53':'indexer_owner_chain_done'},20)
    assert result[-2:]==[(52,53,'index_owner_chain'),(53,54,'index_owner_tail_marker')]


def test_historical_local_path_labels_unchanged():
    assert index_intervals({'20:53':'indexer_logits_done'},20)[-2:]==[
        (52,53,'index_logits_plus_metadata'),(53,54,'index_topk_plus_metadata')]
    assert index_intervals({'20:53':'indexer_owner_chain_done'},22)[-1][2]=='index_topk_plus_metadata'
