"""Match all downstream initialization to the paper coarse-code model.

Only the source lookup vocabulary/parameter sharing changes. No trained
checkpoint, item-to-code map, or coarse user list is read for initialization.
"""
import hashlib
import torch
from residual_experiment import seed_all
from small_task_comparison import ResidualList
from list_coarse_sid_model import make_model
from list_coarse_sid_energy_model import make_energy_model
from list_ambiguity_story import parameter_hash


def shared_hash(model, trainable_only=False):
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if n.startswith('source_items.') or (trainable_only and not p.requires_grad):continue
        h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def matched_pair(lists,raw_vocab,coarse_vocab,items,base,cfg,seed,family):
    device=lists.device
    dummy=torch.zeros((1,cfg['k']),device=device,dtype=torch.long)
    seed_all(seed)
    v1=ResidualList(dummy,raw_vocab,items,base,cfg).to(device)
    common={n:p.detach().clone() for n,p in v1.named_parameters() if not n.startswith('source_items.')}
    del v1
    def construct(tokens,vocab):
        seed_all(seed)
        model=(make_energy_model(tokens,vocab,items,base,cfg) if family=='energy'
               else make_model(tokens,vocab,items,base,cfg,'gaussian')).to(device)
        with torch.no_grad():
            for n,p in model.named_parameters():
                if n in common:p.copy_(common[n])
        return model
    coarse=construct(dummy,coarse_vocab)
    raw=construct(lists,raw_vocab)
    old_raw_hash=parameter_hash(raw)
    cp=dict(coarse.named_parameters())
    differences=[n for n,p in raw.named_parameters() if not n.startswith('source_items.') and not torch.equal(p,cp[n])]
    active_differences=[n for n,p in raw.named_parameters() if n in differences and p.requires_grad]
    before_source=raw.source_items.weight.detach().clone()
    if family=='energy':
        with torch.no_grad():
            for n,p in raw.named_parameters():
                if not n.startswith('source_items.'):p.copy_(cp[n])
        assert shared_hash(raw)==shared_hash(coarse)
    else:
        # Gaussian's unused frozen energy head does not affect training/inference.
        assert not active_differences
        assert shared_hash(raw,True)==shared_hash(coarse,True)
    assert torch.equal(before_source,raw.source_items.weight)
    record=dict(coarse_initial_sha256=parameter_hash(coarse),legacy_raw_initial_sha256=old_raw_hash,
        raw_initial_sha256=parameter_hash(raw),coarse_shared_sha256=shared_hash(coarse),
        raw_shared_sha256=shared_hash(raw),trainable_shared_match=shared_hash(raw,True)==shared_hash(coarse,True),
        original_differing_parameters=differences,original_differing_trainable_parameters=active_differences,
        raw_source_initialization_preserved=True)
    del coarse,common,cp,before_source
    return raw,record
