from typing import Any, Dict, List

import torch


def collate_fn(
        batch: List[Any],
        mode: str='train'
) ->  Dict[Any, Any]:
    """Collates batch into stacked tensors and label lists"""

    keys = batch[0].keys()
    eo_keys = batch[0].get('eo', {}).keys()
    collected = {k: ([] if k != 'eo' else {k_1: [] for k_1 in eo_keys}) for k in keys}

    for item in batch:
        for k in keys:
            if k == 'eo':
                for k_1 in eo_keys:
                    collected[k][k_1].append(item[k][k_1])
            else:
                collected[k].append(item[k])

    for k in keys:
        if k == 'eo':
            for k_1 in eo_keys:
                collected[k][k_1] = torch.stack(collected[k][k_1], dim=0)
        else:
            if type(collected[k][0]) == torch.Tensor:
                collected[k] = torch.stack(collected[k], dim=0)

    # TODO conversion from aux to text

    return collected
