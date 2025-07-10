import torch
import numpy as np


@torch.no_grad()
def prep_for_answers(outputs, targets, names=None):
    outputs = outputs.cpu().detach().tolist()
    targets = targets.cpu().detach().tolist()


    if names is not None:
        if isinstance(names[0], list):
            names = [x for y in names for x in y]
        assert len(outputs) == len(targets) == len(names), f"len(outputs) == len(targets) == len(names). got {len(outputs)} == {len(targets)} == {len(names)}"
        return list(zip(names, outputs, targets))
    else:
        assert len(outputs) == len(targets), f"len(outputs) == len(targets). got {len(outputs)} == {len(targets)}"
        return list(zip(outputs, targets))
    
def get_weights(json_data, weight_type, device):
    assert weight_type in ('inv_freq', 'inv_freq_sqrt', None), "weight_type should be 'inv_freq_sqrt', 'sqrt' or None"
    if weight_type is None:
        return None 
    arr = np.concatenate(list(json_data['labels'].values()))
    freqs = np.bincount(arr) / arr.shape
    inv_freqs = 1.0 / torch.tensor(freqs).to(device)

    if weight_type == 'inv_freq':
        return inv_freqs
    elif weight_type == 'inv_freq_sqrt':
        return torch.sqrt(inv_freqs)