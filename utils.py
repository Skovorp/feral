import torch
import numpy as np
import json 
from metrics import generate_empty_logits, ensemble_predictions


@torch.no_grad()
def prep_for_answers(outputs, targets, names=None):
    outputs = outputs.cpu().detach().tolist()
    if targets is not None:
        targets = targets.cpu().detach().tolist()
    if names is not None:
        if isinstance(names[0], list):
            names = [x for y in names for x in y]

    if names is not None:
        if targets is not None:
            assert len(outputs) == len(targets) == len(names), f"len(outputs) == len(targets) == len(names). got {len(outputs)} == {len(targets)} == {len(names)}"
            return list(zip(names, outputs, targets))
        else:
            assert len(outputs) == len(names), f"len(outputs) == len(names). got {len(outputs)} == {len(names)}"
            return list(zip(names, outputs))
    else:
        assert len(outputs) == len(targets), f"len(outputs) == len(targets). got {len(outputs)} == {len(targets)}"
        return list(zip(outputs, targets))
    
def save_inference_results(ans, ema_ans, predict_per_item, labels_json, save_fn):
    out = {}
    ans_logits = generate_empty_logits(labels_json, 'inference')
    out['preds'] = ensemble_predictions(ans, predict_per_item, ans_logits)
    
    if len(ema_ans) > 0:
        ema_logits = generate_empty_logits(labels_json, 'inference')
        out['ema_preds'] = ensemble_predictions(ema_ans, predict_per_item, ema_logits)
    with open(save_fn, 'w') as f:
        json.dump(out, f)
    
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