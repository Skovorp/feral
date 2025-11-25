import torch
import numpy as np
import json 
import random
import os

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
    
def get_weights(json_data, weight_type, device):
    assert weight_type is None or weight_type in ('inv_freq', 'inv_freq_sqrt'), "weight_type should be 'inv_freq', 'inv_freq_sqrt' or None"
    if weight_type is None:
        return None 
    arr = np.concatenate([json_data['labels'][x] for x in json_data['splits']['train']])
    if len(arr.shape) == 1:
        freqs = torch.tensor(np.bincount(arr) / arr.shape)
        ratio = (1.0 / freqs).to(device)
    elif len(arr.shape) == 2:
        freqs = torch.tensor(arr.mean(0))
        ratio = ((1 - freqs) / freqs).to(device)    
    if freqs.min().item() <= 0: 
        print(f"!!! Some classes don't have any examples. Class frequencies: {freqs}")
        ratio = torch.clamp(ratio, max=1000000.0)
        
    if weight_type == 'inv_freq':
        return ratio
    elif weight_type == 'inv_freq_sqrt':
        return torch.sqrt(ratio)
    

def get_random_run_name():
    sizes = [
        "big", "huge", "giant", "massive", "jumbo", "colossal",
        "mega"
    ]

    adjectives = [
        "beautiful", "graceful", "fluid", "lively", "vibrant", "dynamic", "elegant",
        "spirited", "joyous", "expressive", "fiery", "playful",
        "uplifting", "magnetic", "mesmerizing", "soulful",
        "captivating", "hypnotic", "athletic",
        "sparkling", "sensational"]
    
    cool_animals = [
        "lemur", "platypus", "wombat", "armadillo", "capybara",
        "meerkat", "sloth", "pangolin", "koala", "okapi",
        "yak", "ibis", "cassowary", "toucan", "tapir",
        "gazelle", "lynx", "ocelot", "caracal", "manatee",
        "walrus", "narwhal", "aardvark", "marmot", "porcupine",
        "badger", "jackal", "civet", "quail", "peacock",
        "emu", "sea_otter", "red_panda", "mongoose", "alpaca",
        "reindeer", "ibex", "puffin", "heron", "kookaburra"
    ]
    return '_'.join([
        random.choice(sizes),
        random.choice(adjectives),
        random.choice(cool_animals)
    ])

def last_nonzero_index(arr):
    out = np.empty_like(arr, dtype=int)
    last_idx = -1
    for i in range(len(arr)):
        if arr[i] != 0:
            last_idx = i
        out[i] = last_idx
    return out

def next_nonzero_index(arr):
    out = np.empty_like(arr, dtype=int)
    next_idx = -1
    for i in reversed(range(len(arr))):
        if arr[i] != 0:
            next_idx = i
        out[i] = next_idx
    return out

def suggested_num_workers():
    max_num_worker_suggest = None
    if hasattr(os, 'sched_getaffinity'):
        try:
            max_num_worker_suggest = len(os.sched_getaffinity(0))
        except Exception:
            pass
    if max_num_worker_suggest is None:
        # os.cpu_count() could return Optional[int]
        # get cpu count first and check None in order to satify mypy check
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            max_num_worker_suggest = cpu_count
    return max_num_worker_suggest

if __name__ == "__main__":
    print(next_nonzero_index([1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0]))
