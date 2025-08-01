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
    

# def last_nonzero_index(arr):
#     idx = np.arange(len(arr))
#     nonzero_idx = np.where(arr != 0, idx, -1)
#     acc = np.maximum.accumulate(nonzero_idx)
#     acc[acc == -1] = 0
#     return acc

# def next_nonzero_index(arr):
#     idx = np.arange(len(arr))
#     rev_idx = idx[::-1]
#     rev_arr = arr[::-1]
#     nonzero_rev = np.where(rev_arr != 0, rev_idx, -1)
#     acc = np.maximum.accumulate(nonzero_rev)
#     acc[acc == -1] = len(arr) - 1
#     return acc[::-1]

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

if __name__ == "__main__":
    print(next_nonzero_index([1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0]))