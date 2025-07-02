import numpy as np
import json 
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import re


def calc_frame_level_map(ans, is_frame_level, class_names, labels_json, partition):
    logits = {}
    all_data = {}

    for fn in labels_json['splits'][partition]:
        all_data[fn] = labels_json['labels'][fn]

    # only works with 16 frame chunks!
    if is_frame_level:
        tmp = np.linspace(0.4, 0.6, 8)
        window = np.concatenate([tmp, np.flip(tmp)])[:, None]
    else:
        tmp = np.linspace(0, 1, 8)
        window = np.concatenate([tmp, np.flip(tmp)])[:, None]

    for k in all_data.keys():
        logits[k] = np.zeros((len(all_data[k]), len(class_names)))

    for el in ans:
        name = el[0]
        preds = np.array(el[1])
        if is_frame_level:
            match = re.search(r"([^/]+)_from_(\d+)_to_(\d+)_(\d+)", name)
            fn = match.group(1)
            start = int(match.group(2))
            end = int(match.group(3))
            frame = int(match.group(4))
            
            ind = start + frame
            logits[fn][ind, :] += preds * window[frame, 0]
        else:
            match = re.search(r"([^/]+)_from_(\d+)_to_(\d+)", name)
            fn = match.group(1)
            start = int(match.group(2))
            end = int(match.group(3))

            logits[fn][start : end + 1, :] += preds[None, :] * window


    preds = []
    targets = []
    for key in logits.keys():
        preds.append(logits[key])
        targets.append(all_data[key])

    preds = np.concatenate(preds, 0)
    targets = np.concatenate(targets, 0)

    aps = []
    res = {}
    for cls_ind, cls_name in class_names.items():
        if len(targets.shape) == 1:
            is_positive = (targets == cls_ind)
        else:
            is_positive = targets[:, cls_ind]
        ap = average_precision_score(is_positive, preds[:, cls_ind])
        if cls_name != 'other':
            aps.append(ap)
        res[f'ap_{cls_name}'] = ap 
    return sum(aps) / len(aps)

def calculate_multiclass_metrics(ans, class_names, prefix=''):
    preds = np.array([x[-2] for x in ans])
    targets = np.array([x[-1] for x in ans])
    targets = targets.argmax(1)

    aps = []
    res = {}
    for cls_ind, cls_name in class_names.items():
        ap = average_precision_score(targets == cls_ind, preds[:, cls_ind])
        if cls_name != 'other':
            aps.append(ap)
        res[f'{prefix}_ap_{cls_name}'] = ap 
    res[f'{prefix}_map'] = sum(aps) / len(aps)
    return res