import numpy as np
import json 
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve


def calc_frame_level_map(ans, is_frame_level, class_names):
    with open('/data/petr/caltech_mice/task1_classic_classification/calms21_task1_test.json', 'r') as f:
        test_data = json.load(f)

    all_data = {}
    logits = {}

    # only works with 16 frame chunks!
    if is_frame_level:
        tmp = np.linspace(0.4, 0.6, 8)
        window = np.concatenate([tmp, np.flip(tmp)])[:, None]
    else:
        tmp = np.linspace(0, 1, 8)
        window = np.concatenate([tmp, np.flip(tmp)])[:, None]

    for k in test_data['annotator-id_0'].keys():
        all_data[k.split('/')[-1]] = test_data['annotator-id_0'][k]['annotations']
        logits[k.split('/')[-1]] = np.zeros((len(test_data['annotator-id_0'][k]['annotations']), 4))

    for el in ans:
        name = el[0]
        preds = np.array(el[1])
        fn = name.split('F')[1]
        if is_frame_level:
            start = int(name.split('_')[-4])
            end = int(name.split('_')[-2])
            frame = int(name.split('_')[-1])
            
            ind = start + frame
            logits[fn][ind, :] += preds * window[frame, 0]
        else:
            start = int(name.split('_')[-3])
            end = int(name.split('_')[-1])

            logits[fn][start : end + 1, :] += preds[None, :] * window


    preds = []
    targets = []
    for key in logits.keys():
        preds.append(logits[key])
        targets.append(all_data[key])

    preds = np.concatenate(preds, 0)
    targets = np.concatenate(targets, 0)

    class_names = eval(class_names)

    aps = []
    res = {}
    for cls_ind, cls_name in class_names.items():
        ap = average_precision_score(targets == cls_ind, preds[:, cls_ind])
        if cls_name != 'other':
            aps.append(ap)
        res[f'ap_{cls_name}'] = ap 
    return sum(aps) / len(aps)

def calculate_multiclass_metrics(ans, class_names, prefix=''):
    class_names = eval(class_names) # {0: 'attack', 1: 'invest', 2: 'mount', 3: 'other'}
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