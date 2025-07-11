import numpy as np
import json 
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import re
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import json
import matplotlib.cm as cm
import PIL


def calc_frame_level_map(ans, predict_per_item, labels_json, partition):
    class_names = {int(k): v for k, v in labels_json['class_names'].items()}

    logits = {}
    for k in labels_json['splits'][partition]:
        logits[k] = np.zeros((len(labels_json['labels'][k]), len(class_names)))
    logits = ensemble_predictions(ans, predict_per_item, logits)

    preds = []
    targets = []
    for fn in labels_json['splits'][partition]:
        preds.append(logits[fn])
        targets.append(labels_json['labels'][fn])

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


def ensemble_predictions(ans, predict_per_item, logits):
    is_frame_level = predict_per_item > 1
    assert predict_per_item % 2 == 0

    if is_frame_level:
        tmp = np.linspace(0.4, 0.6, predict_per_item // 2)
        window = np.concatenate([tmp, np.flip(tmp)])[:, None]
    else:
        tmp = np.linspace(0, 1, predict_per_item // 2)
        window = np.concatenate([tmp, np.flip(tmp)])[:, None]

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
    return logits


def generate_raster_plot(ans, predict_per_item, labels_json, partition):
    logits = {}
    for k in labels_json['splits'][partition]:
        logits[k] = np.zeros((len(labels_json['labels'][k]), len(labels_json['class_names'])))
    logits = ensemble_predictions(ans, predict_per_item, logits)
    class_names = {int(k): v for k, v in labels_json['class_names'].items()}
    all_data = {fn: labels_json['labels'][fn] for fn in labels_json['splits'][partition]}

    video_names = list(logits.keys())

    preds_list = []
    targets_list = []
    split_positions = []
    start = 0

    for name in video_names:
        pred = logits[name].argmax(1)
        label = np.array(all_data[name])

        preds_list.append(pred)
        targets_list.append(label)

        start += len(pred)
        split_positions.append(start)

    split_positions = split_positions[:-1]

    arr_pred = np.concatenate(preds_list)[None, :]
    arr_label = np.concatenate(targets_list)[None, :]
    arr_diff = (arr_pred != arr_label).astype(int)

    # Prepare colormaps and labels
    all_classes = sorted(set(np.unique(arr_pred)) | set(np.unique(arr_label)))
    num_classes = len(all_classes)
    base_cmap = cm.get_cmap('nipy_spectral')

    # Skip dark colors near 0.0 — start sampling from 0.05 or 0.1
    color_range = np.linspace(0.1, 1.0, num_classes)
    colors = [base_cmap(val) for val in color_range]

    cmap = ListedColormap(colors)
    diff_cmap = ListedColormap(['white', 'red'])

    labels = ['prediction', 'label', 'mismatch']
    class_id_to_name = class_names
    legend_elements = [mpatches.Patch(color=colors[i], label=class_id_to_name[i]) for i in all_classes]

    # Plot
    fig, axs = plt.subplots(3, 1, figsize=(32, 4), sharex=True)

    for i, arr in enumerate([arr_pred, arr_label, arr_diff]):
        cmap_used = cmap if i < 2 else diff_cmap
        axs[i].imshow(arr, aspect='auto', cmap=cmap_used, interpolation='nearest')
        axs[i].set_yticks([0])
        axs[i].set_yticklabels([labels[i]], fontsize=14, rotation=0, va='center')
        axs[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[i].tick_params(axis='y', which='both', left=False)
        for pos in split_positions:
            axs[i].axvline(pos, color='black', linewidth=2)

    # Add video names centered under each segment
    start = 0
    for name in video_names:
        length = logits[name].shape[0]
        center = start + length // 2
        wrapped_name = '\n'.join([name[i:i+15] for i in range(0, len(name), 15)])
        axs[-1].text(center, 1.2, wrapped_name, ha='center', va='top', fontsize=10)  # Moved up slightly
        start += length

    # Adjust layout
    plt.subplots_adjust(left=0.04, right=0.99, bottom=0.45, top=0.95)

    # Embedded legend
    legend_ax = fig.add_axes([0.1, 0.02, 0.8, 0.08])
    legend_ax.axis('off')
    legend_ax.legend(
        handles=legend_elements,
        loc='center',
        ncol=min(len(legend_elements), 6),
        fontsize=14,
        frameon=False
    )

    res = fig2img(fig)
    plt.close(fig)
    return res


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img

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