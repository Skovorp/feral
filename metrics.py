import numpy as np
import json 
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
import re
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import json
import matplotlib.cm as cm
import PIL
import cv2
import os
import traceback

from utils import last_nonzero_index, next_nonzero_index
from dataset import get_frame_count

def calc_frame_level_map(ans, labels_json, partition):
    class_names = {int(k): v for k, v in labels_json['class_names'].items()}

    logits = generate_empty_logits(labels_json, partition)
    logits = ensemble_predictions(ans, logits)

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

def calculate_f1_metrics(ans, labels_json, partition, is_multilabel, prefix):
    class_names = {int(k): v for k, v in labels_json['class_names'].items()}
    valid_classes = [cls_ind for cls_ind, cls_name in class_names.items() if cls_name != 'other']

    logits = generate_empty_logits(labels_json, partition)
    logits = ensemble_predictions(ans, logits)

    preds = []
    targets = []
    for fn in labels_json['splits'][partition]:
        preds.append(logits[fn])
        targets.append(labels_json['labels'][fn])

    preds = np.concatenate(preds, 0)
    targets = np.concatenate(targets, 0)

    if not is_multilabel:
        pred_labels = preds.argmax(1)
        target_labels = targets

        return {
            f'{prefix}_precision': precision_score(target_labels, pred_labels, labels=valid_classes, average='macro'),
            f'{prefix}_recall': recall_score(target_labels, pred_labels, labels=valid_classes, average='macro'),
            f'{prefix}_f1': f1_score(target_labels, pred_labels, labels=valid_classes, average='macro')
        }
    else:
        threshold = 0.85
        pred_labels = (preds > threshold) * 1
        target_labels = targets.astype(int)

        precisions = []
        recalls = []
        f1s = []

        for valid_class_ind in valid_classes:
            precisions.append(precision_score(target_labels[:, valid_class_ind], pred_labels[:, valid_class_ind]))
            recalls.append(recall_score(target_labels[:, valid_class_ind], pred_labels[:, valid_class_ind]))
            f1s.append(f1_score(target_labels[:, valid_class_ind], pred_labels[:, valid_class_ind]))
        return {
            f'{prefix}_precision': sum(precisions) / len(precisions),
            f'{prefix}_recall': sum(recalls) / len(recalls),
            f'{prefix}_f1': sum(f1s) / len(f1s)
        }

def ensemble_predictions(ans, logits):
    predict_per_item = max([int(x[0].split('_chunkind_')[1]) for x in ans]) + 1
    sum_weights = {fn: np.zeros(val.shape[0]) for (fn, val) in logits.items()}

    # uniform weights for now
    weights = np.ones(predict_per_item)[:, None]

    for el in ans:
        name = el[0]
        preds = np.array(el[1])

        # get ind
        match = re.search(r"([^/]+)_globalind_(\d+)_chunkind_(\d+)", name)
        fn = match.group(1)
        global_ind = int(match.group(2))
        chunk_ind = int(match.group(3))

        logits[fn][global_ind, :] += preds * weights[chunk_ind, 0]
        sum_weights[fn][global_ind] += weights[chunk_ind, 0]
    
    for fn in logits.keys():
        left_ind = last_nonzero_index(sum_weights[fn])
        right_ind = next_nonzero_index(sum_weights[fn])

        for i in range(sum_weights[fn].shape[0]):
            if sum_weights[fn][i] > 0:
                logits[fn][i, :] = logits[fn][i, :] / sum_weights[fn][i]
            else:
                # assert left_ind[i] < i and i < right_ind[i], (left_ind[i], i, right_ind[i], sum_weights[fn][i - 5 : i + 5])
                if left_ind[i] == -1 and right_ind[i] == -1:
                    continue
                elif right_ind[i] == -1:
                    logits[fn][i, :] = logits[fn][left_ind[i], :]
                elif left_ind[i] == -1:
                    logits[fn][i, :] = logits[fn][right_ind[i], :]
                else:
                    inv_left_dist  = 1 / (i - left_ind[i])
                    inv_right_dist = 1 / (right_ind[i] - i)
                    divisor = 1 if inv_left_dist + inv_right_dist == 0 else inv_left_dist + inv_right_dist
                    logits[fn][i, :] = (logits[fn][left_ind[i], :] * inv_left_dist + logits[fn][right_ind[i], :] * inv_right_dist) / divisor
    return logits

def generate_empty_logits(labels_json, partition):
    logits = {}
    for k in labels_json['splits'][partition]:
        logits[k] = np.zeros((len(labels_json['labels'][k]), len(labels_json['class_names'])))
    return logits

def generate_raster_plot(ans, labels_json, partition):
    try:
        logits = generate_empty_logits(labels_json, partition)
        logits = ensemble_predictions(ans, logits)
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
            assert len(label.shape) == 1, f"Rn rasters only work for single class classification. Labels must be 1d array, got {label.shape}"

            preds_list.append(pred)
            targets_list.append(label)

            start += len(pred)
            split_positions.append(start)

        split_positions = split_positions[:-1]

        arr_pred = np.concatenate(preds_list)[None, :]
        arr_label = np.concatenate(targets_list)[None, :]
        arr_diff = (arr_pred != arr_label).astype(int)

        # Prepare colormaps and labels
        base_cmap = cm.get_cmap('nipy_spectral')

        # Skip dark colors near 0.0 — start sampling from 0.05 or 0.1
        color_range = np.linspace(0.1, 1.0, len(class_names))
        colors = [base_cmap(val) for val in color_range]
        
        if 'other' in class_names.values():
            ind = list(class_names.values()).index('other')
            colors[ind], colors[-1] = colors[-1], colors[ind]

        cmap = ListedColormap(colors)
        diff_cmap = ListedColormap(['white', 'red'])

        labels = ['prediction', 'label', 'mismatch']
        legend_elements = [mpatches.Patch(color=colors[i], label=f"({i}) {class_names[i]}") for i in range(len(class_names))]

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
            ncol=min(len(legend_elements), 8),
            fontsize=14,
            frameon=False
        )

        res = fig2img(fig)
        plt.close(fig)
        return res
    except Exception:
        traceback.print_exc()

        fig, ax = plt.subplots(figsize=(32, 4))
        ax.text(0.5, 0.5, 'Error. See logs', fontsize=40, ha='center', va='center', color='red')
        ax.axis('off')
        
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
    targets = np.array([x[-1] for x in ans]).astype(int)
    # targets = targets.argmax(1)

    aps = []
    res = {}
    for cls_ind, cls_name in class_names.items():
        ap = average_precision_score(targets[:, cls_ind], preds[:, cls_ind])
        if cls_name != 'other':
            aps.append(ap)
        res[f'{prefix}_ap_{cls_name}'] = ap 
    res[f'{prefix}_map'] = sum(aps) / len(aps)
    return res


def generate_video_mismatches(ans, labels_json, partition, prefix, font_color=(255, 255, 255), look_around=30, output_path='result.mp4'):
    class_names = {int(k): v for k, v in labels_json['class_names'].items()}
    logits = generate_empty_logits(labels_json, partition)
    logits = ensemble_predictions(ans, logits)

    rel_frames = {}

    for k in logits.keys():
        errors = (logits[k].argmax(1) != labels_json['labels'][k]) * 1
        w = np.ones(2 * look_around + 1)
        rel_frames[k] = np.convolve(errors, w, 'same') > 0
    
    outp_buffer = []
    frame_size = None
    fps = 30  # default fps

    font = cv2.FONT_HERSHEY_SIMPLEX


    for fn in rel_frames:
        video_path = os.path.join(prefix, fn)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open {video_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (width, height)
        fps = cap.get(cv2.CAP_PROP_FPS)

        font_scale = height / 1000
        font_thickness = max(1, int(height / 500))

        frame_flags = rel_frames[fn]
        frame_flags = frame_flags[:total_frames]

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_flags[i]:
                pred = logits[fn][i].argmax()
                true = labels_json['labels'][fn][i]
                truncated_fn = fn if len(fn) <= 48 else fn[:45] + "..."
                text_topleft = f"{truncated_fn} {i}"
                text_topright = f"Label: {class_names[true]}  Pred: {class_names[pred]}"

                # Draw top-left
                cv2.putText(frame, text_topleft, (10, 20), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

                # Draw true/pred top-right
                (text_width, _), _ = cv2.getTextSize(text_topright, font, font_scale, font_thickness)
                x_right = width - text_width - 10
                y = 40
                cv2.putText(frame, text_topright, (x_right, y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                
                # Draw logits with class names underneath
                y += 20  # move down from the "True/Pred" line
                for idx, logit_val in enumerate(logits[fn][i]):
                    class_name = class_names[idx]
                    logit_line = f"{class_name}: {logit_val:.2f}"
                    (text_width, _), _ = cv2.getTextSize(logit_line, font, font_scale, font_thickness)
                    x_right = width - text_width - 10
                    cv2.putText(frame, logit_line, (x_right, y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                    y += 15  # adjust spacing between lines

                outp_buffer.append(frame)


        cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    for frame in outp_buffer:
        out.write(frame)
    out.release()


def save_inference_results(ans, ema_ans, video_prefix, labels_json, save_fn):
    out = {}
    
    ans_logits = {} 
    for fn in labels_json['splits']['inference']:
        ans_logits[fn] = np.zeros((get_frame_count(os.path.join(video_prefix, fn)), len(labels_json['class_names'])))

    out['preds'] = ensemble_predictions(ans, ans_logits)
    out['preds'] = {k: v.tolist() for k, v in out['preds'].items()}
    
    if len(ema_ans) > 0:
        ema_logits = {}
        for fn in labels_json['splits']['inference']:
            ema_logits[fn] = np.zeros((get_frame_count(os.path.join(video_prefix, fn)), len(labels_json['class_names'])))
        out['ema_preds'] = ensemble_predictions(ema_ans, ema_logits)
        out['ema_preds'] = {k: v.tolist() for k, v in out['ema_preds'].items()}
    with open(save_fn, 'w') as f:
        json.dump(out, f)