import numpy as np
import json 
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, accuracy_score, precision_recall_curve
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import PIL
import cv2
import os
import traceback
import logging

from feral.utils import last_nonzero_index, next_nonzero_index
from feral.dataset import get_frame_count

logger = logging.getLogger(__name__)

def calc_frame_level_map(ans, labels_json, partition):
    """Ensemble per-frame predictions and return mean average precision over non-'other' classes."""
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

def calculate_f1_metrics(ans, labels_json, partition, is_multilabel, prefix, multilabel_threshold):
    """Compute precision/recall/F1/accuracy (plus per-class F1) for non-'other' classes.

    Single-label uses argmax with macro-averaging; multilabel thresholds logits at
    multilabel_threshold per class and averages. Returns a dict keyed by '{prefix}/...'.
    """
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

        per_class_f1 = f1_score(target_labels, pred_labels, labels=valid_classes, average=None)
        res = {}
        for cls_ind, f1_val in zip(valid_classes, per_class_f1):
            res[f'{prefix}/f1_{class_names[cls_ind]}'] = f1_val

        res[f'{prefix}/precision'] = precision_score(target_labels, pred_labels, labels=valid_classes, average='macro')
        res[f'{prefix}/recall'] = recall_score(target_labels, pred_labels, labels=valid_classes, average='macro')
        res[f'{prefix}/f1'] = f1_score(target_labels, pred_labels, labels=valid_classes, average='macro')
        res[f'{prefix}/accuracy'] = accuracy_score(target_labels, pred_labels)
        return res
    else:
        pred_labels = (preds > multilabel_threshold) * 1
        target_labels = targets.astype(int)

        precisions = []
        recalls = []
        f1s = []

        res = {}
        for valid_class_ind in valid_classes:
            p = precision_score(target_labels[:, valid_class_ind], pred_labels[:, valid_class_ind])
            r = recall_score(target_labels[:, valid_class_ind], pred_labels[:, valid_class_ind])
            f = f1_score(target_labels[:, valid_class_ind], pred_labels[:, valid_class_ind])
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
            res[f'{prefix}/f1_{class_names[valid_class_ind]}'] = f

        res[f'{prefix}/precision'] = sum(precisions) / len(precisions)
        res[f'{prefix}/recall'] = sum(recalls) / len(recalls)
        res[f'{prefix}/f1'] = sum(f1s) / len(f1s)
        # exact match: all class predictions must match all class targets
        res[f'{prefix}/accuracy'] = accuracy_score(target_labels, pred_labels)
        return res

def _per_class_optimal_picks(ans, labels_json, partition):
    """
    Multilabel one-vs-rest sweep with precision_recall_curve. Returns
    dict[class_ind] = (best_f1, best_threshold) for non-'other' classes with at
    least one positive. Classes outside the dict are excluded (no positives).
    """
    class_names = {int(k): v for k, v in labels_json['class_names'].items()}
    valid_classes = [i for i, n in class_names.items() if n != 'other']

    logits = generate_empty_logits(labels_json, partition)
    logits = ensemble_predictions(ans, logits)

    preds = np.concatenate([logits[fn] for fn in labels_json['splits'][partition]], 0)
    targets = np.concatenate([labels_json['labels'][fn] for fn in labels_json['splits'][partition]], 0)

    out = {}
    for c in valid_classes:
        y_true = targets[:, c].astype(int)
        if y_true.sum() == 0:
            continue
        y_score = preds[:, c]
        p, r, thr = precision_recall_curve(y_true, y_score)
        f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
        best = int(np.argmax(f1))
        if not len(thr):
            continue
        out[c] = (float(f1[best]), float(thr[min(best, len(thr) - 1)]))
    return out


def compute_optimal_per_class_thresholds(ans, labels_json, partition):
    """Per-class optimal-F1 thresholds keyed by class_ind. Empty if no valid classes."""
    return {c: t for c, (_f, t) in _per_class_optimal_picks(ans, labels_json, partition).items()}


def calculate_optimal_f1_metrics(ans, labels_json, partition, is_multilabel, prefix):
    """Per-class optimal-threshold best-F1 metrics (multilabel only).

    Returns a dict with '{prefix}/best_f1_{name}' and '{prefix}/best_thr_{name}' per
    class (nan for classes with no positives) and '{prefix}/best_f1' as their mean.
    Returns {} for single-label.
    """
    # Only defined for multilabel — per-class thresholds aren't simultaneously
    # applicable under single-label argmax, so the metric would be misleading.
    if not is_multilabel:
        return {}

    class_names = {int(k): v for k, v in labels_json['class_names'].items()}
    valid_classes = [i for i, n in class_names.items() if n != 'other']
    picks = _per_class_optimal_picks(ans, labels_json, partition)

    res = {}
    f1s = []
    for c in valid_classes:
        name = class_names[c]
        if c in picks:
            f1, thr = picks[c]
            res[f'{prefix}/best_f1_{name}'] = f1
            res[f'{prefix}/best_thr_{name}'] = thr
            f1s.append(f1)
        else:
            res[f'{prefix}/best_f1_{name}'] = float('nan')
            res[f'{prefix}/best_thr_{name}'] = float('nan')
    res[f'{prefix}/best_f1'] = (sum(f1s) / len(f1s)) if f1s else float('nan')
    return res


def ensemble_predictions(ans, logits):
    """Accumulate per-chunk predictions into per-frame logits (uniform weights), averaging
    overlapping predictions and filling gaps by inverse-distance interpolation from the
    nearest predicted frames on either side. Mutates and returns `logits`.
    """
    predict_per_item = max(x[0][2] for x in ans) + 1
    sum_weights = {fn: np.zeros(val.shape[0]) for (fn, val) in logits.items()}

    # uniform weights for now
    weights = np.ones(predict_per_item)[:, None]

    for el in ans:
        fn, global_ind, chunk_ind = el[0]
        preds = np.array(el[1])

        logits[fn][global_ind, :] += preds * weights[chunk_ind, 0]
        sum_weights[fn][global_ind] += weights[chunk_ind, 0]
    
    for fn in logits.keys():
        left_ind = last_nonzero_index(sum_weights[fn])
        right_ind = next_nonzero_index(sum_weights[fn])

        for i in range(sum_weights[fn].shape[0]):
            if sum_weights[fn][i] > 0:
                logits[fn][i, :] = logits[fn][i, :] / sum_weights[fn][i]
            else:
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
    """Return dict[filename -> zeros array of shape (num_frames, num_classes)] for the partition."""
    logits = {}
    for k in labels_json['splits'][partition]:
        logits[k] = np.zeros((len(labels_json['labels'][k]), len(labels_json['class_names'])))
    return logits

def generate_raster_plot(ans, labels_json, partition):
    """Build a single-label ethogram raster (prediction, label, mismatch rows) across all
    videos in the partition, with per-video separators and a class legend. Returns a PIL
    image; on failure logs the traceback and returns an 'Error' image instead.
    """
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


def generate_multilabel_raster_plot(ans, labels_json, partition, thresholds):
    """
    Multilabel ethogram: two stacked rasters (prediction on top, ground truth on
    bottom), one row per class, cell colored when class is active. Black vlines
    separate videos with their names written below. Always returns a PIL image —
    on failure, an 'Error' image (logs traceback) so training never dies here.

    thresholds: float OR dict[class_ind -> float]. Missing classes default to 0.5.
    """
    try:
        class_names = {int(k): v for k, v in labels_json['class_names'].items()}
        n_classes = len(class_names)

        if isinstance(thresholds, dict):
            thr_arr = np.array(
                [thresholds.get(i, 0.5) for i in range(n_classes)], dtype=float
            )
        else:
            thr_arr = np.full(n_classes, float(thresholds))

        logits = generate_empty_logits(labels_json, partition)
        logits = ensemble_predictions(ans, logits)

        video_names = list(logits.keys())
        preds_list, targets_list = [], []
        split_positions = []
        cursor = 0
        for name in video_names:
            pred = (logits[name] > thr_arr[None, :]).astype(int)
            label = np.array(labels_json['labels'][name]).astype(int)
            assert label.ndim == 2 and label.shape[1] == n_classes, (
                f"Multilabel raster expects labels of shape (T, {n_classes}); got {label.shape} for {name}"
            )
            preds_list.append(pred)
            targets_list.append(label)
            cursor += pred.shape[0]
            split_positions.append(cursor)
        split_positions = split_positions[:-1]

        arr_pred = np.concatenate(preds_list, 0).T   # (C, T)
        arr_label = np.concatenate(targets_list, 0).T  # (C, T)

        base_cmap = cm.get_cmap('nipy_spectral')
        color_range = np.linspace(0.1, 1.0, n_classes)
        colors = [base_cmap(v) for v in color_range]
        if 'other' in class_names.values():
            idx = list(class_names.values()).index('other')
            colors[idx], colors[-1] = colors[-1], colors[idx]

        H, W = arr_pred.shape
        rgb_pred = np.ones((H, W, 3), dtype=float)
        rgb_label = np.ones((H, W, 3), dtype=float)
        for c in range(n_classes):
            color = np.array(colors[c][:3])
            rgb_pred[c, arr_pred[c] == 1] = color
            rgb_label[c, arr_label[c] == 1] = color

        per_class_h = 0.6
        fig_h = max(8.0, 2 * (n_classes * per_class_h + 1.0) + 4.0)
        fig, axes = plt.subplots(2, 1, figsize=(32, fig_h), sharex=True)

        for ax, rgb, title in zip(axes, [rgb_pred, rgb_label], ['prediction', 'ground truth']):
            ax.imshow(rgb, aspect='auto', interpolation='nearest')
            ax.set_yticks(range(n_classes))
            ax.set_yticklabels([class_names[i] for i in range(n_classes)], fontsize=12)
            ax.set_title(title, fontsize=16, pad=12)
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            for pos in split_positions:
                ax.axvline(pos, color='black', linewidth=2)

        # video names well below the bottom raster
        bottom_ax = axes[-1]
        cursor = 0
        for name in video_names:
            length = logits[name].shape[0]
            center = cursor + length // 2
            wrapped = '\n'.join([name[i:i+15] for i in range(0, len(name), 15)])
            bottom_ax.text(
                center, n_classes + 0.5, wrapped,
                ha='center', va='top', fontsize=11, clip_on=False,
            )
            cursor += length

        plt.subplots_adjust(left=0.12, right=0.99, bottom=0.28, top=0.92, hspace=0.6)

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
    """Compute per-class average precision and mAP (over non-'other' classes) from a list of
    (..., preds, targets) items. Returns a dict keyed by '{prefix}/ap_{name}' and '{prefix}/map'.
    """
    preds = np.array([x[-2] for x in ans])
    targets = np.array([x[-1] for x in ans]).astype(int)

    aps = []
    res = {}
    for cls_ind, cls_name in class_names.items():
        ap = average_precision_score(targets[:, cls_ind], preds[:, cls_ind])
        if cls_name != 'other':
            aps.append(ap)
        res[f'{prefix}/ap_{cls_name}'] = ap
    res[f'{prefix}/map'] = sum(aps) / len(aps)
    return res


def generate_video_mismatches(ans, labels_json, partition, prefix, font_color=(255, 255, 255), look_around=30, output_path='result.mp4'):
    """Write an mp4 to output_path containing only frames near prediction/label mismatches
    (within look_around frames), overlaying filename, frame index, true/pred class, and
    per-class logits on each frame. Reads videos from `prefix`; skips unreadable files.
    """
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
            logger.warning("Could not open %s", video_path)
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
    """Ensemble inference predictions (and EMA predictions if present) into per-frame logits
    and write them as JSON to save_fn under keys 'preds' and optionally 'ema_preds', each a
    dict[filename -> list-of-lists]. Frame counts are read from the videos under video_prefix.
    """
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