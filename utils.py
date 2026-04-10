import torch
import logging

logger = logging.getLogger(__name__)
import numpy as np
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
    
def get_class_frequencies(labels_arr, num_classes=None):
    """Calculate class frequencies from a labels array.
    
    Args:
        labels_arr: numpy array of shape (N,) for single-label or (N, num_classes) for multi-label
        num_classes: number of classes (required for single-label to ensure all classes are counted)
    
    Returns:
        numpy array of class frequencies
    """
    if len(labels_arr.shape) == 1:
        # Single-label: count occurrences
        if num_classes is None:
            num_classes = int(labels_arr.max()) + 1
        freqs = np.bincount(labels_arr.flatten().astype(int), minlength=num_classes) / labels_arr.size
    else:
        # Multi-label: mean across samples
        freqs = labels_arr.mean(axis=0)
    return freqs

def get_weights(json_data, weight_type, device):
    assert weight_type is None or weight_type in ('inv_freq', 'inv_freq_sqrt'), "weight_type should be 'inv_freq', 'inv_freq_sqrt' or None"
    if weight_type is None:
        return None 
    arr = np.concatenate([json_data['labels'][x] for x in json_data['splits']['train']])
    freqs = torch.tensor(get_class_frequencies(arr))
    
    if len(arr.shape) == 1:
        ratio = (1.0 / freqs).to(device)
    elif len(arr.shape) == 2:
        ratio = ((1 - freqs) / freqs).to(device)    
    if freqs.min().item() <= 0: 
        logger.warning("Some classes don't have any examples. Class frequencies: %s", freqs)
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

def check_environment(compile_enabled):
    """Check that the current hardware and software meet FERAL's requirements.
    Call once at startup before any training work begins."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. FERAL requires an NVIDIA GPU with CUDA support."
        )

    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    gpu_name = torch.cuda.get_device_name(device)

    # bfloat16 requires compute capability >= 8.0 (Ampere+)
    if capability < (8, 0):
        raise RuntimeError(
            f"GPU '{gpu_name}' has compute capability {capability[0]}.{capability[1]}, "
            f"but FERAL requires >= 8.0 (Ampere or newer) for bfloat16 support."
        )

    # flash attention requires compute capability >= 8.0 and PyTorch >= 2.0
    if not hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        raise RuntimeError(
            "PyTorch scaled_dot_product_attention is not available. "
            "FERAL requires PyTorch >= 2.0 for flash attention support."
        )
    # FERAL disables math and mem-efficient SDP backends, so flash attention must work.
    # The flash SDP backend requires SM 80+ which we already checked above,
    # but verify it isn't explicitly disabled.
    if hasattr(torch.backends.cuda, 'flash_sdp_enabled') and not torch.backends.cuda.flash_sdp_enabled():
        raise RuntimeError(
            "Flash attention SDP backend is disabled. "
            "FERAL requires flash attention (math and mem-efficient SDP are turned off)."
        )

    if compile_enabled:
        try:
            import triton  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "torch.compile is enabled but triton is not installed. "
                "Run: pip install -r requirements.txt"
            )

    logger.info(
        "Environment OK: GPU='%s', compute capability=%s.%s, bfloat16=yes, flash_attn=yes, compile=%s",
        gpu_name, capability[0], capability[1], "yes" if compile_enabled else "off",
    )


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

def save_model(model, path, metadata):
    m = model
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    torch.save({
        'state_dict': m.state_dict(),
        **metadata,
    }, path)


def pick_and_save_best(model, model_ema, val_map, ema_map, best_map, path, metadata):
    """Decide whether to save the base model, the EMA model, or neither.

    Returns (new_best_map, saved). `saved` is one of 'base', 'ema', None.
    The caller is responsible for tracking patience / early stopping based on
    whether `saved` is None.
    """
    if val_map > ema_map and val_map > best_map:
        save_model(model, path, metadata)
        return val_map, 'base'
    if model_ema is not None and ema_map > val_map and ema_map > best_map:
        save_model(model_ema.ema, path, metadata)
        return ema_map, 'ema'
    return best_map, None

def validate_labels_json(labels_json, video_folder):
    """Validate the label JSON up-front so users get clear errors instead of
    cryptic crashes deep in training."""
    errors = []

    # --- top-level keys ---
    required_keys = {'class_names', 'is_multilabel', 'labels', 'splits'}
    missing = required_keys - set(labels_json.keys())
    if missing:
        raise ValueError(f"Label JSON missing required top-level keys: {missing}")

    # --- is_multilabel ---
    if not isinstance(labels_json['is_multilabel'], bool):
        errors.append(f"'is_multilabel' must be a boolean, got {type(labels_json['is_multilabel']).__name__}")

    # --- class_names ---
    class_names_raw = labels_json['class_names']
    if not isinstance(class_names_raw, dict) or len(class_names_raw) == 0:
        errors.append("'class_names' must be a non-empty dict")
    else:
        try:
            ids = sorted(int(k) for k in class_names_raw.keys())
        except (ValueError, TypeError):
            errors.append(f"'class_names' keys must be integer strings, got: {list(class_names_raw.keys())}")
            ids = None
        if ids is not None:
            expected = list(range(len(ids)))
            if ids != expected:
                errors.append(
                    f"'class_names' IDs must be sequential starting from 0. "
                    f"Expected {expected}, got {ids}"
                )

    num_classes = len(class_names_raw) if isinstance(class_names_raw, dict) else 0
    is_multilabel = labels_json['is_multilabel']
    if not isinstance(is_multilabel, bool):
        is_multilabel = None  # skip per-label checks below

    # --- labels ---
    labels = labels_json.get('labels', {})
    if not isinstance(labels, dict):
        errors.append("'labels' must be a dict mapping video filenames to frame labels")
    elif num_classes > 0 and is_multilabel is not None:
        for vid, frame_labels in labels.items():
            if not isinstance(frame_labels, list) or len(frame_labels) == 0:
                errors.append(f"Labels for '{vid}' must be a non-empty list")
                continue
            if is_multilabel:
                bad_width = [
                    i for i, fl in enumerate(frame_labels)
                    if not isinstance(fl, list) or len(fl) != num_classes
                ]
                if bad_width:
                    errors.append(
                        f"'{vid}': multilabel frames must each have {num_classes} values. "
                        f"Bad frames (first 5): {bad_width[:5]}"
                    )
            else:
                bad_vals = sorted({v for v in frame_labels if not isinstance(v, int) or v < 0 or v >= num_classes})
                if bad_vals:
                    errors.append(
                        f"'{vid}': single-label IDs must be ints in [0, {num_classes}). "
                        f"Invalid values: {bad_vals}"
                    )

    # --- splits ---
    splits = labels_json.get('splits', {})
    valid_partitions = {'train', 'val', 'test', 'inference'}
    unknown = set(splits.keys()) - valid_partitions
    if unknown:
        errors.append(f"Unknown split names: {unknown}. Allowed: {valid_partitions}")

    for partition, videos in splits.items():
        if not isinstance(videos, list) or len(videos) == 0:
            errors.append(f"Split '{partition}' is empty. Remove the key if you don't need this partition.")
            continue
        for vid in videos:
            if vid not in labels and partition != 'inference':
                errors.append(f"Split '{partition}' references '{vid}' which has no entry in 'labels'")

    # --- frame count vs. video files ---
    if video_folder:
        import cv2
        frame_mismatches = []
        for partition, videos in splits.items():
            for vid in videos:
                vid_path = os.path.join(video_folder, vid)
                if not os.path.isfile(vid_path):
                    errors.append(f"Split '{partition}' references '{vid}' but file not found: {vid_path}")
                    continue
                if vid not in labels:
                    continue
                cap = cv2.VideoCapture(vid_path)
                try:
                    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                finally:
                    cap.release()
                json_frames = len(labels[vid])
                if video_frames != json_frames:
                    frame_mismatches.append(
                        f"  {vid}: video has {video_frames} frames, labels has {json_frames}"
                    )
        if frame_mismatches:
            errors.append(
                "Frame count mismatches between videos and labels:\n" +
                "\n".join(frame_mismatches)
            )

    if errors:
        raise ValueError(
            f"Label JSON validation failed with {len(errors)} error(s):\n" +
            "\n".join(f"  - {e}" for e in errors)
        )


if __name__ == "__main__":
    print(next_nonzero_index([1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0]))
