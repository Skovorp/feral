"""Read-only diagnostics and compact, machine-readable dataset summaries."""
import hashlib
import importlib
import json
import platform
from pathlib import Path


def doctor() -> dict:
    """Check local imports and CUDA without downloading models or contacting services."""
    from feral import __version__
    report = {'schema_version': 1, 'feral_version': __version__,
              'python': platform.python_version(), 'platform': platform.platform(),
              'dependencies': {}, 'gpus': [], 'errors': []}
    for name in ('torch', 'torchvision', 'decord', 'transformers', 'timm', 'cv2', 'wandb', 'yaml'):
        try:
            module = importlib.import_module(name)
            report['dependencies'][name] = getattr(module, '__version__', 'available')
            if name == 'torch':
                report['cuda_version'] = module.version.cuda
                for index in range(module.cuda.device_count()):
                    prop = module.cuda.get_device_properties(index)
                    report['gpus'].append({'index': index, 'name': prop.name,
                                           'memory_bytes': prop.total_memory,
                                           'compute_capability': [prop.major, prop.minor]})
                report['cuda_available'] = module.cuda.is_available()
                if report['cuda_available']:
                    current = module.cuda.current_device()
                    report['current_device'] = current
                    capability = module.cuda.get_device_capability(current)
                    report['bf16_supported'] = capability[0] >= 8 and module.cuda.is_bf16_supported()
                    if not report['bf16_supported']:
                        report['errors'].append('Current CUDA GPU requires compute capability >= 8.0 and native BF16 support. Use an Ampere-or-newer GPU (for example RTX 30-series/A100); T4 and V100 are unsupported. Select a supported GPU with CUDA_VISIBLE_DEVICES.')
        except Exception as exc:
            report['errors'].append(f'{name}: {type(exc).__name__}: {exc}')
    if not report.get('cuda_available'):
        report['errors'].append('CUDA is unavailable; training and inference require an NVIDIA GPU.')
    report['ok'] = not report['errors']
    return report


def validate_dataset(video_folder: str, label_path: str) -> dict:
    """Validate labels and video frame counts; summarize without returning frame arrays."""
    result = {'schema_version': 1, 'ok': False, 'errors': []}
    try:
        from feral.utils import validate_labels_json
        if not Path(video_folder).is_dir():
            raise ValueError(f'Video folder is not a directory: {video_folder}')
        raw = Path(label_path).read_bytes()
        labels = json.loads(raw)
        validate_labels_json(labels, video_folder)
        classes = labels['class_names']
        result.update(labels_sha256=hashlib.sha256(raw).hexdigest(),
                      class_names=classes, is_multilabel=labels['is_multilabel'])
        result['splits'] = {}
        for split, videos in sorted(labels['splits'].items()):
            counts = {key: 0 for key in sorted(classes, key=int)}
            n_frames = 0
            for video in videos:
                frames = labels['labels'].get(video, [])
                n_frames += len(frames)
                for frame in frames:
                    if labels['is_multilabel']:
                        for key in counts:
                            counts[key] += int(frame[int(key)] == 1)
                    else:
                        counts[str(frame)] += 1
            result['splits'][split] = {'videos': len(videos), 'labeled_frames': n_frames,
                                       'class_positive_frames': counts}
        result['ok'] = True
    except Exception as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    return result
