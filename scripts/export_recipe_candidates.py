"""Export a bounded, read-only W&B sample for recipe review; never rank unlike datasets."""
import argparse
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

# Deliberately exclude free-text notes, user paths, credentials and test metrics.
NUMERIC_FIELDS = {
    'seed', 'predict_per_item', 'ema_decay', 'mixup_alpha', 'multilabel_threshold',
    'max_batches', 'max_train_batches', 'training.patience',
    'model.fc_drop_rate', 'model.freeze_encoder_layers', 'model.max_class_weight',
    'model.gradient_checkpointing', 'data.resize_to', 'data.chunk_length',
    'data.chunk_shift', 'data.chunk_step', 'data.eval_chunk_shift',
    'data.eval_smoothing_window', 'data.do_aa', 'data.part_sample',
    'data.subsample_keep_rare_threshold', 'training.epochs', 'training.train_bs',
    'training.val_bs', 'training.lr', 'training.weight_decay', 'training.label_smoothing',
    'training.compile', 'training.grad_clip_norm', 'training.part_warmup',
}
STRING_FIELDS = {
    'backbone': {'vjepa2_vitl_diving48', 'vjepa2_vitl_ssv2', 'vjepa2_vitl',
                 'vjepa2_1_vitb_384', 'vjepa2_1_vitl_384', 'vjepa2_1_vitg_384',
                 'vjepa2_1_vitgg_384', 'videoprism_v1_base', 'videoprism_v1_large'},
    'data.resize_style': {'square', 'rectangle'},
    'model.class_weights': {'inv_freq', 'inv_freq_sqrt'},
}
METRICS = ('val/frame_level_map', 'val/f1', 'val/precision', 'val/recall', '_runtime')


def _get(mapping: dict, dotted: str):
    value = mapping
    for key in dotted.split('.'):
        if not isinstance(value, dict) or key not in value:
            return False, None
        value = value[key]
    return True, value


def candidate(run) -> dict:
    """Return only allowlisted recipe evidence; absent provenance stays unknown."""
    config = run.config
    recipe = {}
    for key in sorted(NUMERIC_FIELDS | STRING_FIELDS.keys()):
        found, value = _get(config, key)
        if not found:
            continue
        if key in NUMERIC_FIELDS:
            valid = value is None or isinstance(value, (bool, int, float))
            valid = valid and (not isinstance(value, float) or math.isfinite(value))
        else:
            valid = value is None or isinstance(value, str) and value in STRING_FIELDS[key]
        if valid:
            recipe[key] = value
    summary = dict(run.summary)
    metrics = {k: summary[k] for k in METRICS
               if isinstance(summary.get(k), (int, float)) and not isinstance(summary[k], bool)
               and math.isfinite(summary[k])}
    missing = []
    for key in ('backbone', 'data.resize_to', 'data.chunk_length', 'data.chunk_shift',
                'data.chunk_step', 'training.lr', 'training.epochs', 'seed'):
        if recipe.get(key) is None:
            missing.append(key)
    if 'val/frame_level_map' not in metrics:
        missing.append('val/frame_level_map')
    # Old runs do not establish dataset/split equivalence from a path name alone.
    provenance = {}
    for key in ('labels_sha256', 'splits_sha256'):
        val = config.get(key)
        if isinstance(val, str) and len(val) == 64 and all(c in '0123456789abcdef' for c in val):
            provenance[key] = val
        else:
            missing.append(key)
    if run.state != 'finished':
        missing.append('finished_run')
    if recipe.get('max_batches') is not None or recipe.get('max_train_batches') is not None:
        missing.append('untruncated_run')
    initialization = None
    if 'starting_checkpoint' in config:
        initialization = 'checkpoint' if config['starting_checkpoint'] else 'pretrained_backbone'
    if initialization is None:
        missing.append('initialization')
    elif initialization == 'checkpoint':
        missing.append('starting_checkpoint_identity')
    return {'run_url': run.url, 'state': run.state, 'recipe': recipe,
            'initialization': initialization,
            'validation_summary': metrics, 'provenance': provenance,
            'status': 'needs_review' if not missing else 'insufficient_evidence',
            'missing': missing}


def main() -> None:
    """Fetch complete run records without updating runs or downloading video artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default='sposiboh/feral_public', help='ENTITY/PROJECT')
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 1000:
        parser.error('--limit must be between 1 and 1000')
    import wandb
    api = wandb.Api(timeout=30)
    rows = []
    for listed in itertools.islice(api.runs(args.project, order='-created_at', per_page=min(50, args.limit)), args.limit):
        # List responses can omit config; explicit fetch resolves the complete record.
        run = api.run(f'{args.project}/{listed.id}')
        rows.append(candidate(run))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        'schema_version': 1, 'fetched_at': datetime.now(timezone.utc).isoformat(),
        'project': args.project, 'runs': rows,
        'note': 'Summary metrics may describe the last epoch, not the best checkpoint. '
                'Do not rank runs across datasets or splits; verify provenance and metric definitions first. '
                '_runtime is elapsed run time, not normalized GPU throughput.',
    }, indent=2, allow_nan=False) + '\n')
    print(f'Exported {len(rows)} candidate records to {args.output}')


if __name__ == '__main__':
    main()
