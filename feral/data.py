import logging

from torch.utils.data import DataLoader

from feral.dataset import ClsDataset, collate_fn_val, collate_fn_inference
from feral.utils import resolve_num_workers

logger = logging.getLogger(__name__)


_PARTITION_SPECS = {
    'train':     {'shuffle': True,  'drop_last': True,  'collate_fn': None},
    'val':       {'shuffle': False, 'drop_last': False, 'collate_fn': collate_fn_val},
    'test':      {'shuffle': False, 'drop_last': False, 'collate_fn': collate_fn_val},
    'inference': {'shuffle': False, 'drop_last': False, 'collate_fn': collate_fn_inference},
}


def build_datasets_and_loaders(cfg, labels_json, num_classes):
    """Build datasets and dataloaders for every partition present in labels_json.

    Returns (datasets, loaders): two dicts keyed by partition name. Partitions
    that are missing or empty in labels_json are simply absent from the dicts.
    """
    datasets = {}
    loaders = {}

    train_bs = cfg['training']['train_bs']
    val_bs = cfg['training']['val_bs']
    num_workers = resolve_num_workers(cfg['training']['num_workers'])
    logger.info("DataLoader num_workers=%d (config: %r)", num_workers, cfg['training']['num_workers'])
    persistent_workers = num_workers > 0

    # eval_chunk_shift / eval_smoothing_window are eval-only knobs, not ClsDataset
    # args. Pull them out of the data kwargs; eval_chunk_shift overrides chunk_shift
    # for val/test/inference so eval/inference can run a denser overlap than training
    # (eval_smoothing_window is applied later, at ensembling time, not here).
    data_kwargs = dict(cfg['data'])
    eval_chunk_shift = data_kwargs.pop('eval_chunk_shift', None)
    data_kwargs.pop('eval_smoothing_window', None)

    for partition, spec in _PARTITION_SPECS.items():
        split = labels_json['splits'].get(partition)
        if not split:
            logger.info("No %s dataset", partition)
            continue

        part_kwargs = dict(data_kwargs)
        if partition != 'train' and eval_chunk_shift is not None:
            logger.info("%s: eval_chunk_shift overrides chunk_shift %s -> %s",
                        partition, part_kwargs.get('chunk_shift'), eval_chunk_shift)
            part_kwargs['chunk_shift'] = eval_chunk_shift

        dataset = ClsDataset(
            partition=partition,
            label_json_dict=labels_json,
            num_classes=num_classes,
            predict_per_item=cfg['predict_per_item'],
            **part_kwargs,
        )

        if partition != 'inference':
            assert labels_json['is_multilabel'] == dataset.is_multilabel, (
                f"Config is_multilabel doesn't match the data! "
                f"config: {labels_json['is_multilabel']} dataset: {dataset.is_multilabel}"
            )

        loader = DataLoader(
            dataset,
            batch_size=train_bs if partition == 'train' else val_bs,
            shuffle=spec['shuffle'],
            drop_last=spec['drop_last'],
            collate_fn=spec['collate_fn'],
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=True,
        )

        datasets[partition] = dataset
        loaders[partition] = loader

    logger.info("Dataset is multilabel: %s", labels_json['is_multilabel'])
    return datasets, loaders
