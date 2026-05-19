import logging

from torch.utils.data import DataLoader

from feral.dataset import ClsDataset, collate_fn_val, collate_fn_inference
from feral.utils import is_classification

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
    num_workers = cfg['training']['num_workers']
    persistent_workers = num_workers > 0

    is_cls = is_classification(labels_json)
    task = 'classification' if is_cls else 'regression'
    num_targets = None if is_cls else len(labels_json['target_names'])

    for partition, spec in _PARTITION_SPECS.items():
        split = labels_json['splits'].get(partition)
        if not split:
            logger.info("No %s dataset", partition)
            continue

        dataset = ClsDataset(
            partition=partition,
            label_json_dict=labels_json,
            num_classes=num_classes,
            predict_per_item=cfg['predict_per_item'],
            task=task,
            num_targets=num_targets,
            **cfg['data'],
        )

        if partition != 'inference' and is_cls:
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

    if is_cls:
        logger.info("Dataset is multilabel: %s", labels_json['is_multilabel'])
    else:
        logger.info("Dataset is regression with %d target(s)", num_targets)
    return datasets, loaders
