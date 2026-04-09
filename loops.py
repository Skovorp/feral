import torch
from tqdm import tqdm

from utils import prep_for_answers


def _to_prob(output, is_multilabel):
    return torch.sigmoid(output) if is_multilabel else torch.nn.functional.softmax(output, 1)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, *,
                    mixup, model_ema, num_classes, is_multilabel,
                    predict_per_item, device, log_fn=None, max_batches=None):
    """Run one training epoch. Returns (answers, avg_loss).

    log_fn: optional callable invoked per batch with a dict of scalars
            (e.g. wandb.log). No-op if None.
    max_batches: if set, stop after this many batches.
    """
    model.train()
    answers = []
    losses = []

    for i, (data, target) in enumerate(tqdm(loader, total=len(loader))):
        data = data.to(device)
        target = target.to(device)

        # Eye matrix for mixup; sized to actual batch (multi-GPU safe).
        batch_size = data.shape[0]
        eye = torch.eye(batch_size, device=device)

        optimizer.zero_grad()

        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
            if mixup is not None:
                N, T, C, A, B = data.shape
                data = data.reshape(N, T, C, A * B)
                data, mix = mixup(data, eye)
                data = data.reshape(N, T, C, A, B)
                if predict_per_item != 1:
                    target = target.permute(1, 0, 2)
                    target = mix.unsqueeze(0) @ target
                    target = target.permute(1, 0, 2)
                else:
                    target = mix @ target
            target = target.reshape(-1, num_classes)
            output = model(data)
            output_prob = _to_prob(output, is_multilabel)
            loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        scheduler.step()
        if model_ema is not None:
            model_ema.update(model)

        if log_fn is not None:
            log_fn({'batch_loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})

        answers.extend(prep_for_answers(output_prob, target))
        losses.append(loss.item())

        if max_batches is not None and i + 1 >= max_batches:
            break

    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return answers, avg_loss


def evaluate(model, loader, criterion=None, *, num_classes, is_multilabel,
             device, max_batches=None):
    """Run model over a labeled loader (val/test). Returns (answers, avg_loss).

    avg_loss is None if criterion is None.
    """
    model.eval()
    answers = []
    losses = []

    with torch.no_grad():
        for i, (data, target, names) in enumerate(tqdm(loader, total=len(loader))):
            data = data.to(device)
            target = target.to(device)
            target = target.view(-1, num_classes)
            with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                output = model(data)
                output_prob = _to_prob(output, is_multilabel)
                answers.extend(prep_for_answers(output_prob, target, names))
                if criterion is not None:
                    losses.append(criterion(output, target).item())

            if max_batches is not None and i + 1 >= max_batches:
                break

    avg_loss = (sum(losses) / len(losses)) if losses else None
    return answers, avg_loss


def run_inference(model, loader, *, is_multilabel, device, max_batches=None):
    """Run model over an unlabeled loader. Returns answers."""
    model.eval()
    answers = []

    with torch.no_grad():
        for i, (data, names) in enumerate(tqdm(loader, total=len(loader))):
            data = data.to(device)
            with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                output = model(data)
                output_prob = _to_prob(output, is_multilabel)
                answers.extend(prep_for_answers(output_prob, None, names))

            if max_batches is not None and i + 1 >= max_batches:
                break

    return answers
