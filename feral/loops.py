import logging
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from feral.utils import prep_for_answers

logger = logging.getLogger(__name__)


def _to_prob(output, is_multilabel):
    return torch.sigmoid(output) if is_multilabel else torch.nn.functional.softmax(output, 1)


def _per_element_loss(output, target, is_multilabel):
    """Unreduced per-sample loss (for the heavy 'loss distribution' histogram).

    Mirrors the training criterion but with reduction='none'; weights and label
    smoothing are intentionally dropped since this is a diagnostic.
    """
    if is_multilabel:
        return F.binary_cross_entropy_with_logits(output, target, reduction='none').mean(dim=-1)
    # single-label: target may be soft (mixup / one-hot) -> cross-entropy with soft targets
    return -(target * F.log_softmax(output.float(), dim=-1)).sum(dim=-1)


@torch.no_grad()
def _heavy_logs(model, output, target, is_multilabel):
    """Expensive, throttled diagnostics (per-layer grad/weight norm distributions,
    global weight norm, per-element loss distribution).

    Cost is bounded by the number of parameter *tensors* (~hundreds), not
    elements: norms are reduced on-device and pulled to host in a single
    transfer. Guarded so a logging failure never kills training.
    """
    import wandb
    logs = {}
    try:
        params = [p for p in model.parameters() if p.requires_grad]
        if params:
            w = torch.stack([p.detach().norm() for p in params])
            logs['heavy/weight_norm_per_layer'] = wandb.Histogram(w.float().cpu().numpy())
            logs['heavy/weight_norm_global'] = w.norm().item()
            grads = [p.grad.detach().norm() for p in params if p.grad is not None]
            if grads:
                g = torch.stack(grads)
                logs['heavy/grad_norm_per_layer'] = wandb.Histogram(g.float().cpu().numpy())
        per_elem = _per_element_loss(output, target, is_multilabel)
        logs['heavy/loss_hist'] = wandb.Histogram(per_elem.detach().float().cpu().numpy())
    except Exception:
        logger.exception("heavy logging failed")
    return logs


def train_one_epoch(model, loader, criterion, optimizer, scheduler, *,
                    mixup, model_ema, num_classes, is_multilabel,
                    predict_per_item, device, log_fn=None, max_batches=None,
                    grad_clip_norm=None, log_grad_norm=True, heavy_log_every=None):
    """Run one training epoch. Returns (answers, avg_loss).

    log_fn: optional callable invoked per batch with a dict of scalars
            (e.g. wandb.log). No-op if None.
    max_batches: if set, stop after this many batches.
    grad_clip_norm: if set, clip the global grad-norm to this value before the
            optimizer step (off by default; useful for rare-class stability).
    log_grad_norm: measure & log the global grad-norm every step even when not
            clipping (cheap; one reduction + one sync). On by default.
    heavy_log_every: if set, every N steps also log expensive diagnostics
            (per-layer grad/weight norm histograms, per-element loss). Off by
            default to keep the hot path fast.
    """
    model.train()
    answers = []
    losses = []
    t_end = time.perf_counter()

    for i, (data, target) in enumerate(tqdm(loader, total=len(loader))):
        t_data = time.perf_counter() - t_end  # dataloader wait (input-bound probe)
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

        # Grad norm: when clipping, use the clip's returned pre-clip norm;
        # otherwise (default) still measure it with an inf threshold (no clip).
        grad_norm = None
        if grad_clip_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        elif log_grad_norm:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))

        # Heavy diagnostics computed before optimizer.step() (grads still live).
        heavy = {}
        if heavy_log_every is not None and (i % heavy_log_every == 0):
            heavy = _heavy_logs(model, output, target, is_multilabel)

        optimizer.step()
        scheduler.step()
        if model_ema is not None:
            model_ema.update(model)

        loss_val = loss.item()  # single GPU->CPU sync, reused below
        if log_fn is not None:
            t_step = time.perf_counter() - t_end
            batch_logs = {
                'train/batch_loss': loss_val,
                'train/lr': scheduler.get_last_lr()[0],
                'perf/samples_per_sec': batch_size / t_step if t_step > 0 else 0.0,
                'perf/step_time': t_step,
                'perf/data_time': t_data,
            }
            if grad_norm is not None:
                batch_logs['train/grad_norm'] = grad_norm.item()
            batch_logs.update(heavy)
            log_fn(batch_logs)

        answers.extend(prep_for_answers(output_prob, target))
        losses.append(loss_val)

        if max_batches is not None and i + 1 >= max_batches:
            break
        t_end = time.perf_counter()

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
