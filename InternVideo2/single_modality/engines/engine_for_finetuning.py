import os
import time
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from datasets.mixup import Mixup
from timm.utils import accuracy, ModelEma
import utils
from scipy.special import softmax
import random

from utils import synchronize_lists
from datetime import datetime
import json
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

def train_class_batch(model, samples, target, criterion):
    outputs = model(samples)
    loss = criterion(outputs[:, 0], target.float())
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma, alpha, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: logits (before sigmoid), shape (batch_size, ...)
        targets: ground truth labels (0 or 1), shape (batch_size, ...)
        """
        BCE_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        # alphas = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = 2 * (1 - pt) ** self.gamma * BCE_loss # alphas

        return focal_loss.mean()

def calculate_metrics_for_different_conditions(ans):
    fn_filters = {
        'deet': lambda x: 'deet' in x.lower(),
        'picaridin': lambda x: 'picaridin' in x.lower(),
        'ethanol': lambda x: 'ethanol' in x.lower(),
    }

    res = {}
    for partition, func in fn_filters.items():
        pred = [x[1] for x in ans if func(x[0])]
        tgt  = [x[2] for x in ans if func(x[0])]
        try:
            pr_auc = average_precision_score(tgt, pred)
        except Exception:
            pr_auc = None
        
        try:
            precision, recall, thresholds = precision_recall_curve(tgt, pred)
        
            recall_mask = recall[:-1] < 0.7
            idx = np.argmax(recall_mask)
            prec_at_thresh = precision[idx]
            cutoff = thresholds[idx]
        except Exception:
            prec_at_thresh = None 
        
        res[f"{partition}_pr_auc"] = pr_auc
        res[f"{partition}_precision@0.7"] = prec_at_thresh
    return res


def train_one_epoch(
        model: torch.nn.Module, criterion: torch.nn.Module,
        data_loader: Iterable, optimizer: torch.optim.Optimizer,
        device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
        model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
        start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
        num_training_steps_per_epoch=None, update_freq=None,
        bf16=False, loss_pos_weight=None, do_lr_scale=True
    ):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    # criterion = torch.nn.CrossEntropyLoss(
    #     label_smoothing=0.1,
    #     weight=torch.tensor([1.0, 3.0]).to(model.device)
    # )
    if loss_pos_weight is None:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(loss_pos_weight))
    # criterion = FocalLoss(gamma=2, alpha=None)
    answers = []


    for data_iter_step, (samples, targets, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # print(a, b)
        # print(data_loader.dataset[a[0]])
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    if do_lr_scale:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    else:
                        param_group["lr"] = lr_schedule_values[it]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        # if random.random() < 0.01:
        #     torch.save((samples.cpu().numpy(), targets.cpu().numpy()), f'/home/petr/veles/debug/{random.random()}_.pt')
        #     print("saved")
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.bfloat16() if bf16 else samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, targets, criterion)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()
        answers.extend(list(zip(output[:, 0].cpu().detach().tolist(), targets.cpu().detach().tolist())))

        pred = (output[:, 0] > 0) * 1

        if mixup_fn is None:
            # class_acc = (output.max(-1)[-1] == targets).float().mean()
            class_acc = (pred == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)

        true_pos = ((pred == targets) & (targets == 1)).sum()
        predicted_pos = (pred == 1).sum()
        real_pos = (targets == 1).sum()
        batch_precision = 0.0 if predicted_pos == 0 else (true_pos / predicted_pos).item()
        batch_recall = 0.0 if real_pos == 0 else (true_pos / real_pos).item()

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)
        metric_logger.meters['precision'].update(batch_precision, n=predicted_pos.item())
        metric_logger.meters['recall'].update(batch_recall, n=real_pos.item())

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    logs_outp = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    logs_outp['ROC_AUC'] = roc_auc_score(np.array([x[1] for x in answers]), np.array([x[0] for x in answers]))
    logs_outp['PR_AUC'] = average_precision_score(np.array([x[1] for x in answers]), np.array([x[0] for x in answers]))
    return logs_outp


@torch.no_grad()
def validation_one_epoch(data_loader, model, device, ds=False, bf16=False, loss_pos_weight=None, run_name=''):
    if loss_pos_weight is None:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(loss_pos_weight))
    # criterion = FocalLoss(gamma=2, alpha=None)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    answers = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        video_names = batch[2]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if ds:
            videos = videos.bfloat16() if bf16 else videos.half()
            output = model(videos)
            loss = criterion(output[:, 0], target.float())
        else:
            with torch.cuda.amp.autocast():
                output = model(videos)
                loss = criterion(output[:, 0], target.float())
        
        assert len(output.shape) == 2 and output.shape[1] == 1

        accuracy_outp = torch.cat([-output, output], 1)
        acc1, acc5 = accuracy(accuracy_outp, target, topk=(1, 5))
        # print(output.shape, target.shape)
        print(output[:, 0], target)
        # print()
        # print(f"Target elements: {target.shape[0]} Positives: {target.sum()}")
        # print(f"Output elements: {output.shape[0]} Positives: {output.sum()}")

        answers.extend(list(zip(video_names, output[:, 0].cpu().detach().tolist(), target.cpu().detach().tolist())))

        pred = (output[:, 0] > 0) * 1

        true_pos = ((pred == target) & (target == 1)).sum()
        predicted_pos = (pred == 1).sum()
        real_pos = (target == 1).sum()
        batch_precision = 0.0 if predicted_pos == 0 else (true_pos / predicted_pos).item()
        batch_recall = 0.0 if real_pos == 0 else (true_pos / real_pos).item()

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['precision'].update(batch_precision, n=predicted_pos.item())
        metric_logger.meters['recall'].update(batch_recall, n=real_pos.item())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Answers before sync:", len(answers))
    answers = synchronize_lists(answers)
    print("Answers after sync:", len(answers))
    with open(f"answers/{run_name}_answers_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.json", 'w') as f:
        json.dump(answers, f)
    # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    print(f'accuracy: {metric_logger.acc1.global_avg:.3f}% | precision: {100 * metric_logger.precision.global_avg:.3f}% | recall: {100 * metric_logger.recall.global_avg:.3f}%')

    logs_outp = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    logs_outp['ROC_AUC'] = roc_auc_score(np.array([x[2] for x in answers]), np.array([x[1] for x in answers]))
    logs_outp['PR_AUC'] = average_precision_score(np.array([x[2] for x in answers]), np.array([x[1] for x in answers]))
    logs_outp.update(calculate_metrics_for_different_conditions(answers))
    return logs_outp


@torch.no_grad()
def final_test(data_loader, model, device, file, ds=False, bf16=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if ds:
            videos = videos.bfloat16() if bf16 else videos.half()
            output = model(videos)
            loss = criterion(output, target)
        else:
            with torch.cuda.amp.autocast():
                output = model(videos)
                loss = criterion(output, target)

        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(ids[i], \
                                                str(output.data[i].float().cpu().numpy().tolist()), \
                                                str(int(target[i].cpu().numpy())), \
                                                str(int(chunk_nb[i].cpu().numpy())), \
                                                str(int(split_nb[i].cpu().numpy())))
            final_result.append(string)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split(' ')[0]
            label = line.split(']')[-1].split(' ')[1]
            chunk_nb = line.split(']')[-1].split(' ')[2]
            split_nb = line.split(']')[-1].split(' ')[3]
            data = np.fromstring(' '.join(line.split(' ')[1:]).split('[')[1].split(']')[0], dtype=np.float32, sep=',')
            data = softmax(data)
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1 ,final_top5 = np.mean(top1), np.mean(top5)
    return final_top1*100 ,final_top5*100

def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]
