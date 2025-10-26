# --------------------------------------------------------
# Based on MAE code bases
# Integrate MAE for Federated Learning
# Reference: https://github.com/facebookresearch/mae
# Author: Rui Yan
# --------------------------------------------------------

import math
from typing import Iterable, Optional

import torch
from timm.data import Mixup
from timm.utils import accuracy

import os
import sys
sys.path.append(os.path.abspath('..'))
import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    proxy_single_client=None,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    """
    Standard fine-tuning epoch:
      - Unpack (samples, targets) from data_loader
      - Optional mixup
      - Autocast with torch.amp.autocast('cuda')
      - Per-iteration LR schedule
      - Gradient accumulation via args.accum_iter
    """
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):

        # global step per client (for FL schedulers/records)
        if proxy_single_client is not None:
            args.global_step_per_client[proxy_single_client] += 1

        # per-iteration LR scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        ##print("DEBUG:", samples.shape, targets.shape)

        # ADD THIS DEBUG BLOCK (it runs only once)
        if epoch == 0 and data_iter_step == 0:
            print("DEBUG outputs/targets pre-check:")
            with torch.no_grad():
                tmp_out = model(samples)           # raw logits expected: [B, num_classes]
            print("outputs.shape:", tuple(tmp_out.shape))
            print("targets:", targets.dtype, torch.unique(targets).tolist())
            # Remove if you use mixup; keep only for plain CE runs.


        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # autocast (new API)
        use_cuda_amp = (device.type == 'cuda')
        ctx = torch.amp.autocast('cuda') if use_cuda_amp else misc.nullcontext()
        with ctx:
            outputs = model(samples)            # [N, num_classes]
            loss = criterion(outputs, targets)  # scalar

        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        # grad accumulation
        loss = loss / accum_iter
        loss_scaler(
            loss, optimizer, clip_grad=max_norm,
            parameters=model.parameters(), create_graph=False,
            update_grad=((data_iter_step + 1) % accum_iter == 0)
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize() if device.type == 'cuda' else None

        # logs
        metric_logger.update(loss=loss_value)
        min_lr, max_lr = 10.0, 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group.get("lr", 0.0))
            max_lr = max(max_lr, group.get("lr", 0.0))
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            # x-axis in TB unified to 1000x epoch scale
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # sync stats
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    """
    Validation loop that mirrors train:
      - Unpacks (images, target)
      - Uses torch.amp.autocast('cuda') if on CUDA
      - Reports top-1 and top-5 (acc5 requires nb_classes >= 5; harmless otherwise)
    """
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()
    use_cuda_amp = (device.type == 'cuda')
    ctx = torch.amp.autocast('cuda') if use_cuda_amp else misc.nullcontext()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with ctx:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
