# --------------------------------------------------------
# Fed-MAE fine-tuning for image classification (paper style)
# Pretrain: mae_vit_* (encoder+decoder)
# Finetune: vit_* (plain ViT encoder + classifier head)
# --------------------------------------------------------

import argparse, datetime, json, numpy as np, time, os, sys
from pathlib import Path
from copy import deepcopy
import re
import torch



import torch, torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter as _TBWriter

# ============ TB Writer Wrapper ============
class TBWriter(_TBWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = 0
        self.writer = self
    def set_step(self, step=None):
        self.step = self.step + 1 if step is None else int(step)
    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        if global_step is None: global_step = self.step
        return super().add_scalar(tag, scalar_value, global_step, *args, **kwargs)
    def update(self, test_acc1=None, test_acc5=None, test_loss=None, head="perf", step=None):
        if test_acc1 is not None: self.add_scalar(f"{head}/test_acc1", test_acc1, step)
        if test_acc5 is not None: self.add_scalar(f"{head}/test_acc5", test_acc5, step)
        if test_loss is not None: self.add_scalar(f"{head}/test_loss", test_loss, step)

# ============ Paths ============
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import fed_mae.models_vit as models_vit
from fed_mae.engine_for_finetuning import train_one_epoch
import util.misc as misc
from util.FedAvg_utils import Partial_Client_Selection, valid, average_model
from util.data_utils import DatasetFLFinetune, create_dataset_and_evalmetrix
from util.start_config import print_options

# ============ Arg Parser ============
def get_args():
    parser = argparse.ArgumentParser('Fed-MAE fine-tuning', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--model', default='vit_large_patch16', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--drop_path', type=float, default=0.1)
    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')

    # optimizer
    parser.add_argument('--clip_grad', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--blr', type=float, default=5e-4)
    parser.add_argument('--layer_decay', type=float, default=0.75)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    # aug
    parser.add_argument('--color_jitter', type=float, default=None)
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1')
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--reprob', type=float, default=0.25)
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--resplit', action='store_true', default=False)
    parser.add_argument('--mixup', type=float, default=0)
    parser.add_argument('--cutmix', type=float, default=0)
    parser.add_argument('--mixup_prob', type=float, default=1.0)
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5)
    parser.add_argument('--mixup_mode', type=str, default='batch')
    # finetune
    parser.add_argument('--finetune', default='', help='path to MAE pretrain ckpt')
    parser.add_argument('--global_pool', action='store_true'); parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool')
    # dataset
    parser.add_argument('--data_set', default='Retina', type=str)
    parser.add_argument('--data_path', default='/../../data/Retina', type=str)
    parser.add_argument('--nb_classes', default=2, type=int)
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--dist_eval', action='store_true', default=False)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true'); parser.set_defaults(pin_mem=True)
    # distributed
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--sync_bn', default=False, action='store_true')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    # FL
    parser.add_argument("--n_clients", default=5, type=int)
    parser.add_argument("--E_epoch", default=1, type=int)
    parser.add_argument("--max_communication_rounds", default=100, type=int)
    parser.add_argument("--num_local_clients", default=-1, type=int)
    parser.add_argument("--split_type", type=str, default="central")
    return parser.parse_args()

# ============ Main Loop ============
def main(args, model):
    print("Initializing distributed mode...")
    misc.init_distributed_mode(args)
    print(f"Distributed mode initialized: world_size={args.world_size}, local_rank={args.local_rank}")
    device = torch.device(args.device)
    misc.fix_random_seeds(args)
    cudnn.benchmark = True

    print("Creating dataset and evaluation metrics...")
    try:
        create_dataset_and_evalmetrix(args, mode='finetune')
        print(f"Dataset created: dis_cvs_files={args.dis_cvs_files}, clients_with_len={args.clients_with_len}")
    except Exception as e:
        print(f"Error in create_dataset_and_evalmetrix: {e}")
        sys.exit(1)

    dataset_val = None if args.disable_eval_during_finetuning else DatasetFLFinetune(args=args, phase='test')
    dataset_test = DatasetFLFinetune(args=args, phase='test') if args.eval else None
    sampler_val = torch.utils.data.SequentialSampler(dataset_val) if dataset_val else None
    data_loader_val = torch.utils.data.DataLoader(dataset_val, sampler=sampler_val,
                        batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=args.pin_mem, drop_last=False) if dataset_val else None
    data_loader_test = torch.utils.data.DataLoader(dataset_test, sampler=torch.utils.data.SequentialSampler(dataset_test),
                        batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=args.pin_mem, drop_last=False) if dataset_test else None

    if args.eval:
        if not args.resume:
            print("Error: --resume required for --eval")
            sys.exit(1)
        model.to(device)
        test_stats = valid(args, model, data_loader_test or data_loader_val)
        print(f"Accuracy on test: {test_stats['acc1']:.1f}%")
        sys.exit(0)

    # Load checkpoint before Partial_Client_Selection to set start_epoch
    if args.resume and os.path.isfile(args.resume):
        print(f"=> Resuming fine-tune from {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location='cpu')
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
            # Try to get epoch from checkpoint, fallback to parsing filename
            epoch = checkpoint.get('epoch', 0)
            if epoch == 0:
                filename = os.path.basename(args.resume)
                match = re.search(r'checkpoint-(\d+)\.pth', filename)
                if match:
                    epoch = int(match.group(1))
            args.start_epoch = epoch + 1
            print(f"=> Resumed at epoch {args.start_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            sys.exit(1)

    print("Running Partial_Client_Selection...")
    try:
        model_all, optimizer_all, criterion_all, loss_scaler_all, mixup_fn_all = Partial_Client_Selection(args, model, mode='finetune')
        print(f"Partial_Client_Selection completed: {len(model_all)} clients initialized")
    except Exception as e:
        print(f"Error in Partial_Client_Selection: {e}")
        sys.exit(1)

    model_avg = deepcopy(model).cpu()
    log_writer = TBWriter(log_dir=args.log_dir) if args.log_dir else None

    # Synchronize client models with the resumed model_avg
    if args.resume and os.path.isfile(args.resume):
        for client in args.dis_cvs_files:
            model_all[client].load_state_dict(model_avg.state_dict())

    print("=============== Running fine-tuning ===============")
    print(f"Starting epoch: {args.start_epoch}, max_communication_rounds: {args.max_communication_rounds}")
    epoch, start_time, max_accuracy = args.start_epoch, time.time(), 0.0
    if epoch > args.max_communication_rounds:
        print(f"Skipping training: start_epoch ({epoch}) > max_communication_rounds ({args.max_communication_rounds})")
        sys.exit(0)
    while epoch <= args.max_communication_rounds:
        print(f'epoch: {epoch}')

        cur_selected_clients = args.proxy_clients if args.num_local_clients == len(args.dis_cvs_files) \
                               else np.random.choice(args.dis_cvs_files, args.num_local_clients, replace=False).tolist()
        cur_tot_len = sum(args.clients_with_len[c] for c in cur_selected_clients)

        for proxy_single_client in cur_selected_clients:
            args.single_client = proxy_single_client
            args.clients_weightes[proxy_single_client] = args.clients_with_len[proxy_single_client] / cur_tot_len

            dataset_train = DatasetFLFinetune(args=args, phase='train')
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            data_loader_train = torch.utils.data.DataLoader(dataset_train, sampler=sampler_train,
                        batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=args.pin_mem, drop_last=True)

            model_c = model_all[proxy_single_client]
            optimizer, criterion = optimizer_all[proxy_single_client], criterion_all[proxy_single_client]
            loss_scaler, mixup_fn = loss_scaler_all[proxy_single_client], mixup_fn_all[proxy_single_client]
            model_without_ddp = model_c.module if args.distributed else model_c

            for inner_epoch in range(args.E_epoch):
                train_stats = train_one_epoch(
                    model_c, criterion, data_loader_train,
                    optimizer, device, epoch, loss_scaler,
                    args.clip_grad, proxy_single_client,
                    mixup_fn, log_writer=log_writer, args=args)

                log_stats = {**{f'train_{k}': v for k,v in train_stats.items()},
                             'client': proxy_single_client, 'epoch': epoch}
                if args.output_dir and misc.is_main_process():
                    with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n")

        average_model(args, model_avg, model_all)
        # Update client models with the averaged model
        for client in args.dis_cvs_files:
            model_all[client].load_state_dict(model_avg.state_dict())

        if args.output_dir and ((epoch+1)%args.save_ckpt_freq==0 or (epoch+1)==args.max_communication_rounds):
            torch.save({
                'model': model_avg.state_dict(),
                'epoch': epoch,
            }, os.path.join(args.output_dir, f"checkpoint-{epoch}.pth"))

        if data_loader_val is not None:
            model_avg.to(args.device)
            test_stats = valid(args, model_avg, data_loader_val)
            print(f"Val acc: {test_stats['acc1']:.2f}%")
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir:
                    torch.save({
                        'model': model_avg.state_dict(),
                        'epoch': epoch,
                    }, os.path.join(args.output_dir, f"checkpoint-best.pth"))

            model_avg.to('cpu')

        epoch += 1  # Increment epoch at the end of the loop

    print("Training time:", str(datetime.timedelta(seconds=int(time.time()-start_time))))

# ============ Entrypoint ============
if __name__ == '__main__':
    args = get_args()
    args.model_name = args.model  # FIX: compatibility for FedAvg_utils

    if args.output_dir: Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Build plain ViT (NOT MAE) for finetune
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
        )
    print(model)
    print_options(args, model)

    # ===== Resume or Finetune =====
    if args.finetune and os.path.isfile(args.finetune):
        print(f"=> Loading MAE pretrain from {args.finetune}")
        try:
            ckpt = torch.load(args.finetune, map_location="cpu")
            state = ckpt.get('model', ckpt)
            enc_state = {k.replace("encoder.",""):v for k,v in state.items() if k.startswith("encoder.")}
            missing, unexpected = model.load_state_dict(enc_state, strict=False)
            print(f"[finetune load] missing={len(missing)} unexpected={len(unexpected)}")
        except Exception as e:
            print(f"Error loading finetune checkpoint: {e}")
            sys.exit(1)
    elif not args.resume:
        print("WARNING: no pretrain or resume checkpoint found, training from scratch!")

    main(args, model)