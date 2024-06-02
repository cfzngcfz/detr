# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path # ��pathlib���е���Path��, ���ڴ����ļ�·��

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler # ��PyTorch��torch.utils.dataģ���е���DataLoader��DistributedSampler��, �������ݼ��غͷֲ�ʽѵ��ʱ�����ݲ���

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset # ��datasetsģ���е���build_dataset��get_coco_api_from_dataset����, ���ڹ������ݼ��ͻ�ȡ��COCO���ݼ���ص�API
from engine import evaluate, train_one_epoch # ��engineģ���е���evaluate��train_one_epoch����, ��Щ������������ѵ��������ģ��
from models import build_model # ��modelsģ���е���build_model����, ���ڹ������ѧϰģ��


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False) # �ý�������������һ��Ŀ����ģ��(Set transformer detector)��ѵ������
    parser.add_argument('--lr', default=1e-4, type=float)           # ������һ����Ϊ lr �������в���, ��������ѧϰ��(learning rate)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)  # ������һ����Ϊ lr_backbone �������в���, ��������backbone���ֵ�ѧϰ��
    parser.add_argument('--batch_size', default=2, type=int)        # ������һ����Ϊ batch_size �������в���, ��������ѵ��ʱ�����δ�С(batch size)
    parser.add_argument('--weight_decay', default=1e-4, type=float) # ������һ����Ϊ weight_decay �������в���, ��������Ȩ��˥��(weight decay)��ֵ, ���������ǿ��
    parser.add_argument('--epochs', default=300, type=int)          # ������һ����Ϊ epochs �������в���, ��������ѵ�������ִ���
    parser.add_argument('--lr_drop', default=200, type=int)         # ������һ����Ϊ lr_drop �������в���, ��������ѧϰ�ʵ�˥������
    parser.add_argument('--clip_max_norm', default=0.1, type=float, # ������һ����Ϊ clip_max_norm �������в���, ���������ݶȲü��������ֵ, ���ṩ��һ��������Ϣ, �������ݶȲü��ĺ���
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None, # ������һ����Ϊ frozen_weights �������в���, ��������Ԥѵ��ģ�͵�·��. ����������������ַ���(str), Ĭ��ֵΪNone, ��ʾ��ʹ��Ԥѵ��ģ��. ��������˸ò�����ֵ, ��ôֻ������ͷ��(mask head)����ѵ��, �������ֽ�������. ������Ϣ�����˸ò���������, ��ָ��Ԥѵ��ģ�͵�·��, �Ծ����Ƿ�Ҫ����ģ��΢��
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str, # ������һ����Ϊ backbone �������в���, ��������Ҫʹ�õľ������(convolutional backbone)������, Ĭ��Ϊ'resnet50'. �û�����ͨ���ò���ָ��Ҫ��ģ����ʹ�õľ��������ļܹ�
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',          # ������һ����Ϊ dilation �������в���, ����һ����־����(flag), �����ڸò���ʱ, �����ÿն����(dilation convolution)��������ͨ���. �ò������ڿ����Ƿ��ھ�����������һ���������ʹ�ÿն����
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), # ������һ����Ϊ position_embedding �������в���, ��������Ҫ��ͼ������֮��ʹ�õ�λ�ñ���(positional embedding)������. ��������ѡ��: 'sine'��ʾʹ������/����λ�ñ���(Ĭ��ѡ��), 'learned'��ʾʹ�ÿ�ѧϰ��λ�ñ���.
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,        # ������һ����Ϊ enc_layers �������в���, ��������Transformer������(encoder)�еı������, Ĭ��ֵΪ6. ������������˱����������, Ӱ��ģ�͵ĸ��Ӷ�
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,        # ������һ����Ϊ dec_layers �������в���, ��������Transformer������(decoder)�еĽ������, Ĭ��ֵΪ6. ������������˽����������, ͬ��Ӱ��ģ�͵ĸ��Ӷ�
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,# ������һ����Ϊ dim_feedforward �������в���, ��������Transformer���е�ǰ����������м���ά��, Ĭ��ֵΪ2048. �������������Transformer���е�ǰ������Ŀ��
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,      # ������һ����Ϊ hidden_dim �������в���, ��������Transformerģ���е�Ƕ��(embedding)��ά��, Ĭ��ֵΪ256. �������������ģ���ڲ�������ά��
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,       # ������һ����Ϊ dropout �������в���, ��������Transformerģ���е�dropout��, Ĭ��ֵΪ0.1. dropout����ģ�͵�����, �Լ��ٹ���Ϸ���
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,            # ������һ����Ϊ nheads �������в���, ��������Transformerģ���е�ע����ͷ������, Ĭ��ֵΪ8. �������������ע�������ƵĶ�ͷ����
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,     # ������һ����Ϊ num_queries �������в���, �������ò�ѯ��(query slots)������, Ĭ��ֵΪ100. ��ѯ����Ŀ����������ͨ�����ڱ�ʾģ�Ͷ�Ŀ���Ԥ��. num_queries ������Ŀ���������о�����Ҫ����, ��Ϊ��������ģ���ܹ�Ԥ���Ŀ������
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')          # ������һ����Ϊ pre_norm �������в���, ����һ����־����(flag). ������ڸò���, ������Ԥ����(pre-norm)����, Ӱ��ģ�͵�ѵ����ʽ

    # * Segmentation ʵ���ָ������
    parser.add_argument('--masks', action='store_true',             # ������һ����Ϊ masks �������в���, ����һ����־����(flag). ����û������г���ʱ�ṩ�˸ñ�־����, �������������а�����--masks, ���ʾҪѵ���ָ�ͷ��(segmentation head). ����������ڿ����Ƿ�ѵ��ʵ���ָ�ģ�͵ķָ�ͷ��, �ָ�ͷ��ͨ�����ڽ�ͼ���еĲ�ͬʵ��(����, ��ͬ������)�ָ����. �˲�������������ѵ��ʱ�����Ƿ�Ҫ����ʵ���ָ������ѵ��, �����Ƿ�ѵ���ָ�ͷ��������û��ṩ���������, ģ�ͽ����ݱ�־������Ӧ��ѵ�����á�
                        help="Train segmentation head if the flag is provided")

    # Loss ����ÿһ��decoder��loss�����л���, ʵ�ʵ���Ҳû����, ֱ����decoder���һ��loss�������
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false', # ������һ����Ϊ no_aux_loss�������в���, ����һ����־����(flag). ����û��ṩ�������־����, �������������а�����--no_aux_loss, �����ø����Ľ�����ʧ(ÿһ�����ʧ). Ĭ�������, �������ΪFalse, ��ʾ����ʹ��ÿһ��Ľ�����ʧ
                        help="Disables auxiliary decoding losses (loss at each layer)")
    
    # * Matcher һ����ƥ������loss����
    parser.add_argument('--set_cost_class', default=1, type=float,  # �����Ȩ��ռ����. ������һ����Ϊ set_cost_class �������в���, ��������ƥ����ʧ�е������ʧ��Ȩ��, Ĭ��ֵΪ1. ����������ڿ���ƥ����ʧ�������ʧ�����Ҫ��
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,   # bboxȨ��ռ����. ������һ����Ϊ set_cost_bbox �������в���, ��������ƥ����ʧ�б߽��(bbox)��ʧ��Ȩ��, Ĭ��ֵΪ5. ����������ڿ���ƥ����ʧ�б߽����ʧ�����Ҫ��
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,   # giou����ʧռ����. ������һ����Ϊ set_cost_giou �������в���, ��������ƥ����ʧ��GIoU��ʧ��Ȩ��, Ĭ��ֵΪ2. ����������ڿ���ƥ����ʧ��GIoU��ʧ�����Ҫ��
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients ����loss�ı���
    parser.add_argument('--mask_loss_coef', default=1, type=float)  # ������һ����Ϊ mask_loss_coef �������в���, �������÷ָ�(mask)��ʧ��Ȩ��, Ĭ��ֵΪ1
    parser.add_argument('--dice_loss_coef', default=1, type=float)  # ������һ����Ϊ dice_loss_coef �������в���, ��������Dice��ʧ��Ȩ��, Ĭ��ֵΪ1
    parser.add_argument('--bbox_loss_coef', default=5, type=float)  # ������һ����Ϊ bbox_loss_coef �������в���, �������ñ߽��(bbox)��ʧ��Ȩ��, Ĭ��ֵΪ5
    parser.add_argument('--giou_loss_coef', default=2, type=float)  # ������һ����Ϊ giou_loss_coef �������в���, ��������GIoU��ʧ��Ȩ��, Ĭ��ֵΪ2
    parser.add_argument('--eos_coef', default=0.1, type=float,      # ������һ����Ϊ eos_coef �������в���, �������÷�Ŀ��������Է���Ȩ��, Ĭ��ֵΪ0.1. �������ͨ�����ڴ���Ŀ�����еı������
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')           # ������һ����Ϊ dataset_file �������в���, �����������ݼ��ļ�������, Ĭ��Ϊ'coco'. �������ͨ������ָ��Ҫʹ�õ����ݼ�
    parser.add_argument('--coco_path', type=str)                    # ������һ����Ϊ coco_path �������в���, ��������COCO���ݼ���·��
    parser.add_argument('--coco_panoptic_path', type=str)           # ������һ����Ϊ coco_panoptic_path �������в���, ��������COCO Panoptic���ݼ���·��
    parser.add_argument('--remove_difficult', action='store_true')  # ������һ����Ϊ remove_difficult �������в���, ����һ����־����(flag). ����û��ṩ�������־����, �������������а�����--remove_difficult, ���ʾ�����ݼ��ع�����ɾ�����Դ��������(����, ����ʶ���Ŀ��)

    parser.add_argument('--output_dir', default='',                 # ������һ����Ϊ output_dir �������в���, ��������ģ��ѵ�������б�������Ŀ¼·��, Ĭ��Ϊ���ַ���, ��ʾ��������
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',                 # ������һ����Ϊ device �������в���, ��������ѵ���Ͳ���ʱʹ�õļ����豸, Ĭ��Ϊ'cuda', ��ʾʹ��GPU���м���. ���Խ�������Ϊ'cpu'����CPU������
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)             # ������һ����Ϊ seed �������в���, ���������������, ��ȷ��ʵ��Ŀ��ظ���
    parser.add_argument('--resume', default='', help='resume from checkpoint') # ������һ����Ϊ resume �������в���, ����ָ�����ĸ�����(checkpoint)�ָ�ѵ��
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', # ������һ����Ϊ start_epoch �������в���, ��������ѵ������ʼ�ִ�
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')              # ������һ����Ϊ eval �������в���, ����һ����־����(flag). ����û��ṩ�������־����, �������������а�����--eval, ���ʾ����ģ�͵�����, ������ѵ��
    parser.add_argument('--num_workers', default=2, type=int)       # ������һ����Ϊ num_workers �������в���, ���������������ݼ��صĹ�����������, Ĭ��ֵΪ2. ����������Լ������ݼ��ع���

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,        # ������һ����Ϊ world_size �������в���, �������÷ֲ�ʽѵ���еĽ�������, Ĭ��ֵΪ1, ��ʾ����ѵ��. ������Ϊ����1��ֵʱ, ��ʾҪ���зֲ�ʽѵ��, ʹ�ö�����̽���ѵ������
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training') # ������һ����Ϊdist_url�������в���, �������÷ֲ�ʽѵ����URL. Ĭ��ֵΪ'env://', ͨ������������ֶ�����, ���Ǹ��ݻ��������Զ����÷ֲ�ʽѵ��
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
