# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path # 从pathlib库中导入Path类, 用于处理文件路径

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler # 从PyTorch的torch.utils.data模块中导入DataLoader和DistributedSampler类, 用于数据加载和分布式训练时的数据采样

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset # 从datasets模块中导入build_dataset和get_coco_api_from_dataset函数, 用于构建数据集和获取与COCO数据集相关的API
from engine import evaluate, train_one_epoch # 从engine模块中导入evaluate和train_one_epoch函数, 这些函数可能用于训练和评估模型
from models import build_model # 从models模块中导入build_model函数, 用于构建深度学习模型


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False) # 该解析器用于配置一个目标检测模型(Set transformer detector)的训练过程
    parser.add_argument('--lr', default=1e-4, type=float)           # 定义了一个名为 lr 的命令行参数, 用于设置学习率(learning rate)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)  # 定义了一个名为 lr_backbone 的命令行参数, 用于设置backbone部分的学习率
    parser.add_argument('--batch_size', default=2, type=int)        # 定义了一个名为 batch_size 的命令行参数, 用于设置训练时的批次大小(batch size)
    parser.add_argument('--weight_decay', default=1e-4, type=float) # 定义了一个名为 weight_decay 的命令行参数, 用于设置权重衰减(weight decay)的值, 即正则化项的强度
    parser.add_argument('--epochs', default=300, type=int)          # 定义了一个名为 epochs 的命令行参数, 用于设置训练的总轮次数
    parser.add_argument('--lr_drop', default=200, type=int)         # 定义了一个名为 lr_drop 的命令行参数, 用于设置学习率的衰减周期
    parser.add_argument('--clip_max_norm', default=0.1, type=float, # 定义了一个名为 clip_max_norm 的命令行参数, 用于设置梯度裁剪的最大范数值, 并提供了一个帮助信息, 解释了梯度裁剪的含义
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None, # 定义了一个名为 frozen_weights 的命令行参数, 用于设置预训练模型的路径. 这个参数的类型是字符串(str), 默认值为None, 表示不使用预训练模型. 如果设置了该参数的值, 那么只有掩码头部(mask head)将被训练, 其他部分将被冻结. 帮助信息解释了该参数的作用, 即指定预训练模型的路径, 以决定是否要进行模型微调
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str, # 定义了一个名为 backbone 的命令行参数, 用于设置要使用的卷积背景(convolutional backbone)的名称, 默认为'resnet50'. 用户可以通过该参数指定要在模型中使用的卷积神经网络的架构
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',          # 定义了一个名为 dilation 的命令行参数, 它是一个标志参数(flag), 当存在该参数时, 将启用空洞卷积(dilation convolution)而不是普通卷积. 该参数用于控制是否在卷积背景的最后一个卷积块中使用空洞卷积
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), # 定义了一个名为 position_embedding 的命令行参数, 用于设置要在图像特征之上使用的位置编码(positional embedding)的类型. 它有两个选择: 'sine'表示使用正弦/余弦位置编码(默认选项), 'learned'表示使用可学习的位置编码.
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,        # 定义了一个名为 enc_layers 的命令行参数, 用于设置Transformer编码器(encoder)中的编码层数, 默认值为6. 这个参数控制了编码器的深度, 影响模型的复杂度
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,        # 定义了一个名为 dec_layers 的命令行参数, 用于设置Transformer解码器(decoder)中的解码层数, 默认值为6. 这个参数控制了解码器的深度, 同样影响模型的复杂度
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,# 定义了一个名为 dim_feedforward 的命令行参数, 用于设置Transformer块中的前馈神经网络的中间层的维度, 默认值为2048. 这个参数控制了Transformer块中的前馈网络的宽度
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,      # 定义了一个名为 hidden_dim 的命令行参数, 用于设置Transformer模型中的嵌入(embedding)的维度, 默认值为256. 这个参数控制了模型内部的向量维度
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,       # 定义了一个名为 dropout 的命令行参数, 用于设置Transformer模型中的dropout率, 默认值为0.1. dropout用于模型的正则化, 以减少过拟合风险
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,            # 定义了一个名为 nheads 的命令行参数, 用于设置Transformer模型中的注意力头的数量, 默认值为8. 这个参数控制了注意力机制的多头数量
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,     # 定义了一个名为 num_queries 的命令行参数, 用于设置查询槽(query slots)的数量, 默认值为100. 查询槽在目标检测任务中通常用于表示模型对目标的预测. num_queries 参数在目标检测任务中具有重要作用, 因为它决定了模型能够预测的目标数量
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')          # 定义了一个名为 pre_norm 的命令行参数, 它是一个标志参数(flag). 如果存在该参数, 将启用预正则化(pre-norm)操作, 影响模型的训练方式

    # * Segmentation 实例分割的内容
    parser.add_argument('--masks', action='store_true',             # 定义了一个名为 masks 的命令行参数, 它是一个标志参数(flag). 如果用户在运行程序时提供了该标志参数, 即输入命令行中包含了--masks, 则表示要训练分割头部(segmentation head). 这个参数用于控制是否训练实例分割模型的分割头部, 分割头部通常用于将图像中的不同实例(例如, 不同的物体)分割出来. 此参数的作用是在训练时决定是否要进行实例分割任务的训练, 具体是否训练分割头部。如果用户提供了这个参数, 模型将根据标志进行相应的训练配置。
                        help="Train segmentation head if the flag is provided")

    # Loss 计算每一层decoder的loss并进行汇总, 实际当中也没有用, 直接用decoder最后一层loss来计算的
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false', # 定义了一个名为 no_aux_loss的命令行参数, 它是一个标志参数(flag). 如果用户提供了这个标志参数, 即输入命令行中包含了--no_aux_loss, 则会禁用辅助的解码损失(每一层的损失). 默认情况下, 这个参数为False, 表示允许使用每一层的解码损失
                        help="Disables auxiliary decoding losses (loss at each layer)")
    
    # * Matcher 一个是匹配器的loss比重
    parser.add_argument('--set_cost_class', default=1, type=float,  # 分类的权重占多少. 定义了一个名为 set_cost_class 的命令行参数, 用于设置匹配损失中的类别损失的权重, 默认值为1. 这个参数用于控制匹配损失中类别损失项的重要性
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,   # bbox权重占多少. 定义了一个名为 set_cost_bbox 的命令行参数, 用于设置匹配损失中边界框(bbox)损失的权重, 默认值为5. 这个参数用于控制匹配损失中边界框损失项的重要性
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,   # giou的损失占多少. 定义了一个名为 set_cost_giou 的命令行参数, 用于设置匹配损失中GIoU损失的权重, 默认值为2. 这个参数用于控制匹配损失中GIoU损失项的重要性
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients 真正loss的比重
    parser.add_argument('--mask_loss_coef', default=1, type=float)  # 定义了一个名为 mask_loss_coef 的命令行参数, 用于设置分割(mask)损失的权重, 默认值为1
    parser.add_argument('--dice_loss_coef', default=1, type=float)  # 定义了一个名为 dice_loss_coef 的命令行参数, 用于设置Dice损失的权重, 默认值为1
    parser.add_argument('--bbox_loss_coef', default=5, type=float)  # 定义了一个名为 bbox_loss_coef 的命令行参数, 用于设置边界框(bbox)损失的权重, 默认值为5
    parser.add_argument('--giou_loss_coef', default=2, type=float)  # 定义了一个名为 giou_loss_coef 的命令行参数, 用于设置GIoU损失的权重, 默认值为2
    parser.add_argument('--eos_coef', default=0.1, type=float,      # 定义了一个名为 eos_coef 的命令行参数, 用于设置非目标类别的相对分类权重, 默认值为0.1. 这个参数通常用于处理目标检测中的背景类别
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')           # 定义了一个名为 dataset_file 的命令行参数, 用于设置数据集文件的名称, 默认为'coco'. 这个参数通常用于指定要使用的数据集
    parser.add_argument('--coco_path', type=str)                    # 定义了一个名为 coco_path 的命令行参数, 用于设置COCO数据集的路径
    parser.add_argument('--coco_panoptic_path', type=str)           # 定义了一个名为 coco_panoptic_path 的命令行参数, 用于设置COCO Panoptic数据集的路径
    parser.add_argument('--remove_difficult', action='store_true')  # 定义了一个名为 remove_difficult 的命令行参数, 它是一个标志参数(flag). 如果用户提供了这个标志参数, 即输入命令行中包含了--remove_difficult, 则表示在数据加载过程中删除难以处理的样本(例如, 难以识别的目标)

    parser.add_argument('--output_dir', default='',                 # 定义了一个名为 output_dir 的命令行参数, 用于设置模型训练过程中保存结果的目录路径, 默认为空字符串, 表示不保存结果
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',                 # 定义了一个名为 device 的命令行参数, 用于设置训练和测试时使用的计算设备, 默认为'cuda', 表示使用GPU进行计算. 可以将其设置为'cpu'以在CPU上运行
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)             # 定义了一个名为 seed 的命令行参数, 用于设置随机种子, 以确保实验的可重复性
    parser.add_argument('--resume', default='', help='resume from checkpoint') # 定义了一个名为 resume 的命令行参数, 用于指定从哪个检查点(checkpoint)恢复训练
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', # 定义了一个名为 start_epoch 的命令行参数, 用于设置训练的起始轮次
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')              # 定义了一个名为 eval 的命令行参数, 它是一个标志参数(flag). 如果用户提供了这个标志参数, 即输入命令行中包含了--eval, 则表示进行模型的评估, 而不是训练
    parser.add_argument('--num_workers', default=2, type=int)       # 定义了一个名为 num_workers 的命令行参数, 用于设置用于数据加载的工作进程数量, 默认值为2. 这个参数可以加速数据加载过程

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,        # 定义了一个名为 world_size 的命令行参数, 用于设置分布式训练中的进程数量, 默认值为1, 表示单机训练. 当设置为大于1的值时, 表示要进行分布式训练, 使用多个进程进行训练任务
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training') # 定义了一个名为dist_url的命令行参数, 用于设置分布式训练的URL. 默认值为'env://', 通常情况下无需手动设置, 而是根据环境变量自动配置分布式训练
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
