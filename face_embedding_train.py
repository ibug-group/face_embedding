import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
import ibug.face_embedding.backbones.iresnet as iresnet
import ibug.face_embedding.backbones.rtnet as rtnet
import ibug.face_embedding.utils.losses as losses
from ibug.face_embedding.utils.dataset import ImageFolderFaceDataset, DataLoaderX
from ibug.face_embedding.utils.partial_fc import PartialFC
from ibug.face_embedding.utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from ibug.face_embedding.utils.utils_logging import AverageMeter, init_logging
from ibug.face_embedding.utils.utils_amp import MaxClipGradScaler
from ibug.face_embedding.utils.train_config import config as cfg

torch.backends.cudnn.benchmark = True

def main(args):

    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])

    dist_url = "tcp://{}:{}".format(os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])

    dist.init_process_group(backend='nccl', init_method=dist_url, rank=rank, world_size=world_size)
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)

    cfg.output = args.output_dir.strip()

    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.output)

    if rank==0: logging.info("Training results will be saved to: {}".format(cfg.output))

    cfg.batch_size = args.batch_size_per_gpu
    if rank == 0: logging.info("Batch size per GPU: {}".format(cfg.batch_size))

    if not os.path.exists(cfg.output) and rank==0:
        os.makedirs(cfg.output)
    else:
        time.sleep(2)

    if rank==0: logging.info("Dataset root directory: {}".format(args.data_root.strip()))

    trainset = ImageFolderFaceDataset(
        root_dir=args.data_root.strip(),
        local_rank=local_rank)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True)
    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=trainset, batch_size=cfg.batch_size,
        sampler=train_sampler, num_workers=0, pin_memory=True, drop_last=True)

    num_classes = len(trainset.image_dataset.classes)
    num_images = len(trainset.image_dataset.imgs)

    if rank==0: logging.info("Total training images: {}, Classes: {}".format(num_images, num_classes,))

    dropout = 0.4 if cfg.dataset=="webface" else 0

    net_type = args.network.strip()
    backbone = eval("iresnet.{}".format(net_type))(False, dropout=dropout, fp16=cfg.fp16).to(local_rank)

    if rank==0: logging.info("Backbone: {}".format(net_type))

    if args.resume:
        try:
            backbone_pth = os.path.join(cfg.output, "backbone.pth")
            backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))
            if rank==0:
                logging.info("backbone resume successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("resume fail, backbone init successfully!")

    for ps in backbone.parameters():
        dist.broadcast(ps, 0)
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank])
    backbone.train()

    margin_softmax = eval("losses.{}".format(args.loss))()
    module_partial_fc = PartialFC(
        rank=rank, local_rank=local_rank, world_size=world_size, resume=args.resume,
        batch_size=cfg.batch_size, margin_softmax=margin_softmax, num_classes=num_classes,
        sample_rate=cfg.sample_rate, embedding_size=cfg.embedding_size, prefix=cfg.output)

    opt_backbone = torch.optim.SGD(
        params=[{'params': backbone.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)
    opt_pfc = torch.optim.SGD(
        params=[{'params': module_partial_fc.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)

    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_backbone, lr_lambda=cfg.lr_func)
    scheduler_pfc = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_pfc, lr_lambda=cfg.lr_func)

    start_epoch = 0
    total_step = int(len(trainset) / cfg.batch_size / world_size * args.num_epoch)
    if rank==0: logging.info("Total training steps: %d" % total_step)

    callback_verification = CallBackVerification(args.verification_frequency, rank, cfg.val_targets,
                                                 args.verification_dir.strip(), backbone_type=net_type)
    callback_logging = CallBackLogging(50, rank, total_step, cfg.batch_size, world_size, None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)

    loss = AverageMeter()
    global_step = 0
    grad_scaler = MaxClipGradScaler(cfg.batch_size, 128 * cfg.batch_size, growth_interval=100) if cfg.fp16 else None
    for epoch in range(start_epoch, args.num_epoch):
        train_sampler.set_epoch(epoch)
        for step, (img, label,) in enumerate(train_loader):
            global_step += 1
            features = F.normalize(backbone(img))
            x_grad, loss_v = module_partial_fc.forward_backward(label, features, opt_pfc)
            if cfg.fp16:
                features.backward(grad_scaler.scale(x_grad))
                grad_scaler.unscale_(opt_backbone)
                clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
                grad_scaler.step(opt_backbone)
                grad_scaler.update()
            else:
                features.backward(x_grad)
                clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
                opt_backbone.step()

            opt_pfc.step()
            module_partial_fc.update()
            opt_backbone.zero_grad()
            opt_pfc.zero_grad()
            loss.update(loss_v, 1)
            callback_logging(global_step, loss, epoch, cfg.fp16, grad_scaler)
            callback_verification(global_step, backbone)

        callback_checkpoint(global_step, backbone, module_partial_fc)
        scheduler_backbone.step()
        scheduler_pfc.step()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--network', type=str, default='iresnet50',
                        help='backbone network, can be iresnet18, iresnet34, iresnet50, iresnet100 '
                             'or iresnet200 (Default: iresnet50)')
    parser.add_argument('--loss', type=str, default='ArcFace', help='loss function (default: ArcFace)')
    parser.add_argument('--resume', type=int, default=0, help='model resuming')
    parser.add_argument('--output_dir', type=str,
                        default="../fr_snapshots/arcface_emore_root",
                        help='The path of the directory to save training data.')
    parser.add_argument('--data_root', type=str,
                        default="/data/yw12414/face_recognition/datasets/megaface/megaface_112/facescrub_images",
                        help='The data root directory that can be read by torchvision.datasets.ImageFolder')
    parser.add_argument('--verification_dir', '--ver_dir', type=str,
                        default="/data/yw12414/face_recognition/datasets/faces_emore",
                        help='The path of the directory with verification bins (e.g. lfw.bin)')
    parser.add_argument('--verification_frequency', '--ver_freq', type=int,
                        default=2000,
                        help='The frequency (in training steps) to perform verification')
    parser.add_argument('--num_epoch', '--epoch', type=int, default=16, help='training epoch number')
    parser.add_argument('--batch_size_per_gpu', '--bs', type=int, default=64, help='the batch size per gpu')

    # parser.add_argument('--project_to_space', type=str, default=None,
    #                     help='The space to project facial images into. '
    #                          'Options: roi_tanh_polar, roi_tanh_circular, roi_tanh')
    # # options for space projection
    # parser.add_argument('--roi_ratio', type=str, default="0.8,0.8",
    #                     help='The ratio of RoI region with respect to the whole image when doing space projection')
    # parser.add_argument('--roi_offset_range', type=str, default="-0.09,0.09",
    #                     help='The RoI offset range during space projection')
    # parser.add_argument('--angular_offset_range', type=str, default="-0.35,0.35",
    #                     help='The angular offset during space projection')
    # parser.add_argument('--ignore_aspect_ratio', default=False, action="store_true",
    #                     help='If specified, will ignore aspect ratio during space projection')

    args_ = parser.parse_args()
    main(args_)
