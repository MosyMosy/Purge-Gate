import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import models as models
import numpy as np
from utils.bftt3d_TCA import TCA
from utils.bftt3d_JDA import JDA
from utils.bftt3d_setup import *
from utils.bftt3d_utils_nn import *
import torch.jit
from copy import deepcopy
from models import Point_NN

from tools import builder
from utils import misc, dist_utils
from utils.logger import *
import datasets.tta_datasets as tta_datasets
from utils.misc import *


def load_tta_dataset(args, config):
    # we have 3 choices - every tta_loader returns only point and labels
    root = config.tta_dataset_path  # being lazy - 1

    if config.dataset.name == "modelnet":
        if args.corruption == "clean":
            inference_dataset = tta_datasets.ModelNet_h5(args, root)
            tta_loader = DataLoader(
                dataset=inference_dataset,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                drop_last=True,
            )
        else:
            inference_dataset = tta_datasets.ModelNet40C(args, root)
            tta_loader = DataLoader(
                dataset=inference_dataset,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                drop_last=True,
            )

    elif config.dataset.name == "scanobject":
        inference_dataset = tta_datasets.ScanObjectNN(args=args, root=root)
        tta_loader = DataLoader(
            inference_dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            drop_last=True,
        )

    elif config.dataset.name == "shapenetcore":
        inference_dataset = tta_datasets.ShapeNetCore(args=args, root=root)
        tta_loader = DataLoader(
            inference_dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            drop_last=True,
        )

    else:
        raise NotImplementedError(f"TTA for {args.tta} is not implemented")

    print_log(f"\n\n Loading data from ::: {root} ::: level ::: {args.severity}\n\n")

    return tta_loader


def load_clean_dataset(args, config):
    (train_sampler, train_dataloader) = builder.dataset_builder(
        args, config.dataset.train
    )
    return train_dataloader


def load_base_model(args, config, logger, load_part_seg=False, pretrained=True):
    base_model = builder.model_builder(config.model)
    if pretrained:
        base_model.load_model_from_ckpt(args.ckpts)
    if args.use_gpu:
        base_model.to(args.local_rank)
    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log("Using Synchronized BatchNorm ...", logger=logger)
        base_model = nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[args.local_rank % torch.cuda.device_count()],
            find_unused_parameters=True,
        )
        print_log("Using Distributed Data parallel ...", logger=logger)
    else:
        print_log("Using Data parallel ...", logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    return base_model


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_bz", type=int, default=24)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--dim", type=int, default=72)
    parser.add_argument("--k", type=int, default=120)
    parser.add_argument("--alpha", type=int, default=1000)
    parser.add_argument("--beta", type=int, default=100)
    parser.add_argument("--gamma", type=int, default=205)
    parser.add_argument("--pth", type=str, default="pointnet")
    args = parser.parse_args()
    return args


@torch.no_grad()
def runner(args, config):
    dataset_name = config.dataset.name
    logger = get_logger(args.log_name)
    resutl_file_path = os.path.join(
        "results_final_tta/",
        args.method,
        f"{dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}.txt",
    )

    print_log("==> Loading args..")
    print_log(args)

    print_log("==> creating model..")
    source_model = load_base_model(args, config, logger)
    source_model.eval()
    if args.BN_reset:
        for m in source_model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.running_mean = None  # for original implementation of tent
                m.running_var = None  # for original implementation of tent

    print_log("==> Preparing data..")
    train_loader = load_clean_dataset(args, config)

    print_log("==> Constructing Memory Bank..")
    feature_memory, label_memory = [], []

    print("==> Preparing the none parametric model..")
    point_nn = Point_NN(
        input_points=config.npoints,
        num_stages=args.bftt3d_stages,
        embed_dim=args.bftt3d_dim,
        k_neighbors=args.bftt3d_k,
        alpha=args.bftt3d_alpha,
        beta=args.bftt3d_beta,
    ).cuda()
    point_nn.eval()

    # other project
    # coral_project, tca_project, gfk_project, jda_project= CORAL(), TCA(), GFK(), JDA()
    for i, (_, _, data) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # points = points.cuda().permute(0, 2, 1)
        points = data[0].cuda()
        labels = data[1].cuda()

        # Pass through the Non-Parametric Encoder
        point_features = point_nn(points.permute(0, 2, 1))
        feature_memory.append(point_features)
        labels = labels.cuda()
        label_memory.append(labels)

    # Label Memory
    label_memory_ys = torch.cat(label_memory, dim=0)
    label_memory = F.one_hot(label_memory_ys.long()).squeeze().float()

    # Feature Memory
    feature_memory = torch.cat(feature_memory, dim=0)
    feature_memory /= feature_memory.norm(dim=-1, keepdim=True)

    # Prototype
    feature_memory, label_memory = protoype_cal(
        feature_memory, label_memory_ys, label_memory.shape[1], herding=True
    )
    # feature_memory.to(device)
    # label_memory = label_memory.cuda()

    # feature_memory /= feature_memory.norm(dim=-1, keepdim=True)
    print_log("==> Saving Test Point Cloud Features..")

    # corrupted modelnet40c and iterate through all corrutptions
    error_list, error_list_source, error_list_mixed, error_list_LAME = [], [], [], []

    args.severity = 5
    time_list = []
    for corr_id, args.corruption in enumerate(corruptions):
        start_time = time.time()
        if corr_id == 0:  # for saving results for easy copying to google sheet
            f_write = get_writer_to_all_result(
                args, config, custom_path=resutl_file_path
            )
            f_write.write(f"All Corruptions: {corruptions}" + "\n\n")
            f_write.write(f"TTA Results for Dataset: {config.dataset.name}" + "\n\n")
            f_write.write(f"Checkpoint Used: {args.ckpts}" + "\n\n")
            f_write.write(f"Corruption LEVEL: {args.severity}" + "\n\n")

        test_loader = load_tta_dataset(args, config)

        tta_model = setup_BFTT3D(source_model)
        tta_model.eval()

        label_domain_list, test_features = [], []
        (
            logits_domain_list,
            logits_source_list,
        ) = [], []

        for points_cpu, labels in tqdm(test_loader):
            points = points_cpu.cuda()
            points = points

            logits_source = source_model(points).detach()
            logits_source_list.append(logits_source)

            # Pass through the Non-Parametric Encoder
            test_features.append(point_nn(points.permute(0, 2, 1)))
            labels = labels.cuda()
            label_domain_list.append(labels)

        # Feature
        test_features = torch.cat(test_features, dim=0)
        test_features /= test_features.norm(dim=-1, keepdim=True)

        logits_source_list = torch.cat(logits_source_list)
        label_domain_list = torch.cat(label_domain_list)

        #  sperate batch test
        bz = 128
        n_batches = math.ceil(test_features.shape[0] / bz)

        for counter in range(n_batches):
            test_feature_curr = test_features[counter * bz : (counter + 1) * bz]
            logits_source_curr = logits_source_list[counter * bz : (counter + 1) * bz]
            # Subspace learning
            if True:
                feature_memory_aligned, test_features_aligned = feature_shift(
                    feature_memory, test_feature_curr
                )
                feature_memory_aligned = feature_memory_aligned.permute(1, 0)
            else:
                feature_memory_aligned = feature_memory.permute(1, 0)
                test_features_aligned = test_feature_curr
                feature_memory_aligned = feature_memory_aligned.permute(1, 0)

            Sim = test_features_aligned @ feature_memory_aligned
            # print_log('==> Starting Predicition..')
            logits = (-args.bftt3d_gamma * (1 - Sim)).exp() @ label_memory.to(
                Sim.device
            )
            s_entropy = softmax_entropy(logits_source_curr).mean(0)
            nn_entropy = softmax_entropy(logits).mean(0)

            # Label Integrate
            p = 1 - (s_entropy / (s_entropy + nn_entropy))
            f_logits = (1 - p) * logits + p * logits_source_curr
            logits_domain_list.append(f_logits)
        logits_domain_list = torch.cat(logits_domain_list)
        domain_acc = cls_acc(logits_domain_list, label_domain_list)
        source_acc = cls_acc(logits_source_list, label_domain_list)
        error_list_mixed.append(domain_acc)
        error_list_source.append(source_acc)
        print_log(f"Source's {corr_id} classification error: {source_acc:.2f}.")
        print_log(f"BFTT3D's {corr_id} classification error: {domain_acc:.2f}.")
        print_log(
            "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        )

        f_write.write(
            " ".join([str(round(float(xx), 3)) for xx in [domain_acc]]) + "\n"
        )
        f_write.flush()
        end_time = time.time()
        time_list.append(end_time - start_time)

        if corr_id == len(corruptions) - 1:
            f_write.write(
                " ".join(
                    [
                        str(round(float(xx), 3))
                        for xx in [
                            min(time_list),
                            max(time_list),
                            sum(time_list) / len(time_list),
                            np.var(time_list),
                        ]
                    ]
                )
                + "\n"
            )
            f_write.flush()
            f_write.close()

            print(
                f"Final Results Saved at:",
                resutl_file_path,
            )

    # print_log("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print_log(
        f"Mean classification error source: {sum(error_list_source) / len(error_list_source):.2f}."
    )
    print_log(
        f"Mean classification error mixed: {sum(error_list_mixed) / len(error_list_mixed):.2f}."
    )


def feature_shift(feature_memory, test_features):
    # feature_memory_shift = coral_project.fit(feature_memory.cpu().numpy(), test_features.cpu().numpy())
    # feature_memory_shift = torch.tensor(feature_memory_shift, dtype=torch.float).cuda()
    # feature_memory_shift = feature_memory_shift.permute(1, 0)
    # feature_memory_shift, test_features_shift = jda_project.fit(feature_memory.cpu().numpy(), test_features.cpu().numpy())

    tca_project = TCA()
    feature_memory_shift, test_features_shift = tca_project.fit(
        feature_memory.cpu().numpy(), test_features.cpu().numpy()
    )

    feature_memory_shift = torch.tensor(feature_memory_shift, dtype=torch.float).cuda()
    test_features_shift = torch.tensor(test_features_shift, dtype=torch.float).cuda()

    feature_memory_shift /= feature_memory_shift.norm(dim=-1, keepdim=True)
    test_features_shift /= test_features_shift.norm(dim=-1, keepdim=True)
    return feature_memory_shift, test_features_shift


def copy_model_and_optimizer(model):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    return model_state


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def mean_square_distance_batch(batch_vector1, vector2):
    squared_differences = (batch_vector1 - vector2) ** 2
    msd = torch.mean(squared_differences, dim=1)
    return msd


def protoype_cal(features, label, class_num, herding=True):
    prototye_list, label_memory_list = [], []
    for i in range(class_num):
        idx = torch.squeeze((label == i))
        mean_emb = features[idx].mean(0).unsqueeze(0)
        if herding:
            class_emb = features[idx]
            k = int(class_emb.shape[0] / 4)
            _, closese_emb_index = torch.topk(
                mean_square_distance_batch(class_emb, torch.squeeze(mean_emb)),
                k,
                largest=False,
            )
            prototye_list.append(class_emb[closese_emb_index])
            label_memory_list.append(torch.ones(k) * i)
        else:
            prototye_list.append(mean_emb)
            label_memory_list.append((torch.tensor(i).unsqueeze(0)))
    prototye_list = torch.cat(prototye_list, dim=0)
    # Label Memory
    label_memory_list = torch.cat(label_memory_list, dim=0).type(torch.LongTensor)
    label_memory_list = F.one_hot(label_memory_list).squeeze().float()
    return prototye_list, label_memory_list
