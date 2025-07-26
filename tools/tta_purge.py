import os
import time
from tools import builder
from utils import misc, dist_utils
from utils.logger import *
import datasets.tta_datasets as tta_datasets
from torch.utils.data import DataLoader
from utils.misc import *
import numpy as np


level = [5]

from tqdm import tqdm


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

    print(f"\n\n Loading data from ::: {root} ::: level ::: {args.severity}\n\n")

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


def runner(args, config):
    dataset_name = config.dataset.name
    logger = get_logger(args.log_name)

    source_model = load_base_model(args, config, logger)
    source_model.eval()

    if args.method in ["prototype_purge"]:
        clean_intermediates_path = (
            f"intermediate_features/{dataset_name}_clean_intermediates.pth"
        )
        if os.path.exists(clean_intermediates_path):
            clean_intermediates = torch.load(clean_intermediates_path)
            clean_intermediates_mean, clean_intermediates_std = (
                clean_intermediates["mean"],
                clean_intermediates["std"],
            )
        else:
            clean_intermediates_mean, clean_intermediates_std = (
                generate_intermediate_embeddings(args, config, source_model)
            )
            torch.save(
                {
                    "mean": clean_intermediates_mean.cpu(),
                    "std": clean_intermediates_std.cpu(),
                },
                clean_intermediates_path,
            )
        clean_intermediates_mean = clean_intermediates_mean.cuda()
        clean_intermediates_std = clean_intermediates_std.cuda()

    else:
        clean_intermediates_mean, clean_intermediates_std = None, None

    resutl_file_path = os.path.join(
        "results_final_tta/",
        args.method,
        f"{args.exp_name}_{dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}.txt",
    )

    if args.method in [
        "prototype_purge",
        "cls_purge",
    ]:
        eval_purge(
            args,
            config,
            logger,
            source_model,
            clean_intermediates_mean,
            clean_intermediates_std,
            resutl_file_path,
        )

    else:
        raise NotImplementedError(f"Method {args.method} not implemented")


def generate_intermediate_embeddings(args, config, source_model):
    def update_stats(x, count, mean, M2):
        count += 1
        delta = x - mean
        mean = mean + delta / count
        delta2 = x - mean
        M2 = M2 + delta * delta2
        return count, mean, M2

    def finalize_stats(count, mean, M2):
        if count < 2:
            # Not enough data points to compute variance
            return mean, torch.full_like(M2, float("nan"))
        variance = M2 / (count - 1)
        std = torch.sqrt(variance)
        return mean, std

    clean_dataloader = load_clean_dataset(args, config)
    count = 0
    mean = torch.zeros(12, 384, dtype=float)
    M2 = torch.zeros(12, 384, dtype=float)
    for i, (_, _, data) in tqdm(
        enumerate(clean_dataloader), total=len(clean_dataloader)
    ):
        points = data[0].cuda()
        points = misc.fps(points, config.npoints)
        with torch.no_grad():
            intermediates = source_model.module.forward_out_intermediate(points)
        intermediates = torch.stack(
            intermediates, dim=0
        )  # (num_layers, batch_size * tokens, emb_dim)

        for i in range(intermediates.shape[1]):
            x = intermediates[:, i, :]  # x has shape (L, C)
            count, mean, M2 = update_stats(x, count, mean, M2)

    return finalize_stats(count, mean, M2)


@torch.jit.script
def softmax_entropy(x: torch.Tensor, dim:int=-1) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)


def eval_purge(
    args,
    config,
    logger,
    source_model,
    clean_intermediates_mean,
    clean_intermediates_std,
    resutl_file_path,
):
    time_list = []
    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions):
            start_time = time.time()
            acc_sliding_window = list()
            acc_avg = list()
            if args.corruption == "clean":
                continue
                # raise NotImplementedError('Not possible to use tta with clean data, please modify the list above')

            # if corr_id not in [2]:
            #     continue

            if "f_write" not in locals():  # for saving results for easy copying to google sheet
                f_write = get_writer_to_all_result(
                    args, config, custom_path=resutl_file_path
                )
                f_write.write(f"All Corruptions: {corruptions}" + "\n\n")
                f_write.write(
                    f"TTA Results for Dataset: {config.dataset.name}" + "\n\n"
                )
                f_write.write(f"Checkpoint Used: {args.ckpts}" + "\n\n")
                f_write.write(f"Corruption LEVEL: {args.severity}" + "\n\n")

            tta_loader = load_tta_dataset(args, config)
            test_pred = []
            test_label = []
            entropy_list = []
            
            
            base_model = load_base_model(args, config, logger, pretrained=False)
            base_model.load_state_dict(source_model.state_dict())
            
            for idx, (data, labels) in enumerate(tta_loader):
                # now inferring on this one sample
                
                # reset batchnorm running stats
                
                base_model.eval()
                
                
                points = data.cuda()
                labels = labels.cuda()
                points = [
                    misc.fps(points, config.npoints, random_start_point=True)
                    for _ in range(args.batch_size_tta)
                ]
                points = torch.cat(points, dim=0)

                labels = [labels for _ in range(args.batch_size_tta)]
                labels = torch.cat(labels, dim=0)
                        

                # reset batchnorm running stats
                if args.BN_reset:
                    for m in base_model.modules():
                        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                            m.running_mean = None  # for original implementation of tent
                            m.running_var = None  # for original implementation of tent
                purge_sizes = args.purge_size_list
                logits = []
                for i in range(len(purge_sizes)):
                    if args.method == "prototype_purge":
                        with torch.no_grad():
                            logits.append(
                                base_model.module.forward_prototype_purge(
                                    points,
                                    source_stats=(
                                        clean_intermediates_mean,
                                        clean_intermediates_std,
                                    ),
                                    layer_idx=[0],
                                    purge_size=purge_sizes[i],
                                ).unsqueeze(1)
                            )
                    elif args.method == "cls_purge":
                        with torch.no_grad():
                            logits.append(
                                base_model.module.forward_cls_purge(
                                    points,
                                    purge_size=purge_sizes[i],
                                ).unsqueeze(1)
                            )
                logits = torch.cat(logits, dim=1)
                entropy = softmax_entropy(logits, dim=-1)
                # entropy_list.append(entropy.mean().cpu())
                logits = logits[torch.arange(logits.shape[0]), entropy.argmin(dim=-1)]

                target = labels.view(-1)
                pred = logits.argmax(-1).view(-1)

                test_pred.append(pred.detach())
                test_label.append(target.detach())

                if idx % 50 == 0:
                    test_pred_ = torch.cat(test_pred, dim=0)
                    test_label_ = torch.cat(test_label, dim=0)

                    if args.distributed:
                        test_pred_ = dist_utils.gather_tensor(test_pred_, args)
                        test_label_ = dist_utils.gather_tensor(test_label_, args)

                    acc = (
                        (test_pred_ == test_label_).sum()
                        / float(test_label_.size(0))
                        * 100.0
                    )

                    print_log(
                        f"\n\n\nIntermediate Accuracy - IDX {idx} - {acc:.1f}\n\n\n",
                        logger=logger,
                    )

                    acc_avg.append(acc.cpu())
            end_time = time.time()
            time_list.append(end_time - start_time)
            test_pred = torch.cat(test_pred, dim=0)
            test_label = torch.cat(test_label, dim=0)

            if args.distributed:
                test_pred = dist_utils.gather_tensor(test_pred, args)
                test_label = dist_utils.gather_tensor(test_label, args)

            acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.0
            print_log(
                f"\n\n######## Final Accuracy ::: {args.corruption} ::: {acc} ########\n\n",
                logger=logger,
            )
            f_write.write(" ".join([str(round(float(xx), 3)) for xx in [acc]]) + "\n")
            # f_write.write(
            #     " ".join([str(round(float(xx), 3)) for xx in [torch.stack(entropy_list).mean().item()]]) + "\n"
            # )
            f_write.flush()

            if corr_id == len(corruptions) - 1:
                # write min, max, and average, variance,  of times
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

