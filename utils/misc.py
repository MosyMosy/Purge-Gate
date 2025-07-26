import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import abc

# from pointnet2_ops import pointnet2_utils
from pytorch3d.ops import sample_farthest_points  # pytorch3d


corruptions_partnet = [
    # 'clean',
    "uniform",
    "gaussian",
    "background",
    "impulse",
    "upsampling",
    "distortion_rbf",
    "distortion_rbf_inv",
    "density",
    "density_inc",
    "shear",
    "rotation",
    "cutout",
    "distortion",
    "occlusion",
    "lidar",
]
corruptions_scanobj = [
    # 'clean',
    "uniform",
    "gaussian",
    "background",
    "impulse",
    "upsampling",
    "distortion_rbf",
    "distortion_rbf_inv",
    "density",
    "density_inc",
    "shear",
    "rotation",
    "cutout",
    "distortion",
    "occlusion",
    "lidar",
]

corruptions = [
    # 'clean',
    "uniform",
    "gaussian",
    "background",
    "impulse",
    "upsampling",
    "distortion_rbf",
    "distortion_rbf_inv",
    "density",
    "density_inc",
    "shear",
    "rotation",
    "cutout",
    "distortion",
    "occlusion",
    "lidar",
    # 'mixed_corruptions_2_0', 'mixed_corruptions_2_1', 'mixed_corruptions_2_2'
]

corruptions_h5 = [
    # 'clean',
    "add_global",
    "add_local",
    "dropout_global",
    "dropout_local",
    "jitter",
    "rotate",
    "scale",
]


def fps(data, number, random_start_point=False):
    """
    data B N 3
    number int
    """
    # fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    # fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    fps_data, _ = sample_farthest_points(
        points=data, K=number, random_start_point=random_start_point
    )  # pytorch3d
    return fps_data


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_lambda_sche(opti, config):
    if config.get("decay_step") is not None:
        lr_lbmd = lambda e: max(
            config.lr_decay ** (e / config.decay_step), config.lowest_decay
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(opti, lr_lbmd)
    else:
        raise NotImplementedError()
    return scheduler


def build_lambda_bnsche(model, config):
    if config.get("decay_step") is not None:
        bnm_lmbd = lambda e: max(
            config.bn_momentum * config.bn_decay ** (e / config.decay_step),
            config.lowest_decay,
        )
        bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd)
    else:
        raise NotImplementedError()
    return bnm_scheduler


def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.
    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.
    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):
    def __init__(self, model, bn_lambda, last_epoch=-1, setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(type(model).__name__)
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def get_momentum(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        return self.lmbd(epoch)


def seprate_point_cloud(xyz, num_points, crop, fixed_points=None, padding_zeros=False):
    """
    seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
    """
    _, n, c = xyz.shape

    assert n == num_points
    assert c == 3
    if crop == num_points:
        return xyz, None

    INPUT = []
    CROP = []
    for points in xyz:
        if isinstance(crop, list):
            num_crop = random.randint(crop[0], crop[1])
        else:
            num_crop = crop

        points = points.unsqueeze(0)

        if fixed_points is None:
            center = F.normalize(torch.randn(1, 1, 3), p=2, dim=-1).cuda()
        else:
            if isinstance(fixed_points, list):
                fixed_point = random.sample(fixed_points, 1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1, 1, 3).cuda()

        distance_matrix = torch.norm(
            center.unsqueeze(2) - points.unsqueeze(1), p=2, dim=-1
        )  # 1 1 2048

        idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0, 0]  # 2048

        if padding_zeros:
            input_data = points.clone()
            input_data[0, idx[:num_crop]] = input_data[0, idx[:num_crop]] * 0

        else:
            input_data = points.clone()[0, idx[num_crop:]].unsqueeze(0)  # 1 N 3

        crop_data = points.clone()[0, idx[:num_crop]].unsqueeze(0)

        if isinstance(crop, list):
            INPUT.append(fps(input_data, 2048))
            CROP.append(fps(crop_data, 2048))
        else:
            INPUT.append(input_data)
            CROP.append(crop_data)

    input_data = torch.cat(INPUT, dim=0)  # B N 3
    crop_data = torch.cat(CROP, dim=0)  # B M 3

    return input_data.contiguous(), crop_data.contiguous()


def get_pointcloud_img(ptcloud, roll, pitch, title=None):
    fig = plt.figure(figsize=(8, 8))

    x, z, y = ptcloud.transpose(1, 0)
    ax = fig.gca(projection=Axes3D.name, adjustable="box")
    ax.axis("off")
    # ax.axis('scaled')
    ax.view_init(roll, pitch)
    max, min = np.max(ptcloud), np.min(ptcloud)
    ax.set_xbound(min, max)
    ax.set_ybound(min, max)
    ax.set_zbound(min, max)
    ax.scatter(x, y, z, zdir="z", c=y, cmap="jet")
    ax.set_title(title)
    # plt.savefig('recon.pdf')

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # fig.close()
    return img


def get_ptcloud_img(ptcloud, roll, pitch):
    fig = plt.figure(figsize=(8, 8))

    x, z, y = ptcloud.transpose(1, 0)
    ax = fig.gca(projection=Axes3D.name, adjustable="box")
    ax.axis("off")
    ax.view_init(roll, pitch)
    max, min = np.max(ptcloud), np.min(ptcloud)
    ax.set_xbound(min, max)
    ax.set_ybound(min, max)
    ax.set_zbound(min, max)
    ax.scatter(x, y, z, zdir="z", c=y, cmap="jet")

    fig.canvas.draw()
    cutoff_ratio = 0
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    cutoff = int(img.shape[0] * cutoff_ratio)

    img = img[cutoff : img.shape[0] - cutoff, cutoff : img.shape[1] - cutoff]
    plt.close(fig)
    return img


def visualize_KITTI(
    path,
    data_list,
    titles=["input", "pred"],
    cmap=["bwr", "autumn"],
    zdir="y",
    xlim=(-1, 1),
    ylim=(-1, 1),
    zlim=(-1, 1),
):
    fig = plt.figure(figsize=(6 * len(data_list), 6))
    cmax = data_list[-1][:, 0].max()

    for i in range(len(data_list)):
        data = data_list[i][:-2048] if i == 1 else data_list[i]
        color = data[:, 0] / cmax
        ax = fig.add_subplot(1, len(data_list), i + 1, projection="3d")
        ax.view_init(30, -120)
        b = ax.scatter(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            zdir=zdir,
            c=color,
            vmin=-1,
            vmax=1,
            cmap=cmap[0],
            s=4,
            linewidth=0.05,
            edgecolors="black",
        )
        ax.set_title(titles[i])

        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.2, hspace=0)
    if not os.path.exists(path):
        os.makedirs(path)

    pic_path = path + ".png"
    fig.savefig(pic_path)

    np.save(os.path.join(path, "input.npy"), data_list[0].numpy())
    np.save(os.path.join(path, "pred.npy"), data_list[1].numpy())
    plt.close(fig)


def random_dropping(pc, e):
    up_num = max(64, 768 // (e // 50 + 1))
    pc = pc
    random_num = torch.randint(1, up_num, (1, 1))[0, 0]
    pc = fps(pc, random_num)
    padding = torch.zeros(pc.size(0), 2048 - pc.size(1), 3).to(pc.device)
    pc = torch.cat([pc, padding], dim=1)
    return pc


def random_scale(partial, scale_range=[0.8, 1.2]):
    scale = torch.rand(1).cuda() * (scale_range[1] - scale_range[0]) + scale_range[0]
    return partial * scale


def mahalanobis_distance(tokens, mean, std):
    """
    Compute Mahalanobis distance between tokens and pre-training statistics.
    Args:
        tokens (torch.Tensor): Test-time token embeddings, shape (B, N, C)
        mean (torch.Tensor): Pre-training mean, shape (1, 1, C)
        std (torch.Tensor): Pre-training std deviation, shape (1, 1, C)
    Returns:
        torch.Tensor: Mahalanobis distance for each token, shape (B, N)
    """
    # Compute variance (assuming diagonal covariance)
    variance = std**2  # Shape: (1, 1, C)
    # Compute squared Mahalanobis distance per token
    diff = tokens - mean  # Shape: (B, N, C)
    mahalanobis_sq = (diff**2) / variance  # Element-wise division (B, N, C)
    # Sum over feature dimensions (C) to get final Mahalanobis distance
    mahalanobis_dist = torch.sqrt(mahalanobis_sq.sum(dim=-1))  # Shape: (B, N)
    return mahalanobis_dist


def zscore_distance(tokens, mean, std):
    """
    Compute Z-score distance between tokens and pre-training statistics.

    Args:
        tokens (torch.Tensor): Test-time token embeddings, shape (B, N, C)
        mean (torch.Tensor): Pre-training mean, shape (1, 1, C)
        std (torch.Tensor): Pre-training std deviation, shape (1, 1, C)

    Returns:
        torch.Tensor: Z-score distance for each token, shape (B, N)
    """
    # Compute Z-score
    z_score = (tokens - mean) / std  # Shape: (B, N, C)

    # Compute Euclidean (L2) norm across the feature dimension (C)
    zscore_dist = torch.sqrt((z_score**2).sum(dim=-1))  # Shape: (B, N)

    return zscore_dist

def cosine_distance(tokens, mean, std=0):
    """
    Compute cosine distance between tokens and pre-training mean.

    Args:
        tokens (torch.Tensor): Test-time token embeddings, shape (B, N, C)
        mean (torch.Tensor): Pre-training mean, shape (1, 1, C)

    Returns:
        torch.Tensor: Cosine distance for each token, shape (B, N)
    """
    # Compute cosine similarity
    tokens_norm = F.normalize(tokens, p=2, dim=-1)  # Shape: (B, N, C)
    mean_norm = F.normalize(mean, p=2, dim=-1)  # Shape: (1, 1, C)
    cosine_sim = (tokens_norm * mean_norm).sum(dim=-1)  # Shape: (B, N)

    # Compute cosine distance
    cosine_dist = 1 - cosine_sim  # Shape: (B, N)

    return cosine_dist


def euclidean_distance(tokens, mean, std=0):
    """
    Compute Euclidean distance between tokens and pre-training mean.

    Args:
        tokens (torch.Tensor): Test-time token embeddings, shape (B, N, C)
        mean (torch.Tensor): Pre-training mean, shape (1, 1, C)

    Returns:
        torch.Tensor: Euclidean distance for each token, shape (B, N)
    """
    # Compute squared difference
    diff = tokens - mean  # Shape: (B, N, C)

    # Compute Euclidean (L2) norm across the feature dimension (C)
    euclidean_dist = torch.sqrt((diff**2).sum(dim=-1))  # Shape: (B, N)

    return euclidean_dist


def dynamic_threshold(mahalanobis_distances, T_base, k=5.0, max_multiplier=2):
    """
    Compute a dynamic threshold that moves from mean toward min or max but never reaches them.

    Args:
        mahalanobis_distances (torch.Tensor): Mahalanobis distances, shape (B, N).
        T_base (float): Base threshold from clean data.
        k (float): Controls the speed of threshold movement.

    Returns:
        torch.Tensor: Binary mask (1 for outliers, 0 for inliers).
        float: Adaptive threshold.
    """
    if mahalanobis_distances.numel() == 0:
        return None  # Handle empty input

    P_min = torch.min(mahalanobis_distances)
    P_max = torch.max(mahalanobis_distances) * max_multiplier
    P_min += (P_max - P_min) * 0.1  # Add a small margin to P_min
    mu = torch.mean(mahalanobis_distances)

    # Sigmoid function to control smooth movement toward P_min or P_max
    weight = 1 / (1 + torch.exp(k * (mu - T_base)))

    # Compute threshold moving toward min or max, but never reaching them
    threshold = P_min + (P_max - P_min) * weight

    return threshold


def best_threshold_model(distances):
    min_val = distances.min().item()
    max_val = distances.max().item()
    mean_val = distances.mean().item()
    return (
        31
        if ((max_val - min_val) > 3.6)
        else max_val * 2
    )


