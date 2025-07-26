# export CUDA_VISIBLE_DEVICES=3

# Prototype Pruning
python main.py --config cfgs/tta_purge/tta_purge_scanobject.yaml --ckpts checkpoints/scan_object_src_only.pth --method prototype_purge --BN_reset
python main.py --config cfgs/tta_purge/tta_purge_modelnet.yaml --ckpts checkpoints/modelnet_src_only.pth --method prototype_purge --BN_reset
python main.py --config cfgs/tta_purge/tta_purge_shapenet.yaml --ckpts checkpoints/shapenet_src_only.pth --method prototype_purge --BN_reset

# Cla_Token Pruning
python main.py --config cfgs/tta_purge/tta_purge_scanobject.yaml --ckpts checkpoints/scan_object_src_only.pth --method cls_purge --BN_reset
python main.py --config cfgs/tta_purge/tta_purge_modelnet.yaml --ckpts checkpoints/modelnet_src_only.pth --method cls_purge --BN_reset
python main.py --config cfgs/tta_purge/tta_purge_shapenet.yaml --ckpts checkpoints/shapenet_src_only.pth --method cls_purge --BN_reset

#-------------------------------------------------- Baselines------------------------------------------------

# source only
python main.py --config cfgs/tta/tta_scanobj.yaml --ckpts checkpoints/scan_object_src_only.pth --method source_only
python main.py --config cfgs/tta/tta_modelnet.yaml --ckpts checkpoints/modelnet_src_only.pth --method source_only
python main.py --config cfgs/tta/tta_shapenet.yaml --ckpts checkpoints/shapenet_src_only.pth --method source_only

# -----------------------------------------------------------------------------------------------------------

# BFTT3D
python main.py --config cfgs/tta_purge/tta_purge_scanobject.yaml --ckpts checkpoints/scan_object_src_only.pth --method bftt3d
python main.py --config cfgs/tta_purge/tta_purge_modelnet.yaml --ckpts checkpoints/modelnet_src_only.pth --method bftt3d
python main.py --config cfgs/tta_purge/tta_purge_shapenet.yaml --ckpts checkpoints/shapenet_src_only.pth --method bftt3d

# -----------------------------------------------------------------------------------------------------------

# tent
python main.py --config cfgs/baselines/scanobject/tta_tent.yaml --ckpts checkpoints/scan_object_src_only.pth --method tent
python main.py --config cfgs/baselines/modelnet/tta_tent.yaml --ckpts checkpoints/modelnet_src_only.pth --method tent
python main.py --config cfgs/baselines/shapenet/tta_tent.yaml --ckpts checkpoints/shapenet_src_only.pth --method tent

# -----------------------------------------------------------------------------------------------------------

# t3a
python main.py --config cfgs/baselines/scanobject/tta_t3a.yaml --ckpts checkpoints/scan_object_src_only.pth --method t3a
python main.py --config cfgs/baselines/modelnet/tta_t3a.yaml --ckpts checkpoints/modelnet_src_only.pth --method t3a
python main.py --config cfgs/baselines/shapenet/tta_t3a.yaml --ckpts checkpoints/shapenet_src_only.pth --method t3a

# -----------------------------------------------------------------------------------------------------------

# shot
python main.py --config cfgs/baselines/scanobject/tta_shot.yaml --ckpts checkpoints/scan_object_src_only.pth --method shot
python main.py --config cfgs/baselines/modelnet/tta_shot.yaml --ckpts checkpoints/modelnet_src_only.pth --method shot
python main.py --config cfgs/baselines/shapenet/tta_shot.yaml --ckpts checkpoints/shapenet_src_only.pth --method shot

# -----------------------------------------------------------------------------------------------------------
