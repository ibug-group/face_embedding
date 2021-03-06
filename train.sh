#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --job-name=fr   # create a short name for your job
#SBATCH --partition=a100    # learnfair or a100
#SBATCH --nodes=1               # node count
#SBATCH --gres=gpu:8             # number of gpus per node
#SBATCH --ntasks-per-node=8      # number of tasks per node
#SBATCH --cpus-per-task=10       # cpu-cores per task (>1 if multi-threaded tasks)

# source activate polarfr

# the dataset root that can be read by torchvision.datasets.ImageFolder
data_root="/data/yw12414/face_recognition/datasets/megaface/megaface_112/facescrub_images"

# the folder containing all verification bins ("agedb_30.bin", "calfw.bin", "cfp_ff.bin", etc.)
ver_dir="/data/yw12414/face_recognition/datasets/faces_emore"

# the path to save training data
output_dir="/data/yw12414/face_recognition/fr_snapshots/img_folder_test"

# nproc_per_node: number of gpus per node; nnodes: number of nodes
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 \
--master_addr="127.0.0.1" --master_port=36005 \
face_embedding_train.py --network "iresnet18" --epoch 16 --bs 64 --ver_freq 200 \
--data_root $data_root --output_dir $output_dir --ver_dir $ver_dir
