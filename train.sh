#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --job-name=fr   # create a short name for your job
#SBATCH --partition=a100    # learnfair or a100
#SBATCH --nodes=1               # node count
#SBATCH --gres=gpu:8             # number of gpus per node
#SBATCH --ntasks-per-node=8      # number of tasks per node
#SBATCH --cpus-per-task=10       # cpu-cores per task (>1 if multi-threaded tasks)

# source activate polarfr
set -e

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 \
--master_addr="127.0.0.1" --master_port=36005 face_embedding_train.py --network iresnet18 \
--output_dir iresnet18_arcface_roiTanhPolar --project_to_space roi_tanh_polar --roi_ratio 0.8,0.8 \
--roi_offset_range " -0.09,0.09" --angular_offset_range " -0.35,0.35"
