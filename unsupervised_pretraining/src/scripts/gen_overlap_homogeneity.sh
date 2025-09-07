#!/bin/bash
#SBATCH --job-name=gen_overlap_homogeneity
#SBATCH --output=gen_overlap_homogeneity.log
#SBATCH --gres=gpu:1


source /local/scratch3/rwan388/anaconda/etc/profile.d/conda.sh
conda activate std

export CUDA_LAUNCH_BLOCKING=1


cd ..
python -u gen_overlap_homogeneity.py \
    --data_path path_to_data_dir \
    --dataset_name dataset_ame \
    --raw_path path_to_raw_data \
    --num_node dataset_node_number

conda deactivate


