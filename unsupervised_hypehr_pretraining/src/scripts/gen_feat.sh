#!/bin/bash
#SBATCH --job-name=gen_feat
#SBATCH --output=gen_feat
#SBATCH --gres=gpu:1


source /local/scratch3/rwan388/anaconda/etc/profile.d/conda.sh
conda activate std

export CUDA_LAUNCH_BLOCKING=1

# promote
cd ..
python -u generate_feat.py \
  --data_dir path_to_dataset \
  --input_file path_to_hyperedges.txt \
  --output_file path_to_node-embeddings-dataset.txt \
  --seed 0 \
  --num_walks 10 \
  --walk_length 40 \
  --vector_size 128 \
  --window 5 \
  --epochs 10

conda deactivate


