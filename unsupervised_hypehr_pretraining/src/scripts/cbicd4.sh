#!/bin/bash
#SBATCH --job-name=cbicd4
#SBATCH --output=cbicd4.log
#SBATCH --gres=gpu:1


source /local/scratch3/rwan388/anaconda/etc/profile.d/conda.sh
conda activate std

export CUDA_LAUNCH_BLOCKING=1


cd ..
python -u train.py --dname=combine_icd4 --epochs=600 --cuda=1 --num_labels=1 --num_nodes=6729 --num_labeled_data=all --All_num_layers 3 --cuda=1 --feature_dim=128 --heads=4 --MLP_num_layers 2 --MLP_hidden 32 --model_lambda=0.01 --vanilla --pretrain_epoch=100 --pretrain_lr=1.0e-03 --pretrain_weight_decay=1.0e-06 --pretrain_drop_feature_rate=0.1 --pretrain_drop_incidence_rate=0.1 --pretrain_tau_n=0.3 --pretrain_tau_g=0.3 --pretrain_tau_m=2 --pretrain_w_gS=1 --pretrain_w_g=1 --pretrain_w_m=1 --pretrain_ng_batch_size=2048 --pretrain_m_batch_size=4096 --train_percentage=1.0  --pretrain=True 
# --weighted_methods=True --methods=gradnorm
#  --simpleModel=True
# --pareto=True
# --weighted_methods=True --methods=mgda


conda deactivate


