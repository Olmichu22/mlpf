#!/bin/bash
set -euo pipefail

IMG="/nfs/cms/arqolmo/GPU_train/mlpf/gatr_v9.sif"

# Variables que quieres dentro del contenedor
export SINGULARITYENV_PYTHONUSERBASE="/nfs/cms/arqolmo/GPU_train/mlpf/extlib"
export SINGULARITYENV_PATH="$SINGULARITYENV_PYTHONUSERBASE/bin:$PATH"
export SINGULARITYENV_PYTHONPATH="$SINGULARITYENV_PYTHONUSERBASE/lib/python3.8/site-packages:$PYTHONPATH"
# Evita login interactivo de W&B
# export SINGULARITYENV_WANDB_API_KEY="$WANDB_API_KEY"

CMD='
pwd
set -e
echo "Inside container:"
hostname
nvidia-smi || true
python -m site --user-site
echo "PYTHONPATH = $PYTHONPATH"
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
wandb login
cd /nfs/cms/arqolmo/GPU_train/mlpf
python -m src.train_lightning1 \
  --data-train /pnfs/ciemat.es/data/calice/arqolmo/TrainTrees/train_tree_{1..300}.root \
  --data-config config_files/config_hits_track_v2_noise.yaml \
  -clust -clust_dim 3 \
  --network-config src/models/wrapper/example_mode_gatr_noise.py \
  --model-prefix tau_trained_models/retrain_300_files \
  --num-workers 0 --gpus 0 --batch-size 10 --start-lr 1e-3 --num-epochs 100 \
  --optimizer ranger --fetch-step 0.01 --condensation \
  --log-wandb --wandb-displayname test_retrain --wandb-projectname mlpf_retrain_tau \
  --frac_cluster_loss 0 --qmin 3 --use-average-cc-pos 0.99 --tracks \
  --load-model-weights "/nfs/cms/arqolmo/GPU_train/mlpf/tau_trained_models/retrain_300_files/_epoch=82_step=178000.ckpt" \
  --train-val-split 0.99 --prefetch-factor 16
'

# Lanza el contenedor con GPU y bind de rutas necesarias
# -B asegura que /pnfs y tu working dir sean visibles dentro
apptainer exec --nv \
  -B /pnfs/ciemat.es/data/calice/arqolmo/TrainTrees -B /nfs/cms/arqolmo/GPU_train/pytorch_cmspepr/ -B /nfs:/nfs --pwd "$PWD" \
  "$IMG" bash -lc "$CMD"
