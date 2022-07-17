#!/bin/sh

OUT_DIR="./temp1"

# depth experiment
python train_wrm.py --outdir "$OUT_DIR" --n_hidden 2 --width 2 --num_classes 2 --input_dim 2 --gamma 2
python train_wrm.py --outdir "$OUT_DIR" --n_hidden 5 --width 2 --num_classes 2 --input_dim 2 --gamma 20
python train_wrm.py --outdir "$OUT_DIR" --n_hidden 10 --width 2 --num_classes 2 --input_dim 2 --gamma 20
python train_wrm.py --outdir "$OUT_DIR" --n_hidden 20 --width 2 --num_classes 2 --input_dim 2 --gamma 20

## width experiment
python train_wrm.py --outdir "$OUT_DIR" --n_hidden 2 --width 4 --num_classes 2 --input_dim 2 --gamma 20
python train_wrm.py --outdir "$OUT_DIR" --n_hidden 2 --width 8 --num_classes 2 --input_dim 2 --gamma 20
python train_wrm.py --outdir "$OUT_DIR" --n_hidden 2 --width 16 --num_classes 2 --input_dim 2 --gamma 20