# ShadowRemovalRefine
Shadow Removal Refinement via Material-Consistent Shadow Edges
Shilin Hu, Hieu Le, ShahRukh Athar, Sagnik Das, Dimitris Samaras. WACV 2025.

## Getting Started
```sh
conda env create -f environment.yml
```

## Fine-tuned SAM
You can directly use the checkpoint.
Or train your own by
```sh
python train_finetunedSAM.py
```

## Run ShadowRemovalRefine
First, change `line83` in `refine.py` to your data path.

Then, run `test_time_adaptation.sh`

## Baseline Models
Check [ShadowFormer](https://github.com/GuoLanqing/ShadowFormer) and [SP+M-Net](https://github.com/cvlab-stonybrook/SID) for baseline models.
