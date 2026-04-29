# ShadowRemovalRefine
Shadow Removal Refinement via Material-Consistent Shadow Edges.

Shilin Hu, Hieu Le, ShahRukh Athar, Sagnik Das, Dimitris Samaras. WACV 2025. [[link]](https://openaccess.thecvf.com/content/WACV2025/papers/Hu_Shadow_Removal_Refinement_via_Material-Consistent_Shadow_Edges_WACV_2025_paper.pdf)

## Proposed Benchmark Testset
400 pairs of test images and annotated edges. [GoogleDrive](https://drive.google.com/drive/folders/1nUyo95g80kkRBOiU2YfmQUi50-TIUTGm?usp=sharing)

## Getting Started
```sh
conda env create -f environment.yml
```

## Fine-tuned SAM
You can directly use the checkpoint. [GoogleDrive](https://drive.google.com/drive/folders/1nUyo95g80kkRBOiU2YfmQUi50-TIUTGm?usp=sharing)
Or train your own by
```sh
python train_finetunedSAM.py
```

## Run ShadowRemovalRefine
First, change `line83` in `refine.py` to your data path.

Then, run `test_time_adaptation.sh`

Our results are on [GoogleDrive](https://drive.google.com/drive/folders/1nUyo95g80kkRBOiU2YfmQUi50-TIUTGm?usp=sharing)

## Baseline Models
Check [ShadowFormer](https://github.com/GuoLanqing/ShadowFormer) and [SP+M-Net](https://github.com/cvlab-stonybrook/SID) for baseline models.
