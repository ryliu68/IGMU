# [CCS'25] IGMU: Rethinking Machine Unlearning in Image Generation Models

**WARNING: This repository contains model outputs that may be offensive in nature.**

## Introduction

This repository contains the PyTorch implementation for the ACM CCS 2025 paper [IGMU: Rethinking Machine Unlearning in Image Generation Models]().


## Getting Started

```bash
# Create and activate the Conda environment
conda env create --file environment.yaml
conda activate aigc

# Run benchmarking
python -W ignore benchmark.py --evaluation-aspect forggeting|fid|lpips|yolo|CSDR

# evaluate fid and lpips for object
python -W ignore benchmark.py --object True --evaluation-aspect  forggeting|fid|lpips

```


### Acknowledgements

We extend our gratitude to the following repositories for their contributions and resources:

- [ESD](https://github.com/rohitgandikota/erasing)
- [FMN](https://github.com/SHI-Labs/Forget-Me-Not)
- [SPM](https://github.com/Con6924/SPM)
- [AdvUnlearn](https://github.com/OPTML-Group/AdvUnlearn)
- [MACE](https://github.com/shilin-lu/mace)
- [RECE](https://github.com/CharlesGong12/RECE)
- [DoCo](https://github.com/yongliang-wu/DoCo)
- [Receler](https://github.com/jasper0314-huang/Receler)
- [ConceptPrune](https://github.com/ruchikachavhan/concept-prune)
- [UCE](https://github.com/rohitgandikota/unified-concept-editing)

- [NudeNet](https://github.com/notAI-tech/NudeNet)
- [UnlearnDiffAtk](https://github.com/OPTML-Group/Diffusion-MU-Attack)
- [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch)


Their works have significantly contributed to the development of our work.


---

## Citation

If you find this work or code useful for your research, please cite our paper:

```bibtex
@inproceedings{liu2025igmu,
  title = {Rethinking Machine Unlearning in Image Generation Models},
  author = {Liu, Renyang and Feng, Wenjie and Zhang, Tianwei and Zhou, Wei and Cheng, Xueqi and Ng, See-Kiong},
  booktitle = {ACM Conference on Computer and Communications Security (CCS)},
  organization = {ACM},
  year = {2025},
}
```

Thank you for your interest and support!
