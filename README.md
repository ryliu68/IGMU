# [CCS'25] IGMU: Rethinking Machine Unlearning in Image Generation Models

**WARNING: This repository contains model outputs that may be offensive in nature.**

## Introduction

This repository contains the PyTorch implementation for the ACM CCS 2025 paper [IGMU: Rethinking Machine Unlearning in Image Generation Models](https://arxiv.org/abs/2506.02761).

## Update (multi_multiC, 2025-08-08)

To support a wider range of unlearning tasks, we have introduced a new checker **`multi_multiC`**.  
Unlike the previous **`bi_multiC`**, which was limited to binary classification, the new **`multi_multiC`** supports multiple unlearning tasks:
- **NSFW**: Supports detection of Nudity (now).
- **Object**: Supports detection of 10 categories from *Imagenette*.
- **Style**: Directly supports detection of 129 artist styles from *WikiArt*.

We welcome everyone to try out the new version.

---

## Example Usage

You can evaluate your unlearning results with the provided script `eval_unlearning.py`.

```bash
python eval_unlearning.py --indicator multi_multiC --concept "Nudity" --batch-size 5
```

### Arguments
- `--concept`: The concept to evaluate.  
  Supported:
  - `"Nudity"`
  - `"Style"` (129 classes, see `utils/name_to_id.py` for task names)
  - `"Object"` (10 classes, see `utils/name_to_id.py` for task names)
- `--indicator`: Choose from:
  - `bi_multiC` (binary classification)
  - `multi_multiC` (multi-class classification)
- `--batch-size`: Batch size for evaluation.


## Getting Started (Benchmarking, 20250606)

```bash
# Create and activate the Conda environment
conda env create --file environment.yaml
conda activate igmu

# Run benchmarking
# We have included some demo images related to Nudity unlearning as examples. Please refer to "dataset/Benchmarking_images_demo"
python -W ignore benchmark.py --evaluation-aspect forgetting|fid|lpips|yolo|CSDR

# evaluate fid and lpips for 
# We have included some demo images related to Church unlearning as examples. Please refer to "dataset/Benchmarking_images_demo"
python -W ignore benchmark.py --object True --evaluation-aspect fid|lpips

```
### NOTE

By default, `/home/mrliu/miniconda3/envs/igmu/lib/python3.12/site-packages/torchmetrics/functional/multimodal/clip_score.py` returns the mean CLIP score for a batch.  
If you wish to obtain the individual CLIP scores for all images in a batch, please comment out line 165 in `clip_score.py`:

```python
model, processor = _get_clip_model_and_processor(model_name_or_path)
device = images.device if isinstance(images, Tensor) else images[0].device
score, _ = _clip_score_update(images, text, model.to(device), processor)
# score = score.mean(0) # line 165
return torch.max(score, torch.zeros_like(score))
```

## Citation

If you find this work helpful, we would greatly appreciate it if you could cite it.

```bibtex
@inproceedings{liu2025igmu,
  title = {Rethinking Machine Unlearning in Image Generation Models},
  author = {Liu, Renyang and Feng, Wenjie and Zhang, Tianwei and Zhou, Wei and Cheng, Xueqi and Ng, See-Kiong},
  booktitle = {ACM Conference on Computer and Communications Security (CCS)},
  organization = {ACM},
  year = {2025},
}

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
