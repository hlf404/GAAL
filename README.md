# Harmonized Tabular-Image Fusion via Gradient-Aligned Alternating Learning
This is the official code for the paper "Harmonized Tabular-Image Fusion via Gradient-Aligned Alternating Learning".

**"Harmonized Tabular-Image Fusion via Gradient-Aligned Alternating Learning"**

**Authors: [Longfei Huang](https://hlf404.github.io/), and [Yang Yang](http://home.njustkmg.cn:4056/Introduction-cn.html)**

**Accepted by: ICME 2026**

**[[arXiv](https://arxiv.org/abs/2604.01579)]** 

<p align="center">
<img src="image/arch.png" alt="wild_settings" width="100%" align=center />
</p>

## ✨ Motivation

Tabular-image fusion methods may be hindered by gradient conflicts between modalities, misleading the optimization of the unimodal learner. 

## 📖 Overview

We propose GAAL to address this issue by aligning modality gradients. GAAL adopts an alternating unimodal learning and shared classifier to decouple the multimodal gradient and facilitate interaction. Furthermore, we design uncertainty-based cross-modal gradient surgery to selectively align cross-modal gradients, thereby steering the shared parameters to benefit all modalities.

## 🚀 Quick Start

**Requirements**

* python 3.8
* pytorch 1.13.1
* torchaudio 0.13.1
* torchvision 0.14.1 
* torch-lightning 1.6.4
* pl-bolts 0.5.0
* opencv 4.9.0.80
* numpy 1.24.1

## Dataset

Download Dataset: 

The DVM cars dataset is open-access and can be found [here](https://deepvisualmarketing.github.io/).

The CelebA dataset is open-access and can be found [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

The SUN dataset is open-access and can be found [here](https://groups.csail.mit.edu/vision/SUN/hierarchy.html).

## Training

For training, we provide hyper-parameter settings in `/configs/configs.yaml`.

Your data should be constructed in `/configs/dataset`.

### Running

```bash
$ CUDA_VISIBLE_DEVICES=0 python run.py pretrain=False test=False evaluate=True test_and_eval=True datatype=imaging_and_tabular dataset={YOUR_DATASET}
```

## 📘 Citation

If you find this work useful, consider giving this repository a star ⭐️ and citing 📑 our paper as follows:

```bibtex
@inproceedings{Huang2026gaal,
  title={Harmonized Tabular-Image Fusion via Gradient-Aligned Alternating Learning},
  author={Longfei Huang and Yang Yang},
  booktitle={ICME},
  year={2026},
}
```

## Acknowledgment

We thank the following repos providing helpful components/functions in our work.

- [MMCL](https://github.com/paulhager/MMCL-Tabular-Imaging)
- [TIP](https://github.com/siyi-wind/TIP)
- [CHARMS](https://github.com/RyanJJP/CHARMS)
