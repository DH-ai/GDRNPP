<!-- ════════════════════════════════════════════════════════════════════ -->
<!--  PROJECT CONTEXT  —  added for synthetic-data-yolo-training pipeline   -->
<!-- ════════════════════════════════════════════════════════════════════ -->

> ## ⚙️ Fork / Project Context
>
> This is **not** a standalone clone. It is a **forked repo of [shanice-l/gdrnpp_bop2022](https://github.com/shanice-l/gdrnpp_bop2022)**, added here as a **git submodule** from my fork: **[DH-ai/gdrnpp_bop2022](https://github.com/DH-ai/gdrnpp_bop2022/)** basically this repo.
>
> It is used as the 6D pose-estimation stage of the parent [Synthetic Data YOLO Training & Pose Estimation](../../README.md) pipeline: BlenderProc generates BOP-format synthetic data, YOLOX/YOLO provides detections, and GDRNPP is trained/fine-tuned here to produce the final 6D pose vector. The upstream usage docs below still apply — this section only adds our project-specific context, TODO, and quick-start.
>
> ### 🎯 Current main task
>
> **Generating the dataset** (BOP-format `train_pbr` from BlenderProc, with a correct `models_info.json`). Everything else below is queued behind getting valid training data in place.
>
> ### ✅ GDRNPP To-Do (pose vector + AWS steps)
>
> - [ ] Prep BOP-format data (BlenderProc `train_pbr` + correct `models_info.json` diameter / symmetries)
> - [ ] Register the custom dataset (`ref/`, dataset loader, `dataset_factory.py`)
> - [ ] Generate test scripts for AWS
> - [ ] Run detector
> - [ ] Train detector (YOLOX) → gives bounding box per object
> - [ ] Test for GDR-Net
> - [ ] Run GDR-Net on AWS
> - [ ] Train GDR-Net pose network on the cropped detections → gives rotation + translation
> - [ ] Run depth refinement (fast or iterative) → final refined 6D pose vector
>
> ### 🚀 How to (quick start)
>
> > **TODO:** Fill in the concrete commands as the pipeline is validated. Outline of the intended flow:
> >
> > 1. **Prepare data** — point GDRNPP at the BlenderProc-generated BOP dataset under `datasets/BOP_DATASETS/<your_dataset>/` (see *Path Setting* below). Ensure `models_info.json` has correct `diameter` and `symmetries`.
> > 2. **Register the dataset** — add a ref file under `ref/`, wire up the dataset loader and `dataset_factory.py`.
> > 3. **Detector** — train / run YOLOX to produce per-object bounding boxes (`det/`).
> > 4. **Pose network** — train GDR-Net on the cropped detections (`configs/gdrn/...`) → rotation + translation.
> > 5. **Refinement** — run depth-based refinement (fast or iterative) → final refined 6D pose.
> > 6. **AWS** — see the generated AWS test/run scripts for cloud execution.
>
> _For full upstream setup (requirements, dataset structure, training/eval commands), continue reading below._

<!-- ════════════════════════════════════════════════════════════════════ -->

# GDRNPP for BOP2022

This repo provides code and models for GDRNPP_BOP2022, **winner (most of the awards) of the BOP Challenge 2022 at ECCV'22 [[slides](http://cmp.felk.cvut.cz/sixd/workshop_2022/slides/bop_challenge_2022_results.pdf)]**.


## News

[18/03/2025] Our paper has been accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)! Paper is available at [[arXiv](https://arxiv.org/pdf/2102.12145)].

## Path Setting

### Dataset Preparation
Download the 6D pose datasets from the
[BOP website](https://bop.felk.cvut.cz/datasets/) and
[VOC 2012](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)
for background images.
Please also download the  `test_bboxes` from
[ModelScope](https://www.modelscope.cn/datasets/wangg12/GDRNPP_bop22_test_bboxes), [OneDrive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/liuxy21_mails_tsinghua_edu_cn/Eq_2aCC0RfhNisW8ZezYtIoBGfJiRIZnFxbITuQrJ56DjA?e=hPbJz2) (password: groupji) or [BaiDuYunPan](https://pan.baidu.com/s/1FzTO4Emfu-DxYkNG40EDKw)(password: vp58).

The structure of `datasets` folder should look like below:
```
datasets/
├── BOP_DATASETS   # https://bop.felk.cvut.cz/datasets/
    ├──tudl
    ├──lmo
    ├──ycbv
    ├──icbin
    ├──hb
    ├──itodd
    └──tless
└──VOCdevkit
```


### Models

Download the trained models at [ModelScope](https://www.modelscope.cn/models/wangg12/GDRNPP), [Onedrive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/liuxy21_mails_tsinghua_edu_cn/EgOQzGZn9A5DlaQhgpTtHBwB2Bwyx8qmvLauiHFcJbnGSw?e=EZ60La) (password: groupji) or [BaiDuYunPan](https://pan.baidu.com/s/1LhXblEic6pYf1i6hOm6Otw)(password: 10t3) and put them in the folder `./output`.


## Requirements
* Ubuntu 18.04/20.04, CUDA 10.1/10.2/11.6, python >= 3.7, PyTorch >= 1.9, torchvision
* Install `detectron2` from [source](https://github.com/facebookresearch/detectron2)
* `sh scripts/install_deps.sh`
* Compile the cpp extensions for 
1. `farthest points sampling (fps)`
2. `flow`
3. `uncertainty pnp`
4. `ransac_voting`
5. `chamfer distance`
6. `egl renderer`

    ```
    sh ./scripts/compile_all.sh
    ```

## Detection

We adopt yolox as the detection method. We used stronger data augmentation and ranger optimizer.

### Training 

Download the pretrained model at [Onedrive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/liuxy21_mails_tsinghua_edu_cn/EkCTrRfHUZVEtD7eHwLkYSkBCTXlh9ekDteSzK6jM4oo-A?e=m0aNCy) (password: groupji) or [BaiDuYunPan](https://pan.baidu.com/s/1AU7DGCmZWsH9VgQnbTRjow)(password: aw68) and put it in the folder `pretrained_models/yolox`. Then use the following command:

`./det/yolox/tools/train_yolox.sh <config_path> <gpu_ids> (other args)`

### Testing 

`./det/yolox/tools/test_yolox.sh <config_path> <gpu_ids> <ckpt_path> (other args)`

## Pose Estimation

The difference between this repo and GDR-Net (CVPR2021) mainly includes:

* Domain Randomization: We used stronger domain randomization operations than the conference version during training.
* Network Architecture: We used a more powerful backbone Convnext rather than resnet-34,  and two  mask heads for predicting amodal mask and visible mask separately.
* Other training details, such as learning rate, weight decay, visible threshold, and bounding box type.

### Training 

`./core/gdrn_modeling/train_gdrn.sh <config_path> <gpu_ids> (other args)`

For example:

`./core/gdrn_modeling/train_gdrn.sh configs/gdrn/ycbv/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ycbv.py 0`

### Testing 

`./core/gdrn_modeling/test_gdrn.sh <config_path> <gpu_ids> <ckpt_path> (other args)`

For example:

`./core/gdrn_modeling/test_gdrn.sh configs/gdrn/ycbv/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ycbv.py 0 output/gdrn/ycbv/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ycbv/model_final_wo_optim.pth`

## Pose Refinement

We utilize depth information to further refine the estimated pose.
We provide two types of refinement: fast refinement and iterative refinement.

For fast refinement, we compare the rendered object depth and the observed depth to refine translation.
Run

`./core/gdrn_modeling/test_gdrn_depth_refine.sh <config_path> <gpu_ids> <ckpt_path> (other args)`

For iterative refinement, please checkout to the [pose_refine branch](https://github.com/shanice-l/gdrnpp_bop2022/tree/pose_refine) for details.

## Citing GDRNPP

If you use GDRNPP in your research, please use the following BibTeX entries.

```BibTeX
@article{liu2025gdrnpp,
  title     = {GDRNPP: A Geometry-guided and Fully Learning-based Object Pose Estimator},
  author    = {Liu, Xingyu and Zhang, Ruida and Zhang, Chenyangguang and Wang, Gu and Tang, Jiwen and Li, Zhigang and Ji, Xiangyang},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year      = {2025},
}

@InProceedings{Wang_2021_GDRN,
    title     = {{GDR-Net}: Geometry-Guided Direct Regression Network for Monocular 6D Object Pose Estimation},
    author    = {Wang, Gu and Manhardt, Fabian and Tombari, Federico and Ji, Xiangyang},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {16611-16621}
}
```

