# CURL: Multi-view 3D Hair Reconstruction via Adaptive Gaussian Ellipsoids under Varying Illumination

[**Paper**](coming soon) | [**Project Page**](coming soon)

This repository contains an official implementation of CURL, a strand-based hair reconstruction approach for monocular videos, incorporating low-light enhancement and vision-language model-based illumination quality and hair complexity perception.

## Getting started

### 1. Gaussian Haircut Environment
Follow the instructions at https://github.com/eth-ait/GaussianHaircut to set up the environment.

### 2. 3D Gaussian Splatting Environment
Follow the instructions at https://github.com/graphdeco-inria/gaussian-splatting to set up the environment.

### 3. Q-Align Environment
Follow the instructions at https://github.com/q-future/q-align to set up the environment.

### 4. HVI Environment
Follow the instructions at https://github.com/fediory/hvi-cidnet to set up the environment.

### 5. Blender
Install Blender version 3.6.19.

## Reconstruction

We recommend familiarizing yourself with the complete Gaussian Haircut pipeline before running CURL.

### 1. Data Preparation
Create a new directory and rename your monocular video to `raw.mp4`, then place it in that directory.

### 2. Low-Light Enhancement, Histogram Equalization, Q-Align Quality Assessment, and Qwen2.5-VL Guided Densification
Launch Qwen2.5-VL and set `--allowed-local-media-path` to your data directory.

Set all command switches in `run2_1_texture_1.sh` to `true`. Note that certain steps should remain `false` as indicated in the comments.

Set the directory in `./aaaa_1.sh` to your data directory.
```shell
./aaaa_1.sh
```

### 3. 3DGS Reconstruction, CURL Reconstruction, and Visualization
Set all command switches in `run11_all_1.sh` to `true`. Note that the three commands in `calc_edge_loss.py` should remain `false`. You may optionally disable `--sobel_loss` in step 3.2 and step 1.3.

Set the CURL data directory, 3DGS output point cloud path, and 3DGS output directory in `dddd_1.sh`.
```shell
./dddd_1.sh
```

### 4. Quantitative Evaluation
Set the directories in `./cccc_resized.sh`. Resize the original image directory `curves_reconstruction/stage3/raw_frames` and the mask directory `masks/hair`, saving them as `raw_frames_resized` and `masks_hair_resized` respectively.
```shell
./cccc_resized.sh
```
Set the data directory in `eeee_psnr.sh`.
```shell
./eeee_psnr.sh
```

## License

This project is built upon [Gaussian Haircut](https://github.com/eth-ait/GaussianHaircut). For terms and conditions, please refer to their respective licenses. The rest of the code is distributed under CC BY-NC-SA 4.0.

If this work is helpful to your research, please consider citing the following papers.

## Citation

```
@inproceedings{zakharov2024gh,
   title = {CURL: Multi-view 3D Hair Reconstruction via Adaptive Gaussian Ellipsoids under Varying Illumination},
   author = {Liu Jingyi+，Zhang Jingsong+，Ma Jian*，Li Kun},
   booktitle = {Journal of Image and Graphics (JIG)},
   year = {2026}
} 
```

## Links

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting): core 3D Gaussian representation and rendering

- [Gaussian Haircut](https://github.com/eth-ait/GaussianHaircut): overall hair reconstruction pipeline, strand prior and geometry optimization

- [HVI-CIDNet](https://github.com/fediory/hvi-cidnet): low-light enhancement

- [Q-Align](https://github.com/q-future/q-align): illumination quality assessment for view selection

- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL): illumination quality and hair complexity perception-based vision-language model guidance for adaptive densification