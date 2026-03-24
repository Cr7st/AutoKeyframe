# AutoKeyframe: Autoregressive Keyframe Generation for Human Motion Synthesis and Editing

### [Project Page](https://cr7st.github.io/AutoKeyframe.github.io/) | [Paper](https://dl.acm.org/doi/10.1145/3721238.3730664)

![teaser](assets/representative.jpg)

Official implementation of our **SIGGRAPH 2025** paper.

## Overview

AutoKeyframe is a diffusion-based autoregressive keyframe generation system for character animation. It uses DDIM with Transformer encoders to generate motion keyframes conditioned on trajectory, action class, and optional joint-position hints.

## Requirements

Python 3.8+ with CUDA support. Install dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies: `torch`, `lightning`, `numpy`, `scipy`, `omegaconf`, `tqdm`, `matplotlib`

## Dataset

This project uses the [LaFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) motion capture dataset. We provide the pre-processed dataset (keyframes extracted from motion sequences via a reinforcement-learning-based selector as described in the paper).

Download the processed dataset and place the files under `datasets/lafan1_keyframes/`:

| Dataset | Link |
|---------|------|
| Processed LaFAN1 keyframes | [Google Drive](https://drive.google.com/file/d/1jMArouG2Fx92mL84BycwYHHriXyPmmxE/view?usp=drive_link) |

We also include 13 sample sequences in `sample_data/` for quick testing without the full dataset.

## Checkpoints

Download pretrained checkpoints and place them in the `exps/` directory:

| Model | Description | Link |
|-------|-------------|------|
| KeyframeGenerator | Main keyframe generation model | [Google Drive](https://drive.google.com/file/d/1XEWOU_OHNAsFcs5gD_TjeqVh6785asIc/view?usp=drive_link) |
| MotionFID | Feature extractor for FID evaluation | [Google Drive](https://drive.google.com/file/d/1BDWq0mxl2qxGN_pd4ua79SryTOYT3iHH/view?usp=drive_link) |

Expected directory structure:
```
exps/
├── KeyframeGenerator/
│   └── <experiment_name>/
│       ├── checkpoint/
│       │   └── <checkpoint>.ckpt
│       ├── config.yaml
│       ├── mean.npy
│       └── std.npy
├── MotionFID/
│   └── <experiment_name>/
│       ├── checkpoint/
│       │   └── <checkpoint>.ckpt
│       └── config.yaml
└── l_position.npy
```

The `config.yaml` in each experiment folder is the frozen training configuration used for inference. This is separate from the configs in `configs/` which are used to launch training.

## Training

Train the keyframe generation model:

```bash
python scripts/train.py --cfg_file configs/KFG.yaml
```

Override config values via CLI:

```bash
python scripts/train.py --cfg_file configs/KFG.yaml train.batch_size=32 train.lr=5e-5
```

Training outputs (checkpoints, config, normalization stats) are saved to `exps/<task>/<exp_name>_<version>/`.

## Inference

### Batch Test

Generate keyframes for all test sequences specified in the config:

```bash
python scripts/run.py --type test --cfg_file exps/KeyframeGenerator/<experiment_name>/config.yaml
```

The test sequences and checkpoint are specified by `test_path` and `test_checkpoint` in the config. Results are saved to `exps/KeyframeGenerator/<experiment_name>/output/`.

### Single Test with Manual Control

For more flexible control over individual sequences:

```bash
python scripts/run.py --type single_test --cfg_file exps/KeyframeGenerator/<experiment_name>/config.yaml
```

In `run_single_test()`, you can directly specify:
- **Input data**: change the `.npz` file path
- **Joint hints**: set specific joint positions at target frames via `hint` and `hint_mask`
- **Keyframe timing**: customize which frames to generate keyframes at via `keyframes_list` (minimum interval of 9 frames)
- **Action class**: modify the `action` variable to change the motion category

Example of specifying joint hints:

```python
hint_mask[:, 111, 13, :] = 1   # frame 111, joint 13
hint_mask[:, 111, 21, :] = 1   # frame 111, joint 21
hint_mask[:, 139, [17, 21], :] = 1  # frame 139, joints 17 and 21
model.enable_grad_guide = True
```

### FID Evaluation

Evaluate generation quality using FID, accuracy, and diversity metrics (averaged over 20 runs):

```bash
python scripts/test_fid.py --cfg_file exps/KeyframeGenerator/<experiment_name>/config.yaml
```

This requires the MotionFID feature extractor checkpoint.

## Visualization

We provide a Blender plugin for visualizing generated keyframes.

1. Install `MotionKeyframes.zip` as a Blender addon
2. Go to **File → Import → Motion Keyframes (.npz)**
3. Select a generated `.npz` file to import

## Note on MIB

The Motion In-Betweening (MIB) component described in the paper is **not included** in this release. This repository only covers the keyframe generation module. For MIB, we recommend using [Two-Stage Transformer](https://github.com/Piotrekszmel/Two-Stage-Transformer-for-Human-Motion-In-Betweening) as a compatible alternative.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zheng2025autokeyframe,
  title={AutoKeyframe: Autoregressive Keyframe Generation for Human Motion Synthesis and Editing},
  author={Zheng, Yicong and Chen, Kecheng and Fang, Qiang and Lin, Junhao and Liao, Jingbo and Zhang, Jian},
  booktitle={Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers '25},
  series = {SIGGRAPH Conference Papers '25},
  year={2025},
  doi={10.1145/3721238.3730664}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
