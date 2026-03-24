import os
import sys
import pprint
import argparse
import numpy as np

from omegaconf import OmegaConf
from utils.debug_util import mkdir


cfg = OmegaConf.load('./configs/base_cfg.yaml')

# reprocess.py
cfg.reprocess = True
cfg.view = False
# cfg.mib = True
cfg.mib_path = ""  # Path to MIB model (not yet open-sourced)

cfg.wandb_offline = True

def set_seed(seed = 42):
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_cfg(cfg, args):
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])

    cfg.type = args.type

    cfg.original_result_dir = args.save_dir 
    cfg.result_dir = os.path.join(cfg.original_result_dir, cfg.task, f"{cfg.exp_name}_{cfg.version}")
    cfg.checkpoint_dir = os.path.join(cfg.result_dir, 'checkpoint')
    cfg.output_dir = os.path.join(cfg.result_dir, 'output')
    if ".npz" not in cfg.eval_motion_name:
        cfg.eval_motion_name += ".npz"

    cfg.cfg_file = os.path.join(cfg.result_dir, 'config.yaml')

    if args.resume_from is not None:
        cfg.resume_from_checkpoint = args.resume_from

    if cfg.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    mkdir(cfg.result_dir)
    mkdir(cfg.checkpoint_dir)
    mkdir(cfg.output_dir)

    print(f'The result will be saved in: {cfg.result_dir}')
    print(f'The checkpoint will be saved in: {cfg.checkpoint_dir}')
    print(f'The output will be saved in: {cfg.output_dir}')

    cfg.fix_seed = cfg.get('fix_seed', None)
    if cfg.fix_seed is not None: set_seed(cfg.fix_seed)


def make_cfg(args):
    current_cfg = OmegaConf.load(args.cfg_file)
    cfg.merge_with(current_cfg)
    cfg.merge_with_dotlist(args.opts)
    
    parse_cfg(cfg, args)

    if 'train' in sys.argv[0]:
        with open(cfg.cfg_file, 'w') as f:
            OmegaConf.save(cfg, f)

    return cfg

parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="configs/KFG.yaml", type=str)
parser.add_argument("--type", type=str, default="vis")
parser.add_argument("--resume_from", default=None, type=str)
parser.add_argument("--save_dir", default='exps', type=str)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()

cfg = make_cfg(args)
