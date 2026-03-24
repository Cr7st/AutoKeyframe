import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from configs import cfg
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import lightning as L
from data.test_gen_dataset import MotionExtractorDataset, EvaluateWrapper
from data.make_dataset import make_data_loader
from lightning_modules import make_evaluator
from lightning_modules import load_lightning


if __name__ == '__main__':
    split = 'test'
    gt_dataloader = make_data_loader(cfg.fid_dataset, cfg.fid, cfg.fid_dataset_module)
    gen_dataloader = make_data_loader(cfg.test_dataset, cfg.test, cfg.test_dataset_module)
    # num_actions = gen_dataloader.dataset.action_num
    trainer = L.Trainer()
    evaluators = make_evaluator.make_evaluators(cfg, trainer)

    model = load_lightning(cfg.lightning_module, cfg.test_checkpoint, cfg.lightning_cfg)
    model.eval()
    model.l_position = np.load('exps/l_position.npy')
    model.mean = np.load(f"{cfg.result_dir}/mean.npy")
    model.std = np.load(f"{cfg.result_dir}/std.npy")
    model = model.to('cuda')
    model.evaluators = evaluators
    eval_wrapper = EvaluateWrapper()

    trainer.validate(evaluators[0], gt_dataloader)

    all_res = []
    for i in range(20):
        all_res.append(trainer.validate(model, gen_dataloader))
    # for evaluator in evaluators:
    #     print(evaluator.evaluate(gt_dataloader))
    print(all_res)
    import json

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, 'kf_fid_gt.json'), 'w') as f:
        json.dump(all_res, f, indent=4)


