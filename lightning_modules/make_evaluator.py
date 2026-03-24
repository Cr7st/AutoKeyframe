import importlib
from lightning_modules import load_lightning
from data import make_data_loader
import os.path as osp
import numpy as np


def make_evaluator(cfg, evaluator_cfg, trainer):
    module_path, module_name = evaluator_cfg.module.rsplit(".", 1)
    kwargs = evaluator_cfg.get('kwargs', {})

    if module_name == 'FIDModule':
        if kwargs.get('fid_cfg', None) is not None:
            dataloader = make_data_loader(cfg.fid_dataset, cfg.fid, cfg.fid_dataset_module)
            lightning_cfg = kwargs.fid_cfg
        elif kwargs.get('mib_cfg', None) is not None:
            dataloader = make_data_loader(cfg.mib_dataset, cfg.mib, cfg.mib_dataset_module)
            lightning_cfg = kwargs.mib_cfg
            cfg.run_mib = True
        evaluator = load_lightning(evaluator_cfg.module, kwargs.path, lightning_cfg)
        evaluator.mean = np.load(osp.join(kwargs.stats_path, 'mean.npy'))
        evaluator.std = np.load(osp.join(kwargs.stats_path, 'std.npy'))
        #TODO make keyframe fid's mean and std
        trainer.predict(evaluator, dataloaders=dataloader)
    elif module_name == 'PenetrationEvaluator' or module_name == 'FootSkateEvaluator' or module_name == 'TrajErrorEvaluator':
        evaluator = importlib.import_module(module_path).__dict__[module_name](**kwargs)
    else:
        raise NotImplementedError(f'{module_name} is not implemented')
    return evaluator


def make_evaluators(cfg, trainer):
    evaluators = []
    for evaluator_cfg in cfg.evaluators_cfg:
        evaluator = make_evaluator(cfg, evaluator_cfg, trainer)
        evaluators.append(evaluator)
    return evaluators