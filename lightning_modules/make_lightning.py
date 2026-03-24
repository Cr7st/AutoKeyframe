import os
import importlib

def make_lightning(cfg):
    lightning_cfg = cfg.lightning_cfg
    module = cfg.lightning_module
    module_path, module_name = module.rsplit(".", 1)
    lightning = importlib.import_module(module_path).__dict__[module_name](**lightning_cfg, train_cfg = cfg.train)
    return lightning

def load_lightning(load_module, load_path, lightning_cfg = {}):
    module_path, module_name = load_module.rsplit(".", 1)

    lightning = importlib.import_module(module_path).__dict__[module_name].load_from_checkpoint(load_path, **lightning_cfg)
    return lightning