import os
import importlib

def make_model(model_cfg):
    module = model_cfg.module
    module_path, module_name = module.rsplit(".", 1)
    model = importlib.import_module(module_path).__dict__[module_name](**model_cfg.kwargs)
    return model