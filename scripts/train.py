import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import lightning as L
from configs import cfg
from data import make_data_loader
from lightning_modules import make_evaluators, make_lightning, make_trainer

train_dataloader = make_data_loader(cfg.train_dataset, cfg.train, cfg.train_dataset_module)
test_dataloader = make_data_loader(cfg.test_dataset, cfg.test, cfg.test_dataset_module)

trainer = make_trainer(cfg)

model = make_lightning(cfg)

if cfg.get('evaluators_cfg', None) is not None:
    model.evaluators = make_evaluators(cfg, trainer)


trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader, ckpt_path=cfg.get('resume_from_checkpoint', None))
