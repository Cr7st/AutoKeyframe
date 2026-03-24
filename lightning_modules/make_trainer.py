import lightning as L
from lightning.pytorch import loggers, callbacks
from omegaconf import OmegaConf


def make_callback(callback_cfg):
    callback_type = callback_cfg.type
    kwargs = callback_cfg.get('kwargs', {})
    
    if callback_type == 'TQDMProgressBar':
        callback = callbacks.TQDMProgressBar(**kwargs)
    elif callback_type == 'ModelCheckpoint':
        callback = callbacks.ModelCheckpoint(**kwargs)
    else:
        raise NotImplementedError
    return callback

def make_logger(cfg):
    if cfg.trainer.logger == 'wandb':
        logger = loggers.WandbLogger(project=cfg.task, name=cfg.exp_name, version=f"{cfg.exp_name}_{cfg.version}")
    elif cfg.trainer.logger == 'tensorboard':
        logger = loggers.TensorBoardLogger(name=cfg.exp_name, version=f"{cfg.exp_name}_{cfg.version}", save_dir=cfg.result_dir)
    elif cfg.trainer.logger is None:
        return None
    else:
        raise NotImplementedError
    return logger

def make_trainer(cfg):
    logger = make_logger(cfg)

    callbacks_cfg = cfg.trainer.callbacks
    trainer_callbacks = []
    for callback_cfg in callbacks_cfg:
        callback = make_callback(callback_cfg)
        trainer_callbacks.append(callback)

    trainer = L.Trainer(
        logger=logger,
        callbacks=trainer_callbacks,
        **cfg.trainer.kwargs
    )

    return trainer