import torch


def make_optimizer(train_cfg, net):
    if train_cfg.optim == 'AdamW':
        optimizer = torch.optim.AdamW(net.trainable_params, lr=train_cfg.lr)
    else:
        raise NotImplementedError(f"Optimizer {train_cfg.optim} not implemented")
    return optimizer

def make_scheduler(scheduler_cfg, optimizer):
    if scheduler_cfg.type == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, scheduler_cfg.gamma)
    elif scheduler_cfg.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, scheduler_cfg.step_size, scheduler_cfg.gamma)
    else:
        raise NotImplementedError(f"Scheduler {scheduler_cfg.type} not implemented")
    return scheduler