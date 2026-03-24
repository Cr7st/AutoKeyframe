from .scheduling_ddpm import DDPMScheduler
from .scheduling_ddim import DDIMScheduler

def make_diffusion_scheduler(scheduler_cfg):
    if scheduler_cfg.type == 'DDPM':
        return DDPMScheduler(**scheduler_cfg.kwargs)
    elif scheduler_cfg.type == 'DDIM':
        return DDIMScheduler(**scheduler_cfg.kwargs)
    else:
        raise NotImplementedError(f'{scheduler_cfg.type} is not implemented')