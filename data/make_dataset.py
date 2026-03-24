import importlib
from torch.utils.data import DataLoader

def make_dataset(cfg, module):
    dataset_cfg = cfg
    module_path, module_name = module.rsplit(".", 1)
    
    dataset = importlib.import_module(module_path).__dict__[module_name](**dataset_cfg)
    return dataset


def make_data_loader(dataset_cfg, dataloader_cfg, module):
    dataset = make_dataset(dataset_cfg, module)
    batch_size = dataloader_cfg.batch_size
    num_workers = dataloader_cfg.get('num_workers', 0)

    if dataset_cfg.split == 'train':
        shuffle = True
    else:
        shuffle = False

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader
