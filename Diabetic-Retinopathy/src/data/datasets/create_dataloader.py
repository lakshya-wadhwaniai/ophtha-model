import copy
import time

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader

import src.data.datasets as datasets_module
import src.utils.collation as collation_module
from src.data.processing.augmentations import Augmentations
import src.utils.sampler as sampler_module

def create_dataloader(data_cfg, phase):
    """
    Returns dataloader corresponding to the Dataset object for fs2_data.
    """
    datasets_list = []
    start = time.time()
    print('Preparing {} dataloader...'.format(phase))
    phase_flag = False
    for data_source in data_cfg['datasets']:
        phase_flag = True if phase == 'train' else False
        transform = Augmentations(cfg = data_cfg['augmentations'], use_augmentation = phase_flag,
                                size = data_cfg['imgsize'])
        dataset_class = getattr(datasets_module, data_cfg[data_source]['Name'])
        dataset = dataset_class(transform=transform, phase=phase, 
                            **data_cfg[data_source]['dataset_params'])
        datasets_list.append(dataset)
    end = time.time()
    data_cfg_copy = copy.deepcopy(data_cfg)
    if data_cfg['dataloader_params']['collate_fn'] is not None:
        data_cfg_copy['dataloader_params']['collate_fn'] = getattr(
            collation_module, data_cfg['dataloader_params']['collate_fn'])
    data_cfg_copy['dataloader_params']['batch_size'] = data_cfg[
        'dataloader_params'][f'{phase}_batch_size']
    del data_cfg_copy['dataloader_params']['train_batch_size']
    del data_cfg_copy['dataloader_params']['val_batch_size']
    del data_cfg_copy['dataloader_params']['test_batch_size']

    if data_cfg['sampler']:
        del data_cfg_copy['sampler_params']['num_samples_ratio']
    merged_dataset = ConcatDataset(datasets_list)

    # print(data_cfg_copy)

    if (data_cfg['sampler'] and phase_flag):
        sampler_class = getattr(sampler_module, data_cfg['sampler'])
        sampler = sampler_class(merged_dataset, num_samples = (data_cfg['sampler_params']['num_samples_ratio'] * len(merged_dataset)), **data_cfg_copy['sampler_params'])
        dataloader = DataLoader(dataset = merged_dataset, sampler = sampler, **data_cfg_copy['dataloader_params'])
        print('Sampling dataset to {}X'.format(data_cfg['sampler_params']['num_samples_ratio']))
    
    else :
        dataloader = DataLoader(dataset = merged_dataset, **data_cfg_copy['dataloader_params'])
    print('> Time to create dataset object : {:.4f} sec'.format(end - start))
    print('> Time to create dataloader object : {:.4f} sec'.format(
        time.time() - end))

    return dataloader
