import torch
from datasets.vs_ds import VSDataset
from torch.utils.data import DataLoader, random_split

def GetLoader(path, batch_size, num_workers, pin_memory = True, split_ratio = [0.8, 0.2]):
    dataset = VSDataset(dataset_path = path, mode = 'train')
    train_ds, val_ds = random_split(dataset, split_ratio,
                                    generator = torch.Generator().manual_seed(0))
    
    val_ds.dataset.mode = 'test'

    TrainLoader = DataLoader(dataset = train_ds,
                             batch_size = batch_size,
                             shuffle = True,
                             num_workers = num_workers,
                             pin_memory = pin_memory)
    
    ValLoader = DataLoader(dataset = val_ds,
                           batch_size = batch_size,
                           shuffle = False,
                           num_workers = num_workers,
                           pin_memory = pin_memory)
    
    return TrainLoader, ValLoader
