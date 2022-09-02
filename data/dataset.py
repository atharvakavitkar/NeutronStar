import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

#user defined modules
from .add_noise import add_noise_np

class Star_Source(Dataset):

    def __init__(self, data_config):
        """
        Args:
            data_config (dict): Data Loading config
        """
        super(Star_Source, self).__init__()
        self.data_config = data_config
        
        if self.data_config['train'] == True:
            self.train()
        else:
            self.test()


    def train(self):
        
        data_source = os.path.join(self.data_config['dataset_dir'],'train.pkl')
        self.data = pd.read_pickle(data_source)
        
        
    def test(self):
        
        data_source = os.path.join(self.data_config['dataset_dir'],'test.pkl')
        self.data = pd.read_pickle(data_source)
        nH,logTeff,dist = add_noise_np(self.data_config,self.data['nH'],self.data['logTeff'],self.data['dist'])
        self.data.loc[:,'nH'] = nH
        self.data.loc[:,'logTeff'] = logTeff
        self.data.loc[:,'dist'] = dist
    
    
    def __len__(self):
        
        return len(self.data.index)


    def __getitem__(self, idx):

        xi = torch.tensor(self.data.iloc[idx,4:])
        yi = torch.tensor(self.data.iloc[idx,:2])

        return xi.float(), yi.float()
        

class Star_Loader(LightningDataModule):
    
    def __init__(self,data_config):
        super(Star_Loader, self).__init__()

        self.data_config = data_config
        self.dataset = Star_Source(self.data_config)

        if self.data_config['train'] == True:
            self.trainval_split()
        else:
            self.test_dataset = self.dataset

        self.batch_size = self.data_config['batch_size']
        self.num_workers = self.data_config['num_workers']
        self.input_size = self.dataset[0][0].shape[-1]
        self.output_size = self.dataset[0][1].shape[-1]
    
    def __len__(self):
        return len(self.dataset)
    
    def trainval_split(self):
        
        self.val_split = self.data_config['validation_split']
        self.random_state = self.data_config['seed_value']
        self.val_len = None
        self.train_len = None
        self.dataset_len = None

        self.train_dataset = None
        self.val_dataset = None
        self.dataset_len = len(self.dataset)
        self.train_len = int(self.dataset_len * (1 - self.val_split))
        self.val_len = int(self.train_len * self.val_split)
        self.train_len = int(self.train_len * (1 - self.val_split))
        
        mismatch = self.dataset_len - (self.train_len + self.val_len)
        if(mismatch):
            self.train_len+=mismatch

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset,
                                                                    [self.train_len, self.val_len],
                                                                    generator=torch.Generator().manual_seed(self.random_state))
                             
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers)



if __name__ == '__main__':
    import yaml
    config_dir = 'config.yaml'
    with open(config_dir) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            
    config['data_config']['train'] = True
    data_module = Star_Loader(config['data_config'])
    
    for x, y in data_module.train_dataloader():
        break

    for x, y in data_module.val_dataloader():
        break
    
    print("\nShape of input:", x.shape, "\nShape of output:",y.shape)
    #print("\nSample input:",x[0],"\nSample output:",y[0])
    print("\nLength of dataset:",len(data_module))
    
    config['data_config']['train'] = False
    data_module = Star_Loader(config['data_config'])
    
    for x, y in data_module.test_dataloader():
        break

    print("\nShape of input: ", x.shape, "\nShape of output:",y.shape)
    #print("\nSample input:",x[0],"\nSample Output:",y[0])
    print("\nLength of dataset:",len(data_module))