import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

#user defined modules
from add_noise import add_noise_np

class Star_Source(Dataset):

    def __init__(self, data_config):
        """
        Args:
            data_config (dict): Data Loading config
        """
        super(Star_Source, self).__init__()

        if data_config['mini_ns']:
            self.data_source = os.path.join(data_config['dataset_dir'],'mini_ns.pkl')
        else:
            self.data_source = os.path.join(data_config['dataset_dir'],'NS_EoS.pkl')
        
        self.data = pd.read_pickle(self.data_source)
        self.data = self.data.sample(frac = 0.01,ignore_index = True)

        nH,logTeff,dist = add_noise_np(data_config,self.data['nH'],self.data['logTeff'],self.data['dist'])
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

        self.batch_size = data_config['batch_size']
        self.num_workers = data_config['num_workers']
        self.val_split = data_config['validation_split']
        self.test_split = data_config['test_split']
        self.random_state = data_config['seed_value']
        self.val_len = None
        self.train_len = None
        self.test_len = None
        self.dataset_len = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        dataset = Star_Source(data_config)
        self.dataset_len = len(dataset)
        self.train_len = int(self.dataset_len * (1 - self.test_split))
        self.test_len = int(self.dataset_len - self.train_len)
        self.val_len = int(self.train_len * self.val_split)
        self.train_len = int(self.train_len * (1 - self.val_split))
        
        mismatch = self.dataset_len - (self.train_len + self.val_len + self.test_len)
        if(mismatch):
            self.train_len+=mismatch

        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(dataset,
                                                                    [self.train_len, self.val_len, self.test_len],
                                                                    generator=torch.Generator().manual_seed(self.random_state))

        self.input_size = self.train_dataset[0][0].shape[-1]
        self.output_size = self.train_dataset[0][1].shape[-1]
                                                  
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



def test_dm(config_file):
    import yaml
    with open(config_file) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    dm = Star_Loader(config['data_config'])
    
    for x, y in dm.train_dataloader():
        break

    for x, y in dm.val_dataloader():
        break

    for x, y in dm.test_dataloader():
        break

    print(x.shape, y.shape)
    print(x[0],y[0])
    print(dm.input_size,dm.output_size)


if __name__ == '__main__':
    config_dir = 'D:/Masters/NS_EoS/NeutronStar/config.yaml'
    test_dm(config_dir)