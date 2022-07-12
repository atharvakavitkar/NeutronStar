import os
import yaml
import torch
import pytorch_lightning as pl

#user defined modules
from models.MultimodalTransformer import MultimodalTransformer
from data.dataset import Star_Loader
from trainer.trainer import RegressionTrainer
from pytorch_lightning.loggers import WandbLogger



def get_config(config_dir):
    config_file = os.path.join(config_dir,'config.yaml')
    with open(config_file) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def train(config_dir):

    config = get_config(config_dir)
    print(yaml.dump(config))

    pl.seed_everything(config['data_config']['seed_value'])
    
    print('\nIs CUDA available to PyTorch?:', torch.cuda.is_available())
    print('Number of GPUs visible to PyTorch: ', torch.cuda.device_count())

    data = Star_Loader(config['data_config'])

    model = MultimodalTransformer(**config['model_config'])
    
    regressor = RegressionTrainer(model=model,
                                  lr = config['train_config']['learning_rate'])

    wandb_logger = WandbLogger(save_dir=config['result_dir'])
    wandb_logger.experiment.config.update(config)
    
    trainer = pl.Trainer(auto_select_gpus=True,
                         max_epochs=config['train_config']['epochs'],
                         logger=wandb_logger)

    trainer.fit(regressor,datamodule=data)

    test_metric = trainer.test(model = regressor,dataloaders=data.test_dataloader())
    print(test_metric,type(test_metric))

    predictions = trainer.predict(dataloaders=data.test_dataloader())
    torch.save(predictions,'predictions.pt')
    
if __name__ == "__main__":
    config_dir = 'D:/Masters/NS_EoS/NeutronStar/'
    train(config_dir)