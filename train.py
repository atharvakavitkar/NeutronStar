import os
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

#user defined modules
from models.MultimodalTransformer import MultimodalTransformer
from models.MultimodalTransformer import FeedForwardNN
from data.dataset import Star_Loader
from trainer.trainer import RegressionTrainer
from inference.inference import Inference


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
    wandb_logger = WandbLogger(save_dir=config['inference_config']['result_dir'])
    wandb_logger.experiment.config.update(config)
    pl.seed_everything(config['data_config']['seed_value'])
    
    print('\nIs CUDA available to PyTorch?:', torch.cuda.is_available())
    print('Number of GPUs visible to PyTorch: ', torch.cuda.device_count())

    config['data_config']['train'] = True
    data = Star_Loader(config['data_config'])

    callbacks = []
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=2,
                                          monitor='val_loss',
                                          filename='{epoch}-{val_loss:.4f}')

    callbacks.append(model_checkpoint)
    
    if config['train_config']['early_stopping']:
        early_stopping = EarlyStopping( monitor='val_loss', 
                                       mode='min', 
                                       verbose=False,
                                       patience=config['train_config']['stopping_patience'],
                                       check_finite=True,)
        callbacks.append(early_stopping)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    model = MultimodalTransformer(**config['model_config'])
    # model = FeedForwardNN(
    #              input_size = 437,
    #              output_size= 2,
    #              hidden_layers = [],
    #              activation=torch.nn.ReLU(),
    #              batch_norm = True,
    #              dropout = 0.2)
    print(model)
    
    regressor = RegressionTrainer(model=model,
                                  lr = config['train_config']['learning_rate'])
    
    trainer = pl.Trainer(accelerator="gpu", devices=1,
                         max_epochs=config['train_config']['epochs'],
                         auto_lr_find=True,
                         callbacks=callbacks,
                         logger=wandb_logger)
    trainer.tune(model=regressor,datamodule=data)
    trainer.fit(regressor,datamodule=data)

    config['data_config']['train'] = False
    post_process = Inference(config = config, model = regressor, logger = True)
    post_process.evaluate()
    
if __name__ == "__main__":
    config_dir = ''
    train(config_dir)