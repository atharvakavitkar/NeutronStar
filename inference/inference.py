import sys
import os
sys.path.insert(1, '/NeutronStar/')
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme()
from typing import Optional
import wandb

#user-defined modules
from data.dataset import Star_Loader
from trainer.trainer import RegressionTrainer


class Inference():
    def __init__(self, 
                 config: dict,
                 model: Optional[RegressionTrainer] = None,
                 logger: bool = False):
        super().__init__()
        
        self.config = config
        if(model):
            self.new_model = model
        else:
            self.new_model = RegressionTrainer.load_from_checkpoint(self.config['inference_config']['model_ckpt'])
        
        self.base_model = RegressionTrainer.load_from_checkpoint(self.config['inference_config']['base_ckpt'])
        self.logger = logger
        self.trainer = pl.Trainer(accelerator="gpu", devices=1,logger=False)
        self.uncertainty_list = ['true','tight','loose']
        
    def evaluate(self):
    
        for self.uncertainty in ['true','tight','loose']:
            self.config['data_config']['nH'] = self.config['data_config']['logTeff'] = self.config['data_config']['dist'] = self.uncertainty
            self.test_dataloader = Star_Loader(self.config['data_config']).test_dataloader()
        
            new_pred = self.post_processing(self.trainer.predict(
                                        model = self.new_model,
                                        dataloaders = self.test_dataloader,
                                        ),'new_model')
            
            base_pred = self.post_processing(self.trainer.predict(
                                        model = self.base_model,
                                        dataloaders = self.test_dataloader,
                                        ),'base_model')
            
            self.results = pd.merge(new_pred,base_pred,on=['true_m1','true_m2'])
            if self.config['inference_config']['visualise']:
                self.visualise()

    def post_processing(self,predictions,model):
        
        true_m1 = []
        true_m2 = []
        pred_m1 = []
        pred_m2 = []
        for batch in predictions:
            for true in batch[0]:
                true_m1.append(np.float64(true[0]))
                true_m2.append(np.float64(true[1]))
            for pred in batch[1]:
                pred_m1.append(np.float64(pred[0]))
                pred_m2.append(np.float64(pred[1]))

        df = pd.DataFrame({'true_m1':true_m1,'true_m2':true_m2,
                        f'pred_m1_{model}':pred_m1,
                        f'pred_m2_{model}':pred_m2})
        
        df[f'res_m1_{model}'] = df[f'pred_m1_{model}'] - df.true_m1
        df[f'res_m2_{model}'] = df[f'pred_m2_{model}'] - df.true_m2
        
        if(self.logger):
            wandb.run.summary[f'{self.uncertainty}_m1_mean'] = df[f'res_m1_{model}'].mean()
            wandb.run.summary[f'{self.uncertainty}_m1_std'] = df[f'res_m1_{model}'].std()
            wandb.run.summary[f'{self.uncertainty}_m2_mean'] = df[f'res_m2_{model}'].mean()
            wandb.run.summary[f'{self.uncertainty}_m2_std'] = df[f'res_m2_{model}'].std()
            
        print(f'\n{model} with {self.uncertainty} nps:\n')
        print(f'm1.mean: ',df[f'res_m1_{model}'].mean())
        print(f'm1.std: ',df[f'res_m1_{model}'].std(),'\n')
        print(f'm2.mean: ',df[f'res_m2_{model}'].mean())
        print(f'm2.std: ',df[f'res_m2_{model}'].std(),'\n\n')
        return df
    
    def visualise(self):
        sns.set(rc={'figure.figsize':(11.7,8.27)})
        
        plt_m1 = sns.histplot(data=self.results, x = self.results['true_m1'] - self.results['pred_m1_new_model'],element="step", fill=False,stat = 'percent',color='purple')
        plt_m1 = sns.histplot(data=self.results, x = self.results['true_m1'] - self.results['pred_m1_base_model'],element="step", fill=False,stat = 'percent',color='green')
        random_m1 = 4.8 + np.random.rand(len(self.results.true_m1))*0.3
        plt_m1 = sns.histplot(x=random_m1 - self.results['true_m1'],element="step", fill=False,stat = 'percent',color = 'orange')

        plt_m1.set_xlabel(f'm1_true - m1_pred')
        plt_m1.set_ylabel("Fraction")
        plt_m1.set_title(f'm1 Prediction residuals with {self.uncertainty} NPs',fontsize=20)
        plt_m1.legend(loc='upper left', labels=['Transformer','MLP','Random'])
        plt_m1.figure.savefig(os.path.join(self.config['inference_config']['result_dir'],f'm1_{self.uncertainty}.png'))
        plt_m1.figure.clf()
        
        plt_m2 = sns.histplot(data=self.results, x = self.results['true_m2'] - self.results['pred_m2_new_model'],element="step", fill=False,stat = 'percent',color='purple')
        plt_m2 = sns.histplot(data=self.results, x = self.results['true_m2'] - self.results['pred_m2_base_model'],element="step", fill=False,stat = 'percent',color='green')
        random_m2 = -2.05 + np.random.rand(len(self.results.true_m2))*0.25
        plt_m2 = sns.histplot(x=random_m2 - self.results['true_m2'],element="step", fill=False,stat = 'percent',color = 'orange')

        plt_m2.set_xlabel("m2_true - m2_pred")
        plt_m2.set_ylabel("Fraction")
        plt_m2.set_title(f'm2 Prediction residuals with {self.uncertainty} NPs',fontsize=20)
        plt_m2.legend(loc='upper left', labels=['Transformer','MLP','Random'])
        plt_m2.figure.savefig(os.path.join(self.config['inference_config']['result_dir'],f'm2_{self.uncertainty}.png'))
        plt_m2.figure.clf()

if __name__ == "__main__":
    import yaml
    config_dir = 'config.yaml'
    with open(config_dir) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    post_process = Inference(config=config)
    post_process.evaluate()