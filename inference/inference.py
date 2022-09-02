import sys
import os
sys.path.insert(1, '/NeutronStar/')
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme()
from typing import Optional

from data.dataset import Star_Loader
from trainer.trainer import RegressionTrainer

#TODO:  1. Create an inference class
#       2. Write better code for post_processing and visualise

def evaluate(config: dict,
             model: Optional[RegressionTrainer]):
    
    test_dataloader = Star_Loader(config['data_config']).test_dataloader()
    if(model):
        my_model = model
    else:
        my_model = RegressionTrainer.load_from_checkpoint("D:/Masters/NS_EoS/Results/NeutronStar/epoch=113-step=52895.ckpt")
    base_model = RegressionTrainer.load_from_checkpoint("D:/Masters/NS_EoS/Results/NeutronStar/epoch=4-val_loss=0.4491.ckpt")
    trainer = pl.Trainer(logger=False)
    my_pred = post_processing(trainer.predict(
                                model = my_model,
                                dataloaders = test_dataloader,
                                #ckpt_path=best_ckpt_path,
                                ),'my_model')
    
    base_pred = post_processing(trainer.predict(
                                model = base_model,
                                dataloaders = test_dataloader,
                                #ckpt_path=best_ckpt_path,
                                ),'base_model')
    
    results = pd.merge(my_pred,base_pred,on=['true_m1','true_m2'])
    visualise(results,config)

def post_processing(predictions,model):
    
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
    
    return df
    
def visualise(results,config):
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    
    plt_m1 = sns.histplot(data=results, x=results['pred_m1_my_model'] - results['true_m1'],element="step", fill=False,stat = 'percent',color='purple')
    plt_m1 = sns.histplot(data=results, x=results['pred_m1_base_model'] - results['true_m1'],element="step", fill=False,stat = 'percent',color='green')
    random_m1 = 4.8 + np.random.rand(len(results.true_m1))*0.3
    plt_m1 = sns.histplot(x=random_m1 - results['true_m1'],element="step", fill=False,stat = 'percent',color = 'orange')

    plt_m1.set_xlabel("Predicted m1 relative to truth")
    plt_m1.set_ylabel("Fraction")
    plt_m1.set_title("m1 prediction residuals",fontsize=20)
    plt_m1.legend(loc='upper left', labels=['Transformer','MLP','Random'])
    plt_m1.figure.savefig(os.path.join(config['result_dir'],"m1.png"))
    plt_m1.figure.clf()
    
    plt_m2 = sns.histplot(data=results, x=results['pred_m2_my_model'] - results['true_m2'],element="step", fill=False,stat = 'percent',color='purple')
    plt_m2 = sns.histplot(data=results, x=results['pred_m2_base_model'] - results['true_m2'],element="step", fill=False,stat = 'percent',color='green')
    random_m2 = -2.05 + np.random.rand(len(results.true_m2))*0.25
    plt_m2 = sns.histplot(x=random_m2 - results['true_m2'],element="step", fill=False,stat = 'percent',color = 'orange')

    plt_m2.set_xlabel("Predicted m2 relative to truth")
    plt_m2.set_ylabel("Fraction")
    plt_m2.set_title("m2 prediction residuals",fontsize=20)
    plt_m2.legend(loc='upper left', labels=['Transformer','MLP','Random'])
    plt_m2.figure.savefig(os.path.join(config['result_dir'],"m2.png"))


if __name__ == "__main__":
    import yaml
    config_dir = 'config.yaml'
    with open(config_dir) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    evaluate(config = config)