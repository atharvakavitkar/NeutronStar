import torch
from pytorch_lightning import LightningModule

class RegressionTrainer(LightningModule):
    def __init__(self, model,lr):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = torch.nn.MSELoss()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return y,y_hat

    # def validation_epoch_end(self):
    #     pass
    
    # def training_epoch_end(self):
    #     pass

    # def test_epoch_end(self):
    #     pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)