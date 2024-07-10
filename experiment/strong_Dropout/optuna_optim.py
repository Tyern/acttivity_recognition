import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch
from torch import nn
import torch.nn.functional as F

from lightning.pytorch.utilities.model_summary import ModelSummary

import lightning as L
from torch.utils.data import DataLoader, Dataset

def create_objective(model_class, EPOCHS, user, missing_sensor_numbers, batch_size=128):
    data_module = DataModule(
        test_user=user, 
        missing_sensor_numbers=missing_sensor_numbers,
        batch_size=batch_size)
    
    def _objective(trial)
        trainer = L.Trainer(
            logger=False,
            enable_checkpointing=False,
            max_epochs=EPOCHS,
            accelerator="gpu", 
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
        )
        
        dropout_rate = trial.suggest_uniform("dropout_rate", 0.0, 0.6)
        net = model_class(dropout_rate=dropout_rate)

        trainer.fit(model=net, datamodule=data_module)
        trainer.test(model=net, datamodule=data_module)

    return trainer.logger.log_dir["test_acc"]

return _objective

