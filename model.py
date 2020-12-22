#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch 
import torch.nn as nn
from tqdm import tqdm


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        
        super(Model, self).__init__()
        self.train_loader = None
        self.valid_loader = None
        self.optimizer = None
        self.scheduler = None
        self.model_state = None
        self.current_epoch = 0
        self.model_path = None
        self.metrics = {}
        self.metrics["train"] = {}
        self.metrics["valid"] = {}
        self.metrics["test"] = {}
    

    def load(self, model_path, device="cuda"):
        if next(self.paramters()).device != device:
            self.to(device)
        model_dict = torch.load(model_path)
        self.load_state_dict(model_dict["state_dict"])
    
    def save(self, model_path):
        model_state_dict = self.state_dict()
        if self.optimizer is not None:
            opt_state_dict = self.optimizer.state_dict()
        else:
            opt_state_dict = None
        
        if self.scheduler is not None:
            sch_state_dict = self.scheduler.state_dict()
        else:
            sch_state_dict = None
        
        model_dict= {}
        model_dict["state_dict"] = model_state_dict
        model_dict["optimizer"] = opt_state_dict
        model_dict["scheduler"] = sch_state_dict
        model_dict["epoch"] = self.current_epoch
        torch.save(model_dict, model_path)



    def fit(
        self,
        train_dataset,
        valid_dataset=None,
        train_sampler=None,
        valid_sampler=None,
        device="cuda",
        epochs=10,
        train_bs=16,
        valid_bs=16,
        n_jobs=8,
        callbacks=None,
        fp16=False,
    ):
        self._init_model(
            device=device,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            train_sampler=train_sampler,
            valid_sampler=valid_sampler,
            train_bs=train_bs,
            valid_bs=valid_bs,
            n_jobs=n_jobs,
            callbacks=callbacks,
            fp16=fp16,
        )

        for _ in range(epochs):
            self.train_state = enums.TrainingState.EPOCH_START
            self.train_state = enums.TrainingState.TRAIN_EPOCH_START
            train_loss = self.train_one_epoch(self.train_loader, device)
            self.train_state = enums.TrainingState.TRAIN_EPOCH_END
            if self.valid_loader:
                self.train_state = enums.TrainingState.VALID_EPOCH_START
                valid_loss = self.validate_one_epoch(self.valid_loader, device)
                self.train_state = enums.TrainingState.VALID_EPOCH_END
            if self.scheduler:
                if self.step_scheduler_after == "epoch":
                    if self.step_scheduler_metric is None:
                        self.scheduler.step()
                    else:
                        self.scheduler.step(self.step_scheduler_metric)
            self.train_state = enums.TrainingState.EPOCH_END
            if self._model_state.value == "end":
                break
            self.current_epoch += 1
        self.train_state = enums.TrainingState.TRAIN_END


# In[ ]:




