import os
import numpy as np
import config
from config import DEVICE
from dataset import IonDataset
from engine import train_fn, eval_fn
from model import Classifier
from utils import EarlyStopping
import torch
import torch.nn as nn
from torchcontrib.optim import SWA
import torchcontrib
from torch.utils.data import Dataset, DataLoader
from data_preprocessing import split
from sklearn.metrics import f1_score


STORE_DIR = config.outdir
if not os.path.exists(STORE_DIR):
    os.makedirs(STORE_DIR)

train, test, train_tr, new_splits = split()

# test_y = np.zeros([int(2000_000/config.GROUP_BATCH_SIZE), config.GROUP_BATCH_SIZE, 1])
# test_dataset = IonDataset(test, test_y, flip=False)
# test_dataloader = DataLoader(test_dataset, config.NNBATCHSIZE, shuffle=False)
# test_preds_all = np.zeros((2000_000, 11))


oof_score = []
for index, (train_index, val_index, _) in enumerate(new_splits[0:], start=0):
    print(f"Fold : {index}")
    train_dataset = IonDataset(train[train_index], train_tr[train_index], 
                        seq_len=config.GROUP_BATCH_SIZE, flip=config.FLIP, noise_level=config.NOISE)
    train_dataloader = DataLoader(train_dataset, config.NNBATCHSIZE, shuffle=True, num_workers = 16)

    valid_dataset = IonDataset(train[val_index], train_tr[val_index], 
                        seq_len=config.GROUP_BATCH_SIZE, flip=False)
    valid_dataloader = DataLoader(valid_dataset, config.NNBATCHSIZE, shuffle=False)

    it = 0
    model = Classifier()
    model = model.to(DEVICE)

    early_stopping = EarlyStopping(patience=40, is_maximize=True,
                                   checkpoint_path=os.path.join(STORE_DIR, f"gru_clean_checkpoint_fold_{index}_iter_{it}.pt"))
                                                                                                             
    weight = None#cal_weights()
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    optimizer = torchcontrib.optim.SWA(optimizer, swa_start=10, swa_freq=2, swa_lr=0.0011)

    #schedular = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=LR, max_lr=0.003, step_size_up=len(train_dataset)/2, cycle_momentum=False)
    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.2)

    avg_train_losses = []
    avg_valid_losses = []
    for epoch in range(config.EPOCHS):

        print('**********************************')
        print(f"Folder : {index} Epoch : {epoch}")
        print("Learning_rate: {:0.7f}".format(optimizer.param_groups[0]['lr']))

        avg_loss = train_fn(data_loader=train_dataloader, 
                            model=model, 
                            optimizer=optimizer,
                            device=DEVICE, 
                            scheduler=schedular,
                            criterion=criterion)

        optimizer.update_swa()
        optimizer.swap_swa_sgd()

        avg_valid_loss, f1_val = eval_fn(data_loader=valid_dataloader,
                                        device=DEVICE, 
                                        model=model,
                                        criterion=criterion)

        optimizer.swap_swa_sgd()

        avg_train_losses.append(avg_loss)
        avg_valid_losses.append(avg_valid_loss)

        schedular.step(f1_val)
        res = early_stopping(f1_val, model)
        if  res == 2:
            print("Early Stopping")
            print('folder %d global best val max f1 model score %0.5f' % (index, early_stopping.best_score))
            break
        elif res == 1:
            print('save folder %d global val max f1 model score %0.5f' % (index, f1_val))

    print(f'Folder {index} finally best global max f1 score is {early_stopping.best_score:0.5f}')
    oof_score.append(round(early_stopping.best_score, 6))


        
