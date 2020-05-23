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
from tqdm import tqdm
import pandas as pd


STORE_DIR = config.outdir
if not os.path.exists(STORE_DIR):
    os.makedirs(STORE_DIR)

train, test, train_tr, new_splits = split()


oof_score = []
for index, (train_index, val_index, _) in enumerate(new_splits[0:], start=0):
    # print(f"Fold : {index}")
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
    # base_opt = torch.optim.SGD(model.parameters(), lr=0.0015)
    # optimizer = torchcontrib.optim.SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.05)

 

    # schedular = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=config.LR, max_lr=0.003, step_size_up=len(train_dataset)/2, cycle_momentum=False) #!
    # schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.2)
    schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    avg_train_losses = []
    avg_valid_losses = []
    for epoch in range(config.EPOCHS):

        print('**********************************')
        print(f"Folder : {index} Epoch : {epoch}")
        # print("Learning_rate: {:0.7f}".format(optimizer.param_groups[0]['lr']))

        # Decay Learning Rate
        schedular.step()
        # Print Learning Rate
        print('Epoch:', epoch,'LR:', schedular.get_lr())

        avg_loss = train_fn(data_loader=train_dataloader, 
                            model=model, 
                            optimizer=optimizer,
                            device=DEVICE, 
                            scheduler=schedular,
                            criterion=criterion)

        # optimizer.update_swa()
        

        avg_valid_loss, f1_val = eval_fn(data_loader=valid_dataloader,
                                        device=DEVICE, 
                                        model=model,
                                        criterion=criterion)

        # optimizer.swap_swa_sgd()
        # optimizer.update_swa()

        avg_train_losses.append(avg_loss)
        avg_valid_losses.append(avg_valid_loss)

        schedular.step() #
        res = early_stopping(f1_val, model)
        if  res == 2:
            print("Early Stopping")
            print('folder %d global best val max f1 model score %0.5f' % (index, early_stopping.best_score))
            break
        elif res == 1:
            print('save folder %d global val max f1 model score %0.5f' % (index, f1_val))

    # optimizer.update_swa()

    print(f'Folder {index} finally best global max f1 score is {early_stopping.best_score:0.5f}')
    oof_score.append(round(early_stopping.best_score, 6))


    test_y = np.zeros([int(2000_000/config.GROUP_BATCH_SIZE), config.GROUP_BATCH_SIZE, 1])
    test_dataset = IonDataset(test, test_y, flip=False)
    test_dataloader = DataLoader(test_dataset, config.NNBATCHSIZE, shuffle=False)
    test_preds_all = np.zeros((2000_000, 11))

    model.eval()
    pred_list = []
    with torch.no_grad():
        for x, y in tqdm(test_dataloader):
            
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            predictions = model(x)
            predictions_ = predictions.view(-1, predictions.shape[-1]) # shape [128, 4000, 11]
            #print(predictions.shape, F.softmax(predictions_, dim=1).cpu().numpy().shape)
            pred_list.append(torch.softmax(predictions_, dim=1).cpu().numpy()) # shape (512000, 11)
            #a = input()
        test_preds = np.vstack(pred_list) # shape [2000000, 11]
        test_preds_all += test_preds


oof_mean = sum(oof_score)/len(oof_score)
print('all folder score is:%s' % str(oof_score))
print(f'OOF mean score is: {oof_mean}')
print('Generate submission.............')
DATA_DIR = "/home/pchlq/workspace/liverpool-ion-switching/data/"
ss = pd.read_csv(DATA_DIR + 'sample_submission.csv', dtype={'time': str})
test_preds_all = test_preds_all / np.sum(test_preds_all, axis=1)[:, None]
test_pred_frame = pd.DataFrame({'time': ss['time'].astype(str),
                                'open_channels': np.argmax(test_preds_all, axis=1)})
test_pred_frame.to_csv(STORE_DIR + f"/subm_preds_{oof_mean}.csv", index=False)
print('over')


# print("I'm gonna sleep...")
# os.system("systemctl suspend")
        
