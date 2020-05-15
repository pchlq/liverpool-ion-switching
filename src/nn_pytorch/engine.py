# from config import DEVICE
from tqdm.notebook import tqdm
import torch
import numpy as np
from sklearn.metrics import f1_score



def train_fn(data_loader, model, optimizer, device, criterion, scheduler=None):
    model.train()

    train_losses = []

    print('TRAINING -> ...')
    model.train()  # prep model for training
    train_preds = torch.Tensor([]).to(device) 
    train_true = torch.LongTensor([]).to(device)

    for x, y in tqdm(data_loader):     # total=len(data_loader)

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        predictions = model(x)

        predictions_ = predictions.view(-1, predictions.shape[-1])
        y_ = y.view(-1)

        loss = criterion(predictions_, y_)

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        
        scheduler.step(loss)
        # record training lossa
        train_losses.append(loss.item())
        train_true = torch.cat([train_true, y_], 0)
        train_preds = torch.cat([train_preds, predictions_], 0)

    train_loss = np.average(train_losses)
    train_score = f1_score(train_true.cpu().detach().numpy(), 
                           train_preds.cpu().detach().numpy().argmax(1),
                           labels=list(range(11)), average='macro')

    print(f"train_loss: {train_loss:0.6f}, train_f1: {train_score:0.6f}")

    return train_loss


def eval_fn(data_loader, model, device, criterion):

    valid_losses = []
    val_preds = torch.Tensor([]).to(device)
    val_true = torch.LongTensor([]).to(device)

    print('EVALUATION -> ...')
    model.eval()

    with torch.no_grad():
        
        for x, y in tqdm(data_loader):
            x = x.to(device)
            y = y.to(device)

            predictions = model(x)
            predictions_ = predictions.view(-1, predictions.shape[-1])
            y_ = y.view(-1)

            loss = criterion(predictions_, y_)
            valid_losses.append(loss.item())

            val_true = torch.cat([val_true, y_], 0)
            val_preds = torch.cat([val_preds, predictions_], 0)

    valid_loss = np.average(valid_losses)
    valid_score = f1_score(val_true.cpu().detach().numpy(), 
                           val_preds.cpu().detach().numpy().argmax(1),
                           labels=list(range(11)), average='macro')


    print(f"valid_loss: {valid_loss:0.6f}, valid_f1: {valid_score:0.6f}")

    return valid_loss, valid_score