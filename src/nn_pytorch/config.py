import datetime
import torch

EPOCHS = 50
NNBATCHSIZE = 32
GROUP_BATCH_SIZE = 4000
SEED = 100
LR = 0.0015
SPLITS = 5
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
FLIP = False
NOISE = False


model_date = datetime.datetime.today().strftime("%d%m")
outdir = 'wavenet_pytorch_' + model_date