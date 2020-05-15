import datetime
import torch

EPOCHS = 2
NNBATCHSIZE = 32
GROUP_BATCH_SIZE = 4000
SEED = 123
LR = 0.001
SPLITS = 5
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
FLIP = False
NOISE = False


model_date = datetime.datetime.today().strftime("%d%m")
outdir = 'wavenet_pytorch_' + model_date