import numpy as np
import torch


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    def __init__(self, patience=7, delta=0, checkpoint_path='checkpoint.pt', is_maximize=True):
        self.patience, self.delta, self.checkpoint_path = patience, delta, checkpoint_path
        self.counter, self.best_score = 0, None
        self.is_maximize = is_maximize


    def load_best_weights(self, model):
        model.load_state_dict(torch.load(self.checkpoint_path))

    def __call__(self, score, model):
        if self.best_score is None or \
                (score > self.best_score + self.delta if self.is_maximize else score < self.best_score - self.delta):
            torch.save(model.state_dict(), self.checkpoint_path)
            self.best_score, self.counter = score, 0
            return 1
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return 2
        return 0


# class EarlyStopping:
#     def __init__(self, patience=7, mode="max", delta=0.001):
#         self.patience = patience
#         self.counter = 0
#         self.mode = mode
#         self.best_score = None
#         self.early_stop = False
#         self.delta = delta
#         if self.mode == "min":
#             self.val_score = np.Inf
#         else:
#             self.val_score = -np.Inf

#     def __call__(self, epoch_score, model, model_path):

#         if self.mode == "min":
#             score = -1.0 * epoch_score
#         else:
#             score = np.copy(epoch_score)

#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(epoch_score, model, model_path)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(epoch_score, model, model_path)
#             self.counter = 0

#     def save_checkpoint(self, epoch_score, model, model_path):
#         if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
#             print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
#             torch.save(model.state_dict(), model_path)
#         self.val_score = epoch_score