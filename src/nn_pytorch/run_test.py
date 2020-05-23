# import os
# import config
# import numpy as np


# # test_y = np.zeros([int(2000_000/config.GROUP_BATCH_SIZE), config.GROUP_BATCH_SIZE, 1])
# # test_dataset = IonDataset(test, test_y, flip=False)
# # test_dataloader = DataLoader(test_dataset, config.NNBATCHSIZE, shuffle=False)
# # test_preds_all = np.zeros((2000_000, 11))


# for dirname, _, filenames in os.walk(config.outdir):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

#         model.eval()
#         pred_list = []
#         with torch.no_grad():
#             for x, y in tqdm(test_dataloader):
                
#                 x = x.to(config.DEVICE)
#                 y = y.to(config.DEVICE)

#                 predictions = model(x)
#                 predictions_ = predictions.view(-1, predictions.shape[-1]) # shape [128, 4000, 11]
#                 #print(predictions.shape, F.softmax(predictions_, dim=1).cpu().numpy().shape)
#                 pred_list.append(F.softmax(predictions_, dim=1).cpu().numpy()) # shape (512000, 11)
#                 #a = input()
#             test_preds = np.vstack(pred_list) # shape [2000000, 11]
#             test_preds_all += test_preds
