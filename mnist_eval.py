import torch
import lightning as pl
from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from lbae import LBAE

model = LBAE.load_from_checkpoint('E:/projects/pixel-level-image-analysis/lightning_logs/version_28/checkpoints/epoch=71-step=103464.ckpt')
model.eval()

digits = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)

print(X_test.shape)

X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test)

X_test = torch.reshape(X_test, (360, 1, 8, 8))

test_dataloader = DataLoader(list(zip(X_test,y_test)), batch_size=1, shuffle=False)

with torch.inference_mode(): 
    y_preds = model(X_test)
    print(y_preds)

# def plot_digits(nrows, ncols, images, targets, preds=None):
#     fig, axes = plt.subplots(nrows, ncols, figsize=(8, 6))
#     for i in range(nrows):
#         for j in range(ncols):
#             axes[i, j].imshow(images[i*ncols + j], cmap='gray')
#             axes[i, j].axis('off')
#             if preds == None:
#                 axes[i, j].set_title(f'Label:{targets[i*ncols + j]}')
#             else:
#                 axes[i, j].set_title(f'Label:{targets[i*ncols + j]}\nPredict:{preds[i*ncols + j]}')
#     plt.show()

