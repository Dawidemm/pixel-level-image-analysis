from sklearn import datasets
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from lbae import LBAE
from pipeline import Pipeline

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

digits = datasets.load_digits()

# nrows = 2
# ncols = 5
# plot_digits(nrows, ncols, digits.images, digits.target)

dataset_split = int(len(digits.data) * 0.8)

X_train, X_test = digits.data[:dataset_split], digits.data[dataset_split:]
y_train, y_test = digits.target[:dataset_split], digits.target[dataset_split:]

X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test)
y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)

train_dataloader = DataLoader(list(zip(X_train,y_train)), batch_size=1, shuffle=False)

NUM_VISIBLE = 60
MAX_EPOCHS = 35

autoencoder = LBAE(input_size=(1, 8, 8), out_channels=8, zsize=NUM_VISIBLE, num_layers=2, quantize=list(range(MAX_EPOCHS)))
pipeline = Pipeline(auto_encoder=autoencoder, rbm=True, classifier=True)

pipeline.fit(train_dataloader)