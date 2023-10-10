import torch
import lightning as pl
from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import rand_score

from lbae import LBAE

model = LBAE.load_from_checkpoint(checkpoint_path='/Users/dawidmazur/Code/pixel-level-image-analysis/lightning_logs/version_3/checkpoints/epoch=19-step=28740.ckpt', 
                                  hparams_file='/Users/dawidmazur/Code/pixel-level-image-analysis/lightning_logs/version_3/hparams.yaml',
                                  map_location=torch.device('cpu'))

digits = datasets.load_digits()
dataset_split = int(len(digits.data) * 0.8)

X_train, X_test = digits.data[:dataset_split]/16, digits.data[dataset_split:]/16
y_train, y_test = digits.target[:dataset_split], digits.target[dataset_split:]

X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test)
y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)

X_test = torch.reshape(X_test, (len(X_test), 1, 8, 8))

preds = model(X_test)

# test_first_image = X_test[0].clone().detach()
# preds_first_image = preds[0].clone().detach()

# plt.imshow(test_first_image.permute(1, 2, 0), cmap='gray')
# plt.show()

# plt.imshow(preds_first_image.permute(1, 2, 0), cmap='gray')
# plt.show()

# test_first_image = test_first_image.reshape(64,)
# preds_first_image = preds_first_image.reshape(64,)

# rand_index = rand_score(test_first_image, preds_first_image)
# print(f'Rand score dla pierwszego elementu z zbioru testowwego: {round(rand_index, 2)}')

X_test = X_test.clone().detach().reshape(360*64,)
preds = preds.clone().detach().reshape(360*64,)

rand_index_test = rand_score(X_test, preds)
print(f'Rand score dla zbioru testowwego: {round(rand_index_test, 2)}')