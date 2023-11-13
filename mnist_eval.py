import torch
import lightning as pl
import numpy as np
from myDataset import myDataset
from torch.utils.data import DataLoader
from torchmetrics.functional.pairwise import pairwise_euclidean_distance
import matplotlib.pyplot as plt

from lbae import LBAE
from rbm import RBM

if torch.cuda.is_available():
    lbae = LBAE.load_from_checkpoint(checkpoint_path='E:/projects/pixel-level-image-analysis/lightning_logs/version_15/checkpoints/epoch=24-step=4500.ckpt', 
                                    hparams_file='E:/projects/pixel-level-image-analysis/lightning_logs/version_15/hparams.yaml',
                                    map_location=torch.device('cpu'))
    
else:
    lbae = LBAE.load_from_checkpoint(checkpoint_path='/Users/dawidmazur/Code/pixel-level-image-analysis/lightning_logs/version_0/checkpoints/epoch=99-step=18000.ckpt', 
                                    hparams_file='/Users/dawidmazur/Code/pixel-level-image-analysis/lightning_logs/version_0/hparams.yaml',
                                    map_location=torch.device('cpu'))


test_dataset = myDataset(dataset_part='test')
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

X = torch.Tensor(test_dataset.dataset_data/16).reshape(len(test_dataset.dataset_data), 1, 8, 8)
preds = lbae(X)

X_test = X.clone().detach().reshape(360 * 64, 1)
preds = preds.clone().detach().reshape(360 * 64, 1)

print(f'Mean pairwise euclidean distance: {torch.mean(pairwise_euclidean_distance(X_test, preds))}')

X_test = X.clone().detach().reshape(360, 64)
preds = preds.clone().detach().reshape(360, 64)

def plot_images_from_tensors(tensor1, tensor2):
    # Sprawdź, czy dane są w tensorach PyTorch
    if not isinstance(tensor1, torch.Tensor) or not isinstance(tensor2, torch.Tensor):
        raise ValueError("Both inputs should be PyTorch tensors.")

    # Wybierz pierwsze 5 obrazów z obu zbiorów
    images1 = tensor1[:9]
    images2 = tensor2[:9]

    # Tworzenie subplota
    fig, axes = plt.subplots(2, 9, figsize=(12, 5))

    # Pierwszy wiersz: Obrazy z pierwszego zbioru
    for i in range(len(images1)):
        image = images1[i].reshape(1, 8, 8)
        image = torch.permute(image, (1, 2, 0))
        axes[0, i].imshow(image, cmap='gray')
        axes[0, i].set_title(f"X_true")
        axes[0, i].axis('off')

    # Drugi wiersz: Obrazy z drugiego zbioru
    for i in range(len(images2)):
        image = images2[i].reshape(1, 8, 8)
        image = torch.permute(image, (1, 2, 0))
        axes[1, i].imshow(image, cmap='gray')
        axes[1, i].set_title(f"Pred")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

# plot_images_from_tensors(X, preds)

NUM_VISIBLE = 64
NUM_HIDDEN = 64
MAX_EPOCHS = 100


rbm_model = RBM(NUM_VISIBLE, NUM_HIDDEN)
rbm_model.load(file='rbm.npz')

sample = []
encoders = []

for i in range(len(test_dataset.dataset_data)):
    Xi = X[i].reshape(1, 1, 8, 8)
    encoder, err = lbae.encoder.forward(Xi, epoch=1)
    enc = encoder.detach().numpy()
    encoders.append(enc)

    rbm_input = encoder.detach().numpy()
    sample_h = rbm_model.sample_v_given_h(rbm_input)
    sample_h[sample_h == 0] = -1
    sample.append(sample_h)

encoders = torch.Tensor(np.array(encoders))
sample = torch.Tensor(np.array(sample))
print(sample.shape)
print(sample[0].shape)

from sklearn.metrics import rand_score

rs = rand_score(encoders.reshape(360*64,), sample.reshape(360*64,))
print(encoders.reshape(360*64,))
print(sample.reshape(360*64,))
print(rs)

print(torch.mean(pairwise_euclidean_distance(encoders.reshape(360*64, 1), sample.reshape(360*64, 1))))